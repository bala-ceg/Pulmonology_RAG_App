"""
Observability utilities for the Medical RAG App.

Provides:
1. ECS-format JSON logging (compatible with Filebeat → Elasticsearch/Kibana)
2. /health endpoint with app + DB status
3. Flask before/after_request middleware for request logging

Usage in main.py:
    from utils.observability import init_observability
    init_observability(app)

Environment variables:
    LOG_LEVEL  — DEBUG / INFO / WARNING / ERROR, default INFO
    LOG_FILE   — path to write JSON logs, default "" (stdout only)

To ship logs to Kibana, start the ELK stack:
    docker compose -f docker-compose.elk.yml up -d
Then open http://localhost:5601 and select the 'pces-rag-logs-*' data view.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from flask import Flask, request, jsonify, g

# ---------------------------------------------------------------------------
# ECS JSON Formatter
# ---------------------------------------------------------------------------

class ECSFormatter(logging.Formatter):
    """
    Formats log records as Elastic Common Schema (ECS) JSON.

    Each line is a self-contained JSON object that Filebeat can ingest
    directly into Elasticsearch without further transformation.
    """

    SERVICE_NAME = os.getenv("SERVICE_NAME", "pces-rag-app")
    SERVICE_VERSION = os.getenv("SERVICE_VERSION", "1.0.0")
    ENVIRONMENT = os.getenv("FLASK_ENV", "production")

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        doc: dict[str, Any] = {
            "@timestamp": datetime.now(timezone.utc).isoformat(),
            "log.level": record.levelname.lower(),
            "message": record.getMessage(),
            "service.name": self.SERVICE_NAME,
            "service.version": self.SERVICE_VERSION,
            "environment": self.ENVIRONMENT,
            "log.logger": record.name,
            "log.origin": {
                "file.name": record.pathname,
                "function": record.funcName,
                "line": record.lineno,
            },
        }
        if record.exc_info:
            doc["error"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else "Exception",
                "message": str(record.exc_info[1]),
                "stack_trace": self.formatException(record.exc_info),
            }
        # Merge any extra fields attached to the record (prefix ecs_ stripped)
        for key, val in record.__dict__.items():
            if key.startswith("ecs_"):
                doc[key[4:]] = val
        return json.dumps(doc, default=str)


# ---------------------------------------------------------------------------
# Root logger setup
# ---------------------------------------------------------------------------

def _setup_logging() -> None:
    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)

    ecs_handler = logging.StreamHandler()
    ecs_handler.setFormatter(ECSFormatter())
    root.addHandler(ecs_handler)

    log_file = os.getenv("LOG_FILE", "")
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(ECSFormatter())
        root.addHandler(fh)


# ---------------------------------------------------------------------------
# Health check helper
# ---------------------------------------------------------------------------

def _db_status() -> dict[str, Any]:
    """Try a lightweight PostgreSQL ping; return status dict."""
    try:
        import psycopg  # type: ignore[import]
        conn_str = (
            f"host={os.getenv('DB_HOST', 'localhost')} "
            f"port={os.getenv('DB_PORT', '5432')} "
            f"dbname={os.getenv('DB_NAME', 'pces_base')} "
            f"user={os.getenv('DB_USER', '')} "
            f"password={os.getenv('DB_PASSWORD', '')} "
            "connect_timeout=2"
        )
        with psycopg.connect(conn_str) as conn:
            conn.execute("SELECT 1")
        return {"status": "ok"}
    except Exception as exc:
        return {"status": "degraded", "error": str(exc)}


# ---------------------------------------------------------------------------
# Flask integration
# ---------------------------------------------------------------------------

def init_observability(app: Flask) -> None:
    """
    Attach observability middleware and endpoints to a Flask app.

    Call this once during app initialisation:
        from utils.observability import init_observability
        init_observability(app)
    """
    _setup_logging()

    logger = logging.getLogger("observability")

    # ── before_request: stamp start time + request ID ──────────────────────
    @app.before_request
    def _before() -> None:
        g._obs_start = time.perf_counter()
        g._obs_request_id = str(uuid.uuid4())

    # ── after_request: log request details as ECS JSON ──────────────────────
    @app.after_request
    def _after(response):  # type: ignore[return]
        duration = time.perf_counter() - getattr(g, "_obs_start", time.perf_counter())
        method = request.method
        status = response.status_code

        logger.info(
            f"{method} {request.path} → {status}",
            extra={
                "ecs_http.request.method": method,
                "ecs_url.path": request.path,
                "ecs_http.response.status_code": status,
                "ecs_event.duration": int(duration * 1e9),
                "ecs_transaction.id": getattr(g, "_obs_request_id", ""),
            },
        )

        return response

    # ── /health endpoint ────────────────────────────────────────────────────
    @app.route("/health", methods=["GET"])
    def health():
        db = _db_status()
        status = "ok" if db["status"] == "ok" else "degraded"
        payload = {
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": ECSFormatter.SERVICE_NAME,
            "version": ECSFormatter.SERVICE_VERSION,
            "components": {
                "database": db,
            },
        }
        return jsonify(payload), (200 if status == "ok" else 207)

    logger.info("Observability initialised (ELK logging enabled)")

