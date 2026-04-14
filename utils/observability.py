"""
Observability utilities for the Medical RAG App.

Provides:
1. ECS-format JSON logging (compatible with Filebeat → Elasticsearch/Kibana)
2. Prometheus metrics endpoint (/metrics) via prometheus_client
3. /health endpoint with app + DB status
4. Flask before/after_request middleware for request logging

Usage in main.py:
    from utils.observability import init_observability
    init_observability(app)

Environment variables:
    LOG_LEVEL  — DEBUG / INFO / WARNING / ERROR, default INFO
    LOG_FILE   — path to write JSON logs, default "" (stdout only)

To ship logs to Elasticsearch, use the bundled docker-compose.elk.yml to run
a local ELK stack and point Filebeat at LOG_FILE (or stdout via Docker logs).
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
# Prometheus (optional import — degrades gracefully if not installed)
# ---------------------------------------------------------------------------
try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

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
# Prometheus metrics (created lazily so they only exist when library is present)
# ---------------------------------------------------------------------------

_REQUEST_COUNT: Any = None
_REQUEST_LATENCY: Any = None
_ERROR_COUNT: Any = None
_ACTIVE_REQUESTS: Any = None


def _init_prometheus() -> None:
    global _REQUEST_COUNT, _REQUEST_LATENCY, _ERROR_COUNT, _ACTIVE_REQUESTS
    if not PROMETHEUS_AVAILABLE or _REQUEST_COUNT is not None:
        return
    _REQUEST_COUNT = Counter(
        "http_requests_total",
        "Total HTTP requests",
        ["method", "endpoint", "status"],
    )
    _REQUEST_LATENCY = Histogram(
        "http_request_duration_seconds",
        "HTTP request latency in seconds",
        ["method", "endpoint"],
        buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
    )
    _ERROR_COUNT = Counter(
        "http_errors_total",
        "Total HTTP errors (4xx/5xx)",
        ["method", "endpoint", "status"],
    )
    _ACTIVE_REQUESTS = Gauge(
        "http_active_requests",
        "Number of requests currently being processed",
    )


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
    if PROMETHEUS_AVAILABLE:
        _init_prometheus()

    logger = logging.getLogger("observability")

    # ── before_request: stamp start time + request ID ──────────────────────
    @app.before_request
    def _before() -> None:
        g._obs_start = time.perf_counter()
        g._obs_request_id = str(uuid.uuid4())
        if PROMETHEUS_AVAILABLE and _ACTIVE_REQUESTS is not None:
            _ACTIVE_REQUESTS.inc()

    # ── after_request: log + record Prometheus metrics ──────────────────────
    @app.after_request
    def _after(response):  # type: ignore[return]
        duration = time.perf_counter() - getattr(g, "_obs_start", time.perf_counter())
        endpoint = request.endpoint or request.path
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

        if PROMETHEUS_AVAILABLE and _REQUEST_COUNT is not None:
            _REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
            _REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)
            if status >= 400:
                _ERROR_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
            if _ACTIVE_REQUESTS is not None:
                _ACTIVE_REQUESTS.dec()

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
                "prometheus": "available" if PROMETHEUS_AVAILABLE else "not_installed",
            },
        }
        return jsonify(payload), (200 if status == "ok" else 207)

    # ── /metrics endpoint (Prometheus) ─────────────────────────────────────
    if PROMETHEUS_AVAILABLE:
        from flask import Response

        @app.route("/metrics", methods=["GET"])
        def metrics():
            return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

    logger.info("Observability initialised (Prometheus=%s)", PROMETHEUS_AVAILABLE)

