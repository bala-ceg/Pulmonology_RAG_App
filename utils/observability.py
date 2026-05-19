"""
Observability utilities for the Medical RAG App.

Provides:
1. ECS-format JSON logging (file only → Filebeat → Elasticsearch/Kibana)
2. Clean human-readable logging to stdout (INFO level)
3. /health endpoint with app + DB status
4. Flask before/after_request middleware for request logging

Log architecture:
  stdout  → human-readable  "2026-05-20 00:27:02 [INFO] routes.query: ..."
  LOG_FILE → ECS JSON        {"@timestamp":..., "log.level":"info", ...}

Usage:
    # Call once, early — before any other imports:
    from utils.observability import setup_logging
    setup_logging()

    # Call once after Flask app is created:
    from utils.observability import init_observability
    init_observability(app)

Environment variables:
    LOG_LEVEL  — DEBUG / INFO / WARNING / ERROR, default INFO
    LOG_FILE   — path to write ECS JSON logs (e.g. ./logs/pces.log)
                 If not set, ECS JSON is skipped (stdout only).

To ship logs to Kibana, start the ELK stack:
    docker compose -f docker-compose.elk.yml up -d
Then open http://localhost:5601 → Discover → 'pces-rag-logs-*'.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from flask import Flask, request, jsonify, g

# ---------------------------------------------------------------------------
# ANSI strip helper (werkzeug injects colour codes into messages)
# ---------------------------------------------------------------------------

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


# ---------------------------------------------------------------------------
# suppress_stdout — silences C-level stdout (e.g. modelscope LOAD REPORT)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def suppress_stdout():
    """
    Context manager that redirects fd 1 (stdout) AND fd 2 (stderr) to /dev/null.
    Used around SentenceTransformer / rlhf_reranker model loads to suppress
    the 'BertModel LOAD REPORT' and tqdm progress bars that write directly to fds.
    """
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        os.close(devnull_fd)
        try:
            yield
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(old_stdout)
            os.close(old_stderr)
    except Exception:
        yield  # fallback: no suppression if fd ops fail


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


# ---------------------------------------------------------------------------
# ECS JSON Formatter  (file handler only — Kibana)
# ---------------------------------------------------------------------------

class ECSFormatter(logging.Formatter):
    """Formats log records as Elastic Common Schema JSON for Filebeat ingestion."""

    SERVICE_NAME = os.getenv("SERVICE_NAME", "pces-rag-app")
    SERVICE_VERSION = os.getenv("SERVICE_VERSION", "1.0.0")
    ENVIRONMENT = os.getenv("FLASK_ENV", "production")

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        doc: dict[str, Any] = {
            "@timestamp": datetime.now(timezone.utc).isoformat(),
            "log.level": record.levelname.lower(),
            "message": _strip_ansi(record.getMessage()),
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
        for key, val in record.__dict__.items():
            if key.startswith("ecs_"):
                doc[key[4:]] = val
        return json.dumps(doc, default=str)


# ---------------------------------------------------------------------------
# Human-readable formatter  (stdout — terminal)
# ---------------------------------------------------------------------------

_CONSOLE_FMT = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ---------------------------------------------------------------------------
# Third-party loggers to silence (set to WARNING to reduce noise)
# ---------------------------------------------------------------------------

_SILENT_LOGGERS = [
    "werkzeug",          # HTTP request lines — our observability handles them
    "arxiv",             # Paper fetch internals
    "urllib3",           # HTTP client internals
    "httpx",
    "httpcore",
    "sentence_transformers",  # BERT/SBERT model load reports
    "transformers",
    "huggingface_hub",
    "pinecone",
    "openai._base_client",
    "langchain_core",
    "chromadb",
    "filelock",
]


# ---------------------------------------------------------------------------
# Public setup — call ONCE early in main.py before any other imports
# ---------------------------------------------------------------------------

_logging_configured = False


def setup_logging() -> None:
    """
    Configure root logger with:
      - stdout  → human-readable INFO handler
      - LOG_FILE → ECS JSON INFO handler (if LOG_FILE env var is set)

    Silences noisy third-party libraries.
    Call this once, early — before importing routes/services.
    """
    global _logging_configured
    if _logging_configured:
        return
    _logging_configured = True

    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)

    root = logging.getLogger()
    # Remove any handlers added by basicConfig or early library imports
    root.handlers.clear()
    root.setLevel(level)

    # ── Terminal: clean human-readable ─────────────────────────────────────
    import sys
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(_CONSOLE_FMT)
    console.setLevel(level)
    root.addHandler(console)

    # ── File: ECS JSON for Kibana ───────────────────────────────────────────
    log_file = os.getenv("LOG_FILE", "")
    if log_file:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        except Exception:
            pass
        fh = logging.FileHandler(log_file)
        fh.setFormatter(ECSFormatter())
        fh.setLevel(level)
        root.addHandler(fh)

    # ── Silence third-party noise ───────────────────────────────────────────
    for name in _SILENT_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    # ── Suppress tqdm progress bars (BERT/SBERT model loading) ─────────────
    os.environ.setdefault("TQDM_DISABLE", "1")
    # Suppress modelscope/ms_swift BertModel LOAD REPORT printouts
    os.environ.setdefault("MODELSCOPE_VERBOSITY", "error")


# ---------------------------------------------------------------------------
# Health check helper
# ---------------------------------------------------------------------------

def _db_status() -> dict[str, Any]:
    """Lightweight PostgreSQL ping."""
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
    Attach HTTP middleware and /health endpoint to the Flask app.
    Call setup_logging() separately (early) before this.
    """
    # Ensure logging is set up even if called standalone
    setup_logging()

    logger = logging.getLogger("observability")

    # ── before_request: stamp start time + request ID ──────────────────────
    @app.before_request
    def _before() -> None:
        g._obs_start = time.perf_counter()
        g._obs_request_id = str(uuid.uuid4())

    # ── after_request: single structured log per request ───────────────────
    @app.after_request
    def _after(response):  # type: ignore[return]
        duration_ms = (time.perf_counter() - getattr(g, "_obs_start", time.perf_counter())) * 1000
        method = request.method
        status = response.status_code
        path = request.path

        # Skip favicon / Chrome devtools noise
        if path in ("/favicon.ico",) or path.startswith("/.well-known/"):
            return response

        level = logging.WARNING if status >= 400 else logging.INFO
        logger.log(
            level,
            "%s %s → %d  (%.0fms)",
            method, path, status, duration_ms,
            extra={
                "ecs_http.request.method": method,
                "ecs_url.path": path,
                "ecs_http.response.status_code": status,
                "ecs_event.duration": int(duration_ms * 1e6),
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
            "components": {"database": db},
        }
        return jsonify(payload), (200 if status == "ok" else 207)

    logger.info("Observability initialised (ELK logging enabled)")


