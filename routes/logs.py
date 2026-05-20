"""
Blueprint: Log Viewer — /admin/logs

Queries Elasticsearch (pces-rag-logs-*) and renders them in a built-in
dashboard so operators never need to open Kibana for day-to-day log review.

Endpoints:
  GET  /admin/logs          — HTML log viewer page
  GET  /api/logs/query      — JSON: query ES and return log entries
  GET  /api/logs/stats      — JSON: log-level counts + top loggers (last hour)
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import requests
from flask import Blueprint, jsonify, render_template, request

from utils.error_handlers import get_logger, handle_route_errors

logger = get_logger(__name__)

logs_bp = Blueprint("logs_bp", __name__)

# ---------------------------------------------------------------------------
# Elasticsearch connection details
# ---------------------------------------------------------------------------
_ES_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
_ES_USER = os.getenv("ELASTICSEARCH_USER", "elastic")
_ES_PASS = os.getenv("ELASTICSEARCH_PASSWORD", "changeme")
_ES_INDEX = "pces-rag-logs-*"
_ES_TIMEOUT = 10  # seconds


def _es_request(method: str, path: str, **kwargs):
    """Helper: make an authenticated request to Elasticsearch."""
    url = f"{_ES_URL.rstrip('/')}/{path.lstrip('/')}"
    resp = requests.request(
        method, url,
        auth=(_ES_USER, _ES_PASS),
        timeout=_ES_TIMEOUT,
        **kwargs,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# HTML page
# ---------------------------------------------------------------------------

@logs_bp.route("/admin/logs")
def logs_page():
    return render_template("admin_logs.html")


# ---------------------------------------------------------------------------
# JSON API — query
# ---------------------------------------------------------------------------

@logs_bp.route("/api/logs/query")
@handle_route_errors
def logs_query():
    """
    Query Elasticsearch for log entries.

    Query params:
      level    — info | warning | error | debug  (omit = all)
      logger   — filter by log.logger prefix
      search   — free-text search in message
      minutes  — time window in minutes (default 60)
      size     — number of results (default 100, max 500)
    """
    level   = request.args.get("level", "").strip().lower()
    logger_ = request.args.get("logger", "").strip()
    search  = request.args.get("search", "").strip()
    minutes = int(request.args.get("minutes", 60))
    size    = min(int(request.args.get("size", 100)), 500)

    must: list[dict] = [
        {"range": {"@timestamp": {"gte": f"now-{minutes}m", "lte": "now"}}}
    ]

    if level:
        must.append({"term": {"log.level": level}})

    if logger_:
        must.append({"match_phrase_prefix": {"log.logger": logger_}})

    if search:
        must.append({"match": {"message": {"query": search, "operator": "or"}}})

    body = {
        "query": {"bool": {"must": must}},
        "sort": [{"@timestamp": {"order": "desc"}}],
        "size": size,
        "_source": [
            "@timestamp", "log.level", "log.logger",
            "message",
            "http.request.method", "url.path", "http.response.status_code",
            "event.duration", "transaction.id",
            "log.origin.file.name", "log.origin.function", "log.origin.line",
            "error.message", "error.stack_trace",
        ],
    }

    try:
        data = _es_request("POST", f"{_ES_INDEX}/_search", json=body)
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Elasticsearch unreachable", "hits": [], "total": 0}), 503
    except requests.exceptions.HTTPError as exc:
        return jsonify({"error": str(exc), "hits": [], "total": 0}), 500

    hits = data.get("hits", {})
    total = hits.get("total", {}).get("value", 0)
    entries = []
    for h in hits.get("hits", []):
        src = h.get("_source", {})
        # Compute duration_ms from nanoseconds (event.duration is in ns)
        dur_ns = src.get("event.duration") or src.get("event", {}).get("duration")
        dur_ms = round(dur_ns / 1_000_000, 1) if dur_ns else None
        entries.append({
            "ts":         src.get("@timestamp", ""),
            "level":      src.get("log.level") or src.get("log", {}).get("level", ""),
            "logger":     src.get("log.logger") or src.get("log", {}).get("logger", ""),
            "message":    src.get("message", ""),
            "method":     src.get("http.request.method") or src.get("http", {}).get("request", {}).get("method"),
            "path":       src.get("url.path") or src.get("url", {}).get("path"),
            "status":     src.get("http.response.status_code") or src.get("http", {}).get("response", {}).get("status_code"),
            "duration_ms": dur_ms,
            "tx_id":      (src.get("transaction.id") or src.get("transaction", {}).get("id", ""))[:8],
        })

    return jsonify({"hits": entries, "total": total})


# ---------------------------------------------------------------------------
# JSON API — stats
# ---------------------------------------------------------------------------

@logs_bp.route("/api/logs/stats")
@handle_route_errors
def logs_stats():
    """Return log-level counts and top 10 loggers for the last hour."""
    body = {
        "query": {"range": {"@timestamp": {"gte": "now-1h", "lte": "now"}}},
        "size": 0,
        "aggs": {
            "by_level": {
                "terms": {"field": "log.level", "size": 10}
            },
            "by_logger": {
                "terms": {"field": "log.logger", "size": 10}
            },
            "over_time": {
                "date_histogram": {
                    "field": "@timestamp",
                    "fixed_interval": "5m",
                    "min_doc_count": 0,
                    "extended_bounds": {
                        "min": "now-1h",
                        "max": "now"
                    }
                }
            }
        }
    }

    try:
        data = _es_request("POST", f"{_ES_INDEX}/_search", json=body)
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Elasticsearch unreachable"}), 503
    except requests.exceptions.HTTPError as exc:
        return jsonify({"error": str(exc)}), 500

    aggs = data.get("aggregations", {})
    total = data.get("hits", {}).get("total", {}).get("value", 0)

    level_counts = {
        b["key"]: b["doc_count"]
        for b in aggs.get("by_level", {}).get("buckets", [])
    }
    top_loggers = [
        {"name": b["key"], "count": b["doc_count"]}
        for b in aggs.get("by_logger", {}).get("buckets", [])
    ]
    timeline = [
        {"ts": b["key_as_string"], "count": b["doc_count"]}
        for b in aggs.get("over_time", {}).get("buckets", [])
    ]

    return jsonify({
        "total": total,
        "level_counts": level_counts,
        "top_loggers": top_loggers,
        "timeline": timeline,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    })
