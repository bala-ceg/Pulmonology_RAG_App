"""
Pinecone KB routes
==================
Provides test/status endpoints for the Pinecone knowledge base integration.

Endpoints
---------
GET  /api/pinecone/test          — connection check + index stats + sample query
POST /api/pinecone/seed          — seed sample data into all namespaces
GET  /api/pinecone/stats         — index stats only
POST /api/pinecone/query         — ad-hoc query  { "query": "...", "namespace": "..." }
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request

try:
    from pinecone_kb import get_pinecone_kb, PINECONE_KB_AVAILABLE
except ImportError:
    PINECONE_KB_AVAILABLE = False

pinecone_bp = Blueprint("pinecone", __name__)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _unavailable():
    return jsonify({"status": "error", "message": "Pinecone KB is not available. Check PINECONE_API_KEY and pinecone package."}), 503


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@pinecone_bp.route("/api/pinecone/test", methods=["GET"])
def pinecone_test():
    """Connection check, index stats, and a sample query per department."""
    if not PINECONE_KB_AVAILABLE:
        return _unavailable()

    try:
        kb = get_pinecone_kb()
        stats = kb.stats()

        # Run a quick sample query in each department namespace
        sample_queries = {
            "cardiology":       "atrial fibrillation treatment",
            "neurology":        "stroke thrombolysis",
            "general_medicine": "diabetes management",
            "dentist":          "dental caries prevention",
            "pulmonology":      "asthma inhaler therapy",
        }
        sample_results: dict = {}
        for dept, q in sample_queries.items():
            hits = kb.query(q, namespace=dept, top_k=1)
            sample_results[dept] = {
                "query":  q,
                "hits":   len(hits),
                "top_score": round(hits[0]["score"], 4) if hits else 0,
                "snippet": hits[0]["text"][:120] if hits else "",
            }

        return jsonify({
            "status":   "ok",
            "index":    stats,
            "samples":  sample_results,
        })
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@pinecone_bp.route("/api/pinecone/seed", methods=["POST"])
def pinecone_seed():
    """Populate namespaces with sample medical data (skips non-empty namespaces)."""
    if not PINECONE_KB_AVAILABLE:
        return _unavailable()

    try:
        kb = get_pinecone_kb()
        seeded = kb.seed_sample_data()
        return jsonify({"status": "ok", "seeded": seeded})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@pinecone_bp.route("/api/pinecone/stats", methods=["GET"])
def pinecone_stats():
    """Return current index statistics."""
    if not PINECONE_KB_AVAILABLE:
        return _unavailable()

    try:
        kb = get_pinecone_kb()
        return jsonify({"status": "ok", "stats": kb.stats()})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@pinecone_bp.route("/api/pinecone/query", methods=["POST"])
def pinecone_query():
    """Ad-hoc query endpoint. Body: { "query": "...", "namespace": "..." }"""
    if not PINECONE_KB_AVAILABLE:
        return _unavailable()

    data = request.get_json(silent=True) or {}
    query_text = (data.get("query") or "").strip()
    namespace  = data.get("namespace") or None

    if not query_text:
        return jsonify({"status": "error", "message": "query field is required"}), 400

    try:
        kb = get_pinecone_kb()
        results = kb.query(query_text, namespace=namespace, top_k=5)
        return jsonify({
            "status":    "ok",
            "query":     query_text,
            "namespace": namespace,
            "results":   results,
        })
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500
