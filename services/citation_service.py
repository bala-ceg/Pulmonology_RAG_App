"""
Citation Service — Phase 2-B: Attach Citations (Step 08)

Pipeline position (spec: 08_Implement Attach Citation v1.0):
  Confidence Scoring → Attach Citations (this module) → Log Interaction → Deliver

Implements all 10 steps from the spec:
  1.  Receive inputs (response, source_documents, citations_raw, confidence, dept, session)
  2.  Extract metadata from each source document
  3.  Select top supporting documents (by relevance score)
  4.  Format citations into readable text
  5.  Standardise source labels (human-readable names)
  6.  Build citation list (structured + HTML)
  7.  Attach citations to response with confidence score
  8.  Handle external API citations (ArXiv, PubMed links)
  9.  Add LoRA model attribution when applicable
  10. Log citations to citation_log table (async, silent fail)

Production enhancements:
  - Source reliability scoring
  - Duplicate citation removal
  - Relevance-based ranking
"""

from __future__ import annotations

import re
import threading
from typing import Any

from utils.error_handlers import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Step 5 — Standardised source labels
# Maps internal tool/source_type names to human-readable labels shown to users.
# ---------------------------------------------------------------------------
SOURCE_LABELS: dict[str, str] = {
    # Tool names (from integrated_rag.py _TOOL_SOURCE_LABELS)
    "Pinecone_KB_Search":          "Organisation Knowledge Base",
    "Internal_VectorDB":           "Doctor's Knowledge Base",
    "AdHocRAG_Search":             "Uploaded Patient Document",
    "PostgreSQL_Diagnosis_Search": "Patient Medical Record (EHR)",
    "ArXiv_Search":                "Clinical Research (arXiv)",
    "Tavily_Search":               "Web Research",
    "Wikipedia_Search":            "Medical Reference (Wikipedia)",
    # source_type values (from tools.py metadata)
    "wikipedia":  "Medical Reference (Wikipedia)",
    "arxiv":      "Clinical Research (arXiv)",
    "tavily":     "Web Research",
    "internal":   "Doctor's Knowledge Base",
    "pinecone":   "Organisation Knowledge Base",
    "main_rag":   "Doctor's Knowledge Base",
    "adhoc_rag":  "Uploaded Patient Document",
    "postgres":   "Patient Medical Record (EHR)",
    "lora_model": "Department AI Model",
    # Fallback-style keys
    "PCES Pinecone Knowledge Base":  "Organisation Knowledge Base",
    "Uploaded Documents (Adhoc VectorDB)": "Uploaded Patient Document",
    "PostgreSQL EHR Database": "Patient Medical Record (EHR)",
}

# ---------------------------------------------------------------------------
# Production enhancement — Source reliability scores (0-100)
# Higher = more trustworthy for clinical decision support.
# ---------------------------------------------------------------------------
SOURCE_RELIABILITY: dict[str, int] = {
    "Clinical Research (arXiv)":       95,
    "Patient Medical Record (EHR)":    90,
    "Organisation Knowledge Base":     90,
    "Doctor's Knowledge Base":         90,
    "Hospital Knowledge Base":         90,
    "Uploaded Patient Document":       85,
    "Department AI Model":             80,
    "Web Research":                    75,
    "Medical Reference (Wikipedia)":   70,
}

_DEFAULT_RELIABILITY = 60

# Maximum number of source documents to include in structured citations
_MAX_CITATIONS = 5


# ---------------------------------------------------------------------------
# Step 2 — Extract metadata from a source document dict
# ---------------------------------------------------------------------------
def _extract_metadata(doc: dict) -> dict:
    """
    Normalise a source_document dict (as built by integrated_rag.py) into a
    flat metadata dict used for citation rendering.
    """
    fields = doc.get("fields", {})
    meta = doc.get("metadata", {})

    source_raw = (
        fields.get("Source")
        or fields.get("Document")
        or meta.get("source")
        or meta.get("filename")
        or ""
    )
    source_type_raw = (
        meta.get("source_type")
        or fields.get("source_type")
        or _infer_source_type(source_raw, doc)
    )

    label = SOURCE_LABELS.get(source_type_raw, source_type_raw or "Unknown Source")

    # Relevance — may be "73%" string or 0.73 float
    rel_raw = fields.get("Relevance") or meta.get("score") or ""
    try:
        if isinstance(rel_raw, str) and "%" in rel_raw:
            relevance = int(rel_raw.rstrip("%"))
        elif isinstance(rel_raw, (int, float)):
            relevance = int(rel_raw * 100) if rel_raw <= 1.0 else int(rel_raw)
        else:
            relevance = 0
    except (ValueError, TypeError):
        relevance = 0

    _url = fields.get("URL") or meta.get("url") or meta.get("link") or ""
    # If source_raw is itself a URL (e.g., web-sourced Pinecone doc), use it as the link
    if not _url and source_raw and source_raw.startswith("http"):
        _url = source_raw

    return {
        "source_label": label,
        "source_raw": source_raw,
        "source_type": source_type_raw or "unknown",
        "department": fields.get("Dept") or meta.get("department") or "",
        "page": fields.get("Page") or meta.get("page") or meta.get("page_number") or "",
        "url": _url,
        "title": meta.get("title") or fields.get("title") or "",
        "relevance": relevance,
        "reliability": SOURCE_RELIABILITY.get(label, _DEFAULT_RELIABILITY),
        "excerpt": doc.get("excerpt", ""),
    }


def _infer_source_type(source_str: str, doc: dict) -> str:
    """Best-effort inference of source_type from the source string or doc content."""
    s = source_str.lower()
    if "pinecone" in s or "pces" in s:
        return "pinecone"
    if "arxiv" in s:
        return "arxiv"
    if "wikipedia" in s or "wiki" in s:
        return "wikipedia"
    if "tavily" in s or "web" in s:
        return "tavily"
    if "postgres" in s or "ehr" in s or "diagnosis" in s:
        return "postgres"
    header = doc.get("header", "").lower()
    if "pinecone" in header:
        return "pinecone"
    return "internal"


# ---------------------------------------------------------------------------
# Step 3 — Select top supporting documents (ranked by relevance × reliability)
# ---------------------------------------------------------------------------
def _rank_and_select(enriched_docs: list[dict], max_citations: int = _MAX_CITATIONS) -> list[dict]:
    """
    Rank by composite score (relevance × 0.7 + reliability × 0.3) and return
    the top `max_citations`.

    Deduplication: one entry per unique source name — when multiple results
    come from the same namespace (e.g. five Pinecone docs all from 'PCES_cardiology'),
    only the highest-scoring one is kept so the citations list stays compact.
    """
    # Compute composite scores first so sorting gives the best entry per source
    for d in enriched_docs:
        d["composite_score"] = round(d["relevance"] * 0.7 + d["reliability"] * 0.3, 1)

    # Sort highest-score first so the first occurrence of each source is the best
    enriched_docs_sorted = sorted(enriched_docs, key=lambda x: x["composite_score"], reverse=True)

    seen: set[str] = set()
    unique: list[dict] = []
    for d in enriched_docs_sorted:
        key = (d["source_raw"] or d["source_label"]).lower()
        if key not in seen:
            seen.add(key)
            unique.append(d)

    return unique[:max_citations]


# ---------------------------------------------------------------------------
# Steps 4, 6, 8 — Build structured citation list
# ---------------------------------------------------------------------------
def _build_structured_citations(
    ranked_docs: list[dict],
    raw_citations: list[str],
    lora_info: dict | None,
) -> list[dict]:
    """
    Build a machine-readable list of citation dicts suitable for the JSON response.
    Each entry contains label, source, relevance, reliability, url, etc.
    """
    structured: list[dict] = []

    # From ranked source documents
    for doc in ranked_docs:
        entry: dict[str, Any] = {
            "source_label": doc["source_label"],
            "source": doc["source_raw"] or doc["title"] or doc["source_label"],
            "source_type": doc["source_type"],
            "relevance_pct": doc["relevance"],
            "reliability_pct": doc["reliability"],
            "composite_score": doc.get("composite_score", 0),
        }
        if doc["department"]:
            entry["department"] = doc["department"]
        if doc["page"]:
            entry["page"] = str(doc["page"])
        if doc["url"]:
            entry["url"] = doc["url"]
        if doc["title"]:
            entry["title"] = doc["title"]
        if doc["excerpt"]:
            entry["excerpt"] = doc["excerpt"][:200]
        structured.append(entry)

    # From raw citation strings (parsed from tool response footers) that don't
    # already appear as source documents — catches external API citations (Step 8)
    existing_sources_lower = {c["source"].lower() for c in structured}
    existing_source_types = {c["source_type"] for c in structured}

    for raw in (raw_citations or []):
        norm = raw.lower()

        # Exact duplicate
        if norm in existing_sources_lower:
            continue

        raw_type = _type_from_raw_citation(raw)
        url_match = re.search(r'https?://\S+', raw)

        # If any existing source name (≥8 chars) appears inside this raw citation,
        # it's a re-statement of an already-structured entry.
        # e.g. "[PCES Pinecone KB] Dept: Cardiology | Source: PCES_cardiology | Relevance: 58%"
        #       contains "pces_cardiology" which is already in existing_sources_lower.
        if any(es and len(es) >= 8 and es in norm for es in existing_sources_lower):
            continue

        # If this source type is already represented AND no unique URL, skip.
        # Prevents generic re-statements like "Source: PCES Pinecone Knowledge Base"
        # from appearing when we already have a structured Pinecone entry.
        if raw_type in existing_source_types and not url_match:
            continue
        # Parse raw citation to extract label + URL (if present)
        label = _label_from_raw_citation(raw)
        url_match = re.search(r'https?://\S+', raw)
        entry = {
            "source_label": label,
            "source": raw,
            "source_type": _type_from_raw_citation(raw),
            "relevance_pct": 0,
            "reliability_pct": SOURCE_RELIABILITY.get(label, _DEFAULT_RELIABILITY),
            "composite_score": 0,
        }
        if url_match:
            entry["url"] = url_match.group(0)
        structured.append(entry)

    # Step 9 — LoRA model attribution
    if lora_info and lora_info.get("used"):
        structured.append({
            "source_label": "Department AI Model",
            "source": f"LoRA fine-tuned model ({lora_info.get('department', 'unknown')})",
            "source_type": "lora_model",
            "relevance_pct": 0,
            "reliability_pct": SOURCE_RELIABILITY.get("Department AI Model", 80),
            "composite_score": 0,
            "model_version": lora_info.get("model_version", ""),
            "department": lora_info.get("department", ""),
        })

    return structured


def _label_from_raw_citation(raw: str) -> str:
    """Map a raw citation string to a standardised source label."""
    low = raw.lower()
    # Doctor KB labels are already formatted — pass through as-is
    if low.startswith("dr.") and low.endswith(" kb"):
        return raw
    if "arxiv" in low:
        return "Clinical Research (arXiv)"
    if "wikipedia" in low or "wiki" in low:
        return "Medical Reference (Wikipedia)"
    if "tavily" in low or "web" in low:
        return "Web Research"
    if "pinecone" in low or "pces" in low:
        return "Organisation Knowledge Base"
    if "postgres" in low or "ehr" in low or "diagnosis" in low:
        return "Patient Medical Record (EHR)"
    if "uploaded" in low or "adhoc" in low or "patient document" in low:
        return "Uploaded Patient Document"
    return "Doctor's Knowledge Base"


def _type_from_raw_citation(raw: str) -> str:
    """Infer source_type from a raw citation string."""
    low = raw.lower()
    if "arxiv" in low:
        return "arxiv"
    if "wikipedia" in low or "wiki" in low:
        return "wikipedia"
    if "tavily" in low or "web" in low:
        return "tavily"
    if "pinecone" in low or "pces" in low:
        return "pinecone"
    if "postgres" in low or "ehr" in low:
        return "postgres"
    return "internal"


# ---------------------------------------------------------------------------
# Step 7 — Build HTML citations block (enhances the existing HTML builder)
# ---------------------------------------------------------------------------
def build_citations_html(
    structured_citations: list[dict],
    confidence_score: int = 0,
) -> str:
    """
    Build the compact HTML citation block appended to the response message.

    Single-section design: confidence badge + one-line-per-source list with
    hyperlinks (where URLs exist) and inline relevance/reliability chips.
    The expanded "Source Documents" card section has been removed to save
    screen real estate — all useful info is in the compact list.
    """
    parts: list[str] = []

    # Confidence badge
    if confidence_score > 0:
        conf_color = (
            "#28a745" if confidence_score >= 80
            else "#fd7e14" if confidence_score >= 60
            else "#dc3545"
        )
        conf_label = (
            "High" if confidence_score >= 80
            else "Medium" if confidence_score >= 60
            else "Low"
        )
        parts.append(
            f'<div style="margin-bottom:8px;">'
            f'<span style="background:{conf_color};color:#fff;border-radius:4px;'
            f'padding:2px 8px;font-size:11px;font-weight:600;">'
            f'Confidence: {confidence_score}% ({conf_label})</span></div>'
        )

    parts.append('<strong>Citations:</strong><ul style="margin:6px 0 0 16px;padding:0;">')
    for c in structured_citations:
        label = c.get("source_label", "Source")
        source = c.get("source", "")
        url = c.get("url", "")
        rel = c.get("relevance_pct", 0)
        reliability = c.get("reliability_pct", _DEFAULT_RELIABILITY)
        dept = c.get("department", "")

        # Source text — wrap in hyperlink when a URL is available,
        # otherwise show an 🏢 Internal badge for internal org sources
        display = source or label
        _INTERNAL_TYPES = {"pinecone", "internal", "main_rag", "adhoc_rag"}
        src_type = c.get("source_type", "")
        if url:
            source_html = f'<a href="{url}" target="_blank" style="color:#0066cc;">{display}</a>'
        elif src_type in _INTERNAL_TYPES:
            source_html = (
                f'<span>{display}</span>'
                f'<span style="background:#6f42c1;color:#fff;border-radius:3px;'
                f'padding:1px 5px;font-size:9px;margin-left:4px;">🏢 Internal</span>'
            )
        else:
            source_html = f'<span>{display}</span>'

        # Compact inline chips
        chips: list[str] = []
        if dept:
            chips.append(
                f'<span style="background:#e9ecef;border-radius:3px;padding:1px 5px;'
                f'font-size:10px;color:#495057;">🏥 {dept}</span>'
            )
        if rel:
            badge_color = '#28a745' if rel >= 70 else '#fd7e14' if rel >= 50 else '#6c757d'
            chips.append(
                f'<span style="background:{badge_color};color:#fff;border-radius:3px;'
                f'padding:1px 5px;font-size:10px;">{rel}% match</span>'
            )
        if reliability:
            rel_color = '#28a745' if reliability >= 85 else '#fd7e14' if reliability >= 70 else '#6c757d'
            chips.append(
                f'<span style="background:{rel_color};color:#fff;border-radius:3px;'
                f'padding:1px 5px;font-size:10px;">🛡️ {reliability}%</span>'
            )

        chips_html = ' '.join(chips)
        parts.append(
            f'<li style="margin-bottom:5px;list-style:disc;">'
            f'<span style="font-size:12px;color:#6c757d;">{label}:</span> '
            f'{source_html}'
            f'{(" &nbsp;" + chips_html) if chips_html else ""}'
            f'</li>'
        )
    parts.append('</ul>')

    return ''.join(parts)


# ---------------------------------------------------------------------------
# Main orchestrator — Steps 1–9 combined
# ---------------------------------------------------------------------------
def attach_citations(
    response: str,
    source_documents: list[dict],
    citations_raw: list[str],
    confidence_score: int = 0,
    department: str = "",
    session_id: str = "",
    lora_info: dict | None = None,
) -> dict:
    """
    Attach structured citations to a RAG response.

    Returns:
        {
            "structured_citations": [...],   # machine-readable list
            "citations_html": "...",         # HTML block for frontend
            "source_count": int,
            "top_source_label": str,
            "avg_reliability": float,
        }
    """
    try:
        # Step 2 — Extract metadata from each source document
        enriched: list[dict] = []
        for doc in source_documents:
            enriched.append(_extract_metadata(doc))

        # Step 3 — Rank and select top documents
        ranked = _rank_and_select(enriched)

        # Steps 4, 6, 8, 9 — Build structured citation list
        structured = _build_structured_citations(ranked, citations_raw, lora_info)

        # Step 7 — Build HTML
        html = build_citations_html(structured, confidence_score)

        # Summary stats
        avg_reliability = (
            round(sum(c.get("reliability_pct", 0) for c in structured) / len(structured), 1)
            if structured else 0
        )
        top_label = structured[0]["source_label"] if structured else "None"

        result = {
            "structured_citations": structured,
            "citations_html": html,
            "source_count": len(structured),
            "top_source_label": top_label,
            "avg_reliability": avg_reliability,
        }

        logger.info(
            "[CITATION] dept=%r  sources=%d  top=%r  avg_reliability=%.1f  confidence=%d",
            department, len(structured), top_label, avg_reliability, confidence_score,
        )

        return result

    except Exception as exc:
        logger.error("[CITATION] Unexpected error: %s", exc, exc_info=True)
        # Fail-open: return empty citations rather than crashing the response
        return {
            "structured_citations": [],
            "citations_html": "",
            "source_count": 0,
            "top_source_label": "None",
            "avg_reliability": 0,
        }


# ---------------------------------------------------------------------------
# Step 10 — Log citations to DB (fire-and-forget, PostgreSQL only, silent fail)
# ---------------------------------------------------------------------------
def log_citations(
    session_id: str,
    structured_citations: list[dict],
    department: str = "",
    prompt_snippet: str = "",
    confidence_score: int = 0,
) -> None:
    """Log citation data to PostgreSQL citation_log table (async, silent fail)."""
    def _worker():
        try:
            import json as _json
            from services.db_service import get_connection

            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                    CREATE TABLE IF NOT EXISTS citation_log (
                        id SERIAL PRIMARY KEY,
                        session_id TEXT,
                        department TEXT,
                        prompt_snippet TEXT,
                        citation_count INTEGER,
                        citations_json TEXT,
                        confidence_score INTEGER,
                        avg_reliability REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """)
                    avg_rel = (
                        round(sum(c.get("reliability_pct", 0) for c in structured_citations) / len(structured_citations), 1)
                        if structured_citations else 0
                    )
                    cur.execute(
                        """
                    INSERT INTO citation_log
                        (session_id, department, prompt_snippet, citation_count,
                         citations_json, confidence_score, avg_reliability)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                        (
                            session_id,
                            department,
                            prompt_snippet[:200],
                            len(structured_citations),
                            _json.dumps(structured_citations, default=str),
                            confidence_score,
                            avg_rel,
                        ),
                    )
                conn.commit()
        except Exception as exc:
            logger.debug("[CITATION] log_citations DB write failed (non-critical): %s", exc)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
