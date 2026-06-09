"""
Confidence Scoring Service — Phase 2-B: Confidence Scoring

Pipeline position (spec: 07_Implement Confidence Scoring v1.0):
  Safety / Guardrails → Confidence Scoring (this module) → Attach Citations

Implements all 11 steps from the spec:
  1.  Receive inputs (response, context, source_documents, validation_score, guardrail_status)
  2.  Retrieval Score (20%)   — avg similarity of retrieved docs
  3.  Evidence Score (20%)    — context substring overlap ratio
  4.  Validation Score (20%)  — pass-through from validation_service
  5.  Guardrail Score (15%)   — 100 if SAFE, 0 if BLOCKED/EMERGENCY
  6.  Model Confidence (15%)  — heuristic: hedging vs certainty language (no extra LLM call)
  7.  Consistency Score (10%) — multi-source agreement (≥2 docs → 90, else 50, else 0)
  8.  Weighted sum → 0–100 confidence score
  9.  Decision: DELIVER (≥80) / REVIEW (60–79) / REGENERATE (<60)
  10. Build confidence result object
  11. Log to confidence_log table — async, PostgreSQL only, silent fail
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from utils.error_handlers import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Weights (must sum to 1.0)
# ---------------------------------------------------------------------------
_WEIGHTS = {
    "retrieval":   0.20,
    "evidence":    0.20,
    "validation":  0.20,
    "guardrail":   0.15,
    "model_conf":  0.15,
    "consistency": 0.10,
}

# ---------------------------------------------------------------------------
# Step 2 — Retrieval Score
# Parse Relevance % from source_documents[i]["fields"]["Relevance"] = "73%"
# ---------------------------------------------------------------------------

def calculate_retrieval_score(source_documents: list[dict]) -> int:
    """Return average relevance score (0–100) from source document fields.

    Parses the 'Relevance' field which is formatted as a percent string
    (e.g. '73%').  Returns 0 when no documents are available.
    """
    if not source_documents:
        logger.debug("[CONFIDENCE] retrieval_score=0  (no source documents)")
        return 0

    scores: list[int] = []
    for doc in source_documents:
        fields = doc.get("fields", {})
        rel = fields.get("Relevance", "")
        try:
            scores.append(int(str(rel).rstrip("% ").strip()))
        except (ValueError, TypeError):
            pass

    if not scores:
        logger.debug("[CONFIDENCE] retrieval_score=0  (no parseable Relevance fields)")
        return 0

    avg = int(sum(scores) / len(scores))
    logger.debug("[CONFIDENCE] retrieval_score=%d  n_docs=%d", avg, len(scores))
    return avg


# ---------------------------------------------------------------------------
# Step 3 — Evidence Score
# Measure how much retrieved context overlaps with the LLM response.
# ---------------------------------------------------------------------------

def calculate_evidence_score(response: str, context: list[str]) -> int:
    """Return evidence overlap score (0–100).

    Uses the same fingerprinting approach as validation_service:
    takes the first 80 chars of each context snippet and checks if it
    appears verbatim in the response.  Returns the ratio of matching
    snippets × 100.
    """
    if not context:
        logger.debug("[CONFIDENCE] evidence_score=0  (empty context)")
        return 0

    resp_lower = response.lower()
    matched = 0
    for fragment in context:
        snippet = fragment.strip()[:80].lower().rstrip(".,:;!?")
        if len(snippet) >= 10 and snippet in resp_lower:
            matched += 1

    score = int((matched / len(context)) * 100)
    logger.debug(
        "[CONFIDENCE] evidence_score=%d  matched=%d/%d",
        score, matched, len(context),
    )
    return score


# ---------------------------------------------------------------------------
# Step 4 — Validation Score  (pass-through, already 0–100)
# ---------------------------------------------------------------------------

def get_validation_score(validation_score: int) -> int:
    """Clamp validation score to 0–100."""
    return max(0, min(100, int(validation_score)))


# ---------------------------------------------------------------------------
# Step 5 — Guardrail Score
# ---------------------------------------------------------------------------

def calculate_guardrail_score(guardrail_status: str) -> int:
    """Return 100 for SAFE, 0 for BLOCKED or EMERGENCY."""
    if guardrail_status == "SAFE":
        logger.debug("[CONFIDENCE] guardrail_score=100")
        return 100
    logger.debug("[CONFIDENCE] guardrail_score=0  status=%s", guardrail_status)
    return 0


# ---------------------------------------------------------------------------
# Step 6 — Model Confidence (heuristic — no extra LLM call)
# Baseline 70 pts; deduct 5 per hedging phrase; add 5 per certainty phrase.
# ---------------------------------------------------------------------------

_HEDGING_PHRASES: list[str] = [
    "may ", "might ", "possibly ", "perhaps ", "unclear ",
    "uncertain ", "not sure", "could be", "approximately",
    "it is possible", "in some cases", "it depends",
    "typically", "generally", "usually",
]

_CERTAINTY_PHRASES: list[str] = [
    "clearly ", "definitely ", "established ", "confirmed ",
    "proven ", "strongly recommended", "standard of care",
    "evidence shows", "studies confirm", "guidelines recommend",
    "consensus ", "well-documented",
]


def get_model_confidence(response: str) -> int:
    """Estimate LLM confidence from response language patterns (0–100).

    Starts at 70 and adjusts based on hedging/certainty language.
    Length bonus: well-developed responses (>500 chars) earn up to +10 pts.
    """
    score = 70
    r = response.lower()

    hedges = sum(1 for phrase in _HEDGING_PHRASES if phrase in r)
    certs  = sum(1 for phrase in _CERTAINTY_PHRASES if phrase in r)

    score -= hedges * 5
    score += certs  * 5

    if len(response) > 500:
        score += 5
    if len(response) > 1000:
        score += 5

    result = max(0, min(100, score))
    logger.debug(
        "[CONFIDENCE] model_confidence=%d  hedges=%d  certs=%d  len=%d",
        result, hedges, certs, len(response),
    )
    return result


# ---------------------------------------------------------------------------
# Step 7 — Consistency Score
# Multi-source agreement: more sources → higher confidence
# ---------------------------------------------------------------------------

def calculate_consistency_score(source_documents: list[dict], context: list[str]) -> int:
    """Return consistency score based on number of agreeing sources.

    ≥ 3 sources → 100
    2 sources   → 90
    1 source    → 50
    0 sources   → 0
    """
    n_sources = len(source_documents) if source_documents else len(context)
    if n_sources >= 3:
        score = 100
    elif n_sources == 2:
        score = 90
    elif n_sources == 1:
        score = 50
    else:
        score = 0
    logger.debug("[CONFIDENCE] consistency_score=%d  n_sources=%d", score, n_sources)
    return score


# ---------------------------------------------------------------------------
# Step 8 — Weighted sum
# ---------------------------------------------------------------------------

def calculate_weighted_score(sub_scores: dict[str, int]) -> int:
    """Compute weighted confidence score from sub-scores dict."""
    total: float = 0.0
    for key, weight in _WEIGHTS.items():
        total += sub_scores.get(key, 0) * weight
    return int(round(total))


# ---------------------------------------------------------------------------
# Step 9 — Decision Logic
# ---------------------------------------------------------------------------

def confidence_decision(score: int) -> str:
    """Map confidence score to pipeline decision.

    ≥ 80  → "DELIVER"     response is confident and safe
    60–79 → "REVIEW"      route to human review
    < 60  → "REGENERATE"  retry generation
    """
    if score >= 80:
        return "DELIVER"
    if score >= 60:
        return "REVIEW"
    return "REGENERATE"


# ---------------------------------------------------------------------------
# Step 10 — Orchestrator
# ---------------------------------------------------------------------------

def score_confidence(
    prompt: str,
    response: str,
    context: list[str],
    source_documents: list[dict] | None = None,
    validation_score: int = 100,
    guardrail_status: str = "SAFE",
) -> dict:
    """Run all 6 sub-scores and return a structured confidence result.

    Returns::

        {
            "score":    int,       # 0–100 weighted composite
            "decision": str,       # "DELIVER" | "REVIEW" | "REGENERATE"
            "breakdown": {
                "retrieval":   int,
                "evidence":    int,
                "validation":  int,
                "guardrail":   int,
                "model_conf":  int,
                "consistency": int,
            },
        }
    """
    _docs = source_documents or []

    # Run independent sub-scores in parallel
    sub_scores: dict[str, int] = {}
    score_fns: dict[str, Any] = {
        "retrieval":   lambda: calculate_retrieval_score(_docs),
        "evidence":    lambda: calculate_evidence_score(response, context),
        "validation":  lambda: get_validation_score(validation_score),
        "guardrail":   lambda: calculate_guardrail_score(guardrail_status),
        "model_conf":  lambda: get_model_confidence(response),
        "consistency": lambda: calculate_consistency_score(_docs, context),
    }

    with ThreadPoolExecutor(max_workers=6, thread_name_prefix="conf-score") as pool:
        futures = {pool.submit(fn): key for key, fn in score_fns.items()}
        for future in as_completed(futures):
            key = futures[future]
            try:
                sub_scores[key] = future.result()
            except Exception as exc:
                logger.error("[CONFIDENCE] sub-score %r raised: %s — defaulting to 0", key, exc)
                sub_scores[key] = 0

    final_score = calculate_weighted_score(sub_scores)
    decision = confidence_decision(final_score)

    logger.info(
        "[CONFIDENCE] score=%d  decision=%s  breakdown=%s",
        final_score, decision, sub_scores,
    )

    return {
        "score": final_score,
        "decision": decision,
        "breakdown": sub_scores,
    }


# ---------------------------------------------------------------------------
# Step 11 — Log Confidence Results (async, fire-and-forget)
# ---------------------------------------------------------------------------

_conf_table_ensured = threading.Event()


def _ensure_confidence_table() -> None:
    """Create confidence_log table if it does not exist yet."""
    try:
        from services.db_service import get_connection
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS confidence_log (
                        id          SERIAL PRIMARY KEY,
                        session_id  TEXT,
                        department  TEXT,
                        score       INTEGER,
                        decision    TEXT,
                        breakdown   TEXT,
                        prompt_snippet TEXT,
                        created_at  TIMESTAMP DEFAULT NOW()
                    )
                    """
                )
            conn.commit()
        logger.info("[CONFIDENCE] confidence_log table ensured")
        _conf_table_ensured.set()
    except Exception:
        pass  # non-critical — silent skip


def log_confidence(
    session_id: str,
    score: int,
    breakdown: dict,
    decision: str = "",
    department: str = "",
    prompt_snippet: str = "",
) -> None:
    """Persist a confidence result to PostgreSQL in a background daemon thread.

    Silently no-ops when PostgreSQL is unavailable.
    """

    def _insert() -> None:
        if not _conf_table_ensured.is_set():
            _ensure_confidence_table()
        if not _conf_table_ensured.is_set():
            return
        try:
            from services.db_service import get_connection, execute_query
            with get_connection() as conn:
                execute_query(
                    conn,
                    """
                    INSERT INTO confidence_log
                        (session_id, department, score, decision, breakdown, prompt_snippet)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id or "unknown",
                        department or "",
                        score,
                        decision or "",
                        json.dumps(breakdown),
                        (prompt_snippet or "")[:200],
                    ),
                )
                conn.commit()
            logger.debug(
                "[CONFIDENCE] logged  session=%s  score=%d  decision=%s",
                session_id, score, decision,
            )
        except Exception:
            pass  # non-critical — never surface DB errors

    threading.Thread(target=_insert, daemon=True, name="conf-log").start()
