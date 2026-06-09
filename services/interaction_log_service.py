"""
Interaction Log Service — Phase 2-B: Log Interaction (Step 09)

Pipeline position (spec: 09_Implement Log Interaction v1.0):
  Attach Citations (✅) → Log Interaction (this module) → Deliver Response

Implements all 15 steps from the spec:
  1.  Receive all pipeline outputs (prompt, response, routing, scores, tokens, timing)
  2.  Mask PII from prompt / response before storage (HIPAA precaution)
  3.  Standardise session / tenant / doctor / patient identifiers
  4.  Capture model name + LoRA model (when applicable)
  5.  Record token usage (actual from OpenAI or estimated from char count)
  6.  Record latency_ms (end-to-end wall-clock time)
  7.  Record validation score
  8.  Record guardrail status
  9.  Record confidence score
  10. Record sources used (list of source labels)
  11. Store original_response (pre-guardrail override) AND final_response
  12. Set error_flag + error_message when an exception was raised during the request
  13. Record deployment environment (prod / staging / dev)
  14. Write to interaction_log table (async, fire-and-forget, silent fail)
  15. Table partitioned by timestamp (monthly) — DDL includes partition comment for ops

Production enhancements:
  - Async write in daemon thread — never blocks the user response
  - Token estimation fallback (chars / 4) when LLM usage object unavailable
  - Graceful degrade if PostgreSQL is unreachable (non-critical warning)
  - Table auto-creation on first write
"""

from __future__ import annotations

import json
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any

from utils.error_handlers import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Step 2 — PII masking patterns (HIPAA precaution before DB storage)
# ---------------------------------------------------------------------------
_PII_PATTERNS: list[tuple[re.Pattern, str]] = [
    # US Social Security Number
    (re.compile(r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b"), "[SSN]"),
    # E-mail address
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "[EMAIL]"),
    # Phone number (10–11 digits, various separators)
    (re.compile(r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"), "[PHONE]"),
    # Date of birth patterns (e.g. "DOB: 01/01/1990", "born 01-01-1990")
    (re.compile(r"\b(?:dob|born|date of birth)[:\s]+\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b", re.I), "[DOB]"),
    # Credit-card-like 16-digit numbers
    (re.compile(r"\b\d{4}[-\s]\d{4}[-\s]\d{4}[-\s]\d{4}\b"), "[CARD]"),
]


def _mask_pii(text: str) -> str:
    """Remove recognised PII patterns before writing to the interaction log."""
    if not text:
        return text
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# ---------------------------------------------------------------------------
# Step 5 — Token usage helpers
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token ≈ 4 characters (GPT tokeniser heuristic)."""
    return max(1, len(text) // 4)


def _extract_token_usage(
    usage_obj: Any | None,
    prompt: str,
    response: str,
) -> dict:
    """Return token counts from an OpenAI usage object or fall back to estimates.

    The OpenAI/LangChain usage object exposes:
        usage_obj.input_tokens  (psycopg3 / openai>=1) or
        usage_obj.prompt_tokens (openai<1 / langchain callback)
        usage_obj.completion_tokens
        usage_obj.total_tokens
    """
    if usage_obj is not None:
        try:
            tokens_in = (
                getattr(usage_obj, "input_tokens", None)
                or getattr(usage_obj, "prompt_tokens", 0)
            )
            tokens_out = (
                getattr(usage_obj, "output_tokens", None)
                or getattr(usage_obj, "completion_tokens", 0)
            )
            total = getattr(usage_obj, "total_tokens", None) or (tokens_in + tokens_out)
            if tokens_in or tokens_out:
                return {
                    "tokens_input": int(tokens_in),
                    "tokens_output": int(tokens_out),
                    "tokens_total": int(total),
                }
        except Exception:
            pass  # fall through to estimation

    tokens_in = _estimate_tokens(prompt)
    tokens_out = _estimate_tokens(response)
    return {
        "tokens_input": tokens_in,
        "tokens_output": tokens_out,
        "tokens_total": tokens_in + tokens_out,
    }


# ---------------------------------------------------------------------------
# Step 1 — Assemble interaction log dict
# ---------------------------------------------------------------------------

def create_interaction_log(
    *,
    session_id: str = "",
    tenant_id: str = "",
    doctor_id: str = "",
    patient_id: str | None = None,
    department: str = "",
    prompt: str = "",
    original_response: str = "",
    final_response: str = "",
    validation_result: dict | None = None,
    guardrail_result: dict | None = None,
    confidence_result: dict | None = None,
    citation_result: dict | None = None,
    lora_info: dict | None = None,
    usage_obj: Any | None = None,
    latency_ms: int = 0,
    error_info: dict | None = None,
    routing_info: dict | None = None,
) -> dict:
    """Assemble a complete interaction log record from all pipeline outputs.

    All string fields are PII-masked before returning.  The caller passes this
    dict to ``log_interaction()`` for async persistence.
    """
    # Step 2 — PII mask
    safe_prompt = _mask_pii(prompt)
    safe_original = _mask_pii(original_response)
    safe_final = _mask_pii(final_response)

    # Step 3 — Identifiers
    try:
        from config import Config as _Cfg
        _tenant = tenant_id or _Cfg.TENANT_ID or "default"
        _model = _Cfg.LLM_MODEL_NAME or "unknown"
        _env = getattr(_Cfg, "FLASK_ENV", None) or __import__("os").getenv("FLASK_ENV", "production")
    except Exception:
        _tenant = tenant_id or "default"
        _model = "unknown"
        _env = "production"

    # Step 4 — Model & LoRA
    _lora_model = ""
    if lora_info and lora_info.get("used"):
        _lora_model = lora_info.get("department", "") or lora_info.get("model_version", "")

    # Step 5 — Token usage
    _tokens = _extract_token_usage(usage_obj, safe_prompt, safe_final)

    # Step 7–9 — Pipeline scores
    _v_score = (validation_result or {}).get("score", 0)
    _g_status = (guardrail_result or {}).get("status", "UNKNOWN")
    _c_score = (confidence_result or {}).get("score", 0)

    # Step 10 — Sources used
    _sources: list[str] = []
    if citation_result:
        for cit in citation_result.get("structured_citations", []):
            label = cit.get("source_label") or cit.get("source_type") or ""
            if label and label not in _sources:
                _sources.append(label)

    # Step 12 — Error info
    _error_flag = bool(error_info and error_info.get("error"))
    _error_message = (error_info or {}).get("message", "")

    # Step 6 — Latency (caller computes wall-clock ms)
    # Step 13 — Environment
    return {
        "session_id": session_id or "unknown",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tenant_id": _tenant,
        "doctor_id": doctor_id or "",
        "patient_id": patient_id or "",
        "department": department or "",
        "prompt": safe_prompt[:2000],           # cap at 2000 chars
        "original_response": safe_original[:4000],
        "final_response": safe_final[:4000],
        "confidence_score": _c_score,
        "validation_score": _v_score,
        "guardrail_status": _g_status,
        "model_name": _model,
        "lora_model": _lora_model,
        "tokens_input": _tokens["tokens_input"],
        "tokens_output": _tokens["tokens_output"],
        "tokens_total": _tokens["tokens_total"],
        "latency_ms": latency_ms,
        "sources_used": json.dumps(_sources),
        "error_flag": _error_flag,
        "error_message": _error_message[:500],
        "environment": _env,
    }


# ---------------------------------------------------------------------------
# Step 14–15 — Async PostgreSQL write
# ---------------------------------------------------------------------------

_table_ensured = threading.Event()


def _ensure_interaction_table() -> None:
    """Create interaction_log table if it does not exist yet.

    The table is partitioned by timestamp (monthly) in production:
        PARTITION BY RANGE (timestamp)
        → interaction_log_2026_01, interaction_log_2026_02, ...
    The DDL comment below is a reminder for ops — psycopg CREATE TABLE does not
    auto-create partitions; that is managed by the database admin / migration scripts.
    """
    try:
        from services.db_service import get_connection
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS interaction_log (
                        id                  SERIAL PRIMARY KEY,
                        session_id          TEXT,
                        timestamp           TIMESTAMPTZ DEFAULT NOW(),
                        tenant_id           TEXT,
                        doctor_id           TEXT,
                        patient_id          TEXT,
                        department          TEXT,
                        prompt              TEXT,
                        original_response   TEXT,
                        final_response      TEXT,
                        confidence_score    INTEGER,
                        validation_score    INTEGER,
                        guardrail_status    TEXT,
                        model_name          TEXT,
                        lora_model          TEXT,
                        tokens_input        INTEGER,
                        tokens_output       INTEGER,
                        tokens_total        INTEGER,
                        latency_ms          INTEGER,
                        sources_used        TEXT,
                        error_flag          BOOLEAN DEFAULT FALSE,
                        error_message       TEXT,
                        environment         TEXT,
                        created_at          TIMESTAMPTZ DEFAULT NOW()
                        -- NOTE: For production, partition this table by timestamp (monthly):
                        --   CREATE TABLE interaction_log_2026_01 PARTITION OF interaction_log
                        --   FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
                    )
                    """
                )
            conn.commit()
        logger.info("[INTERACTION_LOG] interaction_log table ensured")
        _table_ensured.set()
    except Exception as exc:
        logger.debug("[INTERACTION_LOG] table ensure failed (non-critical): %s", exc)


def log_interaction(log_dict: dict) -> None:
    """Persist an interaction log record to PostgreSQL in a background daemon thread.

    Silently no-ops when PostgreSQL is unavailable — logging is non-critical
    and must never block or error the main request path.

    Args:
        log_dict: The dict produced by ``create_interaction_log()``.
    """

    def _insert() -> None:
        if not _table_ensured.is_set():
            _ensure_interaction_table()
        if not _table_ensured.is_set():
            return  # DB unavailable — skip silently
        try:
            from services.db_service import get_connection, execute_query
            with get_connection() as conn:
                execute_query(
                    conn,
                    """
                    INSERT INTO interaction_log (
                        session_id, timestamp, tenant_id, doctor_id, patient_id,
                        department, prompt, original_response, final_response,
                        confidence_score, validation_score, guardrail_status,
                        model_name, lora_model,
                        tokens_input, tokens_output, tokens_total,
                        latency_ms, sources_used,
                        error_flag, error_message, environment
                    ) VALUES (
                        ?, ?, ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?,
                        ?, ?,
                        ?, ?, ?,
                        ?, ?,
                        ?, ?, ?
                    )
                    """,
                    (
                        log_dict["session_id"],
                        log_dict["timestamp"],
                        log_dict["tenant_id"],
                        log_dict["doctor_id"],
                        log_dict["patient_id"],
                        log_dict["department"],
                        log_dict["prompt"],
                        log_dict["original_response"],
                        log_dict["final_response"],
                        log_dict["confidence_score"],
                        log_dict["validation_score"],
                        log_dict["guardrail_status"],
                        log_dict["model_name"],
                        log_dict["lora_model"],
                        log_dict["tokens_input"],
                        log_dict["tokens_output"],
                        log_dict["tokens_total"],
                        log_dict["latency_ms"],
                        log_dict["sources_used"],
                        log_dict["error_flag"],
                        log_dict["error_message"],
                        log_dict["environment"],
                    ),
                )
                conn.commit()

            logger.info(
                "[INTERACTION_LOG] session=%r  dept=%r  latency=%dms  tokens=%d  "
                "confidence=%d  validation=%d  guardrail=%s  lora=%r  error=%s",
                log_dict["session_id"],
                log_dict["department"],
                log_dict["latency_ms"],
                log_dict["tokens_total"],
                log_dict["confidence_score"],
                log_dict["validation_score"],
                log_dict["guardrail_status"],
                log_dict["lora_model"] or "none",
                log_dict["error_flag"],
            )
        except Exception as exc:
            logger.warning(
                "[INTERACTION_LOG] DB write failed (non-critical): %s", exc
            )

    t = threading.Thread(target=_insert, daemon=True)
    t.start()
