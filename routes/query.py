"""
Query Blueprint

Routes:
  POST /generate_summary  — generate clinical summary + conclusion from transcription
  POST /plain_english     — rewrite a query in plain English
  POST /data              — primary JSON query endpoint
  POST /data-html         — HTML-formatted query endpoint

Helper functions (private to this module):
  _clean_response_text()
  _generate_full_html_response()
  _parse_enhanced_response()

Shared resources accessed via current_app.config:
  LLM_INSTANCE      — ChatOpenAI instance
  RAG_MANAGER       — TwoStoreRAGManager (may be None)
  INTEGRATED_RAG    — IntegratedMedicalRAG (may be None)
  SCOPE_GUARD       — DomainScopeGuard (may be None)
  EMBEDDINGS        — OpenAIEmbeddings instance
  TEXT_SPLITTER     — RecursiveCharacterTextSplitter instance
  LAST_SESSION_FOLDER — current session folder (set by disciplines_bp index route)
  DISCIPLINES_CONFIG  — loaded disciplines JSON
  MEDICAL_ROUTER      — MedicalQueryRouter instance (set at startup)
"""

from __future__ import annotations

import glob as glob_module
import os
import re
import threading
import time as _time
import traceback
from datetime import datetime, timezone

from flask import Blueprint, current_app, jsonify, make_response, request
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document

from config import Config
from utils.error_handlers import get_logger, handle_route_errors

logger = get_logger(__name__)

query_bp = Blueprint("query_bp", __name__)

# ---------------------------------------------------------------------------
# Phase 2-A trace collector
# ---------------------------------------------------------------------------
# Each request gets its own list of trace events stored in thread-local storage.
# Call _trace(icon, step, detail) anywhere in the request to append an entry.
# At response time, flush_trace() drains and returns the list.

_tls = threading.local()

# DEV_MODE: read once at import time; can be toggled by restarting the server.
_DEV_MODE: bool = os.getenv("DEV_MODE", "false").strip().lower() == "true"


def _trace(icon: str, step: str, detail: str = "") -> None:
    """Append a structured trace event to the current request's trace list.

    Events are only collected when DEV_MODE=true; in production the list stays
    empty so no trace data is ever sent to the client.
    """
    if not _DEV_MODE:
        return
    if not hasattr(_tls, "events"):
        _tls.events = []
    _tls.events.append({
        "t": round(_time.monotonic() * 1000),   # ms since process start (for delta calc in JS)
        "icon": icon,
        "step": step,
        "detail": detail,
    })
    logger.info("[PHASE2A] %s %s  %s", icon, step, detail)


def _flush_trace() -> list:
    """Return and clear the trace list for this request."""
    events = getattr(_tls, "events", [])
    _tls.events = []
    # Compute elapsed_ms relative to first event
    if events:
        t0 = events[0]["t"]
        for e in events:
            e["elapsed_ms"] = e["t"] - t0
    return events


def _dev_mode_enabled() -> bool:
    """Expose DEV_MODE flag to response payloads so the UI can render its toggle."""
    return _DEV_MODE


# ---------------------------------------------------------------------------
# Dept/LoRA availability cache — built once at first request, never rescanned
# ---------------------------------------------------------------------------
_dept_lora_cache: dict | None = None   # {"all": [...], "with_model": [...]}
_dept_lora_cache_lock = threading.Lock()


def _get_dept_lora_availability() -> dict:
    """Return cached dict of all departments and which have LoRA models.

    The scan (32 DB + disk lookups) runs only once per process lifetime.
    Subsequent calls return the cached result instantly.
    """
    global _dept_lora_cache
    if _dept_lora_cache is not None:
        return _dept_lora_cache
    with _dept_lora_cache_lock:
        if _dept_lora_cache is not None:   # double-checked locking
            return _dept_lora_cache
        try:
            from sft_experiment_manager import DEPARTMENTS as _DEPTS, get_best_model_path_for_dept as _gmp
            all_depts = list(_DEPTS.keys())
            with_model = [d for d in all_depts if _gmp(d)]
        except Exception:
            all_depts, with_model = [], []
        _dept_lora_cache = {"all": all_depts, "with_model": with_model}
        logger.info("[PHASE2A] dept/LoRA cache built: %d depts, %d with model: %s",
                    len(all_depts), len(with_model), with_model)
    return _dept_lora_cache


try:
    from sft_experiment_manager import add_sme_queue_entry as _add_sme_queue_entry, DEPARTMENTS as _DEPARTMENTS
    _SME_QUEUE_AVAILABLE = True
except Exception:
    _DEPARTMENTS = {}
    _SME_QUEUE_AVAILABLE = False

# ---------------------------------------------------------------------------
# Phase 2-B: Validate Response (optional — graceful degrade if unavailable)
# ---------------------------------------------------------------------------
try:
    from services.validation_service import (
        validate_response as _validate_response,
        log_validation as _log_validation,
    )
    _VALIDATION_AVAILABLE = True
    logger.info("ValidationService loaded — Phase 2-B response validation enabled")
except Exception as _val_import_exc:
    _VALIDATION_AVAILABLE = False
    logger.warning("ValidationService unavailable: %s — validation skipped", _val_import_exc)

    def _validate_response(*_a, **_kw):  # type: ignore[misc]
        return {"score": 100, "decision": "PASS", "results": {}, "flags": []}

    def _log_validation(*_a, **_kw):  # type: ignore[misc]
        pass

# ---------------------------------------------------------------------------
# Phase 2-B: Safety / Guardrails (optional — graceful degrade if unavailable)
# ---------------------------------------------------------------------------
try:
    from services.guardrails_service import (
        run_guardrails as _run_guardrails,
        log_guardrail_event as _log_guardrail,
    )
    _GUARDRAILS_AVAILABLE = True
    logger.info("GuardrailsService loaded — Phase 2-B safety guardrails enabled")
except Exception as _guard_import_exc:
    _GUARDRAILS_AVAILABLE = False
    logger.warning("GuardrailsService unavailable: %s — guardrails skipped", _guard_import_exc)

    def _run_guardrails(*_a, **_kw):  # type: ignore[misc]
        return {"status": "SAFE", "results": {}, "flags": [], "emergency_response": None}

    def _log_guardrail(*_a, **_kw):  # type: ignore[misc]
        pass

# ---------------------------------------------------------------------------
# Phase 2-B: Confidence Scoring (optional — graceful degrade if unavailable)
# ---------------------------------------------------------------------------
try:
    from services.confidence_service import (
        score_confidence as _score_confidence,
        log_confidence as _log_confidence,
    )
    _CONFIDENCE_AVAILABLE = True
    logger.info("ConfidenceService loaded — Phase 2-B confidence scoring enabled")
except Exception as _conf_import_exc:
    _CONFIDENCE_AVAILABLE = False
    logger.warning("ConfidenceService unavailable: %s — confidence scoring skipped", _conf_import_exc)

    def _score_confidence(*_a, **_kw):  # type: ignore[misc]
        return {"score": 80, "decision": "DELIVER", "breakdown": {}}

    def _log_confidence(*_a, **_kw):  # type: ignore[misc]
        pass

# ---------------------------------------------------------------------------
# Phase 2-B: Attach Citations (optional — graceful degrade if unavailable)
# ---------------------------------------------------------------------------
try:
    from services.citation_service import (
        attach_citations as _attach_citations,
        log_citations as _log_citations,
    )
    _CITATIONS_AVAILABLE = True
    logger.info("CitationService loaded — Phase 2-B attach citations enabled")
except Exception as _cit_import_exc:
    _CITATIONS_AVAILABLE = False
    logger.warning("CitationService unavailable: %s — citations passthrough", _cit_import_exc)

    def _attach_citations(*_a, **_kw):  # type: ignore[misc]
        return {
            "structured_citations": [],
            "citations_html": "",
            "source_count": 0,
            "top_source_label": "None",
            "avg_reliability": 0,
        }

    def _log_citations(*_a, **_kw):  # type: ignore[misc]
        pass

# ---------------------------------------------------------------------------
# Phase 2-B: Log Interaction (optional — graceful degrade if unavailable)
# ---------------------------------------------------------------------------
try:
    from services.interaction_log_service import (
        create_interaction_log as _create_interaction_log,
        log_interaction as _log_interaction,
    )
    _INTERACTION_LOG_AVAILABLE = True
    logger.info("InteractionLogService loaded — Phase 2-B interaction logging enabled")
except Exception as _ilog_import_exc:
    _INTERACTION_LOG_AVAILABLE = False
    logger.warning("InteractionLogService unavailable: %s — interaction logging skipped", _ilog_import_exc)

    def _create_interaction_log(*_a, **_kw) -> dict:  # type: ignore[misc]
        return {}

    def _log_interaction(*_a, **_kw) -> None:  # type: ignore[misc]
        pass

try:
    from services.session_service import session_service as _session_svc
    _SESSION_SVC_AVAILABLE = True
    logger.info("SessionService loaded — session completion tracking enabled")
except Exception as _sess_import_exc:
    _SESSION_SVC_AVAILABLE = False
    logger.warning("SessionService unavailable: %s — session tracking skipped", _sess_import_exc)
    _session_svc = None  # type: ignore[assignment]

# Explicit mapping for pces_role values that don't match DEPARTMENTS keys directly
_ROLE_TO_DEPT: dict[str, str] = {
    "ophthalmologist": "Ophthalmology",
    "opthalmologist": "Ophthalmology",       # typo present in live DB
    "ophthalmology": "Ophthalmology",
    "cardiologist": "Cardiology",
    "neurologist": "Neurology",
    "pulmonologist": "Pulmonology",
    "gastroenterologist": "Gastroenterology",
    "nephrologist": "Nephrology",
    "oncologist": "Oncology",
    "hematologist": "Hematology",
    "dermatologist": "Dermatology",
    "psychiatrist": "Psychiatry",
    "pediatrician": "Pediatrics",
    "rheumatologist": "Rheumatology",
    "urologist": "Urology",
    "hepatologist": "Hepatology",
    "radiologist": "Radiology",
    "anesthesiologist": "Anesthesiology",
    "geriatrician": "Geriatrics",
    "dentist": "Dentistry",
    "general surgeon": "General Medicine",
    "general_surgeon": "General Medicine",
    "family medicine": "Family Medicine",
    "family_medicine": "Family Medicine",
    "admin": "",
}


def _normalize_department(role: str) -> str:
    """Map a pces_role value (e.g. 'Ophthalmologist', 'CARDIOLOGIST') to a canonical
    DEPARTMENTS key (e.g. 'Ophthalmology', 'Cardiology') so SME queue entries are
    filtered correctly by the domain dropdown."""
    if not role:
        return ""
    role_lower = role.strip().lower()

    # 1. Direct case-insensitive match against DEPARTMENTS keys
    for dept in _DEPARTMENTS:
        if dept.lower() == role_lower:
            return dept

    # 2. Replace underscores with spaces then retry (handles FAMILY_MEDICINE)
    role_spaced = role_lower.replace("_", " ")
    for dept in _DEPARTMENTS:
        if dept.lower() == role_spaced:
            return dept

    # 3. Explicit role → department map
    mapped = _ROLE_TO_DEPT.get(role_lower) or _ROLE_TO_DEPT.get(role_spaced)
    if mapped is not None:
        return mapped

    # 4. Fallback: return the original value unchanged
    return role

# ---------------------------------------------------------------------------
# Shared resource accessors
# ---------------------------------------------------------------------------

def _get_llm():
    return current_app.config.get("LLM_INSTANCE")


def _get_rag_manager():
    return current_app.config.get("RAG_MANAGER")


def _get_integrated_rag():
    return current_app.config.get("INTEGRATED_RAG")


def _get_scope_guard():
    return current_app.config.get("SCOPE_GUARD")


def _get_embeddings():
    return current_app.config.get("EMBEDDINGS")


def _get_session_folder() -> str | None:
    return current_app.config.get("LAST_SESSION_FOLDER")


def _get_disciplines_config() -> dict:
    return current_app.config.get("DISCIPLINES_CONFIG") or {}


def _get_medical_router():
    return current_app.config.get("MEDICAL_ROUTER")


def _queue_sme_entry(prompt: str, response_text: str, doctor_name: str, doctor_department: str) -> None:
    """Fire-and-forget: insert a chat Q&A pair into the SME review queue in a background thread."""
    if not _SME_QUEUE_AVAILABLE or not prompt or not response_text:
        return

    normalized_dept = _normalize_department(doctor_department)

    def _insert():
        try:
            _add_sme_queue_entry(
                prompt=prompt,
                response_text=response_text,
                domain=normalized_dept or "",
                doctor_name=doctor_name or "",
            )
        except Exception as exc:
            logger.warning("SME queue auto-insert failed (non-critical): %s", exc)

    threading.Thread(target=_insert, daemon=True).start()


# ---------------------------------------------------------------------------
# Local helpers (private)
# ---------------------------------------------------------------------------

def _get_latest_vector_db() -> str | None:
    """Find the most recently modified vector DB folder."""
    vector_dbs_folder = Config.VECTOR_DB_PATH
    vector_dbs = glob_module.glob(os.path.join(vector_dbs_folder, "*"))
    if not vector_dbs:
        return None
    return max(vector_dbs, key=os.path.getmtime)


def _create_contextual_llm(patient_context: str | None = None):
    """Create an LLM instance configured with optional patient context."""
    from langchain_openai import ChatOpenAI

    base_msg = Config.MEDICAL_SYSTEM_MESSAGE
    if patient_context:
        system_msg = (
            f"Patient Context: {patient_context}\n\n{base_msg} "
            "Always consider the patient context when providing medical advice and "
            "recommendations. Tailor your responses to the specific patient demographics, "
            "conditions, and medical history provided."
        )
    else:
        system_msg = base_msg

    contextual_llm = ChatOpenAI(
        api_key=os.getenv("openai_api_key"),
        base_url=os.getenv("base_url"),
        model_name=os.getenv("llm_model_name"),
        temperature=Config.LLM_DEFAULT_TEMPERATURE,
        request_timeout=Config.LLM_REQUEST_TIMEOUT,
    )
    contextual_llm._system_message = system_msg  # type: ignore[attr-defined]
    return contextual_llm


def _enhance_with_citations(results: list) -> str:
    pdf_citations: set[str] = set()
    url_citations: set[str] = set()
    org_citations: set[str] = set()

    for doc in results:
        metadata = getattr(doc, "metadata", {})
        doc_type = metadata.get("type")

        if doc_type == "pdf":
            pdf_citations.add(
                f"PDF: {metadata.get('source', 'Unknown PDF')} (Page {metadata.get('page', 'Unknown Page')})"
            )
        elif doc_type == "url":
            url_citations.add(f"URL: {metadata.get('source', 'Unknown URL')}")
        elif doc_type == "organization_pdf":
            discipline = metadata.get("discipline", "Unknown Discipline")
            org_citations.add(
                f"Organization KB - {discipline.replace('_', ' ').title()}: "
                f"{metadata.get('source', 'Unknown Document')} "
                f"(Page {metadata.get('page', 'Unknown Page')})"
            )

    all_citations = pdf_citations | url_citations | org_citations
    return "\n".join(all_citations) or "No citations available"


def _get_discipline_vector_db_path(discipline_id: str) -> str | None:
    cfg = _get_disciplines_config()
    for discipline in cfg.get("disciplines", []):
        if discipline["id"] == discipline_id:
            return discipline.get("vector_db_path", "")
    return None


# ---------------------------------------------------------------------------
# Helper functions (moved from main.py, private to blueprint)
# ---------------------------------------------------------------------------

def _clean_response_text(text: str) -> str:
    """Remove emojis and clean response text while preserving line breaks."""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002600-\U000026FF"
        "\U00002700-\U000027BF"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" +\n", "\n", text)
    return text.strip()


def _generate_full_html_response(result_data: dict) -> str:
    """Return a guaranteed vertical-layout HTML block for the query response."""
    medical_summary = result_data.get("medical_summary", "No medical summary available.")
    sources = result_data.get("sources", [])
    tool_info = result_data.get("tool_info", {})
    primary_tool = tool_info.get("primary_tool", "Unknown")
    confidence = tool_info.get("confidence", "Unknown")
    tools_used = tool_info.get("tools_used", "N/A")
    reasoning = tool_info.get("reasoning", "No reasoning provided.")

    sources_html = (
        "<ul>" + "".join(f"<li>{s}</li>" for s in sources) + "</ul>"
        if sources
        else "<p>No sources available.</p>"
    )

    return f"""
<div style="display:block; width:100%; max-width:100%; line-height:1.6;">
  <h3 style="color:#007bff; font-size:20px; margin-bottom:10px;">📋 Medical Summary</h3>
  <div style="background:#e3f2fd; padding:15px; border-radius:8px; margin-bottom:30px;">
    {medical_summary}
  </div>

  <h3 style="color:#6f42c1; font-size:20px; margin-bottom:10px;">📖 Sources</h3>
  <div style="background:#f8f9fa; padding:15px; border-radius:8px; margin-bottom:30px;">
    {sources_html}
  </div>

  <h3 style="color:#ff6600; font-size:20px; margin-bottom:10px;">🔧 Tool Selection &amp; Query Routing</h3>
  <div style="background:#fff3cd; padding:15px; border-radius:8px;">
    <p><strong>Primary Tool:</strong> {primary_tool}</p>
    <p><strong>Confidence:</strong> {confidence}</p>
    <p><strong>Tools Used:</strong> {tools_used}</p>
    <p><strong>Reasoning:</strong> {reasoning}</p>
  </div>
</div>
"""


def _parse_enhanced_response(
    answer: str,
    routing_info: dict,
    tools_used: list,
    explanation: str,
) -> dict:
    """Parse enhanced HTML response and extract structured data for the 3-section layout."""
    result_data: dict = {
        "medical_summary": "No medical summary available.",
        "sources": [],
        "tool_info": {},
    }

    if "<div" in answer and "<h4" in answer:
        medical_match = re.search(
            r"<h4[^>]*>.*?Medical Summary.*?</h4><div[^>]*>(.*?)</div>",
            answer,
            re.DOTALL | re.IGNORECASE,
        )
        if medical_match:
            result_data["medical_summary"] = re.sub(
                r"<[^>]+>", "", medical_match.group(1)
            ).strip()

        sources_match = re.search(
            r"<h4[^>]*>.*?Sources.*?</h4><div[^>]*>(.*?)</div>",
            answer,
            re.DOTALL | re.IGNORECASE,
        )
        if sources_match:
            sources_content = sources_match.group(1)
            link_matches = re.findall(
                r'<a[^>]+href="([^"]+)"[^>]*>([^<]+)</a>[^<]*\(([^)]+)\)',
                sources_content,
            )
            for _url, title, source_type in link_matches:
                result_data["sources"].append(f"{title} ({source_type})")
            if not result_data["sources"]:
                plain = re.sub(r"<[^>]+>", "", sources_content).strip()
                if plain:
                    result_data["sources"].append(plain)

        tool_match = re.search(
            r"<h4[^>]*>.*?Tool Selection.*?</h4><div[^>]*>(.*?)</div>",
            answer,
            re.DOTALL | re.IGNORECASE,
        )
        if tool_match:
            tool_content = tool_match.group(1)
            primary_tool_m = re.search(
                r"Primary Tool:\s*<strong>([^<]+)</strong>", tool_content
            )
            confidence_m = re.search(
                r"Confidence:\s*<span[^>]*>([^<]+)</span>", tool_content
            )
            reasoning_m = re.search(r"Reasoning:\s*([^<]+?)(?:<|$)", tool_content)
            result_data["tool_info"] = {
                "primary_tool": (
                    primary_tool_m.group(1)
                    if primary_tool_m
                    else routing_info.get("primary_tool", "Unknown")
                ),
                "confidence": (
                    confidence_m.group(1)
                    if confidence_m
                    else routing_info.get("confidence", "Unknown")
                ),
                "reasoning": (
                    reasoning_m.group(1).strip()
                    if reasoning_m
                    else routing_info.get("reasoning", "No reasoning provided.")
                ),
            }
    else:
        result_data["medical_summary"] = (
            answer[:500] + "..." if len(answer) > 500 else answer
        )
        primary_tool = routing_info.get("primary_tool", "Unknown")
        confidence = routing_info.get("confidence", "Unknown")
        if isinstance(confidence, str):
            confidence_map = {
                "high": "High (≈90%)",
                "medium": "Medium (≈70%)",
                "low": "Low (≈50%)",
            }
            confidence = confidence_map.get(confidence.lower(), confidence)
        elif isinstance(confidence, (int, float)):
            confidence = f"{confidence}%"
        result_data["tool_info"] = {
            "primary_tool": primary_tool,
            "confidence": confidence,
            "reasoning": routing_info.get(
                "reasoning", explanation or "No reasoning provided."
            ),
        }
        result_data["sources"] = ["No specific sources identified."]

    return result_data


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@query_bp.route("/generate_summary", methods=["POST"])
@handle_route_errors
def generate_summary():
    """
    Generate medical summary and conclusion from transcription text.
    Expects: transcription, doctor_name, patient_name
    Returns: summary and conclusion
    """
    try:
        data = request.get_json()
        transcription = data.get("transcription", "").strip()
        doctor_name = data.get("doctor_name", "").strip()
        patient_name = data.get("patient_name", "").strip()

        if not transcription:
            return (
                jsonify({"success": False, "error": "No transcription text provided"}),
                400,
            )

        logger.info(
            "Generating summary for patient: %s by Dr. %s", patient_name, doctor_name
        )
        llm = _get_llm()

        summary_prompt = f"""
        As a medical AI assistant, please analyze the following patient consultation transcript and provide a professional medical summary.
        
        Patient: {patient_name if patient_name else 'Not specified'}
        Doctor: {doctor_name if doctor_name else 'Not specified'}
        Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Transcript:
        {transcription}
        
        Please provide:
        1. A concise clinical summary highlighting key medical information, symptoms, findings, and discussions
        2. Professional conclusions with recommendations, follow-up actions, or treatment plans mentioned
        
        If the transcript does not contain relevant medical information, please provide appropriate default responses indicating the lack of medical content.
        
        When mentioning any dates, use the format "Month Day, Year" (e.g., April 10, 2026).
        
        Format your response exactly as:
        SUMMARY:
        [Provide a clear, professional summary of the medical consultation]
        
        CONCLUSION:
        [Provide conclusions, recommendations, and any follow-up actions mentioned]
        """

        try:
            logger.info("Generating medical summary from transcription...")
            ai_response = llm.invoke(summary_prompt)
            ai_content = (
                ai_response.content.strip()
                if hasattr(ai_response, "content")
                else str(ai_response).strip()
            )
            logger.info("AI summary generated. Length: %d characters", len(ai_content))

            summary_parts = ai_content.split("CONCLUSION:")
            if len(summary_parts) == 2:
                summary = summary_parts[0].replace("SUMMARY:", "").strip()
                conclusion = summary_parts[1].strip()
            else:
                lines = ai_content.split("\n")
                summary_lines: list[str] = []
                conclusion_lines: list[str] = []
                in_conclusion = False

                for line in lines:
                    if "CONCLUSION" in line.upper():
                        in_conclusion = True
                        continue
                    elif "SUMMARY" in line.upper():
                        in_conclusion = False
                        continue
                    if in_conclusion:
                        conclusion_lines.append(line)
                    else:
                        summary_lines.append(line)

                summary = "\n".join(summary_lines).strip()
                conclusion = "\n".join(conclusion_lines).strip()

                if not summary and not conclusion:
                    summary = (
                        "The consultation transcript provided does not contain any "
                        "relevant medical information, symptoms, findings, or discussions "
                        "related to a patient's health."
                    )
                    conclusion = (
                        "As there is no pertinent information available in the transcript, "
                        "no medical conclusions, recommendations, or follow-up actions can "
                        "be provided. It is recommended to ensure accurate and detailed "
                        "documentation of patient consultations for proper medical assessment and care."
                    )

        except Exception as exc:
            logger.error("Error generating summary: %s", exc)
            return (
                jsonify({"success": False, "error": f"Summary generation failed: {exc}"}),
                500,
            )

        return jsonify({"success": True, "summary": summary, "conclusion": conclusion})

    except Exception as exc:
        logger.error("Unexpected error in generate_summary: %s", exc)
        traceback.print_exc()
        return (
            jsonify({"success": False, "error": f"An unexpected error occurred: {exc}"}),
            500,
        )


@query_bp.route("/api/session/patient_change", methods=["POST"])
def session_patient_change():
    """
    Called by the frontend immediately when the user selects a different patient.
    Triggers session completion right at patient selection — not on the next query.
    """
    data = request.get_json(silent=True) or {}
    new_patient_id = data.get("patient_id", "").strip()
    session_id = _get_session_folder() or "guest"

    if _session_svc and new_patient_id:
        completed = _session_svc.on_patient_change(session_id, new_patient_id)
        return jsonify({"status": "ok", "session_completed": completed})

    return jsonify({"status": "ok", "session_completed": False})


@query_bp.route("/plain_english", methods=["POST"])
@handle_route_errors
def plain_english():
    user_text = request.json.get("text", "")
    if not user_text:
        return jsonify({"refined_text": "", "message": "No input provided."})

    try:
        llm = _get_llm()
        prompt = (
            f"Rewrite the following question in plain English for better clarity:\n\n{user_text}"
        )
        response = llm.invoke(prompt)
        refined_text = (
            response.content.strip()
            if hasattr(response, "content")
            else str(response).strip()
        )
        return jsonify({"refined_text": refined_text})
    except Exception as exc:
        return jsonify({"refined_text": "", "message": f"Error: {exc}"})


@query_bp.route("/data", methods=["POST"])
@handle_route_errors
def handle_query():
    """Original JSON endpoint for the UI — returns JSON responses."""
    import time as _time_module
    _req_start: float = _time_module.time()   # Step 6: end-to-end latency start
    user_input = request.json.get("data", "")
    patient_problem = request.json.get("patient_problem", "").strip()
    doctor_name = (request.json.get("doctor_name") or "").strip()
    doctor_department = (request.json.get("doctor_department") or "").strip()
    adhoc_rag_ready = bool(request.json.get("adhoc_rag_ready", False))

    if not user_input:
        return jsonify(
            {
                "response": False,
                "message": "Please provide a valid input to get a medical response.",
            }
        )

    # ── Extract adhoc context from request and inject into AdHocRAG_Search ──
    _doctor_id: str = (request.json.get("doctor_id") or "").strip()
    _patient_id: str | None = (request.json.get("patient_id") or "").strip() or None
    try:
        from tools import _set_adhoc_context as _sac
        from services.context_service import context_service as _ctx_svc
        from config import Config as _Cfg
        _rag_mgr = _get_rag_manager()
        _sac(
            rag_manager=_rag_mgr,
            tenant_id=_Cfg.TENANT_ID,
            doctor_id=_doctor_id,
            patient_id=_patient_id,
            doctor_name=doctor_name,
        )
    except Exception as _adhoc_init_exc:
        logger.debug("AdHoc context injection skipped: %s", _adhoc_init_exc)

    scope_guard = _get_scope_guard()
    guard_disclaimer = ""
    if scope_guard is not None:
        _guard_status, similarity_score, _guard_msg = scope_guard.check(user_input)
        if _guard_status == "rejected":
            logger.warning(
                "Query rejected by scope guard (score=%.3f): %r",
                similarity_score,
                user_input[:80],
            )
            return jsonify(
                {
                    "response": False,
                    "message": _guard_msg,
                    "out_of_scope": True,
                    "similarity_score": round(similarity_score, 4),
                    "validation": {
                        "score": 0,
                        "decision": "REJECTED",
                        "flags": ["Query rejected by scope guard — outside medical scope"],
                    },
                }
            )
        elif _guard_status == "general_medical":
            logger.info(
                "General medical query (score=%.3f) — answering with disclaimer",
                similarity_score,
            )
            guard_disclaimer = _guard_msg

    if patient_problem:
        logger.info("Using patient context as system message: %r", patient_problem)

    # Enrich the RAG search query with patient problem so retrieval is patient-context-aware
    if patient_problem:
        query_input = f"{user_input}. Patient problem: {patient_problem}"
        logger.info("LLM search query enriched with patient problem")
    else:
        query_input = user_input
    integrated_rag_system = _get_integrated_rag()
    rag_manager = _get_rag_manager()
    embeddings = _get_embeddings()
    llm = _get_llm()
    disciplines_config = _get_disciplines_config()
    medical_router = _get_medical_router()

    try:
        # ─────────────────────────────────────────────────────────────────────
        # PHASE 2-A  –  01 Hybrid Router  →  02 Load Dept LoRA
        # ─────────────────────────────────────────────────────────────────────

        # ── TRACE: request entry ──────────────────────────────────────────────
        _trace("►", "ENTRY",
               f"query: {user_input[:100]!r} | doctor_dept: {doctor_department or '(none)'} | app_dept: {current_app.config.get('ACTIVE_DEPARTMENT') or '(none)'}")

        # Resolve active department: prefer request payload, then app-level store
        active_dept = (
            doctor_department.strip()
            or current_app.config.get("ACTIVE_DEPARTMENT", "")
        )

        # Build available-dept list for trace: all known depts + flag which have LoRA models
        _dept_avail = _get_dept_lora_availability()
        _all_depts = _dept_avail["all"]
        _depts_with_model = _dept_avail["with_model"]

        _dept_summary = (
            f"resolved → {active_dept or '(none)'} | "
            f"available ({len(_all_depts)}): {', '.join(_all_depts) or 'none'} | "
            f"LoRA models ({len(_depts_with_model)}): {', '.join(_depts_with_model) or 'none'}"
        )
        _trace("🏥", "01-SELECTS-DEPT", _dept_summary)

        hybrid_route: dict = {}
        if medical_router is None:
            _trace("⚠️", "01-HYBRID-ROUTER", "SKIPPED — MedicalRouter not initialised")
        else:
            try:
                hybrid_route = medical_router.route_with_dept(query_input, active_dept or None)
                _trace("🔀", "01-HYBRID-ROUTER",
                       f"dept={hybrid_route.get('primary_dept')!r} | "
                       f"use_lora={hybrid_route.get('use_dept_lora')} | "
                       f"kb={hybrid_route.get('kb_routing')} | "
                       f"conf={hybrid_route.get('confidence', 0):.0%} | "
                       f"tfidf={hybrid_route.get('tfidf_score', 0):.3f} | "
                       f"method={hybrid_route.get('routing_method')}")
            except Exception as _hr_exc:
                _trace("❌", "01-HYBRID-ROUTER", f"ERROR (non-fatal): {_hr_exc}")

        # Minimum confidence required to invoke LoRA — low-confidence keyword hits
        # (e.g. 4%) should fall straight through to the faster RAG pipeline.
        _MIN_LORA_CONFIDENCE = float(os.environ.get("DEPT_LORA_MIN_CONFIDENCE", "0.60"))
        # Master switch: set DEPT_LORA_INFERENCE_ENABLED=false to skip phi-2 inference
        # (MPS/CPU inference blocks the request thread). The router still identifies dept.
        _LORA_INFERENCE_ENABLED = os.environ.get("DEPT_LORA_INFERENCE_ENABLED", "false").strip().lower() == "true"
        _route_conf = hybrid_route.get("confidence", 0.0)
        _use_lora = (
            _LORA_INFERENCE_ENABLED
            and hybrid_route.get("use_dept_lora")
            and _route_conf >= _MIN_LORA_CONFIDENCE
        )

        if not hybrid_route.get("use_dept_lora"):
            _trace("⏭️", "02-LOAD-DEPT-LORA",
                   f"SKIPPED — no trained LoRA model for dept={hybrid_route.get('primary_dept') or active_dept or '(none)'!r}")
        elif not _LORA_INFERENCE_ENABLED:
            _trace("🔒", "02-LOAD-DEPT-LORA",
                   f"READY (inference disabled) — dept={hybrid_route.get('primary_dept')!r} model available "
                   f"| set DEPT_LORA_INFERENCE_ENABLED=true to enable")
        elif not _use_lora:
            _trace("⏭️", "02-LOAD-DEPT-LORA",
                   f"SKIPPED — confidence {_route_conf:.0%} < threshold {_MIN_LORA_CONFIDENCE:.0%} "
                   f"(dept={hybrid_route.get('primary_dept')!r}) → falling back to RAG")
        elif hybrid_route.get("primary_dept"):
            dept_lora = current_app.config.get("DEPT_LORA_SERVICE")
            if dept_lora is None:
                _trace("⚠️", "02-LOAD-DEPT-LORA", "SKIPPED — DEPT_LORA_SERVICE not registered")
            else:
                _trace("🧠", "02-LOAD-DEPT-LORA",
                       f"invoking LoRA for dept={hybrid_route['primary_dept']!r} | query_len={len(query_input)}")
                try:
                    lora_result = dept_lora.generate(
                        hybrid_route["primary_dept"], query_input
                    )
                    if lora_result.get("success") and lora_result.get("response"):
                        _trace("✅", "02-LOAD-DEPT-LORA",
                               f"SUCCESS | device={lora_result.get('device','?')} | "
                               f"tokens={lora_result.get('tokens_generated','?')} | "
                               f"elapsed={lora_result.get('elapsed_seconds','?')}s")
                        lora_response = lora_result["response"]
                        dept_label = hybrid_route["primary_dept"]
                        lora_footer = (
                            f"\n\n**Dept Model:** {dept_label} fine-tuned LoRA "
                            f"| Device: {lora_result.get('device', 'cpu')} "
                            f"| Tokens: {lora_result.get('tokens_generated', '?')} "
                            f"| {lora_result.get('elapsed_seconds', '?')}s"
                        )
                        final_msg = guard_disclaimer + _clean_response_text(lora_response) + lora_footer
                        _trace("◄", "RESPONSE_VIA_LORA",
                               f"dept={dept_label!r} | response_len={len(lora_response)}")
                        _queue_sme_entry(user_input, final_msg, doctor_name, doctor_department)
                        return jsonify({
                            "response": True,
                            "message": final_msg,
                            "trace": _flush_trace(),
                            "dev_mode": _dev_mode_enabled(),
                            "routing_details": {
                                "disciplines": [hybrid_route.get("primary_dept", "")],
                                "sources": [],
                                "method": "dept_lora",
                                "confidence": f"{hybrid_route.get('confidence', 0):.0%}",
                                "dept_lora_used": True,
                                "kb_routing": hybrid_route.get("kb_routing", ""),
                            },
                        })
                    else:
                        _trace("⏭️", "02-LOAD-DEPT-LORA",
                               f"generate failed → falling through | error={lora_result.get('error','unknown')}")
                except Exception as _lora_exc:
                    _trace("❌", "02-LOAD-DEPT-LORA", f"EXCEPTION (falling through): {_lora_exc}")

        _trace("➡️", "RAG-PIPELINE", "continuing to standard RAG pipeline")

        # 🎯 INTEGRATED MEDICAL RAG SYSTEM WITH INTELLIGENT TOOL ROUTING
        if integrated_rag_system is not None:
            logger.info("Using Integrated Medical RAG System with Tool Routing")

            session_id = request.json.get("session_id")
            if not session_id or session_id == "guest":
                session_id = _get_session_folder() or "guest"
                logger.info("Using current session: %s", session_id)

            integrated_result = integrated_rag_system.query(
                query_input, session_id, patient_problem,
                adhoc_rag_ready=adhoc_rag_ready
            )

            if integrated_result and integrated_result.get("answer"):
                answer = integrated_result["answer"]
                routing_info = integrated_result.get("routing_info", {})
                tools_used = integrated_result.get("tools_used", [])
                citations = integrated_result.get("citations", [])
                source_documents = integrated_result.get("source_documents", [])

                # ── Phase 2-B STEP 1–10: Validate Response ────────────────────
                _trace("🔍", "03-VALIDATE-RESPONSE",
                       f"running validation  dept={doctor_department or active_dept or 'unknown'!r}")
                # Bug fix: use source_document excerpts (actual retrieved text) for
                # evidence matching — citation display strings are too short to match.
                _v_context = [
                    doc.get("excerpt", "")
                    for doc in source_documents
                    if doc.get("excerpt")
                ]
                if not _v_context:   # fall back to citations when no excerpts
                    _v_context = [c for c in citations if isinstance(c, str)]
                _v_patient_data = request.json.get("patient_data") or {}
                _v_result = _validate_response(
                    prompt=user_input,
                    response=answer,
                    context=_v_context,
                    department=doctor_department or active_dept or "",
                    patient_data=_v_patient_data,
                )
                _trace(
                    "📊", "03-VALIDATE-RESPONSE",
                    f"score={_v_result['score']}  decision={_v_result['decision']}  "
                    f"flags={_v_result['flags']}",
                )

                # Phase 2-B STEP 12 — log async (non-blocking)
                _log_validation(
                    session_id=session_id,
                    score=_v_result["score"],
                    results=_v_result["results"],
                    decision=_v_result["decision"],
                    flags=_v_result["flags"],
                    department=doctor_department or active_dept or "",
                    prompt_snippet=user_input,
                )

                # Phase 2-B STEP 11 — REGENERATE: retry LLM once if score < 60
                if _v_result["decision"] == "REGENERATE":
                    _trace("🔄", "03-VALIDATE-RESPONSE-RETRY",
                           f"score={_v_result['score']} < 60 — retrying LLM once")
                    logger.warning(
                        "[VALIDATION] REGENERATE triggered  score=%d  flags=%s  dept=%r",
                        _v_result["score"], _v_result["flags"],
                        doctor_department or active_dept,
                    )
                    try:
                        _retry = integrated_rag_system.query(
                            query_input, session_id, patient_problem,
                            adhoc_rag_ready=adhoc_rag_ready,
                        )
                        if _retry and _retry.get("answer"):
                            answer = _retry["answer"]
                            routing_info = _retry.get("routing_info", routing_info)
                            tools_used = _retry.get("tools_used", tools_used)
                            citations = _retry.get("citations", citations)
                            source_documents = _retry.get("source_documents", source_documents)
                            _trace("✅", "03-VALIDATE-RESPONSE-RETRY",
                                   f"retry succeeded  len={len(answer)}")
                            logger.info("[VALIDATION] Retry succeeded  len=%d", len(answer))
                        else:
                            _trace("⚠️", "03-VALIDATE-RESPONSE-RETRY",
                                   "retry returned no answer — using original")
                    except Exception as _retry_exc:
                        _trace("❌", "03-VALIDATE-RESPONSE-RETRY",
                               f"retry raised: {_retry_exc}")
                        logger.error("[VALIDATION] Retry failed: %s", _retry_exc)

                elif _v_result["decision"] == "REVIEW":
                    _trace("📋", "03-VALIDATE-RESPONSE",
                           f"score={_v_result['score']} in 60–79 — routed to HITL review queue")
                    logger.info(
                        "[VALIDATION] REVIEW flagged  score=%d  flags=%s",
                        _v_result["score"], _v_result["flags"],
                    )

                # ── Phase 2-B STEP 04: Safety / Guardrails ───────────────────
                _trace("🛡️", "04-SAFETY-GUARDRAILS",
                       f"running guardrails  dept={doctor_department or active_dept or 'unknown'!r}")
                _g_result = _run_guardrails(
                    prompt=user_input,
                    response=answer,
                    department=doctor_department or active_dept or "",
                    patient_data=_v_patient_data,
                    validation_score=_v_result["score"],
                )
                _trace(
                    "🛡️", "04-SAFETY-GUARDRAILS",
                    f"status={_g_result['status']}  flags={_g_result['flags']}",
                )
                _log_guardrail(
                    session_id=session_id,
                    status=_g_result["status"],
                    results=_g_result["results"],
                    flags=_g_result["flags"],
                    department=doctor_department or active_dept or "",
                    prompt_snippet=user_input,
                )

                # EMERGENCY → override LLM answer entirely
                if _g_result["status"] == "EMERGENCY":
                    _trace("🚨", "04-SAFETY-GUARDRAILS",
                           "EMERGENCY detected — overriding answer with emergency response")
                    logger.warning(
                        "[GUARDRAILS] EMERGENCY override  dept=%r  flags=%s",
                        doctor_department or active_dept, _g_result["flags"],
                    )
                    answer = _g_result["emergency_response"]

                elif _g_result["status"] == "BLOCKED":
                    _trace("🚫", "04-SAFETY-GUARDRAILS",
                           f"BLOCKED  flags={_g_result['flags']} — delivering with guardrail flag")
                    logger.warning(
                        "[GUARDRAILS] BLOCKED  dept=%r  flags=%s",
                        doctor_department or active_dept, _g_result["flags"],
                    )

                # ── Phase 2-B STEP 05: Confidence Scoring ────────────────────
                _trace("📈", "05-CONFIDENCE-SCORING",
                       f"computing confidence  validation_score={_v_result['score']}  "
                       f"guardrail_status={_g_result['status']}")
                _c_result = _score_confidence(
                    prompt=user_input,
                    response=answer,
                    context=_v_context,
                    source_documents=source_documents,
                    validation_score=_v_result["score"],
                    guardrail_status=_g_result["status"],
                )
                _trace(
                    "📈", "05-CONFIDENCE-SCORING",
                    f"score={_c_result['score']}  decision={_c_result['decision']}  "
                    f"breakdown={_c_result['breakdown']}",
                )
                _log_confidence(
                    session_id=session_id,
                    score=_c_result["score"],
                    breakdown=_c_result["breakdown"],
                    decision=_c_result["decision"],
                    department=doctor_department or active_dept or "",
                    prompt_snippet=user_input,
                )

                # Last-resort secret sanitizer — strip any API keys that may have
                # leaked through from tool outputs before sending to the user
                import re as _re_secrets
                _SECRET_RE = [
                    _re_secrets.compile(r'pcsk_[A-Za-z0-9_]{10,}', _re_secrets.I),
                    _re_secrets.compile(r'sk-[A-Za-z0-9_\-]{20,}', _re_secrets.I),
                    _re_secrets.compile(
                        r'(?:PINECONE|OPENAI|TAVILY|AZURE|APIFY|HUGGINGFACE)_[A-Z_]+=[\"\']?[A-Za-z0-9_\-\.]{8,}[\"\']?',
                        _re_secrets.I,
                    ),
                ]
                def _scrub(text: str) -> str:
                    for pat in _SECRET_RE:
                        text = pat.sub('[REDACTED]', text)
                    return text

                answer = _scrub(answer)
                citations = [_scrub(c) for c in citations]

                # ── Phase 2-B STEP 06: Attach Citations ──────────────────────
                # Determine LoRA info for attribution (Step 9)
                _lora_info = None
                if hybrid_route and hybrid_route.get("use_dept_lora"):
                    _lora_info = {
                        "used": True,
                        "department": hybrid_route.get("primary_dept", ""),
                        "model_version": hybrid_route.get("model_path", ""),
                    }

                _cit_result = _attach_citations(
                    response=answer,
                    source_documents=source_documents,
                    citations_raw=citations,
                    confidence_score=_c_result["score"],
                    department=doctor_department or active_dept or "",
                    session_id=session_id,
                    lora_info=_lora_info,
                )
                _trace(
                    "📎", "06-ATTACH-CITATIONS",
                    f"sources={_cit_result['source_count']}  "
                    f"top={_cit_result['top_source_label']!r}  "
                    f"avg_reliability={_cit_result['avg_reliability']}",
                )
                _log_citations(
                    session_id=session_id,
                    structured_citations=_cit_result["structured_citations"],
                    department=doctor_department or active_dept or "",
                    prompt_snippet=user_input,
                    confidence_score=_c_result["score"],
                )

                # ── Phase 2-B STEP 07: Log Interaction ───────────────────────
                import time as _t_mod
                _latency_ms = int((_t_mod.time() - _req_start) * 1000)
                _ilog = _create_interaction_log(
                    session_id=session_id,
                    tenant_id="",
                    doctor_id=_doctor_id,
                    patient_id=_patient_id,
                    department=doctor_department or active_dept or "",
                    prompt=user_input,
                    original_response=answer,
                    final_response=answer,
                    validation_result=_v_result,
                    guardrail_result=_g_result,
                    confidence_result=_c_result,
                    citation_result=_cit_result,
                    lora_info=_lora_info,
                    latency_ms=_latency_ms,
                )
                _log_interaction(_ilog)
                _trace(
                    "📝", "07-LOG-INTERACTION",
                    f"latency={_latency_ms}ms  tokens={_ilog.get('tokens_total', 0)}  "
                    f"session={session_id!r}",
                )

                # ── Phase 2-B: Session Completion Tracking ────────────────────
                if _SESSION_SVC_AVAILABLE and _session_svc is not None:
                    _sme_review_needed = (
                        "YES"
                        if _c_result.get("decision") == "REVIEW" or _v_result.get("decision") == "REVIEW"
                        else "NO"
                    )
                    _sess_summary = {
                        "timestamp": _ilog.get("timestamp", ""),
                        "prompt_snippet": (user_input or "")[:120],
                        "response": (answer or "")[:500],
                        "sme_review_needed": _sme_review_needed,
                        "confidence": _c_result.get("score", 0),
                        "sources": [
                            c.get("source", "") for c in _cit_result.get("structured_citations", [])
                        ],
                        # Full text used only for RLHF/HITL push — not stored in session JSON
                        "full_prompt": user_input or "",
                        "full_response": answer or "",
                    }
                    _sess_reason = _session_svc.on_query(
                        session_id=session_id,
                        patient_id=_patient_id or "",
                        doctor_id=_doctor_id or "",
                        department=doctor_department or active_dept or "",
                        interaction_summary=_sess_summary,
                        query_started_at=datetime.fromtimestamp(_req_start, tz=timezone.utc),
                    )
                    if _sess_reason:
                        _trace("🔔", "SESSION-COMPLETE", f"reason={_sess_reason}  session={session_id!r}")

                # Build the citations HTML block using the citation service output
                citations_html = _cit_result["citations_html"]

                # Main message body: disclaimer + clean answer (plain text/markdown only)
                message_body = guard_disclaimer + answer
                # Append the HTML citations block — frontend detects '<div' and renders as HTML
                message_body += "\n\n**Citations:**\n" + citations_html

                _queue_sme_entry(user_input, message_body, doctor_name, doctor_department)
                return jsonify(
                    {
                        "response": True,
                        "message": message_body,
                        "trace": _flush_trace(),
                            "dev_mode": _dev_mode_enabled(),
                        "validation": {
                            "score": _v_result["score"],
                            "decision": _v_result["decision"],
                            "flags": _v_result["flags"],
                        },
                        "guardrails": {
                            "status": _g_result["status"],
                            "flags": _g_result["flags"],
                        },
                        "confidence": {
                            "score": _c_result["score"],
                            "decision": _c_result["decision"],
                            "breakdown": _c_result["breakdown"],
                        },
                        "citations": {
                            "source_count": _cit_result["source_count"],
                            "top_source": _cit_result["top_source_label"],
                            "avg_reliability": _cit_result["avg_reliability"],
                            "structured": _cit_result["structured_citations"],
                        },
                        "routing_details": {
                            "disciplines": tools_used,
                            "planned_tools": routing_info.get("planned_tools", []),
                            "sources": citations,
                            "method": routing_info.get("confidence", "medium"),
                            "confidence": routing_info.get("confidence", "medium"),
                            "reasoning": routing_info.get("reasoning", ""),
                        },
                    }
                )
            else:
                logger.warning(
                    "No response from Integrated RAG system, falling back to Two-Store RAG"
                )

        # 🚀 FALLBACK: TWO-STORE RAG ARCHITECTURE WITH LEXICAL GATE
        if rag_manager is not None:
            logger.info("Using Two-Store RAG Architecture with Lexical Gate")

            session_id = request.json.get("session_id", "guest")
            rag_result = rag_manager.query_with_routing(query_input, session_id)

            if rag_result["responses"]:
                rag_result["responses"].sort(
                    key=lambda x: x["confidence"], reverse=True
                )
                final_response = ""
                for i, resp in enumerate(rag_result["responses"][:2], 1):
                    if i > 1:
                        final_response += "\n\n"
                    final_response += _clean_response_text(resp["content"])

                if rag_result["citations"]:
                    final_response += "\n\n**Citations:**\n"
                    for citation in rag_result["citations"]:
                        final_response += f"{citation}\n"

                routing_info = rag_result["routing_info"]
                sources_info = ", ".join(routing_info.get("sources_queried", []))
                final_response += (
                    f"\n**RAG Routing:** TF-IDF similarity: "
                    f"{routing_info.get('similarity_score', 0):.3f}, Sources: {sources_info}"
                )

                # Phase 2-B: Validate Response (Two-Store RAG path)
                _v_context_rag = [c for c in rag_result.get("citations", []) if isinstance(c, str)]
                _v_result_rag = _validate_response(
                    prompt=user_input,
                    response=final_response,
                    context=_v_context_rag,
                    department=doctor_department or active_dept or "",
                    patient_data=request.json.get("patient_data") or {},
                )
                _trace("📊", "03-VALIDATE-RESPONSE",
                       f"[Two-Store RAG] score={_v_result_rag['score']}  "
                       f"decision={_v_result_rag['decision']}")
                logger.info(
                    "[VALIDATION] Two-Store RAG  score=%d  decision=%s  flags=%s",
                    _v_result_rag["score"], _v_result_rag["decision"], _v_result_rag["flags"],
                )
                _log_validation(
                    session_id=session_id,
                    score=_v_result_rag["score"],
                    results=_v_result_rag["results"],
                    decision=_v_result_rag["decision"],
                    flags=_v_result_rag["flags"],
                    department=doctor_department or active_dept or "",
                    prompt_snippet=user_input,
                )

                # Phase 2-B: Safety / Guardrails (Two-Store RAG path)
                _g_result_rag = _run_guardrails(
                    prompt=user_input,
                    response=final_response,
                    department=doctor_department or active_dept or "",
                    patient_data=request.json.get("patient_data") or {},
                    validation_score=_v_result_rag["score"],
                )
                _trace("🛡️", "04-SAFETY-GUARDRAILS",
                       f"[Two-Store RAG] status={_g_result_rag['status']}  "
                       f"flags={_g_result_rag['flags']}")
                _log_guardrail(
                    session_id=session_id,
                    status=_g_result_rag["status"],
                    results=_g_result_rag["results"],
                    flags=_g_result_rag["flags"],
                    department=doctor_department or active_dept or "",
                    prompt_snippet=user_input,
                )
                if _g_result_rag["status"] == "EMERGENCY":
                    final_response = _g_result_rag["emergency_response"]

                # Phase 2-B: Confidence Scoring (Two-Store RAG path)
                _c_result_rag = _score_confidence(
                    prompt=user_input,
                    response=final_response,
                    context=_v_context_rag,
                    source_documents=[],
                    validation_score=_v_result_rag["score"],
                    guardrail_status=_g_result_rag["status"],
                )
                _trace("📈", "05-CONFIDENCE-SCORING",
                       f"[Two-Store RAG] score={_c_result_rag['score']}  "
                       f"decision={_c_result_rag['decision']}")
                _log_confidence(
                    session_id=session_id,
                    score=_c_result_rag["score"],
                    breakdown=_c_result_rag["breakdown"],
                    decision=_c_result_rag["decision"],
                    department=doctor_department or active_dept or "",
                    prompt_snippet=user_input,
                )

                # Phase 2-B: Attach Citations (Two-Store RAG path)
                _cit_result_rag = _attach_citations(
                    response=final_response,
                    source_documents=[],
                    citations_raw=rag_result.get("citations", []),
                    confidence_score=_c_result_rag["score"],
                    department=doctor_department or active_dept or "",
                    session_id=session_id,
                )
                _trace("📎", "06-ATTACH-CITATIONS",
                       f"[Two-Store RAG] sources={_cit_result_rag['source_count']}  "
                       f"top={_cit_result_rag['top_source_label']!r}")
                _log_citations(
                    session_id=session_id,
                    structured_citations=_cit_result_rag["structured_citations"],
                    department=doctor_department or active_dept or "",
                    prompt_snippet=user_input,
                    confidence_score=_c_result_rag["score"],
                )

                # ── Phase 2-B STEP 07: Log Interaction (Two-Store RAG) ───────
                import time as _t_mod_rag
                _latency_ms_rag = int((_t_mod_rag.time() - _req_start) * 1000)
                _ilog_rag = _create_interaction_log(
                    session_id=session_id,
                    tenant_id="",
                    doctor_id=_doctor_id,
                    patient_id=_patient_id,
                    department=doctor_department or active_dept or "",
                    prompt=user_input,
                    original_response=final_response,
                    final_response=guard_disclaimer + final_response,
                    validation_result=_v_result_rag,
                    guardrail_result=_g_result_rag,
                    confidence_result=_c_result_rag,
                    citation_result=_cit_result_rag,
                    latency_ms=_latency_ms_rag,
                )
                _log_interaction(_ilog_rag)
                _trace(
                    "📝", "07-LOG-INTERACTION",
                    f"[Two-Store RAG] latency={_latency_ms_rag}ms  "
                    f"tokens={_ilog_rag.get('tokens_total', 0)}",
                )

                # ── Phase 2-B: Session Completion Tracking (Two-Store RAG) ────
                if _SESSION_SVC_AVAILABLE and _session_svc is not None:
                    _sme_review_needed_rag = (
                        "YES"
                        if _c_result_rag.get("decision") == "REVIEW" or _v_result_rag.get("decision") == "REVIEW"
                        else "NO"
                    )
                    _sess_summary_rag = {
                        "timestamp": _ilog_rag.get("timestamp", ""),
                        "prompt_snippet": (user_input or "")[:120],
                        "response": (final_response or "")[:500],
                        "sme_review_needed": _sme_review_needed_rag,
                        "confidence": _c_result_rag.get("score", 0),
                        "sources": [
                            c.get("source", "") for c in _cit_result_rag.get("structured_citations", [])
                        ],
                        "full_prompt": user_input or "",
                        "full_response": final_response or "",
                    }
                    _sess_reason_rag = _session_svc.on_query(
                        session_id=session_id,
                        patient_id=_patient_id or "",
                        doctor_id=_doctor_id or "",
                        department=doctor_department or active_dept or "",
                        interaction_summary=_sess_summary_rag,
                        query_started_at=datetime.fromtimestamp(_req_start, tz=timezone.utc),
                    )
                    if _sess_reason_rag:
                        _trace("🔔", "SESSION-COMPLETE", f"reason={_sess_reason_rag}  session={session_id!r}")

                _queue_sme_entry(user_input, guard_disclaimer + final_response, doctor_name, doctor_department)
                return jsonify(
                    {
                        "response": True,
                        "message": guard_disclaimer + final_response,
                        "trace": _flush_trace(),
                            "dev_mode": _dev_mode_enabled(),
                        "validation": {
                            "score": _v_result_rag["score"],
                            "decision": _v_result_rag["decision"],
                            "flags": _v_result_rag["flags"],
                        },
                        "guardrails": {
                            "status": _g_result_rag["status"],
                            "flags": _g_result_rag["flags"],
                        },
                        "confidence": {
                            "score": _c_result_rag["score"],
                            "decision": _c_result_rag["decision"],
                            "breakdown": _c_result_rag["breakdown"],
                        },
                        "citations": {
                            "source_count": _cit_result_rag["source_count"],
                            "top_source": _cit_result_rag["top_source_label"],
                            "avg_reliability": _cit_result_rag["avg_reliability"],
                            "structured": _cit_result_rag["structured_citations"],
                        },
                        "routing_details": {
                            "disciplines": routing_info.get("sources_queried", []),
                            "sources": rag_result["citations"],
                            "method": "Two-Store RAG",
                            "confidence": f"{routing_info.get('similarity_score', 0):.0%}",
                        },
                    }
                )
            else:
                logger.warning(
                    "No responses from RAG system, falling back to legacy implementation"
                )

        # 🏥 LEGACY IMPLEMENTATION
        logger.info("Using legacy medical routing system")

        routing_result = medical_router.analyze_query(query_input)
        relevant_disciplines = routing_result["disciplines"]
        confidence_scores = routing_result["confidence_scores"]

        logger.info("Query: %r", user_input)
        if patient_problem:
            logger.info("Patient Context: %r", patient_problem)
        logger.info("Routed to disciplines: %s", relevant_disciplines)
        logger.info("Confidence scores: %s", confidence_scores)

        all_responses: list[dict] = []
        all_citations: list[str] = []

        for discipline_id in relevant_disciplines:
            discipline_cfg = next(
                (
                    d
                    for d in disciplines_config.get("disciplines", [])
                    if d["id"] == discipline_id
                ),
                None,
            )
            if discipline_cfg and discipline_cfg.get("is_session_based", False):
                continue

            try:
                vector_db_path = _get_discipline_vector_db_path(discipline_id)
                if vector_db_path and os.path.exists(vector_db_path):
                    logger.info("Querying Organization KB: %s", discipline_id)
                    vector_store = Chroma(
                        persist_directory=vector_db_path, embedding_function=embeddings
                    )
                    retriever = vector_store.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": Config.RETRIEVAL_K},
                    )
                    contextual_llm = _create_contextual_llm(patient_problem)
                    contextual_qa_chain = RetrievalQA.from_chain_type(
                        llm=contextual_llm, retriever=retriever
                    )
                    org_response = contextual_qa_chain.invoke(query_input)
                    search_results = retriever.invoke(query_input)

                    if org_response["result"].strip():
                        all_responses.append(
                            {
                                "source": f"Organization KB - {discipline_id.replace('_', ' ').title()}",
                                "content": org_response["result"],
                                "confidence": confidence_scores.get(discipline_id, 70),
                            }
                        )
                        org_cits = _enhance_with_citations(search_results)
                        if org_cits != "No citations available":
                            for line in _clean_response_text(org_cits).split("\n"):
                                if line.strip():
                                    all_citations.append(
                                        f"**{discipline_id.replace('_', ' ').title()}: {line.strip()}**"
                                    )
            except Exception as exc:
                logger.error("Error querying %s: %s", discipline_id, exc)

        doctors_files_selected = "doctors_files" in relevant_disciplines
        try:
            latest_vector_db = _get_latest_vector_db()
            if latest_vector_db and os.path.exists(latest_vector_db):
                label = "Doctor's Files" if doctors_files_selected else "Adhoc KB (user uploads)"
                logger.info("Querying %s", label)
                vector_store = Chroma(
                    persist_directory=latest_vector_db, embedding_function=embeddings
                )
                retriever = vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": Config.RETRIEVAL_K},
                )
                contextual_llm = _create_contextual_llm(patient_problem)
                contextual_qa = RetrievalQA.from_chain_type(
                    llm=contextual_llm, retriever=retriever
                )
                adhoc_response = contextual_qa.invoke(query_input)
                search_results = retriever.invoke(query_input)

                if adhoc_response["result"].strip():
                    source_name = (
                        "Doctor's Files"
                        if doctors_files_selected
                        else "User Uploaded Documents"
                    )
                    confidence = (
                        confidence_scores.get("doctors_files", 85)
                        if doctors_files_selected
                        else 85
                    )
                    all_responses.append(
                        {
                            "source": source_name,
                            "content": adhoc_response["result"],
                            "confidence": confidence,
                        }
                    )
                    adhoc_cits = _enhance_with_citations(search_results)
                    if adhoc_cits != "No citations available":
                        prefix = "Doctor's Files" if doctors_files_selected else "User Documents"
                        for line in _clean_response_text(adhoc_cits).split("\n"):
                            if line.strip():
                                all_citations.append(f"**{prefix}: {line.strip()}**")
        except Exception as exc:
            logger.error("Error querying adhoc KB: %s", exc)

        if all_responses:
            all_responses.sort(key=lambda x: x["confidence"], reverse=True)
            final_response = ""
            for i, resp in enumerate(all_responses[:2], 1):
                if i > 1:
                    final_response += "\n\n"
                final_response += _clean_response_text(resp["content"])

            if all_citations:
                final_response += "\n\n**Citations:**\n"
                for citation in all_citations:
                    final_response += f"{citation}\n"

            routing_str = ", ".join(
                d.replace("_", " ").title() for d in relevant_disciplines
            )
            final_response += f"\n**Query Routing:** Analyzed and routed to {routing_str}"

            _queue_sme_entry(user_input, guard_disclaimer + final_response, doctor_name, doctor_department)
            return jsonify(
                {
                    "response": True,
                    "message": guard_disclaimer + final_response,
                    "trace": _flush_trace(),
                            "dev_mode": _dev_mode_enabled(),
                    "routing_details": {
                        "disciplines": relevant_disciplines,
                        "sources": all_citations,
                        "method": routing_result.get("routing_method", "hybrid"),
                        "confidence": (
                            f"{max(confidence_scores.values()):.0%}"
                            if confidence_scores
                            else "0%"
                        ),
                    },
                }
            )
        else:
            fallback_response = (
                f'\n            I understand you\'re asking about: "{user_input}"\n            \n'
                f"            However, I couldn't find specific information in the available medical "
                f"knowledge bases for the disciplines I identified: "
                f"{', '.join(d.replace('_', ' ').title() for d in relevant_disciplines)}.\n"
                f"            \n"
                f"            This could be because:\n"
                f"            1. The Organization KB doesn't have information on this specific topic yet\n"
                f"            2. No user documents have been uploaded that relate to this query\n"
                f"            3. The query might need to be more specific\n"
                f"            \n"
                f"            **Query was routed to:** "
                f"{', '.join(d.replace('_', ' ').title() for d in relevant_disciplines)}\n"
                f"            \n"
                f"            Consider uploading relevant medical documents or rephrasing your "
                f"question for better results.\n"
                f"            "
            )

            return jsonify(
                {
                    "response": True,
                    "message": _clean_response_text(fallback_response),
                    "trace": _flush_trace(),
                            "dev_mode": _dev_mode_enabled(),
                    "routing_details": {
                        "disciplines": relevant_disciplines,
                        "sources": [],
                        "method": routing_result.get("routing_method", "hybrid"),
                        "confidence": "Low (≈30%)",
                    },
                }
            )

    except Exception as exc:
        logger.error("Error in handle_query: %s", exc)
        return jsonify(
            {
                "response": False,
                "message": f"An error occurred while processing your query: {exc}",
                "trace": _flush_trace(),
                            "dev_mode": _dev_mode_enabled(),
            }
        )


@query_bp.route("/data-html", methods=["POST"])
@handle_route_errors
def handle_query_html():
    """HTML endpoint that returns complete HTML documents with 3-section structure."""
    user_input = request.json.get("data", "")
    patient_problem = request.json.get("patient_problem", "").strip()

    if not user_input:
        result_data = {
            "medical_summary": "Please provide a valid input to get a medical response.",
            "sources": ["Input Validation"],
            "tool_info": {
                "primary_tool": "Input Validator",
                "confidence": "N/A",
                "reasoning": "No query text provided in the request.",
            },
        }
        html_response = _generate_full_html_response(result_data)
        response = make_response(html_response)
        response.headers["Content-Type"] = "text/html; charset=utf-8"
        return response

    scope_guard = _get_scope_guard()
    guard_disclaimer = ""
    if scope_guard is not None:
        _guard_status, similarity_score, _guard_msg = scope_guard.check(user_input)
        if _guard_status == "rejected":
            logger.warning(
                "Query rejected by scope guard (score=%.3f): %r",
                similarity_score,
                user_input[:80],
            )
            result_data = {
                "medical_summary": _guard_msg,
                "sources": ["Domain Scope Guard"],
                "tool_info": {
                    "primary_tool": "Domain Scope Guard",
                    "confidence": "N/A",
                    "reasoning": (
                        f"Query similarity score {similarity_score:.3f} is below threshold "
                        f"{scope_guard.threshold}. Query is outside trained domain."
                    ),
                },
            }
            html_response = _generate_full_html_response(result_data)
            response = make_response(html_response)
            response.headers["Content-Type"] = "text/html; charset=utf-8"
            return response
        elif _guard_status == "general_medical":
            logger.info(
                "General medical query (score=%.3f) — answering with disclaimer",
                similarity_score,
            )
            guard_disclaimer = _guard_msg

    if patient_problem:
        logger.info("Using patient context as system message: %r", patient_problem)

    # Enrich the RAG search query with patient problem so retrieval is patient-context-aware
    if patient_problem:
        query_input = f"{user_input}. Patient problem: {patient_problem}"
        logger.info("LLM search query enriched with patient problem")
    else:
        query_input = user_input
    integrated_rag_system = _get_integrated_rag()

    try:
        if integrated_rag_system is not None:
            logger.info("Using Integrated Medical RAG System with Tool Routing")

            session_id = request.json.get("session_id")
            if not session_id or session_id == "guest":
                session_id = _get_session_folder() or "guest"
                logger.info("Using current session: %s", session_id)

            _result = integrated_rag_system.query(query_input, session_id=session_id)
            answer = _result.get("answer", "")
            routing_info = _result.get("routing_info", {})
            tools_used = _result.get("tools_used", [])
            explanation = _result.get("explanation", "")

            if tools_used and any("Enhanced" in tool for tool in tools_used):
                logger.info("Processing Enhanced Tools Response (HTML format)")
                result_data = _parse_enhanced_response(
                    answer, routing_info, tools_used, explanation
                )
                if guard_disclaimer:
                    key = (
                        "medical_summary"
                        if "medical_summary" in result_data
                        else next(iter(result_data), None)
                    )
                    if key:
                        result_data[key] = guard_disclaimer + str(result_data[key])
            else:
                logger.info("Processing Two-Store RAG Response")
                result_data = {
                    "Answer": guard_disclaimer + answer,
                    "sources": (
                        ["Internal Document (PDF)", "Organization Knowledge Base"]
                        if tools_used and "organization" in tools_used[0].lower()
                        else ["Internal Document (PDF)"]
                    ),
                    "tool_info": {
                        "primary_tool": (
                            tools_used[0] if tools_used else "Internal_VectorDB"
                        ),
                        "confidence": (
                            f"{routing_info.get('confidence', 'medium').title()} (≈70%)"
                        ),
                        "reasoning": (
                            explanation
                            or f"Queried internal knowledge base due to uploaded content. "
                               f"{routing_info.get('reasoning', '')}"
                        ),
                    },
                }

            html_response = _generate_full_html_response(result_data)
            response = make_response(html_response)
            response.headers["Content-Type"] = "text/html; charset=utf-8"
            return response

        else:
            logger.warning("Integrated RAG system not available for HTML endpoint")
            result_data = {
                "medical_summary": (
                    "The HTML endpoint requires the integrated RAG system to be available. "
                    "Please use the regular /data endpoint."
                ),
                "sources": ["System Configuration"],
                "tool_info": {
                    "primary_tool": "Error Handler",
                    "confidence": "N/A",
                    "reasoning": (
                        "HTML endpoint requires integrated RAG system which is not currently available."
                    ),
                },
            }
            html_response = _generate_full_html_response(result_data)
            response = make_response(html_response)
            response.headers["Content-Type"] = "text/html; charset=utf-8"
            return response

    except Exception as exc:
        logger.error("Error in HTML query handler: %s", exc)
        traceback.print_exc()
        result_data = {
            "medical_summary": f"An error occurred while processing your query: {exc}",
            "sources": ["System Error"],
            "tool_info": {
                "primary_tool": "Error Handler",
                "confidence": "N/A",
                "reasoning": "An unexpected error occurred during query processing.",
            },
        }
        html_response = _generate_full_html_response(result_data)
        response = make_response(html_response)
        response.headers["Content-Type"] = "text/html; charset=utf-8"
        return response
