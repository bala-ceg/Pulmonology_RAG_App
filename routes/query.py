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
from datetime import datetime

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
# Optional SME queue auto-insert
# ---------------------------------------------------------------------------
try:
    from sft_experiment_manager import add_sme_queue_entry as _add_sme_queue_entry, DEPARTMENTS as _DEPARTMENTS
    _SME_QUEUE_AVAILABLE = True
except Exception:
    _DEPARTMENTS = {}
    _SME_QUEUE_AVAILABLE = False

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
    user_input = request.json.get("data", "")
    patient_problem = request.json.get("patient_problem", "").strip()
    doctor_name = (request.json.get("doctor_name") or "").strip()
    doctor_department = (request.json.get("doctor_department") or "").strip()

    if not user_input:
        return jsonify(
            {
                "response": False,
                "message": "Please provide a valid input to get a medical response.",
            }
        )

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
        try:
            from sft_experiment_manager import DEPARTMENTS as _DEPTS, get_best_model_path_for_dept as _gmp
            _all_depts = list(_DEPTS.keys())
            _depts_with_model = [d for d in _all_depts if _gmp(d)]
        except Exception:
            _all_depts = []
            _depts_with_model = []

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
                query_input, session_id, patient_problem
            )

            if integrated_result and integrated_result.get("answer"):
                answer = integrated_result["answer"]
                routing_info = integrated_result.get("routing_info", {})
                tools_used = integrated_result.get("tools_used", [])
                citations = integrated_result.get("citations", [])
                source_documents = integrated_result.get("source_documents", [])

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

                # ── Build the citations + source-documents section as HTML ──────────
                # The frontend checks for '<div' to decide whether to render as HTML
                # vs markdown. We use HTML so the source document cards render cleanly.
                cit_html_parts = ['<strong>Citations:</strong><ul style="margin:6px 0 0 16px;padding:0;">']
                for c in citations:
                    cit_html_parts.append(f'<li style="margin-bottom:4px;">{c}</li>')
                cit_html_parts.append('</ul>')

                if source_documents:
                    cit_html_parts.append(
                        '<hr style="margin:12px 0;border:none;border-top:1px solid #dee2e6;">'
                        '<strong style="color:#495057;">&#128196; Source Documents</strong>'
                    )
                    for i, doc in enumerate(source_documents, start=1):
                        fields = doc.get('fields', {})
                        dept    = fields.get('Dept', '')
                        src     = fields.get('Source', fields.get('Document', ''))
                        rel     = fields.get('Relevance', '')
                        page    = fields.get('Page', '')
                        url     = fields.get('URL', '')
                        excerpt = doc.get('excerpt', '')

                        # Badge colour by relevance
                        try:
                            rel_val = int(rel.rstrip('%'))
                            badge_color = '#28a745' if rel_val >= 70 else '#fd7e14' if rel_val >= 50 else '#6c757d'
                        except Exception:
                            badge_color = '#6c757d'

                        meta_items = []
                        if dept:
                            meta_items.append(f'<span style="background:#e9ecef;border-radius:4px;padding:1px 6px;font-size:11px;">🏥 {dept}</span>')
                        if src:
                            meta_items.append(f'<span style="background:#e9ecef;border-radius:4px;padding:1px 6px;font-size:11px;">📄 {src}</span>')
                        if page:
                            meta_items.append(f'<span style="background:#e9ecef;border-radius:4px;padding:1px 6px;font-size:11px;">p.{page}</span>')
                        if rel:
                            meta_items.append(f'<span style="background:{badge_color};color:#fff;border-radius:4px;padding:1px 6px;font-size:11px;font-weight:600;">{rel} match</span>')
                        if url:
                            meta_items.append(f'<a href="{url}" target="_blank" style="font-size:11px;">🔗 Link</a>')

                        meta_html = ' &nbsp;'.join(meta_items)
                        excerpt_html = excerpt.replace('<', '&lt;').replace('>', '&gt;')

                        cit_html_parts.append(
                            f'<div style="margin-top:10px;border:1px solid #dee2e6;border-radius:6px;'
                            f'padding:10px 12px;background:#f8f9fa;">'
                            f'<div style="font-size:12px;font-weight:600;color:#343a40;margin-bottom:6px;">'
                            f'Document {i} &nbsp; {meta_html}</div>'
                            f'<div style="font-size:13px;color:#495057;line-height:1.6;'
                            f'border-left:3px solid #007bff;padding-left:10px;'
                            f'background:#fff;border-radius:0 4px 4px 0;padding:8px 10px;">'
                            f'&ldquo;{excerpt_html}&rdquo;</div>'
                            f'</div>'
                        )

                citations_html = ''.join(cit_html_parts)

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
                        "routing_details": {
                            "disciplines": tools_used,
                            "sources": citations,
                            "method": routing_info.get("confidence", "medium"),
                            "confidence": routing_info.get("confidence", "medium"),
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

                _queue_sme_entry(user_input, guard_disclaimer + final_response, doctor_name, doctor_department)
                return jsonify(
                    {
                        "response": True,
                        "message": guard_disclaimer + final_response,
                        "trace": _flush_trace(),
                            "dev_mode": _dev_mode_enabled(),
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
