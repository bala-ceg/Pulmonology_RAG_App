"""
Disciplines Blueprint

Routes:
  GET  /                          — index (creates new session)
  GET  /api/disciplines           — list available disciplines
  POST /api/validate_disciplines  — validate user's selection
  GET  /search_doctors            — autocomplete search in pces_users
  GET  /search_patients           — patient autocomplete (CCM/EHR API)
  GET  /api/patient/search        — advanced patient search (CCM/EHR API)
  GET  /api/hospitals/search      — hospital search (CCM/EHR API)
  GET  /api/patient/<id>/history  — last 3 encounter diagnoses (pces_ehr_ccm)
  GET  /api/patient/<id>/allergies — known allergens (pces_ehr_ccm)
  GET  /api/patient/<id>/ai_summary — AI-generated clinical summary via LLM

Also owns:
  load_disciplines_config()
  get_available_disciplines()
  validate_discipline_selection()
  get_discipline_vector_db_path()
  create_organization_vector_db()
  initialize_session()
  MedicalQueryRouter class
"""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from datetime import date as date_cls, datetime, timedelta
from typing import Generator

import psycopg
from flask import Blueprint, current_app, jsonify, render_template, request

from config import Config
from utils.error_handlers import get_logger, handle_route_errors
from services.ccm_ehr_client import CCMEHRAuthError, CCMEHRError, ccm_ehr_client

# Optional — graceful if sft_experiment_manager or dept_lora_service are unavailable
try:
    from sft_experiment_manager import get_best_model_path_for_dept as _get_best_model_path
    _DEPT_MODEL_LOOKUP_AVAILABLE = True
except Exception:
    _DEPT_MODEL_LOOKUP_AVAILABLE = False
    def _get_best_model_path(_dept: str):  # type: ignore[misc]
        return None

logger = get_logger(__name__)

disciplines_bp = Blueprint("disciplines_bp", __name__)

# ---------------------------------------------------------------------------
# Constants (mirror main.py)
# ---------------------------------------------------------------------------
BASE_STORAGE_PATH = Config.KB_PATH


# ---------------------------------------------------------------------------
# Database helper (per-request connection; no global pool reference needed)
# ---------------------------------------------------------------------------

@contextmanager
def _db_conn() -> Generator:
    """Yield a psycopg connection using Config credentials (pces_base)."""
    with psycopg.connect(**Config.db_kwargs()) as conn:
        yield conn


@contextmanager
def _ehr_conn() -> Generator:
    """Yield a psycopg connection to the EHR database (pces_ehr_ccm via PG_TOOL_* vars)."""
    kwargs = {
        k: v for k, v in {
            "host":            Config.PG_TOOL_HOST,
            "port":            Config.PG_TOOL_PORT,
            "dbname":          Config.PG_TOOL_NAME,
            "user":            Config.PG_TOOL_USER,
            "password":        Config.PG_TOOL_PASSWORD,
            "connect_timeout": 10,
        }.items() if v is not None
    }
    with psycopg.connect(**kwargs) as conn:
        yield conn


# ---------------------------------------------------------------------------
# Disciplines configuration helpers
# ---------------------------------------------------------------------------

def load_disciplines_config() -> dict:
    """Load disciplines configuration from JSON file."""
    try:
        with open(Config.DISCIPLINES_CONFIG_PATH, "r") as fh:
            return json.load(fh)
    except FileNotFoundError:
        logger.warning("disciplines.json not found. Using default configuration.")
        return {
            "disciplines": [
                {
                    "id": "family_medicine",
                    "name": "Family Medicine",
                    "description": "Comprehensive primary healthcare",
                    "is_default": True,
                    "kb_path": "Organization_KB/Family_Medicine",
                    "vector_db_path": "vector_dbs/organization/family_medicine",
                }
            ],
            "selection_rules": {
                "min_selections": 1,
                "max_selections": 3,
                "default_discipline": "family_medicine",
            },
        }


def get_available_disciplines() -> list[dict]:
    """Return list of available disciplines for UI dropdown."""
    cfg = current_app.config.get("DISCIPLINES_CONFIG") or load_disciplines_config()
    return cfg.get("disciplines", [])


def validate_discipline_selection(selected: list[str]) -> tuple[bool, str]:
    """Validate user's discipline selection against configured rules."""
    cfg = current_app.config.get("DISCIPLINES_CONFIG") or load_disciplines_config()
    rules = cfg.get("selection_rules", {})
    min_sel = rules.get("min_selections", 1)
    max_sel = rules.get("max_selections", 3)

    if len(selected) < min_sel:
        return False, f"Please select at least {min_sel} discipline(s)"
    if len(selected) > max_sel:
        return False, f"Please select no more than {max_sel} discipline(s)"

    valid_ids = [d["id"] for d in cfg.get("disciplines", [])]
    invalid_ids = [d for d in selected if d not in valid_ids]
    if invalid_ids:
        return False, f"Invalid discipline(s): {', '.join(invalid_ids)}"

    return True, "Valid selection"


def get_discipline_vector_db_path(discipline_id: str) -> str | None:
    """Return vector database path for a specific discipline."""
    cfg = current_app.config.get("DISCIPLINES_CONFIG") or load_disciplines_config()
    for discipline in cfg.get("disciplines", []):
        if discipline["id"] == discipline_id:
            return discipline.get("vector_db_path", "")
    return None


def create_organization_vector_db(discipline_id: str, documents: list) -> object:
    """Create or update organization vector database for a specific discipline."""
    from langchain_chroma import Chroma

    vector_db_path = get_discipline_vector_db_path(discipline_id)
    if not vector_db_path:
        raise ValueError(f"Unknown discipline: {discipline_id}")

    persist_dir = os.path.join(".", vector_db_path)
    os.makedirs(persist_dir, exist_ok=True)

    embeddings = current_app.config.get("EMBEDDINGS")
    vector_store = Chroma.from_documents(
        documents, embedding=embeddings, persist_directory=persist_dir
    )
    return vector_store


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def _get_timestamp() -> str:
    return time.strftime("%m%d%Y%H%M")


def initialize_session(user: str = "guest") -> str:
    """Initialise a new session folder and persist it in app.config."""
    timestamp = _get_timestamp()
    session_folder = f"{user}_{timestamp}"

    os.makedirs(
        os.path.join(BASE_STORAGE_PATH, "PDF", session_folder), exist_ok=True
    )
    os.makedirs(
        os.path.join(BASE_STORAGE_PATH, "URL", session_folder), exist_ok=True
    )

    current_app.config["LAST_SESSION_FOLDER"] = session_folder
    logger.info("New session folder created: %s", session_folder)
    return session_folder


# ---------------------------------------------------------------------------
# MedicalQueryRouter
# ---------------------------------------------------------------------------

class MedicalQueryRouter:
    """Intelligent router that determines which medical disciplines are relevant for a query."""

    def __init__(self, llm, disciplines_config: dict) -> None:
        self.llm = llm
        self.disciplines = disciplines_config.get("disciplines", [])
        self.discipline_keywords = self._build_keyword_map()

    def _build_keyword_map(self) -> dict:
        return {
            "family_medicine": [
                "primary care", "general practice", "family doctor", "annual checkup",
                "preventive care", "common cold", "flu", "hypertension", "diabetes",
                "vaccination", "routine care", "wellness exam", "physical exam",
                "blood pressure", "cholesterol", "general health",
            ],
            "cardiology": [
                "heart", "cardiac", "cardiovascular", "chest pain", "heart attack",
                "myocardial infarction", "heart failure", "arrhythmia",
                "atrial fibrillation", "coronary", "angina", "pacemaker",
                "cardiologist", "EKG", "ECG", "echocardiogram", "blood pressure",
                "hypertension", "heart rate", "cardiac arrest", "valve", "aorta",
                "coronary artery",
            ],
            "neurology": [
                "brain", "neurological", "nervous system", "stroke", "seizure",
                "epilepsy", "migraine", "headache", "Parkinson's", "Alzheimer's",
                "dementia", "multiple sclerosis", "MS", "neurologist", "MRI brain",
                "CT brain", "memory loss", "confusion", "dizziness", "numbness",
                "tingling", "weakness", "paralysis", "spinal cord", "nerve",
            ],
            "pulmonology": [
                "lung", "pulmonary", "respiratory", "breathing", "breath", "COPD",
                "asthma", "bronchitis", "pneumonia", "emphysema", "IPF",
                "interstitial lung", "pleural", "pleura", "trachea", "bronchial",
                "inhaler", "spirometry", "oxygen", "hypoxia", "dyspnea",
                "shortness of breath", "cough", "wheeze", "wheezing",
                "pulmonologist", "ventilation", "ARDS", "pulmonary fibrosis",
                "sarcoidosis", "alpha-1 antitrypsin", "A1AT",
            ],
            "oncology": [
                "cancer", "tumor", "malignant", "malignancy", "chemotherapy",
                "radiation", "oncology", "carcinoma", "lymphoma", "leukemia",
                "metastasis", "biopsy", "immunotherapy", "targeted therapy",
                "oncologist", "staging", "remission",
            ],
            "diabetes": [
                "diabetes", "diabetic", "insulin", "glucose", "blood sugar",
                "hyperglycemia", "hypoglycemia", "HbA1c", "metformin",
                "type 1 diabetes", "type 2 diabetes", "gestational diabetes",
                "endocrinology", "pancreas",
            ],
            "nephrology": [
                "kidney", "renal", "nephrology", "dialysis", "creatinine",
                "GFR", "proteinuria", "kidney disease", "CKD", "AKI",
                "glomerulonephritis", "nephrologist",
            ],
            "gastroenterology": [
                "stomach", "gastric", "intestine", "bowel", "colon", "liver",
                "hepatic", "GI", "gastrointestinal", "Crohn", "colitis",
                "IBD", "IBS", "gastroenterology", "endoscopy", "colonoscopy",
                "diarrhea", "constipation", "nausea", "vomiting", "acid reflux",
                "GERD", "peptic ulcer",
            ],
            "infectious_disease": [
                "infection", "infectious", "bacteria", "viral", "virus", "sepsis",
                "antibiotic", "antiviral", "HIV", "AIDS", "tuberculosis", "TB",
                "COVID", "SARS", "influenza", "malaria", "fungal", "parasitic",
            ],
            "doctors_files": [
                "my files", "my documents", "uploaded", "document", "file", "PDF",
                "article", "my upload", "personal documents", "doctor's files",
                "my records", "uploaded content", "session files", "my PDFs",
                "document I uploaded", "file I shared", "my data",
            ],
        }

    def _has_session_files(self) -> bool:
        last_folder = current_app.config.get("LAST_SESSION_FOLDER")
        if not last_folder:
            return False
        pdf_path = os.path.join(BASE_STORAGE_PATH, "PDF", last_folder)
        url_path = os.path.join(BASE_STORAGE_PATH, "URL", last_folder)
        pdf_files = (
            [f for f in os.listdir(pdf_path) if f.endswith(".pdf")]
            if os.path.exists(pdf_path)
            else []
        )
        url_files = (
            [f for f in os.listdir(url_path) if f.endswith(".txt")]
            if os.path.exists(url_path)
            else []
        )
        return len(pdf_files) > 0 or len(url_files) > 0

    def analyze_query(self, query: str) -> dict:
        query_lower = query.lower()
        relevant_disciplines: list[str] = []
        confidence_scores: dict[str, float] = {}

        for discipline_id, keywords in self.discipline_keywords.items():
            keyword_matches = sum(1 for kw in keywords if kw in query_lower)
            if keyword_matches > 0:
                # Scale by absolute match count, not fraction of keyword list.
                # 1 specific match → 65%, 2 matches → 80%, 3+ → 90%+
                confidence = min(50 + keyword_matches * 15, 95)
                relevant_disciplines.append(discipline_id)
                confidence_scores[discipline_id] = confidence

        has_files = self._has_session_files()
        if has_files and "doctors_files" not in relevant_disciplines:
            user_file_kws = [
                "my", "document", "file", "upload", "PDF", "article",
                "personal", "doctor", "record",
            ]
            if any(kw in query_lower for kw in user_file_kws):
                relevant_disciplines.append("doctors_files")
                confidence_scores["doctors_files"] = 85

        if not relevant_disciplines:
            # No keyword match — include session files if present, else fall through to default
            if has_files:
                relevant_disciplines.append("doctors_files")
                confidence_scores["doctors_files"] = 75

        if not relevant_disciplines:
            relevant_disciplines = ["family_medicine"]
            confidence_scores["family_medicine"] = 60

        relevant_disciplines.sort(
            key=lambda d: confidence_scores.get(d, 0), reverse=True
        )

        return {
            "disciplines": relevant_disciplines[:2],
            "confidence_scores": confidence_scores,
            "routing_method": "hybrid" if relevant_disciplines else "default",
        }

    def route_with_dept(
        self,
        query: str,
        selected_dept: str | None = None,
    ) -> dict:
        """Hybrid routing combining explicit dept selection, TF-IDF gate, and keyword matching.

        This is the Phase 2-A "01 – Hybrid Router" implementation.

        Priority order:
        1. **Explicit dept selection** (confidence ≈ 0.90) — highest weight.
        2. **TF-IDF lexical gate** — decides whether local or external KB should be queried.
        3. **Keyword matching** (existing ``analyze_query`` logic) — tiebreaker / augmentation.

        Args:
            query:         The user's natural-language medical query.
            selected_dept: Department explicitly chosen by the user in the UI (may be None).

        Returns:
            A dict with keys:
            - ``primary_dept``   (str)  — canonical dept name to route to.
            - ``use_dept_lora``  (bool) — True if a trained LoRA model exists for primary_dept.
            - ``kb_routing``     (str)  — ``"local"`` | ``"external"`` | ``"both"``.
            - ``confidence``     (float) 0-1.
            - ``routing_method`` (str)  — description of signals used.
            - ``disciplines``    (list) — ordered list of relevant discipline IDs (compat shim).
        """
        kb_routing = "external"
        tfidf_score = 0.0

        logger.info(
            "[PHASE2A] HybridRouter.route_with_dept  ENTER  selected_dept=%r  query=%r",
            selected_dept, query[:100],
        )

        # --- Signal 2: TF-IDF lexical gate ---
        rag_manager = current_app.config.get("RAG_MANAGER")
        if rag_manager and hasattr(rag_manager, "lexical_gate"):
            try:
                use_local, tfidf_score = rag_manager.lexical_gate.should_query_local_first(query)
                kb_routing = "local" if use_local else "external"
                logger.info(
                    "[PHASE2A] HybridRouter  SIGNAL-2 TF-IDF  score=%.4f  threshold=%.2f  kb_routing=%s",
                    tfidf_score, rag_manager.lexical_gate.threshold, kb_routing,
                )
            except Exception as exc:
                logger.warning("[PHASE2A] HybridRouter  SIGNAL-2 TF-IDF  ERROR: %s", exc)
        else:
            logger.info("[PHASE2A] HybridRouter  SIGNAL-2 TF-IDF  SKIPPED — no RAG_MANAGER/lexical_gate")

        # --- Signal 1: Explicit dept selection ---
        if selected_dept:
            primary_dept = selected_dept
            confidence = 0.90
            routing_method = "explicit_dept"
            if kb_routing == "local":
                routing_method += "+tfidf_local"
            logger.info(
                "[PHASE2A] HybridRouter  SIGNAL-1 EXPLICIT DEPT  dept=%r  confidence=%.2f",
                primary_dept, confidence,
            )
        else:
            # --- Signal 3: keyword + AI fallback ---
            logger.info("[PHASE2A] HybridRouter  SIGNAL-1 skipped (no explicit dept) → SIGNAL-3 KEYWORD")
            keyword_result = self.analyze_query(query)
            primary_dept_id = keyword_result["disciplines"][0] if keyword_result["disciplines"] else "family_medicine"
            # Map discipline id back to a display name for LoRA lookup
            primary_dept = self._discipline_id_to_dept_name(primary_dept_id)
            confidence = max(keyword_result["confidence_scores"].values(), default=0.60) / 100.0
            routing_method = keyword_result.get("routing_method", "keyword")
            if tfidf_score >= 0.3:
                routing_method += "+tfidf_local"
            logger.info(
                "[PHASE2A] HybridRouter  SIGNAL-3 KEYWORD  primary_disc=%r  primary_dept=%r  conf=%.2f",
                primary_dept_id, primary_dept, confidence,
            )

        # --- Check LoRA availability ---
        use_dept_lora = False
        if _DEPT_MODEL_LOOKUP_AVAILABLE and primary_dept:
            try:
                model_path = _get_best_model_path(primary_dept)
                use_dept_lora = bool(model_path)
                logger.info(
                    "[PHASE2A] HybridRouter  LORA-CHECK  dept=%r  use_dept_lora=%s  model_path=%r",
                    primary_dept, use_dept_lora, model_path,
                )
            except Exception as _le:
                logger.warning("[PHASE2A] HybridRouter  LORA-CHECK  ERROR: %s", _le)
                use_dept_lora = False
        else:
            logger.info(
                "[PHASE2A] HybridRouter  LORA-CHECK  SKIPPED  "
                "(lookup_available=%s  primary_dept=%r)",
                _DEPT_MODEL_LOOKUP_AVAILABLE, primary_dept,
            )

        # Also fall back to kb_routing="both" when tfidf score is borderline
        if 0.2 <= tfidf_score < 0.3:
            kb_routing = "both"

        # Build compat disciplines list
        disciplines = (
            [self._dept_name_to_discipline_id(primary_dept)]
            if primary_dept
            else ["family_medicine"]
        )

        result = {
            "primary_dept": primary_dept,
            "use_dept_lora": use_dept_lora,
            "kb_routing": kb_routing,
            "confidence": round(confidence, 3),
            "routing_method": routing_method,
            "tfidf_score": round(tfidf_score, 4),
            # backward-compat keys consumed by existing query route
            "disciplines": disciplines,
            "confidence_scores": {primary_dept: round(confidence * 100, 1)},
        }
        logger.info(
            "[PHASE2A] HybridRouter.route_with_dept  EXIT  → %s",
            {k: v for k, v in result.items() if k not in ("disciplines", "confidence_scores")},
        )
        return result

    # ------------------------------------------------------------------
    # Department ↔ discipline ID helpers
    # ------------------------------------------------------------------

    def _discipline_id_to_dept_name(self, discipline_id: str) -> str:
        """Map a discipline config id (e.g. ``"cardiology"``) to a dept display name."""
        _DISC_TO_DEPT = {
            "family_medicine": "General Medicine",
            "cardiology": "Cardiology",
            "neurology": "Neurology",
            "pulmonology": "Pulmonology",
            "doctors_files": "General Medicine",
        }
        # Try built-in map first; fall back to title-casing the id
        return _DISC_TO_DEPT.get(discipline_id, discipline_id.replace("_", " ").title())

    def _dept_name_to_discipline_id(self, dept_name: str) -> str:
        """Map a dept display name back to a discipline config id where possible."""
        _DEPT_TO_DISC = {
            "cardiology": "cardiology",
            "neurology": "neurology",
            "pulmonology": "pulmonology",
            "family medicine": "family_medicine",
            "general medicine": "family_medicine",
        }
        return _DEPT_TO_DISC.get(dept_name.lower(), dept_name.lower().replace(" ", "_"))


        try:
            discipline_names = [d["name"] for d in self.disciplines]
            prompt = f"""
            Analyze this medical query and determine which medical specialties are most relevant:
            
            Query: "{query}"
            
            Available specialties: {', '.join(discipline_names)}
            
            Guidelines:
            - If the query mentions "my files", "my documents", "uploaded", or refers to user's personal documents, include "Doctor's Files"
            - If the query is general or could apply to multiple specialties, include Family Medicine
            - If unclear, default to Family Medicine
            - Consider that "Doctor's Files" contains user-uploaded PDFs and documents
            
            Return only the specialty names that are relevant, separated by commas.
            Response format: Specialty1, Specialty2 (max 3)
            """
            response = self.llm.invoke(prompt)
            content = (
                response.content if hasattr(response, "content") else str(response)
            )
            ai_specialties = [s.strip() for s in content.split(",")]
            result: list[str] = []
            for specialty in ai_specialties:
                for discipline in self.disciplines:
                    if discipline["name"].lower() in specialty.lower():
                        result.append(discipline["id"])
                        break
            return result
        except Exception as exc:
            logger.error("AI analysis failed: %s", exc)
            return ["family_medicine"]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@disciplines_bp.route("/", methods=["GET"])
@handle_route_errors
def index():
    """Refresh page to create a new session folder."""
    user = request.args.get("user", "guest")
    initialize_session(user)
    from config import Config  # noqa: PLC0415

    # Pre-fetch hospital name for the login modal (best-effort)
    default_hospital: str = "Default PCES"
    try:
        with _db_conn() as _hconn:
            with _hconn.cursor() as _hcur:
                _hcur.execute(
                    "SELECT organization_name FROM pces_affiliates WHERE org_code = 'PCES101' LIMIT 1"
                )
                _hrow = _hcur.fetchone()
                if _hrow and _hrow[0]:
                    default_hospital = _hrow[0]
    except Exception:
        pass

    return render_template(
        "index.html",
        yodha_chat_url=Config.YODHA_CHAT_URL,
        doc_patient_v2_url=Config.DOC_PATIENT_V2_URL,
        default_hospital=default_hospital,
    )


@disciplines_bp.route("/api/login", methods=["POST"])
def login():
    """Authenticate a PCES user against the pces_users table."""
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    if not username or not password:
        return jsonify({"success": False, "message": "Username and password are required"}), 400

    try:
        with _db_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT username, password_hash, pces_role, first_name, last_name, email FROM pces_users WHERE username = %s LIMIT 1",
                    (username,),
                )
                row = cursor.fetchone()

        if row is None:
            return jsonify({"success": False, "message": "Invalid username or password"}), 401

        db_username, password_hash, pces_role, first_name, last_name, email = row
        if password != password_hash:
            return jsonify({"success": False, "message": "Invalid username or password"}), 401

        # Look up party_id from p_party by email match (best-effort)
        doctor_party_id: str | None = None
        if email:
            try:
                with _ehr_conn() as ehr_conn:
                    with ehr_conn.cursor() as ehr_cur:
                        ehr_cur.execute(
                            "SELECT party_id FROM public.p_party WHERE party_type = 'DOCTOR' AND LOWER(email) = LOWER(%s) LIMIT 1",
                            (email,),
                        )
                        party_row = ehr_cur.fetchone()
                        if party_row:
                            doctor_party_id = str(party_row[0])
            except Exception:
                pass  # non-critical — falls back to username

        full_name = f"{first_name or ''} {last_name or ''}".strip() or db_username

        # Look up the default hospital name from pces_affiliates (best-effort)
        hospital_name: str = "Default PCES"
        try:
            with _db_conn() as _hconn:
                with _hconn.cursor() as _hcur:
                    _hcur.execute(
                        "SELECT organization_name FROM pces_affiliates WHERE org_code = 'PCES101' LIMIT 1"
                    )
                    _hrow = _hcur.fetchone()
                    if _hrow and _hrow[0]:
                        hospital_name = _hrow[0]
        except Exception:
            pass  # non-critical — keep default

        return jsonify({
            "success": True,
            "username": db_username,
            "party_id": doctor_party_id,   # UUID from p_party; None if no email match
            "pces_role": pces_role,
            "full_name": full_name,
            "email": email or "",
            "department": pces_role or "",   # pces_role IS the specialty (CARDIOLOGIST, etc.)
            "hospital_name": hospital_name,
        })
    except Exception as exc:
        logger.error("Login error: %s", exc)
        return jsonify({"success": False, "message": "Server error during login"}), 500


@disciplines_bp.route("/api/disciplines", methods=["GET"])
@handle_route_errors
def get_disciplines():
    """Return available disciplines for UI dropdown."""
    try:
        disciplines = get_available_disciplines()
        cfg = current_app.config.get("DISCIPLINES_CONFIG") or load_disciplines_config()
        return jsonify(
            {
                "success": True,
                "disciplines": disciplines,
                "selection_rules": cfg.get("selection_rules", {}),
            }
        )
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@disciplines_bp.route("/api/validate_disciplines", methods=["POST"])
@handle_route_errors
def validate_disciplines():
    """Validate selected disciplines."""
    try:
        selected = request.json.get("selected_disciplines", [])
        is_valid, message = validate_discipline_selection(selected)
        return jsonify({"success": True, "is_valid": is_valid, "message": message})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@disciplines_bp.route("/api/set_department", methods=["POST"])
@handle_route_errors
def set_department():
    """Store the user's selected department in app.config for the current session.

    Phase 2-A: '01 – Selects Dept' step.

    Request JSON::

        { "department": "Cardiology", "session_id": "<optional>" }

    Response JSON::

        {
            "success": true,
            "department": "Cardiology",
            "has_trained_model": true,
            "model_path": "sft_models/Cardiology_experiment_1"
        }
    """
    data = request.get_json(silent=True) or {}
    dept = (data.get("department") or "").strip()
    if not dept:
        return jsonify({"success": False, "error": "department is required"}), 400

    current_app.config["ACTIVE_DEPARTMENT"] = dept

    # Reinitialise hybrid router for new dept context
    llm = current_app.config.get("LLM_INSTANCE")
    if llm:
        cfg = current_app.config.get("DISCIPLINES_CONFIG") or load_disciplines_config()
        current_app.config["MEDICAL_ROUTER"] = MedicalQueryRouter(llm, cfg)

    # Check if a trained LoRA model is available for this dept
    model_path = None
    if _DEPT_MODEL_LOOKUP_AVAILABLE:
        try:
            model_path = _get_best_model_path(dept)
        except Exception:
            pass

    logger.info("Active department set to: %s (has_model=%s)", dept, model_path is not None)
    return jsonify({
        "success": True,
        "department": dept,
        "has_trained_model": model_path is not None,
        "model_path": model_path or "",
    })


@disciplines_bp.route("/api/dept_model_status", methods=["GET"])
@handle_route_errors
def dept_model_status():
    """Return whether a trained LoRA model is available for the currently active department.

    Phase 2-A: '02 – Trained Model – Load Dept LoRA' readiness check.

    Query params:
        department (optional): override the active dept for a one-off check.

    Response JSON::

        {
            "success": true,
            "department": "Cardiology",
            "has_trained_model": true,
            "model_path": "sft_models/Cardiology_experiment_1",
            "lora_available": true
        }
    """
    dept = (
        request.args.get("department", "").strip()
        or current_app.config.get("ACTIVE_DEPARTMENT", "")
    )

    if not dept:
        return jsonify({
            "success": True,
            "department": None,
            "has_trained_model": False,
            "model_path": "",
            "lora_available": False,
        })

    model_path = None
    if _DEPT_MODEL_LOOKUP_AVAILABLE:
        try:
            model_path = _get_best_model_path(dept)
        except Exception:
            pass

    return jsonify({
        "success": True,
        "department": dept,
        "has_trained_model": model_path is not None,
        "model_path": model_path or "",
        "lora_available": model_path is not None,
    })
@handle_route_errors
def search_doctors():
    """Search for doctors by first_name and last_name from pces_users table."""
    try:
        query = request.args.get("q", "").strip().lower()
        if not query:
            return jsonify([])

        with _db_conn() as conn:
            with conn.cursor() as cursor:
                search_query = """
                SELECT DISTINCT first_name, last_name 
                FROM pces_users 
                WHERE LOWER(first_name) LIKE %s 
                   OR LOWER(last_name) LIKE %s 
                   OR LOWER(CONCAT(first_name, ' ', last_name)) LIKE %s
                ORDER BY first_name, last_name
                LIMIT 10
                """
                pattern = f"%{query}%"
                cursor.execute(search_query, (pattern, pattern, pattern))
                results = cursor.fetchall()

        doctors = [
            {
                "first_name": row[0],
                "last_name": row[1],
                "full_name": f"{row[0]} {row[1]}",
            }
            for row in results
            if row[0] and row[1]
        ]
        return jsonify(doctors)

    except Exception as exc:
        logger.error("Error searching doctors: %s", exc)
        return jsonify({"error": str(exc)}), 500


@disciplines_bp.route("/search_patients", methods=["GET"])
@handle_route_errors
def search_patients():
    """Patient autocomplete — calls the CCM/EHR external API.

    Accepts ?q=<name> and splits it across first_name / last_name.
    Returns max 10 results in [{patient_id, first_name, last_name, full_name}] format.
    """
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify([])

    # Split into first / last guess for the CCM/EHR API
    parts = query.split(None, 1)
    first = parts[0] if parts else ""
    last  = parts[1] if len(parts) > 1 else ""

    try:
        # Try last_name first (more discriminating), then full query as first_name
        results = ccm_ehr_client.search_patients(last_name=query)
        if not results and first:
            results = ccm_ehr_client.search_patients(first_name=first, last_name=last)
        return jsonify(results[:10])

    except CCMEHRAuthError as exc:
        logger.warning("search_patients: CCM/EHR auth error — %s", exc)
        return jsonify({"error": "CCM/EHR token missing or expired. Contact admin.", "auth_error": True}), 401

    except CCMEHRError as exc:
        logger.warning("search_patients: CCM/EHR unavailable — %s", exc)
        return jsonify([])

    except Exception as exc:
        logger.error("search_patients: unexpected error — %s", exc)
        return jsonify([])


@disciplines_bp.route("/api/patients/first20", methods=["GET"])
@handle_route_errors
def get_first_20_patients():
    try:
        with _ehr_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
    SELECT
        pp.party_id,
        pp.first_name,
        pp.middle_name,
        pp.last_name,
        pp.date_of_birth,
        pp.phone,
        pp.email,
        pa.address_id,
        pa.line1,
        pa.line2,
        pa.city,
        pa.state,
        pa.postal_code
    FROM p_party pp
    LEFT JOIN p_address pa
        ON pa.party_id = pp.party_id
       AND pa.is_active = true
    WHERE pp.party_type = 'PATIENT'
      AND pp.is_active = true
    ORDER BY pp.last_name, pp.first_name
    LIMIT 20
""")

                rows = cursor.fetchall()

        patients = []

        for row in rows:
            (
                party_id,
                first_name,
                middle_name,
                last_name,
                dob,
                phone,
                email,
                address_id,
                address1,
                address2,
                city,
                state,
                postal_code
            ) = row

            full_name = " ".join(
                x for x in
                [first_name, middle_name, last_name]
                if x
            )

            patients.append({
                "patient_id": str(party_id),
                "first_name": first_name or "",
                "middle_name": middle_name or "",
                "last_name": last_name or "",
                "full_name": full_name,
                "dob": str(dob)[:10] if dob else "",
                "phone": phone or "",
                "email": email or "",
                "address_id": str(address_id) if address_id else "",
                "address1": address1 or "",
                "address2": address2 or "",
                "city": city or "",
                "state": state or "",
                "zip": postal_code or ""
            })

        return jsonify(patients)

    except Exception as exc:
        logger.error("Unable to retrieve first 20 patients: %s", exc)
        return jsonify({"error": str(exc)}), 500
@disciplines_bp.route("/api/doctors/first20", methods=["GET"])
@handle_route_errors
def get_first_20_doctors():
    try:
        with _ehr_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                  SELECT
                   pp.party_id,
                   pp.first_name,
                   pp.middle_name,
                   pp.last_name,
                   pp.date_of_birth,
                   pp.phone,
                   pp.email,
                   pa.address_id,
                   pa.line1,
                   pa.line2,
                   pa.city,
                   pa.state,
                   pa.postal_code
                   FROM p_party pp
                   LEFT JOIN p_address pa
                  ON pa.party_id = pp.party_id
                 AND pa.is_active = true
                 WHERE pp.party_type = 'DOCTOR'
                 AND pp.is_active = true
                ORDER BY pp.last_name, pp.first_name
                LIMIT 20;
                """)

                rows = cursor.fetchall()

                doctors = []

        for row in rows:
            (
                party_id,
                first_name,
                middle_name,
                last_name,
                dob,
                phone,
                email,
                address_id,
                address1,
                address2,
                city,
                state,
                postal_code
            ) = row

            full_name = " ".join(
                x for x in
                [first_name, middle_name, last_name]
                if x
            )

            doctors.append({
                "doctor_id": str(party_id),
                "first_name": first_name or "",
                "middle_name": middle_name or "",
                "last_name": last_name or "",
                "full_name": full_name,
                "dob": str(dob)[:10] if dob else "",
                "phone": phone or "",
                "email": email or "",
                "address_id": str(address_id) if address_id else "",
                "address1": address1 or "",
                "address2": address2 or "",
                "city": city or "",
                "state": state or "",
                "zip": postal_code or ""
            })

        return jsonify(doctors)

    except Exception as exc:
        logger.error("Unable to retrieve first 20 doctors: %s", exc)
        return jsonify({"error": str(exc)}), 500
# ============================================================
# Patient -> Doctor relationship
# ============================================================

@disciplines_bp.route(
    "/api/patients/<patient_id>/doctors",
    methods=["GET"]
)
@handle_route_errors
def get_patient_doctors(patient_id: str):
    """
    Return doctors associated with the selected patient
    through p_encounter.
    """
    try:
        with _ehr_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        D.party_id AS provider_id,
                        D.first_name,
                        D.middle_name,
                        D.last_name,
                        E.encounter_type,
                        E.encounter_date
                    FROM p_encounter E
                    JOIN p_party P
                        ON E.patient_id = P.party_id
                    JOIN p_party D
                        ON E.provider_id = D.party_id
                    WHERE P.party_id = %s
                    ORDER BY E.encounter_date DESC
                    """,
                    (patient_id,)
                )

                rows = cursor.fetchall()

        doctors = []

        for row in rows:
            (
                provider_id,
                first_name,
                middle_name,
                last_name,
                encounter_type,
                encounter_date
            ) = row

            full_name = " ".join(
                x for x in
                [first_name, middle_name, last_name]
                if x
            )

            doctors.append({
                "doctor_id": str(provider_id),
                "first_name": first_name or "",
                "middle_name": middle_name or "",
                "last_name": last_name or "",
                "full_name": full_name,
                "encounter_type": encounter_type or "",
                "encounter_dt":
                    str(encounter_date)
                    if encounter_date
                    else ""
            })

        return jsonify(doctors)

    except Exception as exc:
        logger.exception(
            "Unable to retrieve doctors for patient %s: %s",
            patient_id,
            exc
        )

        return jsonify({
            "error": str(exc)
        }), 500


# ============================================================
# Doctor -> Patient relationship
# ============================================================

@disciplines_bp.route(
    "/api/doctors/<doctor_id>/patients",
    methods=["GET"]
)
@handle_route_errors
def get_doctor_patients(doctor_id: str):
    """
    Return distinct patients associated with the selected
    doctor through p_encounter.
    """
    try:
        with _ehr_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT DISTINCT
                        P.party_id AS patient_id,
                        P.first_name,
                        P.middle_name,
                        P.last_name,
                        P.date_of_birth,
                        P.phone,
                        P.email
                    FROM p_encounter E
                    JOIN p_party D
                        ON E.provider_id = D.party_id
                    JOIN p_party P
                        ON E.patient_id = P.party_id
                    WHERE D.party_id = %s
                    ORDER BY
                        P.last_name,
                        P.first_name
                    LIMIT 20
                    """,
                    (doctor_id,)
                )

                rows = cursor.fetchall()

        patients = []

        for row in rows:
            (
                patient_id,
                first_name,
                middle_name,
                last_name,
                dob,
                phone,
                email
            ) = row

            full_name = " ".join(
                x for x in
                [first_name, middle_name, last_name]
                if x
            )

            patients.append({
                "patient_id": str(patient_id),
                "first_name": first_name or "",
                "middle_name": middle_name or "",
                "last_name": last_name or "",
                "full_name": full_name,
                "dob": str(dob)[:10] if dob else "",
                "phone": phone or "",
                "email": email or ""
            })

        return jsonify(patients)

    except Exception as exc:
        logger.exception(
            "Unable to retrieve patients for doctor %s: %s",
            doctor_id,
            exc
        )

        return jsonify({
            "error": str(exc)
        }), 500

# ============================================================
# Doctor's Schedule
# ============================================================

@disciplines_bp.route(
    "/api/doctors/<doctor_id>/schedule",
    methods=["GET"]
)
@handle_route_errors
def get_doctor_schedule(doctor_id: str):
    """
    Return today's scheduled appointments for the selected doctor.
    """
    try:
        with _ehr_conn() as conn:
            with conn.cursor() as cursor:

                cursor.execute(
                    """
                    SELECT
                        PS.patient_id,
                        PP.first_name,
                        PP.middle_name,
                        PP.last_name,
                        PP.date_of_birth,
                        PP.gender,
                        PS.appointment_time
                    FROM p_schedule PS
                    JOIN p_party PP
                        ON PS.patient_id = PP.party_id
                    WHERE PS.provider_id = %s
                      AND PS.appointment_date = CURRENT_DATE
                      AND PS.status = 'SCHEDULED'
                    ORDER BY PS.appointment_time
                    """,
                    (doctor_id,)
                )

                rows = cursor.fetchall()

        schedule = []

        for row in rows:
            (
                patient_id,
                first_name,
                middle_name,
                last_name,
                dob,
                gender,
                appointment_time
            ) = row

            full_name = " ".join(
                value
                for value in [
                    first_name,
                    middle_name,
                    last_name
                ]
                if value
            )

            schedule.append({
                "patient_id": str(patient_id),
                "first_name": first_name or "",
                "middle_name": middle_name or "",
                "last_name": last_name or "",
                "full_name": full_name,
                "dob": str(dob)[:10] if dob else "",
                "gender": gender or "",
                "appointment_time":
                    str(appointment_time)
                    if appointment_time
                    else ""
            })

        return jsonify(schedule)

    except Exception as exc:
        logger.exception(
            "Unable to retrieve schedule for doctor %s: %s",
            doctor_id,
            exc
        )

        return jsonify({
            "error": str(exc)
        }), 500

# ============================================================
# Doctor Schedule by Date - Patient Scheduling Stage A
# ============================================================

@disciplines_bp.route(
    "/api/doctors/<doctor_id>/schedule-by-date",
    methods=["GET"]
)
@handle_route_errors
def get_doctor_schedule_by_date(doctor_id: str):
    """Return active p_schedule rows for one doctor on one date."""
    appointment_date = (request.args.get("date") or "").strip()

    if not appointment_date:
        return jsonify({"error": "date is required"}), 400

    try:
        # Validate YYYY-MM-DD before sending the value to PostgreSQL.
        parsed_date = date_cls.fromisoformat(appointment_date)
    except ValueError:
        return jsonify({
            "error": "date must be in YYYY-MM-DD format"
        }), 400

    try:
        with _ehr_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        s.schedule_id,
                        s.appointment_date,
                        s.appointment_time,
                        s.patient_id,
                        CONCAT_WS(
                            ' ',
                            p.first_name,
                            p.middle_name,
                            p.last_name
                        ) AS patient_name,
                        s.status
                    FROM ehr_ccm_schema.p_schedule s
                    LEFT JOIN ehr_ccm_schema.p_party p
                        ON p.party_id = s.patient_id
                    WHERE s.provider_id = %s
                      AND s.appointment_date = %s
                      AND s.is_active = TRUE
                    ORDER BY s.appointment_time
                    """,
                    (doctor_id, parsed_date)
                )

                rows = cursor.fetchall()

        # Stage B (temporary availability model):
        # Build a full 30-minute workday and mark gaps as available.
        # These hours can later be replaced by doctor-specific availability.
        WORK_DAY_START = "09:00"
        WORK_DAY_END = "18:00"
        SLOT_MINUTES = 30

        booked_slots = {}

        for row in rows:
            (
                schedule_id,
                schedule_date,
                appointment_time,
                patient_id,
                patient_name,
                status
            ) = row

            if not appointment_time:
                continue

            time_key = appointment_time.strftime("%H:%M")

            booked_slots[time_key] = {
                "schedule_id": str(schedule_id),
                "appointment_date": (
                    schedule_date.isoformat()
                    if schedule_date
                    else appointment_date
                ),
                "appointment_time": time_key,
                "patient_id": (
                    str(patient_id)
                    if patient_id
                    else ""
                ),
                "patient_name": patient_name or "",
                "status": status or "",
                "available": False
            }

        schedule = []
        current_slot = datetime.strptime(WORK_DAY_START, "%H:%M")
        final_slot = datetime.strptime(WORK_DAY_END, "%H:%M")

        while current_slot <= final_slot:
            time_key = current_slot.strftime("%H:%M")

            if time_key in booked_slots:
                schedule.append(booked_slots[time_key])
            else:
                schedule.append({
                    "schedule_id": "",
                    "appointment_date": appointment_date,
                    "appointment_time": time_key,
                    "patient_id": "",
                    "patient_name": "Spot Available",
                    "status": "AVAILABLE",
                    "available": True
                })

            current_slot += timedelta(minutes=SLOT_MINUTES)

        return jsonify(schedule)

    except Exception as exc:
        logger.exception(
            "Unable to retrieve schedule for doctor %s on %s: %s",
            doctor_id,
            appointment_date,
            exc
        )
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Patient / Doctor CRUD routes used by the Patient/Doctor data panel
# ---------------------------------------------------------------------------



def _entity_payload() -> dict:
    """Normalize the Patient/Doctor form payload from index.html."""
    data = request.get_json(silent=True) or {}
    return {
        "address_id": (data.get("address_id") or "").strip() or None,
        "first_name": (data.get("first_name") or "").strip(),
        "middle_name": (data.get("middle_name") or "").strip(),
        "last_name": (data.get("last_name") or "").strip(),
        "dob": (data.get("dob") or "").strip() or None,
        "phone": (data.get("phone") or "").strip(),
        "email": (data.get("email") or "").strip(),
        "address1": (data.get("address1") or "").strip(),
        "address2": (data.get("address2") or "").strip(),
        # Existing code in this module proves line1/city/state/postal_code.
        # No line2 column is assumed until the DB schema confirms one.
        "city": (data.get("city") or "").strip(),
        "state": (data.get("state") or "").strip(),
        "zip": (data.get("zip") or "").strip(),
    }


def _validate_entity_payload(data: dict):
    if not data["first_name"] or not data["last_name"]:
        return jsonify({
            "success": False,
            "error": "First Name and Last Name are required."
        }), 400
    return None


def _upsert_party_address(cursor, party_id: str, data: dict) -> None:
    address_id = data.get("address_id")

    # Existing address: update exact row
    if address_id:
        cursor.execute(
            """
            UPDATE p_address
            SET line1 = %s,
                line2 = %s,
                city = %s,
                state = %s,
                postal_code = %s,
                updated_at = CURRENT_TIMESTAMP
            WHERE address_id = %s
            """,
            (
                data["address1"],
                data["address2"],
                data["city"],
                data["state"],
                data["zip"],
                address_id
            ),
        )

        if cursor.rowcount == 0:
            raise ValueError(
                f"Address {address_id} was not found for party {party_id}"
            )

        return

    # No address_id means this is genuinely a new address
    new_address_id = str(uuid.uuid4())

    cursor.execute(
        """
        INSERT INTO p_address (
            address_id,
            party_id,
            line1,
            line2,
            city,
            state,
            postal_code,
            country,
            is_active,
            created_at,
            updated_at,
            version_no
        )
        VALUES (
            %s, %s, %s, %s, %s, %s, %s,
            %s,
            true,
            CURRENT_TIMESTAMP,
            CURRENT_TIMESTAMP,
            1
        )
        """,
        (
            new_address_id,
            party_id,
            data["address1"],
            data["address2"],
            data["city"],
            data["state"],
            data["zip"],
            "USA",
        ),
    )
def _create_party(party_type: str):
    data = _entity_payload()
    validation_error = _validate_entity_payload(data)
    if validation_error:
        return validation_error

    party_id = str(uuid.uuid4())

    try:
        with _ehr_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO p_party (
                        party_id,
                        party_type,
                        first_name,
                        middle_name,
                        last_name,
                        date_of_birth,
                        phone,
                        email,
                        is_active
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, true)
                    """,
                    (
                        party_id,
                        party_type,
                        data["first_name"],
                        data["middle_name"] or None,
                        data["last_name"],
                        data["dob"],
                        data["phone"] or None,
                        data["email"] or None,
                    ),
                )

                _upsert_party_address(cursor, party_id, data)

        logger.info("Created %s party_id=%s", party_type, party_id)
        return jsonify({
            "success": True,
            "message": f"{party_type.title()} created successfully.",
            "party_id": party_id,
        }), 201

    except Exception as exc:
        logger.exception("Unable to create %s: %s", party_type, exc)
        return jsonify({
            "success": False,
            "error": str(exc),
        }), 500


def _update_party(party_id: str, party_type: str):
    data = _entity_payload()
    validation_error = _validate_entity_payload(data)
    if validation_error:
        return validation_error

    try:
        with _ehr_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE p_party
                    SET first_name = %s,
                        middle_name = %s,
                        last_name = %s,
                        date_of_birth = %s,
                        phone = %s,
                        email = %s
                    WHERE party_id = %s
                    """,
                    (
                        data["first_name"],
                        data["middle_name"] or None,
                        data["last_name"],
                        data["dob"],
                        data["phone"] or None,
                        data["email"] or None,
                        party_id
                    ),
                )

                if cursor.rowcount == 0:
                    return jsonify({
                        "success": False,
                        "error": f"{party_type.title()} record not found."
                    }), 404

                _upsert_party_address(cursor, party_id, data)

        logger.info("Updated %s party_id=%s", party_type, party_id)
        return jsonify({
            "success": True,
            "message": f"{party_type.title()} updated successfully.",
            "party_id": party_id,
        })

    except Exception as exc:
        logger.exception("Unable to update %s %s: %s", party_type, party_id, exc)
        return jsonify({
            "success": False,
            "error": str(exc),
        }), 500


@disciplines_bp.route("/api/patients", methods=["POST"])
@handle_route_errors
def create_patient():
    return _create_party("PATIENT")


@disciplines_bp.route("/api/patients/<party_id>", methods=["PUT"])
@handle_route_errors
def update_patient(party_id: str):
    return _update_party(party_id, "PATIENT")


@disciplines_bp.route("/api/doctors", methods=["POST"])
@handle_route_errors
def create_doctor():
    return _create_party("DOCTOR")


@disciplines_bp.route("/api/doctors/<party_id>", methods=["PUT"])
@handle_route_errors
def update_doctor(party_id: str):
    return _update_party(party_id, "DOCTOR")


@disciplines_bp.route("/api/patient/search", methods=["GET"])
@handle_route_errors
def search_patients_advanced():
    """Advanced patient search — source-aware.

    Query params (at least one required):
      first, last, middle, dob (YYYY-MM-DD), phone, email
    Optional:
      source  — "PCES_BASE" or "PCES101" (default) = local p_party on New VM;
                any other org_code (e.g. "Contoso101") = CCM/EHR external API on Old VM

    Case 1 (PCES_BASE / PCES101): queries p_party directly via _ehr_conn()
    Case 2 (Contoso101, etc.):    calls CCM/EHR API at CCM_EHR_BASE_URL
    """
    first  = (request.args.get("first",  "") or "").strip()
    last   = (request.args.get("last",   "") or "").strip()
    middle = (request.args.get("middle", "") or "").strip()
    dob    = (request.args.get("dob",    "") or "").strip()
    phone  = (request.args.get("phone",  "") or "").strip()
    email  = (request.args.get("email",  "") or "").strip()
    source = (request.args.get("source", "PCES_BASE") or "PCES_BASE").strip()
    # Treat PCES101 (the org_code for the default row) the same as PCES_BASE
    _LOCAL_SOURCES = {"PCES_BASE", "PCES101", "DEFAULT"}
    is_local = source.upper() in _LOCAL_SOURCES

    if not any([first, last, middle, dob, phone, email]):
        return jsonify([])

    # ── Case 2: external CCM/EHR API (e.g. Contoso101 → Old VM) ─────────────
    if not is_local:
        # Resolve org_code → display name for user-facing error messages
        source_label = source
        try:
            with _db_conn() as _sc:
                with _sc.cursor() as _scur:
                    _scur.execute(
                        "SELECT organization_name FROM pces_affiliates WHERE org_code = %s LIMIT 1",
                        (source,),
                    )
                    _srow = _scur.fetchone()
                    if _srow and _srow[0]:
                        source_label = _srow[0]
        except Exception:
            pass  # keep source_label as org_code fallback

        try:
            results = ccm_ehr_client.search_patients(
                first_name=first,
                last_name=last,
                date_of_birth=dob,
                phone=phone,
                email=email,
            )
            return jsonify(results)
        except CCMEHRAuthError as exc:
            logger.warning("search_patients_advanced[ext]: auth error — %s", exc)
            return jsonify({"error": "CCM/EHR token missing or expired. Contact admin.", "auth_error": True}), 401
        except CCMEHRError as exc:
            logger.warning("search_patients_advanced[ext]: server unavailable (%s) — %s", source_label, exc)
            return jsonify({
                "error": f"{source_label} data not available, please try later",
                "server_error": True,
            }), 503
        except Exception as exc:
            logger.error("search_patients_advanced[ext]: unexpected — %s", exc)
            return jsonify({
                "error": f"{source_label} data not available, please try later",
                "server_error": True,
            }), 503

    # ── Case 1: local PCES_BASE (p_party direct query) ───────────────────────
    try:
        conditions = ["pp.party_type = 'PATIENT'", "pp.is_active = true"]
        params: list = []

        if first:
            conditions.append("LOWER(pp.first_name) LIKE %s")
            params.append(f"%{first.lower()}%")
        if last:
            conditions.append("LOWER(pp.last_name) LIKE %s")
            params.append(f"%{last.lower()}%")
        if middle:
            conditions.append("LOWER(pp.middle_name) LIKE %s")
            params.append(f"%{middle.lower()}%")
        if dob:
            conditions.append("CAST(pp.date_of_birth AS TEXT) LIKE %s")
            params.append(f"%{dob}%")
        if phone:
            conditions.append("pp.phone LIKE %s")
            params.append(f"%{phone}%")
        if email:
            conditions.append("LOWER(pp.email) LIKE %s")
            params.append(f"%{email.lower()}%")

        where = " AND ".join(conditions)
        sql_q = f"""
            SELECT pp.party_id, pp.first_name, pp.middle_name, pp.last_name,
                   pp.date_of_birth, pa.line1, pa.city, pa.state, pa.postal_code
            FROM p_party pp
            LEFT JOIN p_address pa ON pa.party_id = pp.party_id
            WHERE {where}
            ORDER BY pp.last_name, pp.first_name
            LIMIT 20
        """
        with _ehr_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql_q, params)
                rows = cursor.fetchall()

        results = []
        for row in rows:
            pid, fn, mn, ln, dob_val, line1, city, state, postal = row
            name_parts = [p for p in [fn, mn, ln] if p]
            results.append({
                "patient_id":  str(pid) if pid else "",
                "first_name":  fn or "",
                "middle_name": mn or "",
                "last_name":   ln or "",
                "full_name":   " ".join(name_parts),
                "dob":         str(dob_val)[:10] if dob_val else "",
                "address1":    line1 or "",
                "city":        city or "",
                "state":       state or "",
                "zip":         postal or "",
            })
        return jsonify(results)

    except Exception as exc:
        logger.warning("search_patients_advanced[local]: DB unavailable — %s", exc)
        return jsonify([])



# ============================================================
# Patient Scheduling - Hospital picker
# ============================================================

@disciplines_bp.route("/api/hospitals/first20", methods=["GET"])
@handle_route_errors
def get_first_20_hospitals():
    """Return active ORGANIZATION parties for the Scheduling hospital picker.

    In the deployed EHR model, hospitals/health-care organizations are stored
    in ehr_ccm_schema.p_party with party_type = 'ORGANIZATION'.  The party_id
    is the identifier retained by the Scheduling UI and later maps to
    p_schedule.hospital_id.
    """
    try:
        with _ehr_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        party_id,
                        name,
                        hospital_code,
                        hospital_location
                    FROM ehr_ccm_schema.p_party
                    WHERE party_type = 'ORGANIZATION'
                      AND is_active = TRUE
                    ORDER BY name
                    LIMIT 20
                    """
                )
                rows = cursor.fetchall()

        hospitals = []

        for (
            party_id,
            name,
            hospital_code,
            hospital_location,
        ) in rows:
            hospitals.append({
                "hospital_id": str(party_id),
                "hospital_name": name or "",
                "hospital_code": hospital_code or "",
                "location": hospital_location or "",
            })

        return jsonify(hospitals)

    except Exception as exc:
        logger.exception(
            "Unable to retrieve hospitals for Scheduling: %s",
            exc
        )
        return jsonify({"error": str(exc)}), 500


@disciplines_bp.route("/api/hospitals/search", methods=["GET"])
@handle_route_errors
def search_hospitals():
    """Hospital search — calls the CCM/EHR external API.

    Query params (at least one required):
      firm_name, location, hospital_code, phone, email
    Returns [{hospital_id, firm_name, location, hospital_code, phone, email}]
    """
    firm_name     = (request.args.get("firm_name",     "") or "").strip()
    location      = (request.args.get("location",      "") or "").strip()
    hospital_code = (request.args.get("hospital_code", "") or "").strip()
    phone         = (request.args.get("phone",         "") or "").strip()
    email         = (request.args.get("email",         "") or "").strip()

    if not any([firm_name, location, hospital_code, phone, email]):
        return jsonify({"success": False, "message": "At least one search parameter is required."}), 400

    try:
        results = ccm_ehr_client.search_hospitals(
            firm_name=firm_name,
            location=location,
            hospital_code=hospital_code,
            phone=phone,
            email=email,
        )
        return jsonify({"success": True, "count": len(results), "data": results})

    except CCMEHRAuthError as exc:
        logger.warning("search_hospitals: CCM/EHR auth error — %s", exc)
        return jsonify({"success": False, "message": "CCM/EHR token missing or expired. Contact admin.", "auth_error": True}), 401

    except CCMEHRError as exc:
        logger.warning("search_hospitals: CCM/EHR unavailable — %s", exc)
        return jsonify({"success": False, "message": "CCM/EHR service unavailable.", "data": []}), 503

    except Exception as exc:
        logger.error("search_hospitals: unexpected error — %s", exc)
        return jsonify({"success": False, "message": "Internal error.", "data": []}), 500


@disciplines_bp.route("/api/affiliated-sources", methods=["GET"])
@handle_route_errors
def get_affiliated_sources():
    """Return the list of affiliated organisation sources for the Select Source dropdown.

    Queries pces_affiliates from pces_base. The first entry is always
    Default PCES (local New VM search). All other active entries are
    external sources (routed to CCM/EHR API on Old VM).

    Returns:
        [
          {"id": "PCES_BASE",   "firm_name": "Default PCES",      "is_default": true},
          {"id": "Contoso101",  "firm_name": "Contoso Hospitals",  "is_default": false},
          …
        ]
    """
    sources: list[dict] = []
    try:
        with _db_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT affl_id, organization_name, org_code, city, state
                    FROM pces_affiliates
                    WHERE (enddate IS NULL OR enddate >= CURRENT_DATE)
                    ORDER BY affl_id
                    LIMIT 50
                    """
                )
                rows = cursor.fetchall()

        for affl_id, org_name, org_code, city, state in rows:
            is_default = (org_code or "").upper() in ("PCES101", "PCES_BASE", "DEFAULT")
            location_parts = [p for p in [city, state] if p]
            sources.append({
                "id":         "PCES_BASE" if is_default else (org_code or str(affl_id)),
                "firm_name":  org_name or org_code or f"Source {affl_id}",
                "location":   ", ".join(location_parts),
                "is_default": is_default,
            })

    except Exception as exc:
        logger.warning("get_affiliated_sources: pces_affiliates query failed — %s", exc)

    # Always guarantee at least the default source
    if not any(s["is_default"] for s in sources):
        sources.insert(0, {
            "id": "PCES_BASE", "firm_name": "Default PCES",
            "location": "", "is_default": True
        })
    else:
        # Move default to front
        sources.sort(key=lambda s: (0 if s["is_default"] else 1, s["firm_name"]))

    return jsonify(sources)


@disciplines_bp.route("/api/patient/<patient_id>", methods=["GET"])
@handle_route_errors
def get_patient_info(patient_id: str):
    """Return basic demographic info for a single patient from p_party + p_address."""
    try:
        with _ehr_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT pp.party_id, pp.first_name, pp.middle_name, pp.last_name,
                           pp.date_of_birth, pp.gender, pp.phone, pp.email,
                           pa.line1, pa.city, pa.state, pa.postal_code
                    FROM p_party pp
                    LEFT JOIN p_address pa ON pa.party_id = pp.party_id
                    WHERE pp.party_id = %s AND pp.party_type = 'PATIENT'
                    LIMIT 1
                    """,
                    (patient_id,),
                )
                row = cursor.fetchone()

        if not row:
            return jsonify({"db_available": True, "error": "Patient not found"}), 404

        pid, first, middle, last, dob, gender, phone, email, line1, city, state, postal = row
        parts = [p for p in [first, middle, last] if p]
        full_name = " ".join(parts)

        age = None
        if dob:
            from datetime import date as _date
            today = _date.today()
            try:
                birth = dob if hasattr(dob, "year") else _date.fromisoformat(str(dob)[:10])
                age = today.year - birth.year - (
                    (today.month, today.day) < (birth.month, birth.day)
                )
            except Exception:
                age = None

        addr_parts = [p for p in [line1, city, state, postal] if p]
        return jsonify({
            "patient_id": str(pid),
            "full_name":  full_name,
            "dob":        str(dob)[:10] if dob else "",
            "age":        age,
            "gender":     gender or "—",
            "phone":      phone or "",
            "email":      email or "",
            "address":    ", ".join(addr_parts),
        })

    except Exception as exc:
        logger.warning("get_patient_info: DB unavailable for %s (%s) — returning partial", patient_id, exc)
        return jsonify({
            "patient_id":   patient_id,
            "full_name":    None,
            "age":          None,
            "gender":       "—",
            "db_available": False,
        })


@disciplines_bp.route("/api/patient/<patient_id>/history", methods=["GET"])
@handle_route_errors
def get_patient_history(patient_id: str):
    """Return last 3 encounters with diagnosis codes for a patient from pces_ehr_ccm."""
    try:
        with _ehr_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    WITH latest_encounters AS (
                        SELECT encounter_id, provider_id, encounter_date
                        FROM p_encounter
                        WHERE patient_id = %s
                        ORDER BY encounter_date DESC
                        LIMIT 3
                    )
                    SELECT
                        e.encounter_id,
                        d.diagnosis_id,
                        d.code,
                        d.description,
                        provider.first_name  AS doctor_first_name,
                        provider.last_name   AS doctor_last_name,
                        e.encounter_date
                    FROM latest_encounters e
                    INNER JOIN p_diagnosis d   ON d.encounter_id = e.encounter_id
                    INNER JOIN p_party provider ON e.provider_id  = provider.party_id
                    ORDER BY e.encounter_date DESC, d.diagnosis_id
                    """,
                    (patient_id,),
                )
                rows = cursor.fetchall()

        if not rows:
            return jsonify({"rows": [], "db_available": True})

        result = []
        for idx, row in enumerate(rows, 1):
            enc_id, diag_id, code, description, doc_first, doc_last, enc_date = row
            date_str = ""
            if enc_date:
                try:
                    from datetime import date as _date
                    d = enc_date if hasattr(enc_date, "strftime") else _date.fromisoformat(str(enc_date)[:10])
                    date_str = d.strftime("%d/%m/%Y")
                except Exception:
                    date_str = str(enc_date)[:10]
            result.append({
                "s_no":        idx,
                "date":        date_str,
                "code":        code or "",
                "description": description or "",
                "doctor":      f"{doc_first or ''} {doc_last or ''}".strip(),
            })

        return jsonify({"rows": result, "db_available": True})

    except Exception as exc:
        logger.warning("get_patient_history: DB unavailable for %s (%s)", patient_id, exc)
        return jsonify({"rows": [], "db_available": False, "error": str(exc)})


@disciplines_bp.route("/api/patient/<patient_id>/allergies", methods=["GET"])
@handle_route_errors
def get_patient_allergies(patient_id: str):
    """Return known allergens for a patient from p_allergy (pces_ehr_ccm)."""
    try:
        with _ehr_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT allergen FROM p_allergy WHERE patient_id = %s ORDER BY allergen",
                    (patient_id,),
                )
                rows = cursor.fetchall()

        allergens = [row[0] for row in rows if row[0]]
        return jsonify({"allergens": allergens, "db_available": True})

    except Exception as exc:
        logger.warning("get_patient_allergies: DB unavailable for %s (%s)", patient_id, exc)
        return jsonify({"allergens": [], "db_available": False, "error": str(exc)})


@disciplines_bp.route("/api/patient/<patient_id>/ai_summary", methods=["GET"])
@handle_route_errors
def get_patient_ai_summary(patient_id: str):
    """Generate an AI clinical summary using a patient's FULL diagnosis history and allergies."""
    import re as _re
    import time as _time
    from datetime import date as _date

    # ICD-10-like code pattern (e.g. I10, R07.9, R07.89, E11.9) used to distinguish
    # real coded diagnoses from free-text medication/dosage rows that share the same
    # p_diagnosis table in this dataset (no dedicated p_medication data exists yet).
    _ICD10_RE = _re.compile(r"^[A-Z][0-9]{2}(\.[0-9A-Z]{1,4})?$")

    # ── 1. Fetch ALL clinical data from pces_ehr_ccm ───────────────────────
    history_rows = []
    allergens = []
    db_available = True

    try:
        with _ehr_conn() as conn:
            with conn.cursor() as cur:
                # No LIMIT — use complete encounter history for a richer summary
                cur.execute(
                    """
                    SELECT
                        d.code,
                        d.description,
                        provider.first_name,
                        provider.last_name,
                        e.encounter_date
                    FROM p_encounter e
                    INNER JOIN p_diagnosis d    ON d.encounter_id = e.encounter_id
                    INNER JOIN p_party provider ON e.provider_id  = provider.party_id
                    WHERE e.patient_id = %s
                    ORDER BY e.encounter_date DESC, d.diagnosis_id
                    """,
                    (patient_id,),
                )
                history_rows = cur.fetchall()

                cur.execute(
                    "SELECT allergen FROM p_allergy WHERE patient_id = %s ORDER BY allergen",
                    (patient_id,),
                )
                allergens = [row[0] for row in cur.fetchall() if row[0]]

    except Exception as exc:
        logger.warning("get_patient_ai_summary: DB unavailable for %s (%s)", patient_id, exc)
        db_available = False

    if not db_available:
        return jsonify({"summary": "", "db_available": False,
                        "error": "Clinical database unavailable"}), 503

    if not history_rows:
        return jsonify({"summary": "", "db_available": True,
                        "message": "No clinical history available for this patient."})

    # ── 2. Build clinical prompt ───────────────────────────────────────────
    history_lines = []
    active_conditions = []
    medications = []
    recent_encounters = []
    _seen_conditions = set()
    _seen_medications = set()
    _seen_encounters = set()

    for code, description, doc_first, doc_last, enc_date in history_rows:
        try:
            d = enc_date if hasattr(enc_date, "strftime") else _date.fromisoformat(str(enc_date)[:10])
            date_str = d.strftime("%d/%m/%Y")
        except Exception:
            date_str = str(enc_date)[:10] if enc_date else "Unknown date"
        doctor = f"{doc_first or ''} {doc_last or ''}".strip() or "Unknown physician"
        history_lines.append(f"- [{date_str}] {code}: {description} (Treating: Dr. {doctor})")

        # ── Classify each diagnosis row as a real coded condition or a
        # free-text medication/dosage entry (this dataset stores both in
        # p_diagnosis since p_medication has no rows). Deterministic, no LLM.
        code_clean = (code or "").strip()
        if _ICD10_RE.match(code_clean):
            if code_clean not in _seen_conditions and len(active_conditions) < 5:
                _seen_conditions.add(code_clean)
                active_conditions.append({
                    "code": code_clean,
                    "description": (description or "").strip(),
                    "date": date_str,
                })
        else:
            if code_clean and code_clean not in _seen_medications and len(medications) < 5:
                _seen_medications.add(code_clean)
                medications.append({
                    "name": code_clean,
                    "instructions": (description or "").strip(),
                    "date": date_str,
                })

        enc_key = (date_str, doctor)
        if enc_key not in _seen_encounters and len(recent_encounters) < 5:
            _seen_encounters.add(enc_key)
            recent_encounters.append({"date": date_str, "doctor": doctor})

    allergy_text = ", ".join(allergens) if allergens else "No known allergies"

    # ── Condense each active condition to a single concise clinical sentence
    # via the LLM (quick-glance summary card should not show raw/verbose
    # diagnosis text). Best-effort — falls back to the raw description if the
    # LLM call or response parsing fails.
    if active_conditions:
        try:
            llm = current_app.config.get("LLM_INSTANCE")
            if llm:
                condense_prompt = (
                    "You are a clinical assistant. For each diagnosis below, rewrite its "
                    "description as ONE concise clinical sentence (no more than 20 words), "
                    "suitable for a quick-glance patient summary card. Preserve the same "
                    "medical meaning; do not add new information.\n\n"
                    "Return ONLY a JSON array of strings, one per diagnosis, in the exact "
                    "same order as given, with no extra commentary or markdown.\n\n"
                    "Diagnoses:\n"
                    + "\n".join(
                        f"{i + 1}. [{c['code']}] {c['description']}"
                        for i, c in enumerate(active_conditions)
                    )
                )
                condense_response = llm.invoke(condense_prompt)
                condense_text = (
                    condense_response.content.strip()
                    if hasattr(condense_response, "content")
                    else str(condense_response).strip()
                )
                condense_text = _re.sub(
                    r"^```(?:json)?|```$", "", condense_text, flags=_re.MULTILINE
                ).strip()
                concise_list = json.loads(condense_text)
                if isinstance(concise_list, list) and len(concise_list) == len(active_conditions):
                    for cond, concise in zip(active_conditions, concise_list):
                        if isinstance(concise, str) and concise.strip():
                            cond["description"] = concise.strip()
        except Exception as exc:
            logger.warning(
                "get_patient_ai_summary: condense active_conditions failed for %s (%s)",
                patient_id, exc,
            )

    # No lab-results table/data exists in the current schema — render an
    # honest empty state on the frontend rather than fabricating values.
    labs = []
    labs_message = "No recent lab results recorded"

    structured_sections = {
        "active_conditions": active_conditions,
        "medications": medications,
        "recent_encounters": recent_encounters,
        "allergies": allergens,
        "labs": labs,
        "labs_message": labs_message,
    }

    prompt = (
        "You are a clinical decision-support assistant. "
        "Given the following complete patient encounter history and known allergies, "
        "write a concise AI-generated clinical summary in no more than 500 words covering:\n"
        "1. Clinical presentation and key diagnoses\n"
        "2. Treatment considerations and risk factors\n"
        "3. Allergy implications for management\n\n"
        f"Patient Encounter History ({len(history_lines)} records, most recent first):\n"
        + "\n".join(history_lines)
        + f"\n\nKnown Allergies: {allergy_text}\n\n"
        "Write the summary in a professional clinical tone suitable for a treating physician. "
        "Keep the total response under 500 words."
    )

    # ── 3. Invoke LLM with timing ──────────────────────────────────────────
    try:
        llm = current_app.config.get("LLM_INSTANCE")
        if not llm:
            return jsonify({"summary": "", "error": "LLM not initialised"}), 503

        model_name = getattr(llm, "model_name", None) or getattr(llm, "model", None) or "unknown"
        t_start = _time.time()
        response = llm.invoke(prompt)
        generation_time_ms = round((_time.time() - t_start) * 1000)

        summary = (
            response.content.strip()
            if hasattr(response, "content")
            else str(response).strip()
        )
        logger.info(
            "get_patient_ai_summary: patient=%s model=%s encounters=%d time=%dms chars=%d",
            patient_id, model_name, len(history_lines), generation_time_ms, len(summary),
        )
        return jsonify({
            "summary": summary,
            "db_available": True,
            "model": model_name,
            "generation_time_ms": generation_time_ms,
            "encounter_count": len(history_lines),
            **structured_sections,
        })

    except Exception as exc:
        logger.error("get_patient_ai_summary: LLM error for %s (%s)", patient_id, exc)
        return jsonify({"summary": "", "error": f"LLM error: {exc}", **structured_sections}), 500


