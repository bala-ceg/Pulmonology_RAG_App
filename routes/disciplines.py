"""
Disciplines Blueprint

Routes:
  GET  /                          — index (creates new session)
  GET  /api/disciplines           — list available disciplines
  POST /api/validate_disciplines  — validate user's selection
  GET  /search_doctors            — autocomplete search in pces_users
  GET  /search_patients           — autocomplete search in patient table

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
from contextlib import contextmanager
from typing import Generator

import psycopg
from flask import Blueprint, current_app, jsonify, render_template, request

from config import Config
from utils.error_handlers import get_logger, handle_route_errors

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
    """Yield a psycopg connection using Config credentials."""
    with psycopg.connect(**Config.db_kwargs()) as conn:
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
                confidence = min(keyword_matches / len(keywords) * 100, 95)
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
            relevant_disciplines = self._ai_analyze_query(query)
            for d in relevant_disciplines:
                confidence_scores[d] = 70
            if has_files and "doctors_files" not in relevant_disciplines:
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

    def _ai_analyze_query(self, query: str) -> list[str]:
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
    return render_template("index.html")


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


@disciplines_bp.route("/search_doctors", methods=["GET"])
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
    """Search for patients by first_name and last_name from patient table."""
    try:
        query = request.args.get("q", "").strip().lower()
        if not query:
            return jsonify([])

        with _db_conn() as conn:
            with conn.cursor() as cursor:
                search_query = """
                SELECT DISTINCT patient_id, first_name, last_name 
                FROM patient 
                WHERE LOWER(first_name) LIKE %s 
                   OR LOWER(last_name) LIKE %s 
                   OR LOWER(CONCAT(first_name, ' ', last_name)) LIKE %s
                ORDER BY first_name, last_name
                LIMIT 10
                """
                pattern = f"%{query}%"
                cursor.execute(search_query, (pattern, pattern, pattern))
                results = cursor.fetchall()

        patients = [
            {
                "patient_id": row[0],
                "first_name": row[1],
                "last_name": row[2],
                "full_name": f"{row[1]} {row[2]}",
            }
            for row in results
            if row[1] and row[2]
        ]
        return jsonify(patients)

    except Exception as exc:
        logger.error("Error searching patients: %s", exc)
        return jsonify({"error": str(exc)}), 500
