"""
Patient Context Service — query-time clinical context for the LLM.

Builds a small, deterministic "Patient Context" block (age, sex, ethnicity,
known medications, allergies) that is:
  1. Logged to the console for every query that has a patient selected.
  2. Appended to the context passed into the LLM/RAG pipeline.

Data sources (all read-only, `pces_ehr_ccm` via ``Config.PG_TOOL_*``):
  - p_party      → date_of_birth, gender, ethnicity (age is computed, sex =
                   gender, ethnicity is read directly from the column)
  - p_diagnosis  → non-ICD10-coded rows are free-text medication entries in
                   this dataset (there is no dedicated p_medication table —
                   same deterministic classification used by
                   routes/disciplines.py::get_patient_ai_summary, but without
                   the LLM condense/structure steps since only raw names are
                   needed here)
  - p_allergy    → allergen list

If ethnicity is NULL/blank for a given patient row, "Unknown" is reported
rather than guessed or fabricated.

If the patient has no medication rows on record (or the DB is unreachable),
a fixed fallback list is used so the context is never empty. This fallback
is for demo/continuity purposes only and is never preferred over real data.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from datetime import date as _date
from typing import Any, Generator

import psycopg

from config import Config
from utils.error_handlers import get_logger

logger = get_logger(__name__)

# Same ICD-10-like pattern used in routes/disciplines.py::get_patient_ai_summary
# to distinguish real coded diagnoses from free-text medication/dosage rows
# that share the p_diagnosis table in this dataset.
_ICD10_RE = re.compile(r"^[A-Z][0-9]{2}(\.[0-9A-Z]{1,4})?$")

# Fallback medication history — used only when the patient has zero
# medication rows on record in p_diagnosis (or the DB is unreachable).
_FALLBACK_MEDICATIONS: list[str] = [
    "Mentho, Tylenol 250mg, Gabapentin, Menthocorbomol 500mg",
    "Oxycodone, Mentho, Tylenol PM, Gabapentin, Menthocorbomol 500mg",
    "Oxycodone 5ml/5ml",
    "Menthocorbomol 500mg",
    "Gabapentin 300mg",
]


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


def _compute_age(dob: Any) -> int | None:
    if not dob:
        return None
    try:
        today = _date.today()
        birth = dob if hasattr(dob, "year") else _date.fromisoformat(str(dob)[:10])
        return today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
    except Exception:
        return None


def build_patient_context(patient_id: str | None) -> dict[str, Any]:
    """Fetch age/sex/ethnicity/medications/allergies for *patient_id*.

    Returns an empty dict when no patient is selected. Returns a
    best-effort partial dict (never raises) when the DB is unreachable —
    medications will fall back to :data:`_FALLBACK_MEDICATIONS` in that case.
    """
    if not patient_id:
        return {}

    age: int | None = None
    sex: str = "Unknown"
    ethnicity: str = "Unknown"
    medications: list[str] = []
    allergies: list[str] = []
    db_available = True

    try:
        with _ehr_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT date_of_birth, gender, ethnicity FROM p_party "
                    "WHERE party_id = %s AND party_type = 'PATIENT' LIMIT 1",
                    (patient_id,),
                )
                row = cur.fetchone()
                if row:
                    dob, gender, party_ethnicity = row
                    age = _compute_age(dob)
                    sex = (gender or "Unknown").strip() or "Unknown"
                    ethnicity = (party_ethnicity or "Unknown").strip() or "Unknown"

                # Medications: free-text (non-ICD10-code) rows from p_diagnosis,
                # grouped per encounter date (most recent first), deduplicated
                # per encounter so the same medication list isn't repeated.
                cur.execute(
                    """
                    SELECT d.code, d.description, e.encounter_date
                    FROM p_diagnosis d
                    INNER JOIN p_encounter e ON e.encounter_id = d.encounter_id
                    WHERE e.patient_id = %s
                    ORDER BY e.encounter_date DESC, d.diagnosis_id
                    """,
                    (patient_id,),
                )
                med_rows = cur.fetchall()

                cur.execute(
                    "SELECT allergen FROM p_allergy WHERE patient_id = %s ORDER BY allergen",
                    (patient_id,),
                )
                allergies = [r[0] for r in cur.fetchall() if r[0]]

        # Group non-ICD10 rows (medications) by encounter date, preserving
        # order and de-duplicating identical names within the same encounter.
        by_encounter: dict[str, list[str]] = {}
        order: list[str] = []
        for code, description, enc_date in med_rows:
            code_clean = (code or "").strip()
            if not code_clean or _ICD10_RE.match(code_clean):
                continue  # real coded diagnosis, not a medication row
            key = str(enc_date)
            if key not in by_encounter:
                by_encounter[key] = []
                order.append(key)
            name = code_clean
            if name not in by_encounter[key]:
                by_encounter[key].append(name)

        for key in order:
            medications.append(", ".join(by_encounter[key]))

    except Exception as exc:
        logger.warning("build_patient_context: DB unavailable for %s (%s)", patient_id, exc)
        db_available = False

    used_fallback_medications = False
    if not medications:
        medications = list(_FALLBACK_MEDICATIONS)
        used_fallback_medications = True

    return {
        "patient_id": patient_id,
        "age": age,
        "sex": sex,
        # Read from p_party.ethnicity; falls back to "Unknown" if NULL/blank
        # or if the DB was unreachable (see except block above).
        "ethnicity": ethnicity,
        "medications": medications,
        "used_fallback_medications": used_fallback_medications,
        "allergies": allergies,
        "db_available": db_available,
    }


def format_patient_context_block(data: dict[str, Any]) -> str:
    """Render *data* (from :func:`build_patient_context`) as an LLM-readable text block."""
    if not data:
        return ""

    age = data.get("age")
    age_str = str(age) if age is not None else "Unknown"
    sex = data.get("sex") or "Unknown"
    ethnicity = data.get("ethnicity") or "Unknown"
    medications = data.get("medications") or []
    allergies = data.get("allergies") or []

    lines = [
        "Patient Context:",
        f"- Age: {age_str}, Sex: {sex}, Ethnicity: {ethnicity}",
    ]

    if medications:
        med_summary = "; ".join(medications)
        lines.append(f"- So far patient had the following medicines: {med_summary}")

    allergy_summary = ", ".join(allergies) if allergies else "No known allergies"
    lines.append(f"- Allergies: {allergy_summary}")

    return "\n".join(lines)


__all__ = ["build_patient_context", "format_patient_context_block"]
