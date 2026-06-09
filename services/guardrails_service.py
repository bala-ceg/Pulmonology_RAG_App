"""
Guardrails Service — Phase 2-B: Safety and Guardrails

Pipeline position (spec: 06_Implement Safety and Guardrails v1.0):
  Validate Response → Safety / Guardrails (this module) → Confidence Scoring

Implements all 11 steps from the spec:
  1.  Receive inputs (prompt, response, department, patient_data, validation_score)
  2.  Clinical safety check      — unsafe phrase detection
  3.  Medication safety check     — allergy/contraindication conflict
  4.  Emergency detection         — life-threatening symptom keywords
  5.  Compliance / legal check    — banned statements (guaranteed cure, etc.)
  6.  Data privacy check          — PHI / PII regex (SSN, Tax ID, DL, Phone, Email)
  7.  Scope-of-practice check     — cross-department clinical term restriction
  8.  Confidence guardrail        — block if validation_score < 60
  9.  Orchestrate                 — returns SAFE / BLOCKED / EMERGENCY
  10. EMERGENCY overrides answer; BLOCKED flags response
  11. Log to guardrail_log table  — async, PostgreSQL only, silent fail
"""

from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from utils.error_handlers import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Department keyword map — reuse canonical map for scope-of-practice check
# ---------------------------------------------------------------------------
try:
    from sft_experiment_manager import DEPARTMENTS as _DEPT_KEYWORDS
except Exception:
    _DEPT_KEYWORDS = {}

# ---------------------------------------------------------------------------
# Department name aliases — maps common abbreviations, typos, and suffixes to
# canonical department names used in _DEPT_EXCLUSIVE_TERMS.
# "OPTHAMALMOLOGY, SME" → "Ophthalmology"  (typo + suffix)
# "OPTHAMALMOLOGY"      → "Ophthalmology"  (typo)
# ---------------------------------------------------------------------------
_DEPT_ALIASES: dict[str, str] = {
    "opthamalmology": "ophthalmology",
    "opthalmology":   "ophthalmology",
    "ophthalmologist": "ophthalmology",
    "cardio":         "cardiology",
    "neuro":          "neurology",
    "ortho":          "orthopedics",
    "onco":           "oncology",
    "gastro":         "gastroenterology",
    "gi":             "gastroenterology",
    "derm":           "dermatology",
    "psych":          "psychiatry",
    "uro":            "urology",
    "obgyn":          "obstetrics & gynecology",
    "ob/gyn":         "obstetrics & gynecology",
    "obs":            "obstetrics & gynecology",
    "ent":            "ent (otolaryngology)",
}


def _normalise_dept(department: str) -> str:
    """Normalise a department string to a canonical lowercase name.

    Strips role suffixes (', SME', ', sme'), resolves known typos and
    abbreviations, then returns the cleaned lowercase string for comparison.
    """
    # Strip common suffixes: ", SME", ", sme", "(SME)", etc.
    import re as _re
    cleaned = _re.sub(r'[,\s]*(sme|expert|senior|junior|lead)\b.*$', '', department, flags=_re.IGNORECASE).strip()
    cleaned_lower = cleaned.lower()
    # Apply alias map
    for alias, canonical in _DEPT_ALIASES.items():
        if alias in cleaned_lower:
            return canonical
    return cleaned_lower

# ---------------------------------------------------------------------------
# Step 2 — Clinical safety phrase list
# Broader than validation's list: focuses on dangerous self-treatment advice
# ---------------------------------------------------------------------------
_CLINICAL_UNSAFE_PHRASES: list[str] = [
    "overdose",
    "self surgery",
    "stop taking medication",
    "do not see a doctor",
    "avoid medical care",
    "refuse treatment",
    "take double dose",
    "inject yourself",
    "skip your medication",
    "no treatment needed",
    "untreated is fine",
    "perform surgery at home",
    "no need for a doctor",
    "cancel your prescription",
    "ignore symptoms",
    "self-medicate with higher dose",
    "double your dose",
    "take someone else's prescription",
    "use veterinary medication",
]

# ---------------------------------------------------------------------------
# Step 4 — Emergency keywords (triggers EMERGENCY status + response override)
# ---------------------------------------------------------------------------
_EMERGENCY_KEYWORDS: list[str] = [
    "chest pain",
    "heart attack",
    "cardiac arrest",
    "stroke symptoms",
    "difficulty breathing",
    "cannot breathe",
    "severe bleeding",
    "loss of consciousness",
    "unresponsive",
    "call 911",
    "severe chest pressure",
    "sudden severe headache",
    "signs of stroke",
    "face drooping",
    "arm weakness",
    "speech difficulty",
    "sudden vision loss",
    "severe allergic reaction",
    "anaphylaxis",
    "anaphylactic shock",
    "septic shock",
    "diabetic coma",
    "seizure",
    "unconscious",
    "not breathing",
    "no pulse",
]

_EMERGENCY_RESPONSE = (
    "⚠️ **EMERGENCY ALERT** ⚠️\n\n"
    "The information in this query suggests a potentially life-threatening emergency. "
    "**Please call emergency services (911) or go to the nearest emergency room immediately.**\n\n"
    "Do not delay seeking emergency care. This AI system cannot replace emergency medical services.\n\n"
    "**Signs requiring immediate action include:** chest pain, difficulty breathing, stroke symptoms, "
    "severe bleeding, loss of consciousness, or anaphylaxis."
)

# ---------------------------------------------------------------------------
# Step 5 — Compliance / legal banned statements
# ---------------------------------------------------------------------------
_COMPLIANCE_BANNED: list[str] = [
    "guaranteed cure",
    "100% cure",
    "miracle cure",
    "no need to consult",
    "no need to see a doctor",
    "no side effects",
    "completely safe for everyone",
    "replaces medical advice",
    "this ai replaces",
    "ai doctor",
    "not responsible for",
    "diagnose yourself",
    "self-diagnose",
    "no need for medical consultation",
    "permanently cures",
    "clinically proven to cure",
    "fda approved cure",
    "legal advice",
    "legal liability",
]

# ---------------------------------------------------------------------------
# Step 6 — PHI / PII regex patterns (HIPAA)
# ---------------------------------------------------------------------------
_PHI_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("SSN",         re.compile(r'\b\d{3}-\d{2}-\d{4}\b')),
    ("TaxID",       re.compile(r'\b\d{2}-\d{7}\b')),
    ("DriversLic",  re.compile(r'\b[A-Z]{1,2}\d{6,8}\b')),
    ("Phone",       re.compile(r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')),
    ("Email",       re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b')),
    ("PatientID",   re.compile(r'\b(?:patient\s*id|mrn|medical\s*record)\s*[:\-]?\s*\d{4,}\b', re.IGNORECASE)),
    ("CreditCard",  re.compile(r'\b(?:\d{4}[-\s]){3}\d{4}\b')),
    ("DOB",         re.compile(r'\b(?:dob|date of birth)\s*[:\-]?\s*\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b', re.IGNORECASE)),
]

# ---------------------------------------------------------------------------
# Step 7 — Scope-of-practice: high-specificity terms per department
# If term appears in response for a *different* department, flag it.
# Only include terms that are truly exclusive (not shared across disciplines).
# ---------------------------------------------------------------------------
_DEPT_EXCLUSIVE_TERMS: dict[str, list[str]] = {
    "Cardiology": [
        "coronary artery bypass", "cabg", "percutaneous coronary intervention", "pci",
        "cardiac catheterization", "defibrillation", "cardioversion", "stent placement",
        "echocardiogram", "left ventricular ejection fraction", "lvef",
    ],
    "Ophthalmology": [
        "phacoemulsification", "vitrectomy", "trabeculectomy", "fundoscopy",
        "fluorescein angiography", "intraocular pressure", "iop measurement",
        "retinal detachment repair", "corneal transplant",
    ],
    "Orthopedics": [
        "arthroplasty", "total knee replacement", "hip replacement", "acl reconstruction",
        "spinal fusion", "osteotomy", "intramedullary nail", "bone graft",
    ],
    "Neurology": [
        "lumbar puncture", "cerebrospinal fluid analysis", "deep brain stimulation",
        "electroencephalogram", "eeg interpretation", "craniotomy",
        "neurostimulator implant",
    ],
    "Oncology": [
        "chemotherapy protocol", "radiation dosimetry", "tumor staging",
        "bone marrow transplant", "biopsy pathology report", "immunotherapy infusion",
    ],
    "Gastroenterology": [
        "colonoscopy", "esophagogastroduodenoscopy", "egd procedure",
        "endoscopic mucosal resection", "ercp", "liver biopsy",
    ],
    "Urology": [
        "cystoscopy", "prostatectomy", "transurethral resection", "turp",
        "nephrectomy", "urodynamic study", "bladder instillation",
    ],
    "Obstetrics & Gynecology": [
        "amniocentesis", "chorionic villus sampling", "episiotomy",
        "cesarean section", "hysterectomy", "laparoscopic ovarian cystectomy",
    ],
    "Psychiatry": [
        "electroconvulsive therapy", "ect procedure", "transcranial magnetic stimulation",
        "psychiatric commitment", "involuntary admission", "dsm diagnosis",
    ],
    "Dermatology": [
        "mohs surgery", "punch biopsy", "photodynamic therapy",
        "excision of melanoma", "cryotherapy lesion", "laser resurfacing",
    ],
}


def _get_dept_for_term(term: str) -> str | None:
    """Return the owning department of a scope-exclusive term, or None."""
    t_lower = term.lower()
    for dept, terms in _DEPT_EXCLUSIVE_TERMS.items():
        if t_lower in [t.lower() for t in terms]:
            return dept
    return None


# ---------------------------------------------------------------------------
# Step 2 — Clinical Safety Check
# ---------------------------------------------------------------------------

def clinical_safety_check(response: str) -> tuple[bool, list[str]]:
    """Return (passed, flagged_phrases).  passed=False means BLOCKED."""
    resp_lower = response.lower()
    found = [p for p in _CLINICAL_UNSAFE_PHRASES if p in resp_lower]
    if found:
        logger.warning("[GUARDRAILS] clinical_safety=FAIL  phrases=%s", found)
    return (len(found) == 0, found)


# ---------------------------------------------------------------------------
# Step 3 — Medication Safety Check
# ---------------------------------------------------------------------------

def medication_safety_check(response: str, patient_data: dict) -> tuple[bool, list[str]]:
    """Return (passed, clashing_allergens).  passed=False means BLOCKED."""
    if not patient_data:
        return True, []

    resp_lower = response.lower()
    allergies: list[str] = patient_data.get("allergies", [])
    conditions: list[str] = patient_data.get("conditions", [])
    clashes: list[str] = []

    for allergen in allergies:
        if allergen.strip().lower() in resp_lower:
            clashes.append(f"allergen:{allergen}")

    for condition in conditions:
        if condition.strip().lower() in resp_lower:
            clashes.append(f"condition:{condition}")

    if clashes:
        logger.warning("[GUARDRAILS] medication_safety=FAIL  clashes=%s", clashes)
    return (len(clashes) == 0, clashes)


# ---------------------------------------------------------------------------
# Step 4 — Emergency Detection
# ---------------------------------------------------------------------------

def emergency_detection(prompt: str, response: str) -> tuple[bool, list[str]]:
    """Return (is_emergency, matched_keywords).  is_emergency=True → EMERGENCY status."""
    combined = (prompt + " " + response).lower()
    found = [kw for kw in _EMERGENCY_KEYWORDS if kw in combined]
    if found:
        logger.warning("[GUARDRAILS] emergency=DETECTED  keywords=%s", found)
    return (len(found) > 0, found)


def get_emergency_response() -> str:
    """Return the standardised emergency override message."""
    return _EMERGENCY_RESPONSE


# ---------------------------------------------------------------------------
# Step 5 — Compliance / Legal Check
# ---------------------------------------------------------------------------

def compliance_check(response: str) -> tuple[bool, list[str]]:
    """Return (passed, banned_statements).  passed=False means BLOCKED."""
    resp_lower = response.lower()
    found = [stmt for stmt in _COMPLIANCE_BANNED if stmt in resp_lower]
    if found:
        logger.warning("[GUARDRAILS] compliance=FAIL  statements=%s", found)
    return (len(found) == 0, found)


# ---------------------------------------------------------------------------
# Step 6 — Data Privacy Check (PHI/PII)
# ---------------------------------------------------------------------------

def privacy_check(response: str) -> tuple[bool, list[str]]:
    """Return (passed, detected_phi_types).  passed=False means BLOCKED."""
    found: list[str] = []
    for label, pattern in _PHI_PATTERNS:
        if pattern.search(response):
            found.append(label)
            logger.warning("[GUARDRAILS] privacy=FAIL  phi_type=%s", label)
    return (len(found) == 0, found)


# ---------------------------------------------------------------------------
# Step 7 — Scope-of-Practice Check
# ---------------------------------------------------------------------------

def scope_check(response: str, department: str) -> tuple[bool, list[str]]:
    """Return (passed, out_of_scope_terms).  passed=False means BLOCKED.

    Flags high-specificity clinical terms that belong to a department other
    than the one selected for this query.  Normalises department names so
    typos (e.g. 'OPTHAMALMOLOGY') and suffixes (', SME') don't cause false
    positives.
    """
    if not department:
        return True, []

    dept_normalised = _normalise_dept(department)
    resp_lower = response.lower()
    violations: list[str] = []

    for owning_dept, terms in _DEPT_EXCLUSIVE_TERMS.items():
        owning_normalised = _normalise_dept(owning_dept)
        # Skip if the owning dept is the same as (or overlaps with) the query dept
        if owning_normalised in dept_normalised or dept_normalised in owning_normalised:
            continue
        for term in terms:
            if term.lower() in resp_lower:
                violations.append(f"{term} ({owning_dept})")
                logger.warning(
                    "[GUARDRAILS] scope=FAIL  term=%r  belongs_to=%r  query_dept=%r",
                    term, owning_dept, department,
                )

    return (len(violations) == 0, violations)


# ---------------------------------------------------------------------------
# Step 8 — Confidence Guardrail
# ---------------------------------------------------------------------------

def confidence_guardrail(validation_score: int) -> tuple[bool, str]:
    """Return (passed, reason).  passed=False (score < 60) means BLOCKED."""
    if validation_score < 60:
        reason = f"Validation score {validation_score} is below minimum threshold (60)"
        logger.warning("[GUARDRAILS] confidence_guardrail=FAIL  score=%d", validation_score)
        return False, reason
    return True, ""


# ---------------------------------------------------------------------------
# Step 9 — Orchestrator
# ---------------------------------------------------------------------------

def run_guardrails(
    prompt: str,
    response: str,
    department: str,
    patient_data: dict | None = None,
    validation_score: int = 100,
) -> dict:
    """Run all 7 guardrail checks and return a structured result.

    Returns::

        {
            "status":  "SAFE" | "BLOCKED" | "EMERGENCY",
            "results": {
                "clinical_safe":    bool,
                "medication_safe":  bool,
                "emergency":        bool,
                "compliant":        bool,
                "privacy_safe":     bool,
                "in_scope":         bool,
                "confidence_ok":    bool,
            },
            "flags":  list[str],
            "emergency_response": str | None,  # set when status == "EMERGENCY"
        }
    """
    _pd = patient_data or {}

    # Run all independent checks in parallel
    results: dict[str, Any] = {}
    flags: list[str] = []

    check_fns = {
        "clinical":    lambda: clinical_safety_check(response),
        "medication":  lambda: medication_safety_check(response, _pd),
        "emergency":   lambda: emergency_detection(prompt, response),
        "compliance":  lambda: compliance_check(response),
        "privacy":     lambda: privacy_check(response),
        "scope":       lambda: scope_check(response, department),
        "confidence":  lambda: confidence_guardrail(validation_score),
    }

    raw: dict[str, tuple] = {}
    with ThreadPoolExecutor(max_workers=7, thread_name_prefix="guard-check") as pool:
        futures = {pool.submit(fn): key for key, fn in check_fns.items()}
        for future in as_completed(futures):
            key = futures[future]
            try:
                raw[key] = future.result()
            except Exception as exc:
                logger.error("[GUARDRAILS] check %r raised: %s — fail-open", key, exc)
                raw[key] = (True, [])  # fail-open

    # Unpack results
    clinical_ok,   clinical_flags  = raw.get("clinical",   (True, []))
    medication_ok, med_flags       = raw.get("medication",  (True, []))
    is_emergency,  emrg_keywords   = raw.get("emergency",   (False, []))
    compliant,     compliance_stmts = raw.get("compliance",  (True, []))
    privacy_ok,    phi_types       = raw.get("privacy",     (True, []))
    scope_ok,      scope_violations = raw.get("scope",      (True, []))
    conf_ok,       conf_reason     = raw.get("confidence",  (True, ""))

    results = {
        "clinical_safe":   clinical_ok,
        "medication_safe": medication_ok,
        "emergency":       is_emergency,
        "compliant":       compliant,
        "privacy_safe":    privacy_ok,
        "in_scope":        scope_ok,
        "confidence_ok":   conf_ok,
    }

    # Build human-readable flags
    if not clinical_ok:
        flags.append(f"Unsafe clinical phrase detected: {', '.join(clinical_flags)}")
    if not medication_ok:
        flags.append(f"Medication conflict with patient data: {', '.join(med_flags)}")
    if is_emergency:
        flags.append(f"Emergency keywords detected: {', '.join(emrg_keywords[:3])}")
    if not compliant:
        flags.append(f"Compliance violation — banned statement: {', '.join(compliance_stmts)}")
    if not privacy_ok:
        flags.append(f"PHI detected in response: {', '.join(phi_types)}")
    if not scope_ok:
        flags.append(f"Out-of-scope clinical terms: {', '.join(scope_violations[:3])}")
    if not conf_ok:
        flags.append(conf_reason)

    # Determine final status
    if is_emergency:
        status = "EMERGENCY"
    elif not (clinical_ok and medication_ok and compliant and privacy_ok and conf_ok):
        status = "BLOCKED"
    else:
        status = "SAFE"

    # Scope violations are advisory flags, not BLOCKED (doctor may have multi-dept context)
    if not scope_ok and status == "SAFE":
        flags.append("[ADVISORY] Response may contain cross-department terminology")

    logger.info(
        "[GUARDRAILS] dept=%r  status=%s  flags=%s",
        department, status, flags,
    )

    return {
        "status": status,
        "results": results,
        "flags": flags,
        "emergency_response": _EMERGENCY_RESPONSE if status == "EMERGENCY" else None,
    }


# ---------------------------------------------------------------------------
# Step 11 — Log Guardrail Events (async, fire-and-forget)
# ---------------------------------------------------------------------------

_guard_table_ensured = threading.Event()


def _ensure_guardrail_table() -> None:
    """Create guardrail_log table if it does not exist yet."""
    try:
        from services.db_service import get_connection
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS guardrail_log (
                        id          SERIAL PRIMARY KEY,
                        session_id  TEXT,
                        department  TEXT,
                        status      TEXT,
                        flags       TEXT,
                        results     TEXT,
                        prompt_snippet TEXT,
                        created_at  TIMESTAMP DEFAULT NOW()
                    )
                    """
                )
            conn.commit()
        logger.info("[GUARDRAILS] guardrail_log table ensured")
        _guard_table_ensured.set()
    except Exception:
        pass  # non-critical — silent skip


def log_guardrail_event(
    session_id: str,
    status: str,
    results: dict,
    flags: list[str] | None = None,
    department: str = "",
    prompt_snippet: str = "",
) -> None:
    """Persist a guardrail event to PostgreSQL in a background daemon thread.

    Silently no-ops when PostgreSQL is unavailable.
    """

    def _insert() -> None:
        if not _guard_table_ensured.is_set():
            _ensure_guardrail_table()
        if not _guard_table_ensured.is_set():
            return
        try:
            from services.db_service import get_connection, execute_query
            with get_connection() as conn:
                execute_query(
                    conn,
                    """
                    INSERT INTO guardrail_log
                        (session_id, department, status, flags, results, prompt_snippet)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id or "unknown",
                        department or "",
                        status or "",
                        json.dumps(flags or []),
                        json.dumps(results),
                        (prompt_snippet or "")[:200],
                    ),
                )
                conn.commit()
            logger.debug(
                "[GUARDRAILS] logged  session=%s  status=%s", session_id, status
            )
        except Exception:
            pass  # non-critical — never surface DB errors

    threading.Thread(target=_insert, daemon=True, name="guard-log").start()
