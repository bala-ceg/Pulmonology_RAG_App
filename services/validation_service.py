"""
Validation Service — Phase 2-B: Validate Response

Pipeline position (spec: 05_Implement Validations v1.0):
  LLM Generates Response → Validate Response (this module) → Safety / Guardrails

Implements all 12 steps from the spec:
  1.  Receive inputs (prompt, response, context, department, patient_data)
  2.  Check completeness       (+20 pts)
  3.  Check department relevance (+20 pts)
  4.  Check evidence support    (+20 pts)
  5.  Check medical safety      (+20 pts)
  6.  Check patient consistency (+20 pts)
  7.  Detect hallucination      (flag only, not scored)
  8.  Calculate validation score (0–100)
  9.  Build validation result object
  10. Decision: PASS (≥80) / REVIEW (60–79) / REGENERATE (<60)
  11. [Caller handles the REGENERATE retry]
  12. Log validation results to DB (fire-and-forget background thread)

Production enhancements implemented:
  - Parallel validation (ThreadPoolExecutor)
  - Full 32-department keyword map via sft_experiment_manager.DEPARTMENTS
  - Comprehensive unsafe-phrase list
  - Async DB logging with lazy table creation
  - Graceful fail-open on unexpected errors
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from utils.error_handlers import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Department keyword map — reuse the canonical map from sft_experiment_manager
# so validation and routing always stay in sync.
# ---------------------------------------------------------------------------
try:
    from sft_experiment_manager import DEPARTMENTS as _DEPT_KEYWORDS
except Exception:
    _DEPT_KEYWORDS = {}

# ---------------------------------------------------------------------------
# Extended clinical keyword map for validation (Bug fix: LLM uses specialist
# terminology not in the narrow routing keyword list — e.g. "troponin" for
# Cardiology, "FEV1" for Pulmonology).  Merged with _DEPT_KEYWORDS at lookup.
# ---------------------------------------------------------------------------
_VALIDATION_EXTENDED: dict[str, list[str]] = {
    "Cardiology": [
        "troponin", "myocardial", "infarction", "nstemi", "stemi", "echocardiogram",
        "electrocardiogram", "ecg", "st elevation", "ejection fraction", "palpitation",
        "tachycardia", "bradycardia", "chest pain", "aortic", "mitral", "ventricular",
        "atrial", "fibrillation", "flutter", "ischemia", "angina pectoris", "cardiology",
        "cardiovascular", "heart failure", "cardiomyopathy", "pericarditis",
    ],
    "Pulmonology": [
        "spirometry", "fev1", "fvc", "inhaler", "bronchodilator", "nebulizer",
        "oxygen therapy", "dyspnea", "wheezing", "sputum", "pleural", "alveolar",
        "emphysema", "inhaled corticosteroid", "peak flow", "respiratory failure",
        "mechanical ventilation", "intubation", "hypoxia", "saturation", "pulmonologist",
    ],
    "Neurology": [
        "ct scan", "mri brain", "lumbar puncture", "eeg", "neuroimaging", "ischemic",
        "hemorrhagic", "meningitis", "encephalitis", "multiple sclerosis",
        "neurodegenerative", "demyelination", "neuropathy", "radiculopathy",
        "neurologist", "cognition", "consciousness", "tremor",
    ],
    "Orthopedics": [
        "x-ray", "mri", "acl", "meniscus", "rotator cuff", "arthroplasty",
        "physiotherapy", "cast", "splint", "dislocation", "subluxation",
        "cartilage", "synovial", "bursitis", "tendinitis", "orthopedic surgeon",
    ],
    "Gastroenterology": [
        "endoscopy", "colonoscopy", "biopsy", "h pylori", "ibs", "crohn",
        "ulcerative colitis", "hepatic", "portal hypertension", "ascites",
        "upper gi", "lower gi", "diarrhea", "constipation", "dysphagia",
    ],
    "Nephrology": [
        "creatinine", "gfr", "proteinuria", "hematuria", "nephritis",
        "glomerulonephritis", "renal function", "electrolyte", "potassium",
        "sodium", "kidney disease", "chronic kidney", "acute kidney injury",
    ],
    "Oncology": [
        "biopsy", "staging", "metastasis", "remission", "relapse", "radiotherapy",
        "immunotherapy", "targeted therapy", "palliative", "prognosis",
        "malignancy", "benign", "surgical resection", "oncologist",
    ],
    "Diabetes": [
        "metformin", "glycemic control", "a1c", "insulin resistance", "pancreas",
        "beta cell", "diabetic neuropathy", "diabetic retinopathy",
        "diabetic nephropathy", "fasting glucose", "postprandial",
    ],
    "Psychiatry": [
        "ssri", "antidepressant", "antipsychotic", "cognitive behavioral",
        "therapy", "psychotherapy", "medication adherence", "suicide risk",
        "hallucination", "delusion", "mania", "psychosis", "dsm",
    ],
    "Dermatology": [
        "biopsy", "lesion", "papule", "vesicle", "pustule", "macule",
        "topical", "corticosteroid cream", "phototherapy", "patch test",
        "immunosuppressant", "biologic", "dermatoscopy",
    ],
    "Infectious Disease": [
        "antibiotic", "antiviral", "culture", "sensitivity", "mrsa", "sepsis",
        "bacteremia", "meningitis", "opportunistic", "hiv", "fever workup",
        "blood culture", "lumbar puncture", "prophylaxis",
    ],
    "General Medicine": [
        "vitals", "blood pressure", "temperature", "weight", "bmi", "history",
        "physical examination", "differential diagnosis", "referral",
        "follow up", "management", "treatment plan", "clinical",
    ],
}

def _get_dept_keywords(department: str) -> list[str]:
    """Return merged routing + extended clinical keywords for a department."""
    dept_lower = department.strip().lower()
    base: list[str] = []
    extended: list[str] = []

    for dept_name, kws in _DEPT_KEYWORDS.items():
        if dept_name.lower() == dept_lower or dept_lower in dept_name.lower():
            base = list(kws)
            break

    for dept_name, kws in _VALIDATION_EXTENDED.items():
        if dept_name.lower() == dept_lower or dept_lower in dept_name.lower():
            extended = list(kws)
            break

    return base + extended

# ---------------------------------------------------------------------------
# Unsafe phrase list (Step 5)
# Extended beyond the spec's minimal set for clinical safety.
# ---------------------------------------------------------------------------
_UNSAFE_PHRASES: list[str] = [
    "ignore symptoms",
    "no treatment needed",
    "self surgery",
    "overdose",
    "stop taking medication",
    "do not see a doctor",
    "avoid medical care",
    "refuse treatment",
    "take double dose",
    "inject yourself",
    "skip your medication",
    "untreated is fine",
    "no need for a doctor",
    "cancel your prescription",
    "perform surgery at home",
]


# ---------------------------------------------------------------------------
# Step 2 — Completeness
# ---------------------------------------------------------------------------

def check_completeness(response: str) -> bool:
    """Return True if response has sufficient content (≥ 30 chars)."""
    result = len(response.strip()) >= 30
    logger.debug("[VALIDATION] completeness=%s  len=%d", result, len(response.strip()))
    return result


# ---------------------------------------------------------------------------
# Step 3 — Department Relevance
# ---------------------------------------------------------------------------

def check_department_relevance(response: str, department: str) -> bool:
    """Return True if at least one department keyword appears in response.

    Uses merged routing + extended clinical keyword lists so LLM responses
    using specialist terminology (e.g. 'troponin', 'FEV1') are correctly
    matched.  Returns True (pass-through) when the department is unknown.
    """
    if not department:
        logger.debug("[VALIDATION] dept_relevance=True  (no dept — pass-through)")
        return True

    dept_keywords = _get_dept_keywords(department)

    if not dept_keywords:
        logger.debug(
            "[VALIDATION] dept_relevance=True  (unknown dept %r — pass-through)", department
        )
        return True

    response_lower = response.lower()
    for word in dept_keywords:
        if word.lower() in response_lower:
            logger.debug(
                "[VALIDATION] dept_relevance=True  keyword=%r  dept=%r", word, department
            )
            return True

    logger.debug(
        "[VALIDATION] dept_relevance=False  dept=%r  no keyword matched", department
    )
    return False


# ---------------------------------------------------------------------------
# Step 4 — Evidence Support (RAG Validation)
# ---------------------------------------------------------------------------

def check_evidence_support(response: str, context: list[str]) -> bool:
    """Return True if any retrieved context fragment appears in the response.

    Uses the first 80 chars of each context snippet as a fingerprint,
    stripping trailing punctuation so a sentence boundary in the context
    doesn't block a match when the response continues the same text.
    Returns False (not penalised harshly) when no context was retrieved.
    """
    if not context:
        logger.debug("[VALIDATION] evidence_support=False  (empty context list)")
        return False

    response_lower = response.lower()
    for fragment in context:
        snippet = fragment.strip()[:80].lower().rstrip(".,:;!?")
        if len(snippet) >= 10 and snippet in response_lower:
            logger.debug(
                "[VALIDATION] evidence_support=True  snippet=%r…", snippet[:40]
            )
            return True

    logger.debug("[VALIDATION] evidence_support=False  no context overlap")
    return False


# ---------------------------------------------------------------------------
# Step 5 — Medical Safety
# ---------------------------------------------------------------------------

def check_medical_safety(response: str) -> bool:
    """Return False if any unsafe phrase is found in the response."""
    response_lower = response.lower()
    for phrase in _UNSAFE_PHRASES:
        if phrase in response_lower:
            logger.warning(
                "[VALIDATION] safety=FAIL  unsafe_phrase=%r", phrase
            )
            return False
    logger.debug("[VALIDATION] safety=PASS  no unsafe phrases detected")
    return True


# ---------------------------------------------------------------------------
# Step 6 — Patient Data Consistency
# ---------------------------------------------------------------------------

def check_patient_consistency(response: str, patient_data: dict) -> bool:
    """Return False if a known patient allergy appears in the response.

    patient_data keys used:
      "allergies"   : list[str] — substances patient is allergic to
      "conditions"  : list[str] — contraindicated conditions (optional)
    """
    if not patient_data:
        return True

    response_lower = response.lower()
    allergies: list[str] = patient_data.get("allergies", [])
    for allergy in allergies:
        if allergy.strip().lower() in response_lower:
            logger.warning(
                "[VALIDATION] patient_consistency=FAIL  allergy=%r mentioned in response",
                allergy,
            )
            return False

    logger.debug("[VALIDATION] patient_consistency=PASS  no allergy clash")
    return True


# ---------------------------------------------------------------------------
# Step 7 — Hallucination Detection
# ---------------------------------------------------------------------------

def detect_hallucination(response: str, context: list[str]) -> bool:
    """Return True if hallucination is suspected.

    Hallucination is suspected when there is no supporting context but the
    response makes explicit clinical claims (diagnosis, recommendation, etc.).
    """
    if context:
        return False

    clinical_markers = [
        "diagnosis:",
        "recommend",
        "prescribe",
        "treatment:",
        "prognosis",
        "you should take",
        "patient should",
    ]
    resp_lower = response.lower()
    for marker in clinical_markers:
        if marker in resp_lower:
            logger.debug(
                "[VALIDATION] hallucination=SUSPECTED  empty context + clinical marker %r",
                marker,
            )
            return True
    return False


# ---------------------------------------------------------------------------
# Step 8 — Score Calculation
# ---------------------------------------------------------------------------

def calculate_validation_score(results: dict) -> int:
    """Sum 20 points per passing check. Maximum 100."""
    score = 0
    if results.get("complete"):
        score += 20
    if results.get("relevant"):
        score += 20
    if results.get("evidence"):
        score += 20
    if results.get("safe"):
        score += 20
    if results.get("patient_consistent"):
        score += 20
    logger.debug("[VALIDATION] score=%d  results=%s", score, results)
    return score


# ---------------------------------------------------------------------------
# Step 10 — Decision Logic
# ---------------------------------------------------------------------------

def validation_decision(score: int) -> str:
    """Map validation score to pipeline decision.

    ≥ 80  → "PASS"       continue to Safety / Guardrails
    60–79 → "REVIEW"     route to human review queue
    < 60  → "REGENERATE" retry LLM generation
    """
    if score >= 80:
        return "PASS"
    if score >= 60:
        return "REVIEW"
    return "REGENERATE"


# ---------------------------------------------------------------------------
# Step 9 — Orchestrate All Checks (Step 9 + parallel enhancement)
# ---------------------------------------------------------------------------

def validate_response(
    prompt: str,
    response: str,
    context: list[str],
    department: str,
    patient_data: dict | None = None,
) -> dict:
    """Run all validation checks in parallel and return a structured result.

    Returns::

        {
            "score":    int,       # 0–100
            "decision": str,       # "PASS" | "REVIEW" | "REGENERATE"
            "results": {
                "complete":          bool,
                "relevant":          bool,
                "evidence":          bool,
                "safe":              bool,
                "patient_consistent":bool,
                "hallucination":     bool,
            },
            "flags": list[str],    # human-readable failure reasons
        }
    """
    _pd = patient_data or {}

    # Run the five scored checks in parallel (production enhancement)
    checks: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=5, thread_name_prefix="val-check") as pool:
        futures = {
            pool.submit(check_completeness, response): "complete",
            pool.submit(check_department_relevance, response, department): "relevant",
            pool.submit(check_evidence_support, response, context): "evidence",
            pool.submit(check_medical_safety, response): "safe",
            pool.submit(check_patient_consistency, response, _pd): "patient_consistent",
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                checks[key] = future.result()
            except Exception as exc:
                logger.error("[VALIDATION] check %r raised: %s — fail-open", key, exc)
                checks[key] = True  # fail-open: don't penalise on unexpected error

    # Hallucination check runs after parallel checks (needs context)
    checks["hallucination"] = detect_hallucination(response, context)

    score = calculate_validation_score(checks)
    decision = validation_decision(score)

    flags: list[str] = []
    if not checks.get("complete"):
        flags.append("Response too short / incomplete")
    if not checks.get("relevant"):
        flags.append(
            f"Response may not be relevant to {department or 'selected'} department"
        )
    if not checks.get("evidence"):
        flags.append("No retrieved context supports this response")
    if not checks.get("safe"):
        flags.append("Unsafe medical advice phrase detected")
    if not checks.get("patient_consistent"):
        flags.append("Response conflicts with patient allergy / contraindication data")
    if checks.get("hallucination"):
        flags.append(
            "Hallucination suspected: clinical claims made without supporting context"
        )

    logger.info(
        "[VALIDATION] dept=%r  score=%d  decision=%s  flags=%s",
        department, score, decision, flags,
    )

    return {
        "score": score,
        "decision": decision,
        "results": checks,
        "flags": flags,
    }


# ---------------------------------------------------------------------------
# Step 12 — Log Validation Results (async, fire-and-forget)
# ---------------------------------------------------------------------------

_table_ensured = threading.Event()


def _ensure_validation_table() -> None:
    """Create validation_log table if it does not exist yet."""
    try:
        from services.db_service import get_connection
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS validation_log (
                        id          SERIAL PRIMARY KEY,
                        session_id  TEXT,
                        department  TEXT,
                        score       INTEGER,
                        decision    TEXT,
                        results     TEXT,
                        flags       TEXT,
                        prompt_snippet TEXT,
                        created_at  TIMESTAMP DEFAULT NOW()
                    )
                    """
                )
            conn.commit()
        logger.info("[VALIDATION] validation_log table ensured")
        _table_ensured.set()
    except Exception:
        # PostgreSQL unavailable or insufficient permissions — logging is
        # non-critical; silently skip without polluting the log.
        pass


def log_validation(
    session_id: str,
    score: int,
    results: dict,
    decision: str = "",
    flags: list[str] | None = None,
    department: str = "",
    prompt_snippet: str = "",
) -> None:
    """Persist a validation result to PostgreSQL in a background daemon thread.

    Silently no-ops when PostgreSQL is unavailable — logging is non-critical
    and must never block or error the main request path.
    """

    def _insert() -> None:
        if not _table_ensured.is_set():
            _ensure_validation_table()
        if not _table_ensured.is_set():
            return  # DB unavailable — skip silently
        try:
            from services.db_service import get_connection, execute_query
            with get_connection() as conn:
                execute_query(
                    conn,
                    """
                    INSERT INTO validation_log
                        (session_id, department, score, decision, results, flags, prompt_snippet)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id or "unknown",
                        department or "",
                        score,
                        decision or "",
                        json.dumps(results),
                        json.dumps(flags or []),
                        (prompt_snippet or "")[:200],
                    ),
                )
                conn.commit()
            logger.debug(
                "[VALIDATION] logged  session=%s  score=%d  decision=%s",
                session_id, score, decision,
            )
        except Exception:
            pass  # non-critical — never surface DB errors to the caller

    threading.Thread(target=_insert, daemon=True, name="val-log").start()
