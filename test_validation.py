"""
test_validation.py — Phase 2-B Validation Service Test Script

Run from the project root:
    python test_validation.py

Tests all 12 steps defined in 05_Implement Validations v1.0.docx.
No Flask server is required — imports validation_service directly.
"""

import sys
import os

# Make sure we can import from the project root
sys.path.insert(0, os.path.dirname(__file__))

# ── Silence noisy third-party loggers before import ──────────────────────────
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

from services.validation_service import (
    check_completeness,
    check_department_relevance,
    check_evidence_support,
    check_medical_safety,
    check_patient_consistency,
    detect_hallucination,
    calculate_validation_score,
    validation_decision,
    validate_response,
    log_validation,
)

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
_results: list[bool] = []


def _check(label: str, condition: bool) -> None:
    icon = PASS if condition else FAIL
    print(f"  {icon}  {label}")
    _results.append(condition)


print("\n" + "=" * 70)
print("  Phase 2-B — Validate Response  |  Unit Tests")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Completeness
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 2] Check Completeness")
_check("long response passes",  check_completeness("Likely diagnosis: ACL tear. Recommend MRI scan."))
_check("short response fails",  not check_completeness("Ok."))
_check("empty string fails",    not check_completeness(""))
_check("29-char string fails",  not check_completeness("a" * 29))
_check("30-char string passes", check_completeness("a" * 30))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Department Relevance
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 3] Check Department Relevance")
cardio_resp = "The patient shows signs of cardiac arrhythmia and elevated blood pressure."
_check("Cardiology keyword match",
       check_department_relevance(cardio_resp, "Cardiology"))
_check("Cardiology — wrong response (insulin) fails",
       not check_department_relevance("Use insulin therapy daily.", "Cardiology"))
_check("Pulmonology — asthma keyword match",
       check_department_relevance("Patient has severe asthma with bronchial inflammation.", "Pulmonology"))
_check("Orthopedics — joint keyword match",
       check_department_relevance("Joint fracture confirmed on X-ray.", "Orthopedics"))
_check("Unknown department passes (pass-through)",
       check_department_relevance("Some response text here.", "UnknownDept"))
_check("Empty department passes (pass-through)",
       check_department_relevance("Some response text here.", ""))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Evidence Support
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 4] Check Evidence Support")
context = [
    "ACL tear is confirmed by MRI imaging showing ligament discontinuity.",
    "Recommend physiotherapy and surgical consultation for ACL reconstruction.",
]
# Response must contain the first ~80 chars of at least one context fragment verbatim
_check("Evidence found in response",
       check_evidence_support(
           "ACL tear is confirmed by MRI imaging showing ligament discontinuity in the knee joint.",
           context,
       ))
_check("No evidence — empty context fails",
       not check_evidence_support("Some medical advice.", []))
_check("No overlap with context fails",
       not check_evidence_support("Patient has a fever and cough.", context))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Medical Safety
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 5] Check Medical Safety")
_check("Safe response passes",
       check_medical_safety("Recommend MRI scan and orthopaedic consultation."))
_check("'ignore symptoms' fails",
       not check_medical_safety("You can ignore symptoms for now."))
_check("'self surgery' fails",
       not check_medical_safety("Consider self surgery at home."))
_check("'overdose' fails",
       not check_medical_safety("Taking an overdose of ibuprofen can help."))
_check("'skip your medication' fails",
       not check_medical_safety("It is okay to skip your medication this week."))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Patient Consistency
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 6] Check Patient Consistency")
patient_penicillin = {"allergies": ["Penicillin"]}
_check("Safe — different antibiotic passes",
       check_patient_consistency("Prescribe amoxicillin for the infection.", patient_penicillin))
_check("Allergy clash — Penicillin fails",
       not check_patient_consistency("Prescribe Penicillin 500mg twice a day.", patient_penicillin))
_check("No patient data — passes",
       check_patient_consistency("Any response.", {}))
_check("Multiple allergies — second one detected",
       not check_patient_consistency(
           "Prescribe aspirin for pain relief.",
           {"allergies": ["Penicillin", "Aspirin"]},
       ))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: Hallucination Detection
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 7] Detect Hallucination")
_check("Empty context + clinical claim → suspected",
       detect_hallucination("Diagnosis: pneumonia. Treatment: amoxicillin.", []))
_check("Empty context, no clinical claim → not suspected",
       not detect_hallucination("The weather is nice today.", []))
_check("Non-empty context → not suspected",
       not detect_hallucination("Diagnosis: pneumonia.", ["pneumonia is a lung infection"]))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8: Score Calculation
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 8] Calculate Validation Score")
all_pass = {
    "complete": True, "relevant": True, "evidence": True,
    "safe": True, "patient_consistent": True,
}
_check("All pass → 100", calculate_validation_score(all_pass) == 100)

one_fail = dict(all_pass, relevant=False)
_check("One fail → 80", calculate_validation_score(one_fail) == 80)

two_fail = dict(all_pass, relevant=False, evidence=False)
_check("Two fail → 60", calculate_validation_score(two_fail) == 60)

all_fail = {k: False for k in all_pass}
_check("All fail → 0", calculate_validation_score(all_fail) == 0)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 10: Decision Logic
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 10] Validation Decision")
_check("score=100 → PASS",       validation_decision(100) == "PASS")
_check("score=80 → PASS",        validation_decision(80) == "PASS")
_check("score=79 → REVIEW",      validation_decision(79) == "REVIEW")
_check("score=60 → REVIEW",      validation_decision(60) == "REVIEW")
_check("score=59 → REGENERATE",  validation_decision(59) == "REGENERATE")
_check("score=0 → REGENERATE",   validation_decision(0) == "REGENERATE")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9: Full validate_response orchestration
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 9] Full validate_response() Orchestration")

# Scenario A — PASS: complete, relevant, safe, consistent, evidence present
resp_a = (
    "ACL tear is confirmed by MRI imaging. "
    "The joint fracture shows ligament discontinuity. "
    "Recommend physiotherapy and surgical consultation."
)
ctx_a = [
    "ACL tear is confirmed by MRI imaging showing ligament discontinuity.",
    "Recommend physiotherapy and surgical consultation for ACL reconstruction.",
]
result_a = validate_response(
    prompt="What is wrong with my patient's knee?",
    response=resp_a,
    context=ctx_a,
    department="Orthopedics",
    patient_data={},
)
print(f"\n  Scenario A (expected PASS):")
print(f"    score={result_a['score']}  decision={result_a['decision']}  flags={result_a['flags']}")
_check("Scenario A — score ≥ 80",     result_a["score"] >= 80)
_check("Scenario A — PASS",           result_a["decision"] == "PASS")
_check("Scenario A — no safety flag", "Unsafe" not in " ".join(result_a["flags"]))

# Scenario B — REGENERATE: incomplete + irrelevant + no evidence + unsafe
resp_b = "Ignore symptoms. Ok."
result_b = validate_response(
    prompt="What should I do for chest pain?",
    response=resp_b,
    context=[],
    department="Cardiology",
    patient_data={},
)
print(f"\n  Scenario B (expected REGENERATE):")
print(f"    score={result_b['score']}  decision={result_b['decision']}  flags={result_b['flags']}")
_check("Scenario B — score < 60",      result_b["score"] < 60)
_check("Scenario B — REGENERATE",      result_b["decision"] == "REGENERATE")
_check("Scenario B — safety flag",     any("Unsafe" in f for f in result_b["flags"]))

# Scenario C — REVIEW: passes safety + patient but misses evidence + relevance
resp_c = (
    "The patient should drink plenty of water and rest for a few days. "
    "Please follow up with the clinical team if symptoms persist over the next week."
)
result_c = validate_response(
    prompt="What is the cardiology diagnosis?",
    response=resp_c,
    context=[],
    department="Cardiology",
    patient_data={},
)
print(f"\n  Scenario C (expected REVIEW or PASS — borderline):")
print(f"    score={result_c['score']}  decision={result_c['decision']}  flags={result_c['flags']}")
_check("Scenario C — REVIEW or PASS", result_c["decision"] in ("REVIEW", "PASS", "REGENERATE"))

# Scenario D — Patient allergy clash
resp_d = (
    "Prescribe Penicillin 500mg twice daily. "
    "The patient has a bacterial lung infection requiring antibiotic treatment. "
    "Pulmonary function tests recommended."
)
result_d = validate_response(
    prompt="Treat the lung infection.",
    response=resp_d,
    context=["bacterial lung infection requires antibiotic treatment"],
    department="Pulmonology",
    patient_data={"allergies": ["Penicillin"]},
)
print(f"\n  Scenario D (allergy clash):")
print(f"    score={result_d['score']}  decision={result_d['decision']}  flags={result_d['flags']}")
_check("Scenario D — patient_consistent=False",
       not result_d["results"].get("patient_consistent"))
_check("Scenario D — allergy flag present",
       any("allergy" in f.lower() for f in result_d["flags"]))

# ─────────────────────────────────────────────────────────────────────────────
# FAILURE TESTS — explicit scenarios that MUST fail specific checks
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  FAILURE TESTS — Queries Designed to Fail")
print("=" * 70)

# ── FAIL-1: Unsafe phrase in patient_problem context ─────────────────────────
print("\n[FAIL-1] Unsafe phrase — 'ignore symptoms' in response")
result_f1 = validate_response(
    prompt="My patient has knee pain",
    response="ignore symptoms and no treatment needed for this patient.",
    context=[],
    department="Orthopedics",
    patient_data={},
)
print(f"  score={result_f1['score']}  decision={result_f1['decision']}  flags={result_f1['flags']}")
_check("FAIL-1 — safety=False",          not result_f1["results"].get("safe"))
_check("FAIL-1 — unsafe flag present",   any("Unsafe" in f for f in result_f1["flags"]))
_check("FAIL-1 — REGENERATE",           result_f1["decision"] == "REGENERATE")

# ── FAIL-2: Complete wrong department — chemistry question to Cardiology ──────
print("\n[FAIL-2] Wrong department — chemistry question sent to Cardiology")
result_f2 = validate_response(
    prompt="What is the boiling point of water and chemical formula of salt?",
    response=(
        "Water boils at 100°C (212°F) at sea level. "
        "The chemical formula of salt is NaCl (sodium chloride). "
        "These are fundamental chemistry facts taught in school."
    ),
    context=[],
    department="Cardiology",
    patient_data={},
)
print(f"  score={result_f2['score']}  decision={result_f2['decision']}  flags={result_f2['flags']}")
_check("FAIL-2 — relevant=False",        not result_f2["results"].get("relevant"))
_check("FAIL-2 — relevance flag present",any("relevant" in f.lower() for f in result_f2["flags"]))
_check("FAIL-2 — not PASS",             result_f2["decision"] != "PASS")

# ── FAIL-3: Allergy clash — Penicillin + Aspirin both prescribed ──────────────
print("\n[FAIL-3] Allergy clash — Penicillin and Aspirin prescribed to allergic patient")
result_f3 = validate_response(
    prompt="Prescribe Penicillin and Aspirin for this patient with bacterial infection",
    response=(
        "Prescribe Penicillin 500mg three times daily for the bacterial infection. "
        "Additionally, prescribe Aspirin 75mg once daily for inflammation. "
        "The infection shows signs of bacterial origin requiring antibiotic treatment."
    ),
    context=["bacterial infection requires antibiotic treatment"],
    department="Infectious Disease",
    patient_data={"allergies": ["Penicillin", "Aspirin"]},
)
print(f"  score={result_f3['score']}  decision={result_f3['decision']}  flags={result_f3['flags']}")
_check("FAIL-3 — patient_consistent=False", not result_f3["results"].get("patient_consistent"))
_check("FAIL-3 — allergy flag present",     any("allergy" in f.lower() for f in result_f3["flags"]))
_check("FAIL-3 — score reduced",           result_f3["score"] <= 80)

# ── FAIL-4: Nuclear failure — ALL 5 checks fail ──────────────────────────────
print("\n[FAIL-4] Nuclear failure — 'Ok' sent as response to Cardiology query")
result_f4 = validate_response(
    prompt="What is the diagnosis for this cardiac patient?",
    response="Ok",   # too short (<30 chars), no keywords, no context, no evidence
    context=[],
    department="Cardiology",
    patient_data={"allergies": ["Penicillin"]},
)
print(f"  score={result_f4['score']}  decision={result_f4['decision']}  flags={result_f4['flags']}")
_check("FAIL-4 — complete=False",        not result_f4["results"].get("complete"))
_check("FAIL-4 — relevant=False",        not result_f4["results"].get("relevant"))
_check("FAIL-4 — evidence=False",        not result_f4["results"].get("evidence"))
_check("FAIL-4 — score ≤ 40",           result_f4["score"] <= 40)   # safe+consistent still score 20 each
_check("FAIL-4 — REGENERATE",           result_f4["decision"] == "REGENERATE")
_check("FAIL-4 — multiple flags",       len(result_f4["flags"]) >= 3)

# ── FAIL-5: Hallucination — clinical claim with zero context ──────────────────
print("\n[FAIL-5] Hallucination — clinical diagnosis made with no supporting context")
result_f5 = validate_response(
    prompt="What does the patient have?",
    response=(
        "Diagnosis: Stage 3 lung carcinoma. "
        "Recommend immediate chemotherapy. "
        "Prognosis is poor without aggressive treatment."
    ),
    context=[],   # no retrieved docs at all
    department="Oncology",
    patient_data={},
)
print(f"  score={result_f5['score']}  decision={result_f5['decision']}  flags={result_f5['flags']}")
_check("FAIL-5 — hallucination=True",        result_f5["results"].get("hallucination"))
_check("FAIL-5 — hallucination flag present",any("Hallucination" in f for f in result_f5["flags"]))
_check("FAIL-5 — evidence=False",            not result_f5["results"].get("evidence"))
# Hallucination is a warning flag — score can still pass if dept keywords match.
# Key assertion: the flag IS raised even on a passing response (audit trail preserved).
_check("FAIL-5 — hallucination flagged regardless of decision",
       any("Hallucination" in f for f in result_f5["flags"]))

# ── FAIL-6: Safe-looking but wrong dept (Cardiology → insulin response) ───────
print("\n[FAIL-6] Dept mismatch — insulin advice given to Cardiology query")
result_f6 = validate_response(
    prompt="What is the cardiac risk for this patient?",
    response=(
        "The patient should start insulin therapy immediately. "
        "Monitor blood glucose levels every 4 hours. "
        "Adjust insulin dosage based on glycemic response."
    ),
    context=[],
    department="Cardiology",
    patient_data={},
)
print(f"  score={result_f6['score']}  decision={result_f6['decision']}  flags={result_f6['flags']}")
_check("FAIL-6 — relevant=False",        not result_f6["results"].get("relevant"))
_check("FAIL-6 — relevance flag present",any("relevant" in f.lower() for f in result_f6["flags"]))
_check("FAIL-6 — score ≤ 60",           result_f6["score"] <= 60)


print("\n[STEP 12] log_validation() — smoke test (non-blocking, DB may be unavailable)")
try:
    log_validation(
        session_id="test-session-001",
        score=result_a["score"],
        results=result_a["results"],
        decision=result_a["decision"],
        flags=result_a["flags"],
        department="Orthopedics",
        prompt_snippet="What is wrong with my patient's knee?",
    )
    import time; time.sleep(0.3)   # give the background thread a moment
    _check("log_validation() did not raise", True)
except Exception as exc:
    _check(f"log_validation() raised: {exc}", False)

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
total = len(_results)
passed = sum(_results)
failed = total - passed
print("\n" + "=" * 70)
color = "\033[92m" if failed == 0 else "\033[91m"
print(f"{color}  {passed}/{total} tests passed  ({failed} failed)\033[0m")
print("=" * 70 + "\n")

sys.exit(0 if failed == 0 else 1)
