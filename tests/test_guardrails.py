"""
Unit tests for services/guardrails_service.py

Tests all 7 guardrail checks + orchestrator (run_guardrails).
No Flask required — all tests run directly.

Run:  python tests/test_guardrails.py
"""

import sys
import os
import unittest

# Make project root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.guardrails_service import (
    clinical_safety_check,
    medication_safety_check,
    emergency_detection,
    get_emergency_response,
    compliance_check,
    privacy_check,
    scope_check,
    confidence_guardrail,
    run_guardrails,
)


class TestClinicalSafetyCheck(unittest.TestCase):

    def test_safe_response_passes(self):
        response = "The patient should consult an ophthalmologist for this condition."
        passed, flags = clinical_safety_check(response)
        self.assertTrue(passed)
        self.assertEqual(flags, [])

    def test_overdose_phrase_fails(self):
        response = "An overdose of timolol can cause systemic toxicity."
        passed, flags = clinical_safety_check(response)
        self.assertFalse(passed)
        self.assertIn("overdose", flags)

    def test_stop_medication_fails(self):
        response = "The patient should stop taking medication immediately."
        passed, flags = clinical_safety_check(response)
        self.assertFalse(passed)
        self.assertIn("stop taking medication", flags)

    def test_no_doctor_fails(self):
        response = "You can do not see a doctor and treat yourself."
        passed, flags = clinical_safety_check(response)
        self.assertFalse(passed)

    def test_clinical_advice_passes(self):
        response = (
            "Glaucoma management includes prostaglandin analogues such as latanoprost. "
            "Regular IOP monitoring is recommended every 3 months."
        )
        passed, flags = clinical_safety_check(response)
        self.assertTrue(passed)


class TestMedicationSafetyCheck(unittest.TestCase):

    def test_no_patient_data_passes(self):
        passed, clashes = medication_safety_check("Some response", {})
        self.assertTrue(passed)
        self.assertEqual(clashes, [])

    def test_allergen_in_response_fails(self):
        response = "Timolol eye drops are commonly prescribed for glaucoma."
        patient_data = {"allergies": ["timolol", "latanoprost"]}
        passed, clashes = medication_safety_check(response, patient_data)
        self.assertFalse(passed)
        self.assertTrue(any("timolol" in c for c in clashes))

    def test_safe_response_with_patient_data(self):
        response = "Bimatoprost (Lumigan) is an effective alternative for IOP reduction."
        patient_data = {"allergies": ["timolol", "latanoprost"]}
        passed, clashes = medication_safety_check(response, patient_data)
        self.assertTrue(passed)

    def test_condition_conflict(self):
        response = "Metformin is used for blood sugar control in diabetic patients."
        patient_data = {"allergies": [], "conditions": ["metformin"]}
        passed, clashes = medication_safety_check(response, patient_data)
        self.assertFalse(passed)


class TestEmergencyDetection(unittest.TestCase):

    def test_emergency_prompt_detected(self):
        prompt = "My patient has chest pain and cannot breathe."
        is_emergency, keywords = emergency_detection(prompt, "")
        self.assertTrue(is_emergency)
        self.assertTrue(any(kw in keywords for kw in ["chest pain", "cannot breathe"]))

    def test_cardiac_arrest_detected(self):
        prompt = "Patient collapsed — cardiac arrest suspected."
        is_emergency, keywords = emergency_detection(prompt, "")
        self.assertTrue(is_emergency)

    def test_normal_ophthalmology_query_not_emergency(self):
        prompt = "What is the standard treatment for open-angle glaucoma?"
        is_emergency, _ = emergency_detection(prompt, "")
        self.assertFalse(is_emergency)

    def test_emergency_in_response(self):
        response = "If the patient shows signs of stroke, call 911 immediately."
        is_emergency, keywords = emergency_detection("", response)
        self.assertTrue(is_emergency)

    def test_emergency_response_is_non_empty(self):
        msg = get_emergency_response()
        self.assertIn("EMERGENCY", msg)
        self.assertIn("911", msg)


class TestComplianceCheck(unittest.TestCase):

    def test_clean_response_passes(self):
        response = "Please refer to your specialist for further evaluation."
        passed, flags = compliance_check(response)
        self.assertTrue(passed)

    def test_guaranteed_cure_fails(self):
        response = "This treatment is a guaranteed cure for glaucoma."
        passed, flags = compliance_check(response)
        self.assertFalse(passed)
        self.assertIn("guaranteed cure", flags)

    def test_no_need_to_consult_fails(self):
        response = "There is no need to consult a doctor for this condition."
        passed, flags = compliance_check(response)
        self.assertFalse(passed)

    def test_miracle_cure_fails(self):
        response = "This is a miracle cure discovered recently."
        passed, flags = compliance_check(response)
        self.assertFalse(passed)


class TestPrivacyCheck(unittest.TestCase):

    def test_clean_response_passes(self):
        response = "The patient should be monitored regularly for IOP changes."
        passed, phi_types = privacy_check(response)
        self.assertTrue(passed)

    def test_ssn_in_response_fails(self):
        response = "Patient SSN is 123-45-6789 according to records."
        passed, phi_types = privacy_check(response)
        self.assertFalse(passed)
        self.assertIn("SSN", phi_types)

    def test_email_in_response_fails(self):
        response = "Contact the patient at john.doe@hospital.com for follow-up."
        passed, phi_types = privacy_check(response)
        self.assertFalse(passed)
        self.assertIn("Email", phi_types)

    def test_phone_in_response_fails(self):
        response = "Call the patient at 555-867-5309 to confirm appointment."
        passed, phi_types = privacy_check(response)
        self.assertFalse(passed)
        self.assertIn("Phone", phi_types)

    def test_tax_id_fails(self):
        response = "Patient tax ID: 98-7654321 on file."
        passed, phi_types = privacy_check(response)
        self.assertFalse(passed)
        self.assertIn("TaxID", phi_types)


class TestScopeCheck(unittest.TestCase):

    def test_ophthalmology_response_passes_for_ophthalmology(self):
        response = "Phacoemulsification is the gold standard for cataract surgery."
        passed, violations = scope_check(response, "Ophthalmology")
        self.assertTrue(passed)

    def test_cardiac_term_in_ophthalmology_response_fails(self):
        response = "Coronary artery bypass surgery may affect eye pressure."
        passed, violations = scope_check(response, "Ophthalmology")
        self.assertFalse(passed)
        self.assertTrue(any("coronary artery bypass" in v for v in violations))

    def test_no_department_passes(self):
        passed, violations = scope_check("Any response", "")
        self.assertTrue(passed)

    def test_ortho_term_in_cardiology_response_fails(self):
        response = "Arthroplasty outcomes depend on cardiovascular fitness."
        passed, violations = scope_check(response, "Cardiology")
        self.assertFalse(passed)


class TestConfidenceGuardrail(unittest.TestCase):

    def test_high_score_passes(self):
        passed, reason = confidence_guardrail(80)
        self.assertTrue(passed)
        self.assertEqual(reason, "")

    def test_exactly_60_passes(self):
        passed, reason = confidence_guardrail(60)
        self.assertTrue(passed)

    def test_59_fails(self):
        passed, reason = confidence_guardrail(59)
        self.assertFalse(passed)
        self.assertIn("59", reason)

    def test_zero_fails(self):
        passed, reason = confidence_guardrail(0)
        self.assertFalse(passed)


class TestRunGuardrails(unittest.TestCase):
    """Integration tests for run_guardrails orchestrator."""

    def test_safe_ophthalmology_query(self):
        result = run_guardrails(
            prompt="What is the first-line treatment for open-angle glaucoma?",
            response=(
                "Prostaglandin analogues such as bimatoprost and travoprost are first-line "
                "treatments for open-angle glaucoma. They reduce intraocular pressure by "
                "increasing aqueous humour outflow. Regular IOP monitoring is recommended."
            ),
            department="Ophthalmology",
            patient_data={"allergies": []},
            validation_score=80,
        )
        self.assertEqual(result["status"], "SAFE")
        self.assertIn("results", result)
        self.assertIn("flags", result)
        self.assertIsNone(result["emergency_response"])

    def test_emergency_detected_overrides(self):
        result = run_guardrails(
            prompt="Patient has chest pain and cannot breathe.",
            response="This may indicate a cardiac event. Seek immediate care.",
            department="Cardiology",
            patient_data={},
            validation_score=80,
        )
        self.assertEqual(result["status"], "EMERGENCY")
        self.assertIsNotNone(result["emergency_response"])
        self.assertIn("911", result["emergency_response"])

    def test_blocked_by_low_validation_score(self):
        result = run_guardrails(
            prompt="Routine glaucoma query",
            response="Treatment options include several eye drops.",
            department="Ophthalmology",
            patient_data={},
            validation_score=40,  # < 60 → confidence_guardrail fails
        )
        self.assertEqual(result["status"], "BLOCKED")

    def test_blocked_by_phi_in_response(self):
        result = run_guardrails(
            prompt="What should I prescribe?",
            response="Contact patient at 555-123-4567 or john@clinic.com for follow-up.",
            department="Ophthalmology",
            patient_data={},
            validation_score=80,
        )
        self.assertEqual(result["status"], "BLOCKED")
        self.assertTrue(any("PHI" in f or "Email" in f or "Phone" in f for f in result["flags"]))

    def test_blocked_by_guaranteed_cure(self):
        result = run_guardrails(
            prompt="Is there a cure?",
            response="This is a guaranteed cure for all types of glaucoma.",
            department="Ophthalmology",
            patient_data={},
            validation_score=80,
        )
        self.assertEqual(result["status"], "BLOCKED")

    def test_allergen_conflict_blocks(self):
        result = run_guardrails(
            prompt="What eye drops should I prescribe?",
            response="Timolol eye drops are highly effective for reducing IOP.",
            department="Ophthalmology",
            patient_data={"allergies": ["timolol"]},
            validation_score=80,
        )
        self.assertEqual(result["status"], "BLOCKED")

    def test_result_has_all_required_keys(self):
        result = run_guardrails(
            prompt="Query",
            response="Response",
            department="General Medicine",
            patient_data={},
            validation_score=100,
        )
        self.assertIn("status", result)
        self.assertIn("results", result)
        self.assertIn("flags", result)
        self.assertIn("emergency_response", result)
        self.assertIn("clinical_safe", result["results"])
        self.assertIn("medication_safe", result["results"])
        self.assertIn("emergency", result["results"])
        self.assertIn("compliant", result["results"])
        self.assertIn("privacy_safe", result["results"])
        self.assertIn("in_scope", result["results"])
        self.assertIn("confidence_ok", result["results"])


if __name__ == "__main__":
    print("=" * 60)
    print("Guardrails Service — Unit Tests")
    print("=" * 60)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [
        TestClinicalSafetyCheck,
        TestMedicationSafetyCheck,
        TestEmergencyDetection,
        TestComplianceCheck,
        TestPrivacyCheck,
        TestScopeCheck,
        TestConfidenceGuardrail,
        TestRunGuardrails,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print(f"\n{'='*60}")
    print(f"Results: {result.testsRun} tests, "
          f"{len(result.failures)} failures, "
          f"{len(result.errors)} errors")
    sys.exit(0 if result.wasSuccessful() else 1)
