"""
Unit tests for services/interaction_log_service.py

Run:
    source myenv/bin/activate && python tests/test_interaction_log.py

Coverage:
  - PII masking (SSN, email, phone, DOB, credit-card)
  - Token extraction from usage object (actual + estimated fallback)
  - create_interaction_log: schema completeness, field truncation, source list
  - create_interaction_log: LoRA attribution, error fields, environment
  - log_interaction: async fire (no DB required — silent fail)
"""

from __future__ import annotations

import json
import sys
import os
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.interaction_log_service import (
    _mask_pii,
    _estimate_tokens,
    _extract_token_usage,
    create_interaction_log,
    log_interaction,
)


class TestMaskPII(unittest.TestCase):

    def test_ssn_masked(self):
        result = _mask_pii("Patient SSN: 123-45-6789 is on file.")
        self.assertIn("[SSN]", result)
        self.assertNotIn("123-45-6789", result)

    def test_ssn_space_separated(self):
        result = _mask_pii("SSN 123 45 6789.")
        self.assertIn("[SSN]", result)

    def test_email_masked(self):
        result = _mask_pii("Contact patient@hospital.org for follow-up.")
        self.assertIn("[EMAIL]", result)
        self.assertNotIn("patient@hospital.org", result)

    def test_phone_masked(self):
        result = _mask_pii("Call 555-867-5309 now.")
        self.assertIn("[PHONE]", result)
        self.assertNotIn("555-867-5309", result)

    def test_phone_parentheses(self):
        result = _mask_pii("Call (555) 867-5309 now.")
        self.assertIn("[PHONE]", result)

    def test_dob_masked(self):
        result = _mask_pii("DOB: 01/15/1985 recorded.")
        self.assertIn("[DOB]", result)
        self.assertNotIn("01/15/1985", result)

    def test_credit_card_masked(self):
        result = _mask_pii("Card: 4111 1111 1111 1111.")
        self.assertIn("[CARD]", result)

    def test_no_pii_unchanged(self):
        text = "Patient presents with chest pain and shortness of breath."
        self.assertEqual(_mask_pii(text), text)

    def test_empty_string(self):
        self.assertEqual(_mask_pii(""), "")

    def test_multiple_pii_types(self):
        text = "Email: doc@clinic.com, Phone: 555-123-4567, SSN: 111-22-3333"
        result = _mask_pii(text)
        self.assertIn("[EMAIL]", result)
        self.assertIn("[PHONE]", result)
        self.assertIn("[SSN]", result)


class TestEstimateTokens(unittest.TestCase):

    def test_empty_string(self):
        self.assertEqual(_estimate_tokens(""), 1)  # minimum 1

    def test_short_text(self):
        # 40 chars → 10 tokens
        self.assertEqual(_estimate_tokens("a" * 40), 10)

    def test_long_text(self):
        self.assertEqual(_estimate_tokens("a" * 400), 100)


class TestExtractTokenUsage(unittest.TestCase):

    def test_real_usage_object_prompt_tokens(self):
        usage = MagicMock()
        usage.prompt_tokens = 120
        usage.completion_tokens = 80
        usage.total_tokens = 200
        usage.input_tokens = None
        usage.output_tokens = None
        result = _extract_token_usage(usage, "prompt", "response")
        self.assertEqual(result["tokens_input"], 120)
        self.assertEqual(result["tokens_output"], 80)
        self.assertEqual(result["tokens_total"], 200)

    def test_real_usage_object_input_tokens(self):
        usage = MagicMock()
        usage.input_tokens = 150
        usage.output_tokens = 50
        usage.total_tokens = 200
        usage.prompt_tokens = 0
        usage.completion_tokens = 0
        result = _extract_token_usage(usage, "p", "r")
        self.assertEqual(result["tokens_input"], 150)
        self.assertEqual(result["tokens_output"], 50)

    def test_none_usage_falls_back_to_estimate(self):
        prompt = "a" * 400   # 100 tokens
        response = "b" * 200  # 50 tokens
        result = _extract_token_usage(None, prompt, response)
        self.assertEqual(result["tokens_input"], 100)
        self.assertEqual(result["tokens_output"], 50)
        self.assertEqual(result["tokens_total"], 150)

    def test_usage_zero_falls_back(self):
        usage = MagicMock()
        usage.input_tokens = 0
        usage.output_tokens = 0
        usage.total_tokens = 0
        usage.prompt_tokens = 0
        usage.completion_tokens = 0
        result = _extract_token_usage(usage, "a" * 100, "b" * 100)
        # Zero counts are falsy → falls through to estimation
        self.assertGreater(result["tokens_total"], 0)


class TestCreateInteractionLog(unittest.TestCase):

    def _build(self, **kwargs) -> dict:
        defaults = dict(
            session_id="sess-001",
            doctor_id="doc-42",
            patient_id="pat-99",
            department="Cardiology",
            prompt="What is the treatment for hypertension?",
            original_response="Prescribe a thiazide diuretic.",
            final_response="Prescribe a thiazide diuretic.",
            validation_result={"score": 80, "decision": "PASS", "flags": []},
            guardrail_result={"status": "SAFE", "flags": [], "results": {}},
            confidence_result={"score": 72, "decision": "REVIEW", "breakdown": {}},
            citation_result={"structured_citations": [], "source_count": 0,
                             "top_source_label": "None", "avg_reliability": 0},
            latency_ms=1500,
        )
        defaults.update(kwargs)
        return create_interaction_log(**defaults)

    # ── Schema completeness ────────────────────────────────────────────────

    def test_all_required_keys_present(self):
        log = self._build()
        required = [
            "session_id", "timestamp", "tenant_id", "doctor_id", "patient_id",
            "department", "prompt", "original_response", "final_response",
            "confidence_score", "validation_score", "guardrail_status",
            "model_name", "lora_model",
            "tokens_input", "tokens_output", "tokens_total",
            "latency_ms", "sources_used",
            "error_flag", "error_message", "environment",
        ]
        for key in required:
            self.assertIn(key, log, f"Missing key: {key}")

    def test_identifiers_propagated(self):
        log = self._build()
        self.assertEqual(log["session_id"], "sess-001")
        self.assertEqual(log["doctor_id"], "doc-42")
        self.assertEqual(log["patient_id"], "pat-99")
        self.assertEqual(log["department"], "Cardiology")

    def test_scores_propagated(self):
        log = self._build()
        self.assertEqual(log["validation_score"], 80)
        self.assertEqual(log["confidence_score"], 72)
        self.assertEqual(log["guardrail_status"], "SAFE")

    def test_latency_propagated(self):
        log = self._build(latency_ms=2500)
        self.assertEqual(log["latency_ms"], 2500)

    def test_tenant_id_from_config(self):
        log = self._build()
        # tenant_id should be populated from Config (exact value depends on .env)
        self.assertIsInstance(log["tenant_id"], str)
        self.assertGreater(len(log["tenant_id"]), 0)

    def test_model_name_from_config(self):
        log = self._build()
        # model_name should be populated from Config (exact value depends on .env)
        self.assertIsInstance(log["model_name"], str)
        self.assertGreater(len(log["model_name"]), 0)

    def test_environment_from_config(self):
        log = self._build()
        # environment should be a known value (exact value depends on FLASK_ENV)
        self.assertIsInstance(log["environment"], str)
        self.assertGreater(len(log["environment"]), 0)

    # ── PII masking ────────────────────────────────────────────────────────

    def test_pii_masked_in_prompt(self):
        log = self._build(prompt="SSN: 123-45-6789 — need therapy")
        self.assertNotIn("123-45-6789", log["prompt"])
        self.assertIn("[SSN]", log["prompt"])

    def test_pii_masked_in_response(self):
        log = self._build(final_response="Contact patient@example.com for results")
        self.assertNotIn("patient@example.com", log["final_response"])
        self.assertIn("[EMAIL]", log["final_response"])

    def test_pii_masked_in_original_response(self):
        log = self._build(original_response="Call 555-123-4567 for appointment")
        self.assertNotIn("555-123-4567", log["original_response"])
        self.assertIn("[PHONE]", log["original_response"])

    # ── Field truncation ───────────────────────────────────────────────────

    def test_prompt_truncated_at_2000_chars(self):
        log = self._build(prompt="x" * 5000)
        self.assertEqual(len(log["prompt"]), 2000)

    def test_response_truncated_at_4000_chars(self):
        log = self._build(final_response="y" * 6000)
        self.assertEqual(len(log["final_response"]), 4000)

    def test_error_message_truncated_at_500_chars(self):
        log = self._build(error_info={"error": True, "message": "e" * 1000})
        self.assertEqual(len(log["error_message"]), 500)

    # ── Sources ────────────────────────────────────────────────────────────

    def test_sources_used_is_json_string(self):
        log = self._build(
            citation_result={
                "structured_citations": [
                    {"source_label": "Clinical Research (ArXiv)", "score": 0.9},
                    {"source_label": "Organisation Knowledge Base", "score": 0.8},
                ],
                "source_count": 2,
                "top_source_label": "Clinical Research (ArXiv)",
                "avg_reliability": 90,
            }
        )
        sources = json.loads(log["sources_used"])
        self.assertIn("Clinical Research (ArXiv)", sources)
        self.assertIn("Organisation Knowledge Base", sources)

    def test_sources_deduped(self):
        log = self._build(
            citation_result={
                "structured_citations": [
                    {"source_label": "Wikipedia", "score": 0.7},
                    {"source_label": "Wikipedia", "score": 0.6},
                ],
                "source_count": 2,
                "top_source_label": "Wikipedia",
                "avg_reliability": 70,
            }
        )
        sources = json.loads(log["sources_used"])
        self.assertEqual(sources.count("Wikipedia"), 1)

    def test_empty_citations_gives_empty_list(self):
        log = self._build()
        sources = json.loads(log["sources_used"])
        self.assertEqual(sources, [])

    # ── LoRA attribution ───────────────────────────────────────────────────

    def test_lora_model_set_when_used(self):
        log = self._build(lora_info={"used": True, "department": "Neurology"})
        self.assertEqual(log["lora_model"], "Neurology")

    def test_lora_model_empty_when_not_used(self):
        log = self._build(lora_info={"used": False, "department": "Neurology"})
        self.assertEqual(log["lora_model"], "")

    def test_lora_model_empty_when_none(self):
        log = self._build(lora_info=None)
        self.assertEqual(log["lora_model"], "")

    def test_lora_falls_back_to_model_version(self):
        log = self._build(lora_info={"used": True, "department": "", "model_version": "v2.1"})
        self.assertEqual(log["lora_model"], "v2.1")

    # ── Error info ─────────────────────────────────────────────────────────

    def test_error_flag_true_when_error(self):
        log = self._build(error_info={"error": True, "message": "Timeout"})
        self.assertTrue(log["error_flag"])
        self.assertEqual(log["error_message"], "Timeout")

    def test_error_flag_false_by_default(self):
        log = self._build()
        self.assertFalse(log["error_flag"])
        self.assertEqual(log["error_message"], "")

    def test_error_flag_false_when_error_false(self):
        log = self._build(error_info={"error": False, "message": ""})
        self.assertFalse(log["error_flag"])

    # ── Token estimation ───────────────────────────────────────────────────

    def test_tokens_estimated_when_no_usage_obj(self):
        log = self._build(prompt="a" * 400, final_response="b" * 200)
        self.assertEqual(log["tokens_input"], 100)
        self.assertEqual(log["tokens_output"], 50)
        self.assertEqual(log["tokens_total"], 150)

    def test_real_token_usage_passed_through(self):
        usage = MagicMock()
        usage.prompt_tokens = 300
        usage.completion_tokens = 100
        usage.total_tokens = 400
        usage.input_tokens = None
        usage.output_tokens = None
        log = self._build(usage_obj=usage)
        self.assertEqual(log["tokens_input"], 300)
        self.assertEqual(log["tokens_output"], 100)
        self.assertEqual(log["tokens_total"], 400)

    # ── Timestamp ─────────────────────────────────────────────────────────

    def test_timestamp_is_utc_iso(self):
        import re
        log = self._build()
        ts = log["timestamp"]
        # Should match ISO 8601 format with timezone offset or Z
        self.assertRegex(ts, r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")

    # ── Missing / None inputs handled gracefully ───────────────────────────

    def test_none_results_handled(self):
        log = create_interaction_log(
            session_id="s",
            validation_result=None,
            guardrail_result=None,
            confidence_result=None,
            citation_result=None,
        )
        self.assertEqual(log["validation_score"], 0)
        self.assertEqual(log["guardrail_status"], "UNKNOWN")
        self.assertEqual(log["confidence_score"], 0)

    def test_empty_session_defaults(self):
        log = create_interaction_log()
        self.assertEqual(log["session_id"], "unknown")
        self.assertFalse(log["error_flag"])


class TestLogInteraction(unittest.TestCase):
    """log_interaction must fire async and never raise — even with DB errors."""

    def test_fires_without_raising(self):
        """log_interaction should not raise even with no DB."""
        log = create_interaction_log(session_id="s", prompt="test", final_response="ok")
        try:
            log_interaction(log)
        except Exception as exc:
            self.fail(f"log_interaction raised unexpectedly: {exc}")

    def test_fires_with_empty_dict(self):
        """log_interaction must not raise with a partially populated dict."""
        try:
            log_interaction({})
        except Exception as exc:
            self.fail(f"log_interaction raised with empty dict: {exc}")

    def test_returns_none(self):
        """log_interaction always returns None (fire-and-forget)."""
        log = create_interaction_log(session_id="s")
        result = log_interaction(log)
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    total = result.testsRun
    passed = total - len(result.failures) - len(result.errors)
    print(f"\n{'='*60}")
    print(f"Interaction Log Tests: {passed}/{total} passed")
    print(f"{'='*60}")
    sys.exit(0 if result.wasSuccessful() else 1)
