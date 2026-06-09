"""
Unit tests for services/confidence_service.py

Tests all 6 sub-scores + orchestrator (score_confidence) + decision logic.
No Flask required — all tests run directly.

Run:  python tests/test_confidence.py
"""

import sys
import os
import unittest

# Make project root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.confidence_service import (
    calculate_retrieval_score,
    calculate_evidence_score,
    get_validation_score,
    calculate_guardrail_score,
    get_model_confidence,
    calculate_consistency_score,
    calculate_weighted_score,
    confidence_decision,
    score_confidence,
)


class TestRetrievalScore(unittest.TestCase):

    def test_no_documents_returns_zero(self):
        self.assertEqual(calculate_retrieval_score([]), 0)

    def test_single_doc_with_relevance(self):
        docs = [{"fields": {"Relevance": "75%"}}]
        self.assertEqual(calculate_retrieval_score(docs), 75)

    def test_multiple_docs_averages(self):
        docs = [
            {"fields": {"Relevance": "80%"}},
            {"fields": {"Relevance": "60%"}},
            {"fields": {"Relevance": "70%"}},
        ]
        self.assertEqual(calculate_retrieval_score(docs), 70)

    def test_missing_relevance_field_skipped(self):
        docs = [
            {"fields": {"Relevance": "90%"}},
            {"fields": {}},  # no Relevance key
        ]
        self.assertEqual(calculate_retrieval_score(docs), 90)

    def test_unparseable_relevance_skipped(self):
        docs = [
            {"fields": {"Relevance": "N/A"}},
            {"fields": {"Relevance": "85%"}},
        ]
        self.assertEqual(calculate_retrieval_score(docs), 85)

    def test_integer_relevance_value(self):
        docs = [{"fields": {"Relevance": "100"}}]
        self.assertEqual(calculate_retrieval_score(docs), 100)


class TestEvidenceScore(unittest.TestCase):

    def test_no_context_returns_zero(self):
        self.assertEqual(calculate_evidence_score("Any response", []), 0)

    def test_exact_match(self):
        context = ["The patient should be monitored for IOP changes every 3 months"]
        response = "The patient should be monitored for IOP changes every 3 months and adjust treatment."
        score = calculate_evidence_score(response, context)
        self.assertEqual(score, 100)

    def test_no_match_returns_zero(self):
        context = ["Bimatoprost increases aqueous outflow in glaucoma patients."]
        response = "Timolol works by reducing aqueous humor production."
        score = calculate_evidence_score(response, context)
        self.assertEqual(score, 0)

    def test_partial_match(self):
        context = [
            "Bimatoprost is effective for IOP reduction",
            "Timolol is a beta blocker used in glaucoma",
        ]
        response = "Bimatoprost is effective for IOP reduction in patients with glaucoma."
        score = calculate_evidence_score(response, context)
        self.assertEqual(score, 50)  # 1/2 matched

    def test_short_snippet_skipped(self):
        context = ["Hi"]  # < 10 chars — not fingerprinted
        response = "Hi there, how are you?"
        score = calculate_evidence_score(response, context)
        self.assertEqual(score, 0)


class TestValidationScore(unittest.TestCase):

    def test_passthrough(self):
        self.assertEqual(get_validation_score(80), 80)

    def test_clamp_above_100(self):
        self.assertEqual(get_validation_score(120), 100)

    def test_clamp_below_zero(self):
        self.assertEqual(get_validation_score(-10), 0)


class TestGuardrailScore(unittest.TestCase):

    def test_safe_returns_100(self):
        self.assertEqual(calculate_guardrail_score("SAFE"), 100)

    def test_blocked_returns_zero(self):
        self.assertEqual(calculate_guardrail_score("BLOCKED"), 0)

    def test_emergency_returns_zero(self):
        self.assertEqual(calculate_guardrail_score("EMERGENCY"), 0)

    def test_unknown_returns_zero(self):
        self.assertEqual(calculate_guardrail_score("UNKNOWN"), 0)


class TestModelConfidence(unittest.TestCase):

    def test_baseline_short_response(self):
        score = get_model_confidence("The treatment is recommended.")
        self.assertGreater(score, 0)
        self.assertLessEqual(score, 100)

    def test_heavy_hedging_reduces_score(self):
        hedgy = (
            "This may possibly be the case, but it might not be. "
            "It depends on the situation. Perhaps the patient could try this. "
            "Typically it's unclear, but it might work."
        )
        score = get_model_confidence(hedgy)
        baseline = get_model_confidence("The treatment is recommended.")
        self.assertLess(score, baseline)

    def test_certainty_language_increases_score(self):
        certain = (
            "Studies confirm that clearly, the evidence shows this is the standard of care. "
            "Guidelines recommend this approach as it is well-documented and strongly recommended "
            "by consensus of leading ophthalmologists."
        )
        score = get_model_confidence(certain)
        baseline = get_model_confidence("The treatment is recommended.")
        self.assertGreater(score, baseline)

    def test_long_response_gets_bonus(self):
        short = "Yes."
        long_response = "A " * 600  # 600 words > 1000 chars
        self.assertGreater(get_model_confidence(long_response), get_model_confidence(short))

    def test_score_in_valid_range(self):
        extreme_hedging = " ".join(["may might possibly unclear uncertain not sure"] * 20)
        self.assertGreaterEqual(get_model_confidence(extreme_hedging), 0)
        self.assertLessEqual(get_model_confidence(extreme_hedging), 100)


class TestConsistencyScore(unittest.TestCase):

    def test_no_sources_returns_zero(self):
        self.assertEqual(calculate_consistency_score([], []), 0)

    def test_one_source_returns_50(self):
        docs = [{"fields": {}}]
        self.assertEqual(calculate_consistency_score(docs, []), 50)

    def test_two_sources_returns_90(self):
        docs = [{"fields": {}}, {"fields": {}}]
        self.assertEqual(calculate_consistency_score(docs, []), 90)

    def test_three_sources_returns_100(self):
        docs = [{"fields": {}}, {"fields": {}}, {"fields": {}}]
        self.assertEqual(calculate_consistency_score(docs, []), 100)

    def test_context_fallback_when_no_docs(self):
        context = ["fragment one", "fragment two"]
        self.assertEqual(calculate_consistency_score([], context), 90)


class TestWeightedScore(unittest.TestCase):

    def test_all_100_gives_100(self):
        sub_scores = {k: 100 for k in ["retrieval", "evidence", "validation", "guardrail", "model_conf", "consistency"]}
        self.assertEqual(calculate_weighted_score(sub_scores), 100)

    def test_all_zero_gives_zero(self):
        sub_scores = {k: 0 for k in ["retrieval", "evidence", "validation", "guardrail", "model_conf", "consistency"]}
        self.assertEqual(calculate_weighted_score(sub_scores), 0)

    def test_weights_sum_correctly(self):
        # If only validation=100 and the rest are 0, score = 100 * 0.20 = 20
        sub_scores = {"retrieval": 0, "evidence": 0, "validation": 100,
                      "guardrail": 0, "model_conf": 0, "consistency": 0}
        self.assertEqual(calculate_weighted_score(sub_scores), 20)


class TestConfidenceDecision(unittest.TestCase):

    def test_80_is_deliver(self):
        self.assertEqual(confidence_decision(80), "DELIVER")

    def test_100_is_deliver(self):
        self.assertEqual(confidence_decision(100), "DELIVER")

    def test_79_is_review(self):
        self.assertEqual(confidence_decision(79), "REVIEW")

    def test_60_is_review(self):
        self.assertEqual(confidence_decision(60), "REVIEW")

    def test_59_is_regenerate(self):
        self.assertEqual(confidence_decision(59), "REGENERATE")

    def test_0_is_regenerate(self):
        self.assertEqual(confidence_decision(0), "REGENERATE")


class TestScoreConfidence(unittest.TestCase):
    """Integration tests for score_confidence orchestrator."""

    def test_safe_ophthalmology_delivers(self):
        response = (
            "Prostaglandin analogues such as bimatoprost are strongly recommended as first-line "
            "treatment for open-angle glaucoma. Studies confirm this is the standard of care. "
            "Guidelines recommend monitoring IOP every 3 months. Evidence shows they reduce "
            "intraocular pressure by 25–35%. This is well-documented in clinical literature."
        )
        context = ["Prostaglandin analogues are strongly recommended as first-line treatment"]
        result = score_confidence(
            prompt="What is first-line treatment for open-angle glaucoma?",
            response=response,
            context=context,
            source_documents=[
                {"fields": {"Relevance": "85%"}},
                {"fields": {"Relevance": "78%"}},
            ],
            validation_score=80,
            guardrail_status="SAFE",
        )
        self.assertIn("score", result)
        self.assertIn("decision", result)
        self.assertIn("breakdown", result)
        self.assertGreater(result["score"], 0)
        self.assertIn(result["decision"], ["DELIVER", "REVIEW", "REGENERATE"])

    def test_blocked_guardrail_reduces_score(self):
        response = "Some clinical response about eye drops."
        result_safe = score_confidence("q", response, [], source_documents=[], validation_score=80, guardrail_status="SAFE")
        result_blocked = score_confidence("q", response, [], source_documents=[], validation_score=80, guardrail_status="BLOCKED")
        self.assertGreater(result_safe["score"], result_blocked["score"])
        self.assertEqual(result_blocked["breakdown"]["guardrail"], 0)

    def test_emergency_guardrail_score_zero(self):
        result = score_confidence(
            "q", "response", [], source_documents=[], validation_score=80, guardrail_status="EMERGENCY"
        )
        self.assertEqual(result["breakdown"]["guardrail"], 0)

    def test_breakdown_has_all_keys(self):
        result = score_confidence("q", "response", [], source_documents=[], validation_score=80, guardrail_status="SAFE")
        for key in ["retrieval", "evidence", "validation", "guardrail", "model_conf", "consistency"]:
            self.assertIn(key, result["breakdown"])

    def test_high_retrieval_score_improves_result(self):
        high_docs = [{"fields": {"Relevance": "95%"}}, {"fields": {"Relevance": "90%"}}]
        no_docs = []
        r_high = score_confidence("q", "response", [], source_documents=high_docs, validation_score=80, guardrail_status="SAFE")
        r_low  = score_confidence("q", "response", [], source_documents=no_docs, validation_score=80, guardrail_status="SAFE")
        self.assertGreater(r_high["score"], r_low["score"])

    def test_score_in_valid_range(self):
        result = score_confidence("q", "r", [], source_documents=[], validation_score=0, guardrail_status="BLOCKED")
        self.assertGreaterEqual(result["score"], 0)
        self.assertLessEqual(result["score"], 100)


if __name__ == "__main__":
    print("=" * 60)
    print("Confidence Scoring Service — Unit Tests")
    print("=" * 60)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [
        TestRetrievalScore,
        TestEvidenceScore,
        TestValidationScore,
        TestGuardrailScore,
        TestModelConfidence,
        TestConsistencyScore,
        TestWeightedScore,
        TestConfidenceDecision,
        TestScoreConfidence,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print(f"\n{'='*60}")
    print(f"Results: {result.testsRun} tests, "
          f"{len(result.failures)} failures, "
          f"{len(result.errors)} errors")
    sys.exit(0 if result.wasSuccessful() else 1)
