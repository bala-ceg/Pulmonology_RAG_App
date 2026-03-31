"""
Unit tests for domain_scope_guard.py
-------------------------------------
Tests the three-tier filtering logic (accepted / general_medical / rejected)
without loading a real SBERT model — the model is mocked via numpy arrays.
"""
import os
import sys
import types
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_vec(n, idx):
    """Return a unit vector of length n with 1.0 at position idx."""
    v = np.zeros(n, dtype=np.float32)
    v[idx] = 1.0
    return v


# ---------------------------------------------------------------------------
# Import module under test (guard disabled by default in conftest)
# ---------------------------------------------------------------------------

import domain_scope_guard as dsg


# ---------------------------------------------------------------------------
# Module-level constant tests
# ---------------------------------------------------------------------------

class TestConstants:
    def test_thresholds_are_positive_floats(self):
        assert isinstance(dsg.SCOPE_GUARD_THRESHOLD, float)
        assert isinstance(dsg.MEDICAL_FALLBACK_THRESHOLD, float)
        assert dsg.SCOPE_GUARD_THRESHOLD > 0
        assert dsg.MEDICAL_FALLBACK_THRESHOLD > 0

    def test_accepted_threshold_greater_than_medical(self):
        assert dsg.SCOPE_GUARD_THRESHOLD > dsg.MEDICAL_FALLBACK_THRESHOLD

    def test_general_medical_phrases_not_empty(self):
        assert len(dsg._GENERAL_MEDICAL_PHRASES) >= 10

    def test_status_constants(self):
        assert dsg.STATUS_ACCEPTED == "accepted"
        assert dsg.STATUS_GENERAL_MEDICAL == "general_medical"
        assert dsg.STATUS_REJECTED == "rejected"

    def test_general_medical_disclaimer_not_empty(self):
        assert len(dsg.GENERAL_MEDICAL_DISCLAIMER) > 10


# ---------------------------------------------------------------------------
# DomainScopeGuard — disabled mode
# ---------------------------------------------------------------------------

class TestGuardDisabled:
    """When SCOPE_GUARD_ENABLED=false, every query is accepted."""

    def setup_method(self):
        os.environ["SCOPE_GUARD_ENABLED"] = "false"
        self.guard = dsg.DomainScopeGuard()

    def teardown_method(self):
        os.environ["SCOPE_GUARD_ENABLED"] = "false"  # keep disabled for other tests

    def test_disabled_guard_accepts_medical(self):
        status, score, msg = self.guard.check("What is hypertension?")
        assert status == dsg.STATUS_ACCEPTED

    def test_disabled_guard_accepts_off_topic(self):
        status, score, msg = self.guard.check("How do I bake a chocolate cake?")
        assert status == dsg.STATUS_ACCEPTED

    def test_disabled_guard_returns_empty_message(self):
        _, _, msg = self.guard.check("anything")
        assert msg == ""

    def test_disabled_guard_score_is_one(self):
        # Disabled guard passes all queries through with score=1.0
        _, score, _ = self.guard.check("test query")
        assert score == 1.0


# ---------------------------------------------------------------------------
# DomainScopeGuard — no embedder (SBERT unavailable)
# ---------------------------------------------------------------------------

class TestGuardNoEmbedder:
    """When embedder is None (SBERT not loadable), guard passes everything through."""

    def setup_method(self):
        os.environ["SCOPE_GUARD_ENABLED"] = "true"
        self.guard = dsg.DomainScopeGuard.__new__(dsg.DomainScopeGuard)
        self.guard.threshold         = 0.45
        self.guard.medical_threshold = 0.20
        self.guard.enabled           = True
        self.guard._embedder         = None
        self.guard._corpus_matrix    = None
        self.guard._medical_matrix   = None
        self.guard._training_prompts = []
        self.guard._topic_summary    = []
        self.guard._corpus_size      = 0
        self.guard._last_refresh     = None

    def teardown_method(self):
        os.environ["SCOPE_GUARD_ENABLED"] = "false"

    def test_no_embedder_accepts_query(self):
        status, score, msg = self.guard.check("any query at all")
        assert status == dsg.STATUS_ACCEPTED
        assert score == 1.0   # pass-through returns 1.0


# ---------------------------------------------------------------------------
# DomainScopeGuard — mocked embedder (core logic tests)
# ---------------------------------------------------------------------------

class TestGuardWithMockedEmbedder:
    """
    Inject a mock embedder that returns deterministic unit vectors so we can
    test accepted / general_medical / rejected tiers precisely.
    """

    DIM = 8

    def _make_guard(self, training_vecs, medical_vecs):
        """Build a guard instance with pre-computed matrices (bypasses __init__)."""
        guard = dsg.DomainScopeGuard.__new__(dsg.DomainScopeGuard)
        guard.threshold         = 0.45
        guard.medical_threshold = 0.20
        guard.enabled           = True
        guard._training_prompts = [f"prompt_{i}" for i in range(len(training_vecs))]
        guard._topic_summary    = []
        guard._corpus_size      = len(training_vecs)
        guard._last_refresh     = None

        # Set pre-built matrices
        guard._corpus_matrix  = np.vstack(training_vecs).astype(np.float32) if training_vecs else None
        guard._medical_matrix = np.vstack(medical_vecs).astype(np.float32) if medical_vecs else None

        # Mock embedder: encode returns the first training vec for "match" or zero vec
        class MockEmbedder:
            def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
                return np.zeros((len(texts) if isinstance(texts, list) else 1, TestGuardWithMockedEmbedder.DIM), dtype=np.float32)

        guard._embedder = MockEmbedder()
        return guard

    def test_accepted_when_high_training_similarity(self):
        """Query vec identical to a training vec → cosine = 1.0 → accepted."""
        v = _unit_vec(self.DIM, 0)
        guard = self._make_guard([v], [_unit_vec(self.DIM, 1)])

        # Override encode to return the same vector as training
        def encode(texts, **kw):
            return v.reshape(1, -1)

        guard._embedder.encode = encode

        status, score, _ = guard.check("diabetes treatment")
        assert status == dsg.STATUS_ACCEPTED
        assert score >= 0.45

    def test_rejected_when_zero_similarity(self):
        """Query vec orthogonal to training and medical vecs → rejected."""
        train_v   = _unit_vec(self.DIM, 0)
        medical_v = _unit_vec(self.DIM, 1)
        query_v   = _unit_vec(self.DIM, 2)   # orthogonal to both

        guard = self._make_guard([train_v], [medical_v])

        def encode(texts, **kw):
            return query_v.reshape(1, -1)

        guard._embedder.encode = encode

        status, score, msg = guard.check("how to bake a cake")
        assert status == dsg.STATUS_REJECTED
        assert score < 0.20
        assert len(msg) > 0

    def test_general_medical_tier(self):
        """Query moderately similar to medical but not training → general_medical."""
        train_v   = _unit_vec(self.DIM, 0)
        medical_v = _unit_vec(self.DIM, 1)
        # Construct a query vec that is similar to medical (0.5) but not to training (0.0)
        query_v   = medical_v.copy()   # cosine=1.0 with medical, 0.0 with training

        guard = self._make_guard([train_v], [medical_v])
        guard.threshold = 0.45
        guard.medical_threshold = 0.20

        def encode(texts, **kw):
            return query_v.reshape(1, -1)

        guard._embedder.encode = encode

        status, score, msg = guard.check("general health question")
        # training score = 0.0, medical score = 1.0 → general_medical
        assert status == dsg.STATUS_GENERAL_MEDICAL
        assert dsg.GENERAL_MEDICAL_DISCLAIMER in msg

    def test_empty_corpus_falls_back_to_medical_check(self):
        """No training data → uses only medical matrix."""
        medical_v = _unit_vec(self.DIM, 0)
        guard = self._make_guard([], [medical_v])
        guard._corpus_matrix = None
        guard._corpus_size   = 0

        def encode(texts, **kw):
            return medical_v.reshape(1, -1)   # matches medical exactly

        guard._embedder.encode = encode

        status, _, _ = guard.check("what is blood pressure?")
        # No training corpus → falls to medical check: score=1.0 → general_medical
        assert status in (dsg.STATUS_GENERAL_MEDICAL, dsg.STATUS_ACCEPTED)

    def test_decline_message_not_empty(self):
        """_build_decline_message returns a non-empty string."""
        guard = self._make_guard([], [])
        guard._topic_summary = ["hypertension", "diabetes"]
        msg = guard._build_decline_message()
        assert isinstance(msg, str)
        assert len(msg) > 5


# ---------------------------------------------------------------------------
# Cosine similarity helper
# _cosine_similarity_matrix(query_vec_1d, corpus_matrix_2d) → 1D scores array
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors_give_1(self):
        v = _unit_vec(4, 0)          # 1-D query vector, shape (4,)
        m = _unit_vec(4, 0).reshape(1, 4)   # 2-D corpus, shape (1, 4)
        scores = dsg._cosine_similarity_matrix(v, m)
        assert abs(scores[0] - 1.0) < 1e-5

    def test_orthogonal_vectors_give_0(self):
        v = _unit_vec(4, 0)
        m = _unit_vec(4, 1).reshape(1, 4)
        scores = dsg._cosine_similarity_matrix(v, m)
        assert abs(scores[0]) < 1e-5

    def test_returns_max_over_corpus(self):
        v = _unit_vec(4, 0)
        m = np.vstack([_unit_vec(4, 1), _unit_vec(4, 0)])  # second row matches
        scores = dsg._cosine_similarity_matrix(v, m)
        assert abs(np.max(scores) - 1.0) < 1e-5
