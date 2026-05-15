"""Unit tests for BuildContextService (services/context_service.py).

Run with:
    python tests/test_context_service.py
"""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeDoc:
    def __init__(self, content: str, metadata: dict | None = None):
        self.page_content = content
        self.metadata = metadata or {}


def _make_rag_manager(main_docs=None, adhoc_docs=None):
    mgr = MagicMock()
    # _retrieve_main_rag calls rag_manager.kb_local.similarity_search(...)
    _main = main_docs or [_FakeDoc("main rag result")]
    mgr.kb_local.similarity_search.return_value = _main
    # _retrieve_adhoc_rag calls rag_manager.retrieve_adhoc(...)
    mgr.retrieve_adhoc.return_value = adhoc_docs or [_FakeDoc("adhoc result")]
    return mgr


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildContextService(unittest.TestCase):

    def _get_service(self, rag_manager=None):
        """Import and return a fresh BuildContextService instance."""
        from services.context_service import BuildContextService
        svc = BuildContextService()
        svc.initialize(rag_manager or _make_rag_manager())
        return svc

    def test_build_returns_combined_context(self):
        from services.context_service import ContextRequest
        svc = self._get_service()
        req = ContextRequest(
            prompt="What is COPD?",
            department="Pulmonology",
            doctor_id="dr1",
            tenant_id="t1",
            session_id="sess1",
        )
        result = svc.build(req)
        # combined is a list[str]
        self.assertTrue(len(result.combined) > 0, "Combined context should not be empty")
        self.assertTrue(len(result.main_rag) > 0)

    def test_keyword_gate_triggers_external(self):
        from services.context_service import BuildContextService
        svc = BuildContextService.__new__(BuildContextService)
        self.assertTrue(svc._should_call_external("What is the latest research on rare COPD?"))
        self.assertFalse(svc._should_call_external("How do I prescribe amoxicillin?"))

    def test_keyword_gate_case_insensitive(self):
        from services.context_service import BuildContextService
        svc = BuildContextService.__new__(BuildContextService)
        self.assertTrue(svc._should_call_external("Clinical Trial results for asthma"))
        self.assertTrue(svc._should_call_external("Unknown etiology in patient"))

    def test_limit_context_trims_to_max(self):
        from services.context_service import BuildContextService
        svc = BuildContextService.__new__(BuildContextService)
        chunks = [f"chunk {i}" for i in range(20)]
        limited = svc._limit_context(chunks, max_chunks=10)
        self.assertEqual(len(limited), 10)

    def test_limit_context_no_change_when_under_max(self):
        from services.context_service import BuildContextService
        svc = BuildContextService.__new__(BuildContextService)
        chunks = ["a", "b", "c"]
        self.assertEqual(svc._limit_context(chunks, max_chunks=10), chunks)

    def test_build_handles_retrieval_error_gracefully(self):
        """If RAG retrieval raises, build should not propagate the exception."""
        from services.context_service import ContextRequest
        mgr = MagicMock()
        mgr.retrieve.side_effect = RuntimeError("DB unreachable")
        mgr.retrieve_adhoc.return_value = []
        svc = self._get_service(rag_manager=mgr)
        req = ContextRequest(
            prompt="test query",
            department="Cardiology",
            doctor_id="dr2",
            tenant_id="t1",
            session_id="sess2",
        )
        result = svc.build(req)
        # Should succeed with empty/fallback combined context, not crash
        self.assertIsNotNone(result)

    def test_build_without_initialize_raises(self):
        """Calling build() before initialize() should raise a clear error."""
        from services.context_service import BuildContextService, ContextRequest
        svc = BuildContextService()
        req = ContextRequest(
            prompt="test",
            department="Cardiology",
            doctor_id="dr3",
            tenant_id="t1",
            session_id="sess3",
        )
        with self.assertRaises(RuntimeError):
            svc.build(req)

    def test_adhoc_rag_included_in_combined(self):
        """Ad hoc results should appear in combined context."""
        from services.context_service import ContextRequest
        adhoc = [_FakeDoc("adhoc case note — critical")]
        svc = self._get_service(rag_manager=_make_rag_manager(adhoc_docs=adhoc))
        req = ContextRequest(
            prompt="review case notes",
            department="Oncology",
            doctor_id="dr4",
            patient_id="pt99",
            tenant_id="t1",
            session_id="sess4",
        )
        result = svc.build(req)
        combined_text = " ".join(result.combined).lower()
        self.assertIn("adhoc", combined_text, "Combined context should include adhoc results")

    def test_context_result_dataclass_fields(self):
        from services.context_service import ContextResult
        r = ContextResult(
            main_rag="main",
            adhoc_rag="adhoc",
            patient_data="pd",
            external="ext",
            combined="combined",
        )
        self.assertEqual(r.main_rag, "main")
        self.assertEqual(r.combined, "combined")


if __name__ == "__main__":
    unittest.main(verbosity=2)
