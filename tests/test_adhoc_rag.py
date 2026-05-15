"""Unit tests for the Ad Hoc RAG store (rag_architecture.py adhoc methods).

Run with:
    python tests/test_adhoc_rag.py
"""

import os
import sys
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import TwoStoreRAGManager directly (venv has all deps installed)
from rag_architecture import TwoStoreRAGManager  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeDoc:
    def __init__(self, content: str, metadata: dict | None = None):
        self.page_content = content
        self.metadata = metadata or {}


def _make_mock_chroma(docs: list | None = None):
    store = MagicMock()
    # retrieve_adhoc uses similarity_search (no scores)
    store.similarity_search.return_value = [
        _FakeDoc("result text", {"source": "x", "rag_type": "adhoc"})
    ] if docs is None else docs
    store.add_documents = MagicMock()
    # cleanup_expired_adhoc_docs accesses store._collection.get / .delete
    store._collection.get.return_value = {
        "ids": ["id-1", "id-2"],
        "metadatas": [
            {"rag_type": "adhoc", "scope": "patient",
             "upload_time": (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()},
            {"rag_type": "adhoc", "scope": "patient",
             "upload_time": (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()},
        ],
    }
    store._collection.delete = MagicMock()
    return store


def _bare_manager(**kwargs) -> TwoStoreRAGManager:
    """Create a TwoStoreRAGManager without calling __init__ (bypasses ChromaDB / disk IO)."""
    mgr = TwoStoreRAGManager.__new__(TwoStoreRAGManager)
    mgr.adhoc_kb = kwargs.get("adhoc_kb", _make_mock_chroma())
    mgr.kb_local = MagicMock()
    mgr.kb_external = MagicMock()
    mgr.embeddings = MagicMock()
    return mgr


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestIngestAdhocDoc(unittest.TestCase):
    """Test TwoStoreRAGManager.ingest_adhoc_doc()."""

    def test_ingest_adds_metadata(self):
        mgr = _bare_manager()
        docs = [_FakeDoc("clinical note content")]
        extra = {"scope": "doctor", "doctor_id": "dr001", "patient_id": ""}
        mgr.ingest_adhoc_doc(docs, extra)
        call_args = mgr.adhoc_kb.add_documents.call_args
        ingested = call_args[0][0]
        self.assertEqual(len(ingested), 1)
        self.assertEqual(ingested[0].metadata["rag_type"], "adhoc")
        self.assertEqual(ingested[0].metadata["scope"], "doctor")
        self.assertEqual(ingested[0].metadata["doctor_id"], "dr001")

    def test_ingest_noop_when_no_store(self):
        mgr = _bare_manager(adhoc_kb=None)
        docs = [_FakeDoc("content")]
        # Should not raise
        mgr.ingest_adhoc_doc(docs, {})


class TestRetrieveAdhoc(unittest.TestCase):
    """Test TwoStoreRAGManager.retrieve_adhoc()."""

    def _make_manager_with_counter(self, doctor_docs=None, patient_docs=None):
        doctor_results = doctor_docs or [_FakeDoc("doctor doc", {"source": "d1"})]
        patient_results = patient_docs or [_FakeDoc("patient doc", {"source": "p1"})]
        call_count = [0]

        def side_effect(prompt, k, filter):
            call_count[0] += 1
            if filter.get("doctor_id"):
                return doctor_results
            return patient_results

        store = MagicMock()
        # retrieve_adhoc uses similarity_search (not _with_relevance_scores)
        store.similarity_search.side_effect = side_effect
        return _bare_manager(adhoc_kb=store), call_count

    def test_retrieves_both_scopes(self):
        mgr, calls = self._make_manager_with_counter()
        results = mgr.retrieve_adhoc("test prompt", tenant_id="t1", doctor_id="dr1", patient_id="pt1")
        self.assertEqual(calls[0], 2, "Should make two separate queries")
        self.assertGreater(len(results), 0)

    def test_deduplicates_results(self):
        shared_doc = _FakeDoc("same content", {"source": "shared"})
        same = [shared_doc]   # similarity_search returns plain Documents, not (doc, score) tuples
        mgr, _ = self._make_manager_with_counter(doctor_docs=same, patient_docs=same)
        results = mgr.retrieve_adhoc("q", tenant_id="t1", doctor_id="dr1", patient_id="pt1")
        sources = [d.metadata.get("source") for d in results]
        self.assertEqual(len(sources), len(set(sources)), "No duplicate sources")

    def test_noop_when_no_store(self):
        mgr = _bare_manager(adhoc_kb=None)
        results = mgr.retrieve_adhoc("q", tenant_id="t1", doctor_id="dr1", patient_id="pt1")
        self.assertEqual(results, [])


class TestCleanupExpiredAdhocDocs(unittest.TestCase):
    """Test TwoStoreRAGManager.cleanup_expired_adhoc_docs()."""

    def test_deletes_only_expired_patient_docs(self):
        store = _make_mock_chroma()
        mgr = _bare_manager(adhoc_kb=store)
        deleted = mgr.cleanup_expired_adhoc_docs(retention_days=90)
        # Mock has 2 docs: one 100 days old (expired), one 10 days old (fresh)
        self.assertEqual(deleted, 1)
        store._collection.delete.assert_called_once()

    def test_noop_when_no_store(self):
        mgr = _bare_manager(adhoc_kb=None)
        deleted = mgr.cleanup_expired_adhoc_docs(retention_days=90)
        self.assertEqual(deleted, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
