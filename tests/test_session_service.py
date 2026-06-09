"""
Tests for services/session_service.py
=======================================
Covers all three completion triggers, state management, and upload path.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from datetime import datetime, timedelta, timezone
from io import StringIO
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Ensure the project root is on the path
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Stub out azure_storage before importing session_service so Azure is optional
_azure_stub = types.ModuleType("azure_storage")
_azure_stub.AzureStorageManager = None  # type: ignore[attr-defined]
sys.modules.setdefault("azure_storage", _azure_stub)

# Stub config if not importable in test env
try:
    from config import Config  # noqa: F401
except Exception:
    _cfg = types.ModuleType("config")
    class _Config:
        TENANT_ID = "test-tenant"
    _cfg.Config = _Config  # type: ignore[attr-defined]
    sys.modules["config"] = _cfg

from services.session_service import (  # noqa: E402
    SESSION_IDLE_TIMEOUT_SECONDS,
    SESSION_MAX_CONVERSATIONS,
    SessionService,
    SessionState,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_service() -> SessionService:
    """Create a SessionService without starting the watchdog thread."""
    svc = object.__new__(SessionService)
    svc._sessions = {}  # type: ignore[attr-defined]
    svc._patient_registry = {}  # type: ignore[attr-defined]
    svc._lock = threading.Lock()
    return svc


def _make_summary(prompt: str = "test query", confidence: int = 75, sme_review_needed: str = "NO") -> dict:
    return {
        "timestamp": "",
        "prompt_snippet": prompt[:120],
        "response": "test answer",
        "sme_review_needed": sme_review_needed,
        "confidence": confidence,
        "sources": [],
        "full_prompt": prompt,
        "full_response": "test answer full",
    }


# ---------------------------------------------------------------------------
# Tests: initial state
# ---------------------------------------------------------------------------

class TestSessionInitialState(unittest.TestCase):

    def test_new_session_created_on_first_query(self):
        svc = _make_service()
        result = svc.on_query("sess_001", patient_id="PAT_A", department="Cardiology")
        self.assertIsNone(result)
        state = svc._sessions.get("sess_001")
        self.assertIsNotNone(state)
        self.assertEqual(state.conversation_count, 1)
        self.assertEqual(state.patient_id, "PAT_A")

    def test_blank_session_id_returns_none(self):
        svc = _make_service()
        result = svc.on_query("", patient_id="PAT_A")
        self.assertIsNone(result)
        self.assertEqual(len(svc._sessions), 0)

    def test_get_state_returns_none_for_unknown_session(self):
        svc = _make_service()
        self.assertIsNone(svc.get_state("no_such_session"))

    def test_get_state_snapshot(self):
        svc = _make_service()
        svc.on_query("sess_snap", patient_id="PAT_X", department="Neurology")
        snap = svc.get_state("sess_snap")
        self.assertEqual(snap["session_id"], "sess_snap")
        self.assertEqual(snap["conversation_count"], 1)
        self.assertFalse(snap["session_completed"])


# ---------------------------------------------------------------------------
# Tests: conversation_count trigger
# ---------------------------------------------------------------------------

class TestConversationLimitTrigger(unittest.TestCase):

    def _run_queries(self, svc: SessionService, session_id: str, count: int):
        for i in range(count):
            svc.on_query(session_id, patient_id="PAT_A", interaction_summary=_make_summary(f"q{i}"))

    def test_no_completion_before_limit(self):
        svc = _make_service()
        limit = SESSION_MAX_CONVERSATIONS
        for i in range(limit - 1):
            result = svc.on_query("sess_c", patient_id="PAT_A", interaction_summary=_make_summary())
            self.assertIsNone(result, f"Expected no completion at query {i+1}")

    def test_completion_at_limit(self):
        svc = _make_service()
        limit = SESSION_MAX_CONVERSATIONS

        with patch.object(svc, "_complete_session") as mock_complete:
            for _ in range(limit - 1):
                svc.on_query("sess_limit", patient_id="PAT_A")
            # This query reaches the limit
            result = svc.on_query("sess_limit", patient_id="PAT_A")

        self.assertEqual(result, "conversation_limit")
        mock_complete.assert_called_once_with("sess_limit", "conversation_limit")

    def test_session_resets_after_completion(self):
        """After completion, next query with the SAME patient starts fresh (no trigger)."""
        svc = _make_service()

        with patch.object(svc, "_complete_session"):
            # Complete the session
            for _ in range(SESSION_MAX_CONVERSATIONS):
                svc.on_query("sess_reset", patient_id="PAT_A")

        # Mark as completed manually (since _complete_session was mocked)
        svc._sessions["sess_reset"].session_completed = True

        # Same patient — should start fresh with no trigger
        result = svc.on_query("sess_reset", patient_id="PAT_A")
        state = svc._sessions["sess_reset"]
        self.assertEqual(state.conversation_count, 1)
        self.assertFalse(state.session_completed)
        self.assertIsNone(result)

    def test_interaction_summary_stored(self):
        svc = _make_service()
        summary = _make_summary("chest pain", confidence=80)
        svc.on_query("sess_store", patient_id="PAT_A", interaction_summary=summary)
        state = svc._sessions["sess_store"]
        self.assertEqual(len(state.interactions), 1)
        self.assertEqual(state.interactions[0]["prompt_snippet"], "chest pain")


# ---------------------------------------------------------------------------
# Tests: patient_change trigger
# ---------------------------------------------------------------------------

class TestPatientChangeTrigger(unittest.TestCase):

    def test_no_trigger_when_patient_unchanged(self):
        svc = _make_service()
        svc.on_query("sess_same", patient_id="PAT_A")
        result = svc.on_query("sess_same", patient_id="PAT_A")
        self.assertIsNone(result)

    def test_trigger_on_patient_change(self):
        svc = _make_service()

        with patch.object(svc, "_complete_session") as mock_complete:
            svc.on_query("sess_pc", patient_id="PAT_A")
            result = svc.on_query("sess_pc", patient_id="PAT_B")

        self.assertEqual(result, "patient_change")
        mock_complete.assert_called_once_with("sess_pc", "patient_change")

    def test_trigger_after_idle_timeout_reset(self):
        """
        Patient change must be detected even when the session was reset by an
        idle_timeout between the two queries (the _patient_registry survives resets).
        """
        svc = _make_service()

        with patch.object(svc, "_complete_session"):
            svc.on_query("sess_idle_reset", patient_id="PAT_A")

        # Simulate idle_timeout reset — mark session completed
        svc._sessions["sess_idle_reset"].session_completed = True

        # Next query with different patient — should detect change via _patient_registry
        with patch.object(svc, "_complete_session") as mock_complete:
            result = svc.on_query("sess_idle_reset", patient_id="PAT_B")

        self.assertEqual(result, "patient_change")
        mock_complete.assert_called_once_with("sess_idle_reset", "patient_change")

    def test_no_trigger_when_new_session_has_no_prior_patient(self):
        """First query on fresh session — no prior patient_id, no trigger."""
        svc = _make_service()
        result = svc.on_query("sess_fresh", patient_id="PAT_A")
        self.assertIsNone(result)

    def test_patient_updated_after_change_detected(self):
        """After patient change, session state is reset with new patient_id."""
        svc = _make_service()

        with patch.object(svc, "_complete_session"):
            svc.on_query("sess_upd", patient_id="PAT_OLD")
            svc.on_query("sess_upd", patient_id="PAT_NEW")

        # Mark old session as completed
        svc._sessions["sess_upd"].session_completed = True

        # Next query starts fresh with PAT_NEW
        svc.on_query("sess_upd", patient_id="PAT_NEW")
        state = svc._sessions["sess_upd"]
        self.assertEqual(state.patient_id, "PAT_NEW")

    def test_empty_patient_id_does_not_trigger(self):
        """Empty patient_id treated as anonymous — no change trigger."""
        svc = _make_service()
        svc.on_query("sess_anon", patient_id="PAT_A")
        result = svc.on_query("sess_anon", patient_id="")
        self.assertIsNone(result)


class TestOnPatientChange(unittest.TestCase):
    """Tests for on_patient_change() — the immediate UI-triggered path."""

    def test_first_selection_registers_patient(self):
        """First patient selection has no prior — registers without completing."""
        svc = _make_service()
        with patch.object(svc, "_complete_session") as mock_complete:
            completed = svc.on_patient_change("sess_new", "PAT_A")
        self.assertFalse(completed)
        mock_complete.assert_not_called()
        self.assertEqual(svc._patient_registry.get("sess_new"), "PAT_A")

    def test_same_patient_reselected_no_trigger(self):
        """Re-selecting the same patient does not complete the session."""
        svc = _make_service()
        svc.on_patient_change("sess_same", "PAT_A")
        with patch.object(svc, "_complete_session") as mock_complete:
            completed = svc.on_patient_change("sess_same", "PAT_A")
        self.assertFalse(completed)
        mock_complete.assert_not_called()

    def test_different_patient_completes_session_immediately(self):
        """Selecting a different patient fires completion before any new query."""
        svc = _make_service()
        svc.on_patient_change("sess_chg", "PAT_A")
        with patch.object(svc, "_complete_session") as mock_complete:
            completed = svc.on_patient_change("sess_chg", "PAT_B")
        self.assertTrue(completed)
        mock_complete.assert_called_once_with("sess_chg", "patient_change")

    def test_registry_updated_to_new_patient_after_change(self):
        """After patient_change, registry holds the NEW patient_id."""
        svc = _make_service()
        svc.on_patient_change("sess_reg", "PAT_A")
        with patch.object(svc, "_complete_session"):
            svc.on_patient_change("sess_reg", "PAT_B")
        self.assertEqual(svc._patient_registry.get("sess_reg"), "PAT_B")

    def test_empty_patient_id_ignored(self):
        """Empty patient_id is a no-op."""
        svc = _make_service()
        svc.on_patient_change("sess_empty", "PAT_A")
        with patch.object(svc, "_complete_session") as mock_complete:
            completed = svc.on_patient_change("sess_empty", "")
        self.assertFalse(completed)
        mock_complete.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: idle_timeout trigger (watchdog)
# ---------------------------------------------------------------------------

class TestIdleTimeoutTrigger(unittest.TestCase):

    def test_check_idle_sessions_triggers_completion(self):
        svc = _make_service()
        # Create an "old" session
        old_time = datetime.now(timezone.utc) - timedelta(seconds=SESSION_IDLE_TIMEOUT_SECONDS + 60)
        state = SessionState(session_id="sess_idle", last_activity=old_time, patient_id="PAT_A")
        svc._sessions["sess_idle"] = state

        with patch.object(svc, "_complete_session") as mock_complete:
            svc._check_idle_sessions()

        mock_complete.assert_called_once_with("sess_idle", "idle_timeout")

    def test_active_session_not_timed_out(self):
        svc = _make_service()
        state = SessionState(session_id="sess_active", last_activity=datetime.now(timezone.utc))
        svc._sessions["sess_active"] = state

        with patch.object(svc, "_complete_session") as mock_complete:
            svc._check_idle_sessions()

        mock_complete.assert_not_called()

    def test_already_completed_session_skipped(self):
        svc = _make_service()
        old_time = datetime.now(timezone.utc) - timedelta(seconds=SESSION_IDLE_TIMEOUT_SECONDS + 60)
        state = SessionState(session_id="sess_done", last_activity=old_time, session_completed=True)
        svc._sessions["sess_done"] = state

        with patch.object(svc, "_complete_session") as mock_complete:
            svc._check_idle_sessions()

        mock_complete.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: log payload and tmp file
# ---------------------------------------------------------------------------

class TestSessionLogPayload(unittest.TestCase):

    def test_build_log_payload_structure(self):
        svc = _make_service()
        state = SessionState(
            session_id="sess_log",
            patient_id="PAT_001",
            doctor_id="drsmith",
            department="Cardiology",
            conversation_count=5,
        )
        state.interactions.append(_make_summary("bp query"))
        payload = svc._build_log_payload(state, "conversation_limit")

        self.assertEqual(payload["session_id"], "sess_log")
        self.assertEqual(payload["completion_reason"], "conversation_limit")
        self.assertEqual(payload["patient_id"], "PAT_001")
        self.assertEqual(payload["doctor_id"], "drsmith")
        self.assertEqual(payload["department"], "Cardiology")
        self.assertEqual(payload["conversation_count"], 5)
        self.assertIn("interactions", payload)
        self.assertEqual(len(payload["interactions"]), 1)
        # interaction must have response and sme_review_needed fields
        interaction = payload["interactions"][0]
        self.assertIn("response", interaction)
        self.assertIn("sme_review_needed", interaction)
        self.assertNotIn("answer_snippet", interaction)
        self.assertIn("started_at", payload)
        self.assertIn("completed_at", payload)

    def test_make_log_filename(self):
        from datetime import datetime, timezone
        payload = {
            "doctor_id": "drsmith",
            "patient_id": "509ebb39-6721-47c8-bb16-4ca52a0adb78",
            "started_at": "2026-06-06T15:00:50+00:00",
        }
        name = SessionService._make_log_filename(payload)
        self.assertTrue(name.startswith("drsmith_509ebb39-6721-47c8-bb16-4ca52a0adb78_"))
        self.assertTrue(name.endswith(".json"))
        self.assertIn("060620261500", name)  # MMDDYYYYHH24MM

    def test_make_log_filename_unknown_fallbacks(self):
        payload = {"doctor_id": "", "patient_id": None, "started_at": None}
        name = SessionService._make_log_filename(payload)
        self.assertTrue(name.startswith("unknown_unknown_"))
        self.assertTrue(name.endswith(".json"))

    def test_upload_session_log_writes_tmp_file(self):
        svc = _make_service()
        state = SessionState(session_id="sess_tmp", patient_id="PAT_X", conversation_count=3)
        payload = svc._build_log_payload(state, "conversation_limit")

        written_paths = []

        def capture_upload(session_id, tmp_path, log_payload, filename=None, date_str=None):
            written_paths.append(tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            # Azure receives a .zip; extract and verify JSON inside
            import zipfile
            with zipfile.ZipFile(tmp_path) as zf:
                inner = zf.namelist()[0]
                with zf.open(inner) as f:
                    data = json.load(f)
            self.assertEqual(data["session_id"], "sess_tmp")

        with patch.object(svc, "_azure_upload", side_effect=capture_upload):
            svc._upload_session_log("sess_tmp", payload)

        # Both tmp json and tmp zip should be cleaned up after upload
        for p in written_paths:
            self.assertFalse(os.path.exists(p), f"Tmp file {p} was not cleaned up")

    def test_tmp_file_cleaned_up_on_azure_failure(self):
        svc = _make_service()
        state = SessionState(session_id="sess_cleanup", patient_id="PAT_X", conversation_count=2)
        payload = svc._build_log_payload(state, "idle_timeout")

        with patch.object(svc, "_azure_upload", side_effect=RuntimeError("Azure down")):
            svc._upload_session_log("sess_cleanup", payload)

        # Should not raise; tmp files should be cleaned up
        import glob as _glob
        leftover_json = _glob.glob("/tmp/pces_session_sess_cleanup_*.json")
        leftover_zip = _glob.glob("/tmp/pces_session_sess_cleanup_*.zip")
        self.assertEqual(len(leftover_json), 0)
        self.assertEqual(len(leftover_zip), 0)


# ---------------------------------------------------------------------------
# Tests: Azure upload path
# ---------------------------------------------------------------------------

class TestAzureUpload(unittest.TestCase):

    def test_azure_upload_skipped_when_module_unavailable(self):
        svc = _make_service()
        import importlib
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "azure_storage":
                raise ImportError("not installed")
            return original_import(name, *args, **kwargs)

        import tempfile
        _, tmp = tempfile.mkstemp(suffix=".json")
        try:
            with open(tmp, "w") as f:
                json.dump({"session_id": "x"}, f)
            with patch("builtins.__import__", side_effect=mock_import):
                # Should not raise
                svc._azure_upload("sess_no_azure", tmp, {"session_id": "x"})
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def test_azure_upload_sends_correct_blob_path(self):
        svc = _make_service()
        mock_manager = MagicMock()
        mock_blob_client = MagicMock()
        mock_manager.blob_service_client.get_blob_client.return_value = mock_blob_client

        mock_azure_module = types.ModuleType("azure_storage")
        mock_azure_module.AzureStorageManager = MagicMock(return_value=mock_manager)  # type: ignore

        import tempfile, zipfile as _zipfile
        # Build a tmp zip (as _upload_session_log would create)
        _, tmp_zip = tempfile.mkstemp(suffix=".zip")
        zip_filename = "drsmith_P1_060920261445.zip"
        try:
            with _zipfile.ZipFile(tmp_zip, "w") as zf:
                zf.writestr("drsmith_P1_060920261445.json", json.dumps({"session_id": "sess_upload"}))

            with patch.dict(sys.modules, {"azure_storage": mock_azure_module}):
                svc._azure_upload(
                    "sess_upload", tmp_zip,
                    {"session_id": "sess_upload", "patient_id": "P1", "completion_reason": "idle_timeout"},
                    filename=zip_filename,
                )
        finally:
            if os.path.exists(tmp_zip):
                os.remove(tmp_zip)

        self.assertTrue(mock_manager.blob_service_client.get_blob_client.called)
        call_kwargs = mock_manager.blob_service_client.get_blob_client.call_args
        blob_path: str = call_kwargs[1].get("blob") or (call_kwargs[0][1] if len(call_kwargs[0]) > 1 else "")
        self.assertIn("pces/session-logs/", blob_path)
        self.assertTrue(blob_path.endswith(".zip"), f"Expected .zip blob, got: {blob_path}")
        # Date folder must use MMM format e.g. 2026-JUN-06
        import re
        self.assertRegex(blob_path, r"\d{4}-[A-Z]{3}-\d{2}")


# ---------------------------------------------------------------------------
# Tests: HITL queue and RLHF push
# ---------------------------------------------------------------------------

class TestHitlRlhfPush(unittest.TestCase):

    def test_write_hitl_queue_creates_file(self):
        svc = _make_service()
        interaction = {
            "full_prompt": "What is the first-line treatment?",
            "full_response": "Thiazide diuretic.",
            "confidence": 65,
        }
        import services.session_service as _ss_mod
        with tempfile.TemporaryDirectory() as tmpdir:
            orig = _ss_mod.HITL_QUEUE_DIR
            _ss_mod.HITL_QUEUE_DIR = tmpdir
            try:
                svc._write_hitl_queue(interaction, "Cardiology", "test-uuid-1234")
                dept_dir = os.path.join(tmpdir, "cardiology")
                files = os.listdir(dept_dir)
                self.assertEqual(len(files), 1)
                with open(os.path.join(dept_dir, files[0])) as f:
                    data = json.load(f)
                self.assertEqual(data["SME_REVIEW_NEEDED"], "YES")
                self.assertEqual(data["validation_status"], "REVIEW")
                self.assertEqual(data["interaction_id"], "test-uuid-1234")
                self.assertEqual(data["department"], "Cardiology")
            finally:
                _ss_mod.HITL_QUEUE_DIR = orig

    def test_hitl_queue_dept_safe_name(self):
        """Multi-word dept like 'CARDIOLOGY, GENERAL SURGEON' uses first word only."""
        svc = _make_service()
        interaction = {"full_prompt": "q", "full_response": "a", "confidence": 65}
        import services.session_service as _ss_mod
        with tempfile.TemporaryDirectory() as tmpdir:
            orig = _ss_mod.HITL_QUEUE_DIR
            _ss_mod.HITL_QUEUE_DIR = tmpdir
            try:
                svc._write_hitl_queue(interaction, "CARDIOLOGY, GENERAL SURGEON, SME", "uid-2")
                # First word "CARDIOLOGY" → "cardiology"
                self.assertIn("cardiology", os.listdir(tmpdir))
            finally:
                _ss_mod.HITL_QUEUE_DIR = orig

    def test_fire_review_tasks_called_for_review_summary(self):
        """on_query() fires review tasks when sme_review_needed == YES."""
        svc = _make_service()
        fired = []

        with patch.object(svc, "_fire_review_tasks", side_effect=lambda i, d: fired.append(d)):
            svc.on_query("sess_rv", patient_id="PAT_A",
                         interaction_summary=_make_summary(sme_review_needed="YES"),
                         department="Cardiology")

        self.assertEqual(len(fired), 1)
        self.assertEqual(fired[0], "Cardiology")

    def test_fire_review_tasks_not_called_for_no_review(self):
        """on_query() does NOT fire review tasks when sme_review_needed == NO."""
        svc = _make_service()
        fired = []

        with patch.object(svc, "_fire_review_tasks", side_effect=lambda i, d: fired.append(d)):
            svc.on_query("sess_norv", patient_id="PAT_A",
                         interaction_summary=_make_summary(sme_review_needed="NO"),
                         department="Cardiology")

        self.assertEqual(len(fired), 0)


# ---------------------------------------------------------------------------
# Tests: _complete_session idempotency
# ---------------------------------------------------------------------------

class TestCompleteSessionIdempotency(unittest.TestCase):

    def test_double_complete_not_triggered_twice(self):
        svc = _make_service()
        svc.on_query("sess_idem", patient_id="PAT_A")

        upload_calls = []

        def fake_upload(session_id, payload):
            upload_calls.append(session_id)

        with patch.object(svc, "_upload_session_log", side_effect=fake_upload):
            svc._complete_session("sess_idem", "conversation_limit")
            # Wait for the daemon thread to run
            time.sleep(0.05)
            svc._complete_session("sess_idem", "conversation_limit")
            time.sleep(0.05)

        self.assertEqual(len(upload_calls), 1, "Upload should be called only once")


# ---------------------------------------------------------------------------
# Tests: thread safety (basic smoke)
# ---------------------------------------------------------------------------

class TestThreadSafety(unittest.TestCase):

    def test_concurrent_queries_do_not_raise(self):
        svc = _make_service()
        errors = []

        def run():
            try:
                for _ in range(5):
                    svc.on_query("sess_thr", patient_id="PAT_X", interaction_summary=_make_summary())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=run) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])

    def test_concurrent_different_sessions(self):
        svc = _make_service()
        errors = []

        def run(sid):
            try:
                for _ in range(3):
                    svc.on_query(sid, patient_id="PAT_A")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=run, args=(f"sess_par_{i}",)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        self.assertEqual(len(svc._sessions), 5)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [
        TestSessionInitialState,
        TestConversationLimitTrigger,
        TestPatientChangeTrigger,
        TestIdleTimeoutTrigger,
        TestSessionLogPayload,
        TestAzureUpload,
        TestCompleteSessionIdempotency,
        TestThreadSafety,
        TestOnPatientChange,
        TestHitlRlhfPush,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
