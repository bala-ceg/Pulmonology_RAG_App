"""
Session Service
===============
Tracks per-session state and triggers completion when any of three criteria are met:

  1. conversation_count >= SESSION_MAX_CONVERSATIONS (default 10)
  2. idle time > SESSION_IDLE_TIMEOUT_SECONDS (default 300 = 5 minutes)
  3. patient_id changes within the same session

On completion:
  - Serialises the session log to a tmp JSON file
  - Uploads to Azure Blob Storage: pces/session-logs/{YYYY-MMM-DD}/{session_id}_session.json
    e.g. pces/session-logs/2026-JUN-04/guest_060420262013_session.json
  - Writes a permanent local copy: session_logs/{YYYY-MMM-DD}/{session_id}_session.json
  - Cleans up the tmp file after a successful upload
  - Resets the session state so subsequent queries start fresh

Usage::

    from services.session_service import session_service

    # Called once per query in routes/query.py
    completed = session_service.on_query(
        session_id="guest_010620261200",
        patient_id="PAT_001",
        department="Cardiology",
        interaction_summary={
            "prompt_snippet": "...",
            "answer_snippet": "...",
            "confidence": 75,
            "sources": [...],
        },
        query_started_at=datetime.fromtimestamp(req_start, tz=timezone.utc),
    )
    # completed is the completion reason ("conversation_limit"|"patient_change") or None
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import time
import uuid
import zipfile
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from config import Config
from utils.error_handlers import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SESSION_MAX_CONVERSATIONS: int = int(os.getenv("SESSION_MAX_CONVERSATIONS", "10"))
SESSION_IDLE_TIMEOUT_SECONDS: int = int(os.getenv("SESSION_IDLE_TIMEOUT_SECONDS", "300"))  # 5 min
SESSION_WATCHDOG_INTERVAL_SECONDS: int = int(os.getenv("SESSION_WATCHDOG_INTERVAL_SECONDS", "60"))
AZURE_SESSION_LOG_PREFIX: str = "pces/session-logs"
AZURE_CONTAINER: str = "contoso"

# Local session log copy:  session_logs/{YYYY-MMM-DD}/{doctorID}_{patientID}_{MMDDYYYYHH24MM}.json
SESSION_LOGS_DIR: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "session_logs",
)

# HITL queue:  hitl_queue/{dept_safe}/{uuid4}.json
HITL_QUEUE_DIR: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "hitl_queue",
)

# ---------------------------------------------------------------------------
# In-memory log buffer — captures all app log lines with timestamps
# ---------------------------------------------------------------------------

_LOG_FMT = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                               datefmt="%Y-%m-%d %H:%M:%S")


class _SessionLogBuffer(logging.Handler):
    """
    Lightweight logging.Handler attached to the root logger.
    Stores (datetime_utc, formatted_line) in a bounded circular buffer.
    Used to extract log lines that fall within a session's time window.
    """

    _MAX_LINES = int(os.getenv("SESSION_LOG_BUFFER_SIZE", "5000"))

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.setFormatter(_LOG_FMT)
        # deque of (datetime_utc, line_str)
        self._buf: deque[tuple[datetime, str]] = deque(maxlen=self._MAX_LINES)
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
            ts = datetime.fromtimestamp(record.created, tz=timezone.utc)
            with self._lock:
                self._buf.append((ts, line))
        except Exception:
            pass

    def get_lines(self, start: datetime, end: datetime) -> list[str]:
        """Return all buffered log lines whose timestamp falls within [start, end]."""
        with self._lock:
            return [line for ts, line in self._buf if start <= ts <= end]

    def purge_before(self, cutoff: datetime) -> None:
        """Remove entries older than cutoff to free memory."""
        with self._lock:
            while self._buf and self._buf[0][0] < cutoff:
                self._buf.popleft()


# Singleton buffer — attached to root logger once at import time
_log_buffer = _SessionLogBuffer()
logging.getLogger().addHandler(_log_buffer)

# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------


@dataclass
class SessionState:
    session_id: str
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    conversation_count: int = 0
    patient_id: str = ""
    doctor_id: str = ""
    department: str = ""
    session_completed: bool = False
    interactions: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class SessionService:
    """Thread-safe in-memory session registry with Azure upload on completion."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._lock = threading.Lock()
        # Survives session resets — tracks last known patient_id per session_id
        # so patient_change is detected even when an idle_timeout reset happened between queries
        self._patient_registry: dict[str, str] = {}
        self._start_idle_watchdog()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_query(
        self,
        session_id: str,
        patient_id: str = "",
        doctor_id: str = "",
        department: str = "",
        interaction_summary: dict | None = None,
        query_started_at: datetime | None = None,
    ) -> str | None:
        """
        Record a new query against this session. Returns the completion reason
        ("conversation_limit" | "patient_change") if the session just completed,
        else None. Idle-timeout completions are handled by the watchdog.

        ``query_started_at`` should be the UTC datetime when the HTTP request
        began processing — used as ``started_at`` for new sessions so that all
        log lines emitted during the request are captured in app_logs.
        """
        if not session_id:
            return None

        reason: str | None = None

        with self._lock:
            state = self._sessions.get(session_id)

            # ── Patient change detection (survives session resets) ────────────
            last_known_patient = self._patient_registry.get(session_id, "")
            if patient_id and last_known_patient and patient_id != last_known_patient:
                logger.info(
                    "[SESSION] Patient changed in session %s: %s → %s",
                    session_id,
                    last_known_patient,
                    patient_id,
                )
                reason = "patient_change"

            if patient_id:
                self._patient_registry[session_id] = patient_id

            if state is None or state.session_completed:
                # New session or reset after completion.
                # Use query_started_at as started_at so all log lines emitted
                # during the current request fall within get_lines(started_at, ...)
                state = SessionState(
                    session_id=session_id,
                    patient_id=patient_id or "",
                    doctor_id=doctor_id or "",
                    department=department or "",
                    started_at=query_started_at or datetime.now(timezone.utc),
                )
                self._sessions[session_id] = state
                logger.debug("[SESSION] New session started: %s patient=%s doctor=%s", session_id, patient_id, doctor_id)

            # Update state
            state.last_activity = datetime.now(timezone.utc)
            state.conversation_count += 1
            if patient_id:
                state.patient_id = patient_id
            if doctor_id:
                state.doctor_id = doctor_id
            if department:
                state.department = department
            if interaction_summary:
                state.interactions.append(interaction_summary)

            # Conversation count check
            if not reason and state.conversation_count >= SESSION_MAX_CONVERSATIONS:
                reason = "conversation_limit"

        # Fire HITL/RLHF push for REVIEW interactions (non-blocking)
        if interaction_summary and interaction_summary.get("sme_review_needed") == "YES":
            self._fire_review_tasks(interaction_summary, department or (state.department if state else ""))

        if reason:
            self._complete_session(session_id, reason)

        return reason

    def on_patient_change(self, session_id: str, new_patient_id: str) -> bool:
        """
        Called immediately when the user selects a different patient in the UI —
        before any new query is submitted.  Completes the current session if the
        patient has changed.  Returns True if a session was completed.
        """
        if not session_id or not new_patient_id:
            return False

        with self._lock:
            last_known_patient = self._patient_registry.get(session_id, "")
            if not last_known_patient or new_patient_id == last_known_patient:
                # First patient selection or same patient — just register
                self._patient_registry[session_id] = new_patient_id
                return False

            logger.info(
                "[SESSION] Patient changed (UI select) in session %s: %s → %s",
                session_id,
                last_known_patient,
                new_patient_id,
            )
            # Register new patient BEFORE completing so the new session picks it up
            self._patient_registry[session_id] = new_patient_id

        self._complete_session(session_id, "patient_change")
        return True

    def get_state(self, session_id: str) -> dict | None:
        """Return a snapshot of the session state (for debug/testing)."""
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return None
            return {
                "session_id": state.session_id,
                "conversation_count": state.conversation_count,
                "patient_id": state.patient_id,
                "department": state.department,
                "session_completed": state.session_completed,
                "started_at": state.started_at.isoformat(),
                "last_activity": state.last_activity.isoformat(),
                "interaction_count": len(state.interactions),
            }

    # ------------------------------------------------------------------
    # Completion
    # ------------------------------------------------------------------

    def _complete_session(self, session_id: str, reason: str) -> None:
        """Mark session complete and kick off async upload."""
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None or state.session_completed:
                return
            state.session_completed = True
            log_payload = self._build_log_payload(state, reason)

        logger.info(
            "[SESSION] Session completed: %s  reason=%s  conversations=%d  patient=%s",
            session_id,
            reason,
            log_payload["conversation_count"],
            log_payload["patient_id"],
        )

        t = threading.Thread(
            target=self._upload_session_log,
            args=(session_id, log_payload),
            daemon=True,
            name=f"session-upload-{session_id}",
        )
        t.start()

    def _build_log_payload(self, state: SessionState, reason: str) -> dict:
        completed_at = datetime.now(timezone.utc)
        app_logs = _log_buffer.get_lines(state.started_at, completed_at)
        # Purge entries older than this session to keep memory lean
        _log_buffer.purge_before(state.started_at)
        return {
            "session_id": state.session_id,
            "completion_reason": reason,
            "started_at": state.started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "conversation_count": state.conversation_count,
            "patient_id": state.patient_id,
            "doctor_id": state.doctor_id,
            "department": state.department,
            "interactions": state.interactions,
            "app_logs": app_logs,
        }

    @staticmethod
    def _make_log_filename(log_payload: dict) -> str:
        """Build filename: {doctorID}_{patientID}_{MMDDYYYYHH24MM}.json"""
        doctor_id = re.sub(r"[^\w\-]", "", log_payload.get("doctor_id", "") or "unknown") or "unknown"
        patient_id = re.sub(r"[^\w\-]", "", log_payload.get("patient_id", "") or "unknown") or "unknown"
        try:
            started_at = datetime.fromisoformat(log_payload.get("started_at", ""))
        except (ValueError, TypeError):
            started_at = datetime.now(timezone.utc)
        ts = started_at.strftime("%m%d%Y%H%M")
        return f"{doctor_id}_{patient_id}_{ts}.json"

    # ------------------------------------------------------------------
    # Azure upload
    # ------------------------------------------------------------------

    def _upload_session_log(self, session_id: str, log_payload: dict) -> None:
        """Write local JSON copy, ZIP it, upload ZIP to Azure, then clean up tmp files."""
        tmp_json_path: str | None = None
        tmp_zip_path: str | None = None
        try:
            date_str = datetime.now(timezone.utc).strftime("%Y-%b-%d").upper()  # e.g. 2026-JUN-06
            filename = self._make_log_filename(log_payload)          # e.g. drsmith_pat_ts.json
            zip_filename = filename.replace(".json", ".zip")          # e.g. drsmith_pat_ts.zip

            # ── Permanent local copy (plain JSON) ────────────────────────────
            local_dir = os.path.join(SESSION_LOGS_DIR, date_str)
            os.makedirs(local_dir, exist_ok=True)
            local_path = os.path.join(local_dir, filename)
            with open(local_path, "w", encoding="utf-8") as f:
                json.dump(log_payload, f, indent=2, ensure_ascii=False)
            logger.info("[SESSION] Local session log saved: %s", local_path)

            # ── Write JSON to tmp file ───────────────────────────────────────
            tmp_fd, tmp_json_path = tempfile.mkstemp(
                suffix=".json",
                prefix=f"pces_session_{session_id}_",
            )
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(log_payload, f, indent=2, ensure_ascii=False)

            # ── Create tmp ZIP containing the JSON ───────────────────────────
            tmp_zip_fd, tmp_zip_path = tempfile.mkstemp(
                suffix=".zip",
                prefix=f"pces_session_{session_id}_",
            )
            os.close(tmp_zip_fd)
            with zipfile.ZipFile(tmp_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(tmp_json_path, arcname=filename)

            self._azure_upload(session_id, tmp_zip_path, log_payload, zip_filename, date_str)

        except Exception as exc:
            logger.warning(
                "[SESSION] Failed to write/upload session log for %s: %s",
                session_id,
                exc,
            )
        finally:
            for path in (tmp_json_path, tmp_zip_path):
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

    def _azure_upload(self, session_id: str, tmp_path: str, log_payload: dict,
                      filename: str | None = None, date_str: str | None = None) -> None:
        """Upload the session ZIP (or JSON fallback) to Azure Blob Storage."""
        try:
            from azure_storage import AzureStorageManager  # local import — optional dep
        except ImportError:
            logger.debug("[SESSION] azure_storage not available; skipping upload for %s", session_id)
            return

        try:
            manager = AzureStorageManager()
            if not date_str:
                date_str = datetime.now(timezone.utc).strftime("%Y-%b-%d").upper()
            if not filename:
                filename = self._make_log_filename(log_payload)
            blob_path = f"{AZURE_SESSION_LOG_PREFIX}/{date_str}/{filename}"

            with open(tmp_path, "rb") as f:
                content = f.read()

            blob_client = manager.blob_service_client.get_blob_client(
                container=AZURE_CONTAINER, blob=blob_path
            )
            blob_client.upload_blob(
                content,
                overwrite=True,
                metadata={
                    "session_id": session_id,
                    "patient_id": log_payload.get("patient_id", ""),
                    "doctor_id": log_payload.get("doctor_id", ""),
                    "completion_reason": log_payload.get("completion_reason", ""),
                    "conversation_count": str(log_payload.get("conversation_count", 0)),
                },
            )
            logger.info(
                "[SESSION] Uploaded session log to Azure: %s/%s", AZURE_CONTAINER, blob_path
            )
        except Exception as exc:
            logger.warning(
                "[SESSION] Azure upload failed for session %s (non-critical): %s", session_id, exc
            )

    # ------------------------------------------------------------------
    # HITL / RLHF push (review_sme = "YES" interactions)
    # ------------------------------------------------------------------

    def _fire_review_tasks(self, interaction: dict, department: str) -> None:
        """Launch RLHF DB insert + HITL file write in a background thread. Non-blocking."""
        t = threading.Thread(
            target=self._do_review_tasks,
            args=(interaction, department),
            daemon=True,
            name="session-review-push",
        )
        t.start()

    def _do_review_tasks(self, interaction: dict, department: str) -> None:
        interaction_id = str(uuid.uuid4())
        self._push_rlhf_review(interaction, department, interaction_id)
        self._write_hitl_queue(interaction, department, interaction_id)

    def _push_rlhf_review(self, interaction: dict, department: str, interaction_id: str) -> None:
        """Insert a REVIEW interaction into rlhf_interactions with sme_review_needed=true.

        The app DB user (pcesuser) lacks USAGE on the SERIAL sequence, so we compute
        the next interaction_id manually as MAX(interaction_id)+1 within the same
        transaction. A PK conflict on rare concurrent inserts is caught and logged.
        """
        try:
            from services.db_service import get_connection, execute_query
            prompt = interaction.get("full_prompt") or interaction.get("prompt_snippet", "")
            response = interaction.get("full_response") or interaction.get("response", "")
            with get_connection() as conn:
                # Compute next ID manually — pcesuser has no GRANT on the SERIAL sequence
                cur = conn.cursor()
                cur.execute("SELECT COALESCE(MAX(interaction_id), 0) + 1 FROM rlhf_interactions")
                next_id = cur.fetchone()[0]
                execute_query(
                    conn,
                    """
                    INSERT INTO rlhf_interactions
                        (interaction_id, user_prompt, ai_response, target_sme_roles,
                         sme_review_needed, bias_flag)
                    VALUES (?, ?, ?, ?, true, false)
                    """,
                    (next_id, prompt[:4000], response[:4000], department or ""),
                )
                conn.commit()
            logger.info(
                "[SESSION] RLHF review row inserted  id=%d  dept=%s",
                next_id, department,
            )
        except Exception as exc:
            logger.warning("[SESSION] RLHF push failed (non-critical): %s", exc)

    def _write_hitl_queue(self, interaction: dict, department: str, interaction_id: str) -> None:
        """Write a HITL queue JSON file for this REVIEW interaction."""
        try:
            dept_safe = re.sub(r"[^\w]", "_", (department or "general").split(",")[0].strip()).lower()
            queue_dir = os.path.join(HITL_QUEUE_DIR, dept_safe)
            os.makedirs(queue_dir, exist_ok=True)
            hitl_record = {
                "interaction_id": interaction_id,
                "department": department,
                "prompt": interaction.get("full_prompt") or interaction.get("prompt_snippet", ""),
                "response": interaction.get("full_response") or interaction.get("response", ""),
                "validation_status": "REVIEW",
                "confidence": interaction.get("confidence", 0),
                "SME_REVIEW_NEEDED": "YES",
            }
            file_path = os.path.join(queue_dir, f"{interaction_id}.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(hitl_record, f, indent=2, ensure_ascii=False)
            logger.info("[SESSION] HITL queue file written: %s", file_path)
        except Exception as exc:
            logger.warning("[SESSION] HITL queue write failed (non-critical): %s", exc)

    # ------------------------------------------------------------------
    # Idle watchdog
    # ------------------------------------------------------------------

    def _start_idle_watchdog(self) -> None:
        t = threading.Thread(
            target=self._idle_watchdog_loop,
            daemon=True,
            name="session-idle-watchdog",
        )
        t.start()
        logger.debug("[SESSION] Idle watchdog started (interval=%ds)", SESSION_WATCHDOG_INTERVAL_SECONDS)

    def _idle_watchdog_loop(self) -> None:
        while True:
            time.sleep(SESSION_WATCHDOG_INTERVAL_SECONDS)
            try:
                self._check_idle_sessions()
            except Exception as exc:
                logger.debug("[SESSION] Watchdog error (non-critical): %s", exc)

    def _check_idle_sessions(self) -> None:
        now = datetime.now(timezone.utc)
        to_complete: list[str] = []

        with self._lock:
            for sid, state in self._sessions.items():
                if state.session_completed:
                    continue
                idle_secs = (now - state.last_activity).total_seconds()
                if idle_secs > SESSION_IDLE_TIMEOUT_SECONDS:
                    to_complete.append(sid)

        for sid in to_complete:
            logger.info("[SESSION] Idle timeout for session %s", sid)
            self._complete_session(sid, "idle_timeout")


# Module-level singleton
session_service = SessionService()
