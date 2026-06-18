"""
Voice Command Engine for PCES
==============================
Implements the "Yodha" wake-word pattern described in
12_Voice activated to Trigger Commands.docx.

Architecture (web-app adaptation):
  Browser MediaRecorder → /transcribe (Whisper) → text
  JS detects wake word → POST /api/voice/dispatch
  parse_voice() + dispatch() → {success, action, message}
  JS executes the mapped function OR speaks the error via speechSynthesis.

No audio I/O here — the browser handles microphone access.
"""

from __future__ import annotations

from utils.error_handlers import get_logger

logger = get_logger(__name__)

WAKE_WORD: str = "yodha"

# ---------------------------------------------------------------------------
# Command Registry
# ---------------------------------------------------------------------------
# Keys   : normalised command phrase (all lower-case, no wake word)
# action : JavaScript function name called by the frontend on success
# enabled: default state; overridden per-request by the frontend's live
#          button.disabled state so COMMANDS always stays in sync.
# label  : human-readable name used in TTS error messages
# ---------------------------------------------------------------------------
COMMANDS: dict[str, dict] = {
    "generate": {
        "action": "sendMessage",
        "enabled": True,
        "label": "Generate",
    },
    "save to pdf": {
        "action": "saveChat",
        "enabled": False,
        "label": "Save to PDF",
    },
    "save as pdf": {
        "action": "saveChat",
        "enabled": False,
        "label": "Save as PDF",
    },
    "plain english": {
        "action": "refineInput",
        "enabled": True,
        "label": "Plain English",
    },
    "start recording": {
        "action": "startRecording",
        "enabled": True,
        "label": "Start Recording",
    },
    "stop recording": {
        "action": "stopRecording",
        "enabled": False,
        "label": "Stop Recording",
    },
    "update knowledge bank": {
        "action": "createVectorDB",
        "enabled": True,
        "label": "Update Knowledge Bank",
    },
    "update kb": {
        "action": "createVectorDB",
        "enabled": True,
        "label": "Update KB",
    },
    "create adhoc rag": {
        "action": "createVectorDB",
        "enabled": True,
        "label": "Create AdHoc RAG",
    },
    "generate summary": {
        "action": "generateSummaryFromTranscription",
        "enabled": True,
        "label": "Generate Summary",
    },
    "translation": {
        "action": "openTranslationModal",
        "enabled": True,
        "label": "Translation",
    },
    "record patient notes": {
        "action": "startPatientRecording",
        "enabled": True,
        "label": "Record Patient Notes",
    },
}


# ---------------------------------------------------------------------------
# parse_voice
# ---------------------------------------------------------------------------

def parse_voice(text: str) -> str | None:
    """Strip wake word and return the command phrase, or None if wake word absent.

    Examples:
        "Yodha Generate"       → "generate"
        "Yodha Save to PDF"    → "save to pdf"
        "Generate"             → None   (no wake word)
        "Yodha"                → None   (wake word only, no command)
    """
    normalised = text.lower().strip()
    if not normalised.startswith(WAKE_WORD):
        logger.debug("parse_voice: wake word '%s' not found in %r", WAKE_WORD, text[:60])
        return None

    command = normalised[len(WAKE_WORD):].strip()
    return command if command else None


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

def dispatch(command: str, button_enabled: bool | None = None) -> dict:
    """Resolve a command phrase and determine whether to execute it.

    Args:
        command:        Normalised command text (output of parse_voice).
        button_enabled: Live enabled state supplied by the frontend DOM.
                        When None, falls back to COMMANDS[command]["enabled"].

    Returns:
        {
          "success":  bool,
          "action":   str | None,   # JS function name to call on the frontend
          "message":  str,          # TTS-ready response string
        }
    """
    cmd = COMMANDS.get(command)
    if cmd is None:
        # Fuzzy fall-back: substring match
        for key, entry in COMMANDS.items():
            if command in key or key in command:
                cmd = entry
                command = key
                break

    if cmd is None:
        msg = f"I don't know the command '{command}'."
        logger.info("dispatch: unknown command %r", command)
        return {"success": False, "action": None, "message": msg}

    enabled = button_enabled if button_enabled is not None else cmd["enabled"]
    if not enabled:
        msg = f"That '{cmd['label']}' button is not enabled."
        logger.info("dispatch: command %r disabled (button_enabled=%s)", command, button_enabled)
        return {"success": False, "action": cmd["action"], "message": msg}

    msg = f"Running {cmd['label']}."
    logger.info("dispatch: executing command %r → %s", command, cmd["action"])
    return {"success": True, "action": cmd["action"], "message": msg}
