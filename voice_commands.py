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
# _fuzzy_match  — token-overlap scoring
# ---------------------------------------------------------------------------

def _fuzzy_match(command: str) -> tuple[str, dict] | tuple[None, None]:
    """Find the best COMMANDS entry for *command* using word-token overlap.

    Scoring:
        overlap = number of words shared between *command* and candidate key
        score   = overlap / max(len(spoken_words), len(key_words))

    A match requires score ≥ 0.4 (at least 40% word overlap).  The candidate
    with the highest score wins.  Ties are broken by the longer key (more
    specific wins).

    Examples:
        "save pdf"        → "save to pdf"          (2/3 ≈ 0.67 ✓)
        "start"           → "start recording"      (1/2 = 0.50 ✓)
        "update bank"     → "update knowledge bank" (2/3 ≈ 0.67 ✓)
        "generate"        → "generate"             (1/1 = 1.00 ✓)
        "plain"           → "plain english"        (1/2 = 0.50 ✓)
        "stop"            → "stop recording"       (1/2 = 0.50 ✓)
        "record"          → "record patient notes" (1/3 = 0.33 ✗ — below threshold)
        "record notes"    → "record patient notes" (2/3 ≈ 0.67 ✓)
    """
    spoken_words = set(command.lower().split())
    if not spoken_words:
        return None, None

    best_key: str | None = None
    best_entry: dict | None = None
    best_score: float = 0.0

    for key, entry in COMMANDS.items():
        key_words = set(key.split())
        overlap = len(spoken_words & key_words)
        if overlap == 0:
            continue
        score = overlap / max(len(spoken_words), len(key_words))
        if score > best_score or (score == best_score and len(key) > len(best_key or "")):
            best_score = score
            best_key = key
            best_entry = entry

    if best_score >= 0.40:
        logger.debug(
            "_fuzzy_match: %r → %r (score=%.2f)", command, best_key, best_score
        )
        return best_key, best_entry
    return None, None


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

    Resolution order:
        1. Exact match in COMMANDS
        2. Token-overlap fuzzy match (score ≥ 0.4)
        3. Legacy substring fallback

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
    # 1. Exact match
    cmd = COMMANDS.get(command)
    matched_key = command if cmd else None

    # 2. Token-overlap fuzzy match
    if cmd is None:
        matched_key, cmd = _fuzzy_match(command)

    # 3. Legacy substring fallback (safety net)
    if cmd is None:
        for key, entry in COMMANDS.items():
            if command in key or key in command:
                matched_key, cmd = key, entry
                break

    if cmd is None:
        msg = f"I don't know the command '{command}'."
        logger.info("dispatch: unknown command %r", command)
        return {"success": False, "action": None, "message": msg}

    enabled = button_enabled if button_enabled is not None else cmd["enabled"]
    if not enabled:
        msg = f"That '{cmd['label']}' button is not enabled."
        logger.info("dispatch: command %r disabled (button_enabled=%s)", matched_key, button_enabled)
        return {"success": False, "action": cmd["action"], "message": msg}

    msg = f"Running {cmd['label']}."
    logger.info("dispatch: executing %r → %s (spoken=%r)", matched_key, cmd["action"], command)
    return {"success": True, "action": cmd["action"], "message": msg}
