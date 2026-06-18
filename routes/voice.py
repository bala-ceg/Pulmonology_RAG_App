"""
Voice Commands Blueprint
========================
Routes:
  POST /api/voice/dispatch   — parse wake word + dispatch command
  GET  /api/voice/commands   — list all registered commands (dev/debug)
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from utils.error_handlers import get_logger, handle_route_errors
from voice_commands import COMMANDS, _fuzzy_match, dispatch, parse_voice

logger = get_logger(__name__)

voice_bp = Blueprint("voice_bp", __name__)


@voice_bp.route("/api/voice/dispatch", methods=["POST"])
@handle_route_errors
def voice_dispatch():
    """Parse transcribed text for the wake word and execute the matching command.

    Request JSON:
        {
          "text": "Yodha Generate",
          "button_states": {            // optional — live DOM disabled states
            "sendMessage":              false,   // false = button IS enabled
            "saveChat":                 true,    // true  = button is DISABLED
            ...
          }
        }

    Response JSON:
        {
          "success":  bool,
          "action":   str | null,   // JS function name to call
          "message":  str,          // TTS-ready feedback string
          "command":  str | null,   // parsed command phrase (null if no wake word)
        }
    """
    body = request.get_json(silent=True) or {}
    text: str = body.get("text", "").strip()
    # button_states: maps JS action name → True if the DOM button is DISABLED
    button_states: dict = body.get("button_states", {})

    if not text:
        return jsonify({
            "success": False,
            "action": None,
            "message": "No text provided.",
            "command": None,
        }), 400

    command = parse_voice(text)
    if command is None:
        return jsonify({
            "success": False,
            "action": None,
            "message": "Wake word 'Yodha' not detected.",
            "command": None,
        })

    # Resolve the matching COMMANDS entry (exact → fuzzy → substring inside dispatch).
    # We need the action name HERE so we can look up the live button state before
    # calling dispatch() — otherwise fuzzy matches always get button_enabled=None.
    cmd_entry = COMMANDS.get(command)
    if cmd_entry is None:
        _, cmd_entry = _fuzzy_match(command)   # may still be None for unknown commands

    action = cmd_entry["action"] if cmd_entry else None
    # button_states[action] == True  → button IS disabled in the DOM → not allowed
    # button_states[action] == False → button is enabled                → allowed
    dom_disabled = button_states.get(action) if action else None
    button_enabled: bool | None = None if dom_disabled is None else not dom_disabled

    result = dispatch(command, button_enabled=button_enabled)

    logger.info(
        "voice_dispatch: text=%r command=%r success=%s action=%s",
        text[:60], command, result["success"], result.get("action"),
    )

    return jsonify({**result, "command": command})


@voice_bp.route("/api/voice/commands", methods=["GET"])
def voice_commands_list():
    """Return the full command registry (useful for frontend reference)."""
    return jsonify({
        "wake_word": "yodha",
        "commands": [
            {
                "phrase": phrase,
                "action": entry["action"],
                "label": entry["label"],
                "default_enabled": entry["enabled"],
            }
            for phrase, entry in COMMANDS.items()
        ],
    })
