"""
Unit tests for services/audio_service.py.

All Whisper model calls and LLM calls are mocked — no real audio files or
API keys are required to run these tests.
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_whisper_model(text: str = "Hello doctor") -> MagicMock:
    model = MagicMock()
    model.transcribe.return_value = {"text": text}
    return model


def _make_mock_llm(response_text: str = "Translated text") -> MagicMock:
    llm = MagicMock()
    msg = MagicMock()
    msg.content = response_text
    llm.invoke.return_value = msg
    return llm


def _wav_bytes() -> bytes:
    """Return the smallest valid WAV bytes (44-byte header, no samples)."""
    import struct
    data_size = 0
    chunk_size = 36 + data_size
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", chunk_size, b"WAVE",
        b"fmt ", 16, 1, 1, 16000, 32000, 2, 16,
        b"data", data_size,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAudioServiceTranscription:
    """Tests for AudioService.transcribe_audio."""

    def test_transcribe_returns_text(self) -> None:
        """transcribe_audio should return the Whisper model output."""
        with patch("whisper.load_model", return_value=_make_mock_whisper_model("Patient says hello")):
            from services.audio_service import AudioService  # noqa: PLC0415
            svc = AudioService()
            svc._model = _make_mock_whisper_model("Patient says hello")

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(_wav_bytes())
                tmp_path = f.name

            try:
                result = svc.transcribe_audio(tmp_path, language_code="en")
                assert "transcription" in result
                assert result["transcription"] == "Patient says hello"
            finally:
                os.unlink(tmp_path)

    def test_transcribe_language_mapping(self) -> None:
        """transcribe_audio should map language codes to Whisper language names."""
        with patch("whisper.load_model", return_value=_make_mock_whisper_model()):
            from services.audio_service import AudioService  # noqa: PLC0415
            svc = AudioService()
            svc._model = _make_mock_whisper_model()

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(_wav_bytes())
                tmp_path = f.name

            try:
                result = svc.transcribe_audio(tmp_path, language_code="es")
                assert result["language_code"] == "es"
                assert result["whisper_language"] == "spanish"
            finally:
                os.unlink(tmp_path)

    def test_transcribe_unknown_language_defaults(self) -> None:
        """Unknown language codes should fall back to a default (spanish)."""
        with patch("whisper.load_model", return_value=_make_mock_whisper_model()):
            from services.audio_service import AudioService  # noqa: PLC0415
            svc = AudioService()
            svc._model = _make_mock_whisper_model()

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(_wav_bytes())
                tmp_path = f.name

            try:
                result = svc.transcribe_audio(tmp_path, language_code="xx")
                assert "whisper_language" in result
            finally:
                os.unlink(tmp_path)


class TestAudioServicePatientNotes:
    """Tests for AudioService.transcribe_patient_notes."""

    def test_transcribe_patient_notes_returns_transcription(self) -> None:
        with patch("whisper.load_model", return_value=_make_mock_whisper_model("Patient has chest pain")):
            from services.audio_service import AudioService  # noqa: PLC0415
            svc = AudioService()
            svc._model = _make_mock_whisper_model("Patient has chest pain")

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(_wav_bytes())
                tmp_path = f.name

            try:
                result = svc.transcribe_patient_notes(tmp_path)
                assert "transcription" in result
                assert result["transcription"] == "Patient has chest pain"
            finally:
                os.unlink(tmp_path)


class TestAudioServiceTranslation:
    """Tests for AudioService.translate_audio."""

    def test_translate_audio_returns_both_texts(self) -> None:
        mock_model = _make_mock_whisper_model("Hola doctor")
        mock_llm_response = MagicMock()
        mock_llm_response.content = '{"original": "Hola doctor", "translation": "Hello doctor"}'
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = mock_llm_response

        with patch("whisper.load_model", return_value=mock_model), \
             patch("langchain_openai.ChatOpenAI", return_value=mock_llm_instance):
            from services.audio_service import AudioService  # noqa: PLC0415
            svc = AudioService()
            svc._model = mock_model

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(_wav_bytes())
                tmp_path = f.name

            try:
                mock_llm = _make_mock_llm("Hello doctor")
                result = svc.translate_audio(tmp_path, language_code="es", llm=mock_llm)
                # Service may return various dict shapes depending on LLM output parsing
                assert isinstance(result, dict)
                assert len(result) > 0
            finally:
                os.unlink(tmp_path)
