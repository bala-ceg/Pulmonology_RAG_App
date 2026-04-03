"""
Audio processing service.

Encapsulates all Whisper-based transcription and LLM-based translation
logic that lives in the /transcribe, /translate_audio, and
/transcribe_patient_notes routes of main.py.

The Whisper model is loaded **lazily** on first use so that importing this
module never triggers a GPU allocation at startup.
"""

from __future__ import annotations

import json
import os
import tempfile
import traceback
from typing import Any

from config import Config
from utils.error_handlers import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Language mapping
# ---------------------------------------------------------------------------

LANGUAGE_MAP: dict[str, str] = {
    "es": "spanish",
    "zh": "chinese",
    "yue": "chinese",   # Cantonese – Whisper uses the generic Chinese model
    "tl": "tagalog",
    "hi": "hindi",
    "te": "telugu",
    "ta": "tamil",
    "gu": "gujarati",
    "pa": "punjabi",
}


class AudioService:
    """Service for audio transcription and translation using OpenAI Whisper.

    All heavy dependencies (``torch``, ``whisper``) are imported on first
    access via the :py:attr:`model` property so the module is safe to import
    even when those packages are not installed.
    """

    def __init__(self) -> None:
        """Create the service without loading the Whisper model."""
        self._model: Any | None = None
        import torch
        self._device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def model(self) -> Any:
        """Return the Whisper model, loading it on first access."""
        if self._model is None:
            import whisper

            logger.info(
                "Loading Whisper model '%s' on %s",
                Config.WHISPER_MODEL,
                self._device,
            )
            self._model = whisper.load_model(Config.WHISPER_MODEL, device=self._device)
            logger.info("Whisper model loaded successfully")
        return self._model

    @property
    def device(self) -> str:
        """Return the compute device string (``'cuda'`` or ``'cpu'``)."""
        return self._device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe_audio(
        self,
        audio_file_path: str,
        language_code: str = "en",
    ) -> dict[str, str]:
        """Transcribe an audio file using Whisper with speed optimisations.

        Args:
            audio_file_path: Absolute or relative path to the audio file.
            language_code: BCP-47 language code (e.g. ``'es'``, ``'zh'``).
                Defaults to ``'en'`` (English).

        Returns:
            dict with keys:

            * ``transcription`` – the transcribed text.
            * ``language_code`` – the ``language_code`` argument echoed back.
            * ``whisper_language`` – the Whisper language name used internally.
        """
        whisper_language = LANGUAGE_MAP.get(language_code, language_code)

        logger.info(
            "Transcribing audio in '%s' (Whisper language: '%s')",
            language_code,
            whisper_language,
        )

        result = self.model.transcribe(
            audio_file_path,
            language=whisper_language,
            fp16=self.device == "cuda",
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
        )

        transcription: str = result["text"]
        logger.info("Transcription complete. Length: %d characters", len(transcription))

        return {
            "transcription": transcription,
            "language_code": language_code,
            "whisper_language": whisper_language,
        }

    def translate_audio(
        self,
        audio_file_path: str,
        language_code: str,
        llm: Any,
    ) -> dict[str, Any]:
        """Transcribe audio and segment/translate the conversation by speaker.

        Uses a single LLM call to both identify speakers (doctor vs. patient)
        *and* translate each turn to English, matching the optimised logic in
        the ``/translate_audio`` route.  Falls back to plain translation when
        the LLM returns invalid JSON.

        Args:
            audio_file_path: Path to the audio file.  The caller is
                responsible for cleanup.
            language_code: BCP-47 code of the spoken language (e.g. ``'es'``).
            llm: LangChain LLM instance used only in the fallback translation
                path when JSON parsing fails.

        Returns:
            On success (structured segments)::

                {
                    "segments": [
                        {"speaker": "doctor", "text": "...", "translated_text": "..."},
                        ...
                    ],
                    "has_segments": True
                }

            On fallback (plain translation)::

                {
                    "original_text": "...",
                    "translated_text": "...",
                    "has_segments": False
                }
        """
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI

        whisper_language = LANGUAGE_MAP.get(language_code, "spanish")

        # ----------------------------------------------------------------
        # Step 1 – Transcription
        # ----------------------------------------------------------------
        logger.info(
            "Starting fast transcription in '%s' from: %s",
            whisper_language,
            audio_file_path,
        )

        result = self.model.transcribe(
            audio_file_path,
            language=whisper_language,
            fp16=self.device == "cuda",
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
        )
        original_text: str = result["text"]
        logger.info(
            "Transcription completed in %s. Length: %d characters",
            whisper_language,
            len(original_text),
        )

        needs_translation = language_code != "en" and whisper_language != "english"

        # ----------------------------------------------------------------
        # Step 2 – Segmentation + optional translation (single LLM call)
        # ----------------------------------------------------------------
        result_content: str = ""
        try:
            if needs_translation:
                combined_prompt = (
                    f"Analyze this doctor-patient conversation in {whisper_language} and do TWO things:\n"
                    "1. Segment it by speaker (doctor vs patient) - EACH TURN/EXCHANGE should be a SEPARATE segment\n"
                    "2. Translate each segment to English\n\n"
                    f"Transcript:\n{original_text}\n\n"
                    "Provide your response in this exact JSON format:\n"
                    "{\n"
                    '  "segments": [\n'
                    '    {"speaker": "doctor", "text": "original text", "translated_text": "English translation"},\n'
                    '    {"speaker": "patient", "text": "original text", "translated_text": "English translation"},\n'
                    '    {"speaker": "doctor", "text": "original text", "translated_text": "English translation"},\n'
                    '    {"speaker": "patient", "text": "original text", "translated_text": "English translation"}\n'
                    "  ]\n"
                    "}\n\n"
                    "CRITICAL Rules:\n"
                    "- Split the conversation into INDIVIDUAL turns/exchanges - do NOT combine all doctor statements into one segment\n"
                    "- Each time the speaker changes, create a NEW segment\n"
                    "- Medical professionals use medical terminology, ask clinical questions, give advice\n"
                    "- Patients describe symptoms, ask questions about their health\n"
                    "- Provide accurate medical translations\n"
                    "- Be precise in segmentation - multiple back-and-forth exchanges should result in multiple segments"
                )
            else:
                combined_prompt = (
                    "Analyze this doctor-patient conversation and segment it by speaker.\n\n"
                    f"Transcript:\n{original_text}\n\n"
                    "Provide your response in this exact JSON format:\n"
                    "{\n"
                    '  "segments": [\n'
                    '    {"speaker": "doctor", "text": "the text spoken", "translated_text": "the text spoken"},\n'
                    '    {"speaker": "patient", "text": "the text spoken", "translated_text": "the text spoken"}\n'
                    "  ]\n"
                    "}\n\n"
                    "Rules:\n"
                    "- Medical professionals use medical terminology, ask clinical questions, give advice\n"
                    "- Patients describe symptoms, ask questions about their health"
                )

            messages = [
                SystemMessage(
                    content=(
                        "You are an expert medical conversation analyzer and translator. "
                        "Provide responses in valid JSON format only."
                    )
                ),
                HumanMessage(content=combined_prompt),
            ]

            fast_llm = ChatOpenAI(
                api_key=Config.OPENAI_API_KEY,
                base_url=Config.OPENAI_BASE_URL,
                model_name=Config.LLM_MODEL_NAME,
                temperature=0.3,
                max_tokens=2000,
                request_timeout=Config.LLM_REQUEST_TIMEOUT,
            )

            response = fast_llm.invoke(messages)
            result_content = response.content.strip()

            # Strip markdown code fences if present
            if "```json" in result_content:
                result_content = result_content.split("```json")[1].split("```")[0].strip()
            elif "```" in result_content:
                result_content = result_content.split("```")[1].split("```")[0].strip()

            segments_data = json.loads(result_content)
            segments: list[dict] = segments_data.get("segments", [])

            for segment in segments:
                if "translated_text" not in segment:
                    segment["translated_text"] = segment.get("text", "")

            logger.info(
                "Completed: %d segments identified and translated in single call",
                len(segments),
            )
            return {"segments": segments, "has_segments": True}

        except json.JSONDecodeError as je:
            logger.warning(
                "JSON parsing error: %s. Raw response: %s",
                je,
                result_content or "N/A",
            )
            # Fallback – plain translation
            if needs_translation:
                fallback_prompt = (
                    f"Translate this {whisper_language} medical conversation to English. "
                    f"Provide only the translation.\n\n{original_text}"
                )
                fb_messages = [
                    SystemMessage(content="You are a professional medical translator."),
                    HumanMessage(content=fallback_prompt),
                ]
                fb_response = llm.invoke(fb_messages)
                translated_text = fb_response.content.strip()
            else:
                translated_text = original_text

            return {
                "original_text": original_text,
                "translated_text": translated_text,
                "has_segments": False,
            }

        except Exception as exc:
            logger.error(
                "Error during segmentation/translation: %s\n%s",
                exc,
                traceback.format_exc(),
            )
            # Fallback – plain translation
            if needs_translation:
                translation_prompt = (
                    f"Translate the following {whisper_language} text to English.\n"
                    f"Provide only the English translation.\n\n"
                    f"Original text:\n{original_text}"
                )
                fb_messages = [
                    SystemMessage(content="You are a professional medical translator."),
                    HumanMessage(content=translation_prompt),
                ]
                fb_response = llm.invoke(fb_messages)
                translated_text = fb_response.content.strip()
            else:
                translated_text = original_text

            return {
                "original_text": original_text,
                "translated_text": translated_text,
                "has_segments": False,
            }

    def transcribe_patient_notes(self, audio_file_path: str) -> dict[str, str]:
        """Transcribe an English patient notes recording.

        Uses the default Whisper settings (no language hint, no speed
        optimisations) to maximise accuracy for patient-recorded audio.

        Args:
            audio_file_path: Path to the audio WAV file.  The caller is
                responsible for cleanup.

        Returns:
            dict with key ``transcription`` containing the transcribed text.
        """
        logger.info("Transcribing patient notes from: %s", audio_file_path)

        result = self.model.transcribe(audio_file_path)
        transcribed_text: str = result["text"]

        logger.info(
            "Patient notes transcription complete. Length: %d characters",
            len(transcribed_text),
        )
        return {"transcription": transcribed_text}


# ---------------------------------------------------------------------------
# Module-level singleton (lazy – Whisper model loaded on first method call)
# ---------------------------------------------------------------------------
audio_service = AudioService()
