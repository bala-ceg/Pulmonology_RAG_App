"""
Audio Blueprint — /transcribe, /translate_audio, /transcribe_patient_notes

Shared resources are accessed via current_app.config:
  WHISPER_MODEL  — loaded Whisper model instance
  LLM_INSTANCE   — ChatOpenAI instance
"""

from __future__ import annotations

import json
import os
import tempfile
import traceback
from datetime import datetime

from flask import Blueprint, current_app, jsonify, request

from config import Config  # noqa: F401 – available for future use
from utils.error_handlers import get_logger, handle_route_errors

logger = get_logger(__name__)

audio_bp = Blueprint("audio_bp", __name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_whisper():
    return current_app.config.get("WHISPER_MODEL")


def _get_llm():
    return current_app.config.get("LLM_INSTANCE")


def _get_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@audio_bp.route("/transcribe", methods=["POST"])
@handle_route_errors
def transcribe():
    audio_file = request.files["audio"]
    whisper_model = _get_whisper()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
        audio_file.save(temp.name)
        result = whisper_model.transcribe(temp.name)
    return jsonify({"text": result["text"]})


@audio_bp.route("/translate_audio", methods=["POST"])
@handle_route_errors
def translate_audio():
    """
    Transcribe audio in selected language, identify speakers (doctor/patient),
    and translate to English.
    OPTIMIZED: Combined segmentation + translation in single LLM call.
    Expects: audio file, language code
    Returns: segmented conversation with speaker roles and translations
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    try:
        audio_file = request.files.get("audio")
        language_code = request.form.get("language", "es")

        if not audio_file:
            return jsonify({"error": "No audio file provided"}), 400

        language_map = {
            "es": "spanish",
            "zh": "chinese",
            "yue": "chinese",
            "tl": "tagalog",
            "hi": "hindi",
            "te": "telugu",
            "ta": "tamil",
            "gu": "gujarati",
            "pa": "punjabi",
        }
        whisper_language = language_map.get(language_code, "spanish")

        whisper_model = _get_whisper()
        device = _get_device()

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
                temp_path = temp.name
                audio_file.save(temp_path)
                logger.info("Translation audio file saved to: %s", temp_path)

                result = whisper_model.transcribe(
                    temp_path,
                    language=whisper_language,
                    fp16=device == "cuda",
                    beam_size=1,
                    best_of=1,
                    temperature=0.0,
                    condition_on_previous_text=False,
                )
                original_text = result["text"]
                logger.info(
                    "Fast transcription completed in %s. Length: %d characters",
                    whisper_language,
                    len(original_text),
                )
        except Exception as exc:
            logger.error("Error during transcription: %s", exc)
            return jsonify({"error": f"Transcription failed: {exc}"}), 500
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

        needs_translation = language_code != "en" and whisper_language != "english"
        llm = _get_llm()

        try:
            if needs_translation:
                combined_prompt = f"""Analyze this doctor-patient conversation in {whisper_language} and do TWO things:
1. Segment it by speaker (doctor vs patient) - EACH TURN/EXCHANGE should be a SEPARATE segment
2. Translate each segment to English

Transcript:
{original_text}

Provide your response in this exact JSON format:
{{
  "segments": [
    {{"speaker": "doctor", "text": "original text", "translated_text": "English translation"}},
    {{"speaker": "patient", "text": "original text", "translated_text": "English translation"}},
    {{"speaker": "doctor", "text": "original text", "translated_text": "English translation"}},
    {{"speaker": "patient", "text": "original text", "translated_text": "English translation"}}
  ]
}}

CRITICAL Rules:
- Split the conversation into INDIVIDUAL turns/exchanges - do NOT combine all doctor statements into one segment
- Each time the speaker changes, create a NEW segment
- Medical professionals use medical terminology, ask clinical questions, give advice
- Patients describe symptoms, ask questions about their health
- Provide accurate medical translations
- Be precise in segmentation - multiple back-and-forth exchanges should result in multiple segments"""
            else:
                combined_prompt = f"""Analyze this doctor-patient conversation and segment it by speaker.

Transcript:
{original_text}

Provide your response in this exact JSON format:
{{
  "segments": [
    {{"speaker": "doctor", "text": "the text spoken", "translated_text": "the text spoken"}},
    {{"speaker": "patient", "text": "the text spoken", "translated_text": "the text spoken"}}
  ]
}}

Rules:
- Medical professionals use medical terminology, ask clinical questions, give advice
- Patients describe symptoms, ask questions about their health"""

            messages = [
                SystemMessage(
                    content="You are an expert medical conversation analyzer and translator. Provide responses in valid JSON format only."
                ),
                HumanMessage(content=combined_prompt),
            ]

            fast_llm = ChatOpenAI(
                api_key=os.getenv("openai_api_key"),
                base_url=os.getenv("base_url"),
                model_name=os.getenv("llm_model_name"),
                temperature=0.3,
                max_tokens=2000,
                request_timeout=30,
            )

            response = fast_llm.invoke(messages)
            result_content = response.content.strip()

            if "```json" in result_content:
                result_content = result_content.split("```json")[1].split("```")[0].strip()
            elif "```" in result_content:
                result_content = result_content.split("```")[1].split("```")[0].strip()

            segments_data = json.loads(result_content)
            segments = segments_data.get("segments", [])

            for segment in segments:
                if "translated_text" not in segment:
                    segment["translated_text"] = segment.get("text", "")

            logger.info(
                "Completed: %d segments identified and translated in single call",
                len(segments),
            )
            return jsonify({"segments": segments, "has_segments": True})

        except json.JSONDecodeError as je:
            logger.warning("JSON parsing error: %s", je)

            if needs_translation:
                fallback_prompt = (
                    f"Translate this {whisper_language} medical conversation to English."
                    f" Provide only the translation.\n\n{original_text}"
                )
                messages = [
                    SystemMessage(content="You are a professional medical translator."),
                    HumanMessage(content=fallback_prompt),
                ]
                response = llm.invoke(messages)
                translated_text = response.content.strip()
            else:
                translated_text = original_text

            return jsonify(
                {
                    "original_text": original_text,
                    "translated_text": translated_text,
                    "has_segments": False,
                }
            )

        except Exception as exc:
            logger.error("Error during segmentation/translation: %s", exc)
            traceback.print_exc()

            if needs_translation:
                translation_prompt = (
                    f"Translate the following {whisper_language} text to English.\n"
                    f"Provide only the English translation.\n\nOriginal text:\n{original_text}"
                )
                messages = [
                    SystemMessage(content="You are a professional medical translator."),
                    HumanMessage(content=translation_prompt),
                ]
                response = llm.invoke(messages)
                translated_text = response.content.strip()
            else:
                translated_text = original_text

            return jsonify(
                {
                    "original_text": original_text,
                    "translated_text": translated_text,
                    "has_segments": False,
                }
            )

    except Exception as exc:
        logger.error("Unexpected error in translate_audio: %s", exc)
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@audio_bp.route("/transcribe_patient_notes", methods=["POST"])
@handle_route_errors
def transcribe_patient_notes():
    """
    Transcribe patient audio recording and generate medical summary.
    Expects: audio file, doctor_name, patient_name
    Returns: transcribed text, summary, and conclusion
    """
    try:
        audio_file = request.files.get("audio")
        doctor_name = request.form.get("doctor_name", "")
        patient_name = request.form.get("patient_name", "")

        logger.info(
            "Processing patient recording for: %s by Dr. %s", patient_name, doctor_name
        )

        if not audio_file:
            return jsonify({"error": "No audio file provided"}), 400

        if not doctor_name or not patient_name:
            return jsonify({"error": "Doctor name and patient name are required"}), 400

        whisper_model = _get_whisper()
        llm = _get_llm()

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
                temp_path = temp.name
                audio_file.save(temp_path)
                logger.info("Audio file saved to: %s", temp_path)

                result = whisper_model.transcribe(temp_path)
                transcribed_text = result["text"]
                logger.info(
                    "Transcription completed. Length: %d characters", len(transcribed_text)
                )
        except Exception as exc:
            logger.error("Error during transcription: %s", exc)
            return jsonify({"error": f"Transcription failed: {exc}"}), 500
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

        summary_prompt = f"""
        As a medical AI assistant, please analyze the following patient consultation transcript and provide a professional medical summary.
        
        Patient: {patient_name}
        Doctor: {doctor_name}
        Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Transcript:
        {transcribed_text}
        
        Please provide:
        1. A concise clinical summary highlighting key medical information, symptoms, findings, and discussions
        2. Professional conclusions with recommendations, follow-up actions, or treatment plans mentioned
        
        Format your response exactly as:
        SUMMARY:
        [Provide a clear, professional summary of the medical consultation]
        
        CONCLUSION:
        [Provide conclusions, recommendations, and any follow-up actions mentioned]
        """

        try:
            logger.info("Generating medical summary...")
            ai_response = llm.invoke(summary_prompt)
            ai_content = (
                ai_response.content.strip()
                if hasattr(ai_response, "content")
                else str(ai_response).strip()
            )
            logger.info("AI summary generated. Length: %d characters", len(ai_content))

            summary_parts = ai_content.split("CONCLUSION:")
            if len(summary_parts) == 2:
                summary = summary_parts[0].replace("SUMMARY:", "").strip()
                conclusion = summary_parts[1].strip()
            else:
                lines = ai_content.split("\n")
                summary_lines: list[str] = []
                conclusion_lines: list[str] = []
                in_conclusion = False

                for line in lines:
                    if "CONCLUSION" in line.upper():
                        in_conclusion = True
                        continue
                    elif "SUMMARY" in line.upper():
                        in_conclusion = False
                        continue
                    if in_conclusion:
                        conclusion_lines.append(line)
                    else:
                        summary_lines.append(line)

                summary = "\n".join(summary_lines).strip()
                conclusion = "\n".join(conclusion_lines).strip()

                if not summary and not conclusion:
                    summary = ai_content[: len(ai_content) // 2]
                    conclusion = ai_content[len(ai_content) // 2 :]

        except Exception as exc:
            logger.error("Error generating summary: %s", exc)
            return jsonify({"error": f"Summary generation failed: {exc}"}), 500

        response_data = {
            "success": True,
            "transcribed_text": transcribed_text,
            "summary": summary,
            "conclusion": conclusion,
            "doctor_name": doctor_name,
            "patient_name": patient_name,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info("Successfully processed patient recording for %s", patient_name)
        return jsonify(response_data)

    except Exception as exc:
        logger.error("Unexpected error in transcribe_patient_notes: %s", exc)
        traceback.print_exc()
        return jsonify({"error": f"An unexpected error occurred: {exc}"}), 500
