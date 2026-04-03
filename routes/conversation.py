"""
Conversation Blueprint

Routes:
  POST /transcribe_doctor_patient_conversation

Handles doctor-patient conversation recording with voice diarization
and optional translation.

Shared resources accessed via current_app.config:
  WHISPER_MODEL  — loaded Whisper model instance
  LLM_INSTANCE   — ChatOpenAI instance
  EMBEDDINGS     — OpenAIEmbeddings instance (reserved for future use)
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
import traceback
from datetime import datetime

from flask import Blueprint, current_app, jsonify, request

from utils.error_handlers import get_logger, handle_route_errors

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guard
# ---------------------------------------------------------------------------
try:
    from voice_diarization import get_diarization_processor  # type: ignore[import]

    DIARIZATION_AVAILABLE = True
    logger.info("Voice diarization module loaded successfully.")
except ImportError:
    DIARIZATION_AVAILABLE = False
    logger.warning(
        "Voice diarization not available. Install dependencies: pip install pyannote.audio torch"
    )

conversation_bp = Blueprint("conversation_bp", __name__)

# ---------------------------------------------------------------------------
# Resource accessors
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
# Route
# ---------------------------------------------------------------------------


@conversation_bp.route("/transcribe_doctor_patient_conversation", methods=["POST"])
@handle_route_errors
def transcribe_doctor_patient_conversation():
    """Handle doctor-patient conversation recording with voice diarization and optional translation."""
    if not DIARIZATION_AVAILABLE:
        return (
            jsonify(
                {
                    "error": "Voice diarization not available. Please install required dependencies."
                }
            ),
            500,
        )

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI

        audio_file = request.files["audio"]
        doctor_name = request.form.get("doctor_name", "Unknown Doctor")
        patient_name = request.form.get("patient_name", "Unknown Patient")
        language = request.form.get("language", "en")

        logger.info(
            "Processing conversation for %s and %s in language: %s",
            doctor_name,
            patient_name,
            language,
        )

        whisper_model = _get_whisper()
        llm = _get_llm()
        device = _get_device()

        # Save uploaded audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
            audio_file.save(temp_audio.name)
            temp_audio_path = temp_audio.name

        # Convert audio to WAV (mono 16 kHz) for Whisper
        try:
            from pydub import AudioSegment

            logger.info("Loading audio file: %s", temp_audio_path)
            audio = AudioSegment.from_file(temp_audio_path)
            audio = audio.set_channels(1).set_frame_rate(16000)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_file:
                audio.export(wav_file.name, format="wav")
                wav_audio_path = wav_file.name

            os.unlink(temp_audio_path)
            temp_audio_path = wav_audio_path
            logger.info("Audio converted to WAV: %s", temp_audio_path)

        except ImportError:
            try:
                import soundfile as sf

                audio_data, sample_rate = sf.read(temp_audio_path)
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1)
                if sample_rate != 16000:
                    from scipy import signal

                    num_samples = int(len(audio_data) * 16000 / sample_rate)
                    audio_data = signal.resample(audio_data, num_samples)
                    sample_rate = 16000

                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as converted:
                    sf.write(
                        converted.name,
                        audio_data,
                        sample_rate,
                        format="WAV",
                        subtype="PCM_16",
                    )
                    converted_path = converted.name

                os.unlink(temp_audio_path)
                temp_audio_path = converted_path
                logger.info("Audio converted using soundfile: %s", temp_audio_path)

            except Exception as exc:
                logger.warning("Audio conversion error: %s — using original format", exc)

        try:
            conversation_data: dict = {}

            if language == "es":
                logger.info(
                    "Processing Spanish conversation — using translate_audio approach"
                )

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
                whisper_language = language_map.get(language, "spanish")

                start_time = time.time()
                logger.info(
                    "Step 1/3: Starting fast transcription in %s...", whisper_language
                )

                result = whisper_model.transcribe(
                    temp_audio_path,
                    language=whisper_language,
                    fp16=device == "cuda",
                    beam_size=1,
                    best_of=1,
                    temperature=0.0,
                    condition_on_previous_text=False,
                )
                original_text = result["text"]
                transcription_time = time.time() - start_time
                logger.info(
                    "Step 1/3 completed in %.2fs. Text length: %d characters",
                    transcription_time,
                    len(original_text),
                )

                logger.info("Step 2/3: Starting AI segmentation and translation...")
                segmentation_start = time.time()

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
                    result_content = (
                        result_content.split("```json")[1].split("```")[0].strip()
                    )
                elif "```" in result_content:
                    result_content = result_content.split("```")[1].split("```")[0].strip()

                segments_data = json.loads(result_content)
                segments = segments_data.get("segments", [])

                for segment in segments:
                    if "translated_text" not in segment:
                        segment["translated_text"] = segment.get("text", "")

                segmentation_time = time.time() - segmentation_start
                logger.info(
                    "Step 2/3 completed in %.2fs. Processed %d segments",
                    segmentation_time,
                    len(segments),
                )

                logger.info("Step 3/3: Formatting conversation data...")
                format_start = time.time()

                transcript: list[dict] = []
                for i, segment in enumerate(segments):
                    role = segment.get("speaker", "unknown").lower()
                    if role == "doctor":
                        role = "Doctor"
                    elif role == "patient":
                        role = "Patient"
                    else:
                        role = "Unknown"
                    transcript.append(
                        {
                            "role": role,
                            "text": segment.get(
                                "translated_text", segment.get("text", "")
                            ),
                            "start": f"{i * 10}s",
                            "end": f"{(i + 1) * 10}s",
                            "confidence": 0.95,
                        }
                    )

                raw_transcript = "\n\n".join(
                    f"{seg.get('speaker', 'unknown').title()}: "
                    f"{seg.get('translated_text', seg.get('text', ''))}"
                    for seg in segments
                )
                original_transcript = "\n\n".join(
                    f"{seg.get('speaker', 'unknown').title()}: {seg.get('text', '')}"
                    for seg in segments
                )

                format_time = time.time() - format_start
                total_time = time.time() - start_time
                logger.info(
                    "Step 3/3 completed in %.2fs. Total: %.2fs "
                    "(Transcription: %.2fs, Segmentation: %.2fs, Formatting: %.2fs)",
                    format_time,
                    total_time,
                    transcription_time,
                    segmentation_time,
                    format_time,
                )

                conversation_data = {
                    "doctor_name": doctor_name,
                    "patient_name": patient_name,
                    "session_date": datetime.now().isoformat(),
                    "duration": "Unknown",
                    "total_segments": len(transcript),
                    "doctor_segments": sum(
                        1 for t in transcript if t.get("role", "").lower() == "doctor"
                    ),
                    "patient_segments": sum(
                        1 for t in transcript if t.get("role", "").lower() == "patient"
                    ),
                    "speakers_detected": 2,
                    "transcript": transcript,
                    "raw_transcript": raw_transcript,
                    "original_transcript": original_transcript,
                    "role_mapping": {"Doctor": doctor_name, "Patient": patient_name},
                    "language": language,
                    "translated": True,
                }

            else:
                logger.info("Using OpenAI-only processing (bypassing pyannote)")
                diarization_processor = get_diarization_processor()

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(
                    diarization_processor.process_doctor_patient_conversation_openai_only(
                        temp_audio_path
                    )
                )

                if result.get("error"):
                    return jsonify({"success": False, "error": result["error"]}), 500

                conversation_data = {
                    "doctor_name": doctor_name,
                    "patient_name": patient_name,
                    "session_date": datetime.now().isoformat(),
                    "duration": result.get("total_duration", "Unknown"),
                    "total_segments": result.get("total_segments", 0),
                    "doctor_segments": result.get("doctor_segments", 0),
                    "patient_segments": result.get("patient_segments", 0),
                    "speakers_detected": result.get("speakers_detected", 0),
                    "transcript": result.get("transcript", []),
                    "raw_transcript": result.get("raw_transcript", ""),
                    "original_transcript": result.get("raw_transcript", ""),
                    "role_mapping": result.get("role_mapping", {}),
                    "language": language,
                    "translated": False,
                }

            # Generate medical summary from conversation transcript
            full_transcript = conversation_data.get("raw_transcript", "")
            if full_transcript:
                try:
                    logger.info(
                        "Generating medical summary and conclusion for conversation..."
                    )

                    summary_prompt = f"""
                    As a medical AI assistant, please analyze the following doctor-patient conversation transcript and provide a professional medical summary.
                    
                    Doctor: {doctor_name}
                    Patient: {patient_name}
                    Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    
                    Conversation Transcript:
                    {full_transcript}
                    
                    Please provide:
                    1. A concise clinical summary highlighting key medical information, symptoms, findings, and discussions from the conversation
                    2. Professional conclusions with recommendations, follow-up actions, or treatment plans mentioned during the conversation
                    
                    Format your response exactly as:
                    SUMMARY:
                    [Provide a clear, professional summary of the medical conversation]
                    
                    CONCLUSION:
                    [Provide conclusions, recommendations, and any follow-up actions mentioned]
                    """

                    ai_response = llm.invoke(summary_prompt)
                    ai_content = (
                        ai_response.content.strip()
                        if hasattr(ai_response, "content")
                        else str(ai_response).strip()
                    )
                    logger.info(
                        "AI summary generated for conversation. Length: %d characters",
                        len(ai_content),
                    )

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
                            summary = (
                                "Medical conversation analysis completed. "
                                "Please refer to the full transcript for detailed information."
                            )
                            conclusion = (
                                "Further medical review and assessment may be required "
                                "based on the conversation content."
                            )

                    conversation_data["summary"] = summary
                    conversation_data["conclusion"] = conclusion

                except Exception as exc:
                    logger.error(
                        "Error generating summary for conversation: %s", exc
                    )
                    conversation_data["summary"] = (
                        "Summary generation was not available for this conversation."
                    )
                    conversation_data["conclusion"] = (
                        "Please review the conversation transcript for medical conclusions."
                    )
            else:
                conversation_data["summary"] = (
                    "No transcript available for summary generation."
                )
                conversation_data["conclusion"] = (
                    "Please ensure audio quality is sufficient for transcription."
                )

            return jsonify(
                {
                    "success": True,
                    "conversation_data": conversation_data,
                    "message": "Doctor-patient conversation processed successfully",
                }
            )

        finally:
            try:
                os.unlink(temp_audio_path)
            except Exception:
                pass

    except Exception as exc:
        logger.error("Error processing doctor-patient conversation: %s", exc)
        traceback.print_exc()
        return jsonify({"success": False, "error": str(exc)}), 500
