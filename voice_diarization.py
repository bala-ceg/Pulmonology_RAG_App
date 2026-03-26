"""
Voice diarization utilities for separating doctor and patient voices
Based on speaker diarization and OpenAI for role assignment
"""

import os
import json
import tempfile
import logging
from datetime import timedelta
from typing import List, Dict, Tuple
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceDiarization:
    def __init__(self):
        """Initialize voice diarization with required models"""
        self.openai_api_key = os.getenv('OPENAI_API_KEY') or os.getenv('openai_api_key')
        self.hf_token = os.getenv('HUGGINGFACE_TOKEN')
        
        if not self.openai_api_key:
            logger.warning("OpenAI API key not found. Voice role assignment may not work.")
        
        if not self.hf_token:
            logger.warning("Hugging Face token not found. Speaker diarization may not work.")
    
    def format_timedelta(self, td: float) -> str:
        """Format time duration for display"""
        return str(timedelta(seconds=int(td)))
    
    async def transcribe_with_whisper(self, audio_file_path: str) -> Dict:
        """
        Transcribe audio using Whisper
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            Dictionary containing transcription results
        """
        try:
            # Import whisper here to avoid import errors if not installed
            import whisper
            
            logger.info("Loading Whisper model...")
            whisper_model = whisper.load_model("small")
            
            logger.info("Transcribing audio...")
            whisper_result = whisper_model.transcribe(audio_file_path)
            
            logger.info("Whisper transcription completed")
            return whisper_result
            
        except ImportError:
            logger.error("Whisper not installed. Install with: pip install openai-whisper")
            return {"segments": [], "text": ""}
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            return {"segments": [], "text": ""}
    
    async def perform_diarization(self, audio_file_path: str) -> List[Dict]:
        """
        Perform speaker diarization using pyannote
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            List of diarization segments
        """
        try:
            # Import pyannote here to avoid import errors if not installed
            from pyannote.audio import Pipeline
            
            if not self.hf_token:
                logger.error("Hugging Face token not found. Please set HUGGINGFACE_TOKEN in .env file")
                return []
            
            logger.info("Loading diarization pipeline...")
            
            try:
                # Try to load the speaker diarization pipeline with auth token
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization@3.1",
                    use_auth_token=self.hf_token
                )
            except Exception as e:
                logger.warning(f"Failed to load speaker-diarization-3.1: {e}")
                logger.info("Trying fallback model...")
                try:
                    # Fallback to older version
                    pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization@2.1",
                        use_auth_token=self.hf_token
                    )
                except Exception as e2:
                    logger.error(f"Failed to load diarization models: {e2}")
                    logger.error("Please visit https://huggingface.co/pyannote/speaker-diarization and accept the user conditions")
                    logger.error("Also visit https://huggingface.co/pyannote/segmentation and accept the user conditions")
                    return []
            
            if pipeline is None:
                logger.error("Pipeline could not be loaded - please check your Hugging Face token and model access")
                return []
            
            logger.info("Performing speaker diarization...")
            diarization = pipeline(audio_file_path)
            
            # Convert diarization results to list format
            segments = []
            for segment in diarization.itersegments():
                segments.append({
                    "start": segment.start,
                    "end": segment.end,
                    "speaker": diarization.label(segment)
                })
            
            logger.info(f"Diarization completed. Found {len(segments)} segments")
            return segments
            
        except ImportError:
            logger.error("Pyannote not installed. Install with: pip install pyannote.audio")
            return []
        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            logger.error("Please ensure you have accepted the user conditions for pyannote models:")
            logger.error("1. Visit https://huggingface.co/pyannote/speaker-diarization")
            logger.error("2. Visit https://huggingface.co/pyannote/segmentation")
            logger.error("3. Click 'Agree and access repository' on both pages")
            return []
            return []
        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            return []
    
    def align_transcription_with_diarization(self, whisper_result: Dict, diarization_segments: List[Dict]) -> List[Dict]:
        """
        Align Whisper transcription with diarization segments
        
        Args:
            whisper_result: Results from Whisper transcription
            diarization_segments: Speaker diarization segments
            
        Returns:
            List of structured transcript segments
        """
        structured_transcript = []
        
        for dia_segment in diarization_segments:
            start, end = dia_segment["start"], dia_segment["end"]
            speaker = dia_segment["speaker"]
            
            # Find matching whisper segments
            matching_texts = []
            for whisper_seg in whisper_result.get("segments", []):
                w_start, w_end = whisper_seg["start"], whisper_seg["end"]
                
                # Check for overlap
                if (w_start >= start and w_start <= end) or (w_end >= start and w_end <= end) or (w_start <= start and w_end >= end):
                    matching_texts.append(whisper_seg["text"])
            
            # Combine matching texts
            combined_text = " ".join(matching_texts).strip()
            
            if combined_text:
                structured_transcript.append({
                    "speaker": speaker,
                    "start": self.format_timedelta(start),
                    "end": self.format_timedelta(end),
                    "start_seconds": start,
                    "end_seconds": end,
                    "text": combined_text
                })
        
        return structured_transcript
    
    async def segment_and_assign_roles_with_openai(self, whisper_result: Dict) -> List[Dict]:
        """
        Use OpenAI to intelligently segment the conversation and assign roles
        
        Args:
            whisper_result: Results from Whisper transcription
            
        Returns:
            List of conversation segments with roles assigned
        """
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=self.openai_api_key)
            
            # Get the full transcript text
            full_transcript = whisper_result.get("text", "")
            
            # Get individual segments for timing reference
            whisper_segments = whisper_result.get("segments", [])
            
            print(f"🤖 DEBUG: Full transcript length: {len(full_transcript)}")
            print(f"📝 DEBUG: Whisper segments count: {len(whisper_segments)}")
            
            # Create a detailed prompt for OpenAI to analyze and segment the conversation
            prompt = f"""
You are analyzing a medical conversation transcript between a Doctor and a Patient. Your task is to:

1. Identify speaker changes in the conversation
2. Segment the conversation into logical speaking turns
3. Assign each segment to either "Doctor" or "Patient"
4. Provide timing estimates based on the context

Guidelines for role identification:
- Doctors typically: ask questions, use medical terminology, give instructions, make diagnoses, suggest treatments
- Patients typically: describe symptoms, answer questions, express concerns, ask for clarification

Full Transcript:
{full_transcript}

Timing Reference (for context):
{json.dumps([{"start": seg["start"], "end": seg["end"], "text": seg["text"]} for seg in whisper_segments[:10]], indent=2)}

Please return a JSON array of conversation segments in this exact format:
[
  {{
    "role": "Doctor",
    "text": "How are you feeling today?",
    "start_seconds": 0.0,
    "end_seconds": 2.5,
    "confidence": 0.9
  }},
  {{
    "role": "Patient", 
    "text": "I've been having some chest pain.",
    "start_seconds": 3.0,
    "end_seconds": 6.0,
    "confidence": 0.8
  }}
]

Important:
- Split the conversation into natural speaking turns
- Assign realistic timing based on text length and pauses
- Use "Doctor" or "Patient" for roles
- Include confidence score (0.0-1.0) based on how certain you are about the role
- Ensure segments don't overlap and cover the full conversation
"""
            
            logger.info("Requesting intelligent segmentation and role assignment from OpenAI...")
            print("🔄 DEBUG: Sending request to OpenAI for segmentation...")
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an AI assistant that analyzes medical conversations. Return ONLY a valid JSON array without any markdown formatting, code blocks, or additional text. The response must start with '[' and end with ']'."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            response_text = response.choices[0].message.content.strip()
            print(f"📄 DEBUG: OpenAI response length: {len(response_text)}")
            print(f"🎯 DEBUG: OpenAI response preview: {response_text[:200]}...")
            
            # Extract JSON from markdown code blocks if present
            json_text = response_text
            if response_text.startswith("```json"):
                print("🔧 DEBUG: Detected markdown code blocks, extracting JSON...")
                # Remove ```json at the beginning and ``` at the end
                lines = response_text.split('\n')
                # Find the start and end of JSON content
                start_idx = 0
                end_idx = len(lines)
                
                for i, line in enumerate(lines):
                    if line.strip().startswith("```json"):
                        start_idx = i + 1
                    elif line.strip() == "```" and i > start_idx:
                        end_idx = i
                        break
                
                json_text = '\n'.join(lines[start_idx:end_idx]).strip()
                print(f"🔧 DEBUG: Extracted JSON text: {json_text[:100]}...")
            elif response_text.startswith("```"):
                print("🔧 DEBUG: Detected generic code blocks, extracting content...")
                # Handle generic ``` blocks
                lines = response_text.split('\n')
                if len(lines) > 2:
                    json_text = '\n'.join(lines[1:-1]).strip()
                    print(f"🔧 DEBUG: Extracted JSON text: {json_text[:100]}...")
            
            # Parse JSON response
            try:
                segments = json.loads(json_text)
                print(f"✅ DEBUG: Successfully parsed {len(segments)} segments")
                
                # Validate and enhance segments
                enhanced_segments = []
                for i, seg in enumerate(segments):
                    enhanced_seg = {
                        "speaker": f"Speaker_{1 if seg['role'] == 'Doctor' else 2}",  # For compatibility
                        "role": seg["role"],
                        "text": seg["text"],
                        "start": self.format_timedelta(seg["start_seconds"]),
                        "end": self.format_timedelta(seg["end_seconds"]),
                        "start_seconds": seg["start_seconds"],
                        "end_seconds": seg["end_seconds"],
                        "confidence": seg.get("confidence", 0.5),
                        "segment_id": i
                    }
                    enhanced_segments.append(enhanced_seg)
                    
                    if i < 3:  # Debug first few segments
                        print(f"🎬 DEBUG: Enhanced segment {i}: {enhanced_seg}")
                
                return enhanced_segments
                
            except json.JSONDecodeError as e:
                print(f"❌ DEBUG: JSON parsing failed: {e}")
                print(f"🔍 DEBUG: Response text that failed parsing: {repr(json_text[:500])}")
                logger.error(f"Failed to parse OpenAI JSON response: {e}")
                
                # Try a more aggressive extraction approach
                try:
                    # Look for array start and end in the response
                    start_bracket = json_text.find('[')
                    end_bracket = json_text.rfind(']')
                    
                    if start_bracket != -1 and end_bracket != -1 and end_bracket > start_bracket:
                        json_array = json_text[start_bracket:end_bracket + 1]
                        print(f"🔄 DEBUG: Attempting to parse extracted array: {json_array[:100]}...")
                        segments = json.loads(json_array)
                        
                        # Same enhancement logic
                        enhanced_segments = []
                        for i, seg in enumerate(segments):
                            enhanced_seg = {
                                "speaker": f"Speaker_{1 if seg['role'] == 'Doctor' else 2}",
                                "role": seg["role"],
                                "text": seg["text"],
                                "start": self.format_timedelta(seg["start_seconds"]),
                                "end": self.format_timedelta(seg["end_seconds"]),
                                "start_seconds": seg["start_seconds"],
                                "end_seconds": seg["end_seconds"],
                                "confidence": seg.get("confidence", 0.5),
                                "segment_id": i
                            }
                            enhanced_segments.append(enhanced_seg)
                        
                        print(f"✅ DEBUG: Successfully parsed with fallback method: {len(enhanced_segments)} segments")
                        return enhanced_segments
                        
                except Exception as fallback_error:
                    print(f"❌ DEBUG: Fallback parsing also failed: {fallback_error}")
                
                # If all parsing fails, create a simple fallback
                print("🔄 DEBUG: Creating simple fallback segmentation...")
                return self._create_simple_fallback_segments(full_transcript, whisper_segments)
                
        except ImportError:
            logger.error("OpenAI library not installed. Install with: pip install openai")
            return []
        except Exception as e:
            print(f"💥 DEBUG: Exception in segment_and_assign_roles_with_openai: {str(e)}")
            logger.error(f"OpenAI segmentation failed: {e}")
            return []

    def _create_simple_fallback_segments(self, full_transcript, whisper_segments):
        """Create simple alternating role segments when OpenAI parsing fails"""
        print("🔄 DEBUG: Creating simple fallback segments with alternating roles...")
        
        # Split transcript into sentences or logical breaks
        import re
        sentences = re.split(r'[.!?]+', full_transcript)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        segments = []
        current_time = 0.0
        time_per_char = sum(seg["end"] - seg["start"] for seg in whisper_segments) / len(full_transcript) if full_transcript else 0.1
        
        for i, text in enumerate(sentences):
            duration = len(text) * time_per_char
            role = "Doctor" if i % 2 == 0 else "Patient"
            
            segment = {
                "speaker": f"Speaker_{1 if role == 'Doctor' else 2}",
                "role": role,
                "text": text,
                "start": self.format_timedelta(current_time),
                "end": self.format_timedelta(current_time + duration),
                "start_seconds": current_time,
                "end_seconds": current_time + duration,
                "confidence": 0.3,  # Low confidence for fallback
                "segment_id": i
            }
            segments.append(segment)
            current_time += duration
            
        print(f"✅ DEBUG: Created {len(segments)} fallback segments")
        return segments
    
    async def assign_roles_with_openai(self, structured_transcript: List[Dict]) -> Dict[str, str]:
        """
        Use OpenAI to assign Doctor/Patient roles to speakers
        
        Args:
            structured_transcript: List of transcript segments with speakers
            
        Returns:
            Dictionary mapping speaker IDs to roles
        """
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=self.openai_api_key)
            
            # Create conversation text for analysis
            conversation_text = "\n".join([
                f"{seg['speaker']}: {seg['text']}" 
                for seg in structured_transcript
            ])
            
            prompt = f"""
You are analyzing a medical conversation transcript between a Doctor and a Patient.
Based on the language patterns, medical terminology usage, and conversation flow, 
determine which speaker is the Doctor and which is the Patient.

Guidelines:
- Doctors typically ask questions, use medical terminology, give instructions
- Patients typically describe symptoms, answer questions, express concerns
- Look for patterns like "How are you feeling?", "Any medications?", "I recommend..."

Return ONLY a JSON object with this exact format:
{{"Speaker_1": "Doctor", "Speaker_2": "Patient"}}
or
{{"Speaker_1": "Patient", "Speaker_2": "Doctor"}}

Transcript:
{conversation_text}
"""
            
            logger.info("Requesting role assignment from OpenAI...")
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an assistant that assigns roles in medical conversations. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            
            role_mapping_text = response.choices[0].message.content.strip()
            logger.info(f"OpenAI role mapping response: {role_mapping_text}")
            
            # Parse JSON response
            role_mapping = json.loads(role_mapping_text)
            
            return role_mapping
            
        except ImportError:
            logger.error("OpenAI library not installed. Install with: pip install openai")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse OpenAI JSON response: {e}")
            return {}
        except Exception as e:
            logger.error(f"Role assignment failed: {e}")
            return {}
    
    async def process_doctor_patient_conversation_openai_only(self, audio_file_path: str) -> Dict:
        """
        Streamlined pipeline using only Whisper + OpenAI (no pyannote)
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            Dictionary containing processed conversation data
        """
        try:
            print(f"🚀 DEBUG: Starting OpenAI-only conversation processing for file: {audio_file_path}")
            logger.info("Starting doctor-patient conversation processing (OpenAI-only mode)...")
            
            # Step 1: Transcription with Whisper
            print("🔄 DEBUG: Step 1 - Starting Whisper transcription...")
            whisper_result = await self.transcribe_with_whisper(audio_file_path)
            print(f"📝 DEBUG: Whisper result keys: {list(whisper_result.keys())}")
            print(f"📊 DEBUG: Number of whisper segments: {len(whisper_result.get('segments', []))}")
            
            if not whisper_result.get("segments"):
                print("❌ DEBUG: No transcription segments found in whisper result")
                logger.error("No transcription segments found")
                return {"error": "Transcription failed"}
            
            # Step 2: Use OpenAI for intelligent segmentation and role assignment
            print("🔄 DEBUG: Step 2 - Using OpenAI for segmentation and role assignment...")
            openai_segments = await self.segment_and_assign_roles_with_openai(whisper_result)
            print(f"🤖 DEBUG: OpenAI generated {len(openai_segments)} segments")
            
            if not openai_segments:
                print("⚠️ DEBUG: OpenAI segmentation failed, falling back to simple segments...")
                # Fallback: create simple segments from Whisper with basic role assignment
                openai_segments = []
                for i, seg in enumerate(whisper_result.get("segments", [])):
                    # Alternate between doctor and patient or use simple heuristics
                    role = "Doctor" if i % 2 == 0 else "Patient"
                    openai_segments.append({
                        "speaker": f"Speaker_{1 if role == 'Doctor' else 2}",
                        "role": role,
                        "text": seg["text"],
                        "start": self.format_timedelta(seg["start"]),
                        "end": self.format_timedelta(seg["end"]),
                        "start_seconds": seg["start"],
                        "end_seconds": seg["end"],
                        "confidence": 0.3,  # Low confidence for fallback
                        "segment_id": i
                    })
                print(f"📄 DEBUG: Created {len(openai_segments)} fallback segments")
            
            # Step 3: Create final transcript
            print("🔄 DEBUG: Step 3 - Creating final transcript...")
            final_transcript = []
            for seg in openai_segments:
                final_transcript.append({
                    "role": seg["role"],
                    "speaker_id": seg.get("speaker", "Speaker_1"),
                    "start": seg["start"],
                    "end": seg["end"],
                    "start_seconds": seg["start_seconds"],
                    "end_seconds": seg["end_seconds"],
                    "text": seg["text"],
                    "confidence": seg.get("confidence", 0.8)
                })
            
            print(f"📊 DEBUG: Final transcript total segments: {len(final_transcript)}")
            
            # Calculate conversation statistics
            total_duration = max([seg["end_seconds"] for seg in openai_segments]) if openai_segments else 0
            doctor_segments = [seg for seg in final_transcript if seg["role"] == "Doctor"]
            patient_segments = [seg for seg in final_transcript if seg["role"] == "Patient"]
            
            print(f"⏱️ DEBUG: Total duration: {total_duration}")
            print(f"👨‍⚕️ DEBUG: Doctor segments: {len(doctor_segments)}")
            print(f"🤒 DEBUG: Patient segments: {len(patient_segments)}")
            
            # Create role mapping for compatibility
            role_mapping = {}
            for seg in openai_segments:
                speaker = seg.get("speaker", "Speaker_1")
                role = seg["role"]
                if speaker not in role_mapping:
                    role_mapping[speaker] = role
            
            result = {
                "success": True,
                "total_duration": self.format_timedelta(total_duration),
                "total_duration_seconds": total_duration,
                "total_segments": len(final_transcript),
                "doctor_segments": len(doctor_segments),
                "patient_segments": len(patient_segments),
                "role_mapping": role_mapping,
                "transcript": final_transcript,
                "raw_transcript": whisper_result.get("text", ""),
                "speakers_detected": len(set([seg["speaker_id"] for seg in final_transcript])),
                "method": "openai_only"
            }
            
            print(f"✅ DEBUG: Result summary - Success: {result['success']}, Segments: {result['total_segments']}, Speakers: {result['speakers_detected']}")
            print(f"📄 DEBUG: Raw transcript preview: {result['raw_transcript'][:100]}...")
            logger.info("Doctor-patient conversation processing completed successfully (OpenAI-only)")
            return result
            
        except Exception as e:
            print(f"💥 DEBUG: Exception caught in process_doctor_patient_conversation_openai_only: {str(e)}")
            print(f"🔍 DEBUG: Exception type: {type(e).__name__}")
            import traceback
            print(f"📚 DEBUG: Full traceback:\n{traceback.format_exc()}")
            logger.error(f"Conversation processing failed: {e}")
            return {"error": str(e)}
    
    async def process_doctor_patient_conversation(self, audio_file_path: str) -> Dict:
        """
        Complete pipeline to process doctor-patient conversation
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            Dictionary containing processed conversation data
        """
        try:
            print(f"🚀 DEBUG: Starting conversation processing for file: {audio_file_path}")
            logger.info("Starting doctor-patient conversation processing...")
            
            # Step 1: Transcription with Whisper
            print("🔄 DEBUG: Step 1 - Starting Whisper transcription...")
            whisper_result = await self.transcribe_with_whisper(audio_file_path)
            print(f"📝 DEBUG: Whisper result keys: {list(whisper_result.keys())}")
            print(f"📊 DEBUG: Number of whisper segments: {len(whisper_result.get('segments', []))}")
            
            if not whisper_result.get("segments"):
                print("❌ DEBUG: No transcription segments found in whisper result")
                logger.error("No transcription segments found")
                return {"error": "Transcription failed"}
            
            # Step 2: Speaker diarization (with fallback)
            print("🔄 DEBUG: Step 2 - Starting speaker diarization...")
            diarization_segments = await self.perform_diarization(audio_file_path)
            print(f"🎯 DEBUG: Diarization segments count: {len(diarization_segments)}")
            print(f"🗣️ DEBUG: Diarization segments: {diarization_segments}")
            
            if not diarization_segments:
                print("⚠️ DEBUG: No diarization segments, creating fallback...")
                logger.warning("Speaker diarization failed, falling back to single speaker mode")
                # Create a fallback single speaker segment for the entire transcription
                total_duration = whisper_result.get("segments", [])
                print(f"📏 DEBUG: Total duration segments for fallback: {len(total_duration)}")
                if total_duration:
                    start_time = min([seg["start"] for seg in total_duration])
                    end_time = max([seg["end"] for seg in total_duration])
                    print(f"⏰ DEBUG: Fallback time range: {start_time} - {end_time}")
                    diarization_segments = [{
                        "start": start_time,
                        "end": end_time,
                        "speaker": "Speaker_1"
                    }]
                    print(f"✅ DEBUG: Created fallback segment: {diarization_segments}")
                else:
                    print("❌ DEBUG: No audio content detected for fallback")
                    return {"error": "No audio content detected"}
            
            # Step 3: Align transcription with diarization
            print("🔄 DEBUG: Step 3 - Aligning transcription with diarization...")
            structured_transcript = self.align_transcription_with_diarization(
                whisper_result, diarization_segments
            )
            print(f"📋 DEBUG: Structured transcript count: {len(structured_transcript)}")
            print(f"📝 DEBUG: First few structured segments: {structured_transcript[:2] if structured_transcript else 'None'}")
            
            if not structured_transcript:
                print("⚠️ DEBUG: No structured transcript, creating fallback from whisper segments...")
                logger.error("Failed to align transcription with diarization")
                # Fallback: create transcript without speaker separation
                structured_transcript = []
                for i, seg in enumerate(whisper_result.get("segments", [])):
                    structured_transcript.append({
                        "speaker": "Speaker_1",
                        "start": self.format_timedelta(seg["start"]),
                        "end": self.format_timedelta(seg["end"]),
                        "start_seconds": seg["start"],
                        "end_seconds": seg["end"],
                        "text": seg["text"]
                    })
                    if i < 2:  # Print first 2 for debugging
                        print(f"📄 DEBUG: Fallback segment {i}: {structured_transcript[-1]}")
            
            # Step 4: Assign roles (Doctor/Patient) only if we have meaningful content
            print("🔄 DEBUG: Step 4 - Assigning roles...")
            unique_speakers = set([seg["speaker"] for seg in structured_transcript])
            print(f"🎭 DEBUG: Unique speakers detected: {unique_speakers}")
            print(f"🔢 DEBUG: Number of unique speakers: {len(unique_speakers)}")
            
            role_mapping = {}
            if len(unique_speakers) > 1:
                # Multiple speakers detected, try role assignment
                print("👥 DEBUG: Multiple speakers detected, requesting OpenAI role assignment...")
                role_mapping = await self.assign_roles_with_openai(structured_transcript)
                print(f"🎯 DEBUG: OpenAI role mapping result: {role_mapping}")
            else:
                # Single speaker, assume it's mixed conversation
                print("👤 DEBUG: Single speaker detected, using mixed conversation label")
                logger.info("Single speaker detected, labeling as mixed conversation")
                role_mapping = {"Speaker_1": "Mixed (Doctor & Patient)"}
                print(f"🏷️ DEBUG: Single speaker role mapping: {role_mapping}")
            
            # Step 5: Create final transcript with roles
            print("🔄 DEBUG: Step 5 - Creating final transcript...")
            final_transcript = []
            for i, seg in enumerate(structured_transcript):
                role = role_mapping.get(seg["speaker"], seg["speaker"])
                final_transcript.append({
                    "role": role,
                    "speaker_id": seg["speaker"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "start_seconds": seg["start_seconds"],
                    "end_seconds": seg["end_seconds"],
                    "text": seg["text"]
                })
                if i < 2:  # Print first 2 for debugging
                    print(f"🎬 DEBUG: Final transcript segment {i}: {final_transcript[-1]}")
            
            print(f"📊 DEBUG: Final transcript total segments: {len(final_transcript)}")
            
            # Calculate conversation statistics
            total_duration = max([seg["end_seconds"] for seg in structured_transcript]) if structured_transcript else 0
            doctor_segments = [seg for seg in final_transcript if seg["role"] == "Doctor"]
            patient_segments = [seg for seg in final_transcript if seg["role"] == "Patient"]
            
            print(f"⏱️ DEBUG: Total duration: {total_duration}")
            print(f"👨‍⚕️ DEBUG: Doctor segments: {len(doctor_segments)}")
            print(f"🤒 DEBUG: Patient segments: {len(patient_segments)}")
            
            result = {
                "success": True,
                "total_duration": self.format_timedelta(total_duration),
                "total_duration_seconds": total_duration,
                "total_segments": len(final_transcript),
                "doctor_segments": len(doctor_segments),
                "patient_segments": len(patient_segments),
                "role_mapping": role_mapping,
                "transcript": final_transcript,
                "raw_transcript": whisper_result.get("text", ""),
                "speakers_detected": len(set([seg["speaker_id"] for seg in final_transcript]))
            }
            
            print(f"✅ DEBUG: Result summary - Success: {result['success']}, Segments: {result['total_segments']}, Speakers: {result['speakers_detected']}")
            print(f"📄 DEBUG: Raw transcript preview: {result['raw_transcript'][:100]}...")
            print(result)
            logger.info("Doctor-patient conversation processing completed successfully")
            return result
            
        except Exception as e:
            print(f"💥 DEBUG: Exception caught in process_doctor_patient_conversation: {str(e)}")
            print(f"🔍 DEBUG: Exception type: {type(e).__name__}")
            import traceback
            print(f"📚 DEBUG: Full traceback:\n{traceback.format_exc()}")
            logger.error(f"Conversation processing failed: {e}")
            return {"error": str(e)}

# Global diarization instance
diarization_processor = None

def get_diarization_processor():
    """Get or create diarization processor instance"""
    global diarization_processor
    if diarization_processor is None:
        diarization_processor = VoiceDiarization()
    return diarization_processor
