"""
Test script for multilingual audio transcription
Tests Hindi, Tamil, Telugu, Malayalam, Kannada, and other Indian languages
"""

import requests
import json
import os
import sys
from pathlib import Path

# Base URL for the API
BASE_URL = "http://localhost:5000"

def print_separator(title=""):
    """Print a formatted separator"""
    print("\n" + "=" * 80)
    if title:
        print(f" {title}")
        print("=" * 80)
    print()

def test_supported_languages():
    """Test getting supported languages"""
    print_separator("Testing Supported Languages API")
    
    url = f"{BASE_URL}/api/supported-languages"
    
    try:
        response = requests.get(url)
        result = response.json()
        
        if result.get("success"):
            print("✅ Supported Languages Retrieved Successfully")
            print("\n📋 Supported Languages:")
            
            for lang_name, lang_code in result["supported_languages"].items():
                print(f"   - {lang_name.title()}: {lang_code}")
            
            print("\n🌏 Regional Languages:")
            for region, data in result.get("language_regions", {}).items():
                print(f"\n   {data['name']}:")
                for lang in data['languages']:
                    print(f"      • {lang}")
            
            print(f"\n💡 Recommendation: {result.get('recommended_model', 'N/A')}")
            return True
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def test_hindi_conversation(audio_file_path=None):
    """Test Hindi conversation transcription"""
    print_separator("Testing Hindi (हिन्दी) Conversation")
    
    if not audio_file_path:
        print("⚠️  No audio file provided. Using sample file path...")
        audio_file_path = "test_audio/hindi_conversation.wav"
    
    if not os.path.exists(audio_file_path):
        print(f"❌ Audio file not found: {audio_file_path}")
        print("📝 To test, provide a Hindi audio file path")
        return False
    
    url = f"{BASE_URL}/api/transcribe-multilingual"
    
    try:
        with open(audio_file_path, 'rb') as audio_file:
            files = {'audio': audio_file}
            data = {'language': 'hi'}  # Hindi
            
            print(f"🎤 Processing audio file: {audio_file_path}")
            print("🔄 Sending request to API...")
            
            response = requests.post(url, files=files, data=data)
            result = response.json()
            
            if result.get("success"):
                print("✅ Hindi Transcription Successful")
                print(f"\n🗣️  Detected Language: {result.get('language_name', 'N/A')} ({result.get('detected_language', 'N/A')})")
                print(f"📊 Confidence: {result.get('language_probability', 0) * 100:.1f}%")
                print(f"⏱️  Total Duration: {result.get('total_duration', 'N/A')}")
                print(f"👥 Speakers Detected: {result.get('speakers_detected', 0)}")
                print(f"📝 Total Segments: {result.get('total_segments', 0)}")
                
                print("\n📄 Transcript Preview:")
                raw_text = result.get('raw_transcript', '')
                print(raw_text[:500] + "..." if len(raw_text) > 500 else raw_text)
                
                print("\n💬 Conversation Segments:")
                for i, segment in enumerate(result.get('transcript', [])[:3], 1):
                    print(f"\n   Segment {i}:")
                    print(f"   Role: {segment.get('role', 'Unknown')}")
                    print(f"   Time: {segment.get('start', '')} - {segment.get('end', '')}")
                    print(f"   Text: {segment.get('text', '')[:200]}")
                
                return True
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
                return False
                
    except Exception as e:
        print(f"❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tamil_conversation(audio_file_path=None):
    """Test Tamil conversation transcription"""
    print_separator("Testing Tamil (தமிழ்) Conversation")
    
    if not audio_file_path:
        print("⚠️  No audio file provided. Using sample file path...")
        audio_file_path = "test_audio/tamil_conversation.wav"
    
    if not os.path.exists(audio_file_path):
        print(f"❌ Audio file not found: {audio_file_path}")
        print("📝 To test, provide a Tamil audio file path")
        return False
    
    url = f"{BASE_URL}/api/transcribe-multilingual"
    
    try:
        with open(audio_file_path, 'rb') as audio_file:
            files = {'audio': audio_file}
            data = {'language': 'ta'}  # Tamil
            
            print(f"🎤 Processing audio file: {audio_file_path}")
            response = requests.post(url, files=files, data=data)
            result = response.json()
            
            if result.get("success"):
                print("✅ Tamil Transcription Successful")
                display_transcription_results(result)
                return True
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
                return False
                
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def test_telugu_conversation(audio_file_path=None):
    """Test Telugu conversation transcription"""
    print_separator("Testing Telugu (తెలుగు) Conversation")
    
    if not audio_file_path:
        audio_file_path = "test_audio/telugu_conversation.wav"
    
    if not os.path.exists(audio_file_path):
        print(f"❌ Audio file not found: {audio_file_path}")
        print("📝 To test, provide a Telugu audio file path")
        return False
    
    return test_language(audio_file_path, 'te', 'Telugu')

def test_malayalam_conversation(audio_file_path=None):
    """Test Malayalam conversation transcription"""
    print_separator("Testing Malayalam (മലയാളം) Conversation")
    
    if not audio_file_path:
        audio_file_path = "test_audio/malayalam_conversation.wav"
    
    if not os.path.exists(audio_file_path):
        print(f"❌ Audio file not found: {audio_file_path}")
        print("📝 To test, provide a Malayalam audio file path")
        return False
    
    return test_language(audio_file_path, 'ml', 'Malayalam')

def test_kannada_conversation(audio_file_path=None):
    """Test Kannada conversation transcription"""
    print_separator("Testing Kannada (ಕನ್ನಡ) Conversation")
    
    if not audio_file_path:
        audio_file_path = "test_audio/kannada_conversation.wav"
    
    if not os.path.exists(audio_file_path):
        print(f"❌ Audio file not found: {audio_file_path}")
        print("📝 To test, provide a Kannada audio file path")
        return False
    
    return test_language(audio_file_path, 'kn', 'Kannada')

def test_language(audio_file_path, language_code, language_name):
    """Generic language test function"""
    url = f"{BASE_URL}/api/transcribe-multilingual"
    
    try:
        with open(audio_file_path, 'rb') as audio_file:
            files = {'audio': audio_file}
            data = {'language': language_code}
            
            print(f"🎤 Processing audio file: {audio_file_path}")
            response = requests.post(url, files=files, data=data)
            result = response.json()
            
            if result.get("success"):
                print(f"✅ {language_name} Transcription Successful")
                display_transcription_results(result)
                return True
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
                return False
                
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def test_auto_detect_language(audio_file_path):
    """Test automatic language detection"""
    print_separator("Testing Automatic Language Detection")
    
    if not os.path.exists(audio_file_path):
        print(f"❌ Audio file not found: {audio_file_path}")
        return False
    
    url = f"{BASE_URL}/api/transcribe-multilingual"
    
    try:
        with open(audio_file_path, 'rb') as audio_file:
            files = {'audio': audio_file}
            data = {}  # No language specified - auto-detect
            
            print(f"🎤 Processing audio file: {audio_file_path}")
            print("🔍 Auto-detecting language...")
            
            response = requests.post(url, files=files, data=data)
            result = response.json()
            
            if result.get("success"):
                print("✅ Auto-Detection Successful")
                display_transcription_results(result)
                return True
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
                return False
                
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def display_transcription_results(result):
    """Display transcription results in a formatted way"""
    print(f"\n🗣️  Detected Language: {result.get('language_name', 'N/A')} ({result.get('detected_language', 'N/A')})")
    print(f"📊 Confidence: {result.get('language_probability', 0) * 100:.1f}%")
    print(f"⏱️  Total Duration: {result.get('total_duration', 'N/A')}")
    print(f"👥 Speakers Detected: {result.get('speakers_detected', 0)}")
    print(f"📝 Total Segments: {result.get('total_segments', 0)}")
    
    # Role mapping
    role_mapping = result.get('role_mapping', {})
    if role_mapping:
        print("\n🎭 Role Mapping:")
        for speaker, role in role_mapping.items():
            print(f"   {speaker} → {role}")
    
    print("\n📄 Transcript Preview:")
    raw_text = result.get('raw_transcript', '')
    print(raw_text[:500] + "..." if len(raw_text) > 500 else raw_text)
    
    print("\n💬 Conversation Segments (First 3):")
    for i, segment in enumerate(result.get('transcript', [])[:3], 1):
        print(f"\n   Segment {i}:")
        print(f"   Role: {segment.get('role', 'Unknown')}")
        print(f"   Time: {segment.get('start', '')} - {segment.get('end', '')}")
        print(f"   Text: {segment.get('text', '')[:200]}")

def save_results_to_file(result, filename="transcription_result.json"):
    """Save transcription results to a JSON file"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Results saved to: {filename}")
        return True
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        return False

def main():
    """Main test function"""
    print_separator("Multilingual Audio Transcription Test Suite")
    print("Testing support for Hindi, Tamil, Telugu, Malayalam, Kannada, and more")
    print("\nMake sure the Flask server is running at http://localhost:5000")
    print("\nPress Ctrl+C to exit at any time")
    
    # Test 1: Get supported languages
    print("\n" + "-" * 80)
    test_supported_languages()
    
    # Interactive menu
    while True:
        print("\n" + "-" * 80)
        print("\nSelect a test to run:")
        print("1. Test Hindi (हिन्दी) conversation")
        print("2. Test Tamil (தமிழ்) conversation")
        print("3. Test Telugu (తెలుగు) conversation")
        print("4. Test Malayalam (മലയാളം) conversation")
        print("5. Test Kannada (ಕನ್ನಡ) conversation")
        print("6. Test Auto-detect language")
        print("7. Test with custom audio file")
        print("8. Exit")
        
        try:
            choice = input("\nEnter your choice (1-8): ").strip()
            
            if choice == '1':
                audio_path = input("Enter Hindi audio file path (or press Enter for default): ").strip()
                test_hindi_conversation(audio_path if audio_path else None)
            
            elif choice == '2':
                audio_path = input("Enter Tamil audio file path (or press Enter for default): ").strip()
                test_tamil_conversation(audio_path if audio_path else None)
            
            elif choice == '3':
                audio_path = input("Enter Telugu audio file path (or press Enter for default): ").strip()
                test_telugu_conversation(audio_path if audio_path else None)
            
            elif choice == '4':
                audio_path = input("Enter Malayalam audio file path (or press Enter for default): ").strip()
                test_malayalam_conversation(audio_path if audio_path else None)
            
            elif choice == '5':
                audio_path = input("Enter Kannada audio file path (or press Enter for default): ").strip()
                test_kannada_conversation(audio_path if audio_path else None)
            
            elif choice == '6':
                audio_path = input("Enter audio file path for auto-detection: ").strip()
                if audio_path:
                    test_auto_detect_language(audio_path)
                else:
                    print("❌ Audio file path required for auto-detection")
            
            elif choice == '7':
                audio_path = input("Enter audio file path: ").strip()
                lang_code = input("Enter language code (or press Enter for auto-detect): ").strip()
                
                if audio_path:
                    if lang_code:
                        test_language(audio_path, lang_code, f"Language ({lang_code})")
                    else:
                        test_auto_detect_language(audio_path)
                else:
                    print("❌ Audio file path required")
            
            elif choice == '8':
                print("\n👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice. Please select 1-8.")
        
        except KeyboardInterrupt:
            print("\n\n👋 Test interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
