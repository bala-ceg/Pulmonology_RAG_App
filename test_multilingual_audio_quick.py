"""
Quick Test Script for Multilingual Audio Transcription
Simple tests for Hindi, Tamil, Telugu, Malayalam, and Kannada
"""

import requests
import sys

BASE_URL = "http://localhost:5000"

def quick_test_supported_languages():
    """Quick test to check supported languages"""
    print("\n" + "="*60)
    print("Testing Supported Languages API")
    print("="*60 + "\n")
    
    try:
        response = requests.get(f"{BASE_URL}/api/supported-languages")
        result = response.json()
        
        if result.get("success"):
            print("✅ API is working!")
            print("\n📋 Supported Languages:")
            for lang_name, lang_code in result["supported_languages"].items():
                print(f"   • {lang_name.title()}: {lang_code}")
            return True
        else:
            print(f"❌ Error: {result.get('error')}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Is it running at http://localhost:5000?")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def quick_test_with_file(audio_file_path, language_code=None):
    """Quick test with an audio file"""
    print("\n" + "="*60)
    print(f"Testing Audio Transcription")
    print("="*60 + "\n")
    
    try:
        with open(audio_file_path, 'rb') as f:
            files = {'audio': f}
            data = {}
            
            if language_code:
                data['language'] = language_code
                print(f"🗣️  Language: {language_code}")
            else:
                print("🗣️  Language: Auto-detect")
            
            print(f"📁 File: {audio_file_path}")
            print("⏳ Processing...")
            
            response = requests.post(
                f"{BASE_URL}/api/transcribe-multilingual",
                files=files,
                data=data
            )
            
            result = response.json()
            
            if result.get("success"):
                print("\n✅ Transcription Successful!")
                print(f"\n🗣️  Detected: {result.get('language_name', 'Unknown')} ({result.get('detected_language', 'N/A')})")
                print(f"📊 Confidence: {result.get('language_probability', 0) * 100:.1f}%")
                print(f"⏱️  Duration: {result.get('total_duration', 'N/A')}")
                print(f"👥 Speakers: {result.get('speakers_detected', 0)}")
                print(f"📝 Segments: {result.get('total_segments', 0)}")
                
                print("\n📄 Transcript:")
                print(result.get('raw_transcript', 'No transcript')[:300])
                
                return True
            else:
                print(f"\n❌ Error: {result.get('error')}")
                return False
                
    except FileNotFoundError:
        print(f"❌ File not found: {audio_file_path}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("Multilingual Audio Transcription - Quick Test")
    print("="*60)
    
    # Test 1: Check if API is available
    if not quick_test_supported_languages():
        print("\n⚠️  Server may not be running. Start it with: python main.py")
        sys.exit(1)
    
    # Test 2: Test with audio file if provided
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        language = sys.argv[2] if len(sys.argv) > 2 else None
        quick_test_with_file(audio_file, language)
    else:
        print("\n" + "="*60)
        print("Usage Instructions")
        print("="*60)
        print("\nTo test with an audio file:")
        print("  python test_multilingual_audio_quick.py <audio_file> [language_code]")
        print("\nExamples:")
        print("  python test_multilingual_audio_quick.py recording.wav")
        print("  python test_multilingual_audio_quick.py recording.wav hi")
        print("  python test_multilingual_audio_quick.py recording.wav ta")
        print("\nSupported language codes:")
        print("  hi - Hindi, ta - Tamil, te - Telugu, ml - Malayalam, kn - Kannada")
        print("  en - English, bn - Bengali, mr - Marathi, gu - Gujarati, pa - Punjabi")

if __name__ == "__main__":
    main()
