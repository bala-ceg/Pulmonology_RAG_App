#!/usr/bin/env python3
"""
Script to test and authenticate with Hugging Face pyannote models
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_huggingface_auth():
    """Test Hugging Face authentication for pyannote models"""
    
    # Get token from environment
    hf_token = os.getenv('HUGGINGFACE_TOKEN')
    
    if not hf_token:
        print("❌ HUGGINGFACE_TOKEN not found in .env file")
        return False
    
    print(f"✅ Found Hugging Face token: {hf_token[:10]}...")
    
    try:
        from pyannote.audio import Pipeline
        print("✅ pyannote.audio is installed")
        
        # Test authentication with speaker diarization model
        print("\n🔄 Testing speaker diarization model access...")
        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
            print("✅ Successfully loaded pyannote/speaker-diarization-3.1")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load speaker-diarization-3.1: {e}")
            
            # Try fallback model
            print("\n🔄 Testing fallback model...")
            try:
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization",
                    use_auth_token=hf_token
                )
                print("✅ Successfully loaded pyannote/speaker-diarization")
                return True
                
            except Exception as e2:
                print(f"❌ Failed to load speaker-diarization: {e2}")
                print("\n🔧 AUTHENTICATION REQUIRED:")
                print("Please visit the following URLs and accept the user conditions:")
                print("1. https://huggingface.co/pyannote/speaker-diarization-3.1")
                print("2. https://huggingface.co/pyannote/speaker-diarization")
                print("3. https://huggingface.co/pyannote/segmentation")
                print("4. Click 'Agree and access repository' on each page")
                print("5. Make sure you're logged in with your Hugging Face account")
                return False
                
    except ImportError:
        print("❌ pyannote.audio not installed")
        print("Install with: pip install pyannote.audio")
        return False

def check_environment():
    """Check all required environment variables"""
    print("🔍 Checking environment variables...")
    
    required_vars = [
        'HUGGINGFACE_TOKEN',
        'openai_api_key'
    ]
    
    missing_vars = []
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {value[:10]}...")
        else:
            print(f"❌ {var}: Not found")
            missing_vars.append(var)
    
    return len(missing_vars) == 0

if __name__ == "__main__":
    print("🚀 Testing Hugging Face Authentication for Voice Diarization")
    print("=" * 60)
    
    # Check environment
    env_ok = check_environment()
    
    if env_ok:
        print("\n" + "=" * 60)
        # Test authentication
        auth_ok = test_huggingface_auth()
        
        if auth_ok:
            print("\n🎉 SUCCESS! Voice diarization should work now.")
        else:
            print("\n⚠️  Authentication needed. Follow the instructions above.")
    else:
        print("\n❌ Missing required environment variables. Please check your .env file.")
