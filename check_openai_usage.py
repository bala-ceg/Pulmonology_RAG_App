#!/usr/bin/env python3
"""
OpenAI Usage Checker
Quick script to check your current OpenAI API usage and limits
"""

import os
import requests
from datetime import datetime, timedelta
import json
from dotenv import load_dotenv

def check_openai_usage():
    """Check current OpenAI API usage and limits"""

    # Load environment variables from .env file
    load_dotenv()

    api_key = os.getenv('openai_api_key')  # Match your project's variable name
    if not api_key:
        print("❌ openai_api_key not found in environment variables")
        print("Make sure your .env file contains: openai_api_key=\"your-key-here\"")
        return

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    try:
        print("🔍 Checking OpenAI API status...")
        print(f"📅 API Key ends with: ...{api_key[-4:] if api_key else 'N/A'}")

        # Test API connectivity with a simple models list call
        response = requests.get('https://api.openai.com/v1/models', headers=headers, timeout=10)

        if response.status_code == 200:
            print("✅ API key is valid and working")

            models_data = response.json()
            models = [model['id'] for model in models_data['data']]

            print(f"📋 Available models: {len(models)} models")
            gpt_models = [m for m in models if 'gpt' in m.lower()]
            print(f"🤖 GPT models available: {len(gpt_models)}")
            print("Recent GPT models:", sorted(gpt_models)[-5:])

            # Test with a simple prompt
            print("\n🧪 Testing with a simple prompt...")
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                
                test_response = client.chat.completions.create(
                    model='gpt-4o-mini',
                    messages=[{'role': 'user', 'content': 'Explain the symptoms of Type-2 Diabetes'}],
                    max_tokens=50
                )
                
                print("✅ OpenAI API Response:")
                print(test_response.choices[0].message.content)
                print(f"\n📊 Model used: {test_response.model}")
                print(f"🔢 Tokens used: {test_response.usage.total_tokens}")
                
            except Exception as e:
                print(f"❌ Error during prompt test: {e}")
                if "insufficient_quota" in str(e):
                    print("\n💰 QUOTA ISSUE: You need to add credits to your OpenAI account")
                    print("   1. Go to: https://platform.openai.com/account/billing")
                    print("   2. Add payment method and credits")
                    print("   3. Increase spending limit")

        elif response.status_code == 401:
            print("❌ Invalid API key - authentication failed")
            return
        elif response.status_code == 429:
            print("❌ Rate limit exceeded - QUOTA ISSUE CONFIRMED")
            error_data = response.json()
            error_msg = error_data.get('error', {}).get('message', 'Unknown error')
            print(f"Error: {error_msg}")

            if 'insufficient_quota' in error_msg.lower():
                print("\n💰 SOLUTION: You need to add credits to your OpenAI account")
                print("   1. Go to: https://platform.openai.com/usage")
                print("   2. Add payment method")
                print("   3. Increase spending limit")
            return
        else:
            print(f"⚠️ Unexpected response: {response.status_code}")
            print(f"Response: {response.text}")
            return


    except requests.exceptions.Timeout:
        print("❌ Request timed out - check your internet connection")
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    print("🚀 OpenAI Usage Checker")
    print("=" * 50)
    check_openai_usage()