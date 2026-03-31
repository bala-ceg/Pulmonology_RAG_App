#!/usr/bin/env python3
"""
Test script to verify OpenAI timeout configuration
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Load environment
load_dotenv()

print("🧪 Testing OpenAI Configuration with Timeout")
print("=" * 50)

# Test the LLM configuration with timeout
try:
    llm = ChatOpenAI(
        api_key=os.getenv('openai_api_key'),
        base_url=os.getenv('base_url'),
        model_name=os.getenv('llm_model_name'),
        request_timeout=30  # 30 second timeout
    )
    
    print(f"✅ LLM instance created successfully")
    print(f"   Model: {os.getenv('llm_model_name')}")
    print(f"   Base URL: {os.getenv('base_url')}")
    print(f"   Timeout: 30 seconds")
    print()
    
    # Test a simple invocation
    print("🔄 Testing LLM invocation with simple query...")
    response = llm.invoke([HumanMessage(content='Say hello in one word')])
    print(f"✅ Test query successful!")
    print(f"   Response: {response.content}")
    print()
    
    # Test with a medical query
    print("🔄 Testing with medical query...")
    medical_response = llm.invoke([
        HumanMessage(content='List 3 symptoms of Type 2 Diabetes in one line')
    ])
    print(f"✅ Medical query successful!")
    print(f"   Response: {medical_response.content}")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 50)
print("✅ All tests completed!")
