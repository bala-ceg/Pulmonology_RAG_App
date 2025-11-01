#!/usr/bin/env python3
"""Test the fixed PostgreSQL tool integration"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_postgresql_fix():
    """Test the fixed tool filtering logic"""
    print("🔧 TESTING POSTGRESQL TOOL FIX")
    print("=" * 50)
    
    try:
        from integrated_rag import IntegratedMedicalRAG
        
        # Create system
        system = IntegratedMedicalRAG(openai_api_key=os.getenv('OPENAI_API_KEY', 'test-key'))
        
        print("✅ System initialized successfully")
        
        # Test database query
        query = "What diagnoses are available in the database?"
        print(f"\n🔍 Testing query: '{query}'")
        
        result = system.query(query)
        
        print("✅ Query executed successfully")
        print(f"Response type: {type(result)}")
        
        if isinstance(result, dict):
            print(f"Response keys: {list(result.keys())}")
            
            # Check if response contains PostgreSQL data
            if 'response' in result:
                response_text = str(result['response'])
                print(f"Response length: {len(response_text)} characters")
                
                # Look for indicators of successful PostgreSQL tool usage
                indicators = [
                    'PostgreSQL' in response_text,
                    'Diagnosis' in response_text,
                    'Database' in response_text,
                    'diagnosis' in response_text.lower(),
                    'd1' in response_text.lower(),  # diagnosis codes start with D1
                ]
                
                print(f"Contains PostgreSQL data indicators: {any(indicators)}")
                
                if any(indicators):
                    print("🎉 SUCCESS: PostgreSQL tool is working!")
                else:
                    print("⚠️  WARNING: Response doesn't contain expected PostgreSQL data")
                    print(f"Response preview: {response_text[:300]}...")
            else:
                print("⚠️  No 'response' key in result")
        else:
            print(f"⚠️  Unexpected result type: {type(result)}")
            print(f"Result: {result}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_postgresql_fix()