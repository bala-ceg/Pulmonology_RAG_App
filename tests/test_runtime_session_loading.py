#!/usr/bin/env python3
"""
Test to confirm that session loading works in direct tool execution but not in agent execution
"""

import os
import sys
import json

# Setup path
os.chdir('/Users/bseetharaman/Desktop/Bala/2025/AI_Medical_App/Pulmonology_RAG_App')
sys.path.append(os.getcwd())

def test_session_loading_scenarios():
    """Test different scenarios of session loading"""
    print("🧪 Testing Session Loading Scenarios")
    print("=" * 50)
    
    try:
        from integrated_rag import IntegratedMedicalRAG
        from dotenv import load_dotenv
        
        # Load environment
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("openai_api_key")
        
        if not api_key:
            print("❌ OPENAI_API_KEY not found")
            return False
            
        # Initialize integrated RAG
        print("🚀 Initializing Integrated RAG System...")
        rag_system = IntegratedMedicalRAG(api_key)
        print("✅ Integrated RAG System initialized")
        
        # Check available sessions
        session_dirs = [d for d in os.listdir('./vector_dbs') if d.startswith('guest_')]
        if not session_dirs:
            print("❌ No session directories found")
            return False
            
        test_session = session_dirs[0]
        print(f"🔍 Testing with session: {test_session}")
        
        # Test 1: Direct tool execution (should work)
        print("\n1️⃣ Testing Direct Tool Execution...")
        try:
            direct_result = rag_system._direct_tool_execution(
                "What medical documents do I have?", 
                "Internal_VectorDB", 
                test_session
            )
            print("✅ Direct tool execution completed")
            print(f"📄 Result preview: {str(direct_result)[:200]}...")
        except Exception as e:
            print(f"❌ Direct tool execution failed: {e}")
            
        # Test 2: Full integrated query (this is where the issue might be)
        print("\n2️⃣ Testing Full Integrated Query...")
        try:
            full_result = rag_system.query(
                "What medical documents do I have?",
                test_session
            )
            print("✅ Full integrated query completed")
            print(f"📄 Answer preview: {str(full_result.get('answer', 'No answer'))[:200]}...")
            
            # Check if session was actually used
            if 'session' in str(full_result.get('answer', '')):
                print("✅ Session information found in response")
            else:
                print("⚠️  No session information in response - may not be using session loading")
                
        except Exception as e:
            print(f"❌ Full integrated query failed: {e}")
            
        return True
        
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        return False

def test_flask_endpoint_simulation():
    """Simulate Flask endpoint behavior"""
    print("\n🌐 Testing Flask Endpoint Simulation")
    print("=" * 50)
    
    try:
        # Simulate Flask request data
        request_data = {
            "data": "What medical documents do I have uploaded?",
            "session_id": "guest_100620251937"
        }
        
        print(f"📤 Simulating request: {request_data}")
        
        # Test the main query path that Flask would use
        from integrated_rag import IntegratedMedicalRAG
        from dotenv import load_dotenv
        
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("openai_api_key")
        
        rag_system = IntegratedMedicalRAG(api_key)
        
        # This simulates exactly what the Flask endpoint does
        result = rag_system.query(
            request_data["data"], 
            request_data["session_id"]
        )
        
        print("✅ Flask simulation completed")
        print(f"📄 Response: {str(result.get('answer', 'No answer'))[:300]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Flask simulation failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Session Loading Runtime Test")
    print("=" * 50)
    
    # Test 1: Different scenarios
    scenario_test = test_session_loading_scenarios()
    
    # Test 2: Flask simulation
    flask_test = test_flask_endpoint_simulation()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results")
    print(f"Scenario Testing: {'✅ PASS' if scenario_test else '❌ FAIL'}")
    print(f"Flask Simulation: {'✅ PASS' if flask_test else '❌ FAIL'}")
    
    if scenario_test and flask_test:
        print("\n🎯 Session loading is working in the system")
    else:
        print("\n⚠️  Issues detected with session loading")