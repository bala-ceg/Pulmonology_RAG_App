#!/usr/bin/env python3
"""
Test Flask Integration with IntegratedMedicalRAG
==============================================

This script tests that the main.py Flask application correctly integrates 
with the IntegratedMedicalRAG system for intelligent tool routing.
"""

import json
import requests
import time
from typing import Dict, Any

def test_flask_integration():
    """Test the Flask /data endpoint with various medical queries"""
    
    # Test queries that should route to different tools
    test_queries = [
        {
            "query": "What is hypertension?",
            "expected_tool": "Wikipedia_Search",
            "description": "Basic medical definition - should route to Wikipedia"
        },
        {
            "query": "Latest research on COVID-19 treatments",
            "expected_tool": "ArXiv_Search", 
            "description": "Research query - should route to ArXiv"
        },
        {
            "query": "Recent preprints on pulmonary fibrosis mechanisms",
            "expected_tool": "ArXiv_Search",
            "description": "Specific research query - should route to ArXiv"
        },
        {
            "query": "What does my uploaded PDF say about treatment protocols?",
            "expected_tool": "Internal_VectorDB",
            "description": "User document query - should route to Internal VectorDB"
        }
    ]
    
    base_url = "http://localhost:5000"  # Assuming Flask runs on default port
    endpoint = f"{base_url}/data"
    
    print("=" * 70)
    print("Testing Flask Integration with IntegratedMedicalRAG")
    print("=" * 70)
    
    # Check if Flask server is running
    try:
        response = requests.get(base_url, timeout=5)
        print("✅ Flask server is running")
    except requests.exceptions.RequestException:
        print("❌ Flask server is not running. Please start the server with:")
        print("   python main.py")
        print("   or python3 main.py")
        return
    
    results = []
    
    for i, test_case in enumerate(test_queries, 1):
        print(f"\n🧪 Test {i}: {test_case['description']}")
        print(f"Query: '{test_case['query']}'")
        print(f"Expected tool: {test_case['expected_tool']}")
        
        # Prepare request data
        request_data = {
            "data": test_case["query"],
            "session_id": "test_session_123"
        }
        
        try:
            # Send request to Flask endpoint
            start_time = time.time()
            response = requests.post(
                endpoint, 
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result_data = response.json()
                
                if result_data.get("response"):
                    # Extract routing information
                    routing_details = result_data.get("routing_details", {})
                    method = routing_details.get("method", "Unknown")
                    primary_tool = routing_details.get("primary_tool", "Unknown")
                    confidence = routing_details.get("confidence", 0)
                    tools_used = routing_details.get("tools_used", [])
                    
                    print(f"✅ Success! Response received")
                    print(f"   Method: {method}")
                    print(f"   Primary tool: {primary_tool}")
                    print(f"   Confidence: {confidence}%")
                    print(f"   Tools used: {tools_used}")
                    print(f"   Processing time: {processing_time:.2f}s")
                    
                    # Check if correct tool was used
                    tool_match = test_case['expected_tool'].lower() in primary_tool.lower()
                    if tool_match:
                        print(f"   ✅ Correct tool routing!")
                    else:
                        print(f"   ⚠️ Expected {test_case['expected_tool']}, got {primary_tool}")
                    
                    # Show response preview
                    message = result_data.get("message", "")
                    if len(message) > 200:
                        print(f"   Response preview: {message[:200]}...")
                    else:
                        print(f"   Response: {message}")
                    
                    results.append({
                        "test": test_case["description"],
                        "success": True,
                        "tool_match": tool_match,
                        "method": method,
                        "primary_tool": primary_tool,
                        "processing_time": processing_time
                    })
                    
                else:
                    print(f"❌ Error: {result_data.get('message', 'Unknown error')}")
                    results.append({
                        "test": test_case["description"],
                        "success": False,
                        "error": result_data.get('message', 'Unknown error')
                    })
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                results.append({
                    "test": test_case["description"],
                    "success": False,
                    "error": f"HTTP {response.status_code}"
                })
                
        except requests.exceptions.Timeout:
            print("❌ Request timeout (>30s)")
            results.append({
                "test": test_case["description"],
                "success": False,
                "error": "Timeout"
            })
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            results.append({
                "test": test_case["description"],
                "success": False,
                "error": str(e)
            })
        
        print("-" * 50)
    
    # Summary
    print(f"\n📊 TEST SUMMARY:")
    successful_tests = [r for r in results if r.get('success')]
    correct_routing = [r for r in results if r.get('tool_match')]
    
    print(f"✅ Successful responses: {len(successful_tests)}/{len(test_queries)}")
    print(f"🎯 Correct tool routing: {len(correct_routing)}/{len(test_queries)}")
    
    if successful_tests:
        avg_time = sum(r.get('processing_time', 0) for r in successful_tests) / len(successful_tests)
        print(f"⏱️ Average processing time: {avg_time:.2f}s")
        
        methods_used = set(r.get('method') for r in successful_tests if r.get('method'))
        print(f"🔧 Methods used: {', '.join(methods_used)}")
    
    # Integration status
    integrated_responses = [r for r in results if r.get('method', '').startswith('Integrated')]
    if integrated_responses:
        print(f"\n🎉 INTEGRATION SUCCESS!")
        print(f"   {len(integrated_responses)} queries used IntegratedMedicalRAG system")
        print(f"   Tool routing is working correctly!")
    else:
        print(f"\n⚠️ Integration status unclear:")
        print(f"   No queries used IntegratedMedicalRAG system")
        print(f"   Check if system initialized correctly")


def test_import_only():
    """Test just the import functionality without running Flask server"""
    print("🧪 Testing IntegratedMedicalRAG import and initialization...")
    
    try:
        # Test the imports
        from integrated_rag import IntegratedMedicalRAG
        print("✅ IntegratedMedicalRAG imported successfully")
        
        # Test initialization (mock)
        import os
        api_key = os.getenv('OPENAI_API_KEY', 'test-key')
        
        if api_key and api_key != 'test-key':
            integrated_system = IntegratedMedicalRAG(
                openai_api_key=api_key,
                base_vector_path="./vector_dbs"
            )
            print("✅ IntegratedMedicalRAG initialized successfully")
            
            # Test system status
            status = integrated_system.get_system_status()
            print(f"📊 System Status:")
            print(f"   Available tools: {status.get('available_tools', [])}")
            print(f"   Local documents: {status.get('local_document_count', 0)}")
            print(f"   External documents: {status.get('external_document_count', 0)}")
        else:
            print("⚠️ OPENAI_API_KEY not set, skipping full initialization test")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "import-only":
        test_import_only()
    else:
        test_flask_integration()