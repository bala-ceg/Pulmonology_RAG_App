#!/usr/bin/env python3
"""
AI Agent Integration Test for PostgreSQL Tool
==============================================

This script tests that the PostgreSQL tool is properly integrated with the AI agent
and can be called through the main application.
"""

import requests
import json

def test_ai_integration():
    """Test PostgreSQL tool integration with AI agent"""
    print("🤖 Testing AI Agent Integration with PostgreSQL Tool")
    print("=" * 60)
    
    # Test data for various queries that should trigger the PostgreSQL tool
    test_queries = [
        {
            "query": "Show me diagnosis codes from the database",
            "expected_tool": "PostgreSQL",
            "description": "Should trigger PostgreSQL tool for diagnosis code query"
        },
        {
            "query": "What diagnoses are available in the database?",
            "expected_tool": "PostgreSQL", 
            "description": "Should trigger PostgreSQL tool for general diagnosis query"
        },
        {
            "query": "Find diagnosis code D1010",
            "expected_tool": "PostgreSQL",
            "description": "Should trigger PostgreSQL tool for specific diagnosis code"
        },
        {
            "query": "What is diabetes?",
            "expected_tool": "Wikipedia",
            "description": "Should trigger Wikipedia tool for general medical knowledge"
        }
    ]
    
    base_url = "http://localhost:5000"
    
    print(f"Testing against Flask app at: {base_url}")
    print("Note: Make sure the Flask app is running with 'python main.py'")
    print()
    
    for i, test in enumerate(test_queries, 1):
        print(f"{i}️⃣ Test: {test['description']}")
        print(f"   Query: '{test['query']}'")
        
        try:
            # Send request to the data endpoint
            response = requests.post(
                f"{base_url}/data",
                json={
                    "data": test["query"],
                    "patient_problem": "Test patient for PostgreSQL integration"
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get("response"):
                    message = result.get("message", "")
                    routing_details = result.get("routing_details", {})
                    
                    print(f"   ✅ Response received")
                    print(f"   📝 Message length: {len(message)} characters")
                    print(f"   🔧 Routing method: {routing_details.get('method', 'Unknown')}")
                    print(f"   📊 Confidence: {routing_details.get('confidence', 'Unknown')}")
                    
                    # Check if PostgreSQL tool was used (look for database-related content)
                    if "PostgreSQL" in message or "Database:" in message or "diagnosis_id" in message:
                        print(f"   🗄️ PostgreSQL tool likely used")
                    elif "Wikipedia" in message:
                        print(f"   📚 Wikipedia tool likely used")
                    else:
                        print(f"   🤷 Tool used: unclear")
                    
                    print(f"   📄 Response preview: {message[:150]}...")
                    
                else:
                    print(f"   ❌ No response in result: {result}")
            else:
                print(f"   ❌ HTTP Error {response.status_code}: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print(f"   ❌ Connection Error: Flask app not running at {base_url}")
            print(f"      Please start the Flask app with: python main.py")
            return False
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print()
    
    print("🏥 Integration test complete!")
    print("If PostgreSQL-related queries showed database content, the integration is working.")
    return True

if __name__ == "__main__":
    test_ai_integration()