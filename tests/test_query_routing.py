#!/usr/bin/env python3
"""
PostgreSQL Tool Query Tester
============================

Test various queries to see how the AI agent routes them to different tools,
including the new PostgreSQL tool for diagnosis queries.
"""

import requests
import json
import time

def test_query(query, description=""):
    """Test a single query against the AI system"""
    print(f"\n🔍 Testing: {description}")
    print(f"Query: '{query}'")
    print("-" * 60)
    
    try:
        response = requests.post(
            "http://localhost:5000/data",
            json={
                "data": query,
                "patient_problem": "Test patient for tool routing"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get("response"):
                message = result.get("message", "")
                routing_details = result.get("routing_details", {})
                
                print(f"✅ Response received ({len(message)} chars)")
                print(f"🔧 Method: {routing_details.get('method', 'Unknown')}")
                print(f"📊 Confidence: {routing_details.get('confidence', 'Unknown')}")
                
                # Detect which tool was likely used
                if "diagnosis_id" in message or "Database: pces_ehr_ccm" in message:
                    print(f"🗄️ Tool Used: PostgreSQL Database")
                elif "Wikipedia:" in message:
                    print(f"📚 Tool Used: Wikipedia")
                elif "arXiv:" in message:
                    print(f"📄 Tool Used: ArXiv")
                elif "Web:" in message:
                    print(f"🌐 Tool Used: Tavily Web Search")
                else:
                    print(f"🤷 Tool Used: Unknown/Internal VectorDB")
                
                print(f"📝 Response Preview: {message[:200]}...")
                
            else:
                print(f"❌ No response: {result}")
        else:
            print(f"❌ HTTP Error {response.status_code}: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection Error: Make sure Flask app is running at http://localhost:5000")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return True

def main():
    """Run comprehensive PostgreSQL tool tests"""
    print("🏥 PostgreSQL Tool Query Test Suite")
    print("=" * 80)
    
    # Test queries designed to trigger different tools
    test_queries = [
        # PostgreSQL Tool Tests
        ("What diagnoses are available in the database?", "General diagnosis database query"),
        ("Find diagnosis code D1005", "Specific diagnosis code lookup"),
        ("Show me diagnosis with code D1050", "Another specific code lookup"),
        ("Search for diagnosis containing '10'", "Keyword-based diagnosis search"),
        ("List medical diagnoses from the system", "System diagnosis listing"),
        ("Query the p_diagnosis table", "Direct table reference"),
        
        # Comparative Tests (should trigger other tools)
        ("What is diabetes?", "General medical knowledge (should use Wikipedia)"),
        ("Latest diabetes research papers", "Research query (should use ArXiv)"),
        ("Current diabetes treatment guidelines", "Current info (should use Tavily)"),
        
        # Edge Cases
        ("diagnosis", "Single word diagnosis query"),
        ("medical codes database", "Database-focused medical query"),
    ]
    
    print("Starting tests... Make sure Flask app is running!")
    print("Command: python main.py")
    time.sleep(2)
    
    for query, description in test_queries:
        if not test_query(query, description):
            break
        time.sleep(1)  # Small delay between requests
    
    print("\n" + "=" * 80)
    print("🎉 Test Complete!")
    print("\nLook for:")
    print("- PostgreSQL queries showing diagnosis_id, codes, hospital_id")
    print("- Other queries using Wikipedia, ArXiv, or Tavily")
    print("- Proper tool routing based on query content")

if __name__ == "__main__":
    main()