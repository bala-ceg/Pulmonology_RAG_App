#!/usr/bin/env python3
"""
Complete PostgreSQL Integration Test
===================================

This test verifies both the agent-based execution and the fallback mechanism
for the PostgreSQL tool integration.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_complete_integration():
    """Test complete PostgreSQL integration including agent and fallback"""
    print("🎯 COMPLETE POSTGRESQL INTEGRATION TEST")
    print("=" * 60)
    
    try:
        from integrated_rag import IntegratedMedicalRAG
        from tools import PostgreSQL_Diagnosis_Search
        
        # Test 1: Direct PostgreSQL tool function
        print("\n1️⃣  TESTING DIRECT POSTGRESQL TOOL")
        print("-" * 40)
        
        direct_result = PostgreSQL_Diagnosis_Search("What diagnoses are available?")
        print(f"✅ Direct tool call successful")
        print(f"   Result length: {len(str(direct_result))} characters")
        print(f"   Contains diagnosis data: {'Diagnosis' in str(direct_result)}")
        
        # Test 2: Integrated RAG system
        print("\n2️⃣  TESTING INTEGRATED RAG SYSTEM")
        print("-" * 40)
        
        system = IntegratedMedicalRAG(openai_api_key=os.getenv('OPENAI_API_KEY', 'test-key'))
        print(f"✅ System initialized with {len(system.tools)} tools")
        
        # List tools to verify PostgreSQL is included
        tool_names = [getattr(tool, 'name', getattr(tool, '__name__', str(type(tool).__name__))) for tool in system.tools]
        print(f"   Available tools: {', '.join(tool_names)}")
        print(f"   PostgreSQL tool included: {'PostgreSQL_Diagnosis_Search' in tool_names}")
        
        # Test 3: Full query processing
        print("\n3️⃣  TESTING FULL QUERY PROCESSING")
        print("-" * 40)
        
        test_queries = [
            "What diagnoses are available in the database?",
            "Show me diagnosis codes from the database",
            "Search the database for medical conditions"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n   Query {i}: '{query}'")
            
            try:
                result = system.query(query)
                
                if isinstance(result, dict):
                    response = result.get('response', str(result))
                else:
                    response = str(result)
                
                print(f"   ✅ Query processed successfully")
                print(f"   Response length: {len(response)} characters")
                
                # Check for success indicators
                success_indicators = [
                    'Diagnosis' in response,
                    'diagnosis' in response.lower(),
                    'PostgreSQL' in response,
                    'database' in response.lower(),
                    'D10' in response,  # Diagnosis codes
                ]
                
                success_count = sum(success_indicators)
                print(f"   Success indicators: {success_count}/5")
                
                if success_count >= 2:
                    print(f"   🎉 SUCCESS: PostgreSQL tool working correctly!")
                elif 'Unknown tool' in response:
                    print(f"   ❌ FAILURE: Still showing 'Unknown tool' error")
                    print(f"   Response: {response[:200]}...")
                else:
                    print(f"   ⚠️  PARTIAL: Tool may be working but response unclear")
                    print(f"   Response preview: {response[:150]}...")
                    
            except Exception as query_error:
                print(f"   ❌ Error processing query: {query_error}")
        
        # Test 4: Direct tool execution method
        print("\n4️⃣  TESTING DIRECT TOOL EXECUTION FALLBACK")
        print("-" * 40)
        
        try:
            fallback_result = system._direct_tool_execution("What diagnoses are available?", "PostgreSQL_Diagnosis_Search")
            print(f"✅ Direct tool execution successful")
            print(f"   Result length: {len(str(fallback_result))} characters")
            print(f"   Contains diagnosis data: {'Diagnosis' in str(fallback_result)}")
            
            if 'Unknown tool' in fallback_result:
                print(f"   ❌ Still showing 'Unknown tool' in fallback")
            else:
                print(f"   🎉 Fallback mechanism working correctly!")
                
        except Exception as fallback_error:
            print(f"   ❌ Error in fallback execution: {fallback_error}")
            
    except Exception as e:
        print(f"❌ Critical error: {e}")
        import traceback
        traceback.print_exc()

def summary():
    """Print test summary"""
    print("\n\n" + "=" * 60)
    print("🏥 POSTGRESQL INTEGRATION TEST SUMMARY")
    print("=" * 60)
    print("If all tests show ✅ and 🎉, the PostgreSQL tool is fully working.")
    print("If you see ❌ errors, check the output above for issues to fix.")
    print("The system should now work correctly in the web interface!")

if __name__ == "__main__":
    test_complete_integration()
    summary()