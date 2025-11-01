#!/usr/bin/env python3
"""
Test PostgreSQL Integration in Flask Web Interface
=================================================

This script simulates the exact flow that happens when a user submits
a database query through the web interface.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_web_interface_simulation():
    """Simulate the web interface query processing"""
    print("🌐 SIMULATING WEB INTERFACE POSTGRESQL QUERY")
    print("=" * 60)
    
    try:
        # Step 1: Import the integrated RAG system (same as Flask app)
        from integrated_rag import IntegratedMedicalRAG
        
        print("Step 1: ✅ Imported IntegratedMedicalRAG")
        
        # Step 2: Initialize the system (same as Flask app startup)
        system = IntegratedMedicalRAG(openai_api_key=os.getenv('OPENAI_API_KEY', 'test-key'))
        
        print("Step 2: ✅ Initialized RAG system")
        print(f"         Available tools: {len(system.tools)}")
        
        # List tools to verify PostgreSQL is included
        tool_names = []
        for tool in system.tools:
            if hasattr(tool, 'name'):
                tool_names.append(tool.name)
            elif hasattr(tool, '__name__'):
                tool_names.append(tool.__name__)
            else:
                tool_names.append(type(tool).__name__)
        
        print(f"         Tool names: {', '.join(tool_names)}")
        
        if 'PostgreSQL_Diagnosis_Search' in tool_names:
            print("Step 3: ✅ PostgreSQL tool found in agent")
        else:
            print("Step 3: ❌ PostgreSQL tool NOT found in agent")
            return
        
        # Step 4: Process a database query (same as when user submits via web)
        query = "What diagnoses are available in the database?"
        print(f"\nStep 4: Processing query: '{query}'")
        
        # This is the exact method called by Flask route
        try:
            response = system.query(query)
            print("Step 5: ✅ Query processed successfully")
            print(f"         Response length: {len(str(response))} characters")
            print(f"         Response preview: {str(response)[:200]}...")
            
            # Check if response contains PostgreSQL data
            response_str = str(response)
            if "PostgreSQL" in response_str or "Diagnosis" in response_str:
                print("Step 6: ✅ Response contains PostgreSQL data")
            else:
                print("Step 6: ⚠️  Response may not contain PostgreSQL data")
                print(f"         Full response: {response_str}")
                
        except Exception as query_error:
            print(f"Step 5: ❌ Error processing query: {query_error}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ Error in simulation: {e}")
        import traceback
        traceback.print_exc()

def test_direct_tool_access():
    """Test direct access to PostgreSQL tool (bypassing agent)"""
    print("\n\n🔧 DIRECT POSTGRESQL TOOL ACCESS TEST")
    print("=" * 60)
    
    try:
        from tools import PostgreSQL_Diagnosis_Search
        
        print("Step 1: ✅ Imported PostgreSQL tool directly")
        
        # Test the tool function directly
        result = PostgreSQL_Diagnosis_Search("What diagnoses are available?")
        print("Step 2: ✅ Direct tool call successful")
        print(f"         Result length: {len(str(result))} characters")
        print(f"         Result preview: {str(result)[:200]}...")
        
    except Exception as e:
        print(f"❌ Error in direct tool test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_web_interface_simulation()
    test_direct_tool_access()
    
    print("\n\n" + "=" * 60)
    print("🏥 SUMMARY")
    print("=" * 60)
    print("If all steps show ✅, the PostgreSQL tool should work in the web interface.")
    print("If there are ❌ errors, those need to be fixed for web interface to work.")
    print("Check the output above for any issues that need resolution.")