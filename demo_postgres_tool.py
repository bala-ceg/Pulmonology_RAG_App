#!/usr/bin/env python3
"""
PostgreSQL Tool Demo
====================

This script demonstrates the PostgreSQL tool functionality with real data.
"""

from postgres_tool import PostgreSQLTool, enhanced_postgres_search
from tools import PostgreSQL_Diagnosis_Search

def demo_postgres_tool():
    """Demonstrate the PostgreSQL tool functionality"""
    print("🏥 PostgreSQL Database Tool Demo")
    print("=" * 50)
    
    postgres_tool = PostgreSQLTool()
    
    # Test 1: Get all diagnoses
    print("\n1️⃣ Fetching first 10 diagnosis records...")
    result = postgres_tool.fetch_diagnosis_descriptions(limit=10)
    print(f"Status: {'✅ SUCCESS' if 'Error' not in result['content'] else '❌ FAILED'}")
    print(f"Summary: {result['summary']}")
    print(f"Content Preview:\n{result['content'][:500]}...")
    
    # Test 2: Search for specific diagnosis codes
    print("\n2️⃣ Searching by diagnosis code (D1005)...")
    code_result = postgres_tool.get_diagnosis_by_code("D1005")
    print(f"Status: {'✅ SUCCESS' if 'Error' not in code_result['content'] else '❌ FAILED'}")
    print(f"Content:\n{code_result['content']}")
    
    # Test 3: Search using the enhanced search
    print("\n3️⃣ Using enhanced search for 'Diagnosis 5'...")
    enhanced_result = enhanced_postgres_search("Diagnosis 5")
    print(f"Status: {'✅ SUCCESS' if 'Error' not in enhanced_result['content'] else '❌ FAILED'}")
    print(f"Content:\n{enhanced_result['content']}")
    
    # Test 4: Using the LangChain tool directly
    print("\n4️⃣ Using LangChain tool for generic 'diagnosis' search...")
    langchain_result = PostgreSQL_Diagnosis_Search("diagnosis")
    print(f"Result:\n{langchain_result}")
    
    print("\n🎉 PostgreSQL Tool Demo Complete!")
    print("The tool is ready to be used by the AI agent for medical diagnosis queries.")

if __name__ == "__main__":
    demo_postgres_tool()