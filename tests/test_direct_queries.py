#!/usr/bin/env python3
"""
Direct PostgreSQL Tool Testing
==============================

Test the PostgreSQL tool directly without going through the AI agent.
"""

from postgres_tool import PostgreSQLTool, enhanced_postgres_search
from tools import PostgreSQL_Diagnosis_Search

def test_direct_postgres_queries():
    """Test PostgreSQL tool with various queries"""
    print("🗄️ Direct PostgreSQL Tool Testing")
    print("=" * 50)
    
    postgres_tool = PostgreSQLTool()
    
    # Test queries
    queries = [
        "diagnosis",
        "5",
        "10", 
        "99",
        "heart",
        "diabetes",
        "condition"
    ]
    
    print("1️⃣ Testing keyword searches:")
    for query in queries:
        print(f"\n🔍 Searching for: '{query}'")
        result = postgres_tool.search_diagnosis_by_keyword(query, limit=3)
        print(f"   Results: {len(result['content'])} chars")
        if "No diagnosis" not in result['content']:
            print(f"   ✅ Found matches")
        else:
            print(f"   ❌ No matches")
    
    print("\n2️⃣ Testing specific diagnosis codes:")
    codes = ["D1000", "D1005", "D1010", "D1050", "D1099"]
    for code in codes:
        print(f"\n🔍 Looking up code: {code}")
        result = postgres_tool.get_diagnosis_by_code(code)
        if "No diagnosis found" not in result['content']:
            print(f"   ✅ Found: {code}")
        else:
            print(f"   ❌ Not found: {code}")
    
    print("\n3️⃣ Testing general database fetch:")
    result = postgres_tool.fetch_diagnosis_descriptions(limit=5)
    print(f"   Retrieved {len(result['content'])} chars of diagnosis data")
    
    print("\n4️⃣ Testing LangChain tool interface:")
    langchain_result = PostgreSQL_Diagnosis_Search.invoke("diagnosis")
    print(f"   LangChain tool returned {len(langchain_result)} chars")

if __name__ == "__main__":
    test_direct_postgres_queries()