#!/usr/bin/env python3
"""Test PostgreSQL tool routing and functionality"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Test 1: Direct PostgreSQL tool functionality
print("=" * 60)
print("TEST 1: Direct PostgreSQL Tool Functionality")
print("=" * 60)

try:
    from postgres_tool import enhanced_postgres_search
    
    # Test direct database connection and query  
    print("Testing direct PostgreSQL connection...")
    result = enhanced_postgres_search("SELECT description FROM p_diagnosis LIMIT 5")
    print(f"Direct query result: {result}")
    
except Exception as e:
    print(f"Error in direct PostgreSQL test: {e}")

# Test 2: Routing System
print("\n" + "=" * 60)
print("TEST 2: Query Routing System")
print("=" * 60)

try:
    from rag_architecture import MedicalQueryRouter
    
    router = MedicalQueryRouter()
    
    # Test database-related queries
    test_queries = [
        "What diagnoses are available in the database?",
        "Show me diagnosis codes from the database",
        "Query the p_diagnosis table",
        "What data is in the diagnosis database?",
        "Search database for medical conditions"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        result = router.route_tools(query)
        primary_tool = result.get('primary_tool', 'None')
        confidence = result.get('confidence', 'unknown')
        scores = result.get('tool_scores', {})
        postgres_score = scores.get('PostgreSQL_Diagnosis_Search', 0)
        
        print(f"  → Primary Tool: {primary_tool}")
        print(f"  → Confidence: {confidence}")
        print(f"  → PostgreSQL Score: {postgres_score}")
        
        if primary_tool == 'PostgreSQL_Diagnosis_Search':
            print("  ✅ Correctly routed to PostgreSQL")
        else:
            print("  ❌ NOT routed to PostgreSQL")

except Exception as e:
    print(f"Error in routing test: {e}")

# Test 3: Tool Integration
print("\n" + "=" * 60)
print("TEST 3: Tool Registry Integration")
print("=" * 60)

try:
    from tools import AVAILABLE_TOOLS
    
    postgres_tools = [tool for tool in AVAILABLE_TOOLS if 'postgres' in tool.name.lower() or 'diagnosis' in tool.name.lower()]
    
    print(f"PostgreSQL tools found in registry: {len(postgres_tools)}")
    for tool in postgres_tools:
        print(f"  - {tool.name}: {tool.description}")
        
    if postgres_tools:
        print("✅ PostgreSQL tool is registered")
    else:
        print("❌ PostgreSQL tool NOT found in registry")

except Exception as e:
    print(f"Error in tool registry test: {e}")

print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)
print("All tests completed. Check results above for any issues.")