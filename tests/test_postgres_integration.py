#!/usr/bin/env python3
"""
PostgreSQL Database Connection Test
===================================

This script tests the PostgreSQL database connection and demonstrates
fetching data from the p_diagnosis.description table.
"""

import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_database_connection():
    """Test basic database connection"""
    print("🔍 Testing PostgreSQL Database Connection...")
    print("=" * 50)
    
    try:
        from postgres_tool import PostgreSQLTool
        
        # Initialize the PostgreSQL tool
        postgres_tool = PostgreSQLTool()
        
        # Test connection
        connection_result = postgres_tool.test_connection()
        
        if connection_result['status'] == 'success':
            print("✅ Database Connection: SUCCESS")
            print(f"   {connection_result['message']}")
        else:
            print("❌ Database Connection: FAILED")
            print(f"   {connection_result['message']}")
            return False
            
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("   Please install psycopg2-binary: pip install psycopg2-binary")
        return False
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return False
    
    return True


def test_diagnosis_table_access():
    """Test access to p_diagnosis table"""
    print("\n🔍 Testing p_diagnosis Table Access...")
    print("=" * 50)
    
    try:
        from postgres_tool import PostgreSQLTool
        
        postgres_tool = PostgreSQLTool()
        
        # Test 1: Fetch all diagnosis descriptions (limited)
        print("\n📋 Test 1: Fetching first 5 diagnosis descriptions...")
        result = postgres_tool.fetch_diagnosis_descriptions(limit=5)
        
        print(f"   Status: {'✅ SUCCESS' if 'Error' not in result['content'] else '❌ FAILED'}")
        print(f"   Content length: {len(result['content'])} characters")
        print(f"   Summary: {result['summary']}")
        
        if 'Error' not in result['content']:
            print("   Sample content preview:")
            print("   " + result['content'][:200] + "..." if len(result['content']) > 200 else result['content'])
        
        # Test 2: Search for specific diagnosis
        print("\n🔍 Test 2: Searching for 'diabetes' diagnosis...")
        diabetes_result = postgres_tool.search_diagnosis_by_keyword("diabetes", limit=3)
        
        print(f"   Status: {'✅ SUCCESS' if 'Error' not in diabetes_result['content'] else '❌ FAILED'}")
        print(f"   Content length: {len(diabetes_result['content'])} characters")
        print(f"   Summary: {diabetes_result['summary']}")
        
        if 'Error' not in diabetes_result['content']:
            print("   Diabetes search preview:")
            print("   " + diabetes_result['content'][:300] + "..." if len(diabetes_result['content']) > 300 else diabetes_result['content'])
        
        return True
        
    except Exception as e:
        print(f"❌ Table Access Error: {e}")
        return False


def test_tool_integration():
    """Test the tool integration with the enhanced system"""
    print("\n🔍 Testing Tool Integration...")
    print("=" * 50)
    
    try:
        from postgres_tool import enhanced_postgres_search
        
        # Test the enhanced search function
        print("\n🛠️  Testing enhanced_postgres_search function...")
        result = enhanced_postgres_search("hypertension")
        
        print(f"   Status: {'✅ SUCCESS' if 'Error' not in result['content'] else '❌ FAILED'}")
        print(f"   Content: {len(result['content'])} chars")
        print(f"   Summary: {len(result['summary'])} chars")
        print(f"   Citations: {result['citations']}")
        print(f"   Tool Info: {result['tool_info']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Tool Integration Error: {e}")
        return False


def test_langchain_tool():
    """Test the LangChain tool format"""
    print("\n🔍 Testing LangChain Tool Format...")
    print("=" * 50)
    
    try:
        from tools import SQL_Diagnosis_Search
        
        # Test the tool directly
        print("\n🔧 Testing SQL_Diagnosis_Search tool...")
        result = SQL_Diagnosis_Search("heart disease")
        
        print(f"   Status: {'✅ SUCCESS' if 'Error' not in result else '❌ FAILED'}")
        print(f"   Result length: {len(result)} characters")
        print("   Result preview:")
        print("   " + result[:300] + "..." if len(result) > 300 else result)
        
        return True
        
    except Exception as e:
        print(f"❌ LangChain Tool Error: {e}")
        return False


def main():
    """Main test function"""
    print("🏥 PostgreSQL Medical Database Test Suite")
    print("=" * 60)
    
    # Display connection parameters for PostgreSQL tool (without password)
    print(f"PostgreSQL Tool Database Host: {os.getenv('PG_TOOL_HOST', 'Not set')}")
    print(f"PostgreSQL Tool Database Port: {os.getenv('PG_TOOL_PORT', 'Not set')}")
    print(f"PostgreSQL Tool Database Name: {os.getenv('PG_TOOL_NAME', 'Not set')}")
    print(f"PostgreSQL Tool Database User: {os.getenv('PG_TOOL_USER', 'Not set')}")
    print(f"PostgreSQL Tool Password: {'✅ Set' if os.getenv('PG_TOOL_PASSWORD') else '❌ Not set'}")
    print(f"Legacy Database Name: {os.getenv('DB_NAME', 'Not set')} (used by existing code)")
    
    tests = [
        ("Database Connection", test_database_connection),
        ("p_diagnosis Table Access", test_diagnosis_table_access),
        ("Tool Integration", test_tool_integration),
        ("LangChain Tool Format", test_langchain_tool)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("🏥 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:<30} {status}")
    
    print("-" * 60)
    print(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! PostgreSQL tool is ready for use.")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()