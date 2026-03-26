#!/usr/bin/env python3
"""Test the enhanced PostgreSQL search with intelligent query processing"""

from postgres_tool import enhanced_postgres_search
import sys

def test_enhanced_search():
    """Test the enhanced search functionality"""
    print("🔍 TESTING ENHANCED POSTGRESQL SEARCH")
    print("=" * 50)
    
    # Test queries that should list all diagnoses
    test_queries = [
        'What diagnoses are available in the database?',
        'Show me diagnosis codes from the database',
        'List all diagnoses',
        'What diagnosis codes are available?',
        'Show me available diagnoses',
        'Search for diabetes'  # This should search specifically
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nTest {i}: '{query}'")
        print("-" * 40)
        
        try:
            result = enhanced_postgres_search(query)
            
            if isinstance(result, dict):
                content = result.get('content', '')
                summary = result.get('summary', '')
                
                print(f"Summary: {summary}")
                
                if 'No diagnosis descriptions found' in content:
                    print("❌ Still showing no results")
                    print(f"   Search term was: {query}")
                elif 'Diagnosis' in content:
                    print("✅ Found diagnosis data!")
                    # Count how many diagnoses were returned
                    diagnosis_count = content.count('**Diagnosis ID:**')
                    print(f"   Found {diagnosis_count} diagnoses")
                    
                    # Show first diagnosis as sample
                    lines = content.split('\n')
                    for line in lines[:5]:
                        if line.strip() and '**' in line:
                            print(f"   Sample: {line.strip()[:80]}")
                            break
                else:
                    print("⚠️  Unexpected result")
                    print(f"   Content preview: {content[:100]}...")
            else:
                print(f"⚠️  Unexpected result type: {type(result)}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("Test completed. If you see ✅ results, the fix is working!")

if __name__ == "__main__":
    test_enhanced_search()