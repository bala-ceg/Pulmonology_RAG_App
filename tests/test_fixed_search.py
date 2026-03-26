#!/usr/bin/env python3
"""
Test Fixed Wikipedia Search
==========================

Test that the Wikipedia_Search function now works correctly.
"""

def test_wikipedia_search_function():
    """Test the actual Wikipedia_Search function"""
    
    print("🧪 Testing Fixed Wikipedia_Search Function")
    print("=" * 50)
    
    try:
        # Import the function
        from tools import Wikipedia_Search
        
        # Test with the same query that was failing
        query = "Explain the symptoms of Type-2 Diabetes"
        print(f"Query: '{query}'")
        
        print(f"\n📚 Calling Wikipedia_Search...")
        result = Wikipedia_Search(query)
        
        print(f"✅ Result received:")
        print(f"   Length: {len(result)} characters")
        print(f"   Preview: {result[:300]}...")
        
        if len(result) > 500:
            print("✅ SUCCESS! Wikipedia search now returns substantial content")
            
            # Check if it contains relevant information about Type 2 diabetes
            diabetes_keywords = ['diabetes', 'blood sugar', 'insulin', 'glucose', 'symptoms']
            found_keywords = [kw for kw in diabetes_keywords if kw.lower() in result.lower()]
            print(f"   Relevant keywords found: {found_keywords}")
            
        else:
            print("❌ Still too short - may need further investigation")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_arxiv_search_function():
    """Test ArXiv search as well"""
    
    print(f"\n" + "="*50)
    print("🧪 Testing ArXiv_Search Function")
    print("="*50)
    
    try:  
        from tools import ArXiv_Search
        
        query = "diabetes treatment research"
        print(f"Query: '{query}'")
        
        print(f"\n📚 Calling ArXiv_Search...")
        result = ArXiv_Search(query)
        
        print(f"✅ Result received:")
        print(f"   Length: {len(result)} characters")
        print(f"   Preview: {result[:200]}...")
        
        if len(result) > 500:
            print("✅ SUCCESS! ArXiv search also returns substantial content")
        else:
            print("❌ ArXiv result also short - same issue likely affected it")
            
    except Exception as e:
        print(f"❌ Error: {e}")


def test_integrated_system_query():
    """Test a query through the integrated system"""
    
    print(f"\n" + "="*50)
    print("🧪 Testing Complete Integrated System")
    print("="*50)
    
    try:
        import os
        from integrated_rag import IntegratedMedicalRAG
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("⚠️ No API key - skipping integrated test")
            return
            
        print("🚀 Initializing IntegratedMedicalRAG...")
        system = IntegratedMedicalRAG(
            openai_api_key=api_key,
            base_vector_path="./vector_dbs"
        )
        
        query = "What are the symptoms of Type 2 diabetes?"
        print(f"Query: '{query}'")
        
        print(f"\n🎯 Running integrated query...")
        result = system.query(query, "test_session")
        
        if result and result.get('answer'):
            answer = result['answer']
            routing_info = result.get('routing_info', {})
            
            print(f"✅ Integrated result:")
            print(f"   Answer length: {len(answer)} characters")
            print(f"   Primary tool: {routing_info.get('primary_tool', 'Unknown')}")
            print(f"   Confidence: {routing_info.get('confidence', 'Unknown')}")
            print(f"   Answer preview: {answer[:300]}...")
            
            if len(answer) > 500:
                print("✅ SUCCESS! Integrated system now provides full responses")
            else:
                print("❌ Still getting short responses through integrated system")
        else:
            print("❌ No answer from integrated system")
            
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    test_wikipedia_search_function()
    test_arxiv_search_function()
    test_integrated_system_query()