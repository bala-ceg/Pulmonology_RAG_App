#!/usr/bin/env python3
"""
Debug External KB Performance
"""

import sys
import os
from dotenv import load_dotenv

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_external_kb_directly():
    """Test external KB directly to see what content is available."""
    
    load_dotenv()
    
    try:
        from rag_architecture import TwoStoreRAGManager
        from langchain_openai import OpenAIEmbeddings, ChatOpenAI
        
        # Initialize components
        embeddings = OpenAIEmbeddings(
            api_key=os.getenv('openai_api_key'),
            base_url=os.getenv('base_url'),
            model=os.getenv('embedding_model_name')
        )
        
        llm = ChatOpenAI(
            api_key=os.getenv("openai_api_key"),
            base_url=os.getenv("base_url"),
            model_name=os.getenv("llm_model_name")
        )
        
        rag_manager = TwoStoreRAGManager(embeddings, llm)
        
        print("🔍 Testing External KB Performance")
        print("=" * 40)
        
        # Check if external KB exists
        if not rag_manager.kb_external:
            print("❌ External KB not found!")
            print("Solutions:")
            print("1. Run: python3 setup_external_kb.py setup")
            print("2. Wait for setup to complete")
            print("3. Restart your Flask app")
            return False
        
        # Try to get some basic info about the external KB
        try:
            count = rag_manager.kb_external._collection.count()
            print(f"📊 External KB contains {count} document chunks")
        except:
            print("📊 External KB exists but cannot count documents")
        
        # Test with diabetes-related queries
        test_queries = [
            "diabetes symptoms",
            "type 2 diabetes",
            "diabetes mellitus", 
            "hyperglycemia symptoms",
            "insulin resistance"
        ]
        
        print("\n🧪 Testing retrieval for diabetes-related terms:")
        print("-" * 40)
        
        for query in test_queries:
            try:
                retriever = rag_manager.kb_external.as_retriever(search_type="similarity", search_kwargs={"k": 3})
                results = retriever.invoke(query)
                
                print(f"\n🔍 Query: '{query}'")
                print(f"📋 Found {len(results)} relevant chunks")
                
                for i, doc in enumerate(results, 1):
                    preview = doc.page_content[:100].replace('\n', ' ') + "..."
                    source_type = doc.metadata.get('source_type', 'unknown')
                    title = doc.metadata.get('title', doc.metadata.get('Title', 'No title'))
                    print(f"   {i}. [{source_type}] {title}")
                    print(f"      Preview: {preview}")
                
            except Exception as e:
                print(f"❌ Error testing query '{query}': {e}")
        
        # Test full QA for diabetes
        print(f"\n🤖 Testing full QA chain for diabetes:")
        print("-" * 40)
        
        diabetes_query = "What are the main symptoms of type 2 diabetes?"
        external_response = rag_manager._query_kb_external(diabetes_query)
        
        if external_response:
            print(f"✅ Got response for: {diabetes_query}")
            print(f"📝 Response: {external_response['result'][:300]}...")
            print(f"📚 Citations: {len(external_response.get('citations', []))}")
            
            if external_response.get('citations'):
                print("Citations:")
                for citation in external_response['citations'][:3]:
                    print(f"  - {citation}")
        else:
            print(f"❌ No response for: {diabetes_query}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_external_kb_directly()