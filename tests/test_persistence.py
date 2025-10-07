#!/usr/bin/env python3
"""
Test Persistence Functionality
==============================

This script tests the persistence functionality of the RAG system.
"""

import os
import sys
from dotenv import load_dotenv

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from rag_architecture import TwoStoreRAGManager
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    RAG_AVAILABLE = True
except ImportError as e:
    print(f"❌ RAG architecture not available: {e}")
    RAG_AVAILABLE = False

def test_persistence():
    """Test that the persistence system works correctly."""
    if not RAG_AVAILABLE:
        print("❌ RAG architecture not available")
        return False
    
    # Load environment variables
    load_dotenv()
    
    try:
        print("🧪 Testing RAG System Persistence")
        print("=" * 40)
        
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
        
        # Initialize RAG manager
        print("🔄 Initializing RAG manager...")
        rag_manager = TwoStoreRAGManager(embeddings, llm)
        
        # Check content counts
        local_count = rag_manager.get_local_content_count()
        external_count = rag_manager.get_external_content_count()
        
        print(f"📊 Content Counts:")
        print(f"   Local KB: {local_count} documents")
        print(f"   External KB: {external_count} documents")
        
        # Check persistence status
        has_external = rag_manager.has_external_content()
        print(f"🔍 External KB Status: {'✅ Has content' if has_external else '❌ Empty'}")
        
        # Test a simple query
        if external_count > 0:
            print("\n🤖 Testing simple query...")
            test_query = "What is diabetes?"
            
            # This should use the lexical gate to route appropriately
            response = rag_manager.query_with_routing(test_query)
            
            print(f"Query: {test_query}")
            print(f"Response type: {type(response)}")
            print(f"Response keys: {response.keys() if isinstance(response, dict) else 'Not a dict'}")
            
            if isinstance(response, dict):
                print(f"Response length: {len(response.get('answer', ''))} characters")
                citations = response.get('citations', [])
                print(f"Citations: {len(citations)} sources")
                
                # Show citation display names
                for i, citation in enumerate(citations[:3]):  # Show first 3
                    if isinstance(citation, dict):
                        source_name = citation.get('source_display_name', 'Unknown')
                        print(f"   Citation {i+1}: {source_name}")
                    else:
                        print(f"   Citation {i+1}: {citation}")
            else:
                print(f"Unexpected response type: {response}")
        
        print("\n✅ Persistence test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing persistence: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_persistence()
    sys.exit(0 if success else 1)