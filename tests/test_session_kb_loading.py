#!/usr/bin/env python3
"""
Test script to verify session-specific vector database loading works correctly.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from rag_architecture import TwoStoreRAGManager
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    
    def test_session_kb_loading():
        """Test session-specific vector database loading."""
        print("🧪 Testing Session-Specific Vector Database Loading")
        print("=" * 60)
        
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
        rag_manager = TwoStoreRAGManager(embeddings, llm, "./vector_dbs")
        
        print(f"✅ RAG Manager initialized")
        print(f"📊 Current session: {rag_manager.current_session_id}")
        print(f"🗂️ Session cache size: {len(rag_manager.session_cache)}")
        
        # List available session vector DBs
        vector_dbs_path = "./vector_dbs"
        session_dirs = [d for d in os.listdir(vector_dbs_path) 
                       if os.path.isdir(os.path.join(vector_dbs_path, d)) 
                       and d.startswith('guest_')]
        
        print(f"\n📁 Available session vector DBs: {len(session_dirs)}")
        for session_dir in session_dirs[:5]:  # Show first 5
            print(f"   - {session_dir}")
        
        # Test loading different sessions
        if session_dirs:
            test_session = session_dirs[0]
            print(f"\n🔄 Testing session loading: {test_session}")
            
            # Load session vector DB
            success = rag_manager.load_session_vector_db(test_session)
            print(f"✅ Session load success: {success}")
            print(f"📊 Current session: {rag_manager.current_session_id}")
            print(f"🔍 kb_local loaded: {rag_manager.kb_local is not None}")
            
            if rag_manager.kb_local:
                try:
                    count = rag_manager.kb_local._collection.count()
                    print(f"📄 Documents in session KB: {count}")
                except:
                    print("📄 Could not get document count")
            
            # Test query with session
            print(f"\n🔍 Testing query with session: {test_session}")
            test_query = "What is diabetes?"
            result = rag_manager.query_with_routing(test_query, test_session)
            
            print(f"📝 Query: {test_query}")
            print(f"🎯 Session KB loaded: {result['routing_info'].get('session_kb_loaded', False)}")
            print(f"📊 Similarity score: {result['routing_info'].get('similarity_score', 0):.3f}")
            print(f"🔀 Query local first: {result['routing_info'].get('query_local_first', False)}")
            print(f"📚 Sources queried: {result['routing_info'].get('sources_queried', [])}")
            print(f"💬 Responses: {len(result['responses'])}")
            
            # Test loading different session
            if len(session_dirs) > 1:
                test_session2 = session_dirs[1]
                print(f"\n🔄 Testing different session: {test_session2}")
                success2 = rag_manager.load_session_vector_db(test_session2)
                print(f"✅ Second session load success: {success2}")
                print(f"📊 Current session: {rag_manager.current_session_id}")
                print(f"🗂️ Session cache size: {len(rag_manager.session_cache)}")
        else:
            print("⚠️ No session vector DBs found to test")
        
        print("\n" + "=" * 60)
        print("🎉 Session KB loading test completed!")

    if __name__ == "__main__":
        test_session_kb_loading()
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all required packages are installed and the environment is set up correctly.")
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()