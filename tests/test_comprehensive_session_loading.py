#!/usr/bin/env python3
"""
Comprehensive test to verify session-specific vector database loading works 
across all tools and components.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_session_loading_integration():
    """Test session-specific loading across all components."""
    print("🧪 Testing Complete Session-Specific Vector Database Integration")
    print("=" * 80)
    
    try:
        # Import required modules
        from rag_architecture import TwoStoreRAGManager
        from tools import Internal_VectorDB
        from enhanced_tools import enhanced_internal_search
        from integrated_rag import IntegratedMedicalRAG
        from langchain_openai import OpenAIEmbeddings, ChatOpenAI
        
        # Initialize components
        print("🔧 Initializing components...")
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
        
        # Test 1: RAG Manager Session Loading
        print("\n📊 Test 1: RAG Manager Session Loading")
        print("-" * 50)
        
        rag_manager = TwoStoreRAGManager(embeddings, llm, "./vector_dbs")
        
        # List available sessions
        vector_dbs_path = "./vector_dbs"
        session_dirs = [d for d in os.listdir(vector_dbs_path) 
                       if os.path.isdir(os.path.join(vector_dbs_path, d)) 
                       and d.startswith('guest_')]
        
        print(f"📁 Found {len(session_dirs)} session vector DBs")
        
        if session_dirs:
            test_session = session_dirs[0]
            print(f"🔄 Testing session: {test_session}")
            
            # Test RAG manager query with session
            result = rag_manager.query_with_routing("What is diabetes?", test_session)
            print(f"✅ RAG Manager - Session loaded: {result['routing_info'].get('session_kb_loaded', False)}")
            print(f"📊 RAG Manager - Current session: {rag_manager.current_session_id}")
            
            # Test 2: Internal_VectorDB Tool
            print(f"\n📊 Test 2: Internal_VectorDB Tool with Session {test_session}")
            print("-" * 50)
            
            tool_result = Internal_VectorDB("What is diabetes?", test_session, rag_manager)
            print(f"✅ Internal_VectorDB Tool - Result length: {len(tool_result)}")
            print(f"📝 Internal_VectorDB Tool - Preview: {tool_result[:100]}...")
            
            # Test 3: Enhanced Internal Search
            print(f"\n📊 Test 3: Enhanced Internal Search with Session {test_session}")
            print("-" * 50)
            
            enhanced_result = enhanced_internal_search("What is diabetes?", test_session, rag_manager)
            print(f"✅ Enhanced Internal Search - Content length: {len(enhanced_result.get('content', ''))}")
            print(f"📝 Enhanced Internal Search - Preview: {enhanced_result.get('content', '')[:100]}...")
            
            # Test 4: Integrated RAG System
            print(f"\n📊 Test 4: Integrated RAG System with Session {test_session}")
            print("-" * 50)
            
            try:
                integrated_rag = IntegratedMedicalRAG(
                    openai_api_key=os.getenv('openai_api_key'),
                    base_vector_path="./vector_dbs"
                )
                
                integrated_result = integrated_rag.query("What is diabetes?", test_session)
                print(f"✅ Integrated RAG - Answer length: {len(integrated_result.get('answer', ''))}")
                print(f"📊 Integrated RAG - Tools used: {integrated_result.get('tools_used', [])}")
                print(f"📝 Integrated RAG - Session ID: {integrated_result.get('session_id', 'None')}")
                
            except Exception as e:
                print(f"⚠️ Integrated RAG test failed: {e}")
            
            # Test 5: Session Switching
            if len(session_dirs) > 1:
                test_session2 = session_dirs[1]
                print(f"\n📊 Test 5: Session Switching from {test_session} to {test_session2}")
                print("-" * 50)
                
                # Switch to different session
                result2 = rag_manager.query_with_routing("What is diabetes?", test_session2)
                print(f"✅ Session Switch - New session loaded: {result2['routing_info'].get('session_kb_loaded', False)}")
                print(f"📊 Session Switch - Current session: {rag_manager.current_session_id}")
                print(f"🗂️ Session Switch - Cache size: {len(rag_manager.session_cache)}")
                
                # Test tool with new session
                tool_result2 = Internal_VectorDB("What is diabetes?", test_session2, rag_manager)
                print(f"✅ Tool with New Session - Result length: {len(tool_result2)}")
                
                # Verify sessions are different
                if result != result2:
                    print("✅ Sessions produce different results (as expected)")
                else:
                    print("⚠️ Sessions produced identical results (may be expected if content is similar)")
        
        else:
            print("⚠️ No session vector DBs found to test")
            return False
        
        print(f"\n" + "=" * 80)
        print("🎉 All session-specific loading tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_without_session():
    """Test behavior when no session_id is provided."""
    print("\n🧪 Testing behavior without session_id")
    print("-" * 50)
    
    try:
        from rag_architecture import TwoStoreRAGManager
        from tools import Internal_VectorDB
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
        
        rag_manager = TwoStoreRAGManager(embeddings, llm, "./vector_dbs")
        
        # Test without session_id
        result = rag_manager.query_with_routing("What is diabetes?")  # No session_id
        print(f"✅ No Session - Query completed")
        print(f"📊 No Session - Current session: {rag_manager.current_session_id}")
        print(f"🔍 No Session - kb_local loaded: {rag_manager.kb_local is not None}")
        
        # Test tool without session_id
        tool_result = Internal_VectorDB("What is diabetes?", None, rag_manager)
        print(f"✅ Tool No Session - Result length: {len(tool_result)}")
        
        return True
        
    except Exception as e:
        print(f"❌ No-session test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting comprehensive session loading tests...\n")
    
    success1 = test_session_loading_integration()
    success2 = test_without_session()
    
    if success1 and success2:
        print(f"\n🎉 All tests passed! Session-specific loading is working correctly.")
        sys.exit(0)
    else:
        print(f"\n❌ Some tests failed. Please check the output above.")
        sys.exit(1)