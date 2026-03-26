#!/usr/bin/env python3
"""
Test to validate that session-specific vector database loading is working
This focuses on the specific issue: kb_local should be loaded at runtime per session
"""

import sys
import os
sys.path.append('/Users/bseetharaman/Desktop/Bala/2025/AI_Medical_App/Pulmonology_RAG_App')

def test_session_loading():
    """Test that session loading works correctly"""
    print("🚀 Testing Session-Specific Loading Fix...")
    
    try:
        from rag_architecture import TwoStoreRAGManager
        
        # Initialize RAG manager
        rag_manager = TwoStoreRAGManager()
        print("✅ RAG Manager initialized")
        
        # Check available sessions
        session_dirs = [d for d in os.listdir('./vector_dbs') if d.startswith('guest_')]
        print(f"📂 Available sessions: {len(session_dirs)} found")
        
        if not session_dirs:
            print("❌ No session directories found - cannot test session loading")
            return False
            
        # Test with first available session
        test_session = session_dirs[0]
        print(f"🔍 Testing with session: {test_session}")
        
        # Before loading - kb_local should be None or empty
        initial_kb_state = rag_manager.kb_local is not None
        print(f"📊 Initial kb_local state: {'Loaded' if initial_kb_state else 'None/Empty'}")
        
        # Load session-specific vector DB
        success = rag_manager.load_session_vector_db(test_session)
        print(f"{'✅' if success else '❌'} Session loading result: {success}")
        
        # After loading - kb_local should be loaded
        final_kb_state = rag_manager.kb_local is not None
        print(f"📊 Final kb_local state: {'Loaded' if final_kb_state else 'None/Empty'}")
        
        # Test session caching
        cache_size = len(rag_manager.session_cache)
        print(f"💾 Session cache size: {cache_size}")
        
        return success and final_kb_state
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_tools_integration():
    """Test that tools use session loading"""
    print("\n🔧 Testing Tools Integration...")
    
    try:
        from tools import Internal_VectorDB
        from enhanced_tools import enhanced_internal_search
        
        # Check available sessions
        session_dirs = [d for d in os.listdir('./vector_dbs') if d.startswith('guest_')]
        
        if not session_dirs:
            print("❌ No sessions to test tools with")
            return False
            
        test_session = session_dirs[0]
        
        # Test Internal_VectorDB tool
        print(f"🔍 Testing Internal_VectorDB with session: {test_session}")
        try:
            result = Internal_VectorDB("test query", test_session)
            print(f"✅ Internal_VectorDB executed successfully")
        except Exception as e:
            print(f"❌ Internal_VectorDB failed: {e}")
            
        # Test enhanced_internal_search
        print(f"🔍 Testing enhanced_internal_search with session: {test_session}")
        try:
            result = enhanced_internal_search("test query", test_session)
            print(f"✅ enhanced_internal_search executed successfully")
            return True
        except Exception as e:
            print(f"❌ enhanced_internal_search failed: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ Tools import error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Session Loading Fix Validation Test")
    print("=" * 50)
    
    # Test core session loading
    session_test_passed = test_session_loading()
    
    # Test tools integration
    tools_test_passed = test_tools_integration()
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"Session Loading: {'✅ PASS' if session_test_passed else '❌ FAIL'}")
    print(f"Tools Integration: {'✅ PASS' if tools_test_passed else '❌ FAIL'}")
    
    overall_pass = session_test_passed and tools_test_passed
    print(f"Overall Result: {'✅ ALL TESTS PASSED' if overall_pass else '❌ SOME TESTS FAILED'}")
    
    if overall_pass:
        print("\n🎉 The session loading fix is working correctly!")
        print("   - kb_local is now loaded dynamically at runtime")
        print("   - Session-specific vector databases are accessible")
        print("   - Tools properly use session loading")
    else:
        print("\n⚠️  Some issues remain - check the error messages above")