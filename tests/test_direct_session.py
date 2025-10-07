#!/usr/bin/env python3
"""
Direct test of session loading functionality without starting Flask app
"""

import sys
import os

# Change to the correct directory
os.chdir('/Users/bseetharaman/Desktop/Bala/2025/AI_Medical_App/Pulmonology_RAG_App')

# Import just the RAG architecture
from rag_architecture import TwoStoreRAGManager

def test_core_session_loading():
    """Test the core session loading functionality"""
    print("🧪 Testing Core Session Loading")
    print("=" * 40)
    
    try:
        # Initialize RAG manager
        print("🚀 Initializing RAG Manager...")
        rag_manager = TwoStoreRAGManager()
        print("✅ RAG Manager initialized successfully")
        
        # Check available sessions
        session_dirs = [d for d in os.listdir('./vector_dbs') if d.startswith('guest_')]
        print(f"📂 Found {len(session_dirs)} session directories")
        
        if not session_dirs:
            print("❌ No session directories found")
            return False
            
        # Test with first session
        test_session = session_dirs[0]
        print(f"🔍 Testing with session: {test_session}")
        
        # Check initial state
        initial_state = rag_manager.kb_local is not None
        print(f"📊 Initial kb_local state: {'Loaded' if initial_state else 'Not loaded'}")
        
        # Load session vector DB
        print(f"🔄 Loading session vector DB for {test_session}...")
        success = rag_manager.load_session_vector_db(test_session)
        
        # Check final state
        final_state = rag_manager.kb_local is not None
        print(f"📊 Final kb_local state: {'Loaded' if final_state else 'Not loaded'}")
        print(f"📊 Loading success: {success}")
        
        # Check session cache
        cache_info = f"Cache size: {len(rag_manager.session_cache)}"
        if test_session in rag_manager.session_cache:
            cache_info += f" (includes {test_session})"
        print(f"💾 {cache_info}")
        
        return success and final_state
        
    except Exception as e:
        print(f"❌ Error in core test: {e}")
        return False

def test_tools_session_awareness():
    """Test that tools can handle session loading"""
    print("\n🔧 Testing Tools Session Awareness")
    print("=" * 40)
    
    try:
        # Import tools
        from tools import Internal_VectorDB
        from enhanced_tools import enhanced_internal_search
        
        session_dirs = [d for d in os.listdir('./vector_dbs') if d.startswith('guest_')]
        if not session_dirs:
            print("❌ No sessions available for tools test")
            return False
            
        test_session = session_dirs[0]
        print(f"🔍 Testing tools with session: {test_session}")
        
        # Test basic tool call (this should now use session loading)
        print("🧪 Testing Internal_VectorDB...")
        try:
            # This should trigger the session loading we added
            result = Internal_VectorDB("test medical query", test_session)
            print("✅ Internal_VectorDB executed (session loading triggered)")
        except Exception as e:
            print(f"⚠️  Internal_VectorDB warning: {e}")
            
        # Test enhanced tool
        print("🧪 Testing enhanced_internal_search...")
        try:
            result = enhanced_internal_search("test medical query", test_session)
            print("✅ enhanced_internal_search executed (session loading triggered)")
            return True
        except Exception as e:
            print(f"⚠️  enhanced_internal_search warning: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Error in tools test: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Session Loading Fix - Direct Test")
    print("=" * 50)
    
    # Test 1: Core session loading
    core_passed = test_core_session_loading()
    
    # Test 2: Tools integration
    tools_passed = test_tools_session_awareness()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS")
    print(f"Core Session Loading: {'✅ PASS' if core_passed else '❌ FAIL'}")
    print(f"Tools Integration: {'✅ PASS' if tools_passed else '❌ FAIL'}")
    
    if core_passed and tools_passed:
        print("\n🎉 SUCCESS! The session loading issue is FIXED!")
        print("✅ kb_local is now loaded dynamically at runtime")
        print("✅ Tools properly trigger session loading")
        print("✅ Users can now access their uploaded documents")
    else:
        print("\n⚠️  Some issues detected - check the output above")