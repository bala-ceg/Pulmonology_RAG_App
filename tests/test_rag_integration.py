#!/usr/bin/env python3
"""
Quick integration test for the RAG architecture
"""

import sys
import os

# Test imports
def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        from rag_architecture import TwoStoreRAGManager, TFIDFLexicalGate
        print("✅ RAG architecture modules imported successfully")
    except ImportError as e:
        print(f"❌ RAG architecture import failed: {e}")
        return False
    
    try:
        import sklearn
        print("✅ scikit-learn imported successfully")
    except ImportError as e:
        print(f"❌ scikit-learn import failed: {e}")
        return False
    
    try:
        from langchain_community.document_loaders import WikipediaLoader, ArxivLoader
        print("✅ LangChain community loaders imported successfully")
    except ImportError as e:
        print(f"❌ LangChain community loaders import failed: {e}")
        return False
    
    return True

def test_lexical_gate():
    """Test the TF-IDF lexical gate functionality"""
    print("\n🚪 Testing TF-IDF Lexical Gate...")
    
    try:
        from rag_architecture import TFIDFLexicalGate
        
        # Create a simple lexical gate
        gate = TFIDFLexicalGate(threshold=0.3)
        
        # Test documents
        test_docs = [
            "This is a document about cardiology and heart disease treatment",
            "Pulmonology focuses on lung diseases and respiratory conditions",
            "Machine learning applications in medical diagnosis are growing"
        ]
        
        # Build automation summary
        gate.build_automation_summary(test_docs)
        
        if gate.is_fitted:
            print("✅ Lexical gate fitted successfully")
        else:
            print("❌ Lexical gate fitting failed")
            return False
        
        # Test routing decisions
        test_queries = [
            "What is cardiology?",
            "Tell me about artificial intelligence",
            "How to treat heart disease?"
        ]
        
        for query in test_queries:
            query_local_first, similarity = gate.should_query_local_first(query)
            print(f"   Query: '{query}' → {'Local' if query_local_first else 'External'} first (sim: {similarity:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Lexical gate test failed: {e}")
        return False

def test_document_processing():
    """Test document processing functionality"""
    print("\n📄 Testing document processing...")
    
    try:
        from langchain_core.documents import Document
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        # Create test document
        test_content = """
        This is a test medical document about cardiovascular health.
        It contains information about heart disease prevention and treatment.
        Regular exercise and proper diet are important for heart health.
        """
        
        # Create document
        doc = Document(
            page_content=test_content,
            metadata={"source": "test_doc.txt", "type": "test"}
        )
        
        print("✅ Document creation successful")
        
        # Test text splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20
        )
        
        chunks = splitter.split_documents([doc])
        print(f"✅ Text splitting successful: {len(chunks)} chunks created")
        
        return True
        
    except Exception as e:
        print(f"❌ Document processing test failed: {e}")
        return False

def main():
    """Run all integration tests"""
    print("🏥 RAG Architecture Integration Test")
    print("=" * 40)
    
    tests = [
        ("Import Test", test_imports),
        ("Lexical Gate Test", test_lexical_gate),
        ("Document Processing Test", test_document_processing)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! RAG architecture is ready.")
        return True
    else:
        print("⚠️  Some tests failed. Check dependencies and configuration.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)