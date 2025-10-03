#!/usr/bin/env python3
"""
External KB Persistence Management Utility
==========================================

This script helps manage the persistence of the external knowledge base.
"""

import os
import sys
from dotenv import load_dotenv

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from rag_architecture import TwoStoreRAGManager
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from setup_external_kb import setup_external_kb, setup_external_kb_force
    RAG_AVAILABLE = True
except ImportError as e:
    print(f"❌ RAG architecture not available: {e}")
    RAG_AVAILABLE = False

def check_persistence_status():
    """Check the current persistence status of external KB."""
    if not RAG_AVAILABLE:
        return False
    
    load_dotenv()
    
    try:
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
        
        print("🔍 External KB Persistence Status")
        print("=" * 40)
        
        # Check external KB directory
        kb_external_path = "./vector_dbs/kb_external"
        if os.path.exists(kb_external_path):
            files = os.listdir(kb_external_path)
            if files:
                print(f"✅ External KB directory exists with {len(files)} files")
                for file in files:
                    file_path = os.path.join(kb_external_path, file)
                    if os.path.isfile(file_path):
                        size_mb = os.path.getsize(file_path) / (1024 * 1024)
                        print(f"   📄 {file}: {size_mb:.2f} MB")
                    else:
                        print(f"   📁 {file}/")
            else:
                print("⚠️  External KB directory exists but is empty")
        else:
            print("❌ External KB directory not found")
        
        # Check if external KB loads correctly
        if rag_manager.kb_external:
            count = rag_manager.get_external_content_count()
            print(f"✅ External KB loaded successfully with {count} document chunks")
            
            # Check if it has both Wikipedia and arXiv content
            try:
                # Test retrieval for different source types
                retriever = rag_manager.kb_external.as_retriever(search_type="similarity", search_kwargs={"k": 5})
                sample_results = retriever.invoke("medical")
                
                wikipedia_count = sum(1 for doc in sample_results if doc.metadata.get('source_type') == 'wikipedia')
                arxiv_count = sum(1 for doc in sample_results if doc.metadata.get('source_type') == 'arxiv')
                
                print(f"📊 Sample content analysis:")
                print(f"   🌐 Wikipedia documents: {wikipedia_count}")
                print(f"   🔬 arXiv documents: {arxiv_count}")
                
            except Exception as e:
                print(f"⚠️  Error analyzing content: {e}")
                
        else:
            print("❌ External KB failed to load")
        
        # Check persistence behavior
        print(f"\n🔄 Persistence Behavior:")
        if rag_manager.has_external_content():
            print("✅ External KB has content - will skip reloading on startup")
            print("✅ New app launches will be fast (no Wikipedia/arXiv downloads)")
        else:
            print("⚠️  External KB is empty - will reload content on startup")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking persistence status: {e}")
        return False

def force_rebuild_external_kb():
    """Force rebuild the external KB with fresh content."""
    if not RAG_AVAILABLE:
        return False
    
    load_dotenv()
    
    try:
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
        
        print("🔄 Force Rebuilding External KB")
        print("=" * 40)
        
        # Load content with force_reload=True
        medical_topics = [
            "diabetes mellitus", "type 2 diabetes", "type 1 diabetes",
            "cardiovascular disease", "hypertension", "asthma",
            "chronic obstructive pulmonary disease", "pneumonia"
        ]
        
        print(f"📚 Force loading {len(medical_topics)} Wikipedia topics...")
        rag_manager.load_wikipedia_content(medical_topics, max_docs_per_topic=2, force_reload=True)
        
        arxiv_queries = [
            "diabetes mellitus diagnosis treatment",
            "cardiovascular disease risk factors", 
            "COPD diagnosis management",
            "clinical decision support systems"
        ]
        
        print(f"🔬 Force loading {len(arxiv_queries)} arXiv queries...")
        rag_manager.load_arxiv_content(arxiv_queries, max_docs_per_query=1, force_reload=True)
        
        final_count = rag_manager.get_external_content_count()
        print(f"✅ External KB rebuilt with {final_count} documents")
        
        return True
        
    except Exception as e:
        print(f"❌ Error rebuilding external KB: {e}")
        return False

if __name__ == "__main__":
    print("🏥 External KB Persistence Management")
    print("=" * 50)
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python manage_external_kb.py status    - Check persistence status")
        print("  python manage_external_kb.py rebuild   - Force rebuild external KB")
        print("  python manage_external_kb.py setup     - Setup external KB (skip if exists)")
        print("  python manage_external_kb.py force     - Force setup external KB")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "status":
        success = check_persistence_status()
        sys.exit(0 if success else 1)
    elif command == "rebuild":
        success = force_rebuild_external_kb()
        sys.exit(0 if success else 1)
    elif command == "setup":
        success = setup_external_kb()
        sys.exit(0 if success else 1)
    elif command == "force":
        success = setup_external_kb_force()
        sys.exit(0 if success else 1)
    else:
        print(f"❌ Unknown command: {command}")
        sys.exit(1)