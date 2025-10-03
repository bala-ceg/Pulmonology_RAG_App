#!/usr/bin/env python3
"""
RAG External Knowledge Base Setup Utility
==========================================

This script helps initialize and manage the external knowledge base with
Wikipedia and arXiv content for the two-store RAG architecture.
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
    print("Please install dependencies: pip install -r requirements.txt")
    RAG_AVAILABLE = False

def setup_external_kb():
    """Setup the external knowledge base with medical content."""
    if not RAG_AVAILABLE:
        return False
    
    # Load environment variables
    load_dotenv()
    
    # Initialize components
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
        
        print("🚀 Setting up external knowledge base...")
        
        # Enhanced medical topics for Wikipedia - more specific for better answers
        medical_topics = [
            # Diabetes and endocrine (MORE SPECIFIC)
            "diabetes mellitus",
            "type 2 diabetes", 
            "type 1 diabetes",
            "diabetic complications",
            "insulin resistance",
            "hyperglycemia",
            "hypoglycemia",
            
            # Cardiovascular
            "cardiovascular disease",
            "myocardial infarction", 
            "heart failure",
            "coronary artery disease",
            "hypertension",
            "stroke",
            
            # Respiratory  
            "chronic obstructive pulmonary disease",
            "asthma",
            "pneumonia",
            "pulmonary embolism",
            
            # General medicine
            "obesity",
            "metabolic syndrome", 
            "chronic kidney disease",
            "anemia",
            "sepsis",
            "intensive care medicine"
        ]
        
        # Check if external KB already has content
        if rag_manager.has_external_content():
            count = rag_manager.get_external_content_count()
            print(f"📊 External KB already contains {count} documents")
            print("⚠️  Use 'force' command to rebuild: python setup_external_kb.py force")
            return True
        
        print(f"📚 Loading Wikipedia content for {len(medical_topics)} medical topics...")
        rag_manager.load_wikipedia_content(medical_topics, max_docs_per_topic=3)
        
        # Load some medical research from arXiv
        arxiv_queries = [
            # Clinical medicine focus
            "diabetes mellitus diagnosis treatment",
            "type 2 diabetes complications management", 
            "cardiovascular disease risk factors",
            "hypertension treatment guidelines",
            "COPD diagnosis management",
            "asthma treatment protocols",
            "sepsis early recognition treatment",
            "chronic kidney disease progression",
            "metabolic syndrome diagnosis",
            "clinical decision support systems",
            "medical diagnosis algorithms",
            "healthcare predictive modeling"
        ]
        
        print(f"🔬 Loading arXiv research for {len(arxiv_queries)} medical AI topics...")
        rag_manager.load_arxiv_content(arxiv_queries, max_docs_per_query=2)
        
        print("✅ External knowledge base setup completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error setting up external KB: {e}")
        return False

def setup_external_kb_force():
    """Force setup external knowledge base, even if content already exists."""
    if not RAG_AVAILABLE:
        return False
    
    # Load environment variables
    load_dotenv()
    
    # Initialize components
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
        
        print("🚀 Force setting up external knowledge base...")
        
        # Enhanced medical topics for Wikipedia - more specific for better answers
        medical_topics = [
            # Diabetes and endocrine (MORE SPECIFIC)
            "diabetes mellitus",
            "type 2 diabetes", 
            "type 1 diabetes",
            "diabetic complications",
            "insulin resistance",
            "hyperglycemia",
            "hypoglycemia",
            
            # Cardiovascular
            "cardiovascular disease",
            "myocardial infarction", 
            "heart failure",
            "coronary artery disease",
            "hypertension",
            "stroke",
            
            # Respiratory  
            "chronic obstructive pulmonary disease",
            "asthma",
            "pneumonia",
            "pulmonary embolism",
            
            # General medicine
            "obesity",
            "metabolic syndrome", 
            "chronic kidney disease",
            "anemia",
            "sepsis",
            "intensive care medicine"
        ]
        
        print(f"📚 Force loading Wikipedia content for {len(medical_topics)} medical topics...")
        rag_manager.load_wikipedia_content(medical_topics, max_docs_per_topic=3, force_reload=True)
        
        # Enhanced medical research queries for arXiv - more clinically relevant
        arxiv_queries = [
            # Clinical medicine focus
            "diabetes mellitus diagnosis treatment",
            "type 2 diabetes complications management", 
            "cardiovascular disease risk factors",
            "hypertension treatment guidelines",
            "COPD diagnosis management",
            "asthma treatment protocols",
            "sepsis early recognition treatment",
            "chronic kidney disease progression",
            "metabolic syndrome diagnosis",
            "clinical decision support systems",
            "medical diagnosis algorithms",
            "healthcare predictive modeling"
        ]
        
        print(f"🔬 Force loading arXiv research for {len(arxiv_queries)} medical AI topics...")
        rag_manager.load_arxiv_content(arxiv_queries, max_docs_per_query=2, force_reload=True)
        
        print("✅ External knowledge base force setup completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error force setting up external KB: {e}")
        return False

def check_kb_status():
    """Check the status of both knowledge bases."""
    if not RAG_AVAILABLE:
        return
    
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
        
        print("📊 Knowledge Base Status:")
        print("-" * 40)
        
        # Check kb_local
        if rag_manager.kb_local:
            try:
                local_count = rag_manager.kb_local._collection.count()
                print(f"📚 Local KB (kb_local): {local_count} documents")
            except:
                print("📚 Local KB (kb_local): Available but cannot count documents")
        else:
            print("📚 Local KB (kb_local): Not initialized")
        
        # Check kb_external
        if rag_manager.kb_external:
            try:
                external_count = rag_manager.kb_external._collection.count()
                print(f"🌐 External KB (kb_external): {external_count} documents")
            except:
                print("🌐 External KB (kb_external): Available but cannot count documents")
        else:
            print("🌐 External KB (kb_external): Not initialized")
        
        # Check lexical gate
        if rag_manager.lexical_gate.is_fitted:
            gate_docs = len(rag_manager.lexical_gate.local_documents)
            print(f"🚪 Lexical Gate: Fitted with {gate_docs} local documents")
        else:
            print("🚪 Lexical Gate: Not fitted")
        
        print("-" * 40)
        
    except Exception as e:
        print(f"❌ Error checking KB status: {e}")

def test_query_routing():
    """Test the query routing functionality."""
    if not RAG_AVAILABLE:
        return
    
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
        
        test_queries = [
            "What is pulmonology?",
            "How does machine learning help in medical diagnosis?",
            "What are the symptoms of heart disease?",
            "Explain artificial intelligence in healthcare",
            "What is the treatment for asthma?"
        ]
        
        print("🧪 Testing Query Routing:")
        print("=" * 50)
        
        for query in test_queries:
            print(f"\n📝 Query: {query}")
            
            # Test lexical gate routing decision
            query_local_first, similarity = rag_manager.lexical_gate.should_query_local_first(query)
            print(f"🚪 Gate Decision: {'Local first' if query_local_first else 'External first'} (similarity: {similarity:.3f})")
            
            # Test full query
            result = rag_manager.query_with_routing(query)
            
            if result['responses']:
                print(f"📊 Responses: {len(result['responses'])}")
                for i, resp in enumerate(result['responses'][:1], 1):  # Show first response
                    print(f"   {i}. {resp['source']} (confidence: {resp['confidence']})")
                    print(f"      Preview: {resp['content'][:100]}...")
            else:
                print("📊 No responses found")
            
            print("-" * 30)
            
    except Exception as e:
        print(f"❌ Error testing query routing: {e}")

if __name__ == "__main__":
    print("🏥 RAG External Knowledge Base Setup Utility")
    print("=" * 50)
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python setup_external_kb.py setup    - Initialize external KB")
        print("  python setup_external_kb.py status   - Check KB status")
        print("  python setup_external_kb.py test     - Test query routing")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "setup":
        success = setup_external_kb()
        sys.exit(0 if success else 1)
    elif command == "force":
        # Force setup - rebuild even if content exists
        print("🔄 Force rebuilding external KB...")
        success = setup_external_kb_force()
        sys.exit(0 if success else 1)
    elif command == "status":
        check_kb_status()
    elif command == "test":
        test_query_routing()
    else:
        print(f"❌ Unknown command: {command}")
        sys.exit(1)