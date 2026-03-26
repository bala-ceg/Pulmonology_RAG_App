#!/usr/bin/env python3
"""
Enhanced External KB Setup with Better Medical Content
=====================================================

This script improves the external knowledge base with more comprehensive
medical content that can answer specific questions about diseases and symptoms.
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

def setup_enhanced_external_kb():
    """Setup enhanced external knowledge base with comprehensive medical content."""
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
        
        print("🚀 Setting up enhanced external knowledge base...")
        
        # ENHANCED: More specific medical topics for Wikipedia
        medical_topics = [
            # Diabetes and endocrine
            "diabetes mellitus",
            "type 2 diabetes",
            "type 1 diabetes",
            "diabetic complications",
            "insulin resistance",
            "diabetic nephropathy",
            "diabetic retinopathy",
            "diabetic neuropathy",
            "hyperglycemia",
            "hypoglycemia",
            "endocrinology",
            
            # Cardiovascular
            "cardiovascular disease",
            "myocardial infarction",
            "heart failure",
            "coronary artery disease",
            "hypertension",
            "atrial fibrillation",
            "stroke",
            "atherosclerosis",
            
            # Respiratory
            "chronic obstructive pulmonary disease",
            "asthma",
            "pneumonia",
            "pulmonary embolism",
            "respiratory failure",
            "lung cancer",
            
            # Infectious diseases
            "sepsis",
            "pneumonia",
            "urinary tract infection",
            "meningitis",
            "tuberculosis",
            
            # General medicine
            "hypertension",
            "obesity",
            "metabolic syndrome",
            "chronic kidney disease",
            "liver disease",
            "anemia",
            "thyroid disease",
            "depression",
            "anxiety",
            
            # Critical care
            "intensive care medicine",
            "mechanical ventilation",
            "shock (medical)",
            "acute respiratory distress syndrome",
            "multi-organ failure"
        ]
        
        print(f"📚 Loading Wikipedia content for {len(medical_topics)} specific medical topics...")
        # Load in batches to avoid overwhelming the system
        batch_size = 10
        for i in range(0, len(medical_topics), batch_size):
            batch = medical_topics[i:i+batch_size]
            print(f"   Loading batch {i//batch_size + 1}: {', '.join(batch[:3])}{'...' if len(batch) > 3 else ''}")
            rag_manager.load_wikipedia_content(batch, max_docs_per_topic=2)
        
        # ENHANCED: More relevant medical research queries for arXiv
        arxiv_queries = [
            # Diabetes research
            "diabetes mellitus diagnosis treatment",
            "type 2 diabetes complications",
            "insulin resistance mechanisms",
            "diabetic nephropathy pathophysiology",
            
            # Cardiovascular research
            "myocardial infarction diagnosis",
            "heart failure treatment guidelines",
            "hypertension management",
            "cardiovascular risk assessment",
            
            # Respiratory research
            "COPD diagnosis treatment",
            "asthma management guidelines",
            "respiratory failure ventilation",
            "pulmonary embolism diagnosis",
            
            # General medical research
            "sepsis early recognition",
            "chronic kidney disease progression",
            "critical care monitoring",
            "emergency medicine protocols",
            
            # Medical informatics (keep some AI/ML focus)
            "clinical decision support systems",
            "electronic health records analysis",
            "medical diagnosis algorithms",
            "healthcare predictive modeling"
        ]
        
        print(f"🔬 Loading arXiv research for {len(arxiv_queries)} medical research topics...")
        # Load in smaller batches for arXiv (it's slower)
        batch_size = 5
        for i in range(0, len(arxiv_queries), batch_size):
            batch = arxiv_queries[i:i+batch_size]
            print(f"   Loading research batch {i//batch_size + 1}: {', '.join(batch[:2])}{'...' if len(batch) > 2 else ''}")
            rag_manager.load_arxiv_content(batch, max_docs_per_query=1)
        
        print("✅ Enhanced external knowledge base setup completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error setting up enhanced external KB: {e}")
        return False

def test_diabetes_query():
    """Test the enhanced setup with a diabetes query."""
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
        
        print("🧪 Testing diabetes query with enhanced external KB...")
        
        test_query = "What are the symptoms of Type-2 diabetes?"
        print(f"Query: {test_query}")
        
        # Test just external KB query
        if rag_manager.kb_external:
            external_response = rag_manager._query_kb_external(test_query)
            if external_response:
                print("📊 External KB Response Preview:")
                preview = external_response['result'][:200] + "..." if len(external_response['result']) > 200 else external_response['result']
                print(f"   {preview}")
                print(f"📚 Citations found: {len(external_response.get('citations', []))}")
            else:
                print("❌ No response from external KB")
        else:
            print("❌ External KB not available")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing diabetes query: {e}")
        return False

if __name__ == "__main__":
    print("🏥 Enhanced External Knowledge Base Setup")
    print("=" * 50)
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python enhanced_external_kb.py setup     - Setup enhanced external KB")
        print("  python enhanced_external_kb.py test      - Test diabetes query")
        print("  python enhanced_external_kb.py both      - Setup and test")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "setup":
        success = setup_enhanced_external_kb()
        sys.exit(0 if success else 1)
    elif command == "test":
        success = test_diabetes_query()
        sys.exit(0 if success else 1)
    elif command == "both":
        print("1️⃣ Setting up enhanced external KB...")
        setup_success = setup_enhanced_external_kb()
        
        if setup_success:
            print("\n2️⃣ Testing diabetes query...")
            test_success = test_diabetes_query()
            sys.exit(0 if test_success else 1)
        else:
            sys.exit(1)
    else:
        print(f"❌ Unknown command: {command}")
        sys.exit(1)