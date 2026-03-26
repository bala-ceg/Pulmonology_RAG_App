#!/usr/bin/env python3
"""
Quick Diabetes Content Addition to External KB
"""

import sys
import os
from dotenv import load_dotenv

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def add_diabetes_content():
    """Add specific diabetes content to existing external KB."""
    
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
        
        print("🍯 Adding Diabetes-Specific Content to External KB")
        print("=" * 50)
        
        # Add specific diabetes Wikipedia content
        diabetes_topics = [
            "diabetes mellitus",
            "type 2 diabetes", 
            "type 1 diabetes",
            "diabetic complications",
            "insulin resistance",
            "hyperglycemia",
            "hypoglycemia",
            "diabetic neuropathy",
            "diabetic retinopathy",
            "diabetic nephropathy"
        ]
        
        print(f"📚 Loading {len(diabetes_topics)} diabetes-specific Wikipedia topics...")
        rag_manager.load_wikipedia_content(diabetes_topics, max_docs_per_topic=3)
        
        # Add diabetes research
        diabetes_research = [
            "diabetes mellitus symptoms diagnosis",
            "type 2 diabetes complications management",
            "insulin resistance pathophysiology",
            "hyperglycemia treatment"
        ]
        
        print(f"🔬 Loading {len(diabetes_research)} diabetes research papers...")
        rag_manager.load_arxiv_content(diabetes_research, max_docs_per_query=2)
        
        print("✅ Successfully added diabetes-specific content!")
        
        # Test immediately
        print("\n🧪 Testing diabetes query...")
        test_query = "What are the symptoms of type 2 diabetes?"
        result = rag_manager._query_kb_external(test_query)
        
        if result and len(result['result']) > 100:
            print("✅ Diabetes query test successful!")
            preview = result['result'][:200] + "..." if len(result['result']) > 200 else result['result']
            print(f"Preview: {preview}")
        else:
            print("⚠️ Diabetes query test needs more content")
        
        return True
        
    except Exception as e:
        print(f"❌ Error adding diabetes content: {e}")
        return False

if __name__ == "__main__":
    add_diabetes_content()