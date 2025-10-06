#!/usr/bin/env python3
"""
Test UI Integration for Direct API Tools
========================================

This script simulates the Flask /data route logic to verify that
research queries are correctly routed to direct API tools instead
of the old vector database system.
"""

import json
import time
from typing import Dict, Any

# Import the direct API tools
try:
    from direct_api_tools import search_arxiv_safe, search_wikipedia_safe
    DIRECT_API_AVAILABLE = True
    print("✅ Direct API tools loaded successfully")
except ImportError:
    print("⚠️ Direct API tools not available")
    DIRECT_API_AVAILABLE = False


def is_research_query(query: str) -> bool:
    """Check if query is asking for recent research or studies"""
    research_keywords = ['latest', 'recent', 'research', 'study', 'paper', 'findings', 'oncology', 'cancer']
    return any(keyword in query.lower() for keyword in research_keywords)


def simulate_flask_route(query: str, user_id: str = "test_user") -> Dict[str, Any]:
    """
    Simulate the Flask /data route logic with direct API integration
    """
    print(f"\n🔍 Processing query: '{query}'")
    print(f"👤 User ID: {user_id}")
    
    # Check if this is a research query that should use direct APIs
    if DIRECT_API_AVAILABLE and is_research_query(query):
        print("🔬 Detected research query - using direct API tools")
        
        start_time = time.time()
        
        # For research queries, prioritize ArXiv for medical research
        if any(term in query.lower() for term in ['oncology', 'cancer', 'treatment', 'therapy']):
            print("📚 Using ArXiv search for medical research...")
            try:
                response = search_arxiv_safe(query)
                processing_time = time.time() - start_time
                
                return {
                    "status": "success",
                    "source": "direct_api_arxiv",
                    "response": response,
                    "processing_time": f"{processing_time:.2f}s",
                    "fresh_content": True
                }
            except Exception as e:
                print(f"❌ ArXiv search failed: {e}")
                # Fall through to Wikipedia or existing system
        
        # Try Wikipedia for general medical topics
        print("🌐 Trying Wikipedia search...")
        try:
            response = search_wikipedia_safe(query)
            processing_time = time.time() - start_time
            
            return {
                "status": "success", 
                "source": "direct_api_wikipedia",
                "response": response,
                "processing_time": f"{processing_time:.2f}s",
                "fresh_content": True
            }
        except Exception as e:
            print(f"❌ Wikipedia search failed: {e}")
    
    # Fallback to existing RAG system (simulated)
    print("🗃️ Using existing RAG system (vector database)")
    return {
        "status": "success",
        "source": "vector_database",
        "response": f"[Simulated vector DB response for: '{query}']",
        "processing_time": "0.85s",
        "fresh_content": False
    }


def main():
    """Test the integration with various query types"""
    
    test_queries = [
        "latest oncology research",
        "recent cancer studies", 
        "what is diabetes",
        "research on covid treatments",
        "define hypertension",
        "latest findings in cardiology"
    ]
    
    print("=" * 60)
    print("Testing UI Integration for Direct API Tools")
    print("=" * 60)
    
    results = []
    
    for query in test_queries:
        result = simulate_flask_route(query)
        results.append({
            "query": query,
            "result": result
        })
        
        print(f"📊 Result:")
        print(f"  - Source: {result['source']}")
        print(f"  - Fresh content: {result['fresh_content']}")
        print(f"  - Processing time: {result['processing_time']}")
        if len(result['response']) > 150:
            print(f"  - Response preview: {result['response'][:150]}...")
        else:
            print(f"  - Response: {result['response']}")
        print("-" * 50)
    
    # Summary
    print("\n📈 INTEGRATION TEST SUMMARY:")
    research_queries = [r for r in results if r['result']['fresh_content']]
    vector_queries = [r for r in results if not r['result']['fresh_content']]
    
    print(f"✅ Research queries using direct APIs: {len(research_queries)}")
    print(f"🗃️ Definition queries using vector DB: {len(vector_queries)}")
    
    if research_queries:
        print("\nResearch queries that got fresh content:")
        for r in research_queries:
            print(f"  - '{r['query']}' → {r['result']['source']}")
    
    if vector_queries:
        print("\nDefinition queries using existing system:")
        for r in vector_queries:
            print(f"  - '{r['query']}' → {r['result']['source']}")


if __name__ == "__main__":
    main()