#!/usr/bin/env python3
"""
Debug Wikipedia Search Issue
===========================

This script debugs why Wikipedia search is only returning 30 characters
instead of the expected ~1200 characters.
"""

import os
from langchain_community.document_loaders import WikipediaLoader

def debug_wikipedia_search():
    """Debug the Wikipedia search to see what's happening"""
    
    print("🔍 Debugging Wikipedia Search Issue")
    print("=" * 50)
    
    query = "Explain the symptoms of Type-2 Diabetes"
    print(f"Query: '{query}'")
    
    try:
        # This is exactly what the Wikipedia_Search tool does
        print(f"\n📚 Loading Wikipedia documents...")
        loader = WikipediaLoader(query=query, load_max_docs=3)
        docs = loader.load()
        
        print(f"✅ Found {len(docs)} documents")
        
        if not docs:
            print("❌ No documents returned")
            return
        
        # Analyze each document
        for i, doc in enumerate(docs, 1):
            print(f"\n📄 Document {i}:")
            print(f"   Content length: {len(doc.page_content)} characters")
            print(f"   Content preview: {doc.page_content[:100]}...")
            print(f"   Metadata keys: {list(doc.metadata.keys())}")
            
            # Check specific metadata fields
            metadata = doc.metadata
            print(f"   Title: {metadata.get('title', 'N/A')}")
            print(f"   Source: {metadata.get('source', 'N/A')}")
            
        # Now test the _join_docs function
        print(f"\n🔧 Testing _join_docs function...")
        
        # Add source type metadata (as the tool does)
        for doc in docs:
            doc.metadata['source_type'] = 'wikipedia'
        
        # Import and use the _join_docs function
        import sys
        sys.path.append('.')
        from tools import _join_docs
        
        result = _join_docs(docs, max_chars=1200)
        
        print(f"✅ _join_docs result:")
        print(f"   Length: {len(result)} characters")
        print(f"   Preview: {result[:200]}...")
        
        if len(result) < 100:
            print("❌ Result is too short! Investigating...")
            
            # Check if documents have content
            total_content = sum(len(doc.page_content.strip()) for doc in docs)
            print(f"   Total raw content: {total_content} characters")
            
            if total_content == 0:
                print("❌ All documents have empty content!")
            else:
                print("🤔 Content exists but _join_docs is not processing it correctly")
                
                # Debug the _join_docs logic step by step
                combined_text = ""
                for doc in docs:
                    content = doc.page_content.strip()
                    print(f"   Doc content length: {len(content)}")
                    if content:
                        print(f"   Adding content: {content[:50]}...")
                        combined_text += content + "\n\n"
                    else:
                        print("   Doc has no content")
                
                print(f"   Manual combined length: {len(combined_text)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_simple_wikipedia_query():
    """Test with a simpler query"""
    
    print(f"\n" + "="*50)
    print("🧪 Testing Simple Wikipedia Query")
    print("="*50)
    
    simple_query = "diabetes"
    print(f"Simple query: '{simple_query}'")
    
    try:
        loader = WikipediaLoader(query=simple_query, load_max_docs=1)
        docs = loader.load()
        
        print(f"Found {len(docs)} documents")
        
        if docs:
            doc = docs[0]
            print(f"Content length: {len(doc.page_content)}")
            print(f"Title: {doc.metadata.get('title', 'N/A')}")
            print(f"First 200 chars: {doc.page_content[:200]}...")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    debug_wikipedia_search()
    test_simple_wikipedia_query()