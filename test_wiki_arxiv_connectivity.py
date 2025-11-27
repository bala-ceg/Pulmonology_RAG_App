#!/usr/bin/env python3
"""
Test Wikipedia and ArXiv API connectivity
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("🧪 Testing Wikipedia and ArXiv API Connectivity")
print("=" * 60)

# Test 1: Wikipedia
print("\n📚 Test 1: Wikipedia API")
print("-" * 60)
try:
    from langchain_community.document_loaders import WikipediaLoader
    
    print("✅ WikipediaLoader imported successfully")
    print("🔍 Searching for 'Type 2 Diabetes'...")
    
    loader = WikipediaLoader(query="Type 2 Diabetes", load_max_docs=2)
    docs = loader.load()
    
    if docs:
        print(f"✅ Wikipedia search successful!")
        print(f"   Found {len(docs)} documents")
        for i, doc in enumerate(docs, 1):
            title = doc.metadata.get('title', 'Unknown')
            content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            print(f"   Doc {i}: {title}")
            print(f"   Preview: {content_preview}\n")
    else:
        print("❌ No documents returned from Wikipedia")
        
except Exception as e:
    print(f"❌ Wikipedia test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: ArXiv
print("\n📖 Test 2: ArXiv API")
print("-" * 60)
try:
    from langchain_community.document_loaders import ArxivLoader
    
    print("✅ ArxivLoader imported successfully")
    print("🔍 Searching for 'diabetes treatment'...")
    
    loader = ArxivLoader(query="diabetes treatment", load_max_docs=2)
    docs = loader.load()
    
    if docs:
        print(f"✅ ArXiv search successful!")
        print(f"   Found {len(docs)} documents")
        for i, doc in enumerate(docs, 1):
            title = doc.metadata.get('Title', 'Unknown')
            authors = doc.metadata.get('Authors', 'Unknown')
            content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            print(f"   Doc {i}: {title}")
            print(f"   Authors: {authors}")
            print(f"   Preview: {content_preview}\n")
    else:
        print("❌ No documents returned from ArXiv")
        
except Exception as e:
    print(f"❌ ArXiv test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Network connectivity check
print("\n🌐 Test 3: Network Connectivity")
print("-" * 60)
try:
    import requests
    
    # Test Wikipedia
    print("Testing Wikipedia API endpoint...")
    wiki_response = requests.get("https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=diabetes&format=json", timeout=10)
    if wiki_response.status_code == 200:
        print(f"✅ Wikipedia API accessible (status: {wiki_response.status_code})")
        data = wiki_response.json()
        results = data.get('query', {}).get('search', [])
        print(f"   Found {len(results)} search results")
    else:
        print(f"❌ Wikipedia API returned status: {wiki_response.status_code}")
    
    # Test ArXiv
    print("\nTesting ArXiv API endpoint...")
    arxiv_response = requests.get("http://export.arxiv.org/api/query?search_query=diabetes&max_results=1", timeout=10)
    if arxiv_response.status_code == 200:
        print(f"✅ ArXiv API accessible (status: {arxiv_response.status_code})")
        if 'entry' in arxiv_response.text:
            print("   API returned valid XML with entries")
        else:
            print("   ⚠️ API response doesn't contain expected entries")
    else:
        print(f"❌ ArXiv API returned status: {arxiv_response.status_code}")
        
except Exception as e:
    print(f"❌ Network connectivity test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("✅ All connectivity tests completed!")
