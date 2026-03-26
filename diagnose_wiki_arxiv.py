#!/usr/bin/env python3
"""
Diagnostic script for Wikipedia and ArXiv issues
Run this to identify the specific problem
"""
import sys
import os
import traceback

print("=" * 70)
print("WIKIPEDIA & ARXIV DIAGNOSTIC TOOL")
print("=" * 70)

# Test 1: Check Python environment
print("\n📦 Step 1: Checking Python Environment")
print("-" * 70)
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

# Test 2: Check required packages
print("\n📚 Step 2: Checking Required Packages")
print("-" * 70)
packages_to_check = [
    'wikipedia',
    'arxiv',
    'langchain',
    'langchain_community',
    'requests'
]

for package in packages_to_check:
    try:
        mod = __import__(package)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✅ {package:25} version: {version}")
    except ImportError as e:
        print(f"❌ {package:25} NOT INSTALLED - {e}")

# Test 3: Test direct API calls
print("\n🌐 Step 3: Testing Direct API Access")
print("-" * 70)

# Wikipedia API test
print("\nTesting Wikipedia API...")
try:
    import requests
    import json
    
    # Add user-agent to comply with Wikipedia's policy
    headers = {
        'User-Agent': 'Medical RAG Application/1.0 (Educational/Research Purpose)'
    }
    
    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "list": "search",
            "srsearch": "diabetes",
            "format": "json"
        },
        headers=headers,
        timeout=10
    )
    
    if response.status_code == 200:
        data = response.json()
        results = data.get('query', {}).get('search', [])
        print(f"✅ Wikipedia API accessible")
        print(f"   Status code: {response.status_code}")
        print(f"   Search results: {len(results)} items found")
        if results:
            print(f"   Sample result: {results[0].get('title', 'N/A')}")
    else:
        print(f"❌ Wikipedia API returned status: {response.status_code}")
        print(f"   Response: {response.text[:200]}")
        
except Exception as e:
    print(f"❌ Wikipedia API test failed: {e}")
    traceback.print_exc()

# ArXiv API test
print("\nTesting ArXiv API...")
try:
    import requests
    import xml.etree.ElementTree as ET
    
    response = requests.get(
        "http://export.arxiv.org/api/query",
        params={
            "search_query": "diabetes",
            "max_results": 1
        },
        timeout=10
    )
    
    if response.status_code == 200:
        print(f"✅ ArXiv API accessible")
        print(f"   Status code: {response.status_code}")
        
        # Try to parse XML
        try:
            root = ET.fromstring(response.content)
            entries = root.findall('{http://www.w3.org/2005/Atom}entry')
            print(f"   Found {len(entries)} paper(s)")
            if entries:
                title_elem = entries[0].find('{http://www.w3.org/2005/Atom}title')
                if title_elem is not None:
                    print(f"   Sample paper: {title_elem.text[:80]}...")
        except Exception as parse_error:
            print(f"   ⚠️ Could not parse XML response: {parse_error}")
    else:
        print(f"❌ ArXiv API returned status: {response.status_code}")
        print(f"   Response: {response.text[:200]}")
        
except Exception as e:
    print(f"❌ ArXiv API test failed: {e}")
    traceback.print_exc()

# Test 4: Test LangChain loaders
print("\n🔧 Step 4: Testing LangChain Loaders")
print("-" * 70)

# Wikipedia Loader test
print("\nTesting WikipediaLoader...")
try:
    from langchain_community.document_loaders import WikipediaLoader
    
    print("✅ WikipediaLoader imported")
    print("   Attempting to load 'Type 2 Diabetes' (10 second timeout)...")
    
    import signal
    
    class LoaderTimeout(Exception):
        pass
    
    def timeout_handler(signum, frame):
        raise LoaderTimeout("Loader timed out")
    
    # Set 10 second timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(10)
    
    try:
        loader = WikipediaLoader(query="Type 2 Diabetes", load_max_docs=1)
        docs = loader.load()
        signal.alarm(0)  # Cancel timeout
        
        if docs:
            print(f"✅ WikipediaLoader working!")
            print(f"   Loaded {len(docs)} document(s)")
            print(f"   Title: {docs[0].metadata.get('title', 'N/A')}")
            print(f"   Content length: {len(docs[0].page_content)} chars")
        else:
            print("⚠️ WikipediaLoader returned no documents")
    except LoaderTimeout:
        signal.alarm(0)
        print("❌ WikipediaLoader TIMED OUT after 10 seconds")
        print("   This indicates the loader is hanging!")
        
except Exception as e:
    signal.alarm(0)
    print(f"❌ WikipediaLoader test failed: {e}")
    traceback.print_exc()

# ArXiv Loader test
print("\nTesting ArxivLoader...")
try:
    from langchain_community.document_loaders import ArxivLoader
    
    print("✅ ArxivLoader imported")
    print("   Attempting to load 'diabetes treatment' (10 second timeout)...")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(10)
    
    try:
        loader = ArxivLoader(query="diabetes treatment", load_max_docs=1)
        docs = loader.load()
        signal.alarm(0)  # Cancel timeout
        
        if docs:
            print(f"✅ ArxivLoader working!")
            print(f"   Loaded {len(docs)} document(s)")
            print(f"   Title: {docs[0].metadata.get('Title', 'N/A')}")
            print(f"   Content length: {len(docs[0].page_content)} chars")
        else:
            print("⚠️ ArxivLoader returned no documents")
    except LoaderTimeout:
        signal.alarm(0)
        print("❌ ArxivLoader TIMED OUT after 10 seconds")
        print("   This indicates the loader is hanging!")
        
except Exception as e:
    signal.alarm(0)
    print(f"❌ ArxivLoader test failed: {e}")
    traceback.print_exc()

# Test 5: Check network/firewall
print("\n🔒 Step 5: Network/Firewall Check")
print("-" * 70)
try:
    import socket
    
    # Test DNS resolution
    print("Testing DNS resolution...")
    try:
        ip = socket.gethostbyname("en.wikipedia.org")
        print(f"✅ en.wikipedia.org resolves to {ip}")
    except socket.gaierror as e:
        print(f"❌ DNS resolution failed for en.wikipedia.org: {e}")
    
    try:
        ip = socket.gethostbyname("export.arxiv.org")
        print(f"✅ export.arxiv.org resolves to {ip}")
    except socket.gaierror as e:
        print(f"❌ DNS resolution failed for export.arxiv.org: {e}")
        
    # Test connectivity
    print("\nTesting network connectivity...")
    try:
        socket.create_connection(("en.wikipedia.org", 443), timeout=5)
        print("✅ Can connect to en.wikipedia.org:443")
    except (socket.timeout, socket.error) as e:
        print(f"❌ Cannot connect to en.wikipedia.org:443 - {e}")
        
    try:
        socket.create_connection(("export.arxiv.org", 80), timeout=5)
        print("✅ Can connect to export.arxiv.org:80")
    except (socket.timeout, socket.error) as e:
        print(f"❌ Cannot connect to export.arxiv.org:80 - {e}")
        
except Exception as e:
    print(f"❌ Network check failed: {e}")
    traceback.print_exc()

# Summary
print("\n" + "=" * 70)
print("DIAGNOSIS COMPLETE")
print("=" * 70)
print("\nPlease review the results above to identify the issue.")
print("Common issues and solutions:")
print("  • If APIs are accessible but loaders timeout:")
print("    → Check if you need to reinstall langchain-community")
print("    → Try: pip install --upgrade langchain-community")
print("  • If DNS/connectivity fails:")
print("    → Check your internet connection")
print("    → Check if corporate firewall is blocking access")
print("  • If packages are missing:")
print("    → Install with: pip install wikipedia arxiv langchain-community")
print("\n" + "=" * 70)
