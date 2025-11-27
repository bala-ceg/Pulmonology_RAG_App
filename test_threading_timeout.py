#!/usr/bin/env python3
"""
Test threading-based timeout fix for Wikipedia and ArXiv searches
"""

import sys
import os

# Add the current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("🧪 Testing Threading-Based Timeout Fix for Wikipedia/ArXiv")
print("=" * 70)

# Test 1: Import enhanced_tools
print("\n1️⃣ Testing import of enhanced_tools...")
try:
    from enhanced_tools import enhanced_wikipedia_search, enhanced_arxiv_search
    print("✅ Import successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Wikipedia Search
print("\n2️⃣ Testing Wikipedia Search with threading timeout...")
try:
    result = enhanced_wikipedia_search("pneumonia treatment guidelines")
    print(f"✅ Wikipedia search completed!")
    print(f"   Summary length: {len(result.get('summary', ''))} chars")
    print(f"   Content length: {len(result.get('content', ''))} chars")
    if result.get('summary'):
        print(f"   Summary preview: {result['summary'][:100]}...")
except Exception as e:
    print(f"❌ Wikipedia search failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: ArXiv Search  
print("\n3️⃣ Testing ArXiv Search with threading timeout...")
try:
    result = enhanced_arxiv_search("COPD machine learning")
    print(f"✅ ArXiv search completed!")
    print(f"   Summary length: {len(result.get('summary', ''))} chars")
    print(f"   Content length: {len(result.get('content', ''))} chars")
    if result.get('summary'):
        print(f"   Summary preview: {result['summary'][:100]}...")
except Exception as e:
    print(f"❌ ArXiv search failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("✅ Threading timeout test complete!")
print("\nIf both searches completed without hanging, the fix is working.")
