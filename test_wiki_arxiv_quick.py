#!/usr/bin/env python3
"""
Quick test of Wikipedia and ArXiv in the actual tools
"""
import sys
import os

print("Testing Wikipedia and ArXiv Tools")
print("=" * 60)

# Test Wikipedia_Search
print("\n📚 Testing Wikipedia_Search tool...")
try:
    from tools import Wikipedia_Search
    
    result = Wikipedia_Search("Type 2 diabetes symptoms")
    
    if result and "error" not in result.lower():
        print("✅ Wikipedia_Search WORKING!")
        print(f"   Result length: {len(result)} chars")
        print(f"   Preview: {result[:150]}...")
    else:
        print(f"❌ Wikipedia_Search returned error: {result}")
        
except Exception as e:
    print(f"❌ Wikipedia_Search FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test ArXiv_Search
print("\n📖 Testing ArXiv_Search tool...")
try:
    from tools import ArXiv_Search
    
    result = ArXiv_Search("diabetes treatment research")
    
    if result and "error" not in result.lower():
        print("✅ ArXiv_Search WORKING!")
        print(f"   Result length: {len(result)} chars")
        print(f"   Preview: {result[:150]}...")
    else:
        print(f"❌ ArXiv_Search returned error: {result}")
        
except Exception as e:
    print(f"❌ ArXiv_Search FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test Enhanced Tools
print("\n🔧 Testing Enhanced Tools...")
try:
    from enhanced_tools import enhanced_wikipedia_search, enhanced_arxiv_search
    
    print("\nTesting enhanced_wikipedia_search...")
    wiki_result = enhanced_wikipedia_search("diabetes")
    if wiki_result and 'content' in wiki_result:
        print("✅ enhanced_wikipedia_search WORKING!")
        print(f"   Content length: {len(wiki_result['content'])} chars")
    else:
        print("❌ enhanced_wikipedia_search failed")
    
    print("\nTesting enhanced_arxiv_search...")
    arxiv_result = enhanced_arxiv_search("diabetes")
    if arxiv_result and 'content' in arxiv_result:
        print("✅ enhanced_arxiv_search WORKING!")
        print(f"   Content length: {len(arxiv_result['content'])} chars")
    else:
        print("❌ enhanced_arxiv_search failed")
        
except Exception as e:
    print(f"❌ Enhanced tools test FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
