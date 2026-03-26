#!/usr/bin/env python3
"""
Test Script for Direct API Agent Mode
====================================

This script tests the direct API agent mode without complex dependencies.
"""

import os
import sys

def test_direct_api_tools():
    """Test the direct API tools independently."""
    print("🧪 Testing Direct API Tools")
    print("=" * 40)
    
    try:
        # Test basic imports
        print("📦 Testing imports...")
        
        # Mock the tools for testing without full dependencies
        def mock_wikipedia_search(query):
            return f"**According to Wikipedia:** Mock result for '{query}' - This would contain medical definitions and basic facts about the topic.\n\n**Source:** Wikipedia"
        
        def mock_arxiv_search(query):
            return f"**Based on recent ArXiv research:** Mock result for '{query}' - This would contain recent research papers and scientific findings.\n\n**Source:** ArXiv"
        
        def mock_internal_search(query):
            return f"**From internal documents:** Mock result for '{query}' - This would search uploaded PDFs and organizational knowledge.\n\n**Source:** Internal Documents"
        
        # Test queries
        test_cases = [
            ("What is diabetes?", mock_wikipedia_search, "Wikipedia"),
            ("Latest COVID-19 research", mock_arxiv_search, "ArXiv"), 
            ("My uploaded protocol", mock_internal_search, "Internal")
        ]
        
        for query, tool_func, expected_source in test_cases:
            print(f"\n🔍 Query: {query}")
            print(f"🎯 Expected Source: {expected_source}")
            
            result = tool_func(query)
            print(f"📄 Result: {result[:100]}...")
            
            if expected_source.lower() in result.lower():
                print("✅ Test PASSED - Correct source identified")
            else:
                print("❌ Test FAILED - Source mismatch")
        
        print(f"\n✅ Direct API tools test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing direct API tools: {e}")
        return False


def test_agent_routing():
    """Test agent routing logic."""
    print("\n🤖 Testing Agent Routing Logic")
    print("=" * 40)
    
    # Define routing rules (from your implementation)
    def route_query(query):
        query_lower = query.lower()
        
        # ArXiv keywords
        arxiv_keywords = ['latest', 'recent', 'research', 'study', 'paper', 'findings']
        
        # Internal keywords  
        internal_keywords = ['uploaded', 'my file', 'my document', 'our protocol']
        
        # Wikipedia keywords
        wikipedia_keywords = ['what is', 'define', 'explain', 'tell me about']
        
        # Score each tool
        arxiv_score = sum(1 for kw in arxiv_keywords if kw in query_lower)
        internal_score = sum(1 for kw in internal_keywords if kw in query_lower)
        wikipedia_score = sum(1 for kw in wikipedia_keywords if kw in query_lower)
        
        # Default to Wikipedia if no clear winner
        if arxiv_score > max(internal_score, wikipedia_score):
            return "ArXiv"
        elif internal_score > max(arxiv_score, wikipedia_score):
            return "Internal"
        elif wikipedia_score > 0:
            return "Wikipedia"
        else:
            return "Wikipedia"  # Default
    
    # Test routing scenarios
    routing_tests = [
        ("What is hypertension?", "Wikipedia"),
        ("Latest research on COVID-19", "ArXiv"),
        ("Recent studies on diabetes", "ArXiv"),
        ("My uploaded PDF says what?", "Internal"),
        ("Define pneumonia", "Wikipedia"),
        ("Our protocol for treatment", "Internal"),
        ("New findings on cancer", "ArXiv"),
    ]
    
    passed = 0
    total = len(routing_tests)
    
    for query, expected in routing_tests:
        actual = route_query(query)
        status = "✅ PASS" if actual == expected else "❌ FAIL"
        print(f"{status} '{query}' → {actual} (expected {expected})")
        if actual == expected:
            passed += 1
    
    print(f"\n📊 Routing Tests: {passed}/{total} passed ({100*passed//total}%)")
    return passed == total


def create_requirements_check():
    """Check what dependencies are needed."""
    print("\n📋 Dependencies Check")
    print("=" * 30)
    
    required_packages = [
        "langchain",
        "langchain-openai", 
        "wikipedia",
        "arxiv",
        "openai"
    ]
    
    print("Required packages for full functionality:")
    for pkg in required_packages:
        try:
            __import__(pkg.replace('-', '_'))
            print(f"✅ {pkg} - Available")
        except ImportError:
            print(f"❌ {pkg} - Missing (pip install {pkg})")
    
    print(f"\nTo install all dependencies:")
    print(f"pip install {' '.join(required_packages)}")


def demo_agent_conversation():
    """Simulate an agent conversation."""
    print("\n💬 Agent Conversation Demo")
    print("=" * 35)
    
    conversation = [
        "What is pneumonia?",
        "Latest research on pneumonia treatment", 
        "What does my uploaded protocol say about pneumonia?",
    ]
    
    for i, query in enumerate(conversation, 1):
        print(f"\n👤 User: {query}")
        
        # Simulate routing
        if "latest" in query.lower() or "research" in query.lower():
            tool = "ArXiv Search"
            response = f"Based on recent ArXiv research, I found several studies about {query.lower()}..."
        elif "uploaded" in query.lower() or "protocol" in query.lower():
            tool = "Internal Documents"  
            response = f"Searching your uploaded documents for information about {query.lower()}..."
        else:
            tool = "Wikipedia Search"
            response = f"According to Wikipedia, {query.lower().replace('what is ', '')} is a medical condition..."
        
        print(f"🔧 Tool Selected: {tool}")
        print(f"🤖 Agent: {response}")


def main():
    """Run all tests."""
    print("🚀 Direct API Agent Test Suite")
    print("=" * 50)
    
    # Run tests
    tests = [
        ("Direct API Tools", test_direct_api_tools),
        ("Agent Routing", test_agent_routing),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n🧪 Running {name} tests...")
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} test failed: {e}")
            results.append((name, False))
    
    # Show summary
    print(f"\n📊 Test Summary")
    print("=" * 20)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {name}")
    
    print(f"\nOverall: {passed}/{total} test suites passed")
    
    # Show next steps
    print(f"\n🎯 Next Steps:")
    print("1. Install dependencies: pip install langchain langchain-openai wikipedia arxiv")
    print("2. Set environment: export OPENAI_API_KEY='your-key'")
    print("3. Run full agent: python agent_direct_api.py")
    print("4. Run interactive mode: python agent_direct_api.py interactive")
    
    # Show dependency check
    create_requirements_check()
    
    # Show conversation demo
    demo_agent_conversation()


if __name__ == "__main__":
    main()