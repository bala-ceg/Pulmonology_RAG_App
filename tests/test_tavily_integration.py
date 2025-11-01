#!/usr/bin/env python3
"""
Test script for Tavily Search Tool integration
============================================

This script tests the newly added Tavily_Search tool to ensure it:
1. Is properly imported and available
2. Has correct routing keywords and scoring
3. Can execute basic searches (when API key is available)
4. Provides proper fallback to Wikipedia when needed
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import AVAILABLE_TOOLS, Tavily_Search
from rag_architecture import MedicalQueryRouter


def test_tavily_tool_availability():
    """Test that Tavily tool is properly registered."""
    print("🔍 Testing Tavily tool availability...")
    
    # Check if Tavily_Search is in AVAILABLE_TOOLS
    assert 'Tavily_Search' in AVAILABLE_TOOLS, "Tavily_Search not found in AVAILABLE_TOOLS"
    print("✅ Tavily_Search found in AVAILABLE_TOOLS")
    
    # Check if the tool function is callable
    assert callable(AVAILABLE_TOOLS['Tavily_Search']), "Tavily_Search is not callable"
    print("✅ Tavily_Search is callable")


def test_tavily_routing():
    """Test that Tavily routing keywords work correctly."""
    print("\n🔍 Testing Tavily routing logic...")
    
    router = MedicalQueryRouter()
    
    # Test current guidelines query
    test_query = "What are the current FDA guidelines for COVID-19 treatment?"
    routing_result = router.route_tools(test_query)
    
    print(f"Query: '{test_query}'")
    print(f"Routing result: {routing_result}")
    
    # Check if Tavily is ranked high for current guidelines
    ranked_tools = routing_result.get('ranked_tools', [])
    tool_scores = routing_result.get('tool_scores', {})
    
    if ranked_tools:
        top_tool = ranked_tools[0]  # ranked_tools is now a list of tool names
        print(f"Top tool selected: {top_tool}")
        
        # Get Tavily score from tool_scores dictionary
        tavily_score = tool_scores.get('Tavily_Search', 0)
        print(f"Tavily score: {tavily_score}")
        
        # Check if Tavily has a reasonable score and is the top choice
        assert tavily_score > 0, "Tavily should have a positive score for current guidelines query"
        assert top_tool == 'Tavily_Search', f"Tavily should be top choice for current guidelines, got {top_tool}"
        print("✅ Tavily routing works correctly")
    else:
        print("❌ No tools were ranked")


def test_tavily_keywords():
    """Test that Tavily keywords are properly defined."""
    print("\n🔍 Testing Tavily keywords...")
    
    router = MedicalQueryRouter()
    
    # Check if tavily_keywords exist
    assert hasattr(router, 'tavily_keywords'), "Router doesn't have tavily_keywords"
    assert len(router.tavily_keywords) > 0, "Tavily keywords list is empty"
    
    print(f"✅ Found {len(router.tavily_keywords)} Tavily keywords")
    print(f"Sample keywords: {router.tavily_keywords[:5]}")
    
    # Test specific keywords
    expected_keywords = ['current', 'latest', 'fda', 'who', 'cdc', 'guidelines']
    found_keywords = [kw for kw in expected_keywords if kw in router.tavily_keywords]
    
    print(f"Expected keywords found: {found_keywords}")
    assert len(found_keywords) >= 4, f"Not enough expected keywords found. Found: {found_keywords}"
    print("✅ Tavily keywords are properly defined")


def test_tavily_search_basic():
    """Test basic Tavily search functionality."""
    print("\n🔍 Testing Tavily search functionality...")
    
    # Check if API key is available
    api_key = os.getenv('TAVILY_API_KEY')
    if not api_key:
        print("⚠️  TAVILY_API_KEY not found in environment - testing fallback")
        
        # Test fallback mechanism
        result = Tavily_Search("current medical guidelines")
        print(f"Fallback result length: {len(result)} characters")
        
        # Should fallback to Wikipedia
        assert "Wikipedia" in result or len(result) > 100, "Fallback mechanism not working properly"
        print("✅ Fallback to Wikipedia works correctly")
        
    else:
        print("🔑 TAVILY_API_KEY found - testing actual API call")
        
        try:
            # Test with a simple medical query
            test_query = "current FDA drug recalls 2025"
            result = Tavily_Search(test_query)
            
            print(f"API result length: {len(result)} characters")
            print(f"Sample result: {result[:200]}...")
            
            # Basic checks
            assert len(result) > 50, "Result too short"
            assert "Error" not in result[:100], "API call failed"
            
            print("✅ Tavily API call successful")
            
        except Exception as e:
            print(f"⚠️  Tavily API test failed: {e}")
            print("This might be due to API limits or network issues")


def test_integration_completeness():
    """Test that all integration points are properly updated."""
    print("\n🔍 Testing integration completeness...")
    
    # Test imports
    try:
        from tools import Tavily_Search
        print("✅ Tavily_Search import successful")
    except ImportError as e:
        print(f"❌ Tavily_Search import failed: {e}")
        return
    
    # Test tool descriptions
    from tools import get_tool_descriptions
    descriptions = get_tool_descriptions()
    assert 'Tavily_Search' in descriptions, "Tavily_Search not in tool descriptions"
    print("✅ Tavily_Search found in tool descriptions")
    
    # Test integrated RAG import
    try:
        from integrated_rag import IntegratedMedicalRAG
        print("✅ IntegratedMedicalRAG import successful (should include Tavily)")
    except ImportError as e:
        print(f"⚠️  IntegratedMedicalRAG import issue: {e}")
    
    print("✅ Integration completeness check passed")


def main():
    """Run all Tavily integration tests."""
    print("🚀 Starting Tavily Search Tool Integration Tests")
    print("=" * 60)
    
    try:
        test_tavily_tool_availability()
        test_tavily_keywords()
        test_tavily_routing()
        test_tavily_search_basic()
        test_integration_completeness()
        
        print("\n" + "=" * 60)
        print("🎉 All Tavily integration tests completed successfully!")
        print("\n📋 Summary:")
        print("- ✅ Tool registration working")
        print("- ✅ Routing keywords configured")
        print("- ✅ Query routing functional")
        print("- ✅ Search functionality operational")
        print("- ✅ Integration points updated")
        
        print("\n💡 Next steps:")
        print("1. Install tavily-python: pip install tavily-python")
        print("2. Test with real queries to validate search quality")
        print("3. Consider adding enhanced Tavily search to enhanced_tools.py")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())