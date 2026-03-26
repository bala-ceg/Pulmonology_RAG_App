#!/usr/bin/env python3
"""
Test script to validate the routing fix for query-content-based routing.

This script tests that routing decisions are based on query content, not session state.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_architecture import MedicalQueryRouter, TwoStoreRAGManager
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import tempfile

def create_test_rag_manager():
    """Create a mock RAG manager for testing."""
    class MockRAGManager:
        def __init__(self):
            self.sessions_with_content = set()
            
        def has_session_content(self, session_id: str) -> bool:
            return session_id in self.sessions_with_content
            
        def has_external_content(self) -> bool:
            return True
            
        def add_session_content(self, session_id: str):
            """Simulate adding content to a session."""
            self.sessions_with_content.add(session_id)
            
    return MockRAGManager()

def test_routing_scenarios():
    """Test various routing scenarios to ensure proper query-content-based routing."""
    
    print("🧪 Testing Routing Fix - Query-Content-Based Routing")
    print("=" * 60)
    
    # Create mock RAG manager and router
    mock_rag = create_test_rag_manager()
    router = MedicalQueryRouter(rag_manager=mock_rag)
    
    # Test scenarios
    scenarios = [
        {
            'name': 'PDF-relevant query without session content',
            'query': 'What was the relationship between increasing RDW levels and 28-day mortality?',
            'session_id': 'test_session_1',
            'expected_primary': 'Internal_VectorDB',
            'setup': lambda: None  # No setup needed
        },
        {
            'name': 'PDF-relevant query with session content', 
            'query': 'What was the relationship between increasing RDW levels and 28-day mortality?',
            'session_id': 'test_session_2',
            'expected_primary': 'Internal_VectorDB',
            'setup': lambda: mock_rag.add_session_content('test_session_2')
        },
        {
            'name': 'Wiki-relevant query without session content',
            'query': 'What are the symptoms of Covid19?',
            'session_id': 'test_session_3', 
            'expected_primary': 'Wikipedia_Search',
            'setup': lambda: None
        },
        {
            'name': 'Wiki-relevant query with session content (should route to Wiki)',
            'query': 'What are the symptoms of Covid19?',
            'session_id': 'test_session_4',
            'expected_primary': 'Wikipedia_Search',  # Should NOT stick to PDF
            'setup': lambda: mock_rag.add_session_content('test_session_4')
        },
        {
            'name': 'ArXiv-relevant query without session content',
            'query': 'Get me the latest research on Pulmonology',
            'session_id': 'test_session_5',
            'expected_primary': 'ArXiv_Search',
            'setup': lambda: None
        },
        {
            'name': 'ArXiv-relevant query with session content (should route to ArXiv)', 
            'query': 'Get me the latest research on Pulmonology',
            'session_id': 'test_session_6',
            'expected_primary': 'ArXiv_Search',  # Should NOT stick to PDF
            'setup': lambda: mock_rag.add_session_content('test_session_6')
        },
        {
            'name': 'Mixed query - should prioritize content relevance',
            'query': 'How much insulin does India produce?',
            'session_id': 'test_session_7', 
            'expected_primary': 'Wikipedia_Search',  # General knowledge
            'setup': lambda: mock_rag.add_session_content('test_session_7')
        }
    ]
    
    results = {'passed': 0, 'failed': 0, 'details': []}
    
    for scenario in scenarios:
        print(f"\n🔬 Testing: {scenario['name']}")
        print(f"   Query: '{scenario['query']}'")
        print(f"   Session: {scenario['session_id']}")
        
        # Setup scenario
        scenario['setup']()
        
        try:
            # Route the query
            result = router.route_tools(scenario['query'], scenario['session_id'])
            
            actual_primary = result['primary_tool']
            expected_primary = scenario['expected_primary']
            
            # Check result
            if actual_primary == expected_primary:
                print(f"   ✅ PASS: Routed to {actual_primary} as expected")
                results['passed'] += 1
                results['details'].append(f"✅ {scenario['name']}: {actual_primary}")
            else:
                print(f"   ❌ FAIL: Expected {expected_primary}, got {actual_primary}")
                results['failed'] += 1
                results['details'].append(f"❌ {scenario['name']}: Expected {expected_primary}, got {actual_primary}")
                
            print(f"   🔍 Confidence: {result['confidence']}")
            print(f"   💭 Reasoning: {result['reasoning']}")
            
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            results['failed'] += 1
            results['details'].append(f"❌ {scenario['name']}: ERROR - {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Passed: {results['passed']}")
    print(f"❌ Failed: {results['failed']}")
    print(f"📈 Success Rate: {results['passed']/(results['passed']+results['failed'])*100:.1f}%")
    
    print("\n📋 Detailed Results:")
    for detail in results['details']:
        print(f"   {detail}")
    
    if results['failed'] == 0:
        print("\n🎉 ALL TESTS PASSED! The routing fix is working correctly.")
        return True
    else:
        print(f"\n⚠️  {results['failed']} test(s) failed. Review the routing logic.")
        return False

if __name__ == "__main__":
    success = test_routing_scenarios()
    sys.exit(0 if success else 1)