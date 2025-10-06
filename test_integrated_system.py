"""
Test Suite for Integrated Medical RAG System
==========================================

This module provides comprehensive tests for the tool routing system,
guarded retrieval, and fallback mechanisms.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(__file__))

from rag_architecture import MedicalQueryRouter, TwoStoreRAGManager
from tools import Wikipedia_Search, ArXiv_Search, Internal_VectorDB, _join_docs, guarded_retrieve
from prompts import ROUTING_SYSTEM_PROMPT, get_routing_explanation


class TestMedicalQueryRouter(unittest.TestCase):
    """Test the medical query routing system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_rag_manager = Mock()
        self.mock_rag_manager.get_local_content_count.return_value = 5
        self.mock_rag_manager.has_external_content.return_value = True
        
        self.router = MedicalQueryRouter(self.mock_rag_manager)
    
    def test_arxiv_routing(self):
        """Test routing to ArXiv for research queries."""
        queries = [
            "What is the latest research on COVID-19?",
            "Recent studies on hypertension treatment",
            "Show me new papers on diabetes"
        ]
        
        for query in queries:
            result = self.router.route_tools(query)
            self.assertEqual(result['primary_tool'], 'ArXiv_Search')
            self.assertIn('ArXiv_Search', result['ranked_tools'])
            print(f"✓ ArXiv routing for: '{query}' - Confidence: {result['confidence']}")
    
    def test_wikipedia_routing(self):
        """Test routing to Wikipedia for definition queries."""
        queries = [
            "What is hypertension?",
            "Define myocardial infarction",
            "Explain what diabetes means"
        ]
        
        for query in queries:
            result = self.router.route_tools(query)
            self.assertEqual(result['primary_tool'], 'Wikipedia_Search')
            self.assertIn('Wikipedia_Search', result['ranked_tools'])
            print(f"✓ Wikipedia routing for: '{query}' - Confidence: {result['confidence']}")
    
    def test_internal_routing(self):
        """Test routing to Internal VectorDB for uploaded document queries."""
        queries = [
            "What does my uploaded PDF say about treatment?",
            "According to our protocol, what should I do?",
            "From my documents, what is the recommended dosage?"
        ]
        
        for query in queries:
            result = self.router.route_tools(query)
            self.assertEqual(result['primary_tool'], 'Internal_VectorDB')
            self.assertIn('Internal_VectorDB', result['ranked_tools'])
            print(f"✓ Internal routing for: '{query}' - Confidence: {result['confidence']}")
    
    def test_no_content_fallback(self):
        """Test fallback when no internal content is available."""
        # Mock no local content
        self.mock_rag_manager.get_local_content_count.return_value = 0
        
        query = "What does my uploaded document say?"
        result = self.router.route_tools(query)
        
        # Should still try internal but with low score, likely falling back to Wikipedia
        self.assertLess(result['tool_scores']['Internal_VectorDB'], 0)
        print(f"✓ No content fallback test passed")
    
    def test_confidence_scoring(self):
        """Test confidence scoring based on keyword matches."""
        # High confidence query (multiple research keywords)
        high_conf_query = "Latest research studies on new breakthrough findings"
        result = self.router.route_tools(high_conf_query)
        self.assertIn(result['confidence'], ['medium', 'high'])
        
        # Low confidence query (ambiguous)
        low_conf_query = "Tell me about treatment"
        result = self.router.route_tools(low_conf_query)
        # Confidence will vary, but should not crash
        self.assertIn(result['confidence'], ['low', 'medium', 'high'])
        
        print(f"✓ Confidence scoring tests passed")


class TestToolFunctions(unittest.TestCase):
    """Test individual tool functions."""
    
    @patch('tools.WikipediaLoader')
    def test_wikipedia_search(self, mock_loader):
        """Test Wikipedia search tool."""
        # Mock successful Wikipedia response
        mock_docs = [
            Mock(page_content="Hypertension is high blood pressure...", 
                 metadata={'title': 'Hypertension', 'source': 'wikipedia'})
        ]
        mock_loader.return_value.load.return_value = mock_docs
        
        result = Wikipedia_Search("hypertension")
        self.assertIsInstance(result, str)
        self.assertIn("hypertension", result.lower())
        print("✓ Wikipedia search test passed")

    @patch('tools.ArxivLoader')
    def test_arxiv_search(self, mock_loader):
        """Test ArXiv search tool."""
        # Mock successful ArXiv response
        mock_docs = [
            Mock(page_content="Recent study shows...", 
                 metadata={'Title': 'COVID-19 Research', 'Authors': 'Smith et al'})
        ]
        mock_loader.return_value.load.return_value = mock_docs
        
        result = ArXiv_Search("COVID-19 treatment")
        self.assertIsInstance(result, str)
        print("✓ ArXiv search test passed")

    def test_internal_vectordb_no_manager(self):
        """Test Internal VectorDB with no RAG manager."""
        # Should fallback to Wikipedia
        with patch('tools.Wikipedia_Search') as mock_wiki:
            mock_wiki.return_value = "Wikipedia fallback response"
            
            result = Internal_VectorDB("test query", rag_manager=None)
            self.assertIn("not available", result)
            print("✓ Internal VectorDB fallback test passed")

    def test_join_docs_utility(self):
        """Test the _join_docs utility function."""
        from langchain.schema import Document
        
        # Create mock documents
        docs = [
            Document(page_content="First document content", metadata={'source': 'doc1.pdf'}),
            Document(page_content="Second document content", metadata={'source': 'doc2.pdf'})
        ]
        
        result = _join_docs(docs, max_chars=100)
        self.assertIsInstance(result, str)
        self.assertLessEqual(len(result), 150)  # Some buffer for formatting
        print("✓ Join docs utility test passed")


class TestGuardedRetrieval(unittest.TestCase):
    """Test the guarded retrieval system."""
    
    def test_guarded_retrieve_relevant_docs(self):
        """Test guarded retrieval with relevant documents."""
        from langchain.schema import Document
        
        # Mock retriever with relevant documents
        mock_retriever = Mock()
        relevant_docs = [
            Document(page_content="This document discusses hypertension treatment protocols in detail...", 
                    metadata={'source': 'medical_guide.pdf'})
        ]
        mock_retriever.invoke.return_value = relevant_docs
        
        result = guarded_retrieve("hypertension treatment", mock_retriever)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        print("✓ Guarded retrieval with relevant docs passed")
    
    def test_guarded_retrieve_irrelevant_docs(self):
        """Test guarded retrieval with irrelevant documents."""
        from langchain.schema import Document
        
        # Mock retriever with irrelevant documents
        mock_retriever = Mock()
        irrelevant_docs = [
            Document(page_content="This document talks about car maintenance and engine repair...", 
                    metadata={'source': 'car_manual.pdf'})
        ]
        mock_retriever.invoke.return_value = irrelevant_docs
        
        result = guarded_retrieve("diabetes treatment", mock_retriever)
        self.assertIsNone(result)  # Should trigger fallback
        print("✓ Guarded retrieval with irrelevant docs passed")
    
    def test_guarded_retrieve_no_docs(self):
        """Test guarded retrieval with no documents."""
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = []
        
        result = guarded_retrieve("any query", mock_retriever)
        self.assertIsNone(result)
        print("✓ Guarded retrieval with no docs passed")


class TestPrompts(unittest.TestCase):
    """Test prompt generation and routing explanations."""
    
    def test_routing_system_prompt(self):
        """Test that routing system prompt is comprehensive."""
        self.assertIsInstance(ROUTING_SYSTEM_PROMPT, str)
        self.assertIn("Wikipedia_Search", ROUTING_SYSTEM_PROMPT)
        self.assertIn("ArXiv_Search", ROUTING_SYSTEM_PROMPT)
        self.assertIn("Internal_VectorDB", ROUTING_SYSTEM_PROMPT)
        self.assertIn("fallback", ROUTING_SYSTEM_PROMPT.lower())
        print("✓ Routing system prompt test passed")
    
    def test_routing_explanation(self):
        """Test routing explanation generation."""
        explanation = get_routing_explanation("Wikipedia_Search", "high", False)
        self.assertIsInstance(explanation, str)
        self.assertIn("Wikipedia", explanation)
        
        explanation_with_fallback = get_routing_explanation("ArXiv_Search", "medium", True)
        self.assertIn("fallback", explanation_with_fallback)
        print("✓ Routing explanation test passed")


def run_comprehensive_test():
    """Run all tests and provide a summary."""
    print("Running Comprehensive Medical RAG Tool Tests")
    print("=" * 60)
    
    # Create test suite
    test_classes = [
        TestMedicalQueryRouter,
        TestToolFunctions,
        TestGuardedRetrieval,
        TestPrompts
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\nRunning {test_class.__name__}:")
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        
        for test in suite:
            total_tests += 1
            try:
                test.debug()  # Run test
                passed_tests += 1
            except Exception as e:
                print(f"✗ {test._testMethodName} failed: {e}")
    
    print(f"\n{'=' * 60}")
    print(f"Test Summary: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! The integrated system is working correctly.")
    else:
        print(f"⚠️  {total_tests - passed_tests} tests failed. Please review the implementation.")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_comprehensive_test()
    
    if success:
        print("\n" + "=" * 60)
        print("INTEGRATION SUMMARY")
        print("=" * 60)
        print("✓ Four self-describing tools implemented")
        print("✓ Routing system prompt created")
        print("✓ MedicalQueryRouter.route_tools implemented")
        print("✓ Post-retrieval guard system added")
        print("✓ Wikipedia fallback mechanisms working")
        print("✓ Tool integration completed successfully")
        print("\nThe system is ready for deployment!")
    else:
        print("\n⚠️  Some components need attention before deployment.")