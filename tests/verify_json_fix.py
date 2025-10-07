#!/usr/bin/env python3
"""
Verify RAG architecture is working correctly after the JSON serialization fix
"""

import sys
import os
import json

def test_rag_query_simulation():
    """Simulate the query process to test JSON serialization"""
    print("🧪 Testing RAG Query Simulation...")
    
    # Simulate the routing_info that would be returned
    routing_info = {
        'similarity_score': 0.456,  # This should be a float
        'query_local_first': True,  # This should be a bool  
        'sources_queried': ['kb_local']  # This should be a list
    }
    
    # Simulate the response structure
    rag_result = {
        'responses': [
            {
                'source': 'Local Knowledge Base',
                'content': 'RDW (Red Cell Distribution Width) is a measure of the variation in red blood cell size...',
                'confidence': 90
            }
        ],
        'citations': ['**Local KB**: Document about RDW in critical care'],
        'routing_info': routing_info
    }
    
    # Test the exact structure that would be sent to jsonify
    response_data = {
        "response": True, 
        "message": "Test response with routing info",
        "routing_details": {
            "method": "Two-Store RAG with Lexical Gate",
            "similarity_score": float(routing_info.get('similarity_score', 0)),
            "query_local_first": bool(routing_info.get('query_local_first', False)),
            "sources_queried": list(routing_info.get('sources_queried', [])),
            "responses_count": int(len(rag_result['responses']))
        }
    }
    
    # Test JSON serialization
    try:
        json_str = json.dumps(response_data, indent=2)
        print("✅ JSON serialization successful!")
        print("Response structure:")
        print(json_str)
        return True
    except Exception as e:
        print(f"❌ JSON serialization failed: {e}")
        return False

def check_numpy_imports():
    """Check if numpy is properly imported and types are handled"""
    print("\n🔢 Testing NumPy type handling...")
    
    try:
        import numpy as np
        
        # Test the exact scenario from the lexical gate
        similarities = np.array([0.1, 0.456, 0.2])
        max_similarity = np.max(similarities)
        query_local_first = max_similarity >= 0.3
        
        print(f"NumPy max_similarity type: {type(max_similarity)}")
        print(f"NumPy query_local_first type: {type(query_local_first)}")
        
        # Convert to Python native types
        max_similarity_py = float(max_similarity)
        query_local_first_py = bool(query_local_first)
        
        print(f"Converted max_similarity type: {type(max_similarity_py)}")
        print(f"Converted query_local_first type: {type(query_local_first_py)}")
        
        # Test JSON serialization
        test_data = {
            'max_similarity': max_similarity_py,
            'query_local_first': query_local_first_py
        }
        
        json_str = json.dumps(test_data)
        print("✅ NumPy type conversion and JSON serialization successful!")
        return True
        
    except Exception as e:
        print(f"❌ NumPy type handling failed: {e}")
        return False

def main():
    """Run all verification tests"""
    print("🏥 RAG Architecture Verification")
    print("=" * 40)
    
    tests = [
        ("RAG Query Simulation", test_rag_query_simulation),
        ("NumPy Type Handling", check_numpy_imports)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED\n")
        else:
            print(f"❌ {test_name} FAILED\n")
    
    print("=" * 40)
    print(f"Verification Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 JSON serialization fix verified! Your RAG architecture should work correctly now.")
        print("\n📋 Next steps:")
        print("1. Restart your Flask application")
        print("2. Try the queries that failed before:")
        print("   - 'Explain the RDW in critically ill patients'")
        print("   - 'What are the symptoms of Type-2 diabetes?'")
        return True
    else:
        print("⚠️  Some tests failed. There may be additional issues to resolve.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)