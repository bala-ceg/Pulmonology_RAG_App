#!/usr/bin/env python3
"""
Quick test for JSON serialization issues
"""

import json
import numpy as np
from flask import Flask, jsonify

app = Flask(__name__)

def test_json_serialization():
    """Test various data types for JSON serialization"""
    print("Testing JSON serialization...")
    
    # Test problematic types
    test_data = {
        "numpy_float": np.float64(0.456),
        "numpy_bool": np.bool_(True),
        "numpy_int": np.int64(5),
        "regular_float": 0.456,
        "regular_bool": True,
        "regular_int": 5,
        "list": ["item1", "item2"],
        "string": "test"
    }
    
    print("Original data types:")
    for key, value in test_data.items():
        print(f"  {key}: {type(value)} = {value}")
    
    # Test direct JSON serialization
    try:
        json_str = json.dumps(test_data)
        print("✅ Direct JSON serialization successful")
    except Exception as e:
        print(f"❌ Direct JSON serialization failed: {e}")
    
    # Test with type conversion
    converted_data = {
        "numpy_float": float(test_data["numpy_float"]),
        "numpy_bool": bool(test_data["numpy_bool"]),
        "numpy_int": int(test_data["numpy_int"]),
        "regular_float": test_data["regular_float"],
        "regular_bool": test_data["regular_bool"],
        "regular_int": test_data["regular_int"],
        "list": list(test_data["list"]),
        "string": str(test_data["string"])
    }
    
    print("\nConverted data types:")
    for key, value in converted_data.items():
        print(f"  {key}: {type(value)} = {value}")
    
    try:
        json_str = json.dumps(converted_data)
        print("✅ Converted JSON serialization successful")
        return True
    except Exception as e:
        print(f"❌ Converted JSON serialization failed: {e}")
        return False

@app.route('/test_json')
def test_flask_json():
    """Test Flask jsonify with various data types"""
    routing_details = {
        "method": "Two-Store RAG with Lexical Gate",
        "similarity_score": float(np.float64(0.456)),
        "query_local_first": bool(np.bool_(True)),
        "sources_queried": list(["kb_local"]),
        "responses_count": int(np.int64(1))
    }
    
    return jsonify({
        "response": True,
        "message": "Test message",
        "routing_details": routing_details
    })

if __name__ == "__main__":
    success = test_json_serialization()
    
    if success:
        print("\n🧪 Starting Flask test server...")
        print("Visit: http://localhost:5001/test_json")
        app.run(port=5001, debug=True)
    else:
        print("❌ JSON serialization test failed")