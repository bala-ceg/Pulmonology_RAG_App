#!/usr/bin/env python3
"""
Test script to verify the routing logic for Doctor's Files discipline
"""
import sys
import os

# Add the main directory to path to import from main.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up the environment
os.environ['OPENAI_API_KEY'] = 'test-key'  # We won't actually call OpenAI

# Import the routing components
from main import MedicalQueryRouter, load_disciplines_config

def test_routing():
    """Test the routing logic with different queries"""
    
    # Load disciplines configuration
    disciplines_config = load_disciplines_config()
    print("📋 Loaded disciplines:")
    for discipline in disciplines_config["disciplines"]:
        print(f"  - {discipline['id']}: {discipline['name']}")
    
    # Create a mock LLM for testing (won't be called if keyword matching works)
    class MockLLM:
        def invoke(self, prompt):
            class MockResponse:
                content = "Family Medicine, Doctor's Files"
            return MockResponse()
    
    # Initialize router
    router = MedicalQueryRouter(MockLLM(), disciplines_config)
    
    # Set last_created_folder to simulate an active session with files
    import main
    main.last_created_folder = "guest_072820251001"  # Session with files
    
    # Test queries
    test_queries = [
        "What does my uploaded document say about chest pain?",
        "Tell me about the PDF I uploaded",
        "My files contain information about heart disease",
        "What's in my doctor's files?",
        "Check my personal documents for diabetes info",
        "Heart attack symptoms",  # Should route to cardiology
        "General health checkup",  # Should route to family medicine
    ]
    
    print("\n🧪 Testing routing logic:")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        try:
            result = router.analyze_query(query)
            disciplines = result["disciplines"]
            confidence_scores = result["confidence_scores"]
            
            print(f"🎯 Routed to: {disciplines}")
            print(f"📊 Confidence: {confidence_scores}")
            
            # Check if doctors_files is included
            if "doctors_files" in disciplines:
                print("✅ Doctor's Files discipline INCLUDED")
            else:
                print("❌ Doctor's Files discipline NOT included")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 40)

if __name__ == "__main__":
    test_routing()
