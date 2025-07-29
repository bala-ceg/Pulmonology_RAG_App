#!/usr/bin/env python3
import os
import sys
sys.path.append('.')

from main import MedicalQueryRouter, load_disciplines_config

# Mock LLM
class MockLLM:
    def invoke(self, prompt):
        class MockResponse:
            content = "Doctor's Files, Family Medicine"
        return MockResponse()

# Test
config = load_disciplines_config()
router = MedicalQueryRouter(MockLLM(), config)

# Simulate session with files
import main
main.last_created_folder = 'guest_072820251001'

# Test query
query = "What does my uploaded document say?"
result = router.analyze_query(query)
print(f"Query: {query}")
print(f"Routed to: {result['disciplines']}")

if 'doctors_files' in result['disciplines']:
    print("✅ SUCCESS: Doctor's Files is included in routing!")
else:
    print("❌ ISSUE: Doctor's Files not included")
    print(f"Available disciplines: {[d['id'] for d in config['disciplines']]}")
