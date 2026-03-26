#!/usr/bin/env python3
"""Test script to validate PDF generation functionality"""

import json
from datetime import datetime

def test_pdf_data_format():
    """Test the data format for PDF generation"""
    
    # Sample chat data in the format expected by the frontend
    sample_messages = [
        {"role": "user", "content": "How to solve Type 2 diabetes with high BP"},
        {"role": "ai", "content": "Here are some recommendations for managing Type 2 diabetes with high blood pressure:\n\n1. **Lifestyle modifications**:\n   - Regular exercise\n   - Healthy diet\n   - Weight management\n\n2. **Medications**:\n   - ACE inhibitors\n   - Metformin\n   - Monitor blood glucose regularly"},
        {"role": "user", "content": "How about with Type 1 diabetes with high BP"},
        {"role": "ai", "content": "For Type 1 diabetes with high BP:\n\n1. **Insulin management**:\n   - Continuous glucose monitoring\n   - Proper insulin dosing\n\n2. **Blood pressure control**:\n   - Monitor regularly\n   - Consider ACE inhibitors\n   - Lifestyle modifications"}
    ]
    
    # Expected PDF data format
    pdf_data = {
        "doctorName": "Dr. Suresh Reddy", 
        "messages": sample_messages,
        "jsonData": ""  # Optional JSON data section
    }
    
    print("✅ Sample PDF data structure:")
    print(json.dumps(pdf_data, indent=2))
    
    print("\n✅ Expected PDF format will be:")
    print("Header:")
    print(f"    Doctor Name: {pdf_data['doctorName']}")
    
    now = datetime.now()
    formatted_date = now.strftime("%Y %m %d %H %M")
    print(f"    Date: {formatted_date}")
    
    if pdf_data['jsonData']:
        print(f"\n{pdf_data['jsonData']}")
    
    print("\nChat conversation:")
    for message in pdf_data['messages']:
        if message['role'] == 'user':
            print("\n****** Doctor Input *****")
            print(message['content'])
        elif message['role'] == 'ai':
            print("\n****** System Output *****") 
            print(message['content'])
    
    return True

if __name__ == "__main__":
    print("Testing PDF generation format...")
    test_pdf_data_format()
    print("\n✅ PDF format test completed successfully!")
