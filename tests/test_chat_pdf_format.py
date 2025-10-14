#!/usr/bin/env python3
"""
Test script to validate the updated convert_markdown_to_reportlab function
"""

def test_new_format_parsing():
    """Test the updated convert_markdown_to_reportlab function with new enhanced tools format"""
    
    # Sample text in the new format from the user's example
    sample_text = """Answer
Type 2 diabetes, characterized by high blood sugar and insulin resistance, presents with symptoms like increased thirst, frequent urination, fatigue, unexplained weight loss, increased hunger, and slow-healing sores. Obesity, lack of exercise, and genetic factors contribute to its development. Long-term complications include heart disease, stroke, blindness, kidney failure, and amputations. Early diagnosis through blood tests is crucial for effective management and prevention of complications.
Source
Type 2 diabetes (Wikipedia)
Outline of diabetes (Wikipedia)
Type 1 diabetes (Wikipedia)
Tool Selection & Query Routing
Confidence: high
Tools Used: Wikipedia_Search
Reasoning: Query seeks general medical knowledge and definitions; Wikipedia selected for encyclopedic information"""

    # Define the convert function (copied from the updated main.py)
    def convert_markdown_to_reportlab(text):
        """Convert new enhanced tools format to ReportLab-compatible HTML-like formatting"""
        import re
        
        # Handle new enhanced tools format with structured sections
        sections = {
            'answer': '',
            'source': '',
            'tool_routing': ''
        }
        
        # Split text by the new section headers
        lines = text.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for section headers
            if line.lower().startswith('answer'):
                current_section = 'answer'
                continue
            elif line.lower().startswith('source'):
                current_section = 'source'
                continue
            elif line.lower().startswith('tool selection') or line.lower().startswith('confidence:') or line.lower().startswith('tools used:') or line.lower().startswith('reasoning:'):
                current_section = 'tool_routing'
                if line.lower().startswith('tool selection'):
                    continue  # Skip the header line
            
            # Add content to the appropriate section
            if current_section:
                if sections[current_section]:
                    sections[current_section] += ' ' + line
                else:
                    sections[current_section] = line
        
        # Build formatted text
        formatted_parts = []
        
        # 1. Answer section (main content)
        if sections['answer']:
            formatted_parts.append(sections['answer'])
        
        # 2. Source section
        if sections['source']:
            # Split sources by lines and clean them up
            source_text = sections['source'].strip()
            # Try to split by common patterns like "(Wikipedia)" or line breaks
            sources = []
            if '(Wikipedia)' in source_text:
                # Split by source patterns
                import re
                source_matches = re.findall(r'([^()]+\([^)]+\))', source_text)
                sources = [match.strip() for match in source_matches if match.strip()]
            else:
                # Split by lines
                sources = [s.strip() for s in source_text.split('\n') if s.strip()]
            
            if sources:
                formatted_parts.append(f"<br/><br/><b>Sources:</b><br/>• " + "<br/>• ".join(sources))
        
        # 3. Tool routing section
        if sections['tool_routing']:
            # Parse tool routing details more carefully
            routing_text = sections['tool_routing']
            
            # Split by the key fields and extract clean values
            confidence = ""
            tools_used = ""
            reasoning = ""
            
            # Extract confidence
            confidence_match = re.search(r'Confidence:\s*([^:]+?)(?=\s*Tools Used:|$)', routing_text, re.IGNORECASE)
            if confidence_match:
                confidence = confidence_match.group(1).strip()
            
            # Extract tools used
            tools_match = re.search(r'Tools Used:\s*([^:]+?)(?=\s*Reasoning:|$)', routing_text, re.IGNORECASE)
            if tools_match:
                tools_used = tools_match.group(1).strip()
            
            # Extract reasoning
            reasoning_match = re.search(r'Reasoning:\s*(.+?)$', routing_text, re.IGNORECASE | re.DOTALL)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
            
            # Build the routing section
            routing_parts = []
            if confidence:
                routing_parts.append(f"<b>Confidence:</b> {confidence}")
            if tools_used:
                routing_parts.append(f"<b>Tools Used:</b> {tools_used}")
            if reasoning:
                routing_parts.append(f"<b>Reasoning:</b> {reasoning}")
            
            if routing_parts:
                formatted_parts.append(f"<br/><br/><b>Tool Selection & Query Routing:</b><br/>" + "<br/>".join(routing_parts))
        
        # If no structured sections found, treat as plain text (fallback)
        if not any(sections.values()):
            formatted_parts = [text]
        
        # Combine all parts
        full_text = "".join(formatted_parts)
        
        # Clean up formatting
        # Convert **bold** to <b>bold</b>
        full_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', full_text)
        
        # Convert *italic* to <i>italic</i>
        full_text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', full_text)
        
        # Handle line breaks - convert \n to <br/>
        full_text = full_text.replace('\n', '<br/>')
        
        # Handle multiple line breaks for better paragraph spacing
        full_text = re.sub(r'(<br/>){3,}', '<br/><br/>', full_text)
        
        # Clean up extra spaces
        full_text = re.sub(r'\s+', ' ', full_text)
        
        return full_text.strip()

    print("🧪 Testing New Format Parsing")
    print("=" * 50)
    
    print("\n📝 Input Text:")
    print(sample_text)
    
    print("\n🔄 Processing...")
    result = convert_markdown_to_reportlab(sample_text)
    
    print("\n📋 Formatted Result:")
    print(result)
    
    print("\n✅ Parsing Components:")
    
    # Test individual section extraction
    lines = sample_text.split('\n')
    sections = {'answer': '', 'source': '', 'tool_routing': ''}
    current_section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.lower().startswith('answer'):
            current_section = 'answer'
            continue
        elif line.lower().startswith('source'):
            current_section = 'source'
            continue
        elif line.lower().startswith('tool selection') or line.lower().startswith('confidence:'):
            current_section = 'tool_routing'
            if line.lower().startswith('tool selection'):
                continue
        
        if current_section:
            if sections[current_section]:
                sections[current_section] += ' ' + line
            else:
                sections[current_section] = line
    
    print(f"   📋 Answer: {sections['answer'][:100]}...")
    print(f"   📚 Source: {sections['source']}")
    print(f"   🔧 Tool Routing: {sections['tool_routing']}")
    
    print("\n🎉 Test Complete!")

if __name__ == "__main__":
    test_new_format_parsing()