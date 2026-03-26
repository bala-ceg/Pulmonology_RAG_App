#!/usr/bin/env python3
"""
Test simpler HTML parsing approach for tool routing section
"""

def test_simple_html_parsing():
    """Test a simpler HTML parsing approach"""
    
    # Actual tool routing HTML from the error
    routing_html = '''<div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; line-height: 1.6;"><span style="color: #495057;">Confidence:</span> <span style="color: #000; font-weight: bold;">high</span><br><span style="color: #495057;">Tools Used:</span> Wikipedia_Search<br><span style="color: #495057;">Reasoning:</span> Query seeks general medical knowledge and definitions; Wikipedia selected for encyclopedic information</div>'''
    
    def parse_routing_section(html_content):
        """Simple parsing approach for routing section"""
        from bs4 import BeautifulSoup
        import re
        
        soup = BeautifulSoup(html_content, 'html.parser')
        text = soup.get_text()
        
        # Use simple regex patterns on the plain text
        confidence = ""
        tools_used = ""
        reasoning = ""
        
        # Split by line breaks and parse each line
        lines = text.replace('\n', ' ').split('Tools Used:')
        
        if len(lines) >= 2:
            # First part contains confidence
            confidence_part = lines[0]
            confidence_match = re.search(r'Confidence:\s*(.+?)$', confidence_part.strip(), re.IGNORECASE)
            if confidence_match:
                confidence = confidence_match.group(1).strip()
            
            # Second part contains tools used and reasoning
            rest = lines[1]
            reasoning_split = rest.split('Reasoning:')
            
            if len(reasoning_split) >= 2:
                tools_used = reasoning_split[0].strip()
                reasoning = reasoning_split[1].strip()
            else:
                tools_used = rest.strip()
        
        return confidence, tools_used, reasoning
    
    print("🧪 Testing Simple HTML Parsing")
    print("=" * 50)
    
    print("\n📝 Input HTML:")
    print(routing_html)
    
    print("\n🔄 Processing...")
    confidence, tools_used, reasoning = parse_routing_section(routing_html)
    
    print(f"\n✅ Parsing Results:")
    print(f"   📊 Confidence: '{confidence}'")
    print(f"   🔧 Tools Used: '{tools_used}'")
    print(f"   💭 Reasoning: '{reasoning}'")
    
    # Build the final format
    routing_parts = []
    if confidence:
        routing_parts.append(f"<b>Confidence:</b> {confidence}")
    if tools_used:
        routing_parts.append(f"<b>Tools Used:</b> {tools_used}")
    if reasoning:
        routing_parts.append(f"<b>Reasoning:</b> {reasoning}")
    
    final_result = f"<br/><br/><b>Tool Selection & Query Routing:</b><br/>" + "<br/>".join(routing_parts)
    
    print(f"\n📋 Final Formatted Result:")
    print(final_result)
    
    print("\n🎉 Test Complete!")

if __name__ == "__main__":
    test_simple_html_parsing()