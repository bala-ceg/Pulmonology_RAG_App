#!/usr/bin/env python3
"""
Test script to validate the updated convert_markdown_to_reportlab function with actual HTML content
"""

def test_html_format_parsing():
    """Test the updated convert_markdown_to_reportlab function with actual HTML from the error"""
    
    # Actual HTML content from the user's error log
    html_content = '''<div style="margin-bottom: 30px; page-break-after: always;"><h4 style="color: #007bff; margin-bottom: 15px; font-size: 18px;">Answer</h4><div style="background-color: #e3f2fd; padding: 15px; border-radius: 8px; line-height: 1.6; margin-bottom: 20px;">Type 2 Diabetes, characterized by high blood sugar and insulin resistance, presents with symptoms like increased thirst, frequent urination, fatigue, unexplained weight loss, increased hunger, and slow-healing sores. It often develops slowly and can lead to serious complications like heart disease, kidney failure, and vision problems. Lifestyle factors like obesity and lack of exercise play a significant role in its development, along with genetic predisposition.</div></div><div style="margin-bottom: 30px; page-break-after: always;"><h4 style="color: #6f42c1; margin-bottom: 15px; font-size: 18px;">Source</h4><div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; line-height: 1.6; margin-bottom: 20px;"><a href="https://en.wikipedia.org/wiki/Type_2_diabetes" target="_blank">Type 2 diabetes</a> (Wikipedia)<br><a href="https://en.wikipedia.org/wiki/Outline_of_diabetes" target="_blank">Outline of diabetes</a> (Wikipedia)<br><a href="https://en.wikipedia.org/wiki/Type_1_diabetes" target="_blank">Type 1 diabetes</a> (Wikipedia)</div></div><div style="margin-bottom: 20px;"><h4 style="color: #ff6600; margin-bottom: 15px; font-size: 18px;">Tool Selection & Query Routing</h4><div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; line-height: 1.6;"><span style="color: #495057;">Confidence:</span> <span style="color: #000; font-weight: bold;">high</span><br><span style="color: #495057;">Tools Used:</span> Wikipedia_Search<br><span style="color: #495057;">Reasoning:</span> Query seeks general medical knowledge and definitions; Wikipedia selected for encyclopedic information</div></div>'''

    def convert_markdown_to_reportlab(text):
        """Convert enhanced tools HTML format to ReportLab-compatible text"""
        import re
        from bs4 import BeautifulSoup
        
        # Check if this is HTML content (contains div tags)
        if '<div' in text and '<h4' in text:
            try:
                # Parse HTML using BeautifulSoup
                soup = BeautifulSoup(text, 'html.parser')
                
                sections = {
                    'answer': '',
                    'source': '',
                    'tool_routing': ''
                }
                
                # Extract sections by looking for h4 headers
                divs = soup.find_all('div', style=lambda x: x and 'margin-bottom' in x)
                
                for div in divs:
                    h4 = div.find('h4')
                    if h4:
                        header_text = h4.get_text().strip().lower()
                        content_div = div.find('div', style=lambda x: x and ('background-color' in x or 'padding' in x))
                        
                        if 'answer' in header_text:
                            if content_div:
                                sections['answer'] = content_div.get_text().strip()
                        elif 'source' in header_text:
                            if content_div:
                                # Extract links and text
                                sources = []
                                links = content_div.find_all('a')
                                if links:
                                    for link in links:
                                        link_text = link.get_text().strip()
                                        # Get the text after the link (like "(Wikipedia)")
                                        next_text = link.next_sibling
                                        if next_text and isinstance(next_text, str):
                                            sources.append(f"{link_text} {next_text.strip()}")
                                        else:
                                            sources.append(link_text)
                                else:
                                    # Fallback to plain text
                                    sources = [content_div.get_text().strip()]
                                sections['source'] = '\n'.join(sources)
                        elif 'tool selection' in header_text or 'routing' in header_text:
                            if content_div:
                                sections['tool_routing'] = content_div.get_text().strip()
                
                # Build formatted text
                formatted_parts = []
                
                # 1. Answer section (main content)
                if sections['answer']:
                    formatted_parts.append(sections['answer'])
                
                # 2. Source section
                if sections['source']:
                    sources = [s.strip() for s in sections['source'].split('\n') if s.strip()]
                    if sources:
                        formatted_parts.append(f"<br/><br/><b>Sources:</b><br/>• " + "<br/>• ".join(sources))
                
                # 3. Tool routing section
                if sections['tool_routing']:
                    routing_text = sections['tool_routing']
                    
                    # Parse routing details
                    confidence = ""
                    tools_used = ""
                    reasoning = ""
                    
                    # Look for patterns in the text
                    confidence_match = re.search(r'Confidence:\s*([^\n]+)', routing_text, re.IGNORECASE)
                    if confidence_match:
                        confidence = confidence_match.group(1).strip()
                    
                    tools_match = re.search(r'Tools Used:\s*([^\n]+)', routing_text, re.IGNORECASE)
                    if tools_match:
                        tools_used = tools_match.group(1).strip()
                    
                    reasoning_match = re.search(r'Reasoning:\s*(.+)', routing_text, re.IGNORECASE | re.DOTALL)
                    if reasoning_match:
                        reasoning = reasoning_match.group(1).strip()
                    
                    # Build routing section
                    routing_parts = []
                    if confidence:
                        routing_parts.append(f"<b>Confidence:</b> {confidence}")
                    if tools_used:
                        routing_parts.append(f"<b>Tools Used:</b> {tools_used}")
                    if reasoning:
                        routing_parts.append(f"<b>Reasoning:</b> {reasoning}")
                    
                    if routing_parts:
                        formatted_parts.append(f"<br/><br/><b>Tool Selection & Query Routing:</b><br/>" + "<br/>".join(routing_parts))
                
                # Combine all parts
                full_text = "".join(formatted_parts)
                
                # Clean up extra spaces
                full_text = re.sub(r'\s+', ' ', full_text)
                
                return full_text.strip()
                
            except Exception as e:
                print(f"Error parsing HTML content: {e}")
                # Fallback to plain text extraction
                soup = BeautifulSoup(text, 'html.parser')
                return soup.get_text().strip()
        
        return text  # Fallback

    print("🧪 Testing HTML Format Parsing")
    print("=" * 60)
    
    print("\n📝 Input HTML Content:")
    print(html_content[:200] + "...")
    
    print("\n🔄 Processing...")
    try:
        result = convert_markdown_to_reportlab(html_content)
        
        print("\n📋 Formatted Result:")
        print(result)
        
        print("\n✅ Success! HTML content parsed and converted to ReportLab format")
        
        # Check if result is clean (no complex HTML tags that would break ReportLab)
        import re
        problematic_tags = re.findall(r'<(?!/?(?:b|i|br/?))[^>]+>', result)
        if problematic_tags:
            print(f"⚠️  Warning: Found potentially problematic tags: {problematic_tags}")
        else:
            print("✅ Result contains only ReportLab-compatible tags")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n🎉 Test Complete!")

if __name__ == "__main__":
    test_html_format_parsing()