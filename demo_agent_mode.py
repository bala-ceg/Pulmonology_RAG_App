#!/usr/bin/env python3
"""
Simple Demo of Direct API Agent Mode
===================================

This demonstrates the agent mode with simplified mock responses.
Run this to see how the agent routing and tool selection works.
"""

def demo_agent_mode():
    """
    Demo of direct API agent mode with tool selection.
    """
    print("🤖 Direct API Agent Mode Demo")
    print("=" * 50)
    
    # Simulate the agent tools
    class MockTool:
        def __init__(self, name, description, response_template):
            self.name = name
            self.description = description
            self.response_template = response_template
        
        def run(self, query):
            return self.response_template.format(query=query)
    
    # Create mock tools (matching your direct_api_tools.py)
    wikipedia_tool = MockTool(
        name="wikipedia_search",
        description="Search Wikipedia for medical definitions and basic facts",
        response_template="""**According to Wikipedia:**

{query} is a medical condition/term with established clinical definitions. This information comes from peer-reviewed medical sources and textbooks.

**Key Points:**
- Well-documented medical knowledge
- Established diagnostic criteria  
- Standard treatment approaches

**Source:** Wikipedia Medical Articles
**Note:** This is general medical information. Consult healthcare providers for specific medical advice."""
    )
    
    arxiv_tool = MockTool(
        name="arxiv_search", 
        description="Search ArXiv for recent medical research papers",
        response_template="""**Based on recent ArXiv research:**

Recent studies on {query} show emerging developments in the field:

**Paper 1:** "Novel Approaches to {query}" (2024)
- Authors: Smith et al.
- Findings: Preliminary results suggest new therapeutic targets

**Paper 2:** "Clinical Trials in {query}" (2024)  
- Authors: Johnson et al.
- Findings: Phase II trials showing promising outcomes

**Source:** ArXiv.org Medical Research Papers
**Note:** These are preliminary research findings. Clinical validation may be pending."""
    )
    
    internal_tool = MockTool(
        name="internal_documents_search",
        description="Search uploaded documents and internal knowledge base", 
        response_template="""**From your uploaded documents:**

Searching internal knowledge base for information about {query}...

**Found in:** Protocol_Guidelines.pdf
**Content:** Your organization's specific protocols and guidelines regarding {query} include standardized procedures and institutional recommendations.

**Note:** This information comes from your uploaded documents and organizational knowledge base.

**Source:** Internal Document Repository"""
    )
    
    tools = [wikipedia_tool, arxiv_tool, internal_tool]
    
    # Agent routing logic (simplified version of your MedicalQueryRouter)
    def route_query(query):
        """Route query to appropriate tool based on keywords."""
        query_lower = query.lower()
        
        # Keyword scoring
        arxiv_keywords = ['latest', 'recent', 'research', 'study', 'paper', 'findings', 'new']
        internal_keywords = ['uploaded', 'my file', 'my document', 'our protocol', 'internal', 'organization']
        wikipedia_keywords = ['what is', 'define', 'explain', 'tell me about', 'definition']
        
        arxiv_score = sum(1 for kw in arxiv_keywords if kw in query_lower)
        internal_score = sum(1 for kw in internal_keywords if kw in query_lower)  
        wikipedia_score = sum(1 for kw in wikipedia_keywords if kw in query_lower)
        
        # Add default scoring
        if arxiv_score == internal_score == wikipedia_score == 0:
            wikipedia_score = 1  # Default to Wikipedia
        
        # Select tool with highest score
        scores = [
            (wikipedia_tool, wikipedia_score, "Wikipedia"),
            (arxiv_tool, arxiv_score, "ArXiv"), 
            (internal_tool, internal_score, "Internal")
        ]
        
        selected_tool, score, source_name = max(scores, key=lambda x: x[1])
        confidence = "High" if score >= 2 else "Medium" if score >= 1 else "Low"
        
        return selected_tool, source_name, confidence
    
    # Test queries
    test_queries = [
        "What is pneumonia?",  # Should route to Wikipedia
        "Latest research on COVID-19 vaccines",  # Should route to ArXiv  
        "Recent studies on pulmonary fibrosis",  # Should route to ArXiv
        "What does my uploaded protocol say about treatment?",  # Should route to Internal
        "Define hypertension",  # Should route to Wikipedia
        "Our organization's guidelines for diabetes",  # Should route to Internal
    ]
    
    print("🔍 Testing Agent Routing and Tool Selection:")
    print("-" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Query {i}: {query}")
        
        # Route the query
        selected_tool, source_name, confidence = route_query(query)
        
        print(f"🎯 Selected Tool: {selected_tool.name}")
        print(f"📊 Source: {source_name}")
        print(f"🎚️  Confidence: {confidence}")
        
        # Simulate tool execution
        response = selected_tool.run(query)
        print(f"📄 Response Preview: {response[:150]}...")
        
        print("-" * 60)
    
    print("\n✅ Agent Mode Demo Complete!")
    print("\n🚀 To run with real APIs:")
    print("1. Set OpenAI API key: export OPENAI_API_KEY='your-key'")
    print("2. Run: python agent_direct_api.py")
    print("3. Or interactive: python agent_direct_api.py interactive")


if __name__ == "__main__":
    demo_agent_mode()