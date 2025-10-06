"""
Agent-Based Medical RAG System with Direct API Tools
==================================================

This module demonstrates how to use direct API tools (Wikipedia/ArXiv) 
with LangChain agents for fast, real-time medical question answering.
"""

import os
from typing import Dict, List, Optional, Any
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.schema import HumanMessage

# Import direct API tools
from direct_api_tools import create_agent_tools, get_tool_descriptions
from prompts import ROUTING_SYSTEM_PROMPT


class DirectAPIAgent:
    """
    Medical AI agent using direct Wikipedia and ArXiv API calls.
    
    This is a simpler, faster alternative to the vector-database approach
    that's perfect for real-time queries and general medical knowledge.
    """
    
    def __init__(self, openai_api_key: str, include_internal: bool = False):
        """
        Initialize the direct API agent.
        
        Args:
            openai_api_key: OpenAI API key
            include_internal: Whether to include internal document search
        """
        self.openai_api_key = openai_api_key
        self.include_internal = include_internal
        
        # Initialize OpenAI LLM
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model="gpt-3.5-turbo",  # Faster than GPT-4 for most queries
            temperature=0.1,
            max_tokens=1000
        )
        
        # Create tools
        self.tools = create_agent_tools(include_internal=include_internal)
        
        # Initialize agent
        self.agent = self._initialize_agent()
        
        print(f"DirectAPIAgent initialized with {len(self.tools)} tools")
        print(f"Available tools: {[tool.name for tool in self.tools]}")
    
    def _initialize_agent(self):
        """Initialize the LangChain agent with direct API tools."""
        try:
            # Create a simplified system prompt for direct API usage
            system_prompt = """You are a medical AI assistant with access to real-time Wikipedia and ArXiv searches.

TOOL SELECTION RULES:
- Use wikipedia_search for: definitions, basic medical facts, established knowledge
- Use arxiv_search for: recent research, latest studies, scientific findings  
- Use internal_documents_search for: questions about uploaded files (if available)

RESPONSE FORMAT:
- Always state which source you used
- Provide accurate, helpful medical information
- Include appropriate medical disclaimers
- If unsure, acknowledge limitations

Remember: Prioritize user safety and accurate information."""
            
            agent = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                max_iterations=3,
                early_stopping_method="generate",
                handle_parsing_errors=True
            )
            
            # Inject system prompt
            if hasattr(agent.agent, 'llm_chain') and hasattr(agent.agent.llm_chain, 'prompt'):
                original_template = agent.agent.llm_chain.prompt.template
                agent.agent.llm_chain.prompt.template = f"{system_prompt}\n\n{original_template}"
            
            return agent
            
        except Exception as e:
            print(f"Error initializing agent: {e}")
            return None
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Process a medical query using the direct API agent.
        
        Args:
            question: User's medical question
            
        Returns:
            Dictionary with response and metadata
        """
        try:
            print(f"\n🔍 Processing query: {question}")
            
            if not self.agent:
                return {
                    'answer': "Agent is not available. Please check your OpenAI API key and try again.",
                    'source': 'Error',
                    'success': False
                }
            
            # Run the agent
            response = self.agent.run(question)
            
            # Determine which tool was likely used (simple heuristic)
            source = self._determine_source(response)
            
            return {
                'answer': response,
                'source': source,
                'success': True,
                'tools_available': [tool.name for tool in self.tools]
            }
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            return {
                'answer': f"I encountered an error processing your question: {str(e)}. Please try rephrasing your question.",
                'source': 'Error',
                'success': False,
                'error': str(e)
            }
    
    def _determine_source(self, response: str) -> str:
        """Determine which source was used based on response content."""
        response_lower = response.lower()
        
        if 'wikipedia' in response_lower:
            return 'Wikipedia'
        elif 'arxiv' in response_lower:
            return 'ArXiv'
        elif 'internal' in response_lower or 'document' in response_lower:
            return 'Internal Documents'
        else:
            return 'Unknown'
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the agent system."""
        return {
            'model': 'gpt-3.5-turbo',
            'tools_count': len(self.tools),
            'tools': [tool.name for tool in self.tools],
            'tool_descriptions': get_tool_descriptions(),
            'includes_internal': self.include_internal,
            'agent_available': self.agent is not None
        }


def demo_direct_api_agent():
    """
    Demonstration of the direct API agent in action.
    """
    print("🚀 Direct API Agent Demo")
    print("=" * 50)
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ Please set OPENAI_API_KEY environment variable")
        return
    
    # Initialize agent
    agent = DirectAPIAgent(api_key, include_internal=False)
    
    # Show system info
    info = agent.get_system_info()
    print(f"📋 System Info:")
    print(f"   Model: {info['model']}")
    print(f"   Tools: {', '.join(info['tools'])}")
    print(f"   Agent Ready: {info['agent_available']}")
    
    # Test queries
    test_queries = [
        "What is pneumonia?",  # Should use Wikipedia
        "Latest research on COVID-19 vaccines",  # Should use ArXiv
        "Define hypertension and its causes",  # Should use Wikipedia
        "Recent studies on pulmonary fibrosis treatment",  # Should use ArXiv
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Query {i}: {query}")
        print("-" * 60)
        
        result = agent.query(query)
        
        if result['success']:
            print(f"✅ Source: {result['source']}")
            print(f"📄 Answer: {result['answer'][:300]}...")  # Truncate for demo
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
        
        print("-" * 60)


def create_simple_medical_bot():
    """
    Create a simple medical bot for interactive use.
    """
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return None
    
    agent = DirectAPIAgent(api_key, include_internal=False)
    
    print("🤖 Medical AI Bot Ready!")
    print("Ask me any medical question. Type 'quit' to exit.")
    print("Available sources: Wikipedia (definitions), ArXiv (research)")
    
    while True:
        try:
            user_input = input("\n❓ Your question: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            print("🔄 Processing...")
            result = agent.query(user_input)
            
            print(f"\n🎯 Source: {result['source']}")
            print(f"💬 Answer:\n{result['answer']}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'interactive':
        create_simple_medical_bot()
    else:
        demo_direct_api_agent()