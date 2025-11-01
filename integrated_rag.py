"""
Integrated Medical RAG System with Tool Routing
==============================================

This module demonstrates the integration of the tool-based RAG architecture
with intelligent routing, guarded retrieval, and fallback mechanisms.
"""

import os
from typing import Dict, List, Optional, Any
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.schema import Document
try:
    from langchain.tools import tool
except ImportError:
    # Fallback for older versions or different package structure
    def tool(func):
        """Simple decorator for tool functions."""
        func.is_tool = True
        return func

# Import our custom modules
from rag_architecture import TwoStoreRAGManager, MedicalQueryRouter
from tools import Wikipedia_Search, ArXiv_Search, Tavily_Search, Internal_VectorDB, PostgreSQL_Diagnosis_Search, AVAILABLE_TOOLS
from prompts import ROUTING_SYSTEM_PROMPT, get_routing_explanation


class IntegratedMedicalRAG:
    """
    Integrated Medical RAG system that combines tool routing, guarded retrieval,
    and fallback mechanisms for comprehensive medical question answering.
    """
    
    def __init__(self, openai_api_key: str, base_vector_path: str = "./vector_dbs"):
        """
        Initialize the integrated medical RAG system.
        
        Args:
            openai_api_key: OpenAI API key
            base_vector_path: Base path for vector databases
        """
        self.openai_api_key = openai_api_key
        self.base_vector_path = base_vector_path
        
        # Initialize OpenAI components
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model="gpt-4",
            temperature=0.1
        )
        
        # Initialize RAG manager
        self.rag_manager = TwoStoreRAGManager(
            embeddings=self.embeddings,
            llm=self.llm,
            base_vector_path=base_vector_path
        )
        
        # Initialize query router
        self.router = MedicalQueryRouter(rag_manager=self.rag_manager)
        
        # Setup tools with RAG manager context
        self.tools = self._setup_tools()
        
        # Initialize agent
        self.agent = self._initialize_agent()
    
    def _setup_tools(self) -> List:
        """Setup tools with proper context injection."""
        tools = []
        
        # Create Wikipedia, ArXiv, Tavily, and PostgreSQL tools (these don't need RAG manager)
        tools.extend([Wikipedia_Search, ArXiv_Search, Tavily_Search, PostgreSQL_Diagnosis_Search])
        
        # Create Internal_VectorDB tool with RAG manager injected using @tool decorator
        @tool
        def internal_vectordb_with_context(query: str, session_id: str = None) -> str:
            """Internal VectorDB tool with RAG manager context."""
            return Internal_VectorDB(query, session_id, self.rag_manager)
        
        # Set tool metadata to match original
        internal_vectordb_with_context.name = Internal_VectorDB.name
        internal_vectordb_with_context.description = Internal_VectorDB.description
        
        tools.append(internal_vectordb_with_context)
        
        return tools
    
    def _initialize_agent(self):
        """Initialize the LangChain agent with tools and system prompt."""
        try:
            agent = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                max_iterations=3,
                early_stopping_method="generate"
            )
            
            # Inject our routing system prompt
            agent.agent.llm_chain.prompt.template = f"{ROUTING_SYSTEM_PROMPT}\n\n{agent.agent.llm_chain.prompt.template}"
            
            return agent
            
        except Exception as e:
            print(f"Error initializing agent: {e}")
            return None
    
    def query(self, question: str, session_id: str = None, patient_context: str = None) -> Dict[str, Any]:
        """
        Process a medical query using intelligent tool routing.
        
        Args:
            question: User's medical question
            session_id: Session identifier for user-specific content
            patient_context: Patient problem context for medical consultation
            
        Returns:
            Dictionary with response, routing info, and metadata
        """
        try:
            # Step 1: Route the query to appropriate tools
            routing_result = self.router.route_tools(question, session_id)
            
            print(f"Routing Decision: {routing_result['primary_tool']}")
            print(f"Confidence: {routing_result['confidence']}")
            print(f"Reasoning: {routing_result['reasoning']}")
            
            # Step 2: Use only the top 1-2 tools as suggested
            allowed_tools = routing_result['ranked_tools'][:2]
            
            # Step 3: Execute query with selected tools
            if self.agent:
                # Temporarily restrict agent to selected tools
                original_tools = self.agent.tools
                
                # Debug: Print tool matching
                print(f"Allowed tools: {allowed_tools}")
                print(f"Available tool names: {[tool.name for tool in self.tools]}")
                
                # Fixed tool filtering logic - exact name match (case insensitive)
                self.agent.tools = [tool for tool in self.tools 
                                  if any(tool.name.lower() == allowed_tool.lower() 
                                        for allowed_tool in allowed_tools)]
                
                print(f"Filtered tools: {[tool.name for tool in self.agent.tools]}")
                
                try:
                    response = self.agent.run(question)
                finally:
                    # Restore original tools
                    self.agent.tools = original_tools
            else:
                # Fallback: Direct tool execution
                response = self._direct_tool_execution(question, allowed_tools[0], session_id, patient_context)
            
            # Step 4: Generate routing explanation
            explanation = get_routing_explanation(
                routing_result['primary_tool'],
                routing_result['confidence'],
                len(allowed_tools) > 1
            )
            
            return {
                'answer': response,
                'routing_info': routing_result,
                'explanation': explanation,
                'tools_used': allowed_tools,
                'session_id': session_id
            }
            
        except Exception as e:
            print(f"Error processing query: {e}")
            return {
                'answer': f"I encountered an error processing your question: {str(e)}. Please try rephrasing your question.",
                'routing_info': {'error': str(e)},
                'explanation': "Error occurred during processing",
                'tools_used': ['Error'],
                'session_id': session_id
            }
    
    def _direct_tool_execution(self, question: str, tool_name: str, session_id: str = None, patient_context: str = None) -> str:
        """
        Direct tool execution fallback when agent is not available.
        
        Args:
            question: User question
            tool_name: Name of tool to execute
            session_id: Session identifier
            patient_context: Patient problem context for medical consultation
            
        Returns:
            Tool response string
        """
        try:
            if tool_name == 'Wikipedia_Search':
                # Use enhanced Wikipedia search with summaries and citations
                from enhanced_tools import enhanced_wikipedia_search, format_enhanced_response
                result = enhanced_wikipedia_search(question, patient_context)
                return format_enhanced_response(result)
            elif tool_name == 'ArXiv_Search':
                # Use enhanced ArXiv search with summaries and citations
                from enhanced_tools import enhanced_arxiv_search, format_enhanced_response
                result = enhanced_arxiv_search(question, patient_context)
                return format_enhanced_response(result)
            elif tool_name == 'Internal_VectorDB':
                # Use enhanced Internal VectorDB search with summaries and citations
                from enhanced_tools import enhanced_internal_search, format_enhanced_response
                result = enhanced_internal_search(question, session_id, self.rag_manager, patient_context)
                return format_enhanced_response(result)
            elif tool_name == 'Tavily_Search':
                # Use enhanced Tavily search with summaries and citations
                from enhanced_tools import enhanced_tavily_search, format_enhanced_response
                result = enhanced_tavily_search(question, patient_context)
                return format_enhanced_response(result)
            elif tool_name == 'PostgreSQL_Diagnosis_Search':
                # Use PostgreSQL database search for diagnosis information
                result = PostgreSQL_Diagnosis_Search(question)
                return result
            else:
                return f"Unknown tool: {tool_name}. Falling back to Wikipedia."
                
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"
    
    def add_documents_to_local(self, documents: List[Document]) -> None:
        """Add documents to local knowledge base."""
        self.rag_manager.add_documents_to_local(documents)
    
    def add_documents_to_external(self, documents: List[Document]) -> None:
        """Add documents to external knowledge base."""
        self.rag_manager.add_documents_to_external(documents)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and configuration."""
        return {
            'local_document_count': self.rag_manager.get_local_content_count(),
            'external_document_count': self.rag_manager.get_external_content_count(),
            'available_tools': list(AVAILABLE_TOOLS.keys()),
            'routing_keywords': {
                'arxiv': self.router.arxiv_keywords[:5],  # Show first 5
                'internal': self.router.internal_keywords[:5],
                'wikipedia': self.router.wikipedia_keywords[:5],
                'tavily': self.router.tavily_keywords[:5]
            },
            'vector_db_paths': {
                'local': self.rag_manager.kb_local_path,
                'external': self.rag_manager.kb_external_path
            }
        }


def demo_integration():
    """
    Demonstration function showing how to use the integrated system.
    """
    # Note: Replace with actual API key
    api_key = os.getenv('OPENAI_API_KEY', 'your-api-key-here')
    
    if api_key == 'your-api-key-here':
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    # Initialize system
    rag_system = IntegratedMedicalRAG(api_key)
    
    # Check system status
    status = rag_system.get_system_status()
    print("System Status:")
    print(f"  Local documents: {status['local_document_count']}")
    print(f"  External documents: {status['external_document_count']}")
    print(f"  Available tools: {status['available_tools']}")
    
    # Example queries demonstrating different routing scenarios
    test_queries = [
        "What is hypertension?",  # Should route to Wikipedia
        "Latest research on COVID-19 treatments",  # Should route to ArXiv
        "Recent preprints on pulmonary fibrosis mechanisms and pathogenesis",  # Likely route to ArXiv
        "What does my uploaded PDF say about treatment protocols?",  # Should route to Internal (with fallback)
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        result = rag_system.query(query)
        
        print(f"Primary Tool: {result['routing_info']['primary_tool']}")
        print(f"Confidence: {result['routing_info']['confidence']}")
        print(f"Reasoning: {result['routing_info']['reasoning']}")
        print(f"Answer: {result['answer'][:200]}...")  # Truncate for demo


if __name__ == "__main__":
    demo_integration()