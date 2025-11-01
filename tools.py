"""
Self-Describing Medical Tools for RAG Architecture
===============================================

This module implements three self-describing tools that return plain string outputs:
1. Wikipedia_Search: For definitions, factual explanations, and layman answers.
2. ArXiv_Search: For research questions or "latest paper" style queries.
3. Internal_VectorDB: For questions referring to uploaded PDFs or URLs.

Each tool includes strong descriptions with "use" and "do not use" conditions
and implements a post-retrieval guard system with Wikipedia fallback.
"""

import os
import re
from typing import List, Dict, Optional, Tuple
from langchain.schema import Document
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from langchain_community.document_loaders import WikipediaLoader, ArxivLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
try:
    from langchain.tools import tool
except ImportError:
    # Fallback for older versions or different package structure
    def tool(func):
        """Simple decorator for tool functions."""
        func.is_tool = True
        return func
import numpy as np


def _join_docs(docs: List[Document], max_chars: int = 1200) -> str:
    """
    Utility function to join document chunks with character limit.
    
    Args:
        docs: List of Document objects
        max_chars: Maximum characters to include
        
    Returns:
        Concatenated text string with Sources footer if possible
    """
    if not docs:
        return "No relevant information found."
    
    combined_text = ""
    sources = []
    
    for doc in docs:
        content = doc.page_content.strip()
        metadata = doc.metadata
        
        # Calculate how much space we have left (reserve more space for sources)
        space_left = max_chars - len(combined_text) - 150  # 150 chars buffer for formatting and sources
        
        if space_left <= 0:
            break  # No more space
            
        # Truncate content if necessary to fit within the limit
        if len(content) > space_left:
            content = content[:space_left-3] + "..."  # -3 for ellipsis
        
        # Add the content
        combined_text += content + "\n\n"
        
        # Extract source information
        if metadata.get('source'):
            source = metadata['source']
            if metadata.get('source_type') == 'wikipedia':
                sources.append(f"Wikipedia: {metadata.get('title', 'Unknown')}")
            elif metadata.get('source_type') == 'arxiv':
                sources.append(f"arXiv: {metadata.get('Title', 'Unknown')}")
            elif metadata.get('source_type') == 'tavily':
                # Format Tavily sources consistently with other tools
                sources.append(f"Web: {metadata.get('title', 'Unknown')}")
            else:
                sources.append(os.path.basename(source) if source else "Unknown")
    
    # Add sources footer if we have sources and space
    if sources and len(combined_text) < max_chars - 50:
        sources_text = "\n\nSources: " + ", ".join(list(set(sources)))
        if len(combined_text) + len(sources_text) <= max_chars:
            combined_text += sources_text
    
    return combined_text.strip() if combined_text.strip() else "No relevant information found."


def _extract_main_nouns(query: str) -> List[str]:
    """
    Extract main nouns from query for similarity checking.
    
    Args:
        query: User query string
        
    Returns:
        List of main nouns/keywords
    """
    # Simple keyword extraction - remove stop words and extract meaningful terms
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'what', 'how', 'why', 'when', 'where'}
    
    # Remove punctuation and convert to lowercase
    cleaned_query = re.sub(r'[^\w\s]', '', query.lower())
    words = cleaned_query.split()
    
    # Filter out stop words and short words
    main_terms = [word for word in words if word not in stop_words and len(word) > 2]
    
    return main_terms


def guarded_retrieve(query: str, retriever, similarity_threshold: float = 0.35) -> Optional[List[Document]]:
    """
    Post-retrieval guard that checks similarity and content relevance.
    
    Args:
        query: User query string
        retriever: LangChain retriever object
        similarity_threshold: Minimum average similarity score
        
    Returns:
        Documents if they pass the guard, None if they should trigger fallback
    """
    try:
        # Retrieve documents
        docs = retriever.invoke(query)
        
        if not docs:
            return None
        
        # Extract main query terms
        main_terms = _extract_main_nouns(query)
        
        if not main_terms:
            # If we can't extract terms, use simple length check
            if all(len(doc.page_content.strip()) < 50 for doc in docs):
                return None
            return docs
        
        # Check if any of the top chunks contain main nouns
        relevant_docs = []
        for doc in docs:
            content_lower = doc.page_content.lower()
            if any(term in content_lower for term in main_terms):
                relevant_docs.append(doc)
        
        # If no documents contain main terms, trigger fallback
        if not relevant_docs:
            print(f"Guard: No documents contain main terms {main_terms} - triggering fallback")
            return None
        
        # Check document quality - avoid very short or generic responses
        quality_docs = []
        for doc in relevant_docs:
            content = doc.page_content.strip()
            if len(content) > 50 and not _is_generic_content(content):
                quality_docs.append(doc)
        
        if not quality_docs:
            print("Guard: No quality documents found - triggering fallback")
            return None
        
        return quality_docs
        
    except Exception as e:
        print(f"Error in guarded_retrieve: {e}")
        return None


def _is_generic_content(content: str) -> bool:
    """Check if content is too generic or unhelpful."""
    generic_indicators = [
        "no information available",
        "not found in the context",
        "insufficient data",
        "cannot determine",
        "more information needed"
    ]
    
    content_lower = content.lower()
    return any(indicator in content_lower for indicator in generic_indicators)


@tool
def Wikipedia_Search(query: str) -> str:
    """
    Search Wikipedia for factual, encyclopedic information.
    
    USE this tool when:
    - User asks for definitions, explanations, or basic facts
    - Query seeks general knowledge or background information
    - User wants layman-friendly explanations of medical terms
    - Need factual context about diseases, conditions, or treatments
    - Query contains words like "what is", "define", "explain", "tell me about"
    
    DO NOT use this tool when:
    - User asks about "latest research", "recent papers", or "new studies"
    - Query specifically mentions uploaded documents or "my files"
    - User wants cutting-edge research or experimental findings
    - Query is about very specific clinical protocols or guidelines
    
    Args:
        query: The search query for Wikipedia
        
    Returns:
        Plain string with concatenated Wikipedia content and sources footer
    """
    try:
        print(f"Wikipedia_Search: Searching for '{query}'")
        
        # Load Wikipedia documents
        loader = WikipediaLoader(query=query, load_max_docs=3)
        docs = loader.load()
        
        if not docs:
            return f"No Wikipedia articles found for '{query}'. The topic might be too specific or use different terminology."
        
        # Add source type metadata
        for doc in docs:
            doc.metadata['source_type'] = 'wikipedia'
        
        # Use the utility function to join and limit content
        result = _join_docs(docs, max_chars=1200)
        
        print(f"Wikipedia_Search: Found {len(docs)} articles, returned {len(result)} characters")
        return result
        
    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}. Please try rephrasing your query or check your internet connection."


@tool
def ArXiv_Search(query: str) -> str:
    """
    Search arXiv for recent research papers and scientific studies.
    
    USE this tool when:
    - User asks about "latest research", "recent papers", or "new studies"
    - Query contains words like "research", "study", "paper", "findings"
    - User wants cutting-edge or experimental information
    - Query asks about "recent developments" or "current research"
    - Need scientific evidence or research-backed information
    
    DO NOT use this tool when:
    - User asks for basic definitions or general explanations
    - Query specifically mentions uploaded documents or "my files"
    - User wants simple, layman-friendly information
    - Query is about well-established, non-research topics
    - User asks about clinical protocols from their own documents
    
    Args:
        query: The search query for arXiv papers
        
    Returns:
        Plain string with concatenated arXiv paper content and sources footer
    """
    try:
        print(f"ArXiv_Search: Searching for '{query}'")
        
        # Load arXiv documents
        loader = ArxivLoader(query=query, load_max_docs=3)
        docs = loader.load()
        
        if not docs:
            return f"No arXiv papers found for '{query}'. Try using more specific scientific terminology or check if the topic has published research."
        
        # Add source type metadata
        for doc in docs:
            doc.metadata['source_type'] = 'arxiv'
        
        # Use the utility function to join and limit content
        result = _join_docs(docs, max_chars=1200)
        
        print(f"ArXiv_Search: Found {len(docs)} papers, returned {len(result)} characters")
        return result
        
    except Exception as e:
        return f"Error searching arXiv: {str(e)}. Please try using more specific scientific terms or check your internet connection."


@tool
def Tavily_Search(query: str) -> str:
    """
    Search current web information using Tavily API for real-time medical updates.
    
    USE this tool when:
    - User asks about "current", "latest", "recent" guidelines or protocols
    - Query mentions specific organizations like "FDA", "WHO", "CDC", "AMA"
    - Need real-time regulatory, policy, or breaking medical news
    - Query contains words like "current guidelines", "latest recommendations", "recent updates"
    - User asks about "today", "this year", "2024", "2025" or current time references
    - Need information about drug recalls, safety alerts, or recent approvals
    
    DO NOT use this tool when:
    - User asks for basic definitions or general explanations (use Wikipedia)
    - Query is about research papers or scientific studies (use ArXiv)
    - User wants information from uploaded documents (use Internal_VectorDB)
    - Query is about well-established medical knowledge that doesn't change frequently
    - User asks historical or background information
    
    Args:
        query: The search query for real-time web information
        
    Returns:
        Plain string with concatenated web search results and sources footer
    """
    try:
        print(f"Tavily_Search: Searching web for '{query}'")
        
        # Import Tavily client
        try:
            from tavily import TavilyClient
        except ImportError:
            return f"Tavily package not installed. Falling back to general knowledge...\n\n{Wikipedia_Search(query)}"
        
        # Get API key from environment
        tavily_api_key = os.getenv('TAVILY_API_KEY')
        if not tavily_api_key:
            return f"Tavily API key not configured. Falling back to general knowledge...\n\n{Wikipedia_Search(query)}"
        
        # Initialize Tavily client
        client = TavilyClient(api_key=tavily_api_key)
        
        # Perform search with medical focus
        search_results = client.search(
            query=f"medical {query}",
            search_depth="advanced",
            max_results=5,
            include_domains=["nih.gov", "cdc.gov", "who.int", "fda.gov", "mayoclinic.org", "webmd.com", "medscape.com"]
        )
        
        if not search_results or 'results' not in search_results:
            return f"No current web information found for '{query}'. Falling back to general knowledge...\n\n{Wikipedia_Search(query)}"
        
        # Convert Tavily results to Document format for consistency
        docs = []
        for result in search_results['results']:
            content = result.get('content', '').strip()
            title = result.get('title', 'Web Result')
            url = result.get('url', '')
            
            if content:
                # Create Document object matching other tools
                doc = Document(
                    page_content=content,
                    metadata={
                        'source': url,
                        'source_type': 'tavily',
                        'title': title,
                        'url': url
                    }
                )
                docs.append(doc)
        
        if not docs:
            return f"No relevant current information found for '{query}'. Falling back to general knowledge...\n\n{Wikipedia_Search(query)}"
        
        # Use the same utility function as other tools for consistent formatting
        result_text = _join_docs(docs, max_chars=1200)
        
        print(f"Tavily_Search: Found {len(search_results['results'])} web results, returned {len(result_text)} characters")
        return result_text
        
    except Exception as e:
        error_msg = f"Error searching web with Tavily: {str(e)}"
        print(error_msg)
        # Fallback to Wikipedia on error
        return f"{error_msg}\n\nFalling back to general knowledge...\n\n{Wikipedia_Search(query)}"


@tool
def Internal_VectorDB(query: str, session_id: str = None, rag_manager=None) -> str:
    """
    Search internal vector database containing uploaded PDFs and URLs.
    
    USE this tool when:
    - User specifically mentions "uploaded documents", "my files", or "my PDFs"
    - Query refers to content that was previously uploaded to the system
    - User asks about information "from the documents I uploaded"
    - Query mentions specific document names or content unique to uploaded files
    - User wants to analyze their own organizational knowledge base
    
    DO NOT use this tool when:
    - User asks general medical questions without referencing uploaded content
    - Query seeks widely available information that would be in Wikipedia
    - User asks about latest research that would be in arXiv
    - No documents have been uploaded to the system
    - Query is about general medical knowledge not specific to uploaded files
    
    Args:
        query: The search query for internal documents
        session_id: Session identifier for user-specific documents
        rag_manager: RAG manager instance for accessing vector databases
        
    Returns:
        Plain string with concatenated internal document content and sources footer
    """
    try:
        print(f"Internal_VectorDB: Searching internal KB for '{query}' (session: {session_id})")
        
        if not rag_manager:
            return "Internal knowledge base is not available. Please ensure documents have been uploaded and the system is properly initialized."
        
        # Load session-specific vector database if session_id provided
        session_kb_loaded = False
        if session_id:
            session_kb_loaded = rag_manager.load_session_vector_db(session_id)
            if not session_kb_loaded:
                print(f"Internal_VectorDB: No session vector DB found for {session_id} - triggering Wikipedia fallback")
                return f"No documents found for your session ({session_id}). Searching general knowledge instead...\n\n{Wikipedia_Search(query)}"
        
        # Check if we have local content
        if not rag_manager.kb_local or rag_manager.get_local_content_count() == 0:
            print("Internal_VectorDB: No local content found - triggering Wikipedia fallback")
            # Fallback to Wikipedia for better user experience
            return Wikipedia_Search(query)
        
        # Create retriever for local knowledge base
        retriever = rag_manager.kb_local.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 5}
        )
        
        # Apply guarded retrieval
        docs = guarded_retrieve(query, retriever, similarity_threshold=0.35)
        
        if docs is None:
            print("Internal_VectorDB: Guard triggered - falling back to Wikipedia")
            return f"No relevant information found in uploaded documents. Searching general knowledge instead...\n\n{Wikipedia_Search(query)}"
        
        # Add source type metadata for consistency
        for doc in docs:
            if 'source_type' not in doc.metadata:
                doc.metadata['source_type'] = 'internal'
        
        # Use the utility function to join and limit content
        result = _join_docs(docs, max_chars=1200)
        
        print(f"Internal_VectorDB: Found {len(docs)} relevant chunks, returned {len(result)} characters")
        
        # If result is too generic or empty, fallback to Wikipedia
        if len(result) < 100 or _is_generic_content(result):
            print("Internal_VectorDB: Result too generic - falling back to Wikipedia")
            return f"Limited relevant information in uploaded documents. Supplementing with general knowledge...\n\n{Wikipedia_Search(query)}"
        
        return result
        
    except Exception as e:
        error_msg = f"Error searching internal knowledge base: {str(e)}"
        print(error_msg)
        # Fallback to Wikipedia on error
        return f"{error_msg}\n\nFalling back to general knowledge...\n\n{Wikipedia_Search(query)}"


@tool
def PostgreSQL_Diagnosis_Search(query: str) -> str:
    """
    Search PostgreSQL database for medical diagnosis information from p_diagnosis table.
    
    USE this tool when:
    - User asks about specific medical diagnoses or conditions
    - Query mentions diagnosis codes (ICD, medical codes)
    - User wants information about available diagnoses in the database
    - Query seeks specific diagnostic information from hospital/clinical records
    - User asks "what diagnoses are available" or similar database queries
    - Query contains words like "diagnosis", "diagnostic", "medical code", "condition code"
    
    DO NOT use this tool when:
    - User asks for general medical information (use Wikipedia instead)
    - Query is about treatment protocols or procedures
    - User wants research papers or studies (use ArXiv instead) 
    - Query is about current medical news (use Tavily instead)
    - User asks about uploaded documents (use Internal_VectorDB instead)
    
    Args:
        query: The search query for diagnosis information
        
    Returns:
        Plain string with diagnosis information from database and sources footer
    """
    try:
        print(f"PostgreSQL_Diagnosis_Search: Searching diagnosis database for '{query}'")
        
        # Import the PostgreSQL tool
        try:
            from postgres_tool import enhanced_postgres_search
        except ImportError:
            return "PostgreSQL database tool not available. Please ensure psycopg2 is installed and database credentials are configured."
        
        # Search the diagnosis database
        result = enhanced_postgres_search(query)
        
        # Use the enhanced formatting system like other tools
        try:
            from enhanced_tools import format_enhanced_response
            return format_enhanced_response(result)
        except ImportError:
            # Fallback to simple formatting if enhanced_tools not available
            response_parts = []
            
            # Add summary if available
            if result.get('summary'):
                response_parts.append(f"**Medical Summary:**\n{result['summary']}\n")
            
            # Add content
            if result.get('content'):
                response_parts.append(f"**Detailed Information:**\n{result['content']}\n")
            
            # Add sources
            if result.get('citations'):
                response_parts.append(f"**Sources:** {result['citations']}")
            
            return "\n".join(response_parts) if response_parts else 'No diagnosis information found.'
        
    except Exception as e:
        error_msg = f"Error searching diagnosis database: {str(e)}"
        print(error_msg)
        # Fallback to Wikipedia for general medical information
        return f"{error_msg}\n\nFalling back to general medical knowledge...\n\n{Wikipedia_Search(query)}"


# Tool registry for easy access
AVAILABLE_TOOLS = {
    'Wikipedia_Search': Wikipedia_Search,
    'ArXiv_Search': ArXiv_Search,
    'Tavily_Search': Tavily_Search,
    'Internal_VectorDB': Internal_VectorDB,
    'PostgreSQL_Diagnosis_Search': PostgreSQL_Diagnosis_Search
}


def get_tool_descriptions() -> Dict[str, str]:
    """
    Get descriptions of all available tools for routing decisions.
    
    Returns:
        Dictionary mapping tool names to their descriptions
    """
    return {
        'Wikipedia_Search': "Search Wikipedia for factual, encyclopedic information, definitions, and general medical knowledge.",
        'ArXiv_Search': "Search arXiv for recent research papers, scientific studies, and cutting-edge findings.",
        'Tavily_Search': "Search current web information for real-time medical updates, guidelines, and breaking news.",
        'Internal_VectorDB': "Search uploaded PDFs and URLs in the internal knowledge base for user-specific content.",
        'PostgreSQL_Diagnosis_Search': "Search PostgreSQL database for medical diagnosis information, codes, and clinical records from p_diagnosis table."
    }