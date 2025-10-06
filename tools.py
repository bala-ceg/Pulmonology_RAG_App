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
        
        # Calculate how much space we have left
        space_left = max_chars - len(combined_text) - 50  # 50 chars buffer for formatting and sources
        
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
            else:
                sources.append(os.path.basename(source) if source else "Unknown")
    
    # Add sources footer if we have sources and space
    if sources and len(combined_text) < max_chars - 100:
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
        print(f"Internal_VectorDB: Searching internal KB for '{query}'")
        
        if not rag_manager:
            return "Internal knowledge base is not available. Please ensure documents have been uploaded and the system is properly initialized."
        
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


# Tool registry for easy access
AVAILABLE_TOOLS = {
    'Wikipedia_Search': Wikipedia_Search,
    'ArXiv_Search': ArXiv_Search,
    'Internal_VectorDB': Internal_VectorDB
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
        'Internal_VectorDB': "Search uploaded PDFs and URLs in the internal knowledge base for user-specific content."
    }