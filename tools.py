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

from __future__ import annotations

import os
import re
from typing import List, Dict, Optional, Tuple
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from langchain_community.document_loaders import WikipediaLoader, ArxivLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    from langchain_core.tools import tool
except ImportError:
    def tool(func):
        """Simple decorator for tool functions."""
        func.is_tool = True
        return func
import numpy as np

from utils.error_handlers import get_logger

logger = get_logger(__name__)


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
            logger.debug("Guard: No documents contain main terms %s - triggering fallback", main_terms)
            return None
        
        # Check document quality - avoid very short or generic responses
        quality_docs = []
        for doc in relevant_docs:
            content = doc.page_content.strip()
            if len(content) > 50 and not _is_generic_content(content):
                quality_docs.append(doc)
        
        if not quality_docs:
            logger.debug("Guard: No quality documents found - triggering fallback")
            return None
        
        return quality_docs
        
    except Exception as e:
        logger.error("Error in guarded_retrieve: %s", e)
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
    Search Wikipedia for general knowledge, definitions, and encyclopedic medical information.

    USE this tool when:
    - User asks for definitions, explanations, or basic facts about any topic
    - Query is a general research / general knowledge question
    - Query uses words like "what is", "what are", "define", "explain", "tell me about",
      "overview", "background", "how does", "why does", "causes of", "history of"
    - User wants layman-friendly explanations of medical terms or conditions
    - General search query that does not specifically mention a department protocol,
      uploaded documents, patient history, or latest research papers

    DO NOT use this tool when:
    - Query asks for latest medical research / clinical studies (use ArXiv + Tavily)
    - Query mentions uploaded documents or "my files" (use Internal_VectorDB)
    - Query is about patient history or EHR records (use PostgreSQL)
    - Query is about department-specific clinical protocols (use Pinecone_KB_Search)
    """
    try:
        logger.info(f"Wikipedia_Search: Searching for '{query}'")
        
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
        
        logger.info(f"Wikipedia_Search: Found {len(docs)} articles, returned {len(result)} characters")
        return result
        
    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}. Please try rephrasing your query or check your internet connection."


@tool
def ArXiv_Search(query: str) -> str:
    """
    Search arXiv for medical research papers and scientific studies.
    ALWAYS paired with Tavily_Search for medical research queries.

    USE this tool when:
    - Query is about medical research: clinical trials, randomised controlled trials,
      systematic reviews, meta-analyses, peer-reviewed studies, research papers
    - User asks for "latest research", "recent study", "new paper", "clinical evidence",
      "research findings", "evidence-based", "published results"
    - Query mentions medical journals, preprints, or scientific evidence on a condition
    - User asks for "breakthrough treatment" or "novel therapy" backed by studies

    DO NOT use this tool when:
    - User asks general knowledge / definition questions (use Wikipedia)
    - Query is about uploaded documents or "my files" (use Internal_VectorDB)
    - Query is about patient history or EHR records (use PostgreSQL)
    - Query is about department protocols (use Pinecone_KB_Search)
    """
    try:
        logger.info(f"ArXiv_Search: Searching for '{query}'")
        
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
        
        logger.info(f"ArXiv_Search: Found {len(docs)} papers, returned {len(result)} characters")
        return result
        
    except Exception as e:
        return f"Error searching arXiv: {str(e)}. Please try using more specific scientific terms or check your internet connection."


@tool
def Tavily_Search(query: str) -> str:
    """
    Search current web for real-time medical updates, regulatory news, and breaking guidelines.
    ALWAYS paired with ArXiv_Search for medical research queries.

    USE this tool when:
    - Query is about medical research — use together with ArXiv_Search
    - User asks about "current", "latest", "recent" guidelines or recommendations
    - Query mentions organisations: "FDA", "WHO", "CDC", "AMA", "NICE"
    - Need real-time drug recalls, safety alerts, regulatory approvals, or policy updates
    - User asks about "today", "this year", "2024", "2025", "2026" events
    - Query explicitly uses words: "breaking news", "current guidelines", "recent updates",
      "latest recommendations", "real-time"

    DO NOT use this tool when:
    - User asks general knowledge questions (use Wikipedia)
    - Query is about uploaded documents (use Internal_VectorDB)
    - Query is about patient history or EHR (use PostgreSQL)
    - Query is about historical / background information (use Wikipedia)
    """
    try:
        logger.info(f"Tavily_Search: Searching web for '{query}'")
        
        # Import Tavily client
        try:
            from tavily import TavilyClient
        except ImportError:
            return f"Tavily package not installed. Falling back to general knowledge...\n\n{Wikipedia_Search.invoke(query)}"
        
        # Get API key from environment
        tavily_api_key = os.getenv('TAVILY_API_KEY')
        if not tavily_api_key:
            return f"Tavily API key not configured. Falling back to general knowledge...\n\n{Wikipedia_Search.invoke(query)}"
        
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
            return f"No current web information found for '{query}'. Falling back to general knowledge...\n\n{Wikipedia_Search.invoke(query)}"
        
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
            return f"No relevant current information found for '{query}'. Falling back to general knowledge...\n\n{Wikipedia_Search.invoke(query)}"
        
        # Use the same utility function as other tools for consistent formatting
        result_text = _join_docs(docs, max_chars=1200)
        
        logger.info(f"Tavily_Search: Found {len(search_results['results'])} web results, returned {len(result_text)} characters")
        return result_text
        
    except Exception as e:
        error_msg = f"Error searching web with Tavily: {str(e)}"
        logger.error(error_msg)
        # Fallback to Wikipedia on error
        return f"{error_msg}\n\nFalling back to general knowledge...\n\n{Wikipedia_Search.invoke(query)}"


@tool
def Internal_VectorDB(query: str, session_id: str = None, rag_manager=None) -> str:
    """
    Search the user's uploaded documents (PDFs, URLs) — Main RAG + Adhoc RAG tool.

    USE this tool when:
    - User specifically mentions "uploaded documents", "my files", "my PDFs", "my data"
    - Query refers to content previously uploaded to the system
    - User asks about information "from the documents I uploaded"
    - Query mentions specific document names or content unique to uploaded files
    - User wants to query their own organisational / session knowledge base

    DO NOT use this tool when:
    - User asks general medical questions without referencing uploaded content
    - Query is about latest medical research (use ArXiv + Tavily)
    - Query is about general knowledge / definitions (use Wikipedia)
    - Query is about patient EHR / history records (use PostgreSQL)
    - No documents have been uploaded to the system
    """
    try:
        logger.info(f"Internal_VectorDB: Searching internal KB for '{query}' (session: {session_id})")
        
        if not rag_manager:
            return "Internal knowledge base is not available. Please ensure documents have been uploaded and the system is properly initialized."
        
        # Load session-specific vector database if session_id provided
        session_kb_loaded = False
        if session_id:
            session_kb_loaded = rag_manager.load_session_vector_db(session_id)
            if not session_kb_loaded:
                logger.warning(f"Internal_VectorDB: No session vector DB found for {session_id} - triggering Wikipedia fallback")
                return f"No documents found for your session ({session_id}). Searching general knowledge instead...\n\n{Wikipedia_Search.invoke(query)}"
        
        # Check if we have local content
        if not rag_manager.kb_local or rag_manager.get_local_content_count() == 0:
            logger.warning("Internal_VectorDB: No local content found - triggering Wikipedia fallback")
            # Fallback to Wikipedia for better user experience
            return Wikipedia_Search.invoke(query)
        
        # Create retriever for local knowledge base
        retriever = rag_manager.kb_local.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 5}
        )
        
        # Apply guarded retrieval
        docs = guarded_retrieve(query, retriever, similarity_threshold=0.35)
        
        if docs is None:
            logger.info("Internal_VectorDB: Guard triggered - falling back to Wikipedia")
            return f"No relevant information found in uploaded documents. Searching general knowledge instead...\n\n{Wikipedia_Search.invoke(query)}"
        
        # Add source type metadata for consistency
        for doc in docs:
            if 'source_type' not in doc.metadata:
                doc.metadata['source_type'] = 'internal'
        
        # Use the utility function to join and limit content
        result = _join_docs(docs, max_chars=1200)
        
        logger.info(f"Internal_VectorDB: Found {len(docs)} relevant chunks, returned {len(result)} characters")
        
        # If result is too generic or empty, fallback to Wikipedia
        if len(result) < 100 or _is_generic_content(result):
            logger.info("Internal_VectorDB: Result too generic - falling back to Wikipedia")
            return f"Limited relevant information in uploaded documents. Supplementing with general knowledge...\n\n{Wikipedia_Search.invoke(query)}"
        
        return result
        
    except Exception as e:
        error_msg = f"Error searching internal knowledge base: {str(e)}"
        logger.error(error_msg)
        # Fallback to Wikipedia on error
        return f"{error_msg}\n\nFalling back to general knowledge...\n\n{Wikipedia_Search.invoke(query)}"


@tool
def PostgreSQL_Diagnosis_Search(query: str) -> str:
    """
    Search PostgreSQL EHR database for patient history, diagnosis records, and medical codes.

    USE this tool when:
    - Query is about PATIENT HISTORY - "patient history", "patient record", "my patient",
      "case history", "clinical history", "admission record", "EHR", "electronic health record"
    - User asks about specific diagnosis codes (ICD codes, D1xxx codes)
    - User wants data from hospital/clinical records in the database
    - Query contains "medical records", "diagnosis records", "hospital database",
      "p_diagnosis", "diagnosis code", "condition code", "medical code"

    DO NOT use this tool when:
    - User asks general medical knowledge questions (use Wikipedia or Pinecone)
    - Query is about medical research papers (use ArXiv + Tavily)
    - Query is about uploaded documents (use Internal_VectorDB)
    - Query is about department protocols (use Pinecone_KB_Search)
    """
    try:
        logger.info(f"PostgreSQL_Diagnosis_Search: Searching diagnosis database for '{query}'")
        
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
        logger.error(error_msg)
        # Fallback to Wikipedia for general medical information
        return f"{error_msg}\n\nFalling back to general medical knowledge...\n\n{Wikipedia_Search.invoke(query)}"


# ---------------------------------------------------------------------------
# Pinecone KB tool (optional — gracefully absent if Pinecone not configured)
# ---------------------------------------------------------------------------
try:
    from pinecone_kb import get_pinecone_kb, PINECONE_KB_AVAILABLE as _PINECONE_KB_AVAILABLE
except ImportError:
    _PINECONE_KB_AVAILABLE = False


@tool
def Pinecone_KB_Search(query: str) -> str:
    """
    Search the PCES Pinecone organisation knowledge base - the DEFAULT tool for
    clinical / medical queries that are not general knowledge, research papers,
    patient history, or uploaded documents.

    USE this tool when:
    - Query is a general medical search that does NOT match Wikipedia (general knowledge),
      ArXiv/Tavily (medical research), PostgreSQL (patient history), or Internal_VectorDB (uploads)
    - User asks about clinical guidelines, protocols, treatment pathways, or standard of care
    - Query is department-specific: cardiology, neurology, pulmonology, dentist/dental,
      general medicine / general_medicine
    - Query contains words like "protocol", "guideline", "standard of care", "PCES",
      "management of", "treatment of", "therapy for", "care pathway", "best practice"
    - You need authoritative clinical content beyond what Wikipedia offers

    DO NOT use this tool when:
    - User asks general knowledge / definition questions (use Wikipedia)
    - Query is about medical research papers (use ArXiv + Tavily)
    - Query is about patient EHR / history (use PostgreSQL)
    - User refers to their own uploaded documents (use Internal_VectorDB)
    """
    if not _PINECONE_KB_AVAILABLE:
        return "Pinecone KB is not configured. Please set PINECONE_API_KEY in the environment."

    try:
        logger.info("Pinecone_KB_Search: querying for '%s'", query)
        kb = get_pinecone_kb()

        # Try to detect a department from the query to narrow the namespace
        namespace: Optional[str] = None
        q_lower = query.lower()
        dept_keywords = {
            "cardiology":       ["cardio", "heart", "cardiac", "atrial", "coronary", "ecg", "arrhythmia"],
            "neurology":        ["neuro", "brain", "stroke", "seizure", "epilep", "parkinson", "migraine"],
            "general_medicine": ["diabetes", "hypertension", "anaemia", "anemia", "kidney", "thyroid", "pneumonia"],
            "dentist":          ["dental", "tooth", "teeth", "gum", "periodontal", "caries", "tmj", "oral"],
            "pulmonology":      ["lung", "pulmon", "asthma", "copd", "respiratory", "breath", "inhaler", "sleep apnea"],
        }
        for dept, kws in dept_keywords.items():
            if any(kw in q_lower for kw in kws):
                namespace = dept
                break

        results = kb.query(query, namespace=namespace, top_k=5)

        if not results:
            return "No relevant content found in the PCES Pinecone knowledge base."

        # Format results with sources footer (mirrors _join_docs pattern)
        parts: List[str] = []
        sources: List[str] = []
        total_chars = 0
        max_chars = 1200

        for r in results:
            txt = r.get("text", "").strip()
            if not txt:
                continue
            remaining = max_chars - total_chars
            if remaining <= 0:
                break
            chunk = txt[:remaining]
            parts.append(chunk)
            total_chars += len(chunk)

            ns  = r.get("namespace", "")
            src = r.get("metadata", {}).get("source", f"PCES_{ns}")
            if src and src not in sources:
                sources.append(src)

        body = "\n\n".join(parts)
        if sources:
            body += f"\n\nSources: {', '.join(sources)}"
        return body

    except Exception as exc:
        logger.error("Pinecone_KB_Search error: %s", exc)
        return f"Pinecone KB search error: {exc}"


# Tool registry for easy access
AVAILABLE_TOOLS = {
    'Wikipedia_Search': Wikipedia_Search,
    'ArXiv_Search': ArXiv_Search,
    'Tavily_Search': Tavily_Search,
    'Internal_VectorDB': Internal_VectorDB,
    'PostgreSQL_Diagnosis_Search': PostgreSQL_Diagnosis_Search,
    'Pinecone_KB_Search': Pinecone_KB_Search,
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
        'PostgreSQL_Diagnosis_Search': "Search PostgreSQL database for medical diagnosis information, codes, and clinical records from p_diagnosis table.",
        'Pinecone_KB_Search': "Search PCES organisation Pinecone knowledge base for curated department clinical guidelines, protocols, and standard-of-care content.",
    }