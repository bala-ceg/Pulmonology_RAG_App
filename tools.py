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
import threading
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


# ---------------------------------------------------------------------------
# Threading-based timeout (safe in Flask/WSGI — same pattern as enhanced_tools)
# ---------------------------------------------------------------------------

def _run_with_timeout(func, args=(), kwargs=None, timeout_seconds: int = 15):
    """Run *func* in a daemon thread and return its result within *timeout_seconds*.

    Raises ``TimeoutError`` if the function doesn't finish in time.
    """
    if kwargs is None:
        kwargs = {}
    result_holder: List = [TimeoutError(f"Operation timed out after {timeout_seconds}s")]

    def _target():
        try:
            result_holder[0] = func(*args, **kwargs)
        except Exception as exc:
            result_holder[0] = exc

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout_seconds)

    if thread.is_alive():
        raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")
    if isinstance(result_holder[0], Exception):
        raise result_holder[0]
    return result_holder[0]


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
        
        # Load Wikipedia documents — 15 s timeout prevents indefinite blocking
        loader = WikipediaLoader(query=query, load_max_docs=2)
        try:
            docs = _run_with_timeout(loader.load, timeout_seconds=15)
        except TimeoutError:
            return f"Wikipedia search timed out for '{query}'. Please try a shorter or more specific query."
        
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
        
        # Load arXiv documents — 15 s timeout prevents indefinite blocking
        loader = ArxivLoader(query=query, load_max_docs=2)
        try:
            docs = _run_with_timeout(loader.load, timeout_seconds=15)
        except TimeoutError:
            return f"arXiv search timed out for '{query}'. Please try again with different terms."
        
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
                logger.warning(f"Internal_VectorDB: No session vector DB found for {session_id}")
                return "No documents found for your session — no uploaded content available."
        
        # Check if we have local content
        if not rag_manager.kb_local or rag_manager.get_local_content_count() == 0:
            logger.warning("Internal_VectorDB: No local content found")
            return "No relevant content found — no documents have been uploaded to the knowledge base."
        
        # Create retriever for local knowledge base
        retriever = rag_manager.kb_local.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 5}
        )
        
        # Apply guarded retrieval
        docs = guarded_retrieve(query, retriever, similarity_threshold=0.35)
        
        if docs is None:
            logger.info("Internal_VectorDB: Guard triggered — no sufficiently similar docs")
            return "No relevant content found in uploaded documents."
        
        # Add source type metadata for consistency
        for doc in docs:
            if 'source_type' not in doc.metadata:
                doc.metadata['source_type'] = 'internal'
        
        # Use the utility function to join and limit content
        result = _join_docs(docs, max_chars=1200)
        
        logger.info(f"Internal_VectorDB: Found {len(docs)} relevant chunks, returned {len(result)} characters")
        
        # If result is too generic or empty, let cascade continue
        if len(result) < 100 or _is_generic_content(result):
            logger.info("Internal_VectorDB: Result too generic — returning empty for cascade")
            return "No relevant content found in uploaded documents."
        
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

        # Guard: if the result dict carries no real EHR content, return a clean
        # empty signal so the cascade can continue to the next source.
        content_text = (result.get('summary') or result.get('content') or '').strip()
        _no_data_signals = (
            'no medical diagnosis data',
            'no diagnosis information',
            'no data found',
            'not found in the database',
            'no records found',
            'no current web information found',
            'no relevant current information found',
        )
        if not content_text or any(sig in content_text.lower() for sig in _no_data_signals):
            return "No medical diagnosis data found in the database."

        # Use the enhanced formatting system like other tools
        try:
            from enhanced_tools import format_enhanced_response
            formatted = format_enhanced_response(result)
            # Strip HTML tags for cascade text-matching; if nothing left, treat as empty
            import re as _re_pg
            plain = _re_pg.sub(r'<[^>]+>', ' ', formatted).strip()
            if not plain or any(sig in plain.lower() for sig in _no_data_signals):
                return "No medical diagnosis data found in the database."
            # Append a plain-text Sources footer that citations extractor can parse
            sources_line = "\n\nSources:\n- Source: PostgreSQL EHR Database"
            return formatted + sources_line
        except ImportError:
            # Fallback to simple formatting if enhanced_tools not available
            response_parts = []
            
            # Add summary if available
            if result.get('summary'):
                response_parts.append(f"**Medical Summary:**\n{result['summary']}\n")
            
            # Add content
            if result.get('content'):
                response_parts.append(f"**Detailed Information:**\n{result['content']}\n")
            
            # Always include a citations line — use result citations or a default label
            citation_text = result.get('citations') or "PostgreSQL EHR Database (pces_ehr_ccm)"
            response_parts.append(f"**Sources:** {citation_text}")
            
            return "\n".join(response_parts) if response_parts else 'No diagnosis information found.'
        
    except Exception as e:
        error_msg = f"Error searching diagnosis database: {str(e)}"
        logger.error(error_msg)
        return "No medical diagnosis data found in the database."


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
        # Normalize query: replace lookalike Cyrillic/Unicode characters with ASCII
        # (e.g. Cyrillic Ѕ→S, Т→T so "ЅVТ" becomes "SVT" for matching and embedding)
        import unicodedata as _ud
        _cyrillic_map = str.maketrans("ЅТАЕКМНОРСВХаеокрс", "STAEKMHOPCBXaeokpc")
        query_normalized = query.translate(_cyrillic_map)
        # Further normalize to ASCII where possible
        query_normalized = _ud.normalize("NFKD", query_normalized)
        query_normalized = "".join(c for c in query_normalized if ord(c) < 128 or not _ud.combining(c))
        logger.info("Pinecone_KB_Search: querying for '%s' (normalized: '%s')", query, query_normalized)
        kb = get_pinecone_kb()

        # Try to detect a department from the query to narrow the namespace
        namespace: Optional[str] = None
        q_lower = query_normalized.lower()
        dept_keywords = {
            "cardiology":       ["cardio", "heart", "cardiac", "atrial", "coronary", "ecg", "arrhythmia",
                                 "svt", "tachycardia", "bradycardia", "fibrillation", "flutter",
                                 "diltiazem", "verapamil", "adenosine", "amiodarone", "digoxin",
                                 "palpitation", "supraventricular", "ventricular", "ablation",
                                 "hypertension", "blood pressure", "statin", "antiplatelet",
                                 "rehabilitat", "revascular"],
            "neurology":        ["neuro", "brain", "stroke", "seizure", "epilep", "parkinson", "migraine",
                                 "dementia", "alzheimer", "multiple sclerosis", " ms ", "tpa", "alteplase",
                                 "levetiracetam", "lamotrigine", "valproate", "natalizumab"],
            "general_medicine": ["diabetes", "anaemia", "anemia", "kidney", "thyroid", "pneumonia",
                                 "metformin", "insulin", "glp-1", "sglt", "ferrous", "levothyroxine",
                                 "ckd", "haemoglobin", "hemoglobin", "curb-65"],
            "dentist":          ["dental", "tooth", "teeth", "gum", "periodontal", "caries", "tmj", "oral",
                                 "fluoride", "periodontitis", "abscess", "extraction", "root canal"],
            "pulmonology":      ["lung", "pulmon", "asthma", "copd", "respiratory", "breath", "inhaler",
                                 "sleep apnea", "apnoea", "obstructive sleep", "ics", "saba", "laba", "lama",
                                 "nintedanib", "pirfenidone", "cpap", "embolism", "interstitial lung"],
        }
        for dept, kws in dept_keywords.items():
            if any(kw in q_lower for kw in kws):
                namespace = dept
                break

        results = kb.query(query_normalized, namespace=namespace, top_k=5)

        if not results:
            return "No relevant content found in the PCES Pinecone knowledge base."

        # ── Similarity threshold ────────────────────────────────────────────
        # Only use results with a meaningful cosine similarity score.
        # Results below the threshold are too loosely related to the query
        # and would produce hallucinated or generic answers.
        SIMILARITY_THRESHOLD = 0.60   # lowered from 0.70 — sample data may score 0.62-0.68
        qualified = [r for r in results if r.get("score", 0.0) >= SIMILARITY_THRESHOLD]

        logger.info(
            "Pinecone: %d total results, %d above threshold %.2f (top score: %.3f)",
            len(results),
            len(qualified),
            SIMILARITY_THRESHOLD,
            results[0].get("score", 0.0) if results else 0.0,
        )

        if not qualified:
            top_score = results[0].get("score", 0.0) if results else 0.0
            return (
                f"No relevant content found in the PCES Pinecone knowledge base "
                f"(best similarity score {top_score:.3f} is below threshold {SIMILARITY_THRESHOLD})."
            )

        # Format results with detailed per-result citations + verbatim document excerpts
        answer_parts: List[str] = []        # combined text for LLM answer
        source_doc_blocks: List[str] = []   # per-result "Source Document" blocks shown to user
        citation_lines: List[str] = []
        seen_citations: set = set()
        total_chars = 0
        max_chars = 1200

        for idx, r in enumerate(qualified, start=1):
            txt = r.get("text", "").strip()
            if not txt:
                continue
            remaining = max_chars - total_chars
            if remaining <= 0:
                break
            chunk = txt[:remaining]
            answer_parts.append(chunk)
            total_chars += len(chunk)

            # ── Build metadata ───────────────────────────────────────────
            meta      = r.get("metadata", {})
            ns        = r.get("namespace", meta.get("namespace", ""))
            score     = r.get("score", 0.0)
            src       = meta.get("source") or meta.get("filename") or (f"PCES_{ns}" if ns else "PCES Pinecone KB")
            title     = meta.get("title", "")
            page      = meta.get("page") or meta.get("page_number")
            url       = meta.get("url") or meta.get("link") or ""
            dept      = (meta.get("department") or ns or "").replace("_", " ").title()
            doc_id    = meta.get("id") or meta.get("doc_id") or ""

            # ── Citation line ────────────────────────────────────────────
            cite_parts = []
            if dept:
                cite_parts.append(f"Dept: {dept}")
            if title and title != src:
                cite_parts.append(f'"{title}"')
            cite_parts.append(f"Source: {src}")
            if page is not None:
                cite_parts.append(f"Page {page}")
            cite_parts.append(f"Relevance: {score * 100:.0f}%")
            citation = f"[PCES Pinecone KB] {' | '.join(cite_parts)}"
            if citation not in seen_citations:
                seen_citations.add(citation)
                citation_lines.append(citation)

            # ── Source Document block (verbatim excerpt) ─────────────────
            doc_header_parts = [f"Document {idx}"]
            if dept:
                doc_header_parts.append(f"Dept: {dept}")
            if title and title != src:
                doc_header_parts.append(f'Title: "{title}"')
            doc_header_parts.append(f"Source: {src}")
            if page is not None:
                doc_header_parts.append(f"Page: {page}")
            if url:
                doc_header_parts.append(f"URL: {url}")
            if doc_id:
                doc_header_parts.append(f"ID: {doc_id}")
            doc_header_parts.append(f"Relevance: {score * 100:.0f}%")

            doc_block = (
                f"[Source Document {idx}]\n"
                f"{'  |  '.join(doc_header_parts)}\n"
                f"Excerpt:\n"
                f'  "{chunk}"\n'
            )
            source_doc_blocks.append(doc_block)

        body = "\n\n".join(answer_parts)
        if source_doc_blocks:
            body += "\n\n---\nSource Documents (exact retrieved excerpts):\n\n" + "\n".join(source_doc_blocks)
        if citation_lines:
            body += "\n\nSources:\n" + "\n".join(f"- {c}" for c in citation_lines)
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