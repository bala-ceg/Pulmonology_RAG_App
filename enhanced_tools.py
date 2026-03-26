"""
Enhanced Medical Tools with LLM Summaries and Citations
======================================================

This module provides enhanced versions of the medical tools that include:
1. LLM-generated summaries of the content
2. Proper HTML formatting for tool routing information
3. Comprehensive citations for all sources
"""

import os
import re
import threading
from contextlib import contextmanager
from typing import List, Dict, Optional, Tuple
from langchain.schema import Document
from langchain_community.document_loaders import WikipediaLoader, ArxivLoader
from langchain_openai import ChatOpenAI

# Import original functions
from tools import _join_docs, guarded_retrieve, _is_generic_content

# Timeout handler for API calls using threading (works better than signal in Flask/WSGI)
class TimeoutException(Exception):
    pass

def run_with_timeout(func, args=(), kwargs={}, timeout_duration=30):
    """
    Run a function with a timeout using threading.
    More reliable than signal-based timeout, especially in Flask/WSGI contexts.
    """
    result = [TimeoutException(f"Operation timed out after {timeout_duration} seconds")]
    
    def target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            result[0] = e
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout_duration)
    
    if thread.is_alive():
        # Thread is still running, timeout occurred
        raise TimeoutException(f"Operation timed out after {timeout_duration} seconds")
    
    if isinstance(result[0], Exception):
        raise result[0]
    
    return result[0]


def enhanced_tavily_search(query: str, patient_context: str = None) -> Dict[str, str]:
    """Enhanced Tavily search with LLM summary and HTML formatting"""
    
    try:
        print(f"Enhanced Tavily_Search: Searching web for '{query}'")
        
        # Import Tavily client
        try:
            from tavily import TavilyClient
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            return {
                'content': f"Tavily package not installed. Unable to search for current information on '{query}'.",
                'summary': "Real-time web search is currently unavailable due to missing dependencies.",
                'citations': "",
                'tool_info': ""
            }
        
        # Get API key from environment
        tavily_api_key = os.getenv('TAVILY_API_KEY')
        if not tavily_api_key:
            return {
                'content': f"Tavily API key not configured. Unable to search for current information on '{query}'.",
                'summary': "Real-time web search is currently unavailable due to missing API configuration.",
                'citations': "",
                'tool_info': ""
            }
        
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
            return {
                'content': f"No current web information found for '{query}'.",
                'summary': "No recent information available on this topic from authoritative medical sources.",
                'citations': "",
                'tool_info': ""
            }
        
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
            return {
                'content': f"No relevant current information found for '{query}'.",
                'summary': "No recent information available on this topic from authoritative medical sources.",
                'citations': "",
                'tool_info': ""
            }
        
        # Get the raw content using the same utility function
        raw_content = _join_docs(docs, max_chars=1500)  # Slightly more content for better summaries
        
        # Generate LLM summary focused on current/recent information
        summary = generate_medical_summary_tavily(raw_content, query, patient_context)
        
        # Format citations for web sources
        citations = format_citations_html(docs, 'tavily')
        
        # Tool routing info
        tool_info = format_tool_routing_html(
            primary_tool="Tavily_Search",
            confidence="high", 
            tools_used=["Tavily_Search"],
            reasoning="High confidence in tool selection based on keyword analysis"
        )
        
        print(f"Enhanced Tavily_Search: Found {len(search_results['results'])} web results, generated summary")
        
        return {
            'content': raw_content,
            'summary': summary,
            'citations': citations,
            'tool_info': tool_info
        }
        
    except Exception as e:
        error_msg = f"Error searching web with Tavily: {str(e)}"
        print(error_msg)
        return {
            'content': error_msg,
            'summary': "Unable to retrieve current information due to search error.",
            'citations': "",
            'tool_info': ""
        }


def generate_medical_summary_tavily(content: str, query: str, patient_context: str = None) -> str:
    """Generate LLM summary specifically for Tavily web search results"""
    
    llm = get_llm_instance()
    if not llm:
        return ""  # Return empty if no LLM available
    
    try:
        # Base system message for medical AI with Tavily context
        system_message = "You are a medical AI assistant providing accurate, evidence-based medical information from current web sources and official health organization guidelines."
        
        # Add patient context to system message if provided
        if patient_context:
            system_message = f"Patient Context: {patient_context}\n\n{system_message} Always consider the patient context when providing medical advice and recommendations."
        
        # Create the user prompt for Tavily results
        user_prompt = f"""Please provide a concise, professional summary of the following current medical information in response to the query: "{query}"

This information comes from recent web sources including official health organizations, medical institutions, and authoritative medical websites.

Focus on:
- Current guidelines, recommendations, or recent updates
- Recent policy changes or new medical guidance
- Current best practices or treatment protocols
- Important dates or timelines mentioned
- Regulatory or organizational announcements

Use clear, accessible medical language and emphasize the currency and source authority of the information.

Content to summarize:
{content[:2000]}  # Limit content to avoid token limits

Provide a 2-3 sentence summary that highlights the most current and relevant information for the user's query."""
        
        # Create messages with system context
        from langchain.schema import HumanMessage, SystemMessage
        response = llm.invoke([
            SystemMessage(content=system_message),
            HumanMessage(content=user_prompt)
        ])
        
        if hasattr(response, 'content'):
            return response.content.strip()
        else:
            return str(response).strip()
        
    except Exception as e:
        print(f"Error generating Tavily summary: {e}")
        return ""


def test_enhanced_tools():
    """Test the enhanced medical tools"""
    
    print("🧪 Testing Enhanced Medical Tools")
    print("=" * 50)

def preprocess_medical_query(query: str) -> str:
    """Preprocess medical queries to improve search accuracy"""
    
    # Convert to lowercase for processing
    processed = query.lower()
    
    # Medical term normalizations
    medical_normalizations = {
        # Diabetes variants
        'type-2 diabetes': 'type 2 diabetes',
        'type-1 diabetes': 'type 1 diabetes', 
        'type 2 diabetes mellitus': 'type 2 diabetes',
        'type 1 diabetes mellitus': 'type 1 diabetes',
        'diabetes type 2': 'type 2 diabetes',
        'diabetes type 1': 'type 1 diabetes',
        'adult-onset diabetes': 'type 2 diabetes',
        'juvenile diabetes': 'type 1 diabetes',
        
        # Other medical terms
        'high blood pressure': 'hypertension',
        'heart attack': 'myocardial infarction',
        'stroke': 'cerebrovascular accident',
    }
    
    # Apply normalizations
    for original, normalized in medical_normalizations.items():
        if original in processed:
            processed = processed.replace(original, normalized)
    
    # Remove common query prefixes/suffixes that don't help search
    prefixes_to_remove = [
        'explain the ',
        'what are the ',
        'tell me about ',
        'describe the ',
        'what is ',
        'symptoms of ',
        'causes of ',
        'treatment for '
    ]
    
    for prefix in prefixes_to_remove:
        if processed.startswith(prefix):
            processed = processed[len(prefix):]
            break
    
    # Clean up extra spaces and capitalize properly
    processed = re.sub(r'\s+', ' ', processed).strip()
    
    # Capitalize first letter of each major word for Wikipedia search
    words = processed.split()
    if words:
        # Capitalize medical terms properly
        capitalized_words = []
        for word in words:
            if word in ['diabetes', 'hypertension', 'asthma', 'copd']:
                capitalized_words.append(word.lower())
            elif len(word) > 3:  # Capitalize longer words
                capitalized_words.append(word.capitalize())
            else:
                capitalized_words.append(word.lower())
        processed = ' '.join(capitalized_words)
    
    return processed

def filter_relevant_documents(docs: List[Document], original_query: str, processed_query: str) -> List[Document]:
    """Filter and rank documents by relevance to the query"""
    
    if not docs:
        return docs
    
    # Extract key terms from queries
    original_lower = original_query.lower()
    processed_lower = processed_query.lower()
    
    # Define query-specific relevance scoring
    def calculate_relevance_score(doc: Document) -> Tuple[int, str]:
        title = doc.metadata.get('title', '').lower()
        content = doc.page_content.lower()
        
        score = 0
        debug_info = []
        
        # Critical exact matches in title (highest priority)
        if 'type 2 diabetes' in original_lower or 'type-2 diabetes' in original_lower:
            if 'type 2 diabetes' in title:
                score += 100
                debug_info.append("Title: Type 2 diabetes match (+100)")  
            elif 'type 1 diabetes' in title:
                score -= 50  # Penalize wrong type
                debug_info.append("Title: Type 1 diabetes penalty (-50)")
                
        elif 'type 1 diabetes' in original_lower or 'type-1 diabetes' in original_lower:
            if 'type 1 diabetes' in title:
                score += 100
                debug_info.append("Title: Type 1 diabetes match (+100)")
            elif 'type 2 diabetes' in title:
                score -= 50  # Penalize wrong type  
                debug_info.append("Title: Type 2 diabetes penalty (-50)")
        
        # Processed query terms in title
        if processed_lower in title:
            score += 50
            debug_info.append(f"Title contains processed query (+50)")
        
        # Content relevance (lower weight)
        key_terms = processed_lower.split()
        for term in key_terms:
            if len(term) > 3:  # Ignore short words
                if term in title:
                    score += 10
                    debug_info.append(f"Title: '{term}' (+10)")
                elif term in content[:500]:  # First 500 chars are most relevant
                    score += 5
                    debug_info.append(f"Content: '{term}' (+5)")
        
        # Boost medical condition matches
        medical_terms = ['diabetes', 'hypertension', 'asthma', 'copd', 'symptoms', 'treatment']
        for term in medical_terms:
            if term in original_lower and term in title:
                score += 20
                debug_info.append(f"Medical term '{term}' in title (+20)")
        
        return score, "; ".join(debug_info)
    
    # Score all documents
    scored_docs = []
    for doc in docs:
        score, debug = calculate_relevance_score(doc)
        scored_docs.append((score, doc, debug))
        print(f"Document '{doc.metadata.get('title', 'Unknown')}': Score {score} ({debug})")
    
    # Sort by score (highest first)
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    # Return documents in relevance order
    filtered = [doc for score, doc, debug in scored_docs if score > -25]  # Filter out heavily penalized docs
    
    print(f"Content filtering: {len(docs)} -> {len(filtered)} documents after relevance filtering")
    
    return filtered if filtered else docs  # Return original if all filtered out

def get_llm_instance():
    """Get ChatOpenAI instance for generating summaries"""
    try:
        return ChatOpenAI(
            api_key=os.getenv('openai_api_key'),
            base_url=os.getenv('base_url'),
            model_name=os.getenv('llm_model_name', 'gpt-4o-mini'),
            temperature=0.1,
            request_timeout=30  # Add 30 second timeout
        )
    except:
        return None

def generate_medical_summary(content: str, query: str, tool_name: str, patient_context: str = None) -> str:
    """Generate LLM summary of medical content with optional patient context"""
    
    llm = get_llm_instance()
    if not llm:
        return ""  # Return empty if no LLM available
    
    try:
        # Base system message for medical AI
        system_message = "You are a medical AI assistant providing accurate, evidence-based medical information and guidance."
        
        # Add patient context to system message if provided
        if patient_context:
            system_message = f"Patient Context: {patient_context}\n\n{system_message} Always consider the patient context when providing medical advice and recommendations. Tailor your responses to the specific patient demographics, conditions, and medical history provided."
        
        # Create the user prompt
        user_prompt = f"""Please provide a concise, professional summary of the following medical information in response to the query: "{query}"

Focus on:
- Key medical facts and definitions
- Important symptoms, causes, or treatments mentioned
- Critical information that directly answers the user's question
- Use clear, accessible medical language

Content to summarize:
{content[:2000]}  # Limit content to avoid token limits

Provide a 2-3 sentence summary that directly addresses the user's query."""
        
        # Create messages with system context
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ]
        
        # Use the chat completion format for better context handling
        from langchain.schema import HumanMessage, SystemMessage
        response = llm.invoke([
            SystemMessage(content=system_message),
            HumanMessage(content=user_prompt)
        ])
        
        if hasattr(response, 'content'):
            return response.content.strip()
        else:
            return str(response).strip()
            
    except Exception as e:
        error_str = str(e)
        print(f"Error generating summary: {e}")
        
        # Check for specific error types
        if "insufficient_quota" in error_str.lower() or "429" in error_str:
            print("⚠️ OpenAI quota exceeded - summary generation skipped")
            return "[Summary unavailable: OpenAI quota exceeded. Please add credits to your OpenAI account.]"
        elif "timeout" in error_str.lower():
            print("⏱️ Summary generation timed out")
            return "[Summary unavailable: Request timed out]"
        else:
            return ""

def format_citations_html(docs: List[Document], source_type: str) -> str:
    """Format citations as HTML with proper links and formatting"""
    
    if not docs:
        return ""
    
    citations = []
    
    for doc in docs:
        metadata = doc.metadata
        
        if source_type == 'wikipedia':
            title = metadata.get('title', 'Unknown Wikipedia Article')
            source_url = metadata.get('source', '#')
            citations.append(f'<a href="{source_url}" target="_blank">{title}</a> (Wikipedia)')
            
        elif source_type == 'arxiv':
            title = metadata.get('Title', 'Unknown arXiv Paper')
            # ArXiv URLs are typically in the source field
            source_url = metadata.get('source', '#')
            authors = metadata.get('Authors', 'Unknown Authors')
            citations.append(f'<a href="{source_url}" target="_blank">{title}</a> by {authors} (arXiv)')
            
        elif source_type == 'tavily':
            title = metadata.get('title', 'Web Result')
            source_url = metadata.get('url', metadata.get('source', '#'))
            # Clean up title if it's too long
            display_title = title[:80] + "..." if len(title) > 80 else title
            citations.append(f'<a href="{source_url}" target="_blank">{display_title}</a> (Web)')
            
        elif source_type == 'internal':
            source = metadata.get('source', 'Unknown Document')
            filename = os.path.basename(source) if source else 'Unknown File'
            citations.append(f'{filename} (Internal Document)')
    
    if citations:
        unique_citations = list(set(citations))  # Remove duplicates
        return "<br>".join(unique_citations)
    
    return ""

def format_tool_routing_html(primary_tool: str, confidence: str, tools_used: List[str], reasoning: str = "") -> str:
    """Format tool routing information as HTML content (without header)"""
    
    # Use black text for confidence as requested
    confidence_color = '#000'

    # Make Internal_VectorDB appear bold when shown
    primary_tool_display = f"<strong>{primary_tool}</strong>" if str(primary_tool) == 'Internal_VectorDB' else primary_tool

    # Bold Internal_VectorDB in tools used list
    tools_display = [f"<strong>{t}</strong>" if t == 'Internal_VectorDB' else t for t in tools_used]

    parts = []
    parts.append(f'<span style="color: #495057;">Confidence:</span> <span style="color: {confidence_color}; font-weight: bold;">{confidence}</span><br>')
    parts.append(f'<span style="color: #495057;">Tools Used:</span> {", ".join(tools_display)}<br>')
    if reasoning:
        parts.append(f'<span style="color: #495057;">Reasoning:</span> {reasoning}')

    html = "".join(parts)

    return html

def enhanced_wikipedia_search(query: str, patient_context: str = None) -> Dict[str, str]:
    """Enhanced Wikipedia search with LLM summary and HTML formatting"""
    
    try:
        print(f"Enhanced Wikipedia_Search: Original query: '{query}'")
        
        # Preprocess the query for better search results
        processed_query = preprocess_medical_query(query)
        print(f"Enhanced Wikipedia_Search: Processed query: '{processed_query}'")
        
        # Load Wikipedia documents with processed query with timeout
        print(f"Enhanced Wikipedia_Search: Loading documents (30s timeout)...")
        try:
            def load_wikipedia():
                loader = WikipediaLoader(query=processed_query, load_max_docs=3)
                return loader.load()
            
            docs = run_with_timeout(load_wikipedia, timeout_duration=30)
            print(f"Enhanced Wikipedia_Search: Loaded {len(docs)} documents")
        except TimeoutException as te:
            print(f"⏱️ Wikipedia search timed out: {te}")
            return {
                'content': f"Wikipedia search timed out after 30 seconds for query '{query}'. The service may be experiencing delays or the topic may require more specific terminology.",
                'summary': "Wikipedia search timed out. Please try a more specific query or try again later.",
                'citations': "",
                'tool_info': format_tool_routing_html("Wikipedia_Search", "medium", ["Wikipedia_Search"], "Search operation timed out")
            }
        
        if not docs:
            return {
                'content': f"No Wikipedia articles found for '{query}'. The topic might be too specific or use different terminology.",
                'summary': "",
                'citations': "",
                'tool_info': ""
            }
        
        # Add source type metadata
        for doc in docs:
            doc.metadata['source_type'] = 'wikipedia'
        
        # Filter and rank documents based on relevance to original query
        filtered_docs = filter_relevant_documents(docs, query, processed_query)
        
        # Get the raw content from filtered documents
        raw_content = _join_docs(filtered_docs, max_chars=1200)
        
        # Generate LLM summary
        summary = generate_medical_summary(raw_content, query, "Wikipedia", patient_context)
        
        # Format citations
        citations = format_citations_html(docs, 'wikipedia')
        
        # Tool routing info
        tool_info = format_tool_routing_html(
            primary_tool="Wikipedia_Search",
            confidence="high",
            tools_used=["Wikipedia_Search"],
            reasoning="Query seeks general medical knowledge and definitions; Wikipedia selected for encyclopedic information"
        )
        
        print(f"Enhanced Wikipedia_Search: Found {len(docs)} articles, generated summary")
        
        return {
            'content': raw_content,
            'summary': summary,
            'citations': citations,
            'tool_info': tool_info
        }
        
    except Exception as e:
        error_msg = f"Error searching Wikipedia: {str(e)}"
        return {
            'content': error_msg,
            'summary': "",
            'citations': "",
            'tool_info': ""
        }

def enhanced_arxiv_search(query: str, patient_context: str = None) -> Dict[str, str]:
    """Enhanced ArXiv search with LLM summary and HTML formatting"""
    
    try:
        print(f"Enhanced ArXiv_Search: Searching for '{query}' (30s timeout)...")
        
        # Load arXiv documents with timeout
        try:
            def load_arxiv():
                loader = ArxivLoader(query=query, load_max_docs=3)
                return loader.load()
            
            docs = run_with_timeout(load_arxiv, timeout_duration=30)
            print(f"Enhanced ArXiv_Search: Loaded {len(docs)} documents")
        except TimeoutException as te:
            print(f"⏱️ ArXiv search timed out: {te}")
            return {
                'content': f"ArXiv search timed out after 30 seconds for query '{query}'. The service may be experiencing delays.",
                'summary': "ArXiv search timed out. Please try again later or use a different search term.",
                'citations': "",
                'tool_info': format_tool_routing_html("ArXiv_Search", "medium", ["ArXiv_Search"], "Search operation timed out")
            }
        
        if not docs:
            return {
                'content': f"No arXiv papers found for '{query}'. Try using more specific scientific terminology.",
                'summary': "",
                'citations': "",
                'tool_info': ""
            }
        
        # Add source type metadata
        for doc in docs:
            doc.metadata['source_type'] = 'arxiv'
        
        # Get the raw content
        raw_content = _join_docs(docs, max_chars=1200)
        
        # Generate LLM summary
        summary = generate_medical_summary(raw_content, query, "ArXiv", patient_context)
        
        # Format citations
        citations = format_citations_html(docs, 'arxiv')
        
        # Tool routing info
        tool_info = format_tool_routing_html(
            primary_tool="ArXiv_Search", 
            confidence="high",
            tools_used=["ArXiv_Search"],
            reasoning="Query contains research-oriented keywords; ArXiv selected for recent scientific papers and studies"
        )
        
        print(f"Enhanced ArXiv_Search: Found {len(docs)} papers, generated summary")
        
        return {
            'content': raw_content,
            'summary': summary,
            'citations': citations,
            'tool_info': tool_info
        }
        
    except Exception as e:
        error_msg = f"Error searching arXiv: {str(e)}"
        return {
            'content': error_msg,
            'summary': "",
            'citations': "",
            'tool_info': ""
        }

def enhanced_internal_search(query: str, session_id: str = None, rag_manager=None, patient_context: str = None) -> Dict[str, str]:
    """Enhanced Internal VectorDB search with LLM summary and HTML formatting"""
    
    try:
        print(f"Enhanced Internal_VectorDB: Searching for '{query}' (session: {session_id})")
        
        if not rag_manager:
            return {
                'content': "Internal knowledge base is not available. Please ensure documents have been uploaded.",
                'summary': "",
                'citations': "",
                'tool_info': ""
            }
        
        # Load session-specific vector database if session_id provided
        session_kb_loaded = False
        if session_id:
            session_kb_loaded = rag_manager.load_session_vector_db(session_id)
            if not session_kb_loaded:
                print(f"Enhanced Internal_VectorDB: No session vector DB found for {session_id} - falling back to Wikipedia")
                fallback_result = enhanced_wikipedia_search(query, patient_context)
                fallback_result['content'] = f"No documents found for your session. Showing general knowledge instead:\n\n{fallback_result['content']}"
                return fallback_result
        
        # Check if we have local content
        if not rag_manager.kb_local or rag_manager.get_local_content_count() == 0:
            print("Enhanced Internal_VectorDB: No local content - falling back to Wikipedia")
            return enhanced_wikipedia_search(query, patient_context)
        
        # Create retriever for local knowledge base
        retriever = rag_manager.kb_local.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # Apply guarded retrieval
        docs = guarded_retrieve(query, retriever, similarity_threshold=0.35)
        
        if docs is None:
            print("Enhanced Internal_VectorDB: Guard triggered - falling back to Wikipedia")
            fallback_result = enhanced_wikipedia_search(query, patient_context)
            fallback_result['content'] = f"No relevant information found in uploaded documents. Showing general knowledge instead:\n\n{fallback_result['content']}"
            return fallback_result
        
        # Add source type metadata
        for doc in docs:
            if 'source_type' not in doc.metadata:
                doc.metadata['source_type'] = 'internal'
        
        # Get the raw content
        raw_content = _join_docs(docs, max_chars=1200)
        
        # Check if result is too generic
        if len(raw_content) < 100 or _is_generic_content(raw_content):
            print("Enhanced Internal_VectorDB: Result too generic - falling back to Wikipedia")
            fallback_result = enhanced_wikipedia_search(query, patient_context)
            fallback_result['content'] = f"Limited relevant information in uploaded documents. Supplementing with general knowledge:\n\n{fallback_result['content']}"
            return fallback_result
        
        # Generate LLM summary
        summary = generate_medical_summary(raw_content, query, "Internal VectorDB", patient_context)
        
        # Format citations
        citations = format_citations_html(docs, 'internal')
        
        # Tool routing info
        tool_info = format_tool_routing_html(
            primary_tool="Internal_VectorDB",
            confidence="medium",
            tools_used=["Internal_VectorDB"],
            reasoning="Query references uploaded documents or user-specific content; Internal VectorDB selected for personalized information"
        )
        
        print(f"Enhanced Internal_VectorDB: Found {len(docs)} relevant chunks, generated summary")
        
        return {
            'content': raw_content,
            'summary': summary,
            'citations': citations,
            'tool_info': tool_info
        }
        
    except Exception as e:
        error_msg = f"Error searching internal knowledge base: {str(e)}"
        print(error_msg)
        # Fallback to Wikipedia on error
        fallback_result = enhanced_wikipedia_search(query, patient_context)
        fallback_result['content'] = f"{error_msg}\n\nFalling back to general knowledge:\n\n{fallback_result['content']}"
        return fallback_result

def format_enhanced_response(result: Dict[str, str]) -> str:
    """Format the enhanced response with only Medical Summary, Sources, and Tool Selection sections with page breaks"""
    
    response_parts = []
    
    # Section 1: Answer (renamed from Medical Summary)
    if result.get('summary'):
        response_parts.append(f'<div style="margin-bottom: 30px; page-break-after: always;"><h4 style="color: #007bff; margin-bottom: 15px; font-size: 18px;">Answer</h4><div style="background-color: #e3f2fd; padding: 15px; border-radius: 8px; line-height: 1.6; margin-bottom: 20px;">{result["summary"]}</div></div>')
    else:
        # Use content as summary if no LLM summary available
        if result.get('content'):
            content_preview = result['content'][:500] + "..." if len(result['content']) > 500 else result['content']
            response_parts.append(f'<div style="margin-bottom: 30px; page-break-after: always;"><h4 style="color: #007bff; margin-bottom: 15px; font-size: 18px;">� Medical Summary</h4><div style="background-color: #e3f2fd; padding: 15px; border-radius: 8px; line-height: 1.6; margin-bottom: 20px;">{content_preview}</div></div>')
    
    # Section 2: Source (renamed from Sources)
    if result.get('citations'):
        response_parts.append(f'<div style="margin-bottom: 30px; page-break-after: always;"><h4 style="color: #6f42c1; margin-bottom: 15px; font-size: 18px;">Source</h4><div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; line-height: 1.6; margin-bottom: 20px;">{result["citations"]}</div></div>')
    
    # Section 3: Tool Selection & Query Routing (header text unchanged emoji removed)
    if result.get('tool_info'):
        response_parts.append(f'<div style="margin-bottom: 20px;"><h4 style="color: #ff6600; margin-bottom: 15px; font-size: 18px;">Tool Selection & Query Routing</h4><div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; line-height: 1.6;">{result["tool_info"]}</div></div>')
    
    return "".join(response_parts)

# Test function
def test_enhanced_tools():
    """Test the enhanced tools"""
    
    print("🧪 Testing Enhanced Medical Tools")
    print("=" * 50)
    
    # Test Wikipedia search
    query = "symptoms of type 2 diabetes"
    print(f"\n📚 Testing enhanced Wikipedia search: '{query}'")
    
    result = enhanced_wikipedia_search(query)
    formatted_response = format_enhanced_response(result)
    
    print(f"✅ Enhanced Wikipedia result:")
    print(f"   Content length: {len(result.get('content', ''))}")
    print(f"   Summary length: {len(result.get('summary', ''))}")
    print(f"   Citations length: {len(result.get('citations', ''))}")
    print(f"   Tool info length: {len(result.get('tool_info', ''))}")
    
    print(f"\n📝 Formatted response preview:")
    print(formatted_response[:500] + "..." if len(formatted_response) > 500 else formatted_response)

if __name__ == "__main__":
    test_enhanced_tools()