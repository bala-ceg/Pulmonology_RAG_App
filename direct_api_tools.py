"""
Direct API Tools for LangChain Agents
====================================

This module provides direct API access to Wikipedia and ArXiv that works
seamlessly with LangChain agents, bypassing vector databases for simpler
and faster queries.
"""

import os
import wikipedia
import arxiv
from typing import Optional
from langchain.tools import tool

# Configure Wikipedia for better reliability
wikipedia.set_rate_limiting(True)


@tool
def wikipedia_search(query: str) -> str:
    """
    Search Wikipedia for medical information and definitions.
    
    Use this tool when users ask for:
    - Definitions or explanations of medical terms
    - General background information about diseases/conditions  
    - Well-established medical knowledge
    - Basic facts about treatments or procedures
    
    DO NOT use for:
    - Recent research or latest studies
    - Questions about uploaded documents
    - Cutting-edge experimental treatments
    
    Args:
        query: The medical term or topic to search for
        
    Returns:
        Formatted Wikipedia summary with source attribution
    """
    try:
        print(f"Wikipedia Search: Querying '{query}'")
        
        # Get Wikipedia summary with auto-suggestion for typos
        summary = wikipedia.summary(
            query, 
            sentences=3,  # Get 3 sentences for more context
            auto_suggest=True
        )
        
        # Get the page for additional metadata
        page = wikipedia.page(query, auto_suggest=True)
        
        # Format response with source attribution
        result = f"**According to Wikipedia:**\n\n{summary}\n\n"
        result += f"**Source:** Wikipedia - {page.title}\n"
        result += f"**URL:** {page.url}"
        
        print(f"Wikipedia Search: Successfully found information for '{query}'")
        return result
        
    except wikipedia.exceptions.DisambiguationError as e:
        # Handle multiple possible pages
        try:
            print(f"Wikipedia Search: Multiple pages found, trying '{e.options[0]}'")
            summary = wikipedia.summary(e.options[0], sentences=3)
            page = wikipedia.page(e.options[0])
            
            result = f"**According to Wikipedia (for '{e.options[0]}'):**\n\n{summary}\n\n"
            result += f"**Note:** Multiple pages found. Showing results for '{e.options[0]}'\n"
            result += f"**Other options:** {', '.join(e.options[1:3])}\n"  # Show first 2 alternatives
            result += f"**Source:** Wikipedia - {page.title}\n"
            result += f"**URL:** {page.url}"
            return result
            
        except Exception as inner_e:
            return f"Multiple Wikipedia pages found for '{query}'. Please be more specific. Options include: {', '.join(e.options[:5])}"
    
    except wikipedia.exceptions.PageError:
        return f"No Wikipedia page found for '{query}'. Please check the spelling or try a different term."
    
    except Exception as e:
        print(f"Wikipedia Search Error: {str(e)}")
        return f"Wikipedia search encountered an error: {str(e)}. Please try rephrasing your query."


@tool  
def arxiv_search(query: str) -> str:
    """
    Search ArXiv for recent medical research papers and studies.
    
    Use this tool when users ask for:
    - Latest research or recent studies
    - Cutting-edge medical developments
    - Scientific papers on specific topics
    - Research findings or experimental treatments
    - Questions containing words like 'research', 'study', 'paper', 'latest', 'recent'
    
    DO NOT use for:
    - Basic medical definitions
    - Well-established medical knowledge
    - Questions about uploaded documents
    - General background information
    
    Args:
        query: The research topic to search for
        
    Returns:
        Formatted summary of recent ArXiv papers with source attribution
    """
    try:
        print(f"ArXiv Search: Querying '{query}'")
        
        # Search ArXiv for recent papers
        search = arxiv.Search(
            query=query,
            max_results=5,  # Get top 5 results
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending
        )
        
        papers = []
        for i, paper in enumerate(search.results()):
            if i >= 3:  # Limit to top 3 papers
                break
                
            # Format paper information
            # Truncate summary to ~150 characters for readability
            summary = paper.summary.replace('\n', ' ').strip()
            if len(summary) > 150:
                summary = summary[:147] + "..."
            
            # Format authors (limit to first 3)
            authors = [author.name for author in paper.authors[:3]]
            if len(paper.authors) > 3:
                authors.append("et al.")
            author_str = ", ".join(authors)
            
            # Format publication date
            pub_date = paper.published.strftime("%Y-%m-%d")
            
            paper_info = f"**Title:** {paper.title}\n"
            paper_info += f"**Authors:** {author_str}\n"
            paper_info += f"**Published:** {pub_date}\n"
            paper_info += f"**Summary:** {summary}\n"
            paper_info += f"**ArXiv ID:** {paper.entry_id.split('/')[-1]}"
            
            papers.append(paper_info)
        
        if papers:
            result = f"**Based on recent ArXiv research:**\n\n"
            result += "\n\n---\n\n".join(papers)
            result += f"\n\n**Source:** ArXiv.org\n"
            result += f"**Search Query:** {query}\n"
            result += f"**Papers Found:** {len(papers)} of {len(list(search.results()))}"
            
            print(f"ArXiv Search: Found {len(papers)} relevant papers")
            return result
        else:
            return f"No recent ArXiv papers found for '{query}'. Try using more specific scientific terminology or check the spelling."
            
    except Exception as e:
        print(f"ArXiv Search Error: {str(e)}")
        return f"ArXiv search encountered an error: {str(e)}. Please try rephrasing your research query."


@tool
def internal_documents_search(query: str, session_id: Optional[str] = None) -> str:
    """
    Search uploaded documents and internal knowledge base.
    
    Use this tool when users ask about:
    - Content from uploaded PDFs or documents
    - Information "from my files" or "my documents"
    - Organizational protocols or guidelines
    - Questions referencing specific uploaded content
    
    DO NOT use for:
    - General medical knowledge
    - Recent research (use ArXiv instead)
    - Basic definitions (use Wikipedia instead)
    
    Args:
        query: The question about internal documents
        session_id: Optional session identifier for user-specific content
        
    Returns:
        Results from internal documents or fallback message
    """
    try:
        print(f"Internal Search: Querying '{query}' for session {session_id}")
        
        # Import here to avoid circular imports
        from tools import Internal_VectorDB
        from rag_architecture import TwoStoreRAGManager
        from langchain_openai import OpenAIEmbeddings
        
        # Initialize RAG manager (simplified)
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return "Internal document search is not available - OpenAI API key not configured."
        
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        rag_manager = TwoStoreRAGManager(
            embeddings=embeddings,
            llm=None,  # We don't need LLM for this
            base_vector_path="./vector_dbs"
        )
        
        # Use the existing Internal_VectorDB tool
        result = Internal_VectorDB(query, session_id, rag_manager)
        
        print(f"Internal Search: Retrieved {len(result)} characters")
        return result
        
    except Exception as e:
        print(f"Internal Search Error: {str(e)}")
        # Fallback to Wikipedia for better user experience
        return f"Could not access internal documents: {str(e)}\n\nFalling back to general medical knowledge...\n\n{wikipedia_search(query)}"


# Tool registry for easy access
DIRECT_API_TOOLS = [
    wikipedia_search,
    arxiv_search, 
    internal_documents_search
]


def get_tool_descriptions():
    """Get descriptions of all direct API tools."""
    return {
        'wikipedia_search': wikipedia_search.description,
        'arxiv_search': arxiv_search.description,
        'internal_documents_search': internal_documents_search.description
    }


def search_arxiv_safe(query: str) -> str:
    """
    Safe wrapper for ArXiv search that uses proper invocation method.
    This avoids deprecation warnings.
    """
    try:
        return arxiv_search.invoke({"query": query})
    except Exception as e:
        return f"Error searching ArXiv: {str(e)}"


def search_wikipedia_safe(query: str) -> str:
    """
    Safe wrapper for Wikipedia search that uses proper invocation method.
    This avoids deprecation warnings.
    """
    try:
        return wikipedia_search.invoke({"query": query})
    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"


def create_agent_tools(include_internal: bool = True):
    """
    Create a list of tools for LangChain agent initialization.
    
    Args:
        include_internal: Whether to include internal document search
        
    Returns:
        List of LangChain tools ready for agent use
    """
    tools = [wikipedia_search, arxiv_search]
    
    if include_internal:
        tools.append(internal_documents_search)
    
    return tools