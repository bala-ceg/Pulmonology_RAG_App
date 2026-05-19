"""
Integrated Medical RAG System with Tool Routing
==============================================

This module demonstrates the integration of the tool-based RAG architecture
with intelligent routing, guarded retrieval, and fallback mechanisms.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Any
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.agents import initialize_agent, AgentType
from langchain_core.documents import Document
try:
    from langchain_core.tools import tool
except ImportError:
    def tool(func):
        """Simple decorator for tool functions."""
        func.is_tool = True
        return func

# Import our custom modules
from config import Config
from rag_architecture import TwoStoreRAGManager, MedicalQueryRouter
from tools import Wikipedia_Search, ArXiv_Search, Tavily_Search, Internal_VectorDB, PostgreSQL_Diagnosis_Search, Pinecone_KB_Search, AdHocRAG_Search, _set_adhoc_context, AVAILABLE_TOOLS
from prompts import ROUTING_SYSTEM_PROMPT, get_routing_explanation
from utils.error_handlers import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Citation extraction helper
# ---------------------------------------------------------------------------

# Map tool names to human-readable source labels used when no inline citation
# is found in the response text.
_TOOL_SOURCE_LABELS: Dict[str, str] = {
    'Pinecone_KB_Search':          'PCES Pinecone Knowledge Base',
    'Internal_VectorDB':           'Uploaded Documents (Adhoc VectorDB)',
    'AdHocRAG_Search':             'Ad Hoc RAG (Doctor/Patient Documents)',
    'PostgreSQL_Diagnosis_Search': 'PostgreSQL EHR Database',
    'ArXiv_Search':                'arXiv Research Papers',
    'Tavily_Search':               'Web Search (Tavily)',
    'Wikipedia_Search':            'Wikipedia',
}


def _extract_citations_from_response(response: str, primary_tool: str) -> List[str]:
    """
    Extract citation strings from the final Sources: bullet list in a tool response.

    Handles two formats:
    1. Multi-line bullet list produced by Pinecone_KB_Search:
         Sources:
         - [PCES Pinecone KB] Dept: Cardiology | Source: file.pdf | Relevance: 82%
    2. Single-line comma-separated produced by _join_docs / other tools:
         Sources: Wikipedia, arXiv:2301.12345

    The Source Documents (verbatim excerpts) block that precedes Sources: is
    intentionally ignored — only the structured Sources: bullet list is parsed.
    Falls back to a labelled source so citations are never empty.
    """
    import re as _re

    citations: List[str] = []

    if response:
        # Remove the "Source Documents" excerpt block before parsing, so that
        # "Source: ..." lines inside the excerpt are not captured as citations.
        clean = _re.sub(
            r'---\s*\nSource Documents.*?(?=\nSources?:|\Z)',
            '',
            response,
            flags=_re.IGNORECASE | _re.DOTALL,
        )

        # Now find the final Sources: block
        sources_match = _re.search(
            r'Sources?:\s*(.+?)(?:\n\n|\Z)', clean, _re.IGNORECASE | _re.DOTALL
        )
        if sources_match:
            raw = sources_match.group(1).strip()
            for entry in _re.split(r'\n', raw):
                entry = _re.sub(r'^[\-\*\•]\s*', '', entry).strip()
                if entry:
                    citations.append(entry)

    # Always include a fallback label so citations are never empty
    label = _TOOL_SOURCE_LABELS.get(primary_tool, primary_tool)
    fallback = f"Source: {label}"
    if not any(label in c for c in citations):
        citations.append(fallback)

    return citations


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
        
        # Initialize OpenAI components — use Config so the same model/key/base_url
        # configured in .env is used here, matching llm_service.py everywhere else.
        self.embeddings = OpenAIEmbeddings(
            api_key=Config.OPENAI_API_KEY,
            base_url=Config.OPENAI_BASE_URL,
            model=Config.EMBEDDING_MODEL_NAME,
        )
        self.llm = ChatOpenAI(
            api_key=Config.OPENAI_API_KEY,
            base_url=Config.OPENAI_BASE_URL,
            model_name=Config.LLM_MODEL_NAME,
            temperature=Config.LLM_DEFAULT_TEMPERATURE,
            request_timeout=Config.LLM_REQUEST_TIMEOUT,
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
        
        # Create Wikipedia, ArXiv, Tavily, PostgreSQL, Pinecone KB, and AdHocRAG tools
        # (these don't need RAG manager injected via closure)
        tools.extend([Wikipedia_Search, ArXiv_Search, Tavily_Search, PostgreSQL_Diagnosis_Search, Pinecone_KB_Search])

        # Inject rag_manager into the AdHocRAG_Search tool via module-level holder
        from config import Config as _Config
        _set_adhoc_context(
            rag_manager=self.rag_manager,
            tenant_id=_Config.TENANT_ID,
            doctor_id="",      # overridden per-request via _set_adhoc_context in query route
            patient_id=None,
        )
        tools.append(AdHocRAG_Search)
        
        # Create Internal_VectorDB tool with RAG manager injected using @tool decorator
        # ZeroShotAgent requires single-input tools — only expose `query`
        @tool
        def internal_vectordb_with_context(query: str) -> str:
            """Search uploaded PDFs and URLs in the internal knowledge base for user-specific content."""
            return Internal_VectorDB(query, None, self.rag_manager)
        
        # Set tool metadata to match original
        internal_vectordb_with_context.name = Internal_VectorDB.name
        internal_vectordb_with_context.description = Internal_VectorDB.description
        
        tools.append(internal_vectordb_with_context)
        
        return tools
    
    def _initialize_agent(self):
        """Initialize the LangChain agent with tools and system prompt."""
        import warnings
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                agent = initialize_agent(
                    tools=self.tools,
                    llm=self.llm,
                    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    verbose=True,
                    max_iterations=2,
                    early_stopping_method="generate",
                )

            # Inject our routing system prompt
            agent.agent.llm_chain.prompt.template = (
                f"{ROUTING_SYSTEM_PROMPT}\n\n{agent.agent.llm_chain.prompt.template}"
            )
            return agent

        except Exception as e:
            logger.error(f"Error initializing agent: {e}")
            return None
    
    # -----------------------------------------------------------------------
    # Cascade order: Pinecone → AdHocRAG → Internal_VectorDB →
    #                ArXiv → Tavily → PostgreSQL → Wikipedia
    #
    # PostgreSQL is placed AFTER ArXiv/Tavily so general/research queries
    # never wait on the potentially-slow DB connection timeout.
    # -----------------------------------------------------------------------
    _CASCADE_ORDER: List[str] = [
        'Pinecone_KB_Search',
        'AdHocRAG_Search',
        'Internal_VectorDB',
        'ArXiv_Search',
        'Tavily_Search',
        'PostgreSQL_Diagnosis_Search',
        'Wikipedia_Search',
    ]

    # Regex patterns for secrets that must NEVER appear in any response or prompt
    import re as _re_module
    _SECRET_PATTERNS: tuple = (
        # Pinecone API keys  (pcsk_...)
        _re_module.compile(r'pcsk_[A-Za-z0-9_]{10,}', _re_module.IGNORECASE),
        # OpenAI keys  (sk-...)
        _re_module.compile(r'sk-[A-Za-z0-9_\-]{20,}', _re_module.IGNORECASE),
        # Generic API key assignment in env-file format  KEY="value" or KEY=value
        _re_module.compile(
            r'(?:PINECONE|OPENAI|TAVILY|AZURE|APIFY|HUGGINGFACE)_[A-Z_]+=[\"\']?[A-Za-z0-9_\-\.]{8,}[\"\']?',
            _re_module.IGNORECASE,
        ),
        # Any bearer / token pattern
        _re_module.compile(r'Bearer\s+[A-Za-z0-9_\-\.]{20,}', _re_module.IGNORECASE),
    )

    @classmethod
    def _sanitize_secrets(cls, text: str) -> str:
        """Strip API keys and secrets from any text before it reaches the LLM or user."""
        if not text:
            return text
        for pat in cls._SECRET_PATTERNS:
            text = pat.sub('[REDACTED]', text)
        return text

    # Responses that count as "no content found" — trigger next tool in cascade
    _EMPTY_SIGNALS: tuple = (
        # Generic empties
        "no relevant content found",
        "no results found",
        "not configured",
        "search error",
        "error searching",
        "http error",
        "please try using more specific",
        "i don't know",
        "i cannot",
        "no information",
        "could not find",
        "unable to find",
        # Pinecone
        "below threshold",
        "is below threshold",
        # PostgreSQL — errors & timeouts treated as empty so cascade continues
        "no diagnosis information found",
        "no medical diagnosis data",
        "not found in the database",
        "no data found",
        "error searching diagnosis database",
        "connection timeout",
        "technical error in accessing",
        "unavailable due to a technical error",
        # Internal VectorDB
        "no documents found for your session",
        "not available. please ensure",
        "limited relevant information in uploaded",
        # AdHoc RAG
        "no adhoc documents found",
        "no relevant content found in adhoc",
        "ad hoc rag store is not initialised",
        # Tool internal fallbacks — these mean the tool had no real answer
        "falling back to general knowledge",
        "falling back to general medical knowledge",
        "searching general knowledge instead",
        "supplementing with general knowledge",
        "no relevant information found in uploaded",
    )

    def _is_empty_result(self, text: str, tool_name: str = "") -> bool:
        """Return True when a tool response contains no useful content."""
        if not text or not text.strip():
            return True
        lc = text.lower()
        if any(sig in lc for sig in self._EMPTY_SIGNALS):
            return True
        # If a non-Wikipedia/non-Tavily tool silently fell back to Wikipedia
        # (its response contains wikipedia.org links), treat as empty so the
        # cascade routes properly to Wikipedia at the end.
        if tool_name not in ("Wikipedia_Search", "Tavily_Search", "ArXiv_Search"):
            if "wikipedia.org" in lc or "en.wikipedia.org" in lc:
                logger.info(
                    "Tool '%s' silently returned Wikipedia content — treating as empty for cascade",
                    tool_name,
                )
                return True
        return False

    def _call_tool(self, tool_name: str, question: str, session_id: str) -> str:
        """Execute a single tool by name and return its sanitized raw output."""
        tool_map = {t.name: t for t in self.tools}
        tool_fn = tool_map.get(tool_name)
        if tool_fn is None:
            return ""
        try:
            if tool_name == 'Internal_VectorDB':
                result = Internal_VectorDB(question, session_id, self.rag_manager)
            else:
                result = tool_fn.func(question)
            # Sanitize secrets BEFORE any further processing or LLM calls
            return self._sanitize_secrets(result or "")
        except Exception as exc:
            logger.warning("Tool %s raised %s", tool_name, exc)
            return ""

    def _format_answer(self, raw_content: str, question: str, tool_name: str) -> str:
        """
        Use the LLM to produce a focused Q&A answer from *raw_content*.
        Content may be merged from multiple sources (separated by ---).
        """
        prompt = (
            f"You are a medical Q&A assistant. The content below is retrieved from "
            f"one or more medical knowledge sources. Read ALL sections carefully and "
            f"synthesize a complete, accurate answer to the question.\n\n"
            f"Instructions:\n"
            f"- Combine relevant information from ALL sections provided\n"
            f"- If one section covers the topic better than another, prioritise it but "
            f"do not ignore the others\n"
            f"- Be concise and specific\n"
            f"- Only say the content does not address the question if NONE of the "
            f"sections contain relevant information\n\n"
            f"Question: {question}\n\n"
            f"Retrieved Content:\n{raw_content}\n\n"
            f"Answer:"
        )
        try:
            result = self.llm.invoke(prompt)
            answer = result.content if hasattr(result, "content") else str(result)
            return answer.strip()
        except Exception as exc:
            logger.warning("LLM formatting failed (%s) — returning raw content", exc)
            return raw_content

    # All tools the planner is allowed to select.
    # _FALLBACK_CASCADE is used when the planner fails or selects no tools.
    _ALL_SELECTABLE_TOOLS: List[str] = [
        'Pinecone_KB_Search',
        'Tavily_Search',
        'ArXiv_Search',
        'Wikipedia_Search',
        'AdHocRAG_Search',
        'Internal_VectorDB',
        'PostgreSQL_Diagnosis_Search',
    ]

    _FALLBACK_TOOL: str = 'Tavily_Search'

    _FALLBACK_CASCADE: List[str] = [
        'AdHocRAG_Search',
        'Internal_VectorDB',
        'ArXiv_Search',
        'PostgreSQL_Diagnosis_Search',
        'Wikipedia_Search',
        'Tavily_Search',
    ]

    def _run_parallel_tools(
        self,
        question: str,
        session_id: str,
        tool_names: List[str],
        timeout: float = 20.0,
    ) -> Dict[str, str]:
        """Run *tool_names* concurrently; return {tool_name: result} for non-empty results."""
        import concurrent.futures as _cf
        results: Dict[str, str] = {}
        if not tool_names:
            return results
        with _cf.ThreadPoolExecutor(max_workers=max(len(tool_names), 1)) as executor:
            futures = {
                executor.submit(self._call_tool, name, question, session_id): name
                for name in tool_names
            }
            for future in _cf.as_completed(futures, timeout=timeout):
                name = futures[future]
                try:
                    result = future.result()
                    if not self._is_empty_result(result, name):
                        results[name] = result
                        logger.info("Parallel tool '%s' returned %d chars", name, len(result))
                    else:
                        logger.info("Parallel tool '%s' returned nothing", name)
                except Exception as exc:
                    logger.warning("Parallel tool '%s' raised %s", name, exc)
        return results

    # Compact one-line descriptions shown to the planner LLM
    _PLANNER_TOOL_DESCRIPTIONS: Dict[str, str] = {
        'Pinecone_KB_Search':          'PCES org KB — dept clinical guidelines, protocols, standard-of-care (cardiology, neurology, pulmonology, etc.)',
        'Tavily_Search':               'Live web search — real-time news, current guidelines, general medical questions',
        'ArXiv_Search':                'Scientific papers — clinical trials, RCTs, recent research, evidence-based medicine',
        'Wikipedia_Search':            'Encyclopedic facts — definitions, anatomy, basic biology, general knowledge',
        'AdHocRAG_Search':             'Doctor/patient session docs — uploaded case notes, patient-specific clinical files',
        'Internal_VectorDB':           'Session-uploaded PDFs/URLs — user uploaded documents in current session',
        'PostgreSQL_Diagnosis_Search': 'EHR database — patient diagnosis records, ICD codes, clinical history by patient ID',
    }

    # Planner LLM call timeout (seconds)
    _PLANNER_TIMEOUT: float = 8.0

    def _plan_tools(self, question: str, adhoc_rag_ready: bool = False) -> List[str]:
        """
        Select ALL available tools to run in parallel.

        AdHocRAG_Search is included only when the frontend signals that the
        doctor has successfully built an Ad Hoc KB (adhoc_rag_ready=True).
        All other tools always run so results are merged from every source.
        """
        tools = [t for t in self._ALL_SELECTABLE_TOOLS]

        if not adhoc_rag_ready and 'AdHocRAG_Search' in tools:
            tools.remove('AdHocRAG_Search')
            logger.info("_plan_tools: adhoc_rag_ready=False — skipping AdHocRAG_Search")

        logger.info("_plan_tools: running all tools in parallel: %s", tools)
        return tools

    def query(self, question: str, session_id: str = None, patient_context: str = None,
              adhoc_rag_ready: bool = False) -> Dict[str, Any]:
        """
        LLM-driven multi-tool parallel query.

        Strategy:
          Phase 0 (plan)     — LLM selects relevant tools for this query (~0.3s)
          Phase 1 (parallel) — Selected tools run simultaneously via ThreadPoolExecutor
          Phase 2 (fallback) — If all planned tools return empty, try Tavily → Wikipedia
        """
        try:
            tools_used: List[str] = []
            raw_content: str = ""

            # ── Phase 0: LLM tool planner ────────────────────────────────────
            planned_tools = self._plan_tools(question, adhoc_rag_ready=adhoc_rag_ready)
            logger.info("Planned tools for query: %s", planned_tools)

            # ── Phase 1: run planned tools in parallel ───────────────────────
            parallel_results = self._run_parallel_tools(
                question, session_id or "guest", planned_tools
            )

            if parallel_results:
                # Merge results in planned order so Pinecone always leads
                parts = []
                for name in planned_tools:
                    if name in parallel_results:
                        label = _TOOL_SOURCE_LABELS.get(name, name)
                        parts.append(f"[{label}]\n{parallel_results[name]}")
                        tools_used.append(name)
                raw_content = "\n\n---\n\n".join(parts)
                logger.info("Multi-tool: merged %d/%d planned tools (%d chars)",
                            len(parallel_results), len(planned_tools), len(raw_content))

            # ── Phase 2: fallback when ALL planned tools returned empty ──────
            if not raw_content:
                logger.info("All planned tools empty — running fallback cascade")
                for tool_name in self._FALLBACK_CASCADE:
                    if tool_name in planned_tools:
                        continue   # already tried
                    logger.info("Fallback: trying '%s'", tool_name)
                    result = self._call_tool(tool_name, question, session_id or "guest")
                    if not self._is_empty_result(result, tool_name):
                        raw_content = result
                        tools_used = [tool_name]
                        logger.info("Fallback: '%s' returned %d chars", tool_name, len(result))
                        break
                    logger.info("Fallback: '%s' returned nothing — next", tool_name)


            if not raw_content:
                raw_content = "No relevant information found across all knowledge sources."

            # Sanitize raw_content before sending to LLM — prevent secrets from
            # leaking even if _call_tool result was not fully sanitized
            raw_content = self._sanitize_secrets(raw_content)

            # Extract Source Documents block BEFORE sending raw_content to LLM
            # so it is kept as structured data and NOT embedded in the answer text.
            import re as _re
            source_documents: list = []  # list of dicts, one per retrieved doc
            sd_block_match = _re.search(
                r'---\s*\nSource Documents.*?(?=\nSources?:|\Z)',
                raw_content,
                _re.IGNORECASE | _re.DOTALL,
            )
            if sd_block_match:
                sd_block_text = sd_block_match.group(0)
                # Parse each [Source Document N] entry
                for doc_match in _re.finditer(
                    r'\[Source Document \d+\]\s*\n'
                    r'([^\n]+)\n'          # header line: "Doc N | Dept | Source | Relevance"
                    r'Excerpt:\s*\n'
                    r'\s*"([^"]+)"',       # verbatim text in quotes
                    sd_block_text,
                    _re.IGNORECASE | _re.DOTALL,
                ):
                    header = doc_match.group(1).strip()
                    excerpt = doc_match.group(2).strip()
                    # Parse header fields (pipe-separated)
                    fields = {
                        p.split(':', 1)[0].strip(): p.split(':', 1)[1].strip()
                        for p in header.split('  |  ')
                        if ':' in p
                    }
                    source_documents.append({
                        'header': header,
                        'excerpt': excerpt,
                        'fields': fields,
                    })

            # Format a focused Q&A answer from the raw tool content.
            # The LLM only sees the medical text, NOT the Source Documents block.
            content_for_llm = _re.sub(
                r'---\s*\nSource Documents.*',
                '',
                raw_content,
                flags=_re.IGNORECASE | _re.DOTALL,
            ).strip()
            answer = self._format_answer(content_for_llm, question, ", ".join(tools_used) if tools_used else "unknown")

            # The answer must NOT contain the Source Documents block — strip it
            # if the LLM happened to reproduce it.
            answer = _re.sub(
                r'---\s*\nSource Documents.*',
                '',
                answer,
                flags=_re.IGNORECASE | _re.DOTALL,
            ).strip()

            primary_tool = tools_used[0] if tools_used else "Wikipedia_Search"
            citations = _extract_citations_from_response(raw_content, primary_tool)
            # Final sanitization pass — ensure no secrets reach the user
            answer = self._sanitize_secrets(answer)
            citations = [self._sanitize_secrets(c) for c in citations]

            return {
                'answer': answer,
                'routing_info': {
                    'primary_tool': primary_tool,
                    'confidence': 'high',
                    'reasoning': f'LLM planned: {", ".join(planned_tools)} → executed: {", ".join(tools_used)}',
                    'planned_tools': planned_tools,
                    'ranked_tools': self._ALL_SELECTABLE_TOOLS,
                },
                'explanation': f"Answer sourced from: {', '.join(tools_used)}",
                'tools_used': tools_used,
                'session_id': session_id,
                'citations': citations,
                'source_documents': source_documents,
            }

        except Exception as e:
            logger.error("Error in cascade query: %s", e)
            return {
                'answer': f"I encountered an error processing your question: {str(e)}.",
                'routing_info': {'error': str(e)},
                'explanation': "Error occurred during processing",
                'tools_used': ['Error'],
                'session_id': session_id,
                'citations': [],
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
        logger.info("Please set OPENAI_API_KEY environment variable")
        return
    
    # Initialize system
    rag_system = IntegratedMedicalRAG(api_key)
    
    # Check system status
    status = rag_system.get_system_status()
    logger.info("System Status:")
    logger.info(f"  Local documents: {status['local_document_count']}")
    logger.info(f"  External documents: {status['external_document_count']}")
    logger.info(f"  Available tools: {status['available_tools']}")
    
    # Example queries demonstrating different routing scenarios
    test_queries = [
        "What is hypertension?",  # Should route to Wikipedia
        "Latest research on COVID-19 treatments",  # Should route to ArXiv
        "Recent preprints on pulmonary fibrosis mechanisms and pathogenesis",  # Likely route to ArXiv
        "What does my uploaded PDF say about treatment protocols?",  # Should route to Internal (with fallback)
    ]
    
    for query in test_queries:
        logger.info(f"\n{'='*60}")
        logger.info(f"Query: {query}")
        logger.info('='*60)
        
        result = rag_system.query(query)
        
        logger.info(f"Primary Tool: {result['routing_info']['primary_tool']}")
        logger.info(f"Confidence: {result['routing_info']['confidence']}")
        logger.info(f"Reasoning: {result['routing_info']['reasoning']}")
        logger.info(f"Answer: {result['answer'][:200]}...")  # Truncate for demo


if __name__ == "__main__":
    demo_integration()