"""
RAG service — facade over TwoStoreRAGManager, IntegratedMedicalRAG, and
DomainScopeGuard.

Provides:
- ``RAGService``   — lazy-initialised facade class.
- ``rag_service``  — module-level singleton; import and use directly.

Usage::

    from services.rag_service import rag_service
    from services.llm_service import llm_service

    rag_service.initialize(
        embeddings=llm_service.get_embeddings(),
        llm=llm_service.get_llm(),
    )

    if rag_service.is_rag_available:
        result = rag_service.rag_manager.query(...)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from config import Config
from utils.error_handlers import get_logger

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Optional-import availability flags (mirrors main.py pattern)
# ---------------------------------------------------------------------------
try:
    from rag_architecture import TwoStoreRAGManager  # type: ignore[import]
    _RAG_ARCHITECTURE_AVAILABLE = True
except ImportError:
    _RAG_ARCHITECTURE_AVAILABLE = False
    logger.warning("RAG architecture not available — install scikit-learn")

try:
    from integrated_rag import IntegratedMedicalRAG  # type: ignore[import]
    _INTEGRATED_RAG_AVAILABLE = True
except ImportError:
    _INTEGRATED_RAG_AVAILABLE = False
    logger.warning("Integrated RAG system not available")

try:
    from domain_scope_guard import DomainScopeGuard  # type: ignore[import]
    _SCOPE_GUARD_AVAILABLE = True
except ImportError:
    _SCOPE_GUARD_AVAILABLE = False


# ---------------------------------------------------------------------------
# Service class
# ---------------------------------------------------------------------------

class RAGService:
    """Lazy facade over :class:`TwoStoreRAGManager`, :class:`IntegratedMedicalRAG`,
    and :class:`DomainScopeGuard`.

    The constructor does **not** create any heavyweight objects.  Call
    :meth:`initialize` once (e.g. during Flask app startup) to hydrate the
    service.
    """

    def __init__(self) -> None:
        """Create the service shell without initialising any RAG components."""
        self._rag_manager: Any = None
        self._integrated_rag_system: Any = None
        self._scope_guard: Any = None
        self._initialised: bool = False

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize(
        self,
        embeddings: OpenAIEmbeddings,
        llm: ChatOpenAI,
    ) -> None:
        """Initialise all RAG components using *embeddings* and *llm*.

        This method replicates the startup logic from ``main.py`` (lines
        526–605).  It is idempotent — calling it a second time is a no-op.

        Args:
            embeddings: An :class:`~langchain_openai.OpenAIEmbeddings` instance
                        used by the vector stores.
            llm:        A :class:`~langchain_openai.ChatOpenAI` instance used
                        by the RAG chains.
        """
        if self._initialised:
            logger.warning("RAGService.initialize() called more than once — ignoring")
            return

        vector_db_path: str = Config.VECTOR_DB_PATH

        # ---- Two-Store RAG Manager ----------------------------------------
        if _RAG_ARCHITECTURE_AVAILABLE:
            try:
                self._rag_manager = TwoStoreRAGManager(embeddings, llm, vector_db_path)
                logger.info("Two-Store RAG Manager initialised successfully")

                # Populate external KB on first run if it is empty
                kb_external_has_content = False
                if self._rag_manager.kb_external:
                    try:
                        count = self._rag_manager.kb_external._collection.count()
                        kb_external_has_content = count > 0
                        logger.info("External KB already contains %d documents", count)
                    except Exception:
                        kb_external_has_content = False

                if not kb_external_has_content:
                    logger.info("External KB is empty — loading initial content")

                    medical_topics = [
                        "pulmonology",
                        "cardiology",
                        "neurology",
                        "family medicine",
                        "medical diagnosis",
                        "clinical medicine",
                        "pharmacology",
                    ]
                    logger.info("Loading %d Wikipedia topics", len(medical_topics))
                    self._rag_manager.load_wikipedia_content(
                        medical_topics, max_docs_per_topic=2
                    )

                    arxiv_queries = [
                        "medical diagnosis AI",
                        "clinical decision support",
                        "medical imaging analysis",
                        "healthcare machine learning",
                    ]
                    logger.info("Loading %d arXiv queries", len(arxiv_queries))
                    self._rag_manager.load_arxiv_content(
                        arxiv_queries, max_docs_per_query=1
                    )

                    logger.info("External KB initial content loaded successfully")
                else:
                    logger.info("External KB already populated — skipping content loading")

            except Exception as exc:
                logger.error("Failed to initialise RAG Manager: %s", exc)
                self._rag_manager = None
        else:
            logger.warning("RAG Architecture not available — using legacy mode")

        # ---- Integrated Medical RAG System --------------------------------
        if _INTEGRATED_RAG_AVAILABLE:
            try:
                api_key = Config.OPENAI_API_KEY or os.getenv("openai_api_key")
                if api_key:
                    self._integrated_rag_system = IntegratedMedicalRAG(
                        openai_api_key=api_key,
                        base_vector_path=vector_db_path,
                    )
                    logger.info("Integrated Medical RAG System initialised successfully")
                else:
                    logger.warning(
                        "OpenAI API key not found — Integrated RAG system disabled"
                    )
            except Exception as exc:
                logger.error("Failed to initialise Integrated RAG System: %s", exc)
                self._integrated_rag_system = None
        else:
            logger.warning("Integrated RAG System not available")

        # ---- Domain Scope Guard -------------------------------------------
        if _SCOPE_GUARD_AVAILABLE:
            try:
                self._scope_guard = DomainScopeGuard(db_config=Config.db_kwargs())
                # Inject the instance back into the module so existing code that
                # references ``domain_scope_guard.scope_guard`` continues to work.
                import domain_scope_guard as _dsg_module  # type: ignore[import]
                _dsg_module.scope_guard = self._scope_guard
                logger.info("Domain Scope Guard initialised")
            except Exception as exc:
                logger.warning(
                    "Domain Scope Guard failed to initialise: %s — "
                    "pass-through mode active",
                    exc,
                )
                self._scope_guard = None
        else:
            logger.warning("Domain Scope Guard not available")

        self._initialised = True

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def rag_manager(self) -> Any:
        """The :class:`TwoStoreRAGManager` instance, or ``None``."""
        return self._rag_manager

    @property
    def integrated_rag_system(self) -> Any:
        """The :class:`IntegratedMedicalRAG` instance, or ``None``."""
        return self._integrated_rag_system

    @property
    def scope_guard(self) -> Any:
        """The :class:`DomainScopeGuard` instance, or ``None``."""
        return self._scope_guard

    @property
    def is_rag_available(self) -> bool:
        """``True`` when the Two-Store RAG Manager was initialised successfully."""
        return self._rag_manager is not None

    @property
    def is_integrated_available(self) -> bool:
        """``True`` when the Integrated RAG system was initialised successfully."""
        return self._integrated_rag_system is not None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
rag_service = RAGService()

__all__ = ["RAGService", "rag_service"]
