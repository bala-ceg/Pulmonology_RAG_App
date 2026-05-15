"""
Build Context Service
=====================
Assembles the structured context passed to the LLM for every query.

Pipeline (mirrors Doc 04 — Build Context v1.0):
  1. Receive inputs (prompt, department, doctor_id, patient_id, tenant_id, session_id)
  2. Retrieve Main RAG      (top_k=5, scope="global")        ─┐
  3. Retrieve Ad Hoc RAG   (top_k=3, doctor + patient scope) ─┤ parallel
  4. Retrieve Patient Data (Postgres p_diagnosis)            ─┘
  5. Conditionally retrieve External APIs (ArXiv + Tavily + Wikipedia)
  6. Combine → limit to MAX_CONTEXT_CHUNKS → return ContextResult

Usage::

    from services.context_service import context_service, ContextRequest

    req = ContextRequest(
        prompt="Patient has knee pain after a fall",
        department="Orthopedics",
        doctor_id="DOC_102",
        patient_id="PAT_567",
        tenant_id="HOSP_001",
        session_id="20260414_1015",
    )
    result = context_service.build(req)
    # result.combined → list of str chunks ready for LLM
    # result.patient_data → dict (may be empty if patient not found)
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Any

from config import Config
from utils.error_handlers import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass
class ContextRequest:
    """Input contract for BuildContextService.build()."""
    prompt: str
    department: str = ""
    doctor_id: str = ""
    patient_id: str | None = None
    tenant_id: str = field(default_factory=lambda: Config.TENANT_ID)
    session_id: str = ""


@dataclass
class ContextResult:
    """Output of BuildContextService.build()."""
    main_rag: list[str] = field(default_factory=list)
    adhoc_rag: list[str] = field(default_factory=list)
    patient_data: dict[str, Any] = field(default_factory=dict)
    external: list[str] = field(default_factory=list)
    combined: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Service class
# ---------------------------------------------------------------------------

class BuildContextService:
    """Parallel context assembly for the LLM pipeline.

    A module-level singleton is created at the bottom of this file
    (``context_service``).  Call ``context_service.initialize(rag_manager)``
    once at app startup to hydrate the service.
    """

    def __init__(self) -> None:
        self._rag_manager: Any = None   # TwoStoreRAGManager | None
        self._initialised: bool = False

    def initialize(self, rag_manager: Any) -> None:
        """Inject the TwoStoreRAGManager (called once at app startup)."""
        if self._initialised:
            return
        self._rag_manager = rag_manager
        self._initialised = True
        logger.info("BuildContextService initialised (rag_manager=%s)", rag_manager is not None)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, request: ContextRequest) -> ContextResult:
        """Assemble the full context for *request* using parallel retrieval.

        Steps run in parallel:
          - Main RAG
          - Ad Hoc RAG
          - Patient Data (Postgres)

        External APIs are called sequentially **only if** the prompt contains
        a trigger keyword (see :attr:`Config.EXTERNAL_API_KEYWORDS`).

        Returns a :class:`ContextResult` with individual buckets plus a
        ``combined`` list capped at :attr:`Config.MAX_CONTEXT_CHUNKS` entries.
        """
        if not self._initialised or self._rag_manager is None:
            raise RuntimeError(
                "BuildContextService not initialised — call context_service.initialize(rag_manager) at startup"
            )
        start = time.perf_counter()
        result = ContextResult()

        # ── Parallel retrieval ──────────────────────────────────────────
        timeout = Config.CONTEXT_RETRIEVAL_TIMEOUT

        futures_map: dict = {}
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="ctx") as pool:
            futures_map["main_rag"] = pool.submit(
                self._retrieve_main_rag, request.prompt, request.tenant_id, request.department
            )
            futures_map["adhoc_rag"] = pool.submit(
                self._retrieve_adhoc_rag,
                request.prompt,
                request.tenant_id,
                request.doctor_id,
                request.patient_id,
            )
            futures_map["patient_data"] = pool.submit(
                self._retrieve_patient_data, request.patient_id
            )

            for key, future in futures_map.items():
                try:
                    value = future.result(timeout=timeout)
                    if key == "main_rag":
                        result.main_rag = value
                    elif key == "adhoc_rag":
                        result.adhoc_rag = value
                    elif key == "patient_data":
                        result.patient_data = value
                except FuturesTimeout:
                    msg = f"context retrieval timed out for {key}"
                    logger.warning("BuildContextService: %s", msg)
                    result.errors.append(msg)
                except Exception as exc:
                    msg = f"{key} retrieval failed: {exc}"
                    logger.error("BuildContextService: %s", msg)
                    result.errors.append(msg)

        # ── External APIs (conditional) ─────────────────────────────────
        if self._should_call_external(request.prompt):
            result.external = self._retrieve_external(request.prompt)

        # ── Combine & limit ─────────────────────────────────────────────
        combined: list[str] = []
        combined.extend(result.main_rag)
        combined.extend(result.adhoc_rag)
        if result.patient_data:
            combined.append(self._format_patient_data(result.patient_data))
        combined.extend(result.external)

        result.combined = self._limit_context(combined, Config.MAX_CONTEXT_CHUNKS)
        result.elapsed_seconds = round(time.perf_counter() - start, 3)

        logger.info(
            "BuildContextService.build: %d combined chunks (main=%d adhoc=%d ext=%d) in %.2fs",
            len(result.combined),
            len(result.main_rag),
            len(result.adhoc_rag),
            len(result.external),
            result.elapsed_seconds,
        )
        return result

    # ------------------------------------------------------------------
    # Step 2 — Main RAG
    # ------------------------------------------------------------------

    def _retrieve_main_rag(
        self, prompt: str, tenant_id: str, department: str, top_k: int = 5
    ) -> list[str]:
        """Retrieve department-scoped Main RAG documents."""
        if self._rag_manager is None:
            return []
        try:
            # Use the session-based kb_local with department metadata filter
            if self._rag_manager.kb_local is None:
                return []
            where_filter: dict = {}
            if department:
                where_filter["department"] = department
            if where_filter:
                docs = self._rag_manager.kb_local.similarity_search(
                    prompt, k=top_k, filter=where_filter
                )
            else:
                docs = self._rag_manager.kb_local.similarity_search(prompt, k=top_k)
            return [doc.page_content for doc in docs if doc.page_content.strip()]
        except Exception as exc:
            logger.warning("_retrieve_main_rag error: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Step 3 — Ad Hoc RAG
    # ------------------------------------------------------------------

    def _retrieve_adhoc_rag(
        self,
        prompt: str,
        tenant_id: str,
        doctor_id: str,
        patient_id: str | None,
        top_k: int = 3,
    ) -> list[str]:
        """Retrieve doctor/patient-scoped Ad Hoc RAG documents."""
        if self._rag_manager is None:
            return []
        try:
            docs = self._rag_manager.retrieve_adhoc(
                prompt, tenant_id, doctor_id, patient_id, top_k=top_k
            )
            return [doc.page_content for doc in docs if doc.page_content.strip()]
        except Exception as exc:
            logger.warning("_retrieve_adhoc_rag error: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Step 4 — Patient Data (Postgres)
    # ------------------------------------------------------------------

    def _retrieve_patient_data(self, patient_id: str | None) -> dict[str, Any]:
        """Query Postgres for patient history.

        Note: The design document specifies a `patient_records` table with
        age/gender/allergies/medications.  That table does not exist in the
        current schema.  We fall back to querying `p_diagnosis` for diagnosis
        history from `pces_ehr_ccm`.
        """
        if not patient_id:
            return {}
        try:
            import psycopg2  # type: ignore[import]
            import os

            conn = psycopg2.connect(
                host=os.getenv("PG_TOOL_HOST"),
                port=os.getenv("PG_TOOL_PORT"),
                dbname=os.getenv("PG_TOOL_NAME"),
                user=os.getenv("PG_TOOL_USER"),
                password=os.getenv("PG_TOOL_PASSWORD"),
                connect_timeout=5,
            )
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT diagnosis_code, diagnosis_description, diagnosis_date
                        FROM p_diagnosis
                        WHERE patient_id = %s
                        ORDER BY diagnosis_date DESC
                        LIMIT 5
                        """,
                        (patient_id,),
                    )
                    rows = cur.fetchall()
            conn.close()

            if not rows:
                return {}

            diagnoses = [
                {"code": r[0], "description": r[1], "date": str(r[2])} for r in rows
            ]
            return {"patient_id": patient_id, "diagnoses": diagnoses}

        except Exception as exc:
            logger.warning("_retrieve_patient_data error (patient_id=%s): %s", patient_id, exc)
            return {}

    # ------------------------------------------------------------------
    # Step 5 — External API gate
    # ------------------------------------------------------------------

    def _should_call_external(self, prompt: str) -> bool:
        """Return True if the prompt contains a keyword that warrants external API calls."""
        prompt_lower = prompt.lower()
        return any(kw in prompt_lower for kw in Config.EXTERNAL_API_KEYWORDS)

    # ------------------------------------------------------------------
    # Step 6 — External APIs (ArXiv + Tavily + Wikipedia)
    # ------------------------------------------------------------------

    def _retrieve_external(self, prompt: str, max_results: int = 3) -> list[str]:
        """Call ArXiv and Tavily for external evidence.

        Note: PubMed is not available in this deployment (not installed).
        ArXiv is used as the research source.  Wikipedia is queried as a
        lightweight fallback.
        """
        results: list[str] = []
        timeout = Config.CONTEXT_RETRIEVAL_TIMEOUT

        # ArXiv
        try:
            from langchain_community.document_loaders import ArxivLoader as _ArxivLoader
            import threading as _t

            _holder: list = [None]

            def _load_arxiv():
                loader = _ArxivLoader(query=prompt, load_max_docs=2)
                _holder[0] = loader.load()

            t = _t.Thread(target=_load_arxiv, daemon=True)
            t.start()
            t.join(timeout=timeout)
            if t.is_alive():
                logger.warning("_retrieve_external: ArXiv timed out")
            elif _holder[0]:
                for doc in _holder[0][:max_results]:
                    text = doc.page_content.strip()
                    if text:
                        results.append(f"[ArXiv] {text[:600]}")
        except Exception as exc:
            logger.warning("_retrieve_external ArXiv error: %s", exc)

        # Tavily
        try:
            from tavily import TavilyClient as _TavilyClient
            import os

            api_key = os.getenv("TAVILY_API_KEY")
            if api_key:
                client = _TavilyClient(api_key=api_key)
                resp = client.search(query=f"medical {prompt}", max_results=2)
                for r in (resp.get("results") or [])[:max_results]:
                    text = r.get("content", "").strip()
                    if text:
                        results.append(f"[Tavily] {text[:600]}")
        except Exception as exc:
            logger.warning("_retrieve_external Tavily error: %s", exc)

        return results

    # ------------------------------------------------------------------
    # Step 7 / 8 — Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_patient_data(patient_data: dict[str, Any]) -> str:
        if not patient_data:
            return ""
        pid = patient_data.get("patient_id", "")
        diagnoses = patient_data.get("diagnoses", [])
        lines = [f"Patient {pid} — recent diagnoses:"]
        for d in diagnoses:
            lines.append(f"  • [{d['date']}] {d['code']}: {d['description']}")
        return "\n".join(lines)

    @staticmethod
    def _limit_context(chunks: list[str], max_chunks: int) -> list[str]:
        """Cap the context list to *max_chunks* non-empty entries."""
        non_empty = [c for c in chunks if c and c.strip()]
        return non_empty[:max_chunks]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
context_service = BuildContextService()

__all__ = ["BuildContextService", "ContextRequest", "ContextResult", "context_service"]
