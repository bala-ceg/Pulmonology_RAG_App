"""
domain_scope_guard.py
---------------------
RLHF Domain Scope Guard — three-tier query filtering based on semantic
similarity to the RLHF training data stored in `rlhf_interactions`.

Tiers
-----
1. accepted       score >= SCOPE_GUARD_THRESHOLD (0.45)
                  → answer normally
2. general_medical MEDICAL_FALLBACK_THRESHOLD <= score < SCOPE_GUARD_THRESHOLD (0.20–0.45)
                  → answer with a brief disclaimer, using a built-in general
                    medical corpus as the reference for this tier
3. rejected       score < MEDICAL_FALLBACK_THRESHOLD (< 0.20)
                  → politely decline

Configuration (env vars)
------------------------
  SCOPE_GUARD_THRESHOLD      In-scope cutoff (default 0.45)
  MEDICAL_FALLBACK_THRESHOLD General-medical cutoff (default 0.20)
  SCOPE_GUARD_ENABLED        Set "false" to disable (default "true")
  EMB_MODEL                  SBERT model name (default all-MiniLM-L6-v2)

Graceful degradation
--------------------
  - Empty / unreachable rlhf_interactions → all queries pass through.
  - sentence-transformers not installed    → all queries pass through.
"""

import os
import re
import logging
from datetime import datetime
from typing import Optional

import numpy as np
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────
SCOPE_GUARD_THRESHOLD      = float(os.getenv("SCOPE_GUARD_THRESHOLD", "0.45"))
MEDICAL_FALLBACK_THRESHOLD = float(os.getenv("MEDICAL_FALLBACK_THRESHOLD", "0.20"))
SCOPE_GUARD_ENABLED        = os.getenv("SCOPE_GUARD_ENABLED", "true").lower() != "false"
EMB_MODEL                  = os.getenv("EMB_MODEL", "all-MiniLM-L6-v2")

# Built-in general medical reference phrases used for tier-2 detection
_GENERAL_MEDICAL_PHRASES = [
    "symptoms of a medical condition",
    "treatment options for a disease or illness",
    "patient diagnosis and clinical findings",
    "medical terminology and healthcare concepts",
    "anatomy and physiology of the human body",
    "pharmacology and drug therapy",
    "surgical procedure and medical intervention",
    "public health and disease prevention",
    "laboratory tests and diagnostic results",
    "mental health and psychiatric conditions",
    "pediatric medicine and child health",
    "emergency medicine and acute care",
    "chronic disease management and monitoring",
    "nutrition and dietary health guidelines",
    "reproductive health and obstetrics",
    "infectious disease and microbiology",
    "cardiovascular health and heart disease",
    "respiratory conditions and lung function",
    "oncology and cancer treatment",
    "neurology and brain disorders",
    "endocrinology and hormonal disorders",
    "musculoskeletal disorders and orthopedics",
    "dermatology and skin conditions",
    "ophthalmology and eye health",
    "clinical guidelines and medical protocols",
]

# ── SBERT import ─────────────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    _SBERT_AVAILABLE = True
except ImportError:
    _SBERT_AVAILABLE = False
    logger.warning("sentence-transformers not installed — DomainScopeGuard disabled")


def _cosine_similarity_matrix(query_vec: np.ndarray, corpus_matrix: np.ndarray) -> np.ndarray:
    """Return cosine similarity of query_vec against every row of corpus_matrix."""
    q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    c_norms = corpus_matrix / (np.linalg.norm(corpus_matrix, axis=1, keepdims=True) + 1e-10)
    return c_norms @ q_norm


def _extract_topic_phrases(prompts: list, max_topics: int = 15) -> list:
    strip_patterns = re.compile(
        r"^(what (are|is|were|was|the)|how (do|does|can|should|to)|when (should|is|are|do)|"
        r"why (is|are|does|do)|explain|describe|tell me about|what.*recommend|"
        r"can you|please|list|define|provide)\s+",
        re.IGNORECASE,
    )
    seen: set = set()
    topics: list = []
    for prompt in prompts:
        phrase = strip_patterns.sub("", prompt.strip()).strip("?.,! ")
        short = " ".join(phrase.split()[:8]).strip("?.,!")
        key = short.lower()
        if short and key not in seen:
            seen.add(key)
            topics.append(short[0].upper() + short[1:])
        if len(topics) >= max_topics:
            break
    return topics


# Status constants
STATUS_ACCEPTED       = "accepted"
STATUS_GENERAL_MEDICAL = "general_medical"
STATUS_REJECTED       = "rejected"

GENERAL_MEDICAL_DISCLAIMER = (
    "Even though your question falls outside the scope of topics I have been trained on, "
    "I will answer this generally.\n\n"
)


class DomainScopeGuard:
    """
    Three-tier semantic similarity gate for incoming queries.

    check(query) returns (status, score, message):
      - ("accepted",        score, "")           → proceed normally
      - ("general_medical", score, disclaimer)   → proceed, prepend disclaimer to answer
      - ("rejected",        score, decline_msg)  → return decline to user
    """

    def __init__(self, db_config: Optional[dict] = None, threshold: Optional[float] = None):
        self.threshold          = threshold if threshold is not None else SCOPE_GUARD_THRESHOLD
        self.medical_threshold  = MEDICAL_FALLBACK_THRESHOLD
        self.db_config          = db_config or {}
        self.enabled            = SCOPE_GUARD_ENABLED

        self._embedder                           = None
        self._corpus_matrix: Optional[np.ndarray] = None   # training prompts
        self._medical_matrix: Optional[np.ndarray] = None  # general medical phrases
        self._training_prompts: list             = []
        self._topic_summary: list                = []
        self._corpus_size: int                   = 0
        self._last_refresh: Optional[datetime]   = None

        if not self.enabled:
            logger.info("DomainScopeGuard: disabled via SCOPE_GUARD_ENABLED=false")
            return

        if not _SBERT_AVAILABLE:
            logger.warning("DomainScopeGuard: sentence-transformers unavailable — pass-through mode")
            return

        try:
            logger.info(f"DomainScopeGuard: loading SBERT model '{EMB_MODEL}'…")
            self._embedder = SentenceTransformer(EMB_MODEL)
            # Pre-embed the general medical reference corpus (done once at startup)
            self._medical_matrix = self._embedder.encode(
                _GENERAL_MEDICAL_PHRASES, convert_to_numpy=True, show_progress_bar=False
            ).astype(np.float32)
        except Exception as exc:
            logger.warning(f"DomainScopeGuard: failed to load SBERT model: {exc}")
            self._embedder = None

        self.load_corpus()

    # ── Corpus loading ────────────────────────────────────────────────────────

    def load_corpus(self) -> int:
        if not self._embedder:
            return 0

        prompts = self._fetch_training_prompts()
        if not prompts:
            logger.warning("DomainScopeGuard: no training prompts found — pass-through mode")
            self._corpus_matrix = None
            self._training_prompts = []
            self._topic_summary = []
            self._corpus_size = 0
            self._last_refresh = datetime.utcnow()
            return 0

        logger.info(f"DomainScopeGuard: embedding {len(prompts)} training prompts…")
        try:
            matrix = self._embedder.encode(prompts, convert_to_numpy=True, show_progress_bar=False)
            self._corpus_matrix    = matrix.astype(np.float32)
            self._training_prompts = prompts
            self._topic_summary    = _extract_topic_phrases(prompts)
            self._corpus_size      = len(prompts)
            self._last_refresh     = datetime.utcnow()
            logger.info(
                f"DomainScopeGuard: corpus ready — {self._corpus_size} prompts, "
                f"threshold={self.threshold}, medical_threshold={self.medical_threshold}"
            )
        except Exception as exc:
            logger.warning(f"DomainScopeGuard: embedding failed: {exc}")
            self._corpus_matrix = None
            self._corpus_size   = 0

        return self._corpus_size

    def _fetch_training_prompts(self) -> list:
        try:
            import psycopg
            cfg = {k: v for k, v in self.db_config.items() if v is not None}
            with psycopg.connect(**cfg) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT DISTINCT user_prompt FROM rlhf_interactions "
                        "WHERE user_prompt IS NOT NULL AND user_prompt <> '' "
                        "ORDER BY user_prompt"
                    )
                    rows = cur.fetchall()
            return [r[0] for r in rows if r[0]]
        except Exception as exc:
            logger.debug(f"DomainScopeGuard: psycopg fetch failed ({exc}), trying SQLite")

        try:
            import sqlite3
            db_path = os.path.join(os.path.dirname(__file__), "local_sft.db")
            with sqlite3.connect(db_path) as conn:
                rows = conn.execute(
                    "SELECT DISTINCT user_prompt FROM rlhf_interactions "
                    "WHERE user_prompt IS NOT NULL AND user_prompt <> '' "
                    "ORDER BY user_prompt"
                ).fetchall()
            return [r[0] for r in rows if r[0]]
        except Exception as exc2:
            logger.warning(f"DomainScopeGuard: SQLite fallback failed ({exc2})")

        return []

    # ── Inference gate ────────────────────────────────────────────────────────

    def check(self, query: str) -> tuple:
        """
        Check query against training scope.

        Returns:
            (status, max_score, message)
            status is one of STATUS_ACCEPTED, STATUS_GENERAL_MEDICAL, STATUS_REJECTED.
        """
        if not self.enabled:
            return STATUS_ACCEPTED, 1.0, ""
        if not self._embedder or self._corpus_matrix is None or self._corpus_size == 0:
            return STATUS_ACCEPTED, 1.0, ""

        try:
            q_vec = self._embedder.encode([query], convert_to_numpy=True)[0].astype(np.float32)
        except Exception as exc:
            logger.warning(f"DomainScopeGuard.check: embedding error ({exc}) — allowing through")
            return STATUS_ACCEPTED, 1.0, ""

        # Tier 1: in-scope training match
        training_sims = _cosine_similarity_matrix(q_vec, self._corpus_matrix)
        training_score = float(np.max(training_sims))

        if training_score >= self.threshold:
            return STATUS_ACCEPTED, training_score, ""

        # Tier 2: check against general medical reference corpus
        if self._medical_matrix is not None:
            medical_sims  = _cosine_similarity_matrix(q_vec, self._medical_matrix)
            medical_score = float(np.max(medical_sims))
        else:
            medical_score = 0.0

        if medical_score >= self.medical_threshold:
            logger.info(
                f"DomainScopeGuard: general_medical query "
                f"(training={training_score:.3f}, medical={medical_score:.3f}): {query[:80]!r}"
            )
            return STATUS_GENERAL_MEDICAL, training_score, GENERAL_MEDICAL_DISCLAIMER

        # Tier 3: fully out-of-scope
        logger.info(
            f"DomainScopeGuard: rejected query "
            f"(training={training_score:.3f}, medical={medical_score:.3f}): {query[:80]!r}"
        )
        return STATUS_REJECTED, training_score, self._build_decline_message()

    def _build_decline_message(self) -> str:
        return (
            "I\u2019m sorry, but your question falls outside the scope of topics I have been "
            "trained on."
        )

    # ── Management helpers ────────────────────────────────────────────────────

    def refresh(self) -> int:
        logger.info("DomainScopeGuard: refreshing corpus…")
        return self.load_corpus()

    def set_threshold(self, threshold: float) -> None:
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        self.threshold = threshold
        logger.info(f"DomainScopeGuard: threshold updated to {threshold}")

    def get_status(self) -> dict:
        return {
            "enabled":            self.enabled,
            "sbert_available":    _SBERT_AVAILABLE and self._embedder is not None,
            "corpus_size":        self._corpus_size,
            "threshold":          self.threshold,
            "medical_threshold":  self.medical_threshold,
            "last_refresh":       self._last_refresh.isoformat() if self._last_refresh else None,
            "covered_topics":     self._topic_summary,
            "pass_through_mode":  self._corpus_matrix is None,
        }


# ── Module-level instance (set by main.py at startup) ────────────────────────
scope_guard: Optional["DomainScopeGuard"] = None
