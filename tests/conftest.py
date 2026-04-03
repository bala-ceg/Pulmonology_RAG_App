"""
Shared pytest fixtures for the Medical RAG App test suite.

Provides:
- flask_app / client       — Flask test client with all external deps mocked
- mock_llm                 — LLM that returns canned text without hitting OpenAI
- mock_db                  — In-memory SQLite connection
- sample_pdf_bytes         — Minimal valid PDF bytes for upload tests
- real_chat_openai_class   — The unpatched ChatOpenAI class (for live tests)
"""

from __future__ import annotations

import io
import os
import sqlite3
import sys
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Load .env FIRST so real API credentials are in os.environ before setdefault
# ---------------------------------------------------------------------------
from dotenv import load_dotenv  # noqa: E402
load_dotenv()

# ---------------------------------------------------------------------------
# Save the REAL ChatOpenAI class BEFORE any patches are applied.
# The flask_app session fixture below will patch langchain_openai.ChatOpenAI
# with a MagicMock. Live tests need the original; we capture it here at
# import time so it's available via the real_chat_openai_class fixture.
# ---------------------------------------------------------------------------
from langchain_openai import ChatOpenAI as _RealChatOpenAI  # noqa: E402

# ---------------------------------------------------------------------------
# Minimal env-var fallbacks (setdefault won't override values from .env)
# ---------------------------------------------------------------------------
os.environ.setdefault("openai_api_key", "test-key-not-real")
os.environ.setdefault("base_url", "https://api.openai.com/v1")
os.environ.setdefault("llm_model_name", "gpt-4o-mini")
os.environ.setdefault("embedding_model_name", "text-embedding-3-small")
os.environ.setdefault("SCOPE_GUARD_ENABLED", "false")
# Force SQLite for SFT module — prevents _init_db_backend() from connecting to
# a real PostgreSQL instance before the flask_app fixture can patch psycopg.connect,
# which would leave _use_sqlite=False and cause SFT routes to fail in Flask tests.
os.environ.setdefault("SFT_USE_SQLITE", "true")


# ---------------------------------------------------------------------------
# Flask app fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def flask_app():
    """
    Return the Flask app in test mode (session-scoped for speed).

    All heavy external dependencies (OpenAI, Whisper, ChromaDB, PostgreSQL)
    are patched out so tests run offline without any API keys.
    """
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value = MagicMock(content="Mocked LLM response")

    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.return_value = [[0.0] * 1536]
    mock_embeddings.embed_query.return_value = [0.0] * 1536

    mock_whisper = MagicMock()
    mock_whisper.transcribe.return_value = {"text": "Mocked transcription"}

    patches = [
        patch("langchain_openai.ChatOpenAI", return_value=mock_llm_instance),
        patch("langchain_openai.OpenAIEmbeddings", return_value=mock_embeddings),
        patch("whisper.load_model", return_value=mock_whisper),
        patch("psycopg.connect", side_effect=Exception("DB not available in tests")),
    ]

    started = [p.start() for p in patches]

    from main import app  # noqa: PLC0415

    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False
    # Inject mocked instances so blueprints can access them via current_app.config
    app.config["LLM_INSTANCE"] = mock_llm_instance
    app.config["EMBEDDINGS"] = mock_embeddings
    app.config["WHISPER_MODEL"] = mock_whisper
    app.config["RAG_MANAGER"] = None
    app.config["INTEGRATED_RAG"] = None

    # Provide a real DomainScopeGuard in pass-through mode (SCOPE_GUARD_ENABLED=false)
    # so that endpoints that check scope_guard.get_status() return correct structure.
    try:
        from domain_scope_guard import DomainScopeGuard  # noqa: PLC0415
        app.config["SCOPE_GUARD"] = DomainScopeGuard(db_config={})
    except Exception:
        app.config["SCOPE_GUARD"] = None

    yield app

    for p in patches:
        p.stop()


@pytest.fixture(scope="session")
def client(flask_app):
    """Flask test client."""
    return flask_app.test_client()


# ---------------------------------------------------------------------------
# LLM mock
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_llm() -> MagicMock:
    """
    A MagicMock that mimics ChatOpenAI.

    ``mock_llm.invoke(prompt)`` returns a MagicMock with ``.content`` set to
    ``"Mocked medical response"`` so callers that extract ``.content`` work.

    Customise per-test::

        mock_llm.invoke.return_value = MagicMock(content="custom response")
    """
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="Mocked medical response")
    return llm


# ---------------------------------------------------------------------------
# In-memory SQLite DB
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_db() -> Generator[sqlite3.Connection, None, None]:
    """
    An in-memory SQLite connection that mirrors the pces_base schema
    used by tests (rlhf_interactions, sft_ranked_data, doctors).

    The connection is closed after the test.
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.executescript(
        """
        CREATE TABLE IF NOT EXISTS rlhf_interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            query TEXT,
            response TEXT,
            rating INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS sft_ranked_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            group_id TEXT,
            domain TEXT,
            prompt TEXT,
            response TEXT,
            rank INTEGER,
            sme_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS doctors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            department TEXT,
            email TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.commit()

    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# Sample PDF bytes
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_pdf_bytes() -> bytes:
    """
    Return minimal valid PDF bytes (a single blank page).

    Uses reportlab if available; otherwise returns a raw minimal PDF bytestring
    so tests can run even without reportlab installed.
    """
    try:
        from reportlab.lib.pagesizes import A4  # noqa: PLC0415
        from reportlab.platypus import SimpleDocTemplate  # noqa: PLC0415

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4)
        doc.build([])
        return buf.getvalue()
    except ImportError:
        # Minimal valid PDF that most parsers accept
        return (
            b"%PDF-1.4\n"
            b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
            b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\n"
            b"xref\n0 4\n0000000000 65535 f\n"
            b"0000000009 00000 n\n0000000058 00000 n\n0000000115 00000 n\n"
            b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n190\n%%EOF"
        )


# ---------------------------------------------------------------------------
# Real ChatOpenAI class (for live OpenAI tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def real_chat_openai_class():
    """
    Return the genuine (unpatched) ``ChatOpenAI`` class.

    The flask_app session fixture patches ``langchain_openai.ChatOpenAI`` with a
    MagicMock for the entire test session.  This fixture returns the class that
    was imported at conftest load time — *before* any patches were applied —
    so live tests can create real LLM instances that hit the OpenAI API.
    """
    return _RealChatOpenAI
