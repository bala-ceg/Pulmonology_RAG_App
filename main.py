"""
Medical RAG Application — Bootstrap Entry Point
================================================

This file is the application bootstrap. It:
  1. Loads configuration from environment variables (via config.Config)
  2. Initialises the database connection pool
  3. Initialises LLM, embeddings, and all RAG subsystems
  4. Stores shared state on Flask app.config so Blueprints can access it
  5. Registers all Blueprint modules (routes/)
  6. Starts the Flask development server

All route logic lives in routes/.  Business logic lives in services/.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import asyncio
import glob
import io
import json
import logging
import os
import re
import tempfile
import threading
import time
import traceback
import warnings
from datetime import datetime
from typing import List

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import fitz  # PyMuPDF
import psycopg
import torch
import whisper
from apify_client import ApifyClient
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load env vars before importing Config so all os.getenv() calls see them
load_dotenv()

from config import Config
from utils.error_handlers import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Suppress noisy third-party loggers
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

for _noisy_logger in ("httpx", "httpcore", "sentence_transformers", "transformers"):
    logging.getLogger(_noisy_logger).setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Local — optional subsystems (graceful degradation)
# ---------------------------------------------------------------------------
try:
    from rag_architecture import TwoStoreRAGManager, TFIDFLexicalGate
    RAG_ARCHITECTURE_AVAILABLE = True
except ImportError:
    logger.warning("RAG architecture not available. Install: pip install scikit-learn")
    RAG_ARCHITECTURE_AVAILABLE = False

try:
    from azure_storage import get_storage_manager
    AZURE_AVAILABLE = True
except ImportError:
    logger.warning("Azure storage not available. Install: pip install azure-storage-blob")
    AZURE_AVAILABLE = False

try:
    from voice_diarization import get_diarization_processor
    DIARIZATION_AVAILABLE = True
except ImportError:
    logger.warning("Voice diarization not available. Install: pip install pyannote.audio torch")
    DIARIZATION_AVAILABLE = False

try:
    from integrated_rag import IntegratedMedicalRAG
    INTEGRATED_RAG_AVAILABLE = True
    logger.info("Integrated RAG system loaded successfully")
except ImportError:
    logger.warning("Integrated RAG system not available.")
    INTEGRATED_RAG_AVAILABLE = False

try:
    from domain_scope_guard import DomainScopeGuard
    SCOPE_GUARD_AVAILABLE = True
except ImportError:
    SCOPE_GUARD_AVAILABLE = False

from psycopg import sql

# ---------------------------------------------------------------------------
# Legacy path constants — kept for backward compat with existing test files
# ---------------------------------------------------------------------------
BASE_STORAGE_PATH = Config.KB_PATH
VECTOR_DB_PATH = Config.VECTOR_DB_PATH
ORGANIZATION_KB_PATH = Config.ORGANIZATION_KB_PATH
ORGANIZATION_VECTOR_DB_PATH = Config.ORGANIZATION_VECTOR_DB_PATH
VECTOR_DBS_FOLDER = Config.VECTOR_DB_PATH

Config.ensure_directories()

# ---------------------------------------------------------------------------
# Legacy db_config dict (kept for any code that still references it directly)
# ---------------------------------------------------------------------------
db_config = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "connect_timeout": Config.DB_CONNECT_TIMEOUT,
}

# ---------------------------------------------------------------------------
# Database connection pool
# ---------------------------------------------------------------------------
from services.db_service import init_pool, get_connection as _pg_conn

init_pool()

# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------
app = Flask(__name__)

persist_directory = Config.PERSIST_DIRECTORY

# ---------------------------------------------------------------------------
# LLM and Embeddings
# ---------------------------------------------------------------------------
from services.llm_service import llm_service

llm = llm_service.get_llm()
embeddings = llm_service.get_embeddings()


def create_contextual_llm(patient_context: str | None = None) -> ChatOpenAI:
    """Create an LLM instance with optional patient context. Delegates to LLMService."""
    return llm_service.create_contextual_llm(patient_context)


# ---------------------------------------------------------------------------
# Apify client
# ---------------------------------------------------------------------------
client = ApifyClient(os.getenv("apify_api_key"))

# ---------------------------------------------------------------------------
# Whisper model
# ---------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model(Config.WHISPER_MODEL).to(device)

# ---------------------------------------------------------------------------
# Text splitter
# ---------------------------------------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=Config.CHUNK_SIZE,
    chunk_overlap=Config.CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", " "],
)

# ---------------------------------------------------------------------------
# Metadata helpers (kept for backward compat)
# ---------------------------------------------------------------------------

def get_timestamp() -> str:
    """Generate timestamp in MMDDYYYYHHMM format."""
    return time.strftime("%m%d%Y%H%M")


def get_latest_vector_db() -> str | None:
    """Find the latest vector database in the vector_dbs folder."""
    vector_dbs = glob.glob(os.path.join(VECTOR_DBS_FOLDER, "*"))
    if not vector_dbs:
        logger.info("No existing vector DB found. A new one will be created.")
        return None
    latest_db = max(vector_dbs, key=os.path.getmtime)
    logger.info("Using latest vector DB: %s", latest_db)
    return latest_db


def load_metadata(file_path: str) -> list:
    """Load JSON metadata from the given file path."""
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        logger.warning("Metadata file not found: %s", file_path)
        return []
    except json.JSONDecodeError as e:
        logger.error("Error decoding JSON at %s: %s", file_path, e)
        return []


def process_pdf_metadata(pdf_metadata: list, splitter) -> list:
    """Chunk text content from PDF metadata into LangChain Documents."""
    chunked: list = []
    for doc in pdf_metadata:
        file_name = doc.get("file_name", "Unknown File")
        for page in doc.get("text", []):
            page_number = page.get("page", "Unknown Page")
            page_text = page.get("text", "").strip()
            if not page_text:
                continue
            for chunk in splitter.split_text(page_text):
                chunked.append(
                    Document(
                        page_content=chunk,
                        metadata={"source": file_name, "type": "pdf", "page": page_number},
                    )
                )
    return chunked


def process_url_metadata(url_metadata: list, splitter) -> list:
    """Chunk text content from URL metadata into LangChain Documents."""
    chunked: list = []
    for entry in url_metadata:
        url = entry.get("url", "Unknown URL")
        text_content = entry.get("text", "").strip()
        date_info = entry.get("date", "Unknown Date")
        if not text_content:
            continue
        for chunk in splitter.split_text(text_content):
            chunked.append(
                Document(
                    page_content=chunk,
                    metadata={"source": url, "type": "url", "date": date_info},
                )
            )
    return chunked


pdf_metadata = load_metadata("pdf_metadata.json")
url_metadata = load_metadata("url_metadata.json")
all_documents = (
    process_pdf_metadata(pdf_metadata, text_splitter)
    + process_url_metadata(url_metadata, text_splitter)
)

# ---------------------------------------------------------------------------
# RAG systems
# ---------------------------------------------------------------------------
from services.rag_service import rag_service

rag_service.initialize(embeddings, llm)
rag_manager = rag_service.rag_manager
integrated_rag_system = rag_service.integrated_rag_system
scope_guard = rag_service.scope_guard

# ---------------------------------------------------------------------------
# Store shared state on app.config for Blueprint access
# ---------------------------------------------------------------------------
app.config.update(
    {
        "LLM_INSTANCE": llm,
        "EMBEDDINGS": embeddings,
        "RAG_MANAGER": rag_manager,
        "INTEGRATED_RAG": integrated_rag_system,
        "SCOPE_GUARD": scope_guard,
        "TEXT_SPLITTER": text_splitter,
        "WHISPER_MODEL": model,
        "APIFY_CLIENT": client,
        "DISCIPLINES_CONFIG": None,
        "LAST_SESSION_FOLDER": None,
        "MEDICAL_ROUTER": None,
    }
)

# ---------------------------------------------------------------------------
# Register all Blueprint route modules
# ---------------------------------------------------------------------------
from routes import register_blueprints

register_blueprints(app)

# ---------------------------------------------------------------------------
# Backward-compatibility re-exports
# ---------------------------------------------------------------------------
last_created_folder: str | None = None

try:
    from routes.disciplines import (
        MedicalQueryRouter,
        load_disciplines_config,
        initialize_session,
    )
except Exception:
    pass

try:
    from routes.documents import _can_upload_more_files as can_upload_more_files
except Exception:
    pass

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=3000, use_reloader=False)
