"""
Application configuration.

All runtime parameters are read from environment variables with the original
hardcoded values as defaults, so the app behaves identically without any
changes to existing .env files.
"""

from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ------------------------------------------------------------------
    # Storage paths
    # ------------------------------------------------------------------
    KB_PATH: str = os.getenv("KB_PATH", "./KB/")
    VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", "./vector_dbs/")
    ORGANIZATION_KB_PATH: str = os.getenv("ORG_KB_PATH", "./Organization_KB/")
    ORGANIZATION_VECTOR_DB_PATH: str = os.getenv(
        "ORG_VECTOR_DB_PATH", "./vector_dbs/organization/"
    )
    PERSIST_DIRECTORY: str = os.getenv("PERSIST_DIRECTORY", "./vector_db")
    DISCIPLINES_CONFIG_PATH: str = os.getenv(
        "DISCIPLINES_CONFIG_PATH", "config/disciplines.json"
    )

    # ------------------------------------------------------------------
    # PostgreSQL — primary database (pces_base)
    # ------------------------------------------------------------------
    DB_HOST: str | None = os.getenv("DB_HOST")
    DB_PORT: str | None = os.getenv("DB_PORT")
    DB_NAME: str | None = os.getenv("DB_NAME")
    DB_USER: str | None = os.getenv("DB_USER")
    DB_PASSWORD: str | None = os.getenv("DB_PASSWORD")
    DB_CONNECT_TIMEOUT: int = int(os.getenv("DB_CONNECT_TIMEOUT", "30"))
    DB_POOL_MIN_SIZE: int = int(os.getenv("DB_POOL_MIN_SIZE", "1"))
    DB_POOL_MAX_SIZE: int = int(os.getenv("DB_POOL_MAX_SIZE", "10"))
    DB_POOL_RECONNECT_TIMEOUT: int = int(os.getenv("DB_POOL_RECONNECT_TIMEOUT", "30"))

    # PostgreSQL — EHR/CCM tool database (pces_ehr_ccm)
    PG_TOOL_HOST: str | None = os.getenv("PG_TOOL_HOST")
    PG_TOOL_PORT: str | None = os.getenv("PG_TOOL_PORT")
    PG_TOOL_NAME: str | None = os.getenv("PG_TOOL_NAME")
    PG_TOOL_USER: str | None = os.getenv("PG_TOOL_USER")
    PG_TOOL_PASSWORD: str | None = os.getenv("PG_TOOL_PASSWORD")

    # ------------------------------------------------------------------
    # LLM & Embeddings
    # ------------------------------------------------------------------
    OPENAI_API_KEY: str | None = os.getenv("openai_api_key")
    OPENAI_BASE_URL: str | None = os.getenv("base_url")
    LLM_MODEL_NAME: str | None = os.getenv("llm_model_name")
    EMBEDDING_MODEL_NAME: str | None = os.getenv("embedding_model_name")
    LLM_REQUEST_TIMEOUT: int = int(os.getenv("LLM_REQUEST_TIMEOUT", "30"))
    LLM_DEFAULT_TEMPERATURE: float = float(os.getenv("LLM_DEFAULT_TEMPERATURE", "0.1"))

    MEDICAL_SYSTEM_MESSAGE: str = (
        "You are a medical AI assistant providing accurate, "
        "evidence-based medical information and guidance."
    )

    # ------------------------------------------------------------------
    # RAG / Vector DB parameters
    # ------------------------------------------------------------------
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "4096"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "128"))
    RETRIEVAL_K: int = int(os.getenv("RETRIEVAL_K", "2"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))
    TFIDF_LOCAL_THRESHOLD: float = float(os.getenv("TFIDF_LOCAL_THRESHOLD", "0.3"))

    # ------------------------------------------------------------------
    # Upload limits
    # ------------------------------------------------------------------
    MAX_FILES_PER_SESSION: int = int(os.getenv("MAX_FILES_PER_SESSION", "10"))
    MAX_URLS_PER_SESSION: int = int(os.getenv("MAX_URLS_PER_SESSION", "3"))

    # ------------------------------------------------------------------
    # Audio / Whisper
    # ------------------------------------------------------------------
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "base")

    # ------------------------------------------------------------------
    # RLHF / SFT training
    # ------------------------------------------------------------------
    MIN_TRAINING_SAMPLES: int = int(os.getenv("MIN_TRAINING_SAMPLES", "20"))
    TRAINING_TIMEOUT_SECONDS: int = int(os.getenv("TRAINING_TIMEOUT_SECONDS", "300"))
    TRAINING_JOIN_TIMEOUT: int = int(os.getenv("TRAINING_JOIN_TIMEOUT", "120"))

    # ------------------------------------------------------------------
    # Azure Blob Storage (optional)
    # ------------------------------------------------------------------
    AZURE_STORAGE_CONNECTION_STRING: str | None = os.getenv(
        "AZURE_STORAGE_CONNECTION_STRING"
    )
    AZURE_STORAGE_CONTAINER_NAME: str | None = os.getenv(
        "AZURE_STORAGE_CONTAINER_NAME"
    )

    # ------------------------------------------------------------------
    # External APIs (optional)
    # ------------------------------------------------------------------
    TAVILY_API_KEY: str | None = os.getenv("TAVILY_API_KEY")
    APIFY_API_KEY: str | None = os.getenv("apify_api_key")
    HUGGINGFACE_TOKEN: str | None = os.getenv("HUGGINGFACE_TOKEN")

    # ------------------------------------------------------------------
    # Flask
    # ------------------------------------------------------------------
    DEBUG: bool = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    HOST: str = os.getenv("FLASK_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("FLASK_PORT", "5000"))

    @classmethod
    def db_kwargs(cls) -> dict:
        """Return a dict of non-None DB connection kwargs for psycopg."""
        return {
            k: v
            for k, v in {
                "host": cls.DB_HOST,
                "port": cls.DB_PORT,
                "dbname": cls.DB_NAME,
                "user": cls.DB_USER,
                "password": cls.DB_PASSWORD,
                "connect_timeout": cls.DB_CONNECT_TIMEOUT,
            }.items()
            if v is not None
        }

    @classmethod
    def ensure_directories(cls) -> None:
        """Create all required storage directories if they do not exist."""
        for path in (
            cls.KB_PATH,
            cls.VECTOR_DB_PATH,
            cls.ORGANIZATION_KB_PATH,
            cls.ORGANIZATION_VECTOR_DB_PATH,
        ):
            os.makedirs(path, exist_ok=True)
