"""
Shared pytest fixtures for the Medical RAG App test suite.
"""
import os
import sys
import pytest

# Ensure the project root is on sys.path so all modules can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Provide minimal env vars before any app import
os.environ.setdefault("openai_api_key", "test-key-not-real")
os.environ.setdefault("base_url", "https://api.openai.com/v1")
os.environ.setdefault("llm_model_name", "gpt-4o-mini")
os.environ.setdefault("embedding_model_name", "text-embedding-3-small")
os.environ.setdefault("SCOPE_GUARD_ENABLED", "false")   # disable heavy SBERT at import time


@pytest.fixture(scope="session")
def flask_app():
    """Return the Flask app in test mode (session-scoped for speed)."""
    from main import app
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False
    return app


@pytest.fixture(scope="session")
def client(flask_app):
    """Flask test client."""
    return flask_app.test_client()
