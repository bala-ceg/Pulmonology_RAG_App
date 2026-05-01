"""
Flask routes package.

Call ``register_blueprints(app)`` from main.py after creating the Flask app
to attach all Blueprint modules.
"""

from __future__ import annotations

from flask import Flask


def register_blueprints(app: Flask) -> None:
    """Import and register all application Blueprints."""
    from routes.audio import audio_bp
    from routes.query import query_bp
    from routes.documents import documents_bp
    from routes.disciplines import disciplines_bp
    from routes.azure_routes import azure_bp
    from routes.conversation import conversation_bp
    from routes.pdf_generation import pdf_bp
    from routes.rlhf import rlhf_bp
    from routes.sft import sft_bp
    from routes.doctors import doctors_bp
    from routes.pinecone_routes import pinecone_bp

    for bp in (
        audio_bp,
        query_bp,
        documents_bp,
        disciplines_bp,
        azure_bp,
        conversation_bp,
        pdf_bp,
        rlhf_bp,
        sft_bp,
        doctors_bp,
        pinecone_bp,
    ):
        app.register_blueprint(bp)
