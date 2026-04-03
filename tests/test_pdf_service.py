"""
Unit tests for services/pdf_service.py.

No real Azure connection or disk I/O beyond temp files is needed.
All Azure calls are mocked.
"""

from __future__ import annotations

import io
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------

PATIENT_DATA = {
    "doctorName": "Dr. Smith",
    "patientName": "John Doe",
    "patientId": "P-001",
    "dateTime": "2026-04-01 10:00",
    "transcription": "Patient reports shortness of breath.",
    "summary": "Possible asthma exacerbation.",
    "conclusion": "Prescribe bronchodilator.",
    "patientProblem": "Shortness of breath",
}

CHAT_DATA = {
    "doctorName": "Dr. Smith",
    "patientName": "Jane Doe",
    "patientId": "P-002",
    "dateTime": "2026-04-01 11:00",
    "chatHistory": [
        {"role": "user", "content": "What is asthma?"},
        {"role": "assistant", "content": "Asthma is a chronic lung condition."},
    ],
    "summary": "Explained asthma basics.",
}

CONVERSATION_DATA = {
    "doctorName": "Dr. Jones",
    "patientName": "Bob Smith",
    "patientId": "P-003",
    "dateTime": "2026-04-01 12:00",
    "conversation": [
        {"role": "doctor", "content": "How are you?"},
        {"role": "patient", "content": "Not well."},
    ],
    "summary": "Patient feels unwell.",
    "translation": "Doctor: Comment allez-vous? Patient: Pas bien.",
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPDFServicePatient:
    """Tests for PDFService.generate_patient_pdf."""

    def test_returns_bytes(self) -> None:
        """generate_patient_pdf must return non-empty bytes."""
        pytest.importorskip("reportlab", reason="reportlab not installed")
        from services.pdf_service import PDFService  # noqa: PLC0415
        svc = PDFService()
        with patch.object(svc, "_upload_to_azure", return_value=None):
            pdf_bytes, _ = svc.generate_patient_pdf(PATIENT_DATA)
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 100
        assert pdf_bytes[:4] == b"%PDF"

    def test_azure_url_returned_when_available(self) -> None:
        pytest.importorskip("reportlab", reason="reportlab not installed")
        from services.pdf_service import PDFService  # noqa: PLC0415
        svc = PDFService()
        with patch.object(svc, "_upload_to_azure", return_value="https://storage.example.com/file.pdf"):
            _, azure_url = svc.generate_patient_pdf(PATIENT_DATA)
        assert azure_url == "https://storage.example.com/file.pdf"

    def test_azure_url_empty_when_upload_fails(self) -> None:
        pytest.importorskip("reportlab", reason="reportlab not installed")
        from services.pdf_service import PDFService  # noqa: PLC0415
        svc = PDFService()
        with patch.object(svc, "_upload_to_azure", return_value=None):
            _, azure_url = svc.generate_patient_pdf(PATIENT_DATA)
        assert azure_url is None or azure_url == ""


class TestPDFServiceChat:
    """Tests for PDFService.generate_chat_pdf."""

    def test_returns_bytes(self) -> None:
        pytest.importorskip("reportlab", reason="reportlab not installed")
        from services.pdf_service import PDFService  # noqa: PLC0415
        svc = PDFService()
        with patch.object(svc, "_upload_to_azure", return_value=None):
            pdf_bytes, _ = svc.generate_chat_pdf(CHAT_DATA)
        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes[:4] == b"%PDF"

    def test_empty_chat_history_handled(self) -> None:
        pytest.importorskip("reportlab", reason="reportlab not installed")
        from services.pdf_service import PDFService  # noqa: PLC0415
        svc = PDFService()
        data = {**CHAT_DATA, "chatHistory": []}
        with patch.object(svc, "_upload_to_azure", return_value=None):
            pdf_bytes, _ = svc.generate_chat_pdf(data)
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 0


class TestPDFServiceConversation:
    """Tests for PDFService.generate_conversation_pdf."""

    def test_returns_bytes(self) -> None:
        pytest.importorskip("reportlab", reason="reportlab not installed")
        from services.pdf_service import PDFService  # noqa: PLC0415
        svc = PDFService()
        with patch.object(svc, "_upload_to_azure", return_value=None):
            pdf_bytes, _ = svc.generate_conversation_pdf(CONVERSATION_DATA)
        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes[:4] == b"%PDF"


class TestPDFServiceAzureUpload:
    """Tests for PDFService._upload_to_azure."""

    def test_returns_none_when_azure_not_available(self) -> None:
        from services.pdf_service import PDFService  # noqa: PLC0415
        svc = PDFService()
        with patch("services.pdf_service.AZURE_AVAILABLE", False):
            result = svc._upload_to_azure(b"pdf content", "test.pdf", "research")
        assert result is None

    def test_calls_storage_manager_when_available(self) -> None:
        from services.pdf_service import PDFService  # noqa: PLC0415
        mock_manager = MagicMock()
        mock_manager.upload_research_pdf.return_value = "https://blob.example.com/test.pdf"
        svc = PDFService()
        with patch("services.pdf_service.AZURE_AVAILABLE", True), \
             patch("services.pdf_service.get_storage_manager", return_value=mock_manager):
            result = svc._upload_to_azure(b"pdf content", "test.pdf", "research")
        assert result == "https://blob.example.com/test.pdf"
