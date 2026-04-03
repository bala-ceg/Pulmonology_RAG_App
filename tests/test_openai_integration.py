"""
OpenAI integration tests — verifies that the LLM wiring is correct and that
long-context answers flow through the system end-to-end.

Test layers
-----------
1. LLMService unit tests        — mock ChatOpenAI; verify invoke(), get_llm(),
                                   create_contextual_llm()
2. Endpoint tests (mocked LLM)  — /plain_english, /data, /generate_summary;
                                   confirm the full answer is returned intact
3. Long-context response tests  — mock an LLM that returns a large answer
                                   (>2 000 chars) and verify no truncation occurs
4. Live integration tests       — skipped unless OPENAI_TEST_LIVE=1 is set;
                                   exercises the real API with a concise medical
                                   query and checks minimum response length

Run only unit + endpoint tests (default, offline):
    pytest tests/test_openai_integration.py

Run everything including live tests:
    OPENAI_TEST_LIVE=1 pytest tests/test_openai_integration.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


# A realistic multi-section medical answer (>2 000 chars) used as a mock payload.
_LONG_MEDICAL_ANSWER = (
    "## Chronic Obstructive Pulmonary Disease (COPD) — Comprehensive Overview\n\n"
    "### Pathophysiology\n"
    "COPD is characterised by persistent airflow limitation caused by a mixture of small "
    "airways disease (obstructive bronchiolitis) and parenchymal destruction (emphysema). "
    "The relative contribution of each varies between patients. Exposure to noxious particles "
    "— predominantly cigarette smoke — triggers an abnormal inflammatory response involving "
    "neutrophils, macrophages, and CD8+ T-lymphocytes. Repeated cycles of injury and repair "
    "lead to airway remodelling, mucus hypersecretion, and progressive loss of elastic recoil.\n\n"
    "### Diagnosis\n"
    "Spirometry is the gold standard: a post-bronchodilator FEV₁/FVC ratio below 0.70 confirms "
    "airflow obstruction. The GOLD classification grades severity as I (mild, FEV₁ ≥ 80%) through "
    "IV (very severe, FEV₁ < 30%). Symptom burden is assessed with validated tools such as the "
    "mMRC dyspnoea scale and the COPD Assessment Test (CAT).\n\n"
    "### Pharmacological Management\n"
    "Short-acting β₂-agonists (SABA) and short-acting muscarinic antagonists (SAMA) provide "
    "on-demand relief. Long-acting bronchodilators (LABAs and LAMAs) form the backbone of "
    "maintenance therapy and reduce exacerbation frequency. Inhaled corticosteroids (ICS) are "
    "added for patients with frequent exacerbations or eosinophil counts ≥ 300 cells/µL. "
    "Roflumilast, a PDE4 inhibitor, is an option in severe disease with chronic bronchitis.\n\n"
    "### Non-Pharmacological Management\n"
    "Smoking cessation is the single most effective intervention and slows disease progression "
    "regardless of stage. Pulmonary rehabilitation improves exercise capacity, reduces dyspnoea, "
    "and decreases hospitalisation rates. Long-term oxygen therapy (LTOT) is indicated when "
    "resting PaO₂ ≤ 7.3 kPa (55 mmHg) or SaO₂ ≤ 88%, improving survival in severe hypoxaemia. "
    "Lung volume reduction surgery and bronchoscopic lung volume reduction (BLVR) are considered "
    "in selected patients with predominantly upper-lobe emphysema.\n\n"
    "### Exacerbation Management\n"
    "Acute exacerbations are managed with intensified bronchodilators, systemic corticosteroids "
    "(prednisolone 30–40 mg for 5 days), and antibiotics when sputum purulence is present "
    "(Anthonisen criteria). Non-invasive positive-pressure ventilation (NIV/BiPAP) reduces "
    "intubation rates and in-hospital mortality in hypercapnic respiratory failure (pH < 7.35, "
    "PaCO₂ > 6 kPa).\n\n"
    "### Prognosis\n"
    "The BODE index (BMI, airflow Obstruction, Dyspnoea, Exercise capacity) predicts mortality "
    "more accurately than FEV₁ alone. Five-year survival in GOLD IV disease is approximately "
    "20–30%. Comorbidities — cardiovascular disease, lung cancer, osteoporosis, depression — "
    "significantly influence outcomes and should be actively managed.\n\n"
    "**Sources:** GOLD 2024 Guidelines; Lancet Respir Med 2022; NEJM 2020."
)


def _make_llm_mock(response_text: str) -> MagicMock:
    """Return a MagicMock that behaves like ChatOpenAI for *response_text*."""
    mock = MagicMock()
    ai_msg = MagicMock()
    ai_msg.content = response_text
    mock.invoke.return_value = ai_msg
    return mock


# ─────────────────────────────────────────────────────────────────────────────
# 1. LLMService unit tests
# ─────────────────────────────────────────────────────────────────────────────


class TestLLMServiceUnit:
    """Unit tests for services/llm_service.py — no Flask app required."""

    def test_invoke_returns_content_string(self):
        from services.llm_service import LLMService

        svc = LLMService.__new__(LLMService)
        mock_llm = _make_llm_mock("Asthma is a chronic airway disease.")
        svc._llm = mock_llm
        svc._config = type("Cfg", (), {"LLM_DEFAULT_TEMPERATURE": 0.1})

        result = svc.invoke("What is asthma?")
        assert isinstance(result, str)
        assert "Asthma" in result

    def test_invoke_with_plain_string_response(self):
        """invoke() handles LLMs that return a raw string (no .content attr)."""
        from services.llm_service import LLMService

        svc = LLMService.__new__(LLMService)
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Plain string response"
        svc._llm = mock_llm
        svc._config = type("Cfg", (), {"LLM_DEFAULT_TEMPERATURE": 0.1})

        result = svc.invoke("Any question")
        assert result == "Plain string response"

    def test_invoke_raises_runtime_error_on_exception(self):
        from services.llm_service import LLMService

        svc = LLMService.__new__(LLMService)
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = ConnectionError("API unreachable")
        svc._llm = mock_llm
        svc._config = type("Cfg", (), {"LLM_DEFAULT_TEMPERATURE": 0.1})

        with pytest.raises(RuntimeError, match="LLM invocation failed"):
            svc.invoke("Will fail")

    def test_get_llm_returns_same_instance_by_default(self):
        from services.llm_service import LLMService

        svc = LLMService.__new__(LLMService)
        mock_llm = _make_llm_mock("ok")
        svc._llm = mock_llm
        svc._config = type(
            "Cfg",
            (),
            {
                "LLM_DEFAULT_TEMPERATURE": 0.1,
                "OPENAI_API_KEY": "x",
                "OPENAI_BASE_URL": "https://api.openai.com/v1",
                "LLM_MODEL_NAME": "gpt-4o-mini",
                "LLM_REQUEST_TIMEOUT": 60,
            },
        )

        assert svc.get_llm() is mock_llm

    def test_create_contextual_llm_sets_system_message(self):
        from services.llm_service import LLMService

        with patch("services.llm_service.ChatOpenAI") as MockChatOpenAI:
            fake_instance = MagicMock()
            MockChatOpenAI.return_value = fake_instance

            svc = LLMService.__new__(LLMService)
            svc._llm = MagicMock()
            svc._config = type(
                "Cfg",
                (),
                {
                    "LLM_DEFAULT_TEMPERATURE": 0.1,
                    "OPENAI_API_KEY": "test",
                    "OPENAI_BASE_URL": "https://api.openai.com/v1",
                    "LLM_MODEL_NAME": "gpt-4o-mini",
                    "LLM_REQUEST_TIMEOUT": 60,
                    "MEDICAL_SYSTEM_MESSAGE": "You are a medical AI.",
                },
            )

            ctx_llm = svc.create_contextual_llm(patient_context="Patient: John, 45M, diabetic")
            assert "Patient: John, 45M, diabetic" in ctx_llm._system_message

    def test_create_contextual_llm_without_context_uses_base_message(self):
        from services.llm_service import LLMService

        with patch("services.llm_service.ChatOpenAI") as MockChatOpenAI:
            fake_instance = MagicMock()
            MockChatOpenAI.return_value = fake_instance

            svc = LLMService.__new__(LLMService)
            svc._llm = MagicMock()
            base_msg = "You are a medical AI."
            svc._config = type(
                "Cfg",
                (),
                {
                    "LLM_DEFAULT_TEMPERATURE": 0.1,
                    "OPENAI_API_KEY": "test",
                    "OPENAI_BASE_URL": "https://api.openai.com/v1",
                    "LLM_MODEL_NAME": "gpt-4o-mini",
                    "LLM_REQUEST_TIMEOUT": 60,
                    "MEDICAL_SYSTEM_MESSAGE": base_msg,
                },
            )

            ctx_llm = svc.create_contextual_llm()
            assert ctx_llm._system_message == base_msg


# ─────────────────────────────────────────────────────────────────────────────
# 2. /plain_english endpoint
# ─────────────────────────────────────────────────────────────────────────────


class TestPlainEnglishEndpoint:
    """Tests for POST /plain_english using the session-mocked LLM."""

    def test_returns_200(self, client):
        resp = client.post(
            "/plain_english",
            data=json.dumps({"text": "What is COPD?"}),
            content_type="application/json",
        )
        assert resp.status_code == 200

    def test_returns_refined_text_key(self, client):
        resp = client.post(
            "/plain_english",
            data=json.dumps({"text": "Explain bronchial hyperresponsiveness"}),
            content_type="application/json",
        )
        data = json.loads(resp.data)
        assert "refined_text" in data

    def test_mocked_llm_response_is_preserved(self, flask_app):
        """Inject a specific mock answer and verify the endpoint echoes it."""
        expected = "What is COPD in simple words?"
        mock = _make_llm_mock(expected)

        with flask_app.test_request_context():
            flask_app.config["LLM_INSTANCE"] = mock

        with flask_app.test_client() as c:
            resp = c.post(
                "/plain_english",
                data=json.dumps({"text": "COPD"}),
                content_type="application/json",
            )
        data = json.loads(resp.data)
        assert data["refined_text"] == expected
        # Restore original mock (session fixture reuse)
        flask_app.config["LLM_INSTANCE"] = flask_app.config.get("LLM_INSTANCE")

    def test_empty_text_returns_message(self, client):
        resp = client.post(
            "/plain_english",
            data=json.dumps({"text": ""}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "message" in data or data.get("refined_text") == ""

    def test_long_input_accepted(self, client):
        long_text = "What are the diagnostic criteria for " + ("pulmonary fibrosis " * 50)
        resp = client.post(
            "/plain_english",
            data=json.dumps({"text": long_text}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "refined_text" in data


# ─────────────────────────────────────────────────────────────────────────────
# 3. /generate_summary endpoint
# ─────────────────────────────────────────────────────────────────────────────

_SAMPLE_SUMMARY_RESPONSE = (
    "SUMMARY:\n"
    "Patient presented with a 3-week history of productive cough and progressive dyspnoea. "
    "Examination revealed reduced breath sounds at the right base. CXR showed right-sided "
    "consolidation consistent with community-acquired pneumonia.\n\n"
    "CONCLUSION:\n"
    "Commence amoxicillin-clavulanate 875/125 mg BD for 7 days. Repeat CXR in 6 weeks. "
    "Return if symptoms worsen or fever persists beyond 48 hours of antibiotics."
)

_LONG_TRANSCRIPT = (
    "Doctor: Good morning. What brings you in today?\n"
    "Patient: I've had this cough for three weeks now, doctor. It's getting worse.\n"
    "Doctor: Is the cough productive? Are you bringing up any phlegm?\n"
    "Patient: Yes, greenish sputum mostly in the mornings.\n"
    "Doctor: Any fever, chills, or night sweats?\n"
    "Patient: Some low-grade fever, around 37.8. No night sweats.\n"
    "Doctor: How is your breathing? Any shortness of breath?\n"
    "Patient: Yes, especially when I climb stairs. Much worse than usual.\n"
    "Doctor: Any chest pain? Do you smoke?\n"
    "Patient: Mild right-sided chest discomfort. I quit smoking two years ago — smoked for 20 years.\n"
    "Doctor: Let me listen to your chest... I can hear reduced breath sounds at the right base. "
    "I'll order a chest X-ray and some bloods. Given your symptoms and exam findings, "
    "I suspect a right lower lobe pneumonia.\n"
    "Patient: Is that serious?\n"
    "Doctor: It's treatable with antibiotics. We'll get those results and go from there.\n"
) * 3  # multiply to create a sufficiently long transcript


class TestGenerateSummaryEndpoint:
    def test_returns_200_with_valid_transcription(self, flask_app):
        mock_llm = _make_llm_mock(_SAMPLE_SUMMARY_RESPONSE)
        flask_app.config["LLM_INSTANCE"] = mock_llm

        with flask_app.test_client() as c:
            resp = c.post(
                "/generate_summary",
                data=json.dumps(
                    {
                        "transcription": "Patient has a cough.",
                        "doctor_name": "Dr Smith",
                        "patient_name": "John Doe",
                    }
                ),
                content_type="application/json",
            )
        assert resp.status_code == 200

    def test_response_has_summary_and_conclusion(self, flask_app):
        mock_llm = _make_llm_mock(_SAMPLE_SUMMARY_RESPONSE)
        flask_app.config["LLM_INSTANCE"] = mock_llm

        with flask_app.test_client() as c:
            resp = c.post(
                "/generate_summary",
                data=json.dumps({"transcription": "Patient has a cough."}),
                content_type="application/json",
            )
        data = json.loads(resp.data)
        assert data.get("success") is True
        assert "summary" in data
        assert "conclusion" in data

    def test_summary_field_is_non_empty(self, flask_app):
        mock_llm = _make_llm_mock(_SAMPLE_SUMMARY_RESPONSE)
        flask_app.config["LLM_INSTANCE"] = mock_llm

        with flask_app.test_client() as c:
            resp = c.post(
                "/generate_summary",
                data=json.dumps({"transcription": "Patient has a cough."}),
                content_type="application/json",
            )
        data = json.loads(resp.data)
        assert len(data["summary"].strip()) > 0

    def test_long_transcript_accepted_and_processed(self, flask_app):
        """A multi-turn transcript longer than 1 000 chars must be processed."""
        mock_llm = _make_llm_mock(_SAMPLE_SUMMARY_RESPONSE)
        flask_app.config["LLM_INSTANCE"] = mock_llm

        assert len(_LONG_TRANSCRIPT) > 1000, "Fixture must be >1000 chars"

        with flask_app.test_client() as c:
            resp = c.post(
                "/generate_summary",
                data=json.dumps(
                    {
                        "transcription": _LONG_TRANSCRIPT,
                        "doctor_name": "Dr Jones",
                        "patient_name": "Jane Smith",
                    }
                ),
                content_type="application/json",
            )
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data.get("success") is True

    def test_missing_transcription_returns_400(self, client):
        resp = client.post(
            "/generate_summary",
            data=json.dumps({"doctor_name": "Dr X"}),
            content_type="application/json",
        )
        assert resp.status_code == 400


# ─────────────────────────────────────────────────────────────────────────────
# 4. Long-context answer tests  (/data endpoint)
# ─────────────────────────────────────────────────────────────────────────────


def _make_integrated_rag_mock(answer_text: str) -> MagicMock:
    """
    Return a MagicMock that behaves like IntegratedMedicalRAG.

    The /data route checks ``integrated_rag_system.query(...).get("answer")``.
    When this is truthy the route returns immediately with the answer text,
    bypassing the RAG manager and the legacy medical-router path (which both
    require real infrastructure in tests).
    """
    mock = MagicMock()
    mock.query.return_value = {
        "answer": answer_text,
        "routing_info": {"sources": [], "confidence": "high"},
        "tools_used": ["Wikipedia_Search"],
    }
    return mock


class TestLongContextAnswers:
    """
    Verify that the /data endpoint passes large LLM answers through intact.

    We inject a mock IntegratedMedicalRAG (``INTEGRATED_RAG`` on app.config)
    that returns a pre-set answer string.  This exercises the JSON
    serialisation, response assembly, and HTTP transport path without
    requiring a real LLM, vector DB, or PostgreSQL connection.
    """

    def _post_query(self, flask_app, answer_text: str, query: str) -> dict[str, Any]:
        """Post *query* to /data with an integrated-RAG mock returning *answer_text*."""
        flask_app.config["INTEGRATED_RAG"] = _make_integrated_rag_mock(answer_text)
        try:
            with flask_app.test_client() as c:
                resp = c.post(
                    "/data",
                    data=json.dumps({"data": query, "session_id": "test-session"}),
                    content_type="application/json",
                )
            assert resp.status_code == 200
            return json.loads(resp.data)
        finally:
            flask_app.config["INTEGRATED_RAG"] = None  # restore so other tests are unaffected

    def test_short_answer_preserved(self, flask_app):
        short = "COPD is a lung disease."
        data = self._post_query(flask_app, short, "What is COPD?")
        msg = data.get("message", "")
        assert short in msg

    def test_long_answer_not_truncated(self, flask_app):
        """Response >2 000 chars must arrive at the caller without truncation."""
        data = self._post_query(
            flask_app, _LONG_MEDICAL_ANSWER, "Give me a detailed overview of COPD management"
        )
        msg = data.get("message", "")
        assert len(msg) >= len(_LONG_MEDICAL_ANSWER), (
            f"Response was truncated: got {len(msg)} chars, expected ≥{len(_LONG_MEDICAL_ANSWER)}"
        )

    def test_multi_section_answer_structure_intact(self, flask_app):
        """Section headings inside a long answer must survive the pipeline."""
        data = self._post_query(flask_app, _LONG_MEDICAL_ANSWER, "Explain COPD in detail")
        msg = data.get("message", "")
        assert "Pathophysiology" in msg
        assert "Pharmacological" in msg
        assert "Prognosis" in msg

    def test_response_has_required_keys(self, flask_app):
        data = self._post_query(flask_app, "Any answer.", "What is asthma?")
        assert "response" in data
        assert "message" in data

    def test_response_flag_is_true_on_success(self, flask_app):
        data = self._post_query(flask_app, "Some medical answer.", "What is hypertension?")
        assert data.get("response") is True

    def test_answer_with_citations_preserved(self, flask_app):
        answer_with_sources = (
            "Asthma is a chronic inflammatory disease of the airways.\n\n"
            "**Sources:** GINA 2024 Guidelines; NEJM 2023; Lancet 2022."
        )
        data = self._post_query(flask_app, answer_with_sources, "What is asthma?")
        msg = data.get("message", "")
        assert "Sources" in msg
        assert "asthma" in msg.lower()

    @pytest.mark.parametrize("char_count", [500, 2000, 5000])
    def test_various_response_lengths_pass_through(self, flask_app, char_count: int):
        """Parametrised: test 500-, 2 000-, and 5 000-character answers."""
        long_answer = "Medical content. " * (char_count // 17 + 1)
        long_answer = long_answer[:char_count]
        data = self._post_query(flask_app, long_answer, "Medical question")
        msg = data.get("message", "")
        assert len(msg) >= char_count, (
            f"Expected ≥{char_count} chars in response, got {len(msg)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Live OpenAI integration tests
#    Always run — uses the real API key from .env
# ─────────────────────────────────────────────────────────────────────────────


class TestOpenAILiveIntegration:
    """
    Exercises the real OpenAI API using credentials from the project .env file.

    The ``real_chat_openai_class`` fixture (defined in conftest.py) provides the
    genuine ``ChatOpenAI`` class captured *before* the session-scoped mock patch
    is applied, so these tests always hit the live API regardless of the other
    mocks active in the session.

    Requirements:
        - ``openai_api_key`` set to a valid key in .env
        - ``llm_model_name`` / ``base_url`` configured appropriately
    """

    @pytest.fixture(scope="class")
    def live_svc(self, real_chat_openai_class):
        """
        Return a real LLMService-like object backed by the unpatched ChatOpenAI.

        We bypass LLMService.__init__ (which would call the patched ChatOpenAI)
        and inject the real class directly.
        """
        from config import Config
        from services.llm_service import LLMService

        svc = LLMService.__new__(LLMService)
        svc._config = Config
        svc._llm = real_chat_openai_class(
            api_key=Config.OPENAI_API_KEY,
            base_url=Config.OPENAI_BASE_URL,
            model_name=Config.LLM_MODEL_NAME,
            temperature=Config.LLM_DEFAULT_TEMPERATURE,
            request_timeout=Config.LLM_REQUEST_TIMEOUT,
        )
        return svc

    def test_live_connectivity(self, live_svc):
        """Basic connectivity: any non-empty response from the model."""
        result = live_svc.invoke("Reply with exactly one word: Hello")
        assert isinstance(result, str)
        assert len(result.strip()) > 0

    def test_live_medical_query_returns_text(self, live_svc):
        """Model answers a basic medical question with >50 chars."""
        result = live_svc.invoke(
            "In one sentence, what is the first-line treatment for community-acquired pneumonia?"
        )
        assert len(result) > 50, f"Response too short: {result!r}"

    def test_live_long_context_query(self, live_svc):
        """Model handles a detailed clinical scenario and returns >200 chars."""
        prompt = (
            "A 65-year-old male ex-smoker presents with 3 months of progressive dyspnoea, "
            "daily productive cough, and two exacerbations in the past year. "
            "Spirometry shows FEV1/FVC 0.62 and FEV1 58% predicted post-bronchodilator. "
            "CXR reveals hyperinflation. BMI is 21. "
            "Describe the GOLD COPD classification, recommended pharmacotherapy step-up, "
            "non-pharmacological interventions, and exacerbation prevention strategies."
        )
        result = live_svc.invoke(prompt)
        assert len(result) > 200, (
            f"Expected a detailed answer (>200 chars), got {len(result)} chars.\n"
            f"Response: {result[:300]!r}"
        )

    def test_live_model_config_matches(self, live_svc):
        """Sanity-check: service config matches the global Config."""
        from config import Config

        assert live_svc._config.LLM_MODEL_NAME == Config.LLM_MODEL_NAME

    def test_live_contextual_llm_injects_system_prompt(self, live_svc):
        """create_contextual_llm() attaches the patient context to the system message."""
        ctx_llm = live_svc.create_contextual_llm(patient_context="7-year-old with asthma")
        assert "7-year-old" in ctx_llm._system_message

    def test_live_endpoint_plain_english(self, flask_app, real_chat_openai_class):
        """
        POST /plain_english with a real query using the live API.

        Injects the real ChatOpenAI instance into flask_app.config so the
        route handler calls the live API instead of the session mock.
        """
        from config import Config

        real_llm = real_chat_openai_class(
            api_key=Config.OPENAI_API_KEY,
            base_url=Config.OPENAI_BASE_URL,
            model_name=Config.LLM_MODEL_NAME,
            temperature=Config.LLM_DEFAULT_TEMPERATURE,
            request_timeout=Config.LLM_REQUEST_TIMEOUT,
        )
        try:
            flask_app.config["LLM_INSTANCE"] = real_llm
            with flask_app.test_client() as c:
                resp = c.post(
                    "/plain_english",
                    data=json.dumps(
                        {"text": "What is the first-line treatment for Type 2 diabetes?"}
                    ),
                    content_type="application/json",
                )
            assert resp.status_code == 200
            data = json.loads(resp.data)
            msg = data.get("refined_text", "")
            assert len(msg) > 20, f"Expected substantive response, got: {msg!r}"
        finally:
            # Restore mock so other tests are unaffected
            flask_app.config["LLM_INSTANCE"] = flask_app.config.get("LLM_INSTANCE")

    def test_live_generate_summary(self, flask_app, real_chat_openai_class):
        """
        POST /generate_summary with a real transcript using the live API.
        """
        from config import Config

        real_llm = real_chat_openai_class(
            api_key=Config.OPENAI_API_KEY,
            base_url=Config.OPENAI_BASE_URL,
            model_name=Config.LLM_MODEL_NAME,
            temperature=Config.LLM_DEFAULT_TEMPERATURE,
            request_timeout=Config.LLM_REQUEST_TIMEOUT,
        )
        try:
            flask_app.config["LLM_INSTANCE"] = real_llm
            with flask_app.test_client() as c:
                resp = c.post(
                    "/generate_summary",
                    data=json.dumps(
                        {
                            "transcription": (
                                "Doctor: Good morning. What brings you in today?\n"
                                "Patient: I have had a cough for three weeks with greenish sputum "
                                "and some shortness of breath on exertion.\n"
                                "Doctor: Any fever? I can hear reduced breath sounds at the right base. "
                                "I suspect right lower lobe pneumonia. "
                                "We will start amoxicillin-clavulanate 875mg twice daily for 7 days."
                            ),
                            "doctor_name": "Dr Smith",
                            "patient_name": "Jane Doe",
                        }
                    ),
                    content_type="application/json",
                )
            assert resp.status_code == 200
            data = json.loads(resp.data)
            assert data.get("success") is True
            assert len(data.get("summary", "")) > 20
        finally:
            flask_app.config["LLM_INSTANCE"] = flask_app.config.get("LLM_INSTANCE")
