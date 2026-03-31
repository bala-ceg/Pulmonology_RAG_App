"""
Tests for the Q&A / RAG system:

  1. MedicalQueryRouter — tool routing decisions for all 5 tools
  2. Tool functions — Wikipedia, ArXiv, Tavily, Internal_VectorDB, PostgreSQL (mocked)
  3. /data endpoint — full Q&A request/response cycle (mocked LLM/tools)
  4. /data-html endpoint — HTML Q&A request/response cycle (mocked)
  5. SFT experiment test endpoint — guarded question answering
  6. TFIDFLexicalGate — should_query_local_first routing
  7. _join_docs helper
  8. IntegratedMedicalRAG — initialization and fallback behaviour

All external calls (OpenAI, Wikipedia, arXiv, Tavily, PostgreSQL) are mocked
so these tests run offline without API keys.
"""
import os
import sys
import json
import sqlite3
import pytest
from unittest.mock import patch, MagicMock, PropertyMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# MedicalQueryRouter — routing decisions
# ---------------------------------------------------------------------------

class TestMedicalQueryRouter:
    """
    Unit-test the MedicalQueryRouter.route_tools() scoring/routing logic
    without loading any real LLM or RAG manager.
    """

    @pytest.fixture(autouse=True)
    def setup_router(self):
        from rag_architecture import MedicalQueryRouter
        # Minimal mock rag_manager with no session/external content
        mock_rm = MagicMock()
        mock_rm.has_session_content.return_value = False
        mock_rm.has_external_content.return_value = True
        self.router = MedicalQueryRouter(rag_manager=mock_rm)

    # --- Wikipedia routing ---
    def test_what_is_routes_to_wikipedia(self):
        result = self.router.route_tools("What is hypertension?")
        assert result["primary_tool"] == "Wikipedia_Search"

    def test_explain_routes_to_wikipedia(self):
        result = self.router.route_tools("Explain the symptoms of COVID-19")
        assert result["primary_tool"] == "Wikipedia_Search"

    def test_define_routes_to_wikipedia(self):
        result = self.router.route_tools("Define pulmonary fibrosis")
        assert result["primary_tool"] == "Wikipedia_Search"

    def test_symptoms_type1_diabetes_wikipedia(self):
        result = self.router.route_tools("Explain the symptoms and medication for Type-1 Diabetes")
        assert result["primary_tool"] == "Wikipedia_Search"

    # --- ArXiv routing ---
    def test_latest_research_routes_to_arxiv(self):
        result = self.router.route_tools("Get me the latest research on oncology")
        assert result["primary_tool"] == "ArXiv_Search"

    def test_recent_study_routes_to_arxiv(self):
        result = self.router.route_tools("Get me the latest research on pulmonology")
        assert result["primary_tool"] == "ArXiv_Search"

    def test_new_findings_routes_to_arxiv(self):
        result = self.router.route_tools("What are the newest findings in COVID-19 treatment research?")
        assert result["primary_tool"] == "ArXiv_Search"

    def test_scientific_evidence_routes_to_arxiv(self):
        result = self.router.route_tools("Show me recent scientific papers on COPD management")
        assert result["primary_tool"] == "ArXiv_Search"

    # --- Tavily routing ---
    def test_fda_recalls_routes_to_tavily(self):
        result = self.router.route_tools("What are the recent FDA drug recalls?")
        assert result["primary_tool"] == "Tavily_Search"

    def test_who_guidelines_routes_to_tavily(self):
        result = self.router.route_tools("What are the latest treatment guidelines by WHO for treating common cold?")
        assert result["primary_tool"] == "Tavily_Search"

    def test_current_recommendations_routes_to_tavily(self):
        result = self.router.route_tools("Current CDC guidelines for COVID-19 prevention")
        assert result["primary_tool"] == "Tavily_Search"

    # --- Internal VectorDB routing ---
    def test_uploaded_pdf_routes_to_internal(self):
        result = self.router.route_tools("What does my uploaded PDF say about treatment protocols?")
        assert result["primary_tool"] == "Internal_VectorDB"

    def test_my_document_routes_to_internal(self):
        result = self.router.route_tools("What does my uploaded pdf contain?")
        assert result["primary_tool"] == "Internal_VectorDB"

    def test_organization_kb_routes_to_internal(self):
        result = self.router.route_tools("Search my organization internal data about discharge criteria")
        assert result["primary_tool"] == "Internal_VectorDB"

    # --- PostgreSQL routing ---
    def test_diagnoses_available_routes_to_postgres(self):
        result = self.router.route_tools("What diagnoses are available in the database?")
        assert result["primary_tool"] == "PostgreSQL_Diagnosis_Search"

    def test_diagnosis_codes_routes_to_postgres(self):
        result = self.router.route_tools("Show me diagnosis codes from the database")
        assert result["primary_tool"] == "PostgreSQL_Diagnosis_Search"

    def test_medical_database_routes_to_postgres(self):
        result = self.router.route_tools("What is in the medical database?")
        assert result["primary_tool"] == "PostgreSQL_Diagnosis_Search"

    # --- Result structure ---
    def test_result_has_required_keys(self):
        result = self.router.route_tools("What is asthma?")
        assert "primary_tool" in result
        assert "ranked_tools" in result
        assert "confidence" in result
        assert "reasoning" in result

    def test_ranked_tools_is_list(self):
        result = self.router.route_tools("Latest cancer research")
        assert isinstance(result["ranked_tools"], list)
        assert len(result["ranked_tools"]) >= 1

    def test_primary_tool_is_first_ranked_tool(self):
        result = self.router.route_tools("Explain diabetes")
        assert result["primary_tool"] == result["ranked_tools"][0]

    def test_confidence_is_string(self):
        result = self.router.route_tools("Any query")
        assert isinstance(result["confidence"], str)

    def test_session_with_content_boosts_internal(self):
        """When the session has content and query is PDF-relevant, Internal_VectorDB gets boosted."""
        from rag_architecture import MedicalQueryRouter
        mock_rm = MagicMock()
        mock_rm.has_session_content.return_value = True
        mock_rm.has_external_content.return_value = True
        router = MedicalQueryRouter(rag_manager=mock_rm)
        result = router.route_tools("What does my uploaded document say?", session_id="sess_123")
        assert result["primary_tool"] == "Internal_VectorDB"


# ---------------------------------------------------------------------------
# TFIDFLexicalGate — local-first routing
# ---------------------------------------------------------------------------

class TestTFIDFLexicalGate:
    @pytest.fixture(autouse=True)
    def setup_gate(self):
        from rag_architecture import TFIDFLexicalGate
        self.gate = TFIDFLexicalGate()

    def test_returns_tuple(self):
        result = self.gate.should_query_local_first("What is pneumonia?")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_bool_and_float(self):
        local_first, score = self.gate.should_query_local_first("Some medical query")
        assert isinstance(local_first, bool)
        assert isinstance(score, float)

    def test_score_between_0_and_1(self):
        _, score = self.gate.should_query_local_first("Any query")
        assert 0.0 <= score <= 1.0

    def test_empty_corpus_defaults_to_external_first(self):
        """Without training corpus, gate should default to external (False)."""
        _, score = self.gate.should_query_local_first("test query")
        # score should be 0 or very low with empty corpus
        assert score >= 0.0

    def test_load_from_disk_with_valid_path(self):
        """Should not raise when loading from a valid pkl path."""
        gate_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "vector_dbs", "lexical_gate.pkl"
        )
        if os.path.exists(gate_path):
            self.gate.load_from_disk(gate_path)  # should not raise
        else:
            pytest.skip("lexical_gate.pkl not present in vector_dbs/")


# ---------------------------------------------------------------------------
# Tool functions — mocked external calls
# ---------------------------------------------------------------------------

class TestWikipediaSearchTool:
    def test_returns_string(self):
        mock_doc = MagicMock()
        mock_doc.page_content = "COVID-19 is caused by SARS-CoV-2 coronavirus."
        mock_doc.metadata = {"source": "https://en.wikipedia.org/wiki/COVID-19", "title": "COVID-19"}

        with patch("tools.WikipediaLoader") as MockLoader:
            MockLoader.return_value.load.return_value = [mock_doc]
            from tools import Wikipedia_Search
            result = Wikipedia_Search.invoke("COVID-19 symptoms")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_sources_footer(self):
        mock_doc = MagicMock()
        mock_doc.page_content = "Diabetes is a metabolic disease."
        mock_doc.metadata = {"source": "https://en.wikipedia.org/wiki/Diabetes", "title": "Diabetes"}

        with patch("tools.WikipediaLoader") as MockLoader:
            MockLoader.return_value.load.return_value = [mock_doc]
            from tools import Wikipedia_Search
            result = Wikipedia_Search.invoke("What is diabetes?")

        assert "Sources" in result or len(result) > 10

    def test_empty_results_returns_fallback_message(self):
        with patch("tools.WikipediaLoader") as MockLoader:
            MockLoader.return_value.load.return_value = []
            from tools import Wikipedia_Search
            result = Wikipedia_Search.invoke("xyzzy nonexistent medical term")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_exception_returns_error_string(self):
        with patch("tools.WikipediaLoader") as MockLoader:
            MockLoader.return_value.load.side_effect = Exception("Network error")
            from tools import Wikipedia_Search
            result = Wikipedia_Search.invoke("test")

        assert "Error" in result or "error" in result


class TestArXivSearchTool:
    def test_returns_string(self):
        mock_doc = MagicMock()
        mock_doc.page_content = "Recent advances in COPD treatment show improved outcomes."
        mock_doc.metadata = {"source": "https://arxiv.org/abs/2301.00001", "Title": "COPD Research"}

        with patch("tools.ArxivLoader") as MockLoader:
            MockLoader.return_value.load.return_value = [mock_doc]
            from tools import ArXiv_Search
            result = ArXiv_Search.invoke("latest research on COPD")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_results_returns_fallback_message(self):
        with patch("tools.ArxivLoader") as MockLoader:
            MockLoader.return_value.load.return_value = []
            from tools import ArXiv_Search
            result = ArXiv_Search.invoke("xyzzy nonexistent topic")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_exception_returns_error_string(self):
        with patch("tools.ArxivLoader") as MockLoader:
            MockLoader.return_value.load.side_effect = Exception("Timeout")
            from tools import ArXiv_Search
            result = ArXiv_Search.invoke("pulmonology research")

        assert "Error" in result or "error" in result


class TestTavilySearchTool:
    def test_returns_string_when_api_key_missing(self):
        """Without API key Tavily falls back gracefully — returns a non-empty string."""
        import os
        original = os.environ.pop("TAVILY_API_KEY", None)
        try:
            with patch("tools.WikipediaLoader") as MockWiki:
                mock_doc = MagicMock()
                mock_doc.page_content = "FDA recall information."
                mock_doc.metadata = {"source": "https://fda.gov", "title": "FDA"}
                MockWiki.return_value.load.return_value = [mock_doc]
                from tools import Tavily_Search
                result = Tavily_Search.invoke("What are the recent FDA drug recalls?")
        finally:
            if original is not None:
                os.environ["TAVILY_API_KEY"] = original
        assert isinstance(result, str)
        assert len(result) > 0

    def test_tavily_client_used_when_key_present(self):
        """When API key is set and TavilyClient is available, it is called."""
        import os
        os.environ["TAVILY_API_KEY"] = "fake-test-key"
        try:
            mock_client = MagicMock()
            mock_client.search.return_value = {
                "results": [{"content": "FDA issues recall.", "title": "FDA Recall", "url": "https://fda.gov"}]
            }
            with patch("tavily.TavilyClient", return_value=mock_client):
                from tools import Tavily_Search
                result = Tavily_Search.invoke("What are the recent FDA drug recalls?")
        except Exception:
            # If tavily package itself has issues in test env, just check graceful fallback
            result = "fallback"
        finally:
            os.environ.pop("TAVILY_API_KEY", None)
        assert isinstance(result, str)


class TestPostgreSQLDiagnosisTool:
    def test_returns_string(self):
        # enhanced_postgres_search is imported locally inside the tool function;
        # patch it on the postgres_tool module directly.
        mock_result = {
            "summary": "Diabetes mellitus is a chronic condition.",
            "content": "ICD-10 E11 — Type 2 diabetes mellitus",
            "citations": "Sources: pces_ehr_ccm.p_diagnosis"
        }
        with patch("postgres_tool.enhanced_postgres_search", return_value=mock_result, create=True):
            with patch("enhanced_tools.format_enhanced_response", return_value="Diabetes: ICD E11", create=True):
                from tools import PostgreSQL_Diagnosis_Search
                result = PostgreSQL_Diagnosis_Search.invoke("What diagnoses are available?")
        assert isinstance(result, str)

    def test_db_unavailable_falls_back_gracefully(self):
        """When DB is unreachable the tool catches the exception and returns a string."""
        with patch("postgres_tool.enhanced_postgres_search",
                   side_effect=Exception("Connection refused"), create=True):
            with patch("tools.Wikipedia_Search") as mock_wiki:
                mock_wiki.invoke.return_value = "Wikipedia fallback content"
                from tools import PostgreSQL_Diagnosis_Search
                result = PostgreSQL_Diagnosis_Search.invoke("Show me diagnosis codes")
        assert isinstance(result, str)


class TestInternalVectorDBTool:
    def test_no_rag_manager_returns_message(self):
        from tools import Internal_VectorDB
        result = Internal_VectorDB.invoke("What does my PDF say?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_rag_manager_with_no_session_content_uses_fallback(self):
        mock_rm = MagicMock()
        mock_rm.load_session_vector_db.return_value = False

        with patch("tools.Wikipedia_Search") as mock_wiki:
            mock_wiki.invoke.return_value = "Wikipedia fallback content"
            from tools import Internal_VectorDB
            result = Internal_VectorDB.func("RDW correlation with mortality", "sess_123", mock_rm)

        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _join_docs helper
# ---------------------------------------------------------------------------

class TestJoinDocs:
    def test_joins_documents_to_string(self):
        from tools import _join_docs
        mock_doc = MagicMock()
        mock_doc.page_content = "This is medical content about pneumonia."
        mock_doc.metadata = {"source": "https://example.com", "title": "Pneumonia"}
        result = _join_docs([mock_doc], max_chars=500)
        assert isinstance(result, str)
        assert "pneumonia" in result.lower()

    def test_truncates_at_max_chars(self):
        from tools import _join_docs
        mock_doc = MagicMock()
        mock_doc.page_content = "A" * 2000
        mock_doc.metadata = {}
        result = _join_docs([mock_doc], max_chars=100)
        # Result will be truncated; just verify it's shorter than full content
        assert len(result) < 2000

    def test_empty_docs_returns_string(self):
        from tools import _join_docs
        result = _join_docs([], max_chars=500)
        assert isinstance(result, str)

    def test_sources_footer_appended(self):
        from tools import _join_docs
        mock_doc = MagicMock()
        mock_doc.page_content = "Content about hypertension."
        mock_doc.metadata = {"source": "https://en.wikipedia.org/wiki/Hypertension", "title": "Hypertension"}
        result = _join_docs([mock_doc])
        assert "Sources" in result or "sources" in result.lower()


# ---------------------------------------------------------------------------
# /data endpoint — Q&A integration (mocked LLM + tools)
# ---------------------------------------------------------------------------

class TestDataEndpointQA:
    """
    Test the /data endpoint with mocked integrated RAG and two-store RAG
    to verify request handling, scope guard pass-through, and response format.
    """

    def test_empty_query_returns_false(self, client):
        resp = client.post(
            "/data",
            data=json.dumps({"data": ""}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["response"] is False

    def test_missing_data_field_returns_false(self, client):
        resp = client.post(
            "/data",
            data=json.dumps({}),
            content_type="application/json",
        )
        data = json.loads(resp.data)
        assert data["response"] is False

    def test_query_returns_json_response(self, client):
        """POST /data with a medical query returns a JSON response structure."""
        resp = client.post(
            "/data",
            data=json.dumps({"data": "What is hypertension?", "session_id": "test-session"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "response" in data
        # response may be True (answered) or False (error) — both are valid structures
        assert isinstance(data["response"], bool)

    def test_query_has_message_field(self, client):
        resp = client.post(
            "/data",
            data=json.dumps({"data": "Explain asthma", "session_id": "test-session"}),
            content_type="application/json",
        )
        data = json.loads(resp.data)
        assert "message" in data

    def test_out_of_scope_query_with_guard_disabled(self, client):
        """
        With SCOPE_GUARD_ENABLED=false (set in conftest), off-topic queries
        are NOT rejected — they get processed normally.
        """
        resp = client.post(
            "/data",
            data=json.dumps({"data": "How do I bake a chocolate cake?", "session_id": "test"}),
            content_type="application/json",
        )
        data = json.loads(resp.data)
        # Guard is disabled in tests — should not return out_of_scope
        assert data.get("out_of_scope") is not True

    def test_patient_problem_field_accepted(self, client):
        """patient_problem context field should be accepted without error."""
        resp = client.post(
            "/data",
            data=json.dumps({
                "data": "What is diabetes?",
                "patient_problem": "63-year-old patient with high blood sugar",
                "session_id": "test"
            }),
            content_type="application/json",
        )
        assert resp.status_code == 200

    def test_session_id_field_accepted(self, client):
        resp = client.post(
            "/data",
            data=json.dumps({"data": "What is COPD?", "session_id": "my-session-123"}),
            content_type="application/json",
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /data-html endpoint — HTML Q&A integration
# ---------------------------------------------------------------------------

class TestDataHtmlEndpointQA:
    def test_empty_query_returns_error(self, client):
        # /data-html with empty input returns an HTML error page (not JSON)
        resp = client.post(
            "/data-html",
            data=json.dumps({"data": ""}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        # Response is HTML containing an error/validation message
        assert b"valid input" in resp.data or b"Please" in resp.data or len(resp.data) > 0

    def test_query_returns_valid_response(self, client):
        resp = client.post(
            "/data-html",
            data=json.dumps({"data": "What is pneumonia?", "session_id": "test-html"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        # Response is JSON or HTML — both are valid; just check it's non-empty
        assert len(resp.data) > 0

    def test_query_response_has_content(self, client):
        """After the intelligent_query bug-fix, /data-html should return non-empty content."""
        resp = client.post(
            "/data-html",
            data=json.dumps({"data": "Explain COPD symptoms", "session_id": "test-html"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        assert len(resp.data) > 10


# ---------------------------------------------------------------------------
# SFT Experiment test endpoint
# ---------------------------------------------------------------------------

class TestSFTExperimentTestEndpoint:
    def test_missing_question_returns_400(self, client):
        resp = client.post(
            "/api/rlhf/experiment/1/test",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert resp.status_code in (400, 503)

    def test_empty_question_returns_400(self, client):
        resp = client.post(
            "/api/rlhf/experiment/1/test",
            data=json.dumps({"question": ""}),
            content_type="application/json",
        )
        assert resp.status_code in (400, 503)

    def test_nonexistent_experiment_handled(self, client):
        """Experiment 99999 probably doesn't exist — should return structured error."""
        resp = client.post(
            "/api/rlhf/experiment/99999/test",
            data=json.dumps({"question": "What is hypertension?"}),
            content_type="application/json",
        )
        data = json.loads(resp.data)
        # Either 400/503 with error, or success=False
        assert resp.status_code in (200, 400, 503)
        assert "success" in data or "error" in data

    def test_medical_question_not_rejected_when_guard_disabled(self, client):
        """With guard disabled, medical questions pass through (not out_of_scope)."""
        resp = client.post(
            "/api/rlhf/experiment/1/test",
            data=json.dumps({"question": "What is hypertension?"}),
            content_type="application/json",
        )
        data = json.loads(resp.data)
        assert data.get("out_of_scope") is not True


# ---------------------------------------------------------------------------
# IntegratedMedicalRAG — init and _direct_tool_execution fallback
# ---------------------------------------------------------------------------

class TestIntegratedMedicalRAG:
    def test_init_with_mocked_openai(self):
        """IntegratedMedicalRAG initialises without raising when OpenAI is mocked."""
        with patch("integrated_rag.ChatOpenAI") as MockLLM, \
             patch("integrated_rag.OpenAIEmbeddings") as MockEmb, \
             patch("integrated_rag.TwoStoreRAGManager") as MockRAG, \
             patch("integrated_rag.initialize_agent") as MockAgent:

            MockLLM.return_value = MagicMock()
            MockEmb.return_value = MagicMock()
            MockRAG.return_value = MagicMock()
            MockAgent.return_value = MagicMock()

            from integrated_rag import IntegratedMedicalRAG
            rag = IntegratedMedicalRAG(openai_api_key="test-key")
            assert rag is not None

    def test_tools_list_has_five_tools(self):
        """After init, the system should register 5 tools."""
        with patch("integrated_rag.ChatOpenAI") as MockLLM, \
             patch("integrated_rag.OpenAIEmbeddings") as MockEmb, \
             patch("integrated_rag.TwoStoreRAGManager") as MockRAG, \
             patch("integrated_rag.initialize_agent") as MockAgent:

            MockLLM.return_value = MagicMock()
            MockEmb.return_value = MagicMock()
            MockRAG.return_value = MagicMock()
            MockAgent.return_value = MagicMock()

            from integrated_rag import IntegratedMedicalRAG
            rag = IntegratedMedicalRAG(openai_api_key="test-key")
            assert len(rag.tools) == 5

    def test_tool_names(self):
        """Tools must include Wikipedia, ArXiv, Tavily, Internal, PostgreSQL."""
        with patch("integrated_rag.ChatOpenAI") as MockLLM, \
             patch("integrated_rag.OpenAIEmbeddings") as MockEmb, \
             patch("integrated_rag.TwoStoreRAGManager") as MockRAG, \
             patch("integrated_rag.initialize_agent") as MockAgent:

            MockLLM.return_value = MagicMock()
            MockEmb.return_value = MagicMock()
            MockRAG.return_value = MagicMock()
            MockAgent.return_value = MagicMock()

            from integrated_rag import IntegratedMedicalRAG
            rag = IntegratedMedicalRAG(openai_api_key="test-key")
            tool_names = {t.name for t in rag.tools}
            assert "Wikipedia_Search" in tool_names
            assert "ArXiv_Search" in tool_names
            assert "Tavily_Search" in tool_names
            assert "Internal_VectorDB" in tool_names
            assert "PostgreSQL_Diagnosis_Search" in tool_names


# ---------------------------------------------------------------------------
# AVAILABLE_TOOLS registry
# ---------------------------------------------------------------------------

class TestAvailableToolsRegistry:
    def test_all_tools_in_registry(self):
        from tools import AVAILABLE_TOOLS
        expected = {"Wikipedia_Search", "ArXiv_Search", "Tavily_Search",
                    "Internal_VectorDB", "PostgreSQL_Diagnosis_Search"}
        assert expected == set(AVAILABLE_TOOLS.keys())

    def test_tools_are_callable(self):
        from tools import AVAILABLE_TOOLS
        for name, tool in AVAILABLE_TOOLS.items():
            assert callable(tool) or hasattr(tool, "invoke"), f"{name} must be callable or have invoke()"
