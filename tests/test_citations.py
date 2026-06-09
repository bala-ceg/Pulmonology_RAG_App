"""
Unit tests for services/citation_service.py — Phase 2-B: Attach Citations (Step 08)

Run:  python tests/test_citations.py
"""

from __future__ import annotations

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.citation_service import (
    SOURCE_LABELS,
    SOURCE_RELIABILITY,
    _extract_metadata,
    _infer_source_type,
    _rank_and_select,
    _build_structured_citations,
    _label_from_raw_citation,
    _type_from_raw_citation,
    build_citations_html,
    attach_citations,
)


# ── Helpers ──────────────────────────────────────────────────────────────────
def _make_source_doc(*, dept="Cardiology", source="heart_guide.pdf",
                     relevance="73%", page="5", url="", excerpt="Sample text",
                     source_type=None):
    """Build a source document dict matching integrated_rag.py format."""
    doc = {
        "header": f"Document 1  |  Dept: {dept}  |  Source: {source}  |  Relevance: {relevance}",
        "excerpt": excerpt,
        "fields": {
            "Dept": dept,
            "Source": source,
            "Relevance": relevance,
            "Page": page,
        },
    }
    if url:
        doc["fields"]["URL"] = url
    if source_type:
        doc["metadata"] = {"source_type": source_type}
    return doc


# ── Test Source Label Mapping ────────────────────────────────────────────────
class TestSourceLabels(unittest.TestCase):
    """Step 5: Standardised source labels."""

    def test_all_tool_names_have_labels(self):
        tool_names = [
            "Pinecone_KB_Search", "Internal_VectorDB", "AdHocRAG_Search",
            "PostgreSQL_Diagnosis_Search", "ArXiv_Search", "Tavily_Search",
            "Wikipedia_Search",
        ]
        for name in tool_names:
            self.assertIn(name, SOURCE_LABELS, f"Missing label for tool: {name}")

    def test_all_source_types_have_labels(self):
        source_types = ["wikipedia", "arxiv", "tavily", "internal", "pinecone",
                        "main_rag", "adhoc_rag", "postgres", "lora_model"]
        for st in source_types:
            self.assertIn(st, SOURCE_LABELS, f"Missing label for source_type: {st}")

    def test_labels_are_human_readable(self):
        for key, label in SOURCE_LABELS.items():
            self.assertTrue(len(label) > 5, f"Label too short for {key}: {label}")
            self.assertFalse(label.startswith("_"), f"Label looks internal: {label}")


# ── Test Source Reliability Scores ───────────────────────────────────────────
class TestSourceReliability(unittest.TestCase):
    """Production enhancement: reliability scoring."""

    def test_all_labels_have_reliability(self):
        expected_labels = set(SOURCE_LABELS.values())
        for label in expected_labels:
            self.assertIn(label, SOURCE_RELIABILITY, f"No reliability for: {label}")

    def test_scores_in_range(self):
        for label, score in SOURCE_RELIABILITY.items():
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 100)

    def test_arxiv_most_reliable(self):
        self.assertEqual(SOURCE_RELIABILITY["Clinical Research (arXiv)"], 95)

    def test_wiki_less_reliable_than_ehr(self):
        self.assertLess(
            SOURCE_RELIABILITY["Medical Reference (Wikipedia)"],
            SOURCE_RELIABILITY["Patient Medical Record (EHR)"],
        )


# ── Test Metadata Extraction (Step 2) ────────────────────────────────────────
class TestExtractMetadata(unittest.TestCase):

    def test_basic_extraction(self):
        doc = _make_source_doc(dept="Cardiology", source="heart.pdf",
                               relevance="73%", page="5")
        meta = _extract_metadata(doc)
        self.assertEqual(meta["department"], "Cardiology")
        self.assertEqual(meta["source_raw"], "heart.pdf")
        self.assertEqual(meta["page"], "5")
        self.assertEqual(meta["relevance"], 73)

    def test_percentage_string_parsing(self):
        doc = _make_source_doc(relevance="85%")
        meta = _extract_metadata(doc)
        self.assertEqual(meta["relevance"], 85)

    def test_float_score_parsing(self):
        doc = {"fields": {}, "metadata": {"score": 0.73}, "excerpt": ""}
        meta = _extract_metadata(doc)
        self.assertEqual(meta["relevance"], 73)

    def test_missing_fields_graceful(self):
        doc = {"fields": {}, "excerpt": "test"}
        meta = _extract_metadata(doc)
        self.assertEqual(meta["department"], "")
        self.assertEqual(meta["source_raw"], "")
        self.assertEqual(meta["page"], "")
        self.assertEqual(meta["relevance"], 0)

    def test_url_extraction(self):
        doc = _make_source_doc(url="https://example.com/paper.pdf")
        meta = _extract_metadata(doc)
        self.assertEqual(meta["url"], "https://example.com/paper.pdf")

    def test_source_type_from_metadata(self):
        doc = _make_source_doc(source_type="arxiv")
        meta = _extract_metadata(doc)
        self.assertEqual(meta["source_type"], "arxiv")
        self.assertEqual(meta["source_label"], "Clinical Research (arXiv)")

    def test_reliability_assigned(self):
        doc = _make_source_doc(source_type="arxiv")
        meta = _extract_metadata(doc)
        self.assertEqual(meta["reliability"], 95)


# ── Test Source Type Inference ────────────────────────────────────────────────
class TestInferSourceType(unittest.TestCase):

    def test_pinecone(self):
        self.assertEqual(_infer_source_type("PCES_cardiology", {}), "pinecone")

    def test_arxiv(self):
        self.assertEqual(_infer_source_type("arxiv:2301.12345", {}), "arxiv")

    def test_wikipedia(self):
        self.assertEqual(_infer_source_type("Wikipedia: Glaucoma", {}), "wikipedia")

    def test_tavily(self):
        self.assertEqual(_infer_source_type("web search result", {}), "tavily")

    def test_postgres(self):
        self.assertEqual(_infer_source_type("EHR diagnosis DB", {}), "postgres")

    def test_fallback_internal(self):
        self.assertEqual(_infer_source_type("unknown_source.pdf", {}), "internal")

    def test_header_fallback(self):
        doc = {"header": "PCES Pinecone KB result"}
        self.assertEqual(_infer_source_type("unknown.pdf", doc), "pinecone")


# ── Test Ranking & Selection (Step 3) ────────────────────────────────────────
class TestRankAndSelect(unittest.TestCase):

    def test_ranking_by_composite_score(self):
        docs = [
            {"source_raw": "a.pdf", "page": "1", "relevance": 50, "reliability": 90},
            {"source_raw": "b.pdf", "page": "1", "relevance": 90, "reliability": 70},
        ]
        ranked = _rank_and_select(docs)
        # b: 90*0.7 + 70*0.3 = 63+21 = 84
        # a: 50*0.7 + 90*0.3 = 35+27 = 62
        self.assertEqual(ranked[0]["source_raw"], "b.pdf")
        self.assertEqual(ranked[1]["source_raw"], "a.pdf")

    def test_deduplication(self):
        docs = [
            {"source_raw": "same.pdf", "page": "5", "relevance": 80, "reliability": 90},
            {"source_raw": "same.pdf", "page": "5", "relevance": 75, "reliability": 90},
        ]
        ranked = _rank_and_select(docs)
        self.assertEqual(len(ranked), 1)

    def test_same_namespace_collapsed_to_highest_score(self):
        """Multiple Pinecone docs from same namespace collapse to the best one."""
        docs = [
            {"source_raw": "PCES_cardiology", "page": "", "relevance": 58, "reliability": 90},
            {"source_raw": "PCES_cardiology", "page": "", "relevance": 57, "reliability": 90},
            {"source_raw": "PCES_cardiology", "page": "", "relevance": 56, "reliability": 90},
            {"source_raw": "PCES_cardiology", "page": "", "relevance": 55, "reliability": 90},
            {"source_raw": "PCES_cardiology", "page": "", "relevance": 54, "reliability": 90},
        ]
        ranked = _rank_and_select(docs)
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0]["relevance"], 58)  # highest score kept

    def test_max_citations_limit(self):
        docs = [
            {"source_raw": f"doc{i}.pdf", "page": str(i), "relevance": 50, "reliability": 70}
            for i in range(10)
        ]
        ranked = _rank_and_select(docs, max_citations=3)
        self.assertEqual(len(ranked), 3)

    def test_empty_input(self):
        self.assertEqual(_rank_and_select([]), [])


# ── Test Label From Raw Citation ─────────────────────────────────────────────
class TestLabelFromRawCitation(unittest.TestCase):

    def test_arxiv(self):
        self.assertEqual(_label_from_raw_citation("arXiv: 2301.12345"), "Clinical Research (arXiv)")

    def test_wikipedia(self):
        self.assertEqual(_label_from_raw_citation("Wikipedia: Glaucoma"), "Medical Reference (Wikipedia)")

    def test_pinecone(self):
        self.assertEqual(
            _label_from_raw_citation("[PCES Pinecone KB] Dept: Cardiology"),
            "Organisation Knowledge Base",
        )

    def test_web(self):
        self.assertEqual(_label_from_raw_citation("Web: some result"), "Web Research")

    def test_ehr(self):
        self.assertEqual(
            _label_from_raw_citation("PostgreSQL EHR Database"),
            "Patient Medical Record (EHR)",
        )

    def test_fallback(self):
        self.assertEqual(
            _label_from_raw_citation("Some random source"),
            "Hospital Knowledge Base",
        )


# ── Test Type From Raw Citation ──────────────────────────────────────────────
class TestTypeFromRawCitation(unittest.TestCase):

    def test_arxiv(self):
        self.assertEqual(_type_from_raw_citation("arXiv paper"), "arxiv")

    def test_wikipedia(self):
        self.assertEqual(_type_from_raw_citation("Wikipedia article"), "wikipedia")

    def test_tavily(self):
        self.assertEqual(_type_from_raw_citation("Tavily web search"), "tavily")

    def test_pinecone(self):
        self.assertEqual(_type_from_raw_citation("PCES Pinecone KB"), "pinecone")

    def test_postgres(self):
        self.assertEqual(_type_from_raw_citation("EHR diagnosis"), "postgres")

    def test_fallback(self):
        self.assertEqual(_type_from_raw_citation("unknown"), "internal")


# ── Test Build Structured Citations (Steps 4, 6, 8, 9) ──────────────────────
class TestBuildStructuredCitations(unittest.TestCase):

    def test_basic_build(self):
        ranked = [{
            "source_label": "Clinical Research (arXiv)",
            "source_raw": "arxiv:2301.12345",
            "source_type": "arxiv",
            "department": "",
            "page": "",
            "url": "https://arxiv.org/abs/2301.12345",
            "title": "A Study",
            "relevance": 80,
            "reliability": 95,
            "excerpt": "Important findings...",
            "composite_score": 84.5,
        }]
        result = _build_structured_citations(ranked, [], None)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["source_label"], "Clinical Research (arXiv)")
        self.assertEqual(result[0]["url"], "https://arxiv.org/abs/2301.12345")

    def test_raw_citations_appended(self):
        result = _build_structured_citations(
            [],
            ["Source: Wikipedia", "arXiv: some paper"],
            None,
        )
        self.assertEqual(len(result), 2)

    def test_duplicate_raw_citation_filtered(self):
        ranked = [{
            "source_label": "Medical Reference (Wikipedia)",
            "source_raw": "Wikipedia: Glaucoma",
            "source_type": "wikipedia",
            "department": "",
            "page": "",
            "url": "",
            "title": "Glaucoma",
            "relevance": 70,
            "reliability": 70,
            "excerpt": "",
            "composite_score": 70,
        }]
        result = _build_structured_citations(
            ranked,
            ["Wikipedia: Glaucoma"],  # duplicate
            None,
        )
        # Raw citation should be filtered as duplicate
        self.assertEqual(len(result), 1)

    def test_pinecone_formatted_raw_citation_filtered(self):
        """'[PCES Pinecone KB] Dept:... Source: PCES_cardiology' must be suppressed
        when 'pces_cardiology' is already in structured citations."""
        ranked = [{
            "source_label": "Organisation Knowledge Base",
            "source_raw": "PCES_cardiology",
            "source_type": "pinecone",
            "department": "Cardiology",
            "page": "",
            "url": "",
            "title": "",
            "relevance": 58,
            "reliability": 90,
            "excerpt": "",
            "composite_score": 67.6,
        }]
        raw = [
            "[PCES Pinecone KB] Dept: Cardiology | Source: PCES_cardiology | Relevance: 58%",
            "[PCES Pinecone KB] Dept: Cardiology | Source: PCES_cardiology | Relevance: 57%",
            "Source: PCES Pinecone Knowledge Base",
        ]
        result = _build_structured_citations(ranked, raw, None)
        # All three raw citations are duplicates of the structured entry — only 1 total
        self.assertEqual(len(result), 1)

    def test_generic_pinecone_raw_suppressed_by_type(self):
        """Generic 'Source: PCES Pinecone Knowledge Base' without URL is suppressed
        when a structured Pinecone entry already exists."""
        ranked = [{
            "source_label": "Organisation Knowledge Base",
            "source_raw": "namespace_xyz",
            "source_type": "pinecone",
            "department": "Neurology",
            "page": "",
            "url": "",
            "title": "",
            "relevance": 60,
            "reliability": 90,
            "excerpt": "",
            "composite_score": 69.0,
        }]
        result = _build_structured_citations(
            ranked,
            ["Source: PCES Pinecone Knowledge Base"],
            None,
        )
        self.assertEqual(len(result), 1)

    def test_lora_attribution(self):
        lora_info = {
            "used": True,
            "department": "Ophthalmology",
            "model_version": "v1.0",
        }
        result = _build_structured_citations([], [], lora_info)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["source_type"], "lora_model")
        self.assertEqual(result[0]["source_label"], "Department AI Model")
        self.assertEqual(result[0]["department"], "Ophthalmology")

    def test_lora_not_used(self):
        lora_info = {"used": False}
        result = _build_structured_citations([], [], lora_info)
        self.assertEqual(len(result), 0)

    def test_lora_none(self):
        result = _build_structured_citations([], [], None)
        self.assertEqual(len(result), 0)


# ── Test Build Citations HTML (Step 7) ───────────────────────────────────────
class TestBuildCitationsHTML(unittest.TestCase):

    def test_confidence_badge_high(self):
        html = build_citations_html([], confidence_score=90)
        self.assertIn("90%", html)
        self.assertIn("High", html)
        self.assertIn("#28a745", html)

    def test_confidence_badge_medium(self):
        html = build_citations_html([], confidence_score=70)
        self.assertIn("Medium", html)
        self.assertIn("#fd7e14", html)

    def test_confidence_badge_low(self):
        html = build_citations_html([], confidence_score=40)
        self.assertIn("Low", html)
        self.assertIn("#dc3545", html)

    def test_no_confidence_badge_when_zero(self):
        html = build_citations_html([], confidence_score=0)
        self.assertNotIn("Confidence:", html)

    def test_citation_list_rendered(self):
        citations = [{
            "source_label": "Clinical Research (arXiv)",
            "source": "arxiv:2301.12345",
            "source_type": "arxiv",
        }]
        html = build_citations_html(citations)
        self.assertIn("Clinical Research (arXiv)", html)
        self.assertIn("arxiv:2301.12345", html)

    def test_url_creates_link(self):
        citations = [{
            "source_label": "Web Research",
            "source": "some site",
            "source_type": "tavily",
            "url": "https://example.com",
        }]
        html = build_citations_html(citations)
        self.assertIn('href="https://example.com"', html)

    def test_source_document_cards_with_excerpts(self):
        """Compact list shows dept/match/reliability inline — no card section."""
        citations = [{
            "source_label": "Organisation Knowledge Base",
            "source": "heart.pdf",
            "source_type": "pinecone",
            "department": "Cardiology",
            "page": "5",
            "relevance_pct": 80,
            "reliability_pct": 90,
            "excerpt": "The heart is a muscular organ",
        }]
        html = build_citations_html(citations)
        # Card section should NOT appear — compact list only
        self.assertNotIn("Source Documents", html)
        # Compact list should still show key metadata inline
        self.assertIn("Cardiology", html)
        self.assertIn("80% match", html)
        self.assertIn("90%", html)

    def test_no_document_cards_without_excerpts(self):
        citations = [{
            "source_label": "Web Research",
            "source": "some site",
            "source_type": "tavily",
        }]
        html = build_citations_html(citations)
        self.assertNotIn("Source Documents", html)

    def test_hyperlink_rendered_when_url_present(self):
        """Source with a URL should render as a hyperlink in the compact list."""
        citations = [{
            "source_label": "Clinical Research (arXiv)",
            "source": "Glaucoma treatment meta-analysis",
            "source_type": "arxiv",
            "url": "https://arxiv.org/abs/2301.12345",
            "relevance_pct": 75,
            "reliability_pct": 95,
        }]
        html = build_citations_html(citations)
        self.assertIn('<a href="https://arxiv.org/abs/2301.12345"', html)
        self.assertIn("Glaucoma treatment meta-analysis", html)

    def test_no_hyperlink_without_url(self):
        """Pinecone source without URL shows 🏢 Internal badge instead of anchor."""
        citations = [{
            "source_label": "Organisation Knowledge Base",
            "source": "PCES_cardiology",
            "source_type": "pinecone",
        }]
        html = build_citations_html(citations)
        self.assertNotIn("<a href=", html)
        self.assertIn("🏢 Internal", html)

    def test_non_internal_no_url_no_badge(self):
        """Non-internal source without URL gets plain text (no badge, no link)."""
        citations = [{
            "source_label": "Web Research",
            "source": "Some web result",
            "source_type": "tavily",
        }]
        html = build_citations_html(citations)
        self.assertNotIn("<a href=", html)
        self.assertNotIn("🏢 Internal", html)


# ── Test Main Orchestrator (attach_citations) ────────────────────────────────
class TestAttachCitations(unittest.TestCase):

    def test_basic_attach(self):
        docs = [_make_source_doc(dept="Cardiology", source="heart.pdf", relevance="80%")]
        result = attach_citations(
            response="The heart...",
            source_documents=docs,
            citations_raw=["Source: Wikipedia"],
            confidence_score=85,
            department="Cardiology",
        )
        self.assertIn("structured_citations", result)
        self.assertIn("citations_html", result)
        self.assertGreater(result["source_count"], 0)
        self.assertIn("top_source_label", result)
        self.assertGreater(result["avg_reliability"], 0)

    def test_empty_inputs(self):
        result = attach_citations(
            response="Some answer",
            source_documents=[],
            citations_raw=[],
        )
        self.assertEqual(result["source_count"], 0)
        self.assertEqual(result["structured_citations"], [])

    def test_with_lora_info(self):
        result = attach_citations(
            response="Answer",
            source_documents=[],
            citations_raw=[],
            lora_info={"used": True, "department": "Ophthalmology", "model_version": "v1.0"},
        )
        self.assertEqual(result["source_count"], 1)
        self.assertEqual(result["structured_citations"][0]["source_type"], "lora_model")

    def test_html_contains_confidence(self):
        result = attach_citations(
            response="Answer",
            source_documents=[_make_source_doc()],
            citations_raw=[],
            confidence_score=90,
        )
        self.assertIn("90%", result["citations_html"])

    def test_graceful_error_handling(self):
        # Pass completely invalid data — should not crash
        result = attach_citations(
            response=None,
            source_documents=[{"invalid": True}],
            citations_raw=None,
        )
        # Should return empty result, not crash
        self.assertIn("structured_citations", result)

    def test_multiple_docs_ranked(self):
        docs = [
            _make_source_doc(source="low.pdf", relevance="30%"),
            _make_source_doc(source="high.pdf", relevance="90%"),
            _make_source_doc(source="mid.pdf", relevance="60%"),
        ]
        result = attach_citations(
            response="Answer",
            source_documents=docs,
            citations_raw=[],
        )
        # Highest relevance should be first in structured
        structured = result["structured_citations"]
        self.assertGreater(len(structured), 0)
        # First should have higher composite score than last
        self.assertGreaterEqual(
            structured[0].get("composite_score", 0),
            structured[-1].get("composite_score", 0),
        )


# ── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    unittest.main(verbosity=2)
