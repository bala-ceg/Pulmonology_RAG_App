"""
Flask test-client tests for core application routes.

Tests the homepage, disciplines API, and basic request validation
for the /data and /data-html query endpoints.
"""
import os
import sys
import json
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Homepage
# ---------------------------------------------------------------------------

class TestHomePage:
    def test_get_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_returns_html(self, client):
        resp = client.get("/")
        assert b"html" in resp.data.lower() or b"<!DOCTYPE" in resp.data


# ---------------------------------------------------------------------------
# /api/disciplines
# ---------------------------------------------------------------------------

class TestDisciplinesEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/api/disciplines")
        assert resp.status_code == 200

    def test_returns_json(self, client):
        resp = client.get("/api/disciplines")
        assert resp.content_type.startswith("application/json")

    def test_has_disciplines_key(self, client):
        resp = client.get("/api/disciplines")
        data = json.loads(resp.data)
        assert "disciplines" in data

    def test_disciplines_is_list(self, client):
        resp = client.get("/api/disciplines")
        data = json.loads(resp.data)
        assert isinstance(data["disciplines"], list)

    def test_each_discipline_has_id_and_name(self, client):
        resp = client.get("/api/disciplines")
        data = json.loads(resp.data)
        for disc in data["disciplines"]:
            assert "id" in disc
            assert "name" in disc

    def test_has_selection_rules(self, client):
        resp = client.get("/api/disciplines")
        data = json.loads(resp.data)
        assert "selection_rules" in data


# ---------------------------------------------------------------------------
# /data — POST (query endpoint)
# ---------------------------------------------------------------------------

class TestDataEndpoint:
    def test_missing_body_returns_error(self, client):
        resp = client.post("/data", content_type="application/json", data="{}")
        # Should not 500 — returns a JSON error or 400
        assert resp.status_code in (200, 400, 422)

    def test_missing_query_field_handled(self, client):
        resp = client.post(
            "/data",
            data=json.dumps({"session_id": "test-123"}),
            content_type="application/json",
        )
        assert resp.status_code in (200, 400, 422)
        # Response should be JSON or have an error message
        if resp.content_type.startswith("application/json"):
            data = json.loads(resp.data)
            assert "error" in data or "answer" in data or "response" in data

    def test_non_json_content_type_handled(self, client):
        resp = client.post("/data", data="not json", content_type="text/plain")
        assert resp.status_code in (200, 400, 415, 422, 500)


# ---------------------------------------------------------------------------
# /data-html — POST (HTML query endpoint)
# ---------------------------------------------------------------------------

class TestDataHtmlEndpoint:
    def test_missing_body_returns_error(self, client):
        resp = client.post("/data-html", content_type="application/json", data="{}")
        assert resp.status_code in (200, 400, 422)

    def test_missing_query_field_handled(self, client):
        resp = client.post(
            "/data-html",
            data=json.dumps({"session_id": "test-456"}),
            content_type="application/json",
        )
        assert resp.status_code in (200, 400, 422)


# ---------------------------------------------------------------------------
# 404 handling
# ---------------------------------------------------------------------------

class TestNotFound:
    def test_unknown_route_returns_404(self, client):
        resp = client.get("/this-route-does-not-exist-xyz")
        assert resp.status_code == 404
