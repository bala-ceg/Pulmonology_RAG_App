"""
Flask test-client tests for RLHF API endpoints.

Uses the Flask test client — no running server required.
Heavy dependencies (ChromaDB, SBERT, PostgreSQL pool) are already
handled by conftest.py setting SCOPE_GUARD_ENABLED=false and using
a test key for OpenAI.
"""
import os
import sys
import json
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# /admin/rlhf
# ---------------------------------------------------------------------------

class TestAdminRLHFPage:
    def test_admin_page_returns_200(self, client):
        resp = client.get("/admin/rlhf")
        assert resp.status_code == 200

    def test_admin_page_is_html(self, client):
        resp = client.get("/admin/rlhf")
        assert b"text/html" in resp.content_type.encode() or b"<!DOCTYPE" in resp.data or b"<html" in resp.data.lower()

    def test_admin_page_contains_experiments(self, client):
        resp = client.get("/admin/rlhf")
        assert b"experiments" in resp.data.lower() or b"Experiments" in resp.data


# ---------------------------------------------------------------------------
# /api/rlhf/departments
# ---------------------------------------------------------------------------

class TestDepartmentsEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/api/rlhf/departments")
        assert resp.status_code == 200

    def test_returns_json(self, client):
        resp = client.get("/api/rlhf/departments")
        data = json.loads(resp.data)
        assert isinstance(data, dict)

    def test_success_flag_true(self, client):
        resp = client.get("/api/rlhf/departments")
        data = json.loads(resp.data)
        assert data.get("success") is True

    def test_departments_list_present(self, client):
        resp = client.get("/api/rlhf/departments")
        data = json.loads(resp.data)
        assert "departments" in data
        assert isinstance(data["departments"], list)

    def test_dentistry_in_departments(self, client):
        resp = client.get("/api/rlhf/departments")
        data = json.loads(resp.data)
        assert "Dentistry" in data["departments"]

    def test_ophthalmology_in_departments(self, client):
        resp = client.get("/api/rlhf/departments")
        data = json.loads(resp.data)
        assert "Ophthalmology" in data["departments"]

    def test_at_least_31_departments(self, client):
        resp = client.get("/api/rlhf/departments")
        data = json.loads(resp.data)
        assert len(data["departments"]) >= 31


# ---------------------------------------------------------------------------
# /api/rlhf/scope_guard
# ---------------------------------------------------------------------------

class TestScopeGuardEndpoint:
    def test_get_returns_200(self, client):
        resp = client.get("/api/rlhf/scope_guard")
        assert resp.status_code == 200

    def test_get_returns_json(self, client):
        resp = client.get("/api/rlhf/scope_guard")
        data = json.loads(resp.data)
        assert isinstance(data, dict)

    def test_get_has_enabled_key(self, client):
        resp = client.get("/api/rlhf/scope_guard")
        data = json.loads(resp.data)
        assert "enabled" in data

    def test_post_set_threshold_returns_200(self, client):
        resp = client.post(
            "/api/rlhf/scope_guard",
            data=json.dumps({"action": "set_threshold", "threshold": 0.5}),
            content_type="application/json",
        )
        assert resp.status_code == 200

    def test_post_refresh_returns_200(self, client):
        resp = client.post(
            "/api/rlhf/scope_guard",
            data=json.dumps({"action": "refresh"}),
            content_type="application/json",
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /api/rlhf/ranked-data
# ---------------------------------------------------------------------------

class TestRankedDataEndpoint:
    def test_get_returns_200(self, client):
        resp = client.get("/api/rlhf/ranked-data")
        assert resp.status_code == 200

    def test_get_returns_json(self, client):
        resp = client.get("/api/rlhf/ranked-data")
        data = json.loads(resp.data)
        assert isinstance(data, dict)

    def test_get_has_success_key(self, client):
        resp = client.get("/api/rlhf/ranked-data")
        data = json.loads(resp.data)
        assert "success" in data

    def test_get_with_domain_filter(self, client):
        resp = client.get("/api/rlhf/ranked-data?domain=Dentistry")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data.get("success") is True

    def test_get_with_ophthalmology_filter(self, client):
        resp = client.get("/api/rlhf/ranked-data?domain=Ophthalmology")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /api/rlhf/ranked-data/stats
# ---------------------------------------------------------------------------

class TestRankedDataStats:
    def test_returns_200(self, client):
        resp = client.get("/api/rlhf/ranked-data/stats")
        assert resp.status_code == 200

    def test_returns_json_with_success(self, client):
        resp = client.get("/api/rlhf/ranked-data/stats")
        data = json.loads(resp.data)
        assert data.get("success") is True


# ---------------------------------------------------------------------------
# /api/rlhf/experiments
# ---------------------------------------------------------------------------

class TestExperimentsEndpoint:
    def test_get_returns_200(self, client):
        resp = client.get("/api/rlhf/experiments")
        assert resp.status_code == 200

    def test_get_returns_json(self, client):
        resp = client.get("/api/rlhf/experiments")
        data = json.loads(resp.data)
        assert isinstance(data, dict)

    def test_get_has_success_key(self, client):
        resp = client.get("/api/rlhf/experiments")
        data = json.loads(resp.data)
        assert "success" in data

    def test_get_has_experiments_list(self, client):
        resp = client.get("/api/rlhf/experiments")
        data = json.loads(resp.data)
        if data.get("success"):
            assert "experiments" in data
            assert isinstance(data["experiments"], list)
