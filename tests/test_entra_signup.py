from __future__ import annotations

from flask import Flask

from routes.entra_signup import entra_signup_bp


def _client():
    app = Flask(__name__)
    app.config.update(TESTING=True, SECRET_KEY="signup-test-secret-" + ("x" * 40))
    app.register_blueprint(entra_signup_bp)
    return app.test_client()


def test_signup_requires_verified_pending_entra_identity():
    response = _client().get("/auth/entra/signup")
    assert response.status_code == 401
    assert response.get_json()["success"] is False


def test_office_admin_review_page_requires_session():
    response = _client().get("/auth/entra/admin")
    assert response.status_code == 403
    assert response.get_json()["success"] is False


def test_non_office_admin_cannot_list_signup_requests():
    client = _client()
    with client.session_transaction() as state:
        state["pces_identity"] = {
            "party_id": "00000000-0000-0000-0000-000000000001",
            "tenant_id": "00000000-0000-0000-0000-000000000002",
            "party_type": "DOCTOR",
        }
    response = client.get("/auth/entra/admin/requests")
    assert response.status_code == 403


def test_office_admin_review_requires_csrf_token():
    client = _client()
    with client.session_transaction() as state:
        state["pces_identity"] = {
            "party_id": "00000000-0000-0000-0000-000000000001",
            "tenant_id": "00000000-0000-0000-0000-000000000002",
            "party_type": "OFFICE_ADMIN",
        }
        state["entra_admin_csrf"] = "expected-token"
    response = client.post(
        "/auth/entra/admin/requests/00000000-0000-0000-0000-000000000003/review",
        json={"decision": "APPROVE"},
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid review token"
