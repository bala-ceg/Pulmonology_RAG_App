from __future__ import annotations

from flask import Flask

from routes import entra_auth


def _client():
    app = Flask(__name__)
    app.config.update(TESTING=True, SECRET_KEY="test-session-secret-" + ("x" * 40))
    app.register_blueprint(entra_auth.entra_auth_bp)
    return app.test_client()


def _pending(client):
    with client.session_transaction() as state:
        state["entra_pending_link"] = {
            "tenant_id": "0ec0310b-1be7-4aa5-99f0-4bab9e2558cf",
            "object_id": "b4c2f1b3-5b3d-4db3-9f44-0f7a21b9c2c4",
        }
        state["entra_link_csrf"] = "csrf-test-token"


def test_request_code_renders_generic_confirmation(monkeypatch):
    called = {}

    def fake_request(tid, oid, email, tenant):
        called.update(tid=tid, oid=oid, email=email, tenant=tenant)
        return "ccf8d1b2-32b5-4fb1-a0df-14d95ab9d10a"

    monkeypatch.setattr(entra_auth, "request_email_code", fake_request)
    client = _client()
    _pending(client)
    response = client.post("/auth/entra/link/request", data={
        "csrf_token": "csrf-test-token",
        "email": "verified@example.test",
        "pces_tenant_id": "1f0e868c-f47d-474f-a537-16010f9bd41b",
    })
    assert response.status_code == 200
    assert b"code has been sent" in response.data
    assert called["email"] == "verified@example.test"
    with client.session_transaction() as state:
        assert state["entra_email_challenge_id"] == "ccf8d1b2-32b5-4fb1-a0df-14d95ab9d10a"


def test_verify_code_sets_pces_session_and_redirects(monkeypatch):
    def fake_verify(challenge, code, tid, oid):
        assert challenge == "ccf8d1b2-32b5-4fb1-a0df-14d95ab9d10a"
        assert code == "123456"
        return {
            "party_id": "party-123",
            "tenant_id": "tenant-123",
            "party_type": "DOCTOR",
            "pces_role": "PULMONOLOGIST",
        }

    monkeypatch.setattr(entra_auth, "verify_email_code", fake_verify)
    client = _client()
    _pending(client)
    with client.session_transaction() as state:
        state["entra_email_challenge_id"] = "ccf8d1b2-32b5-4fb1-a0df-14d95ab9d10a"
    response = client.post("/auth/entra/link/verify", data={
        "csrf_token": "csrf-test-token", "code": "123456",
    })
    assert response.status_code == 302
    with client.session_transaction() as state:
        assert state["pces_identity"]["party_id"] == "party-123"
        assert state["pces_identity"]["auth_provider"] == "entra_external_id"
        assert "entra_pending_link" not in state


def test_verify_code_rejects_missing_csrf(monkeypatch):
    def should_not_verify(*_args):
        raise AssertionError("OTP verification must not run without CSRF")

    monkeypatch.setattr(entra_auth, "verify_email_code", should_not_verify)
    client = _client()
    _pending(client)
    with client.session_transaction() as state:
        state["entra_email_challenge_id"] = "ccf8d1b2-32b5-4fb1-a0df-14d95ab9d10a"
    response = client.post("/auth/entra/link/verify", data={"code": "123456"})
    assert response.status_code == 400
