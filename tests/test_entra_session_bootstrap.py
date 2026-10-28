from __future__ import annotations

from flask import Flask

from routes import entra_auth


def _client():
    app = Flask(__name__)
    app.config.update(TESTING=True, SECRET_KEY="test-session-secret-" + ("x" * 40))
    app.register_blueprint(entra_auth.entra_auth_bp)
    return app.test_client()


def test_session_bootstrap_requires_authenticated_session():
    response = _client().get("/auth/entra/session")
    assert response.status_code == 401
    assert response.get_json()["success"] is False


def test_session_bootstrap_returns_profile_for_active_membership(monkeypatch):
    expected = {
        "success": True,
        "username": "doctor.test",
        "party_id": "party-123",
        "party_type": "DOCTOR",
        "first_name": "Test",
        "middle_name": "",
        "last_name": "Doctor",
        "pces_role": "DOCTOR",
        "full_name": "Test Doctor",
        "email": "doctor@example.test",
        "department": "DOCTOR",
        "tenant_id": "tenant-123",
        "tenant_code": "HOSP-A",
        "tenant_name": "Hospital A",
        "hospital_name": "Hospital A",
    }
    called = {}

    def fake_profile(party_id, tenant_id, expected_object_id=None):
        called.update(
            party_id=party_id,
            tenant_id=tenant_id,
            expected_object_id=expected_object_id,
        )
        return expected

    monkeypatch.setattr(entra_auth, "get_pces_profile_for_membership", fake_profile)
    client = _client()
    with client.session_transaction() as state:
        state["pces_identity"] = {
            "party_id": "party-123",
            "tenant_id": "tenant-123",
            "party_type": "DOCTOR",
            "pces_role": "DOCTOR",
            "auth_provider": "entra_external_id",
            "entra_object_id": "entra-oid-123",
        }

    response = client.get("/auth/entra/session")
    assert response.status_code == 200
    assert response.get_json() == expected
    assert called == {
        "party_id": "party-123",
        "tenant_id": "tenant-123",
        "expected_object_id": "entra-oid-123",
    }


def test_session_bootstrap_clears_invalid_identity(monkeypatch):
    def reject_profile(*_args, **_kwargs):
        raise entra_auth.EntraIdentityError("membership inactive")

    monkeypatch.setattr(entra_auth, "get_pces_profile_for_membership", reject_profile)
    client = _client()
    with client.session_transaction() as state:
        state["pces_identity"] = {
            "party_id": "party-123",
            "tenant_id": "tenant-123",
            "auth_provider": "entra_external_id",
            "entra_object_id": "entra-oid-123",
        }

    response = client.get("/auth/entra/session")
    assert response.status_code == 401
    with client.session_transaction() as state:
        assert "pces_identity" not in state
