from __future__ import annotations
from flask import Flask
from routes.pces_admin import pces_admin_bp

ADMIN={"party_id":"b620e197-bbe3-4b26-a43c-9131720e85ff","party_type":"PCES_ADMIN","pces_role":"PCES_ADMIN"}
def client(identity=None):
    app=Flask(__name__); app.config.update(TESTING=True,SECRET_KEY="test-secret")
    app.register_blueprint(pces_admin_bp); c=app.test_client()
    if identity:
        with c.session_transaction() as s: s["pces_identity"]=identity
    return c

def test_hospitals_requires_admin():
    assert client().get("/api/pces-admin/hospitals").status_code==403

def test_users_requires_admin():
    doctor={"party_id":"00000000-0000-0000-0000-000000000001","party_type":"DOCTOR","pces_role":"Cardiology"}
    assert client(doctor).get("/api/pces-admin/users").status_code==403

def test_hospital_lookup(monkeypatch):
    from routes import pces_admin
    monkeypatch.setattr(pces_admin,"list_hospitals",lambda admin_id:[{"tenant_id":"t1","hospital_name":"St Luke"}])
    response=client(ADMIN).get("/api/pces-admin/hospitals")
    assert response.status_code==200
    assert response.get_json()["hospitals"][0]["hospital_name"]=="St Luke"

def test_users_default_to_20(monkeypatch):
    from routes import pces_admin
    seen={}
    def fake(admin_id,tenant_id,limit,offset):
        seen.update(tenant_id=tenant_id,limit=limit,offset=offset); return []
    monkeypatch.setattr(pces_admin,"list_users",fake)
    response=client(ADMIN).get("/api/pces-admin/users?tenant_id=00000000-0000-0000-0000-000000000001")
    assert response.status_code==200 and seen["limit"]==20 and seen["offset"]==0

def test_patch_is_admin_only():
    assert client().patch("/api/pces-admin/users/00000000-0000-0000-0000-000000000001",json={"email":"x@y.com"}).status_code==403

def test_onboard_is_admin_only():
    assert client().post("/api/pces-admin/users/00000000-0000-0000-0000-000000000001/onboard").status_code==403


def test_send_email_uses_bluehost_smtp(monkeypatch):
    from email.message import EmailMessage
    from services import pces_admin_onboarding as onboarding

    class FakeSMTP:
        def __init__(self, **kwargs):
            seen["connection"] = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def login(self, username, password):
            seen["login"] = (username, password)

        def send_message(self, message):
            seen["message"] = message
            return {}

    seen = {}
    monkeypatch.setattr(onboarding.Config, "PCES_SMTP_HOST", "pcesglobal.ai", raising=False)
    monkeypatch.setattr(onboarding.Config, "PCES_SMTP_PORT", 465, raising=False)
    monkeypatch.setattr(onboarding.Config, "PCES_SMTP_USERNAME", "noreply@pcesglobal.ai", raising=False)
    monkeypatch.setattr(onboarding.Config, "PCES_SMTP_PASSWORD", "test-secret", raising=False)
    monkeypatch.setattr(onboarding.Config, "PCES_SMTP_TIMEOUT", 30, raising=False)
    monkeypatch.setattr(onboarding.Config, "PCES_SIGNIN_URL", "https://pces.example.test", raising=False)
    monkeypatch.setattr(onboarding.smtplib, "SMTP_SSL", FakeSMTP)

    onboarding._send_email("doctor@example.test", "Dr.Test", "TempPass123")

    assert seen["connection"]["host"] == "pcesglobal.ai"
    assert seen["connection"]["port"] == 465
    assert seen["connection"]["timeout"] == 30
    assert seen["login"] == ("noreply@pcesglobal.ai", "test-secret")
    message = seen["message"]
    assert isinstance(message, EmailMessage)
    assert message["From"] == "noreply@pcesglobal.ai"
    assert message["To"] == "doctor@example.test"
    assert "TempPass123" in message.get_content()
    assert "https://pces.example.test" in message.get_content()


def test_send_email_requires_bluehost_configuration(monkeypatch):
    from services import pces_admin_onboarding as onboarding

    monkeypatch.setattr(onboarding.Config, "PCES_SMTP_HOST", None, raising=False)
    monkeypatch.setattr(onboarding.Config, "PCES_SMTP_USERNAME", None, raising=False)
    monkeypatch.setattr(onboarding.Config, "PCES_SMTP_PASSWORD", None, raising=False)

    try:
        onboarding._send_email("doctor@example.test", "Dr.Test", "TempPass123")
    except onboarding.OnboardingError as exc:
        assert "Bluehost SMTP configuration is incomplete" in str(exc)
    else:
        raise AssertionError("Expected missing SMTP configuration to fail")


def test_send_email_rejects_refused_recipient(monkeypatch):
    from services import pces_admin_onboarding as onboarding

    class RefusingSMTP:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def login(self, username, password):
            pass

        def send_message(self, message):
            return {"doctor@example.test": (550, b"rejected")}

    monkeypatch.setattr(onboarding.Config, "PCES_SMTP_HOST", "pcesglobal.ai", raising=False)
    monkeypatch.setattr(onboarding.Config, "PCES_SMTP_PORT", 465, raising=False)
    monkeypatch.setattr(onboarding.Config, "PCES_SMTP_USERNAME", "noreply@pcesglobal.ai", raising=False)
    monkeypatch.setattr(onboarding.Config, "PCES_SMTP_PASSWORD", "test-secret", raising=False)
    monkeypatch.setattr(onboarding.smtplib, "SMTP_SSL", RefusingSMTP)

    try:
        onboarding._send_email("doctor@example.test", "Dr.Test", "TempPass123")
    except onboarding.OnboardingError as exc:
        assert "rejected one or more recipients" in str(exc)
    else:
        raise AssertionError("Expected refused recipient to fail")
