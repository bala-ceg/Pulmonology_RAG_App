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
    monkeypatch.setattr(pces_admin,"list_hospitals",lambda:[{"tenant_id":"t1","hospital_name":"St Luke"}])
    response=client(ADMIN).get("/api/pces-admin/hospitals")
    assert response.status_code==200
    assert response.get_json()["hospitals"][0]["hospital_name"]=="St Luke"

def test_users_default_to_20(monkeypatch):
    from routes import pces_admin
    seen={}
    def fake(tenant_id,limit,offset):
        seen.update(tenant_id=tenant_id,limit=limit,offset=offset); return []
    monkeypatch.setattr(pces_admin,"list_users",fake)
    response=client(ADMIN).get("/api/pces-admin/users?tenant_id=00000000-0000-0000-0000-000000000001")
    assert response.status_code==200 and seen["limit"]==20 and seen["offset"]==0

def test_patch_is_admin_only():
    assert client().patch("/api/pces-admin/users/00000000-0000-0000-0000-000000000001",json={"email":"x@y.com"}).status_code==403

def test_onboard_is_admin_only():
    assert client().post("/api/pces-admin/users/00000000-0000-0000-0000-000000000001/onboard").status_code==403
