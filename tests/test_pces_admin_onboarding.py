from __future__ import annotations
import ast
from datetime import date
from pathlib import Path
import shutil
import subprocess
import uuid
from unittest.mock import MagicMock
import pytest
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

def test_user_search_passes_last_name_party_type_and_email(monkeypatch):
    from routes import pces_admin
    seen={}
    def fake(admin_id,tenant_id,last_name,party_type,email):
        seen.update(tenant_id=tenant_id,last_name=last_name,party_type=party_type,email=email)
        return [{"party_id":"p1","on_boarded":True,"on_board_date":"2026-08-15"}]
    monkeypatch.setattr(pces_admin,"search_users",fake)
    response=client(ADMIN).get(
        "/api/pces-admin/users/search?tenant_id=00000000-0000-0000-0000-000000000001"
        "&last_name=Pearson&party_type=DOCTOR&email=doctor%40example.test"
    )
    assert response.status_code==200
    assert seen["last_name"]=="Pearson"
    assert seen["party_type"]=="DOCTOR"
    assert seen["email"]=="doctor@example.test"
    assert response.get_json()["users"][0]["on_boarded"] is True


def test_user_update_requires_selected_hospital_membership(monkeypatch):
    from routes import pces_admin
    seen={}
    def fake(admin_id,party_id,tenant_id,data):
        seen.update(party_id=party_id,tenant_id=tenant_id,data=data)
        return {"party_id":party_id,"updated_fields":["email"]}
    monkeypatch.setattr(pces_admin,"update_user",fake)
    response=client(ADMIN).patch(
        "/api/pces-admin/users/00000000-0000-0000-0000-000000000002",
        json={"tenant_id":"00000000-0000-0000-0000-000000000001","email":"new@example.test"}
    )
    assert response.status_code==200
    assert seen["tenant_id"]=="00000000-0000-0000-0000-000000000001"
    assert seen["data"]=={"email":"new@example.test"}

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
    assert "Microsoft sign-in email: doctor@example.test" in message.get_content()
    assert "PCES username: Dr.Test" in message.get_content()


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
    monkeypatch.setattr(onboarding.Config, "PCES_SIGNIN_URL", "https://pces.example.test")
    monkeypatch.setattr(onboarding.smtplib, "SMTP_SSL", RefusingSMTP)

    try:
        onboarding._send_email("doctor@example.test", "Dr.Test", "TempPass123")
    except onboarding.OnboardingError as exc:
        assert "rejected one or more recipients" in str(exc)
    else:
        raise AssertionError("Expected refused recipient to fail")


def test_onboard_route_passes_explicit_password_reset(monkeypatch):
    from routes import pces_admin
    onboard = MagicMock(return_value={"email_sent": True})
    monkeypatch.setattr(pces_admin, "onboard_user", onboard)
    response = client(ADMIN).post(
        "/api/pces-admin/users/00000000-0000-0000-0000-000000000002/onboard",
        json={"tenant_id": "00000000-0000-0000-0000-000000000001", "reset_password": True},
    )
    assert response.status_code == 200
    onboard.assert_called_once_with(
        ADMIN["party_id"], "00000000-0000-0000-0000-000000000002",
        "00000000-0000-0000-0000-000000000001", reset_password=True,
    )


def test_onboard_route_does_not_reset_password_by_default(monkeypatch):
    from routes import pces_admin
    onboard = MagicMock(return_value={})
    monkeypatch.setattr(pces_admin, "onboard_user", onboard)
    assert client(ADMIN).post(
        "/api/pces-admin/users/00000000-0000-0000-0000-000000000002/onboard",
        json={"tenant_id": "00000000-0000-0000-0000-000000000001"},
    ).status_code == 200
    assert onboard.call_args.kwargs == {"reset_password": False}


def test_onboard_route_rejects_non_boolean_reset(monkeypatch):
    from routes import pces_admin
    onboard = MagicMock()
    monkeypatch.setattr(pces_admin, "onboard_user", onboard)
    response = client(ADMIN).post(
        "/api/pces-admin/users/00000000-0000-0000-0000-000000000002/onboard",
        json={"reset_password": "false"},
    )
    assert response.status_code == 400
    onboard.assert_not_called()


@pytest.fixture
def onboarding_database(monkeypatch):
    from services import pces_admin_onboarding as onboarding
    conn = MagicMock()
    conn.__enter__.return_value = conn
    cur = conn.cursor.return_value.__enter__.return_value
    connect = MagicMock(return_value=conn)
    monkeypatch.setattr(onboarding, "_connect", connect)
    return conn, cur, connect


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("email_failure", [False, True])
def test_onboard_creates_or_resets_then_commits_and_emails(monkeypatch, onboarding_database, existing, email_failure):
    from services import pces_admin_onboarding as onboarding
    conn, cur, connect = onboarding_database
    party_id = "00000000-0000-0000-0000-000000000002"
    tenant_id = "00000000-0000-0000-0000-000000000001"
    oid = "00000000-0000-0000-0000-000000000003"
    row = ("Test", "", "Doctor", "doctor@example.test", "Dr.Test", oid if existing else None, "DOCTOR", party_id)
    cur.fetchone.side_effect = [(1,), row] + ([] if existing else [(party_id,)]) + [(date(2026, 11, 13),)]
    monkeypatch.setattr(onboarding, "_email_settings", lambda: None)
    provision = MagicMock(return_value=(oid, "TempPass123"))
    reset = MagicMock(return_value="TempPass123")
    monkeypatch.setattr(onboarding, "provision_entra_user", provision)
    monkeypatch.setattr(onboarding, "reset_entra_password", reset)

    def send(email, username, password):
        assert conn.commit.called, "Credentials must be mapped before delivery"
        assert (email, username, password) == ("doctor@example.test", "Dr.Test", "TempPass123")
        if email_failure:
            raise onboarding.OnboardingError("SMTP delivery failed")

    monkeypatch.setattr(onboarding, "_send_email", send)
    result = onboarding.onboard_user(ADMIN["party_id"], party_id, tenant_id, reset_password=existing)
    connect.assert_called_once()
    assert result["on_boarded"] is True
    assert result["on_board_date"] == "2026-11-13"
    assert result["email_sent"] is not email_failure
    assert "TempPass123" not in str(result)
    if existing:
        provision.assert_not_called()
        reset.assert_called_once_with(oid, "doctor@example.test")
    else:
        reset.assert_not_called()
        provision.assert_called_once()
    if email_failure:
        assert "SMTP delivery failed" in result["message"]
        assert "reset the password" in result["message"]


def test_existing_account_reconciles_using_locked_connection(monkeypatch, onboarding_database):
    from services import pces_admin_onboarding as onboarding
    conn, cur, connect = onboarding_database
    party_id = "00000000-0000-0000-0000-000000000002"
    tenant_id = "00000000-0000-0000-0000-000000000001"
    cur.fetchone.side_effect = [
        (1,), ("Test", "", "Doctor", "doctor@example.test", "Dr.Test", party_id, "DOCTOR", party_id),
        (date(2026, 10, 1),),
    ]
    reset = MagicMock()
    send = MagicMock()
    monkeypatch.setattr(onboarding, "reset_entra_password", reset)
    monkeypatch.setattr(onboarding, "_send_email", send)
    result = onboarding.onboard_user(ADMIN["party_id"], party_id, tenant_id)
    connect.assert_called_once()  # A second connection would deadlock on the locked membership.
    reset.assert_not_called()
    send.assert_not_called()
    assert result["on_board_date"] == "2026-10-01"


def test_missing_email_config_does_not_change_microsoft_credentials(monkeypatch, onboarding_database):
    from services import pces_admin_onboarding as onboarding
    _, cur, _ = onboarding_database
    party_id = "00000000-0000-0000-0000-000000000002"
    cur.fetchone.side_effect = [
        (1,), ("Test", "", "Doctor", "doctor@example.test", "Dr.Test", None, "DOCTOR", party_id),
    ]
    monkeypatch.setattr(onboarding.Config, "PCES_SMTP_HOST", None)
    provision = MagicMock()
    monkeypatch.setattr(onboarding, "provision_entra_user", provision)
    with pytest.raises(onboarding.OnboardingError, match="SMTP configuration is incomplete"):
        onboarding.onboard_user(ADMIN["party_id"], party_id, party_id)
    provision.assert_not_called()


@pytest.mark.parametrize("identity_matches", [True, False])
def test_reset_only_changes_matching_local_microsoft_account(monkeypatch, identity_matches):
    from services import pces_admin_onboarding as onboarding
    monkeypatch.setattr(onboarding, "_graph_headers", lambda: {"Authorization": "Bearer test"})
    monkeypatch.setattr(onboarding.Config, "PCES_DIRECTORY_DOMAIN", "test.onmicrosoft.com")
    identity = {"signInType": "emailAddress" if identity_matches else "federated",
                "issuer": "test.onmicrosoft.com", "issuerAssignedId": "doctor@example.test"}
    get = MagicMock(return_value=MagicMock(status_code=200, json=lambda: {"identities": [identity]}))
    patch = MagicMock(return_value=MagicMock(status_code=204))
    monkeypatch.setattr(onboarding.requests, "get", get)
    monkeypatch.setattr(onboarding.requests, "patch", patch)
    oid = "00000000-0000-0000-0000-000000000003"
    if not identity_matches:
        with pytest.raises(onboarding.OnboardingError, match="not an email/password customer account"):
            onboarding.reset_entra_password(oid, "doctor@example.test")
        patch.assert_not_called()
        return
    password = onboarding.reset_entra_password(oid, "doctor@example.test")
    assert patch.call_args.args[0].endswith("/users/" + oid)
    assert patch.call_args.kwargs["json"]["passwordProfile"] == {
        "password": password, "forceChangePasswordNextSignIn": True,
    }


def test_graph_reset_permission_error_is_actionable_and_contains_no_credentials(monkeypatch):
    from services import pces_admin_onboarding as onboarding
    monkeypatch.setattr(onboarding, "_graph_headers", lambda: {})
    monkeypatch.setattr(onboarding.Config, "PCES_DIRECTORY_DOMAIN", "test.onmicrosoft.com")
    response = MagicMock(status_code=200)
    response.json.return_value = {"identities": [{
        "signInType": "emailAddress", "issuer": "test.onmicrosoft.com",
        "issuerAssignedId": "doctor@example.test",
    }]}
    monkeypatch.setattr(onboarding.requests, "get", lambda *args, **kwargs: response)
    monkeypatch.setattr(onboarding.requests, "patch", lambda *args, **kwargs: MagicMock(status_code=403))
    monkeypatch.setattr(onboarding, "_temp_password", lambda: "TempPass123")
    with pytest.raises(onboarding.OnboardingError, match="permissions and admin consent") as error:
        onboarding.reset_entra_password("00000000-0000-0000-0000-000000000003", "doctor@example.test")
    assert "TempPass123" not in str(error.value)


def test_provisioned_credentials_match_the_customer_email(monkeypatch):
    from services import pces_admin_onboarding as onboarding
    monkeypatch.setattr(onboarding, "_graph_headers", lambda: {})
    monkeypatch.setattr(onboarding.Config, "PCES_DIRECTORY_DOMAIN", "test.onmicrosoft.com")
    oid = "00000000-0000-0000-0000-000000000003"
    post = MagicMock(return_value=MagicMock(status_code=201, json=lambda: {"id": oid}))
    monkeypatch.setattr(onboarding.requests, "post", post)
    result_oid, password = onboarding.provision_entra_user(
        {"first_name": "Test", "middle_name": "", "last_name": "Doctor", "email": "doctor@example.test"}
    )
    assert result_oid == oid
    payload = post.call_args.kwargs["json"]
    assert payload["identities"] == [{
        "signInType": "emailAddress", "issuer": "test.onmicrosoft.com",
        "issuerAssignedId": "doctor@example.test",
    }]
    assert payload["passwordProfile"] == {"password": password, "forceChangePasswordNextSignIn": True}


@pytest.mark.parametrize("onboarded", [False, True])
def test_user_listing_returns_membership_onboarding_state(onboarded):
    # Load this route without bootstrapping unrelated RAG/model dependencies.
    from flask import jsonify, request
    tree = ast.parse((Path(__file__).resolve().parents[1] / "routes/disciplines.py").read_text())
    route = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "pces_admin_list_users")
    route.decorator_list = []
    conn = MagicMock()
    conn.__enter__.return_value = conn
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = [
        ("party-1", "Test", "", "Doctor", "DOCTOR", "CARDIOLOGY",
         "doctor@example.test", "Dr.Test", "entra-1", None,
         onboarded, date(2026, 11, 13) if onboarded else None),
    ]
    namespace = {"_require_pces_admin": lambda: None, "_ehr_conn": lambda: conn,
                 "request": request, "jsonify": jsonify, "uuid": uuid, "logger": MagicMock()}
    exec(compile(ast.Module(body=[route], type_ignores=[]), "<user-list-route>", "exec"), namespace)
    app = Flask(__name__)
    with app.test_request_context("/?tenant_id=00000000-0000-0000-0000-000000000001"):
        result = namespace["pces_admin_list_users"]().get_json()[0]
    assert result["on_boarded"] is onboarded
    assert result["on_board_date"] == ("2026-11-13" if onboarded else "")
    assert "pt.on_board" in cur.execute.call_args.args[0]


def test_admin_frontend_editing_and_onboarding():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for frontend behavior checks")
    template = (Path(__file__).resolve().parents[1] / "templates/index.html").read_text()
    source = template.split("let pcesUserHospitalCache = [];", 1)[1].split("function applyRoleVisibility", 1)[0]
    script = r"""
const assert = require('node:assert/strict');
const elements = {};
const fieldIds = ['party-id','first-name','middle-name','last-name','party-type','pces-role','email','username','entra-id','save-btn'];
for (const name of fieldIds) {
    const element = {value:'',readOnly:['party-id','entra-id'].includes(name),disabled:false,style:{},options:[],
        appendChild(option){this.options.push(option);},focus(){}};
    Object.defineProperty(element,'innerHTML',{set(){this.options=[];}});
    elements['pces-user-'+name] = element;
}
const button = {textContent:'On Board',disabled:false};
const document = {
    getElementById(id){return elements[id] || null;},
    createElement(){return {};},
    querySelectorAll(selector){return selector === '.pces-user-admin-actions button' ? [button] : [];}
};
let alerts=[], requests=[], confirmation='';
const alert = message => alerts.push(message);
const confirm = message => {confirmation=message;return true;};
let fetch = async () => ({ok:true,json:async()=>['CARDIOLOGY','NURSE']});
""" + source + r"""
(async () => {
    await loadPcesUserRoles();
    assert.equal(elements['pces-user-pces-role'].options.length,2);
    const user={party_id:'party-1',first_name:'Test',middle_name:'',last_name:'Doctor',
        party_type:'DOCTOR',pces_role:'CARDIOLOGY, SME',email:'doctor@example.test',
        username:'Dr.Test',entra_id:'entra-1'};
    selectPcesUser(user);
    for (const name of ['first-name','middle-name','last-name','email','username'])
        assert.equal(elements['pces-user-'+name].readOnly,false,name);
    assert.equal(elements['pces-user-party-type'].disabled,false);
    assert.equal(elements['pces-user-pces-role'].disabled,false);
    assert.equal(elements['pces-user-pces-role'].value,'CARDIOLOGY, SME');
    assert.ok(elements['pces-user-pces-role'].options.some(o=>o.value==='CARDIOLOGY, SME'));
    assert.equal(elements['pces-user-party-id'].readOnly,true);
    assert.equal(elements['pces-user-entra-id'].readOnly,true);
    pcesSelectedUserHospital={tenant_id:'tenant-1'};
    loadPcesUsersForHospital=async()=>{};
    fetch=async(url,options)=>{requests.push({url,...options});return {ok:true,json:async()=>({success:true,party_id:'party-1',email_sent:true})};};
    elements['pces-user-first-name'].value='Edited';
    await onboardSelectedPcesUser();
    assert.equal(requests.length,0);
    assert.ok(alerts.at(-1).includes('Save your user details'));
    await savePcesUser();
    assert.equal(requests[0].method,'PATCH');
    assert.equal(JSON.parse(requests[0].body).first_name,'Edited');
    selectPcesUser(user);
    await onboardSelectedPcesUser();
    assert.ok(confirmation.includes('old Microsoft password will stop working'));
    assert.deepEqual(JSON.parse(requests[1].body),{tenant_id:'tenant-1',reset_password:true});
    selectPcesUser({...user,entra_id:''});
    await onboardSelectedPcesUser();
    assert.deepEqual(JSON.parse(requests[2].body),{tenant_id:'tenant-1',reset_password:false});
    assert.equal(button.disabled,false);
})().catch(error=>{console.error(error);process.exitCode=1;});
"""
    result = subprocess.run([node, "-e", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
