from __future__ import annotations

from contextlib import contextmanager

from flask import Flask

from routes import disciplines


class _FakeCursor:
    def __init__(self, rows, columns=None, tenant_rows=None):
        self._rows = list(rows)
        self._columns = list(columns or [])
        self._tenant_rows = list(tenant_rows or [])
        self.queries: list[str] = []
        self.last_query = ""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query, params=None):
        self.last_query = str(query)
        self.queries.append(self.last_query)

    def fetchone(self):
        if not self._rows:
            return None
        return self._rows.pop(0)

    def fetchall(self):
        if "FROM p_tenant" in self.last_query:
            return list(self._tenant_rows)
        return [(column,) for column in self._columns]


class _FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor


def _register_test_app():
    app = Flask(__name__)
    app.register_blueprint(disciplines.disciplines_bp)
    app.config["TESTING"] = True
    return app


def test_login_maps_nurse_to_nurse_party_id(monkeypatch, caplog):
    caplog.set_level("INFO", logger=disciplines.__name__)
    ehr_cursor = _FakeCursor(
        [
            (
                "nurse-party-123",
                "NURSE",
                "nurse1",
                "secret",
                "NURSE",
                "Nina",
                "Marie",
                "Nurse",
                "nina.nurse@hospital.test",
            ),
            ("tenant-101", "HOSP-101", "KIMS Kolkata"),
        ],
        columns=[
            "party_id",
            "party_type",
            "username",
            "password_hash",
            "pces_role",
            "first_name",
            "middle_name",
            "last_name",
            "email",
            "is_active",
        ],
    )

    @contextmanager
    def fake_db_conn():
        raise AssertionError("login should not use pces_base")

    @contextmanager
    def fake_ehr_conn():
        yield _FakeConnection(ehr_cursor)

    monkeypatch.setattr(disciplines, "_db_conn", fake_db_conn)
    monkeypatch.setattr(disciplines, "_ehr_conn", fake_ehr_conn)

    client = _register_test_app().test_client()
    response = client.post(
        "/api/login",
        json={"username": "nurse1", "password": "secret", "tenant_id": "tenant-101"},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert payload["party_id"] == "nurse-party-123"
    assert payload["party_type"] == "NURSE"
    assert payload["first_name"] == "Nina"
    assert payload["middle_name"] == "Marie"
    assert payload["last_name"] == "Nurse"
    assert payload["pces_role"] == "NURSE"
    assert payload["tenant_id"] == "tenant-101"
    assert payload["tenant_code"] == "HOSP-101"
    assert payload["tenant_name"] == "KIMS Kolkata"
    assert any(
        "party_id=nurse-party-123" in record.message
        and "tenant_name='KIMS Kolkata'" in record.message
        and "secret" not in record.message
        for record in caplog.records
    )
    assert any("FROM p_party" in query for query in ehr_cursor.queries)
    assert any("p_party_tenant" in query and "p_tenant" in query for query in ehr_cursor.queries)
    assert any("NURSE" in query for query in ehr_cursor.queries)


def test_login_rejects_unassociated_hospital(monkeypatch):
    ehr_cursor = _FakeCursor(
        [
            (
                "doctor-party-123",
                "DOCTOR",
                "doctor1",
                "secret",
                "Cardiology, SME",
                "Shruti",
                "",
                "Malee",
                "shruti@example.test",
            ),
        ],
        columns=[
            "party_id",
            "party_type",
            "username",
            "password_hash",
            "pces_role",
            "first_name",
            "middle_name",
            "last_name",
            "email",
            "is_active",
        ],
    )

    @contextmanager
    def fake_ehr_conn():
        yield _FakeConnection(ehr_cursor)

    monkeypatch.setattr(disciplines, "_ehr_conn", fake_ehr_conn)

    response = _register_test_app().test_client().post(
        "/api/login",
        json={"username": "doctor1", "password": "secret", "tenant_id": "other-tenant"},
    )

    assert response.status_code == 403
    assert response.get_json()["success"] is False


def test_login_tenants_lists_all_tenants(monkeypatch):
    ehr_cursor = _FakeCursor(
        [],
        tenant_rows=[
            ("tenant-101", "HOSP-101", "KIMS Kolkata"),
            ("tenant-102", "HOSP-102", "NIMS Delhi"),
        ],
    )

    @contextmanager
    def fake_ehr_conn():
        yield _FakeConnection(ehr_cursor)

    monkeypatch.setattr(disciplines, "_ehr_conn", fake_ehr_conn)

    response = _register_test_app().test_client().get("/api/tenants")

    assert response.status_code == 200
    assert response.get_json() == [
        {"tenant_id": "tenant-101", "tenant_code": "HOSP-101", "tenant_name": "KIMS Kolkata"},
        {"tenant_id": "tenant-102", "tenant_code": "HOSP-102", "tenant_name": "NIMS Delhi"},
    ]


def test_login_requires_hospital_selection(monkeypatch):
    @contextmanager
    def unexpected_db_conn():
        raise AssertionError("validation should happen before connecting to the EHR")
        yield

    monkeypatch.setattr(disciplines, "_ehr_conn", unexpected_db_conn)

    response = _register_test_app().test_client().post(
        "/api/login",
        json={"username": "doctor1", "password": "secret"},
    )

    assert response.status_code == 400
    assert "hospital" in response.get_json()["message"]
