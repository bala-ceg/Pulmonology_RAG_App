from __future__ import annotations

from contextlib import contextmanager
import sqlite3

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
    app.config["SECRET_KEY"] = "test-only-session-key"
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
    with client.session_transaction() as login_session:
        assert login_session["pces_identity"] == {
            "party_id": "nurse-party-123",
            "tenant_id": "tenant-101",
        }
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


def test_patient_list_requires_login(monkeypatch):
    def unexpected_db_conn():
        raise AssertionError("unauthenticated requests must not query patients")

    monkeypatch.setattr(disciplines, "_ehr_conn", unexpected_db_conn)
    client = _register_test_app().test_client()
    response = client.get("/api/patients/first20?tenant_id=other-tenant")
    assert response.status_code == 401


def test_patient_list_filters_by_authenticated_tenant(monkeypatch):
    with sqlite3.connect(":memory:") as database:
        database.executescript("""
            CREATE TABLE p_party (
                party_id TEXT, party_type TEXT, first_name TEXT,
                middle_name TEXT, last_name TEXT, date_of_birth TEXT,
                phone TEXT, email TEXT, is_active BOOLEAN
            );
            CREATE TABLE p_address (
                party_id TEXT, address_id TEXT, line1 TEXT, line2 TEXT,
                city TEXT, state TEXT, postal_code TEXT, is_active BOOLEAN
            );
            CREATE TABLE p_party_tenant (party_id TEXT, tenant_id TEXT);
            INSERT INTO p_party VALUES
                ('patient-a', 'PATIENT', 'Alice', '', 'A', NULL, '', '', TRUE),
                ('patient-b', 'PATIENT', 'Bob', '', 'B', NULL, '', '', TRUE);
            INSERT INTO p_party_tenant VALUES
                ('patient-a', 'tenant-a'), ('patient-a', 'tenant-a'),
                ('patient-b', 'tenant-b');
        """)

        class PatientCursor(_FakeCursor):
            def execute(self, query, params=None):
                self.result = database.execute(query.replace("%s", "?"), params)

            def fetchall(self):
                return self.result.fetchall()

        @contextmanager
        def fake_ehr_conn():
            yield _FakeConnection(PatientCursor([]))

        monkeypatch.setattr(disciplines, "_ehr_conn", fake_ehr_conn)
        client = _register_test_app().test_client()
        with client.session_transaction() as login_session:
            login_session["pces_identity"] = {
                "party_id": "doctor-a",
                "tenant_id": "tenant-a",
            }
        response = client.get("/api/patients/first20?tenant_id=tenant-b")
        assert response.status_code == 200
        assert [row["patient_id"] for row in response.get_json()] == ["patient-a"]

        assert client.post("/api/logout").status_code == 200
        assert client.get("/api/patients/first20").status_code == 401


def test_local_patient_search_requires_login(monkeypatch):
    def unexpected_db_conn():
        raise AssertionError("unauthenticated search must not query patients")

    monkeypatch.setattr(disciplines, "_ehr_conn", unexpected_db_conn)
    response = _register_test_app().test_client().get("/api/patient/search?first=Alice")
    assert response.status_code == 401


def test_local_patient_search_uses_session_tenant(monkeypatch):
    cursor = _FakeCursor([])
    cursor.params = None

    def execute(query, params=None):
        cursor.queries.append(str(query))
        cursor.params = params

    cursor.execute = execute

    @contextmanager
    def fake_ehr_conn():
        yield _FakeConnection(cursor)

    monkeypatch.setattr(disciplines, "_ehr_conn", fake_ehr_conn)
    client = _register_test_app().test_client()
    with client.session_transaction() as login_session:
        login_session["pces_identity"] = {
            "party_id": "doctor-a",
            "tenant_id": "tenant-a",
        }
    response = client.get("/api/patient/search?first=Alice&tenant_id=tenant-b")
    assert response.status_code == 200
    assert cursor.params == ["tenant-a", "%alice%"]
    assert "EXISTS" in cursor.queries[0]
    assert "pt.party_id = pp.party_id" in cursor.queries[0]
