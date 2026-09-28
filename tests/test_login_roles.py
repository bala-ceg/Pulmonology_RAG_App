from __future__ import annotations

from contextlib import contextmanager

from flask import Flask

from routes import disciplines


class _FakeCursor:
    def __init__(self, rows, columns=None):
        self._rows = list(rows)
        self._columns = list(columns or [])
        self.queries: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query, params=None):
        self.queries.append(str(query))

    def fetchone(self):
        if not self._rows:
            return None
        return self._rows.pop(0)

    def fetchall(self):
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


def test_login_maps_nurse_to_nurse_party_id(monkeypatch):
    ehr_cursor = _FakeCursor(
        [
            (
                "nurse-party-123",
                "nurse1",
                "secret",
                "NURSE",
                "Nina",
                "Nurse",
                "nina.nurse@hospital.test",
            ),
        ],
        columns=[
            "party_id",
            "party_type",
            "username",
            "password_hash",
            "pces_role",
            "first_name",
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
    response = client.post("/api/login", json={"username": "nurse1", "password": "secret"})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert payload["party_id"] == "nurse-party-123"
    assert payload["pces_role"] == "NURSE"
    assert any("FROM p_party" in query for query in ehr_cursor.queries)
    assert any("NURSE" in query for query in ehr_cursor.queries)
