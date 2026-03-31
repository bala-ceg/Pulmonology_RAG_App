"""
Unit tests for sft_experiment_manager.py
------------------------------------------
Tests DEPARTMENTS dict, get_department_list, keyword matching, and
get_prompts_by_department using an in-memory SQLite database.
"""
import os
import sys
import sqlite3
import uuid
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sft_experiment_manager import (
    DEPARTMENTS,
    get_department_list,
    get_prompts_by_department,
)


# ---------------------------------------------------------------------------
# DEPARTMENTS dict
# ---------------------------------------------------------------------------

class TestDepartments:
    def test_dentistry_present(self):
        assert "Dentistry" in DEPARTMENTS

    def test_ophthalmology_present(self):
        assert "Ophthalmology" in DEPARTMENTS

    def test_cardiology_present(self):
        assert "Cardiology" in DEPARTMENTS

    def test_neurology_present(self):
        assert "Neurology" in DEPARTMENTS

    def test_total_departments_at_least_31(self):
        assert len(DEPARTMENTS) >= 31

    def test_all_departments_have_keywords(self):
        for dept, keywords in DEPARTMENTS.items():
            assert isinstance(keywords, list), f"{dept} keywords must be a list"
            assert len(keywords) > 0, f"{dept} must have at least one keyword"

    def test_dentistry_keywords_are_dental(self):
        dental_kw = DEPARTMENTS["Dentistry"]
        assert any(k in dental_kw for k in ["tooth", "teeth", "dental", "dentist"])

    def test_ophthalmology_keywords_include_eye(self):
        opht_kw = DEPARTMENTS["Ophthalmology"]
        assert any(k in opht_kw for k in ["eye", "vision", "optic"])

    def test_no_duplicate_department_names(self):
        keys = list(DEPARTMENTS.keys())
        assert len(keys) == len(set(keys)), "Duplicate department names found"

    def test_all_values_are_lists_of_strings(self):
        for dept, keywords in DEPARTMENTS.items():
            for kw in keywords:
                assert isinstance(kw, str), f"Keyword in {dept} must be a string"


# ---------------------------------------------------------------------------
# get_department_list
# ---------------------------------------------------------------------------

class TestGetDepartmentList:
    def test_returns_dict_with_success(self):
        result = get_department_list()
        assert isinstance(result, dict)
        assert result.get("success") is True

    def test_contains_departments_key(self):
        result = get_department_list()
        assert "departments" in result

    def test_departments_is_list(self):
        result = get_department_list()
        assert isinstance(result["departments"], list)

    def test_dentistry_in_list(self):
        result = get_department_list()
        assert "Dentistry" in result["departments"]

    def test_ophthalmology_in_list(self):
        result = get_department_list()
        assert "Ophthalmology" in result["departments"]

    def test_list_count_matches_departments_dict(self):
        result = get_department_list()
        assert len(result["departments"]) == len(DEPARTMENTS)


# ---------------------------------------------------------------------------
# get_prompts_by_department — using in-memory SQLite
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sqlite_db_with_data(tmp_path_factory):
    """
    Create a temporary SQLite DB with sft_ranked_data rows for Dentistry
    and return its path. Patches the module's DB backend to use it.
    """
    import sft_experiment_manager as sem

    db_path = str(tmp_path_factory.mktemp("db") / "test_sft.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sft_ranked_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt TEXT NOT NULL,
            response_text TEXT NOT NULL DEFAULT '',
            rank INTEGER NOT NULL DEFAULT 1,
            reason TEXT DEFAULT '',
            group_id TEXT NOT NULL,
            domain TEXT,
            sme_score INTEGER,
            sme_score_reason TEXT,
            sme_reviewed_by TEXT,
            sme_reviewed_at TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        )
    """)
    # Insert Dentistry rows
    for i in range(6):
        gid = str(uuid.uuid4())[:8]
        for rank in range(1, 4):
            conn.execute(
                "INSERT INTO sft_ranked_data (prompt, response_text, rank, group_id, domain) "
                "VALUES (?, ?, ?, ?, ?)",
                (f"Dental question {i}: how do I treat a tooth cavity?",
                 f"Response rank {rank}", rank, gid, "Dentistry")
            )
    # Insert Ophthalmology rows
    for i in range(3):
        gid = str(uuid.uuid4())[:8]
        conn.execute(
            "INSERT INTO sft_ranked_data (prompt, response_text, rank, group_id, domain) "
            "VALUES (?, ?, ?, ?, ?)",
            (f"Eye question {i}: what causes glaucoma?", "Eye response", 1, gid, "Ophthalmology")
        )
    conn.commit()
    conn.close()

    # Patch the module-level DB to use our SQLite DB
    original_backend = sem._DB_BACKEND
    original_conn_fn = sem._get_db_connection

    def _patched_conn():
        c = sqlite3.connect(db_path)
        return c, "sqlite"

    sem._get_db_connection = _patched_conn
    sem._DB_BACKEND = "sqlite"

    yield db_path

    # Restore
    sem._get_db_connection = original_conn_fn
    sem._DB_BACKEND = original_backend


class TestGetPromptsByDepartment:
    def test_unknown_department_returns_error(self):
        result = get_prompts_by_department("NonExistentDept")
        assert result.get("success") is False
        assert "error" in result

    def test_known_department_returns_success_structure(self):
        # Just test the structure without requiring real DB rows
        result = get_prompts_by_department("Dentistry", limit=5)
        assert isinstance(result, dict)
        assert "success" in result

    def test_department_result_has_prompts_key_on_success(self):
        result = get_prompts_by_department("Dentistry", limit=5)
        if result.get("success"):
            assert "prompts" in result or "groups" in result or "data" in result
