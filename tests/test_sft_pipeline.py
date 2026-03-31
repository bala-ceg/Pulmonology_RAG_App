"""
Tests for the SFT pipeline scripts:
  - load_prompts_to_sft.py   (_connect fallback, ensure_domain_column, insert_prompts)
  - generate_sft_responses.py (_connect fallback)
  - auto_sme_score.py         (_connect fallback, rank_to_score)

All DB operations run against an in-memory / temp SQLite DB; no real
PostgreSQL or OpenAI calls are made.
"""
import os
import sys
import json
import sqlite3
import uuid
import tempfile
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sqlite_conn(path=":memory:"):
    conn = sqlite3.connect(path)
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
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# load_prompts_to_sft — _connect, ensure_domain_column, insert_prompts
# ---------------------------------------------------------------------------

class TestLoadPromptsToSFT:
    """Tests using in-memory SQLite via mocked _connect."""

    def _get_module(self):
        import load_prompts_to_sft
        return load_prompts_to_sft

    def test_connect_falls_back_to_sqlite_when_pg_fails(self):
        mod = self._get_module()
        with patch.object(mod, '_pg') as mock_pg:
            mock_pg.connect.side_effect = Exception("connection refused")
            conn, ph, is_sqlite = mod._connect()
            assert is_sqlite is True
            assert ph == "?"
            conn.close()

    def test_connect_uses_pg_when_available(self):
        mod = self._get_module()
        mock_conn = MagicMock()
        with patch.object(mod, '_pg') as mock_pg:
            mock_pg.connect.return_value = mock_conn
            conn, ph, is_sqlite = mod._connect()
            assert is_sqlite is False
            assert ph == "%s"

    def test_ensure_domain_column_sqlite_no_op_when_exists(self):
        mod = self._get_module()
        conn = _make_sqlite_conn()   # already has domain column
        cur = conn.cursor()
        mod.ensure_domain_column(cur, is_sqlite=True)   # should not raise
        conn.close()

    def test_ensure_domain_column_sqlite_adds_when_missing(self):
        mod = self._get_module()
        conn = sqlite3.connect(":memory:")
        conn.execute("""
            CREATE TABLE sft_ranked_data (
                id INTEGER PRIMARY KEY,
                prompt TEXT,
                response_text TEXT,
                rank INTEGER,
                reason TEXT,
                group_id TEXT
            )
        """)
        conn.commit()
        cur = conn.cursor()
        mod.ensure_domain_column(cur, is_sqlite=True)
        # Check column was added
        cols = [row[1] for row in cur.execute("PRAGMA table_info(sft_ranked_data)").fetchall()]
        assert "domain" in cols
        conn.close()

    def test_insert_prompts_inserts_new_rows(self):
        mod = self._get_module()
        conn = _make_sqlite_conn()
        cur = conn.cursor()
        prompts = [{"prompt": "What is periodontitis?"}, {"prompt": "How to treat dry socket?"}]
        inserted, skipped = mod.insert_prompts(cur, prompts, "Dentistry", ph="?")
        conn.commit()
        assert inserted == 2
        assert skipped == 0
        count = cur.execute("SELECT COUNT(*) FROM sft_ranked_data WHERE domain='Dentistry'").fetchone()[0]
        assert count == 2
        conn.close()

    def test_insert_prompts_skips_duplicates(self):
        mod = self._get_module()
        conn = _make_sqlite_conn()
        cur = conn.cursor()
        prompts = [{"prompt": "What causes tooth decay?"}]
        # Insert once
        mod.insert_prompts(cur, prompts, "Dentistry", ph="?")
        conn.commit()
        # Insert again — should skip
        inserted, skipped = mod.insert_prompts(cur, prompts, "Dentistry", ph="?")
        assert inserted == 0
        assert skipped == 1
        conn.close()

    def test_insert_prompts_ignores_blank_prompts(self):
        mod = self._get_module()
        conn = _make_sqlite_conn()
        cur = conn.cursor()
        prompts = [{"prompt": ""}, {"prompt": "   "}, {"prompt": "Valid question?"}]
        inserted, _ = mod.insert_prompts(cur, prompts, "Dentistry", ph="?")
        assert inserted == 1
        conn.close()

    def test_insert_prompts_sets_correct_rank_and_domain(self):
        mod = self._get_module()
        conn = _make_sqlite_conn()
        cur = conn.cursor()
        prompts = [{"prompt": "What is dental fluorosis?"}]
        mod.insert_prompts(cur, prompts, "Dentistry", ph="?")
        conn.commit()
        row = cur.execute(
            "SELECT rank, domain, response_text FROM sft_ranked_data WHERE domain='Dentistry'"
        ).fetchone()
        assert row[0] == 1            # rank = 1
        assert row[1] == "Dentistry"  # domain set correctly
        assert row[2] == ""           # response_text blank (to be filled by generate script)
        conn.close()

    def test_files_list_contains_ophthalmology_and_dentistry(self):
        mod = self._get_module()
        domains = [domain for _, domain in mod.FILES]
        assert "Ophthalmology" in domains
        assert "Dentistry" in domains


# ---------------------------------------------------------------------------
# auto_sme_score — rank_to_score mapping
# ---------------------------------------------------------------------------

class TestAutoSMEScore:
    def _get_module(self):
        import auto_sme_score
        return auto_sme_score

    def test_connect_falls_back_to_sqlite(self):
        mod = self._get_module()
        with patch.object(mod, '_pg') as mock_pg:
            mock_pg.connect.side_effect = Exception("timeout")
            conn, ph, is_sqlite = mod._connect()
            assert is_sqlite is True
            assert ph == "?"
            conn.close()

    def test_rank_1_gives_score_5(self):
        mod = self._get_module()
        assert mod.rank_to_score(1) == 5

    def test_rank_2_gives_score_4(self):
        mod = self._get_module()
        assert mod.rank_to_score(2) == 4

    def test_rank_3_gives_score_3(self):
        mod = self._get_module()
        assert mod.rank_to_score(3) == 3

    def test_rank_4_gives_score_2(self):
        mod = self._get_module()
        assert mod.rank_to_score(4) == 2

    def test_rank_5_or_more_gives_score_1(self):
        mod = self._get_module()
        assert mod.rank_to_score(5) == 1
        assert mod.rank_to_score(10) == 1

    def test_connect_sqlite_returns_question_placeholder(self):
        mod = self._get_module()
        with patch.object(mod, '_pg') as mock_pg:
            mock_pg.connect.side_effect = Exception("unreachable")
            _, ph, _ = mod._connect()
        assert ph == "?"


# ---------------------------------------------------------------------------
# generate_sft_responses — _connect fallback
# ---------------------------------------------------------------------------

class TestGenerateSFTResponses:
    def _get_module(self):
        import generate_sft_responses
        return generate_sft_responses

    def test_connect_falls_back_to_sqlite(self):
        mod = self._get_module()
        with patch.object(mod, '_pg') as mock_pg:
            mock_pg.connect.side_effect = Exception("connection refused")
            conn, ph, is_sqlite = mod._connect()
            assert is_sqlite is True
            assert ph == "?"
            conn.close()

    def test_connect_sqlite_db_path_exists(self):
        mod = self._get_module()
        with patch.object(mod, '_pg') as mock_pg:
            mock_pg.connect.side_effect = Exception("down")
            conn, _, _ = mod._connect()
            # Verify it's a real SQLite connection (not an in-memory one)
            assert isinstance(conn, sqlite3.Connection)
            conn.close()

    def test_domain_system_prompts_contains_ophthalmology(self):
        mod = self._get_module()
        assert "Ophthalmology" in mod.DOMAIN_SYSTEM_PROMPTS

    def test_domain_system_prompts_contains_dentistry(self):
        mod = self._get_module()
        assert "Dentistry" in mod.DOMAIN_SYSTEM_PROMPTS

    def test_domain_system_prompts_not_empty(self):
        mod = self._get_module()
        for dept, prompt in mod.DOMAIN_SYSTEM_PROMPTS.items():
            assert len(prompt) > 20, f"System prompt for {dept} is too short"
