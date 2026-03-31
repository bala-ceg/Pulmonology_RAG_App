#!/usr/bin/env python3
"""
Automate SME scoring for all sft_ranked_data rows that have a response but
no SME review yet (sme_score IS NULL).

For each row this script:
  1. Calls GPT-4o-mini with the prompt + response + rank to generate a
     concise (~50-word) sme_score_reason.
  2. Derives sme_score from rank:  rank 1 → 5,  rank 2 → 4,  rank 3 → 3,
                                   rank 4 → 2,  rank ≥5 → 1
  3. Sets sme_reviewed_by = "Auto-SME" and sme_reviewed_at = NOW().

The script is idempotent: re-running it only touches rows where
sme_score IS NULL.
"""

import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# DB driver — prefer psycopg (v3), fall back to psycopg2
# ---------------------------------------------------------------------------
try:
    import psycopg as _pg
    _PG_VERSION = 3
except ImportError:
    try:
        import psycopg2 as _pg  # type: ignore
        _PG_VERSION = 2
    except ImportError:
        print("ERROR: neither psycopg nor psycopg2 is installed.")
        sys.exit(1)

_raw = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}
_raw["dbname" if _PG_VERSION == 3 else "database"] = os.getenv("DB_NAME")
DB_CONFIG = {k: v for k, v in _raw.items() if v is not None}
SQLITE_DB  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_sft.db")


def _connect():
    """Try PostgreSQL (5 s timeout); fall back to SQLite. Returns (conn, placeholder, is_sqlite)."""
    try:
        cfg = {**DB_CONFIG, "connect_timeout": 5}
        conn = _pg.connect(**cfg)
        print(f"Connected to PostgreSQL at {DB_CONFIG.get('host')}:{DB_CONFIG.get('port')}/{DB_CONFIG.get('dbname') or DB_CONFIG.get('database')}")
        return conn, "%s", False
    except Exception as pg_err:
        print(f"⚠️  PostgreSQL unavailable ({pg_err}). Falling back to SQLite: {SQLITE_DB}")
        import sqlite3
        conn = sqlite3.connect(SQLITE_DB)
        return conn, "?", True

# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------
try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed.  Run: pip install openai")
    sys.exit(1)

client = OpenAI(
    api_key=os.getenv("openai_api_key"),
    base_url=os.getenv("base_url", "https://api.openai.com/v1"),
)
MODEL = os.getenv("llm_model_name", "gpt-4o-mini")

# ---------------------------------------------------------------------------
# rank → SME score mapping
# ---------------------------------------------------------------------------
RANK_TO_SCORE = {1: 5, 2: 4, 3: 3, 4: 2}

def rank_to_score(rank: int) -> int:
    return RANK_TO_SCORE.get(rank, 1)

# ---------------------------------------------------------------------------
# Domain-specific reviewer personas for the LLM prompt
# ---------------------------------------------------------------------------
DOMAIN_REVIEWER = {
    "Neurology":  "board-certified neurologist",
    "Cardiology": "board-certified cardiologist",
}

SME_REASON_PROMPT = """\
You are a {reviewer} reviewing a clinical AI response for quality.

Clinical question:
{prompt}

AI response (Rank {rank} of 3 — Rank 1 is best):
{response}

Write a concise SME evaluation (exactly 40–55 words) that:
- Explains why this response deserves a score of {score}/5
- Notes its clinical accuracy, completeness, and practical utility
- Is direct and professional

Return ONLY the evaluation text, no labels or preamble."""


def generate_sme_reason(prompt: str, response: str, rank: int, score: int,
                        domain: str) -> str:
    reviewer = DOMAIN_REVIEWER.get(domain, "board-certified physician")
    user_msg = SME_REASON_PROMPT.format(
        reviewer=reviewer,
        prompt=prompt,
        response=response[:1500],   # guard against very long responses
        rank=rank,
        score=score,
    )
    result = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": user_msg}],
        temperature=0.3,
        max_tokens=120,
        timeout=30,
    )
    return result.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    conn, ph, is_sqlite = _connect()

    cur = conn.cursor()
    cur.execute(
        "SELECT id, prompt, response_text, rank, domain "
        "FROM sft_ranked_data "
        "WHERE response_text <> '' AND sme_score IS NULL "
        "ORDER BY domain, rank, id"
    )
    rows = cur.fetchall()
    cur.close()

    if not rows:
        print("No rows need SME scoring — nothing to do.")
        conn.close()
        return

    total = len(rows)
    print(f"Found {total} rows to score.\n")

    updated = 0
    errors  = 0

    for idx, (row_id, prompt, response, rank, domain) in enumerate(rows, 1):
        domain = domain or "General"
        score  = rank_to_score(rank)
        print(f"[{idx:>3}/{total}] {domain[:12]:<12} | rank {rank} → score {score}/5 …",
              end=" ", flush=True)

        try:
            reason = generate_sme_reason(prompt, response, rank, score, domain)
        except Exception as e:
            print(f"LLM ERROR: {e}")
            errors += 1
            time.sleep(2)
            continue

        try:
            cur = conn.cursor()
            if is_sqlite:
                cur.execute(
                    f"UPDATE sft_ranked_data "
                    f"SET sme_score = {ph}, sme_score_reason = {ph}, sme_reviewed_by = 'Auto-SME' "
                    f"WHERE id = {ph}",
                    (score, reason, row_id),
                )
            else:
                cur.execute(
                    f"UPDATE sft_ranked_data "
                    f"SET sme_score = {ph}, sme_score_reason = {ph}, sme_reviewed_by = 'Auto-SME', "
                    f"    sme_reviewed_at = NOW(), updated_at = NOW() "
                    f"WHERE id = {ph}",
                    (score, reason, row_id),
                )
            cur.close()
            conn.commit()
            updated += 1
            print("✅")
        except Exception as e:
            conn.rollback()
            print(f"DB ERROR: {e}")
            errors += 1

        time.sleep(0.4)   # stay within OpenAI rate limits

    conn.close()

    print(f"\n{'='*55}")
    print(f"Rows scored  : {updated}")
    print(f"Errors       : {errors}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
