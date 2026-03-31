#!/usr/bin/env python3
"""
Load neurologist_prompts_100.json and cardiologist_prompts_100.json into
sft_ranked_data in PostgreSQL (pces_base).

Each prompt is inserted as a rank-1 placeholder row with a blank response_text.
Responses should be generated and updated afterwards.
Duplicate prompts (matched by exact text) are skipped.
"""

import json
import os
import sys
import uuid

from dotenv import load_dotenv

load_dotenv()

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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SQLITE_DB  = os.path.join(SCRIPT_DIR, "local_sft.db")

_raw = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}
_raw["dbname" if _PG_VERSION == 3 else "database"] = os.getenv("DB_NAME")
DB_CONFIG = {k: v for k, v in _raw.items() if v is not None}


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

FILES = [
    ("neurologist_prompts_100.json", "Neurology"),
    ("cardiologist_prompts_100.json", "Cardiology"),
    ("pediatrician_prompts_100.json", "Pediatrics"),
    ("orthopedic_prompts_100.json", "Orthopedics"),
    ("ophthalmologist_prompts_100.json", "Ophthalmology"),
    ("dentist_prompts_100.json", "Dentistry"),
]


def ensure_domain_column(cur, is_sqlite=False):
    """Add domain column if it doesn't exist yet (idempotent)."""
    if is_sqlite:
        rows = cur.execute("PRAGMA table_info(sft_ranked_data)").fetchall()
        existing = [row[1] for row in rows]
        if "domain" not in existing:
            cur.execute("ALTER TABLE sft_ranked_data ADD COLUMN domain TEXT")
            print("  ℹ️  Added missing 'domain' column to sft_ranked_data")
    else:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'sft_ranked_data' AND column_name = 'domain'"
        )
        if not cur.fetchone():
            cur.execute("ALTER TABLE sft_ranked_data ADD COLUMN domain TEXT")
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_sft_ranked_data_domain "
                "ON sft_ranked_data(domain)"
            )
            print("  ℹ️  Added missing 'domain' column to sft_ranked_data")


def insert_prompts(cur, prompts, domain, ph="%s"):
    inserted = 0
    skipped = 0
    for item in prompts:
        prompt_text = item.get("prompt", "").strip()
        if not prompt_text:
            continue

        cur.execute(
            f"SELECT 1 FROM sft_ranked_data WHERE prompt = {ph} LIMIT 1",
            (prompt_text,),
        )
        if cur.fetchone():
            skipped += 1
            continue

        group_id = str(uuid.uuid4())[:8]
        cur.execute(
            f"""INSERT INTO sft_ranked_data
               (prompt, response_text, rank, reason, group_id, domain)
               VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph})""",
            (prompt_text, "", 1, "", group_id, domain),
        )
        inserted += 1

    return inserted, skipped


def main():
    conn, ph, is_sqlite = _connect()

    total_inserted = 0
    total_skipped = 0

    try:
        cur = conn.cursor()
        ensure_domain_column(cur, is_sqlite=is_sqlite)
        conn.commit()
        cur.close()

        for filename, domain in FILES:
            path = os.path.join(SCRIPT_DIR, filename)
            print(f"\n📂  {filename}  →  domain: {domain}")
            try:
                with open(path) as f:
                    prompts = json.load(f)
            except FileNotFoundError:
                print(f"  ERROR: file not found at {path}")
                continue

            print(f"  Found {len(prompts)} prompts")
            cur = conn.cursor()
            inserted, skipped = insert_prompts(cur, prompts, domain, ph=ph)
            cur.close()
            conn.commit()
            print(f"  ✅ Inserted: {inserted}   ⏭  Skipped (duplicate): {skipped}")
            total_inserted += inserted
            total_skipped += skipped

    finally:
        conn.close()

    print(f"\n{'='*50}")
    print(f"Total inserted : {total_inserted}")
    print(f"Total skipped  : {total_skipped}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
