#!/usr/bin/env python3
"""
RLHF/SFT Training Pipeline
============================
Phase 1 — User Doctor:      Submit 100 cardiology + 100 neurology prompts to the LLM
                             and save prompt+response to sft_ranked_data.
Phase 2 — Neurologist SME:  Login as Neurologist director (Dr. James Wilson),
                             review all pending neurology LLM responses,
                             provide 1–5 star ratings and clinical feedback.
Phase 3 — Cardiologist SME: Login as Cardiologist director (Dr. Sarah Chen),
                             review all pending cardiology LLM responses,
                             provide 1–5 star ratings and clinical feedback.

Progress is checkpointed to run_pipeline_progress.json so the script can
be safely interrupted and resumed.
"""

import json
import os
import sqlite3
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
SQLITE_PATH = BASE_DIR / "local_sft.db"
PROGRESS_FILE = BASE_DIR / "run_pipeline_progress.json"

CARDIOLOGY_JSON = BASE_DIR / "cardiologist_prompts_100.json"
NEUROLOGY_JSON = BASE_DIR / "neurologist_prompts_100.json"

OPENAI_API_KEY = os.getenv("openai_api_key") or os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("llm_model_name", "gpt-4o-mini")

# Neurologist SME director
NEURO_SME_NAME = "Dr. James Wilson"
NEURO_SME_SPECIALTY = "Stroke / Movement Disorders (Neurology Director)"

# Cardiologist SME director
CARDIO_SME_NAME = "Dr. Sarah Chen"
CARDIO_SME_SPECIALTY = "Interventional Cardiology (Cardiology Director)"

REQUEST_DELAY = 0.05  # minimal delay — we rely on API parallelism
MAX_WORKERS = 8       # parallel API calls
LLM_MAX_TOKENS = 900  # sufficient for detailed medical responses

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------
if not OPENAI_API_KEY:
    print("ERROR: OpenAI API key not found in .env (openai_api_key) or env var OPENAI_API_KEY")
    sys.exit(1)

client = OpenAI(api_key=OPENAI_API_KEY)


# ---------------------------------------------------------------------------
# Progress checkpoint helpers
# ---------------------------------------------------------------------------
def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"phase1_cardiology": {}, "phase1_neurology": {},
            "phase2_neuro_sme": {}, "phase3_cardio_sme": {}}


def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------
def get_conn():
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def insert_ranked_entry(conn, prompt, response_text, domain, group_id):
    """Insert a single prompt/response into sft_ranked_data with domain."""
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO sft_ranked_data
           (prompt, response_text, rank, reason, group_id, domain,
            created_by, updated_by, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)""",
        (prompt, response_text, 1,
         f"LLM response for {domain} training prompt", group_id, domain,
         1001, 1001),
    )
    return cur.lastrowid


def update_sme_review(conn, entry_id, sme_score, sme_score_reason, sme_name):
    """Record SME rating and feedback on a ranked data entry."""
    cur = conn.cursor()
    cur.execute(
        """UPDATE sft_ranked_data
           SET sme_score = ?,
               sme_score_reason = ?,
               sme_reviewed_by = ?,
               sme_reviewed_at = CURRENT_TIMESTAMP,
               updated_at = CURRENT_TIMESTAMP
           WHERE id = ?""",
        (sme_score, sme_score_reason, sme_name, entry_id),
    )


def get_pending_entries(conn, domain):
    """Return entries for a domain that haven't been SME-reviewed yet."""
    cur = conn.cursor()
    cur.execute(
        """SELECT id, prompt, response_text FROM sft_ranked_data
           WHERE LOWER(domain) = LOWER(?) AND sme_score IS NULL
           ORDER BY id""",
        (domain,),
    )
    return cur.fetchall()


# ---------------------------------------------------------------------------
# LLM call helpers
# ---------------------------------------------------------------------------
MEDICAL_SYSTEM_PROMPT = """You are a board-certified medical AI assistant providing
evidence-based clinical guidance to licensed physicians. When given a clinical case,
produce a comprehensive, structured response covering:
1. Differential diagnosis assessment
2. Recommended diagnostic workup (labs, imaging, specialist referrals)
3. Evidence-based management approach
4. Key red flags to monitor
5. Patient education points
Use current clinical guidelines (AHA, AAN, ACC, etc.) where applicable.
Keep the response medically rigorous but clearly organized."""


def get_llm_response(prompt: str) -> str:
    """Call the LLM and return the medical response text."""
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": MEDICAL_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=LLM_MAX_TOKENS,
    )
    return response.choices[0].message.content.strip()


def get_sme_review(prompt: str, response_text: str, sme_name: str,
                   sme_specialty: str, domain: str) -> tuple[int, str]:
    """
    Have the SME LLM persona evaluate an AI response.
    Returns (score 1-5, detailed feedback string).
    """
    sme_system = f"""You are {sme_name}, a senior {domain} director and subject matter expert (SME)
reviewing AI-generated medical responses for quality and accuracy in an RLHF training pipeline.
Your specialty: {sme_specialty}.

For each response you review, provide:
1. A star rating from 1 to 5 (1=poor/dangerous, 3=adequate, 5=excellent/comprehensive)
2. Detailed clinical feedback explaining the rating

Rating criteria:
  5 ★★★★★ – Clinically excellent, guidelines-aligned, comprehensive differential, actionable management plan
  4 ★★★★  – Very good, minor omissions, clinically sound overall
  3 ★★★   – Adequate, partially complete, some important points missing
  2 ★★    – Below standard, significant gaps or misleading information
  1 ★     – Unsafe or grossly incorrect clinical guidance

Respond ONLY in this exact JSON format (no extra text):
{{"score": <integer 1-5>, "feedback": "<detailed_clinical_feedback_string>"}}"""

    user_msg = f"""Clinical Query Submitted by Doctor:
{prompt}

AI Response to Review:
{response_text}

Provide your SME evaluation as JSON."""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": sme_system},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
        max_tokens=600,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content.strip()
    data = json.loads(raw)
    score = max(1, min(5, int(data.get("score", 3))))
    feedback = str(data.get("feedback", "No feedback provided."))
    return score, feedback


# ---------------------------------------------------------------------------
# Phase 1 — User Doctor submits prompts (concurrent)
# ---------------------------------------------------------------------------
def _submit_one(item, domain):
    """Worker: call LLM for a single prompt. Returns (pid, prompt, response, group_id)."""
    group_id = str(uuid.uuid4())[:8]
    response_text = get_llm_response(item["prompt"])
    time.sleep(REQUEST_DELAY)
    return str(item["id"]), item["prompt"], response_text, group_id


def phase1_submit_prompts(prompts: list, domain: str, progress_key: str,
                           progress: dict):
    """Submit prompts as a User doctor, get LLM responses, save to DB."""
    done = progress[progress_key]
    pending_items = [item for item in prompts if str(item["id"]) not in done]
    total = len(prompts)

    print(f"\n{'='*60}")
    print(f"PHASE 1 [{domain}] — User Doctor submitting {total} prompts")
    print(f"  Model: {LLM_MODEL}  |  Workers: {MAX_WORKERS}  |  Already done: {len(done)}/{total}")
    print(f"{'='*60}")

    if not pending_items:
        print(f"  ✅ All {total} prompts already submitted. Skipping.")
        return done

    conn = get_conn()
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_submit_one, item, domain): item for item in pending_items}
        for future in as_completed(futures):
            try:
                pid, prompt_text, response_text, group_id = future.result()
                entry_id = insert_ranked_entry(conn, prompt_text, response_text, domain, group_id)
                conn.commit()
                done[pid] = {"entry_id": entry_id, "group_id": group_id}
                completed += 1
                overall = len(done)
                print(f"  [{overall:3d}/{total}] id={pid} — saved (db_id={entry_id})")
            except Exception as e:
                print(f"  ERROR on prompt {futures[future]['id']}: {e}")

    save_progress(progress)
    conn.close()
    print(f"\n  ✅ Phase 1 [{domain}] complete — {completed} new entries added.")
    return done


# ---------------------------------------------------------------------------
# Phase 2/3 — SME reviews pending entries (concurrent)
# ---------------------------------------------------------------------------
def _review_one(row, sme_name, sme_specialty, domain):
    """Worker: call LLM for a single SME review. Returns (entry_id, score, feedback)."""
    score, feedback = get_sme_review(
        row["prompt"], row["response_text"],
        sme_name, sme_specialty, domain
    )
    time.sleep(REQUEST_DELAY)
    return row["id"], score, feedback


def phase_sme_review(domain: str, sme_name: str, sme_specialty: str,
                     progress_key: str, progress: dict):
    """SME logs in, reviews all pending responses, saves star ratings + feedback."""
    done_reviews = progress[progress_key]
    conn = get_conn()

    all_pending = get_pending_entries(conn, domain)
    pending = [r for r in all_pending if str(r["id"]) not in done_reviews]
    total_pending = len(all_pending)

    print(f"\n{'='*60}")
    print(f"SME LOGIN → {sme_name}  ({sme_specialty})")
    print(f"  Domain: {domain}  |  Workers: {MAX_WORKERS}")
    print(f"  Pending reviews: {total_pending}  |  Remaining: {len(pending)}")
    print(f"{'='*60}")

    if not pending:
        print(f"  ✅ No pending entries to review for {domain}.")
        conn.close()
        return

    reviewed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_review_one, row, sme_name, sme_specialty, domain): row
                   for row in pending}
        for future in as_completed(futures):
            try:
                entry_id, score, feedback = future.result()
                update_sme_review(conn, entry_id, score, feedback, sme_name)
                conn.commit()
                done_reviews[str(entry_id)] = {"score": score, "feedback_preview": feedback[:80]}
                reviewed += 1
                stars = "⭐" * score
                print(f"  [{reviewed:3d}/{len(pending)}] db_id={entry_id} — {stars} ({score}/5)  {feedback[:55]}…")
            except Exception as e:
                print(f"  ERROR reviewing entry {futures[future]['id']}: {e}")

    save_progress(progress)
    conn.close()
    print(f"\n  ✅ SME review [{domain}] complete — {reviewed} entries rated by {sme_name}.")


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------
def print_summary():
    conn = get_conn()
    cur = conn.cursor()
    print(f"\n{'='*60}")
    print("PIPELINE SUMMARY")
    print(f"{'='*60}")
    for domain in ("Cardiology", "Neurology"):
        cur.execute(
            "SELECT COUNT(*), SUM(CASE WHEN sme_score IS NOT NULL THEN 1 ELSE 0 END), "
            "AVG(CASE WHEN sme_score IS NOT NULL THEN sme_score END) "
            "FROM sft_ranked_data WHERE LOWER(domain) = LOWER(?)",
            (domain,),
        )
        total, reviewed, avg_score = cur.fetchone()
        avg_str = f"{avg_score:.2f}" if avg_score else "N/A"
        print(f"  {domain:<15} — total: {total:>3}  |  SME-reviewed: {reviewed:>3}  |  avg score: {avg_str}")
    conn.close()
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("\n" + "="*60)
    print("  RLHF/SFT TRAINING PIPELINE")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Database: {SQLITE_PATH}")
    print(f"  Model:    {LLM_MODEL}")
    print("="*60)

    progress = load_progress()

    # Load prompt files
    with open(CARDIOLOGY_JSON) as f:
        cardio_prompts = json.load(f)
    with open(NEUROLOGY_JSON) as f:
        neuro_prompts = json.load(f)

    print(f"\n  Loaded {len(cardio_prompts)} cardiology prompts")
    print(f"  Loaded {len(neuro_prompts)} neurology prompts")

    # ------------------------------------------------------------------
    # PHASE 1a — User Doctor submits Cardiology prompts
    # ------------------------------------------------------------------
    phase1_submit_prompts(cardio_prompts, "Cardiology",
                          "phase1_cardiology", progress)

    # ------------------------------------------------------------------
    # PHASE 1b — User Doctor submits Neurology prompts
    # ------------------------------------------------------------------
    phase1_submit_prompts(neuro_prompts, "Neurology",
                          "phase1_neurology", progress)

    # ------------------------------------------------------------------
    # PHASE 2 — Neurologist SME reviews all pending Neurology responses
    # ------------------------------------------------------------------
    phase_sme_review("Neurology", NEURO_SME_NAME, NEURO_SME_SPECIALTY,
                     "phase2_neuro_sme", progress)

    # ------------------------------------------------------------------
    # PHASE 3 — Cardiologist SME reviews all pending Cardiology responses
    # ------------------------------------------------------------------
    phase_sme_review("Cardiology", CARDIO_SME_NAME, CARDIO_SME_SPECIALTY,
                     "phase3_cardio_sme", progress)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print_summary()
    print(f"  Progress checkpoint: {PROGRESS_FILE}")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    main()
