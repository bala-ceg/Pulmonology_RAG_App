"""
Test AI Summary generation time across different OpenAI models.

Calls the EHR database and OpenAI API DIRECTLY — no Flask restart needed.
Each model is tested in the same process run.

Usage:
    python tests/test_ai_summary_models.py

Requirements:
    pip install psycopg2-binary openai python-dotenv
    .env must contain: PG_TOOL_*, openai_api_key, base_url

Patient tested: 11111111-1111-1111-1111-111111111001
"""

import os
import sys
import time
from datetime import date as _date
from pathlib import Path

# ── Load .env from repo root ────────────────────────────────────────────────
_env_file = Path(__file__).resolve().parent.parent / ".env"
if not _env_file.exists():
    print(f"❌  .env not found at {_env_file}")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv(_env_file, override=True)
except ImportError:
    # Manual fallback parser
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

# ── Config ──────────────────────────────────────────────────────────────────
PATIENT_ID = "11111111-1111-1111-1111-111111111001"

MODELS_TO_TEST = [
    # ── GPT-3.5 ────────────────────────────────────────────────────────────
    "gpt-3.5-turbo",          # fast, cheap baseline

    # ── GPT-4 ──────────────────────────────────────────────────────────────
    "gpt-4o-mini",            # cost-optimised GPT-4o
    "gpt-4o",                 # full GPT-4o
    "gpt-4-turbo",            # GPT-4 Turbo

    # ── GPT-5 — uncomment when your API key has access ─────────────────────
    # "gpt-5",                # base GPT-5
    # "gpt-5-mini",           # cost-optimised GPT-5
    # "gpt-5-turbo",          # GPT-5 Turbo variant (if released)

    # ── OpenAI reasoning models ────────────────────────────────────────────
    # "o3",                   # o3 reasoning model
    # "o3-mini",              # o3-mini
    # "o4-mini",              # o4-mini reasoning model

    # ── To add any model: append its exact OpenAI API model ID above,
    #    then re-run:  python3 tests/test_ai_summary_models.py
]

PROMPT_TEMPLATE = (
    "You are a clinical decision-support assistant. "
    "Given the following complete patient encounter history and known allergies, "
    "write a concise AI-generated clinical summary in no more than 500 words covering:\n"
    "1. Clinical presentation and key diagnoses\n"
    "2. Treatment considerations and risk factors\n"
    "3. Allergy implications for management\n\n"
    "{history_block}"
    "\n\nKnown Allergies: {allergy_text}\n\n"
    "Write the summary in a professional clinical tone suitable for a treating physician. "
    "Keep the total response under 500 words."
)

# ── DB helpers ───────────────────────────────────────────────────────────────
def _fetch_patient_data(patient_id: str):
    """Fetch encounter history + allergies from pces_ehr_ccm directly."""
    try:
        import psycopg2
    except ImportError:
        print("❌  psycopg2-binary not installed. Run: pip install psycopg2-binary")
        sys.exit(1)

    conn = psycopg2.connect(
        host=os.getenv("PG_TOOL_HOST"),
        port=os.getenv("PG_TOOL_PORT"),
        dbname=os.getenv("PG_TOOL_NAME"),
        user=os.getenv("PG_TOOL_USER"),
        password=os.getenv("PG_TOOL_PASSWORD"),
        connect_timeout=15,
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT d.code, d.description,
                       provider.first_name, provider.last_name,
                       e.encounter_date
                FROM p_encounter e
                INNER JOIN p_diagnosis d    ON d.encounter_id = e.encounter_id
                INNER JOIN p_party provider ON e.provider_id  = provider.party_id
                WHERE e.patient_id = %s
                ORDER BY e.encounter_date DESC, d.diagnosis_id
                """,
                (patient_id,),
            )
            history_rows = cur.fetchall()

            cur.execute(
                "SELECT allergen FROM p_allergy WHERE patient_id = %s ORDER BY allergen",
                (patient_id,),
            )
            allergens = [r[0] for r in cur.fetchall() if r[0]]
    finally:
        conn.close()

    return history_rows, allergens


def _build_prompt(history_rows, allergens):
    lines = []
    for code, desc, doc_first, doc_last, enc_date in history_rows:
        try:
            d = enc_date if hasattr(enc_date, "strftime") else _date.fromisoformat(str(enc_date)[:10])
            date_str = d.strftime("%b %d, %Y")
        except Exception:
            date_str = str(enc_date)[:10]
        lines.append(f"- {date_str}: [{code}] {desc} (Provider: Dr. {doc_first} {doc_last})")

    history_block = (
        f"Patient Encounter History ({len(lines)} records, most recent first):\n"
        + "\n".join(lines)
    )
    allergy_text = ", ".join(allergens) if allergens else "No known allergies"
    return PROMPT_TEMPLATE.format(history_block=history_block, allergy_text=allergy_text), len(lines)


# ── LLM call ────────────────────────────────────────────────────────────────
def _call_llm(model: str, prompt: str):
    """Call OpenAI directly and return (summary_text, elapsed_ms)."""
    try:
        from openai import OpenAI
    except ImportError:
        print("❌  openai package not installed. Run: pip install openai")
        sys.exit(1)

    client = OpenAI(
        api_key=os.getenv("openai_api_key"),
        base_url=os.getenv("base_url") or None,
    )

    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        timeout=120,
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000)
    summary = response.choices[0].message.content or ""
    return summary, elapsed_ms


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  PCES AI Summary — Multi-Model Timing Test (Direct Mode)")
    print(f"  Patient : {PATIENT_ID}")
    print(f"  Mode    : Direct DB + OpenAI API call (no Flask restart needed)")
    print("=" * 65)

    # Pre-flight: fetch patient data once (shared across all model runs)
    print("\n  Fetching patient data from pces_ehr_ccm …", end=" ", flush=True)
    try:
        history_rows, allergens = _fetch_patient_data(PATIENT_ID)
    except Exception as exc:
        print(f"\n❌  DB connection failed: {exc}")
        sys.exit(1)

    if not history_rows:
        print(f"\n❌  No encounter history for patient {PATIENT_ID}")
        sys.exit(1)

    print(f"OK ({len(history_rows)} encounters, allergies: {allergens or ['none']})")

    prompt, encounter_count = _build_prompt(history_rows, allergens)
    print(f"  Prompt  : {len(prompt)} chars | Encounters: {encounter_count}\n")

    results = []

    for model in MODELS_TO_TEST:
        print(f"─── Testing: {model} " + "─" * max(1, 45 - len(model)))

        # Call 1 — cold
        print(f"  Call 1 (cold) … ", end="", flush=True)
        try:
            summary1, ms1 = _call_llm(model, prompt)
            words1 = len(summary1.split())
            print(f"{ms1}ms | {words1} words | {len(summary1)} chars")
        except Exception as exc:
            print(f"FAILED — {exc}")
            results.append({"model": model, "error": str(exc)})
            continue

        # Call 2 — warm
        print(f"  Call 2 (warm) … ", end="", flush=True)
        try:
            summary2, ms2 = _call_llm(model, prompt)
            words2 = len(summary2.split())
            print(f"{ms2}ms | {words2} words | {len(summary2)} chars")
        except Exception as exc:
            print(f"FAILED — {exc}")
            ms2, words2 = None, None

        results.append({
            "model"     : model,
            "encounters": encounter_count,
            "ms1"       : ms1,
            "ms2"       : ms2,
            "words1"    : words1,
            "words2"    : words2,
            "chars"     : len(summary1),
            "error"     : None,
        })
        print()

    # ── Summary table ────────────────────────────────────────────────────────
    print("=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)
    print(f"  {'Model':<20} {'Enc':>4} {'Call1':>7} {'Call2':>7} {'Words1':>6} {'Words2':>6} {'Chars':>6}")
    print("  " + "-" * 63)
    for r in results:
        if r.get("error"):
            print(f"  {r['model']:<20}  ❌ {r['error']}")
        else:
            ms1_s  = f"{r['ms1']}ms"
            ms2_s  = f"{r['ms2']}ms" if r['ms2'] else "—"
            print(f"  {r['model']:<20} {str(r['encounters']):>4} {ms1_s:>7} {ms2_s:>7} "
                  f"{str(r['words1']):>6} {str(r['words2'] or '—'):>6} {str(r['chars']):>6}")
    print("=" * 65)
    print("  Call1=cold, Call2=warm | all times = direct OpenAI API\n")

    # Print best summary (first successful model, call 1)
    for r in results:
        if not r.get("error"):
            print("─── Sample Summary (first model, call 1) ───────────────────")
            _, allergens2 = _fetch_patient_data(PATIENT_ID)
            allergy_text = ", ".join(allergens2) if allergens2 else "None"
            print(f"Known Allergies: {allergy_text}")
            print()
            # Re-run call 1 for the first model to get the actual text
            smry, _ = _call_llm(r['model'], prompt)
            print(smry)
            print("─" * 65)
            break


if __name__ == "__main__":
    main()
