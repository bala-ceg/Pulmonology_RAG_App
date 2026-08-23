"""
Test AI Summary generation time across different OpenAI models.

Usage:
    python tests/test_ai_summary_models.py

Requirement: Flask app must be running on http://localhost:5000
Patient tested: 11111111-1111-1111-1111-111111111001
"""

import os
import sys
import time
import json
import subprocess
import urllib.request
import urllib.error

PATIENT_ID   = "11111111-1111-1111-1111-111111111001"
APP_URL      = "http://localhost:5000"
SUMMARY_URL  = f"{APP_URL}/api/patient/{PATIENT_ID}/ai_summary"
ENV_FILE     = os.path.join(os.path.dirname(__file__), "..", ".env")

MODELS_TO_TEST = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-turbo",
    # ── GPT-5 models — uncomment when your API key has access ──────────
    # "gpt-5",               # base GPT-5 (when GA)
    # "gpt-5-mini",          # cost-optimised GPT-5 variant
    # "o3",                  # OpenAI o3 reasoning model
    # "o4-mini",             # OpenAI o4-mini reasoning model
    # ───────────────────────────────────────────────────────────────────
    # To add any model: append its exact API model ID string above,
    # then run:  python tests/test_ai_summary_models.py
]


def _read_env(path: str) -> dict:
    env = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    env[k.strip()] = v.strip()
    return env


def _write_env_model(path: str, model: str):
    """Patch llm_model_name in .env in-place."""
    with open(path) as f:
        lines = f.readlines()
    patched = False
    for i, line in enumerate(lines):
        if line.strip().startswith("llm_model_name"):
            lines[i] = f"llm_model_name={model}\n"
            patched = True
            break
    if not patched:
        lines.append(f"llm_model_name={model}\n")
    with open(path, "w") as f:
        f.writelines(lines)


def _call_summary() -> dict:
    """Call the summary endpoint and return timing + JSON data."""
    req = urllib.request.Request(SUMMARY_URL)
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            elapsed_ms = round((time.perf_counter() - t0) * 1000)
            body = json.loads(resp.read().decode())
            body["_client_ms"] = elapsed_ms
            return body
    except urllib.error.HTTPError as e:
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        body = json.loads(e.read().decode())
        body["_client_ms"] = elapsed_ms
        body["_http_error"] = e.code
        return body
    except Exception as exc:
        return {"error": str(exc), "_client_ms": 0}


def _check_app_running() -> bool:
    try:
        with urllib.request.urlopen(f"{APP_URL}/health", timeout=5):
            return True
    except Exception:
        return False


def _restart_app():
    """Signal the user to restart — we can't restart Flask from here safely."""
    print("  ⚠️  Please restart the Flask app now (python main.py) and press Enter when ready...")
    input()


def main():
    print("=" * 65)
    print("  PCES AI Summary — Multi-Model Timing Test")
    print(f"  Patient : {PATIENT_ID}")
    print(f"  Endpoint: {SUMMARY_URL}")
    print("=" * 65)

    if not _check_app_running():
        print(f"\n❌  App is NOT running at {APP_URL}.")
        print("    Start with: python main.py   then re-run this script.\n")
        sys.exit(1)

    original_env = _read_env(ENV_FILE)
    original_model = original_env.get("llm_model_name", "(not set)")
    print(f"\n  Current model in .env: {original_model}\n")

    results = []

    for model in MODELS_TO_TEST:
        print(f"─── Testing model: {model} " + "─" * (40 - len(model)))

        # Patch .env
        print(f"  [1/3] Patching .env  → llm_model_name={model}")
        _write_env_model(ENV_FILE, model)

        # Ask user to restart
        _restart_app()

        # Health check
        if not _check_app_running():
            print(f"  ❌  App not responding after restart — skipping {model}\n")
            results.append({"model": model, "error": "App not running after restart"})
            continue

        # Call summary endpoint twice (cold + warm)
        print(f"  [2/3] Calling /api/patient/…/ai_summary (call 1 — cold)")
        r1 = _call_summary()
        print(f"        server={r1.get('generation_time_ms','?')}ms  client={r1.get('_client_ms','?')}ms  "
              f"encounters={r1.get('encounter_count','?')}  chars={len(r1.get('summary',''))}")

        print(f"  [3/3] Calling /api/patient/…/ai_summary (call 2 — warm)")
        r2 = _call_summary()
        print(f"        server={r2.get('generation_time_ms','?')}ms  client={r2.get('_client_ms','?')}ms  "
              f"encounters={r2.get('encounter_count','?')}  chars={len(r2.get('summary',''))}")

        results.append({
            "model"          : r1.get("model", model),
            "encounters"     : r1.get("encounter_count", "?"),
            "call1_server_ms": r1.get("generation_time_ms"),
            "call1_client_ms": r1.get("_client_ms"),
            "call2_server_ms": r2.get("generation_time_ms"),
            "call2_client_ms": r2.get("_client_ms"),
            "chars"          : len(r1.get("summary", "")),
            "error"          : r1.get("error") or r2.get("error"),
        })
        print()

    # Restore original model
    print(f"─── Restoring original model: {original_model}")
    _write_env_model(ENV_FILE, original_model)
    print("    (Restart app to apply)\n")

    # Summary table
    print("=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)
    print(f"  {'Model':<20} {'Encounters':>10} {'S-Call1':>8} {'S-Call2':>8} {'C-Call1':>8} {'Chars':>7}")
    print("  " + "-" * 63)
    for r in results:
        if r.get("error"):
            print(f"  {r['model']:<20}  ERROR: {r['error']}")
        else:
            print(f"  {str(r['model']):<20} {str(r['encounters']):>10} "
                  f"{str(r['call1_server_ms'])+'ms':>8} {str(r['call2_server_ms'])+'ms':>8} "
                  f"{str(r['call1_client_ms'])+'ms':>8} {str(r['chars']):>7}")
    print("=" * 65)
    print("  S = server-side LLM time | C = client round-trip time\n")


if __name__ == "__main__":
    main()
