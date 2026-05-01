"""
RAG Routing Pipeline Test Script
=================================
Tests all 5 routing rules + 6 tools end-to-end.

Priority rules under test:
  1. Patient history       → PostgreSQL_Diagnosis_Search
  2. Medical research      → ArXiv_Search + Tavily_Search (both)
  3. General knowledge     → Wikipedia_Search
  4. Uploaded documents    → Internal_VectorDB
  Default fallback         → Pinecone_KB_Search

Usage:
    python test_rag_routing.py
"""

import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()

# ── Colour helpers ─────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def banner(text: str) -> None:
    print(f"\n{BOLD}{CYAN}{'='*70}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'='*70}{RESET}")

def section(label: str) -> None:
    print(f"\n{BOLD}{YELLOW}── {label} {'─'*(60-len(label))}{RESET}")

def ok(msg: str) -> None:
    print(f"{GREEN}✅ {msg}{RESET}")

def fail(msg: str) -> None:
    print(f"{RED}❌ {msg}{RESET}")

def info(msg: str) -> None:
    print(f"   {msg}")

# ── Imports ────────────────────────────────────────────────────────────────
banner("PCES RAG Routing Pipeline — End-to-End Test")

print("\nLoading modules …")
try:
    from rag_architecture import MedicalQueryRouter
    print("  ✓ MedicalQueryRouter")
except Exception as e:
    print(f"  ✗ MedicalQueryRouter: {e}")
    sys.exit(1)

try:
    from tools import (
        Wikipedia_Search,
        ArXiv_Search,
        Tavily_Search,
        Internal_VectorDB,
        PostgreSQL_Diagnosis_Search,
        Pinecone_KB_Search,
    )
    print("  ✓ All 6 tools imported")
except Exception as e:
    print(f"  ✗ tools import: {e}")
    sys.exit(1)

try:
    from pinecone_kb import get_pinecone_kb, PINECONE_KB_AVAILABLE
    print(f"  ✓ Pinecone KB available: {PINECONE_KB_AVAILABLE}")
except Exception as e:
    print(f"  ✗ pinecone_kb: {e}")
    PINECONE_KB_AVAILABLE = False

router = MedicalQueryRouter()

# ── Test cases ─────────────────────────────────────────────────────────────

TEST_CASES = [
    # (label, query, expected_primary_tool, invoke_tool_fn)
    (
        "PRIORITY 1 — Patient History → PostgreSQL",
        "Show me patient history for a patient admitted with chest pain",
        "PostgreSQL_Diagnosis_Search",
        PostgreSQL_Diagnosis_Search,
    ),
    (
        "PRIORITY 2a — Medical Research → ArXiv",
        "What are the latest medical research papers on atrial fibrillation treatment?",
        "ArXiv_Search",
        ArXiv_Search,
    ),
    (
        "PRIORITY 2b — Medical Research → Tavily (real-time)",
        "What are the current FDA guidelines and recent clinical trial updates for COPD?",
        "Tavily_Search",
        Tavily_Search,
    ),
    (
        "PRIORITY 3a — General Knowledge → Wikipedia",
        "What is atrial fibrillation?",
        "Wikipedia_Search",
        Wikipedia_Search,
    ),
    (
        "PRIORITY 3b — General Search → Wikipedia",
        "Tell me about hypertension causes and symptoms",
        "Wikipedia_Search",
        Wikipedia_Search,
    ),
    (
        "DEFAULT — PCES Org KB → Pinecone",
        "What is the PCES cardiology treatment protocol for heart failure?",
        "Pinecone_KB_Search",
        Pinecone_KB_Search,
    ),
    (
        "DEFAULT — Department-specific → Pinecone",
        "Management of asthma in pulmonology — standard of care",
        "Pinecone_KB_Search",
        Pinecone_KB_Search,
    ),
]

# ── Run tests ──────────────────────────────────────────────────────────────

results = []

for label, query, expected_primary, tool_fn in TEST_CASES:
    section(label)
    print(f"  Query: {BOLD}{query}{RESET}")

    # Step 1: Route
    t0 = time.perf_counter()
    route_result = router.route_tools(query, session_id=None)
    t_route = time.perf_counter() - t0

    primary     = route_result["primary_tool"]
    selected    = route_result["ranked_tools"]
    confidence  = route_result["confidence"]
    reasoning   = route_result["reasoning"]
    scores      = route_result["tool_scores"]

    routing_ok = primary == expected_primary
    results.append((label, routing_ok))

    status = ok if routing_ok else fail
    status(f"Routing → {BOLD}{primary}{RESET}  (expected: {expected_primary})  [{confidence} confidence, {t_route*1000:.0f}ms]")
    info(f"Selected tools : {selected}")
    info(f"Reasoning      : {reasoning}")

    # Print top-3 scores
    top3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
    info(f"Top-3 scores   : {', '.join(f'{k}={v}' for k,v in top3)}")

    # Step 2: Invoke the tool and print response
    print(f"\n  {CYAN}Invoking {tool_fn.name} …{RESET}")
    try:
        t1 = time.perf_counter()
        if tool_fn.name == "Internal_VectorDB":
            response = tool_fn.invoke({"query": query, "session_id": None, "rag_manager": None})
        else:
            response = tool_fn.invoke(query)
        t_invoke = time.perf_counter() - t1

        # Truncate for display
        preview = response[:800].strip()
        if len(response) > 800:
            preview += "\n  … [truncated]"

        print(f"\n  {BOLD}Response ({t_invoke:.1f}s):{RESET}")
        for line in preview.splitlines():
            print(f"  {line}")

    except Exception as exc:
        fail(f"Tool invocation error: {exc}")

print()

# ── Routing-only bonus tests (no tool invocation) ──────────────────────────
banner("Routing Accuracy — Additional Scenarios")

ROUTING_ONLY = [
    ("Medical research: systematic review on stroke thrombolysis",   "ArXiv_Search"),
    ("EHR medical history for patient ID 1234",                       "PostgreSQL_Diagnosis_Search"),
    ("Define pneumonia",                                               "Wikipedia_Search"),
    ("PCES neurology guidelines",                                      "Pinecone_KB_Search"),
    ("Search my uploaded documents for diabetes management",           "Internal_VectorDB"),
    ("What is the treatment of COPD",                                  "Pinecone_KB_Search"),
    ("Latest clinical evidence for metformin in diabetes",             "ArXiv_Search"),
    ("How does the kidney work?",                                      "Wikipedia_Search"),
]

routing_pass = routing_fail = 0
for query, expected in ROUTING_ONLY:
    r = router.route_tools(query)
    primary = r["primary_tool"]
    if primary == expected:
        ok(f"[{primary}] {query}")
        routing_pass += 1
    else:
        fail(f"[got={primary}, want={expected}] {query}")
        routing_fail += 1

# ── Summary ────────────────────────────────────────────────────────────────
banner("Test Summary")

full_pass = sum(1 for _, ok_ in results if ok_)
full_fail = len(results) - full_pass

print(f"\n  End-to-End tests  : {GREEN}{full_pass} passed{RESET}, {RED}{full_fail} failed{RESET}  (total {len(results)})")
print(f"  Routing-only tests: {GREEN}{routing_pass} passed{RESET}, {RED}{routing_fail} failed{RESET}  (total {len(ROUTING_ONLY)})")
total_pass = full_pass + routing_pass
total_fail = full_fail + routing_fail
total      = total_pass + total_fail
status_str = f"{GREEN}ALL PASS{RESET}" if total_fail == 0 else f"{RED}{total_fail} FAILED{RESET}"
print(f"\n  Overall           : {BOLD}{total_pass}/{total}{RESET} — {status_str}")
print()
