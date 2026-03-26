# Copilot Instructions — Medical RAG Application

## Commands

```bash
# Start the app (Flask on http://localhost:5000)
python main.py

# Install dependencies
pip install -r requirements.txt

# Initialize databases
python setup_local_db.py           # PostgreSQL / SQLite schema
python setup_external_kb.py        # Wikipedia + arXiv vector KB
python setup_organization_kb.py    # Discipline-specific KBs

# Run tests (no test runner — execute files directly)
python tests/test_complete_integration.py   # Core RAG integration
python test_rlhf_pipeline.py                # RLHF pipeline
python tests/test_postgres_integration.py   # PostgreSQL tool
python test_wiki_arxiv_connectivity.py      # External source connectivity

# Single test: run any test file directly, e.g.
python tests/test_routing.py

# Train reward model
python train_reward_sbert.py

# Docker
docker build -t medical-rag-app .
docker run -p 5000:5000 --env-file .env medical-rag-app
```

## Architecture

This is a **multi-source medical RAG system** built on Flask + LangChain + OpenAI + ChromaDB. The main entry point is `main.py` (~5 000 lines).

### Two-Store RAG with TF-IDF Lexical Gate (`rag_architecture.py`)

All queries pass through `TFIDFLexicalGate` before retrieval. The gate computes TF-IDF cosine similarity against the local KB corpus to decide routing:

- Score ≥ 0.3 → query `kb_local` first (uploaded PDFs/URLs)
- Score < 0.3 → go directly to external sources (Wikipedia, arXiv, Tavily)

`TwoStoreRAGManager` manages three ChromaDB stores:
- `kb_local` — uploaded PDFs and URLs (`./KB/`)
- `kb_external` — Wikipedia + arXiv documents (`./vector_dbs/`)
- `vector_dbs/organization/{discipline}` — specialty-specific documents

Each query also gets a per-session vector DB for conversation context.

### Tool-Based Agent (`integrated_rag.py`, `tools.py`)

`IntegratedMedicalRAG` initialises a LangChain ZERO_SHOT_REACT agent with five tools. Each tool uses the `@tool` decorator with a descriptive docstring — the docstring is the primary signal the LLM uses for routing:

| Tool | When it's used |
|------|---------------|
| `Wikipedia_Search` | Definitions, factual explanations, general medical knowledge |
| `ArXiv_Search` | "Latest research", recent papers, scientific evidence |
| `Tavily_Search` | Real-time / current medical information |
| `Internal_VectorDB` | User's uploaded documents / specific files |
| `PostgreSQL_Diagnosis_Search` | Structured diagnosis data from `pces_ehr_ccm` |

`guarded_retrieve()` in `tools.py` post-filters results by similarity threshold (0.35) and falls back to Wikipedia when local results are low-confidence.

### RLHF + SFT Pipeline

**RLHF loop:**
1. All query-response pairs logged to `rlhf_interactions` (PostgreSQL `pces_base`)
2. Admin rates responses 1–5 at `/admin/rlhf`
3. `train_reward_sbert.py` trains a logistic regression classifier on SBERT embeddings (`all-MiniLM-L6-v2`)
4. Trained model saved as `reward_model.joblib`
5. `rlhf_reranker.py` loads that model at inference to re-rank candidate responses

**SFT loop:**
1. SME reviewers score ranked responses in the SME Review Queue UI
2. `sft_experiment_manager.py` manages fine-tuning jobs stored in `sft_experiments` table
3. Training uses `microsoft/phi-2` + LoRA adapters (rank=16, alpha=32, ~0.5 % parameters)
4. Experiments are department-scoped (30 specialties, keyword-matched from `sft_ranked_data.domain`)

### Database Layer

The app auto-detects and falls back between backends:

```
PostgreSQL (primary) → SQLite local_sft.db (fallback if PostgreSQL unreachable)
```

SQL parameter syntax is adapted automatically (`%s` for PostgreSQL, `?` for SQLite).

Two PostgreSQL databases:
- `pces_base` — RLHF interactions, reward model training history, users, SFT experiments
- `pces_ehr_ccm` — Medical diagnosis records (`p_diagnosis` table)

Both psycopg v3 (`psycopg`) and v2 (`psycopg2-binary`) are required — `main.py` uses v3; `postgres_tool.py` and some utilities use v2.

### Optional Subsystems

Each optional subsystem uses a try/except import pattern and exposes an `*_AVAILABLE` boolean flag:

- **Azure Blob Storage** (`azure_storage.py`) — PDF/audio cloud storage; `AZURE_AVAILABLE`
- **Voice Diarization** (`voice_diarization.py`) — Whisper + pyannote.audio; `DIARIZATION_AVAILABLE`
- **Integrated RAG** (`integrated_rag.py`) — Full agent mode; `INTEGRATED_RAG_AVAILABLE`

The app degrades gracefully when any of these are missing.

## Key Conventions

### Tool naming
Tool functions use `CamelCase_With_Underscores` (e.g. `Wikipedia_Search`, `Internal_VectorDB`). The docstring of every `@tool` function is the routing signal — keep it precise and discriminative.

### Result formatting
`_join_docs(docs, max_chars=1200)` in `tools.py` is the standard way to join `Document` objects into a string. It truncates at `max_chars`, extracts sources from `doc.metadata`, and appends a `Sources:` footer. Use this helper (or mirror its pattern) for any new tool.

### Timeout handling
Use threading-based timeouts everywhere — signal-based timeouts are unsafe in Flask/WSGI. The pattern in `enhanced_tools.py` is the reference implementation.

### Environment variables
All secrets and configuration come from a `.env` file loaded by `python-dotenv`. Required keys:

```
openai_api_key, base_url, llm_model_name, embedding_model_name
DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD          # pces_base
PG_TOOL_HOST, PG_TOOL_PORT, PG_TOOL_NAME, PG_TOOL_USER, PG_TOOL_PASSWORD  # pces_ehr_ccm
TAVILY_API_KEY, apify_api_key, HUGGINGFACE_TOKEN          # optional integrations
AZURE_STORAGE_CONNECTION_STRING, AZURE_STORAGE_CONTAINER_NAME  # optional
```

### Medical discipline configuration
`config/disciplines.json` defines the medical specialties. Each entry requires `id`, `name`, `kb_path`, and `vector_db_path`. The `is_session_based: true` flag marks the Doctor's Files discipline whose vector DB is scoped per session. Max 4 disciplines can be selected simultaneously (`selection_rules.max_selections`).

### Vector DB paths
- Session-based retrieval: `./vector_dbs/{session_id}/`
- Organization KBs: `./vector_dbs/organization/{discipline_id}/`
- Local KB: `./vector_dbs/` (root)

When adding new document types, persist metadata (`source`, `source_type`, `title`) on `Document.metadata` so `_join_docs` can build correct `Sources:` footers.
