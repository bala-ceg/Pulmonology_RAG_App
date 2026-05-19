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

This is a **multi-source medical RAG system** built on Flask + LangChain + OpenAI + ChromaDB.

`main.py` is the **bootstrap entry point only** — it loads config, initialises the DB pool, LLM, and all RAG subsystems, stores shared state on `app.config`, registers Blueprints, and starts Flask. All route logic lives in `routes/`; all business logic lives in `services/`.

### Project Layout

```
main.py                    # Bootstrap: config → pool → services → blueprints → run
config.py                  # Config class — all runtime parameters via env vars
config/disciplines.json    # Medical specialty definitions

routes/                    # Flask Blueprints (one file per concern)
  audio.py                 # /transcribe, /translate_audio, /transcribe_patient_notes
  azure_routes.py          # Azure Blob Storage upload/download
  conversation.py          # Conversation history management
  disciplines.py           # /, /api/disciplines, /api/validate_disciplines, MedicalQueryRouter
  doctors.py               # Doctor management endpoints
  documents.py             # /upload_pdf, /upload_url, /upload_organization_kb, /create_vector_db
  pdf_generation.py        # PDF report generation
  pinecone_routes.py       # Pinecone vector DB routes
  query.py                 # /data, /data-html, /generate_summary, /plain_english
  rlhf.py                  # /admin/rlhf, RLHF rating endpoints
  sft.py                   # /api/rlhf/experiments, /api/rlhf/ranked-data/*, SFT endpoints

services/                  # Business logic singletons
  llm_service.py           # LLMService / llm_service — ChatOpenAI + OpenAIEmbeddings
  rag_service.py           # RAGService / rag_service — TwoStoreRAGManager + IntegratedMedicalRAG + DomainScopeGuard
  db_service.py            # init_pool(), get_connection(), execute_query()
  dept_lora_service.py     # DeptLoRAService / dept_lora_service — LRU-cached LoRA models
  audio_service.py         # Whisper transcription + LLM translation
  pdf_service.py           # PDF report generation logic

utils/
  error_handlers.py        # get_logger(), handle_route_errors decorator
  observability.py         # ECS JSON logging (Filebeat → Kibana), /health endpoint
```

### Two-Store RAG with TF-IDF Lexical Gate (`rag_architecture.py`)

All queries pass through `TFIDFLexicalGate` before retrieval. The gate computes TF-IDF cosine similarity against the local KB corpus to decide routing:

- Score ≥ 0.3 → query `kb_local` first (uploaded PDFs/URLs)
- Score < 0.3 → go directly to external sources (Wikipedia, arXiv, Tavily)

`TwoStoreRAGManager` manages three ChromaDB stores:
- `kb_local` — uploaded PDFs and URLs (`./KB/`)
- `kb_external` — Wikipedia + arXiv documents (`./vector_dbs/`)
- `vector_dbs/organization/{discipline}` — specialty-specific documents

Each query also gets a per-session vector DB for conversation context.

`RAGService` (`services/rag_service.py`) is a lazy facade that initialises `TwoStoreRAGManager`, `IntegratedMedicalRAG`, and `DomainScopeGuard` in one call (`rag_service.initialize(embeddings, llm)`) and exposes them via properties (`rag_manager`, `integrated_rag_system`, `scope_guard`).

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

### Domain Scope Guard (`domain_scope_guard.py`)

Three-tier SBERT-based query filter applied before every RAG call:

| Tier | Score | Behaviour |
|------|-------|-----------|
| `accepted` | ≥ 0.45 | Answer normally |
| `general_medical` | 0.20–0.45 | Answer with disclaimer using general medical corpus |
| `rejected` | < 0.20 | Politely decline |

Gracefully degrades to pass-through if `rlhf_interactions` is empty or `sentence-transformers` is not installed. Controlled via env vars `SCOPE_GUARD_THRESHOLD`, `MEDICAL_FALLBACK_THRESHOLD`, `SCOPE_GUARD_ENABLED`.

### Hybrid Query Router (`MedicalQueryRouter` in `routes/disciplines.py`)

`MedicalQueryRouter` combines keyword matching and TF-IDF similarity to route queries to the correct discipline vector DB at startup. Auto-initialised at boot so keyword/TF-IDF routing is available before any discipline is explicitly selected.

### Department LoRA Service (`services/dept_lora_service.py`)

`DeptLoRAService` loads and caches department-specific `microsoft/phi-2` + LoRA adapters (produced by the SFT pipeline). Key design:

- **LRU cache** capped at `DEPT_LORA_CACHE_SIZE` (default 3) — each model is ~300 MB
- **Thread-safe** loading with per-dept `threading.Event` to prevent duplicate loads
- **Threading timeouts** for both load (`DEPT_LORA_LOAD_TIMEOUT=120 s`) and inference (`DEPT_LORA_INFERENCE_TIMEOUT=60 s`)
- `is_available(dept)` checks for a trained adapter on disk via `sft_experiment_manager`
- `generate(dept, query)` returns `{"success", "response", "source": "lora"|"fallback", ...}`

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

`services/db_service.py` provides:
- `init_pool()` — opens a `psycopg_pool.ConnectionPool` (non-blocking); falls back to per-request connections if `psycopg[pool]` is not installed
- `get_connection()` — context manager yielding a psycopg connection from the pool or fresh
- `execute_query(conn, query, params)` — auto-rewrites `?` → `%s` for PostgreSQL compatibility

SQL parameter syntax is adapted automatically (`%s` for PostgreSQL, `?` for SQLite).

Two PostgreSQL databases:
- `pces_base` — RLHF interactions, reward model training history, users, SFT experiments
- `pces_ehr_ccm` — Medical diagnosis records (`p_diagnosis` table)

Both psycopg v3 (`psycopg`) and v2 (`psycopg2-binary`) are required — `main.py`/`db_service.py` use v3; `postgres_tool.py` and some utilities use v2.

### Observability (`utils/observability.py`)

Call `init_observability(app)` once at startup (already done in `main.py`). It provides:

- **ECS JSON logging** — all log records emitted as Elastic Common Schema JSON; shipped to Kibana via the bundled `docker-compose.elk.yml` + `filebeat.yml`. Open http://localhost:5601 → Discover → `pces-rag-logs-*` to view logs.
- **`/health` endpoint** — returns `{"status": "ok"|"degraded", "components": {"database": ...}}`

Controlled via env vars: `LOG_LEVEL`, `LOG_FILE`, `SERVICE_NAME`, `SERVICE_VERSION`, `FLASK_ENV`.

### Optional Subsystems

Each optional subsystem uses a try/except import pattern and exposes an `*_AVAILABLE` boolean flag:

| Flag | Module | Capability |
|------|--------|-----------|
| `AZURE_AVAILABLE` | `azure_storage.py` | PDF/audio cloud storage |
| `DIARIZATION_AVAILABLE` | `voice_diarization.py` | Whisper + pyannote.audio speaker diarization |
| `INTEGRATED_RAG_AVAILABLE` | `integrated_rag.py` | Full ZERO_SHOT_REACT agent mode |
| `RAG_ARCHITECTURE_AVAILABLE` | `rag_architecture.py` | Two-store RAG + TF-IDF gate |
| `SCOPE_GUARD_AVAILABLE` | `domain_scope_guard.py` | Three-tier query filtering |
| `DEPT_LORA_AVAILABLE` | `services/dept_lora_service.py` | Department-specific LoRA inference |

The app degrades gracefully when any of these are missing.

## Key Conventions

### Service singletons
Import and use module-level singletons — never re-instantiate:
```python
from services.llm_service import llm_service
from services.rag_service import rag_service
from services.db_service import get_connection
```
Blueprints access shared state via `current_app.config` keys: `LLM_INSTANCE`, `EMBEDDINGS`, `RAG_MANAGER`, `INTEGRATED_RAG`, `SCOPE_GUARD`, `TEXT_SPLITTER`, `WHISPER_MODEL`, `APIFY_CLIENT`, `DEPT_LORA_SERVICE`, `ACTIVE_DEPARTMENT`.

### Logging
Use `get_logger(__name__)` from `utils/error_handlers.py` in every module. Never call `logging.getLogger()` directly. Decorate Flask route handlers with `@handle_route_errors` to catch and JSON-serialize unhandled exceptions automatically.

### Tool naming
Tool functions use `CamelCase_With_Underscores` (e.g. `Wikipedia_Search`, `Internal_VectorDB`). The docstring of every `@tool` function is the routing signal — keep it precise and discriminative.

### Result formatting
`_join_docs(docs, max_chars=1200)` in `tools.py` is the standard way to join `Document` objects into a string. It truncates at `max_chars`, extracts sources from `doc.metadata`, and appends a `Sources:` footer. Use this helper (or mirror its pattern) for any new tool.

### Timeout handling
Use threading-based timeouts everywhere — signal-based timeouts are unsafe in Flask/WSGI. The pattern in `enhanced_tools.py` and `services/dept_lora_service.py` is the reference implementation.

### Environment variables
All secrets and configuration come from a `.env` file loaded by `python-dotenv`. `config.py` (`Config` class) is the single source of truth — read `Config.*` attributes, not `os.getenv()` directly in route/service code.

**Required:**
```
openai_api_key, base_url, llm_model_name, embedding_model_name
DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD          # pces_base
PG_TOOL_HOST, PG_TOOL_PORT, PG_TOOL_NAME, PG_TOOL_USER, PG_TOOL_PASSWORD  # pces_ehr_ccm
```

**Optional integrations:**
```
TAVILY_API_KEY, apify_api_key, HUGGINGFACE_TOKEN
AZURE_STORAGE_CONNECTION_STRING, AZURE_STORAGE_CONTAINER_NAME
```

**Observability:**
```
LOG_LEVEL, LOG_FILE, SERVICE_NAME, SERVICE_VERSION, FLASK_ENV
```

**Tuning:**
```
SCOPE_GUARD_THRESHOLD (0.45), MEDICAL_FALLBACK_THRESHOLD (0.20), SCOPE_GUARD_ENABLED
DEPT_LORA_CACHE_SIZE (3), DEPT_LORA_LOAD_TIMEOUT (120), DEPT_LORA_INFERENCE_TIMEOUT (60)
CHUNK_SIZE (4096), CHUNK_OVERLAP (128), RETRIEVAL_K (2), SIMILARITY_THRESHOLD (0.35)
TFIDF_LOCAL_THRESHOLD (0.3), LLM_DEFAULT_TEMPERATURE (0.1), LLM_REQUEST_TIMEOUT (30)
DB_POOL_MIN_SIZE (1), DB_POOL_MAX_SIZE (10)
YODHA_CHAT_URL, DOC_PATIENT_V2_URL       # external app links
```

### Medical discipline configuration
`config/disciplines.json` defines the medical specialties. Each entry requires `id`, `name`, `kb_path`, and `vector_db_path`. The `is_session_based: true` flag marks the Doctor's Files discipline whose vector DB is scoped per session. Max 4 disciplines can be selected simultaneously (`selection_rules.max_selections`).

### Vector DB paths
- Session-based retrieval: `./vector_dbs/{session_id}/`
- Organization KBs: `./vector_dbs/organization/{discipline_id}/`
- Local KB: `./vector_dbs/` (root)

When adding new document types, persist metadata (`source`, `source_type`, `title`) on `Document.metadata` so `_join_docs` can build correct `Sources:` footers.
