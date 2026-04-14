<div align="center">
  <img src="ClientLogo101.png" alt="PCES Logo" height="80"/>

  # PCES Medical RAG Application

  **A production-grade, multi-source Retrieval-Augmented Generation (RAG) platform for clinical decision support**

  [![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
  [![Flask](https://img.shields.io/badge/Flask-3.x-black?logo=flask)](https://flask.palletsprojects.com)
  [![LangChain](https://img.shields.io/badge/LangChain-0.3-green)](https://langchain.com)
  [![ChromaDB](https://img.shields.io/badge/ChromaDB-vector%20store-orange)](https://trychroma.com)
  [![PostgreSQL](https://img.shields.io/badge/PostgreSQL-primary%20DB-blue?logo=postgresql)](https://postgresql.org)
  [![Docker](https://img.shields.io/badge/Docker-ready-blue?logo=docker)](https://docker.com)
</div>

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
   - [Two-Store RAG with TF-IDF Lexical Gate](#two-store-rag-with-tf-idf-lexical-gate)
   - [Tool-Based Agent](#tool-based-agent)
   - [RLHF + SFT Pipeline](#rlhf--sft-pipeline)
4. [Medical Disciplines](#medical-disciplines)
5. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Local Setup](#local-setup)
   - [Docker Setup](#docker-setup)
6. [Environment Variables](#environment-variables)
7. [Database Setup](#database-setup)
8. [Application Routes (API Reference)](#application-routes-api-reference)
9. [Observability](#observability)
   - [Prometheus Metrics](#prometheus-metrics)
   - [Health Check](#health-check)
   - [ELK Stack (Local)](#elk-stack-local)
10. [Authentication](#authentication)
11. [Integrated Apps](#integrated-apps)
    - [ChatYodha](#chatyodha)
    - [Doc-Patient v2](#doc-patient-v2)
12. [Project Structure](#project-structure)
13. [Training the Reward Model](#training-the-reward-model)
14. [Running Tests](#running-tests)
15. [Deployment on a VM](#deployment-on-a-vm)

---

## Overview

The **PCES Medical RAG Application** is an enterprise-grade clinical intelligence platform that combines:

- **Multi-source RAG** — answers from uploaded documents, Wikipedia, arXiv, and real-time web search, automatically routed by a TF-IDF lexical gate
- **Agent-based reasoning** — a LangChain zero-shot ReAct agent with five specialist tools
- **RLHF feedback loop** — human ratings drive a reward model that re-ranks responses at inference time
- **SFT fine-tuning** — department-scoped LoRA fine-tuning of `microsoft/phi-2` on SME-ranked data
- **Doctor-Patient conversation** — real-time voice transcription, diarization, medical summary generation, and PDF export
- **Observability** — ECS-format structured JSON logging, Prometheus metrics, and an integrated ELK stack option

---

## Key Features

| Feature | Description |
|---------|-------------|
| 🔍 **Intelligent RAG routing** | TF-IDF lexical gate decides local KB vs. external sources per query |
| 🤖 **Multi-tool agent** | Wikipedia, arXiv, Tavily, internal vector DB, and PostgreSQL EHR tools |
| 🎙️ **Voice transcription** | Whisper-based patient note recording + real-time doctor-patient conversation |
| 🗣️ **Speaker diarization** | OpenAI-based speaker separation with segmented conversation output |
| 📄 **PDF generation** | Patient notes, chat history, and conversation reports exported to PDF |
| ☁️ **Azure Blob storage** | PDFs and audio files uploaded to Azure with graceful local fallback |
| 🏥 **4 medical disciplines** | Family Medicine, Cardiology, Neurology, and Doctor's Files (session-based) |
| 🔐 **PCES authentication** | Login-gated UI — all protected actions require a valid session |
| 🎓 **RLHF training** | Admin dashboard to rate responses 1-5; reward model trained on SBERT embeddings |
| 🧪 **SFT experiments** | LoRA fine-tuning jobs scoped per medical department |
| 📊 **Prometheus + ELK** | `/metrics`, `/health` endpoints + Filebeat → Elasticsearch → Kibana |
| 🐳 **Docker-ready** | Single `docker build` to containerize the entire app |

---

## Architecture

### Two-Store RAG with TF-IDF Lexical Gate

```
User Query
    │
    ▼
┌─────────────────────┐
│  TFIDFLexicalGate   │  cosine similarity against local KB corpus
└────────┬────────────┘
         │
    score ≥ 0.3?
    ┌─────┴──────┐
   YES           NO
    │             │
    ▼             ▼
 kb_local    External sources
(PDFs/URLs)  (Wikipedia · arXiv · Tavily)
    │             │
    └──────┬──────┘
           ▼
   TwoStoreRAGManager
   (ChromaDB + reranking)
           │
           ▼
     LLM Response
```

**ChromaDB stores:**

| Store | Path | Contents |
|-------|------|----------|
| `kb_local` | `./KB/` | Uploaded PDFs and URLs (per session) |
| `kb_external` | `./vector_dbs/` | Wikipedia + arXiv documents |
| `organization/{discipline}` | `./vector_dbs/organization/` | Specialty-specific documents |

### Tool-Based Agent

`IntegratedMedicalRAG` runs a LangChain `ZERO_SHOT_REACT_DESCRIPTION` agent with five tools:

| Tool | Trigger signal (docstring) | Backend |
|------|---------------------------|---------|
| `Wikipedia_Search` | Definitions, factual explanations, general medical knowledge | `wikipedia` library |
| `ArXiv_Search` | Latest research, recent papers, scientific evidence | `arxiv` library |
| `Tavily_Search` | Real-time / current medical information | Tavily API |
| `Internal_VectorDB` | User's uploaded documents / specific files | ChromaDB similarity search |
| `PostgreSQL_Diagnosis_Search` | Structured diagnosis data | `pces_ehr_ccm.p_diagnosis` table |

All tools use a `guarded_retrieve()` wrapper with a similarity threshold of **0.35** and a Wikipedia fallback for low-confidence local results.

### RLHF + SFT Pipeline

```
Query ──► LLM response ──► Logged to rlhf_interactions
                                    │
              Admin rates 1-5 at /admin/rlhf
                                    │
              train_reward_sbert.py trains logistic regression
              on SBERT (all-MiniLM-L6-v2) embeddings
                                    │
              reward_model.joblib saved to disk
                                    │
              rlhf_reranker.py re-ranks candidates at inference

SME scores ranked responses ──► sft_ranked_data table
                                    │
              sft_experiment_manager.py creates fine-tuning job
              LoRA (rank=16, α=32) on microsoft/phi-2
              Department-scoped by keyword matching
```

---

## Medical Disciplines

Disciplines are configured in `config/disciplines.json`. Up to **4** can be active simultaneously.

| ID | Name | KB Path | Session-Based |
|----|------|---------|---------------|
| `family_medicine` | Family Medicine | `Organization_KB/Family_Medicine` | No (default) |
| `cardiology` | Cardiology | `Organization_KB/Cardiology` | No |
| `neurology` | Neurology | `Organization_KB/Neurology` | No |
| `doctors_files` | Doctor's Files | `KB/` | **Yes** — per browser session |

To add a new discipline, add an entry to `config/disciplines.json` and run:

```bash
python setup_organization_kb.py
```

---

## Getting Started

### Prerequisites

- Python **3.10+**
- PostgreSQL **14+** (two databases: `pces_base` and `pces_ehr_ccm`)
- `ffmpeg` (for audio transcription)
- A valid **OpenAI API key** (or compatible endpoint)
- Docker (optional, for containerised deployment)

### Local Setup

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd Pulmonology_RAG_App

# 2. Create and activate a virtual environment
python -m venv myenv
source myenv/bin/activate        # Windows: myenv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Copy and fill in environment variables
cp .env.example .env             # edit .env with your keys

# 5. Initialise the databases
python setup_local_db.py         # creates PostgreSQL schema + SQLite fallback

# 6. Build the vector knowledge bases
python setup_external_kb.py      # Wikipedia + arXiv (takes a few minutes)
python setup_organization_kb.py  # discipline-specific KBs

# 7. Start the app
python main.py
# → http://localhost:3000
```

### Docker Setup

```bash
# Build image
docker build -t pces-rag-app .

# Run with env file
docker run -p 3000:3000 --env-file .env pces-rag-app
```

The app is exposed on port **3000** by default (configurable in `main.py`).

---

## Environment Variables

Create a `.env` file in the project root. All variables below are read at startup.

### Required

```dotenv
# LLM / Embeddings (OpenAI-compatible)
openai_api_key=sk-...
base_url=https://api.openai.com/v1
llm_model_name=gpt-4o
embedding_model_name=text-embedding-3-small

# Primary database (pces_base — RLHF, users, SFT)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=pces_base
DB_USER=pcesuser
DB_PASSWORD=your_password

# EHR / CCM database (pces_ehr_ccm — diagnosis records)
PG_TOOL_HOST=localhost
PG_TOOL_PORT=5432
PG_TOOL_NAME=pces_ehr_ccm
PG_TOOL_USER=pcesuser
PG_TOOL_PASSWORD=your_password
```

### Optional Integrations

```dotenv
# Real-time web search
TAVILY_API_KEY=tvly-...

# Web scraping via Apify
apify_api_key=apify_api_...

# Azure Blob Storage (PDF/audio uploads — app degrades gracefully without this)
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...
AZURE_STORAGE_CONTAINER_NAME=contoso

# HuggingFace (required for SFT fine-tuning)
HUGGINGFACE_TOKEN=hf_...

# Linked external apps
YODHA_CHAT_URL=http://127.0.0.1:3000       # ChatYodha app URL
DOC_PATIENT_V2_URL=http://127.0.0.1:3001   # Doc-Patient v2 app URL

# Observability
LOG_LEVEL=INFO                             # DEBUG | INFO | WARNING | ERROR
LOG_FILE=./logs/pces.log                   # leave blank to log to stdout only
SERVICE_NAME=pces-rag-app
```

---

## Database Setup

The app uses two PostgreSQL databases and automatically falls back to SQLite (`local_sft.db`) if PostgreSQL is unreachable.

```
PostgreSQL pces_base ──► Users, RLHF interactions, reward model history, SFT experiments
PostgreSQL pces_ehr_ccm ──► Medical diagnosis records (p_diagnosis table)
SQLite local_sft.db ──► Auto-fallback when PostgreSQL is unavailable
```

### Schema migration

```bash
# Run the SFT/RLHF schema on pces_base
psql -h $DB_HOST -U $DB_USER -d pces_base -f setup_sft_tables.sql

# Or let the app auto-migrate on first run
python setup_local_db.py
```

**Parameter syntax is automatically adapted:** `%s` for PostgreSQL, `?` for SQLite — no code changes needed when switching backends.

---

## Application Routes (API Reference)

The app runs on `http://localhost:3000` by default.

### Core UI

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Main application UI |
| `GET` | `/admin/rlhf` | RLHF admin dashboard |

### Authentication

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/login` | Authenticate against `pces_users` table. Returns `{success, username, pces_role}` |

### Query & RAG

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/data` | Main RAG query — returns AI answer with sources |
| `POST` | `/generate_summary` | Generate structured medical summary from conversation |
| `POST` | `/plain_english` | Rephrase a medical query in plain language |
| `POST` | `/data-html` | RAG query — HTML-formatted response |

### Documents & Knowledge Base

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/upload_pdf` | Upload a PDF to the local KB |
| `POST` | `/upload_url` | Add a URL to the local KB |
| `POST` | `/upload_organization_kb` | Upload a PDF to a discipline-specific KB |
| `POST` | `/create_vector_db` | Rebuild the vector database from the current KB |

### Audio & Voice

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/transcribe` | Transcribe audio via Whisper |
| `POST` | `/translate_audio` | Transcribe + translate audio |
| `POST` | `/transcribe_patient_notes` | Transcribe and structure patient notes |

### Doctor-Patient Conversation

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/transcribe_doctor_patient_conversation` | Diarize + transcribe a doctor-patient recording |

### PDF Generation

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/generate_patient_pdf` | Generate patient notes PDF (Azure upload or local download) |
| `POST` | `/generate_chat_pdf` | Generate chat history PDF |
| `POST` | `/generate_conversation_pdf` | Generate full conversation report PDF |

### Disciplines

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/disciplines` | List available medical disciplines |
| `POST` | `/api/validate_disciplines` | Validate a set of selected discipline IDs |
| `GET` | `/search_doctors` | Search doctors by name / department |

### Doctors

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/rlhf/doctors` | List all doctors |
| `GET` | `/api/rlhf/doctors/by-department` | List doctors grouped by department |
| `POST` | `/api/rlhf/doctors` | Create a new doctor record |
| `PUT` | `/api/rlhf/doctors/<id>` | Update a doctor record |

### RLHF

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/rlhf/stats` | Reward model training statistics |
| `GET` | `/api/rlhf/interactions` | List all logged interactions |
| `GET` | `/api/rlhf/sessions` | List interaction sessions |
| `POST` | `/api/rlhf/add_sample` | Add a rated Q&A sample |

### SFT Experiments

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/rlhf/experiments` | List fine-tuning experiments |
| `POST` | `/api/rlhf/experiment/create` | Start a new SFT job |
| `GET` | `/api/rlhf/experiment/<id>/status` | Get experiment status |
| `DELETE` | `/api/rlhf/experiment/<id>` | Delete an experiment |

### Azure Storage

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/check_azure_files` | List files in Azure Blob container |
| `GET` | `/check_azure_file/<filename>` | Check if a specific file exists |
| `GET` | `/azure_storage_info` | Azure connection info and stats |

### Observability

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | App + database health status |
| `GET` | `/metrics` | Prometheus text-format metrics |

---

## Observability

### Prometheus Metrics

The `/metrics` endpoint exposes four metrics in Prometheus text format:

| Metric | Type | Description |
|--------|------|-------------|
| `http_requests_total` | Counter | Total requests, labelled by method / endpoint / status |
| `http_request_duration_seconds` | Histogram | Request latency with 9 buckets (50ms → 30s) |
| `http_errors_total` | Counter | 4xx/5xx responses, labelled by method / endpoint / status |
| `http_active_requests` | Gauge | Requests currently in flight |

```bash
# View metrics on the VM
curl http://localhost:3000/metrics
```

To scrape with Prometheus, add to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: pces-rag
    static_configs:
      - targets: ['<vm-ip>:3000']
```

### Health Check

```bash
curl http://localhost:3000/health
```

```json
{
  "status": "ok",
  "timestamp": "2026-04-14T12:00:00+00:00",
  "service": "pces-rag-app",
  "version": "1.0.0",
  "components": {
    "database": { "status": "ok" },
    "prometheus": "available"
  }
}
```

Returns `200 OK` when healthy, `207 Multi-Status` when degraded (e.g. database unreachable).

### ELK Stack (Local)

The project ships a ready-to-use Docker Compose file for a local Elasticsearch + Kibana + Filebeat stack.

**Step 1** — Tell the app to write JSON logs to a file:

```bash
# In .env
LOG_FILE=./logs/pces.log
```

**Step 2** — Start the ELK stack:

```bash
mkdir -p logs
docker compose -f docker-compose.elk.yml up -d
```

| Service | Port | URL |
|---------|------|-----|
| Elasticsearch | 9200 | `http://localhost:9200` |
| Kibana | 5601 | `http://localhost:5601` |
| Filebeat | — | Ships logs automatically |

Default credentials: **elastic / changeme** (change for production).

**Step 3** — Open Kibana:

1. Go to `http://localhost:5601`
2. Navigate to **Discover**
3. Create a data view with index pattern `pces-rag-logs-*`
4. All ECS-format JSON log lines from the app will appear here

Every log line includes: `@timestamp`, `log.level`, `message`, `http.request.method`, `url.path`, `http.response.status_code`, `event.duration` (nanoseconds), `transaction.id`.

---

## Authentication

All protected UI actions require a valid PCES login. Authentication state is stored in `sessionStorage` (persists through page refreshes within the same browser tab).

**Protected buttons** (require login):
- Generate (Yodha chat)
- Save as PDF
- Start Recording / Stop Recording
- Plain English
- Translation
- Add PDF / Add URL / Update KB
- Doc/Patient Conversation (start/stop)
- Doc-Patient v2
- Record Patient Notes
- RLHF Training Admin

**Unprotected** (always accessible):
- ChatYodha (opens the external chat app)
- Stop buttons for recordings already in progress that were started pre-auth

### Login flow

```
User presses protected button
  └─► Login modal appears (PCES username + password)
        └─► POST /api/login
              ├─ Success → sessionStorage.pces_auth = 'true' → original action executes
              └─ Failure → error message shown in modal
```

Credentials are validated against the `pces_users` table in the `pces_base` database.

---

## Integrated Apps

### ChatYodha

An external AI chat interface accessible directly from the main app.

- **Button:** `ChatYodha` (top panel — no login required)
- **URL configured by:** `YODHA_CHAT_URL` env var (default: `http://127.0.0.1:3000`)

### Doc-Patient v2

A real-time multilingual doctor-patient conversation system with voice diarization, AI-powered medical summaries, and PDF export.

- **Button:** `Doc-Patient v2` (requires PCES login)
- **URL configured by:** `DOC_PATIENT_V2_URL` env var (default: `http://127.0.0.1:3001`)
- **Source code:** `../doctor_patient_conversation/`

**Doc-Patient v2 PDF export** works with or without Azure:
- If `AZURE_STORAGE_CONNECTION_STRING` is valid → PDF uploaded to Azure Blob, URL returned
- If Azure is unavailable or misconfigured → PDF generated locally and served via `/api/download_pdf/<token>` (5-minute token TTL)

---

## Project Structure

```
Pulmonology_RAG_App/
├── main.py                    # Flask bootstrap — initialises app, registers blueprints
├── config.py                  # All env-var-backed configuration
├── config/
│   └── disciplines.json       # Medical specialty definitions
│
├── routes/                    # Flask Blueprint route modules
│   ├── audio.py               # /transcribe, /translate_audio, /transcribe_patient_notes
│   ├── azure_routes.py        # /check_azure_files, /azure_storage_info
│   ├── conversation.py        # /transcribe_doctor_patient_conversation
│   ├── disciplines.py         # /api/login, /api/disciplines, /search_doctors
│   ├── doctors.py             # /api/rlhf/doctors CRUD
│   ├── documents.py           # /upload_pdf, /upload_url, /create_vector_db
│   ├── pdf_generation.py      # /generate_patient_pdf, /generate_chat_pdf, etc.
│   ├── query.py               # /data, /generate_summary, /plain_english
│   ├── rlhf.py                # /admin/rlhf, /api/rlhf/interactions, /api/rlhf/stats
│   └── sft.py                 # /api/rlhf/experiment CRUD
│
├── services/                  # Business logic (decoupled from HTTP layer)
│   ├── audio_service.py       # Whisper transcription + diarization
│   ├── db_service.py          # PostgreSQL connection pool + fallback
│   ├── llm_service.py         # LLM + embeddings initialisation
│   ├── pdf_service.py         # ReportLab PDF generation
│   └── rag_service.py         # RAG chain assembly
│
├── utils/
│   ├── error_handlers.py      # get_logger(), @handle_route_errors decorator
│   └── observability.py       # ECS logging, Prometheus middleware, /health, /metrics
│
├── rag_architecture.py        # TwoStoreRAGManager + TFIDFLexicalGate
├── integrated_rag.py          # LangChain agent with 5 tools
├── tools.py                   # @tool definitions + guarded_retrieve()
├── enhanced_tools.py          # Threading-based timeout wrappers
├── domain_scope_guard.py      # Domain relevance filtering
│
├── azure_storage.py           # Azure Blob upload helpers
├── voice_diarization.py       # pyannote.audio speaker diarization
│
├── rlhf_reranker.py           # Inference-time response re-ranking
├── train_reward_sbert.py      # SBERT-based reward model training script
├── sft_experiment_manager.py  # LoRA fine-tuning job manager
├── model_utils.py             # Model loading utilities
│
├── templates/
│   ├── index.html             # Main app UI (single-page)
│   └── admin_rlhf.html        # RLHF admin dashboard
│
├── KB/                        # Local PDF/URL knowledge base (Doctor's Files)
├── Organization_KB/           # Discipline-specific knowledge bases
├── vector_dbs/                # ChromaDB persistent stores
│   └── organization/          # Per-discipline vector DBs
│
├── sft_models/                # Saved LoRA fine-tuned model checkpoints
├── reward_model.joblib        # Trained RLHF reward model
├── local_sft.db               # SQLite fallback database
│
├── Dockerfile                 # Container image definition
├── docker-compose.elk.yml     # Local ELK stack (Elasticsearch + Kibana + Filebeat)
├── filebeat.yml               # Filebeat configuration
└── requirements.txt           # Python dependencies
```

---

## Training the Reward Model

The reward model re-ranks candidate LLM responses using a logistic regression classifier trained on SBERT (`all-MiniLM-L6-v2`) embeddings of rated query-response pairs.

```bash
# 1. Rate responses via the admin UI
#    http://localhost:3000/admin/rlhf  (ratings: 1-5)

# 2. Train the reward model
python train_reward_sbert.py

# 3. Model is saved to reward_model.joblib
#    rlhf_reranker.py loads it automatically at inference
```

### SFT Fine-Tuning

```bash
# Start a fine-tuning experiment via the SFT API
curl -X POST http://localhost:3000/api/rlhf/experiment/create \
  -H 'Content-Type: application/json' \
  -d '{
    "experiment_name": "cardiology-v1",
    "department": "cardiology",
    "model_name": "microsoft/phi-2",
    "num_epochs": 10,
    "lora_r": 16,
    "lora_alpha": 32
  }'
```

Fine-tuned checkpoints are saved to `sft_models/`.

---

## Running Tests

```bash
# Activate virtual environment first
source myenv/bin/activate

# Core RAG integration
python tests/test_complete_integration.py

# RLHF pipeline
python test_rlhf_pipeline.py

# PostgreSQL tool
python tests/test_postgres_integration.py

# External source connectivity (Wikipedia + arXiv)
python test_wiki_arxiv_connectivity.py

# Quick connectivity check
python test_wiki_arxiv_quick.py

# Timeout fix validation
python test_timeout_fix.py
```

> **Note:** Tests are run directly as scripts (no pytest runner required), though `pytest.ini` is present for IDE support.

---

## Deployment on a VM

### 1. Install system dependencies

```bash
sudo apt-get update && sudo apt-get install -y \
    python3.10 python3-pip python3-venv \
    ffmpeg tesseract-ocr \
    postgresql-client \
    docker.io docker-compose-plugin
```

### 2. Set up the app

```bash
git clone <repo-url> /opt/pces-rag
cd /opt/pces-rag
python3 -m venv myenv && source myenv/bin/activate
pip install -r requirements.txt
cp .env.example .env && nano .env   # fill in all required values
python setup_local_db.py
```

### 3. Run as a systemd service

```bash
sudo nano /etc/systemd/system/pces-rag.service
```

```ini
[Unit]
Description=PCES Medical RAG Application
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/opt/pces-rag
EnvironmentFile=/opt/pces-rag/.env
ExecStart=/opt/pces-rag/myenv/bin/python main.py
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable pces-rag
sudo systemctl start pces-rag

# View logs
sudo journalctl -u pces-rag -f
```

### 4. Check health and metrics

```bash
curl http://localhost:3000/health
curl http://localhost:3000/metrics
```

### 5. Start the ELK observability stack

```bash
mkdir -p /opt/pces-rag/logs
# Set LOG_FILE=./logs/pces.log in .env, then:
docker compose -f docker-compose.elk.yml up -d

# Kibana at http://<vm-ip>:5601
# Default login: elastic / changeme
```

### 6. Run with Docker

```bash
docker build -t pces-rag-app .
docker run -d \
  --name pces-rag \
  -p 3000:3000 \
  --env-file .env \
  -v $(pwd)/KB:/app/KB \
  -v $(pwd)/vector_dbs:/app/vector_dbs \
  -v $(pwd)/logs:/app/logs \
  pces-rag-app
```

---

## Optional Subsystems

Each optional subsystem is imported with a `try/except` pattern and exposes a boolean flag. The app degrades gracefully when any are absent.

| Subsystem | Flag | Required packages |
|-----------|------|-------------------|
| Azure Blob Storage | `AZURE_AVAILABLE` | `azure-storage-blob` |
| Voice Diarization | `DIARIZATION_AVAILABLE` | `pyannote.audio`, `torch` |
| Integrated RAG Agent | `INTEGRATED_RAG_AVAILABLE` | `langchain`, `tavily-python` |
| RAG Architecture | `RAG_ARCHITECTURE_AVAILABLE` | `scikit-learn` |
| Prometheus Metrics | `PROMETHEUS_AVAILABLE` | `prometheus_client` |

---

<div align="center">
  <sub>Built with ❤️ by the PCES Engineering Team</sub>
</div>
