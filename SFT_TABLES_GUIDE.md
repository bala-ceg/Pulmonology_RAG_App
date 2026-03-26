# SFT Tables Schema Guide

## Overview
Two main tables support the Supervised Fine-Tuning (SFT) system for medical AI models:
1. **sft_ranked_data** - Training data with human quality rankings
2. **sft_experiments** - Training job tracking and configuration

---

## Table 1: `sft_ranked_data`
**Purpose:** Stores medical Q&A training data with quality rankings for RLHF/SFT

### Why This Design?
- **Multiple responses per question:** Allows human experts to rank different AI responses
- **Group-based organization:** Links all responses for the same prompt via `group_id`
- **Rank-based selection:** Training uses only the best response (rank=1) per question

### Fields Explained

| Field | Type | Why It Exists |
|-------|------|---------------|
| `id` | SERIAL | Unique identifier for each response |
| `prompt` | TEXT | The medical question or scenario (e.g., "What are diabetes symptoms?") |
| `response_text` | TEXT | The AI-generated or expert-written answer |
| `rank` | INTEGER | Quality ranking (1=best, 2=second-best, etc.) - Training uses rank-1 only |
| `reason` | TEXT | Expert's explanation for this ranking (appears as "### Rationale:" in training) |
| `group_id` | TEXT | Links all responses to the same prompt (e.g., "diabetes_q1") |
| `created_at` | TIMESTAMP | When this training data was added |
| `updated_at` | TIMESTAMP | When this was last modified (tracks iterative improvements) |
| `created_by` | INTEGER | User ID who added this (default 1001 = system) |
| `updated_by` | INTEGER | User ID who last edited this |

### Example Data

```sql
-- Question with 3 ranked responses
-- group_id: "heart_attack_001"

-- Rank 1 (best)
INSERT INTO sft_ranked_data (prompt, response_text, rank, reason, group_id) VALUES
('What are heart attack warning signs?', 
 'Warning signs include chest pain, shortness of breath, nausea, and cold sweats...', 
 1, 
 'Complete list with clear medical terminology', 
 'heart_attack_001');

-- Rank 2 (second-best)
INSERT INTO sft_ranked_data (prompt, response_text, rank, reason, group_id) VALUES
('What are heart attack warning signs?', 
 'You might feel chest pain and have trouble breathing...', 
 2, 
 'Less comprehensive than rank-1', 
 'heart_attack_001');

-- Rank 3 (worst)
INSERT INTO sft_ranked_data (prompt, response_text, rank, reason, group_id) VALUES
('What are heart attack warning signs?', 
 'Call 911 if you feel bad.', 
 3, 
 'Too vague, lacks medical detail', 
 'heart_attack_001');
```

**Training uses only the rank-1 response.**

### How Department Filtering Works
- No explicit `department` column (schema-less filtering)
- Filtering happens via keyword matching on `prompt` text
- Example: Department "Cardiology" matches prompts containing: "heart", "cardiac", "cardio", "arrhythmia", etc.
- See `DEPARTMENTS` dict in `sft_experiment_manager.py` for keyword mappings

---

## Table 2: `sft_experiments`
**Purpose:** Tracks each model training run with its configuration and results

### Why These Fields?

#### Basic Metadata
| Field | Type | Why It Exists |
|-------|------|---------------|
| `id` | SERIAL | Unique experiment ID (used in model output path) |
| `experiment_name` | TEXT | Human-readable name (e.g., "Cardiology_SFT_Model") |
| `department` | TEXT | Specialization filter (NULL = all data, "Cardiology" = cardio prompts only) |
| `status` | TEXT | Current state: 'pending', 'running', 'completed', 'failed' |

#### Model Architecture
| Field | Default | Why This Value? |
|-------|---------|-----------------|
| `model_name` | microsoft/phi-2 | 2.7B parameter model, good medical knowledge, fits on consumer GPUs |
| `lora_r` | 16 | Rank of LoRA adapter - balance between capacity and memory |
| `lora_alpha` | 32 | Scaling factor (typically 2x rank) for stable training |
| `lora_dropout` | 0.05 | 5% dropout prevents overfitting on small datasets |

#### Training Hyperparameters
| Field | Default | Why This Value? |
|-------|---------|-----------------|
| `num_epochs` | 10 | For 6 samples, 10 epochs = 60 training steps (good for small datasets) |
| `batch_size` | 2 | Conservative - fits Phi-2 + LoRA on most GPUs |
| `gradient_accumulation_steps` | 4 | Effective batch = 2 × 4 = 8 (simulates larger batches) |
| `learning_rate` | 0.0001 | Conservative for medical domain (prevents catastrophic forgetting) |
| `max_seq_length` | 2048 | Handles typical medical Q&A (avg 200-500 tokens) |

#### Training Results
| Field | Type | Purpose |
|-------|------|---------|
| `training_samples` | INTEGER | Actual count after department filtering (e.g., 6 for Cardiology) |
| `started_at` | TIMESTAMP | When training began (for elapsed time calculation) |
| `completed_at` | TIMESTAMP | When training finished (NULL if still running) |
| `error_message` | TEXT | Stack trace if status='failed' (for debugging) |
| `model_output_path` | TEXT | Where model was saved (e.g., "sft_models/Cardiology_experiment_1/") |
| `metrics` | JSONB | Training stats (e.g., `{"final_loss": 1.8780, "steps": 10}`) |

---

## Key Relationships

### 1. Department-Scoped Training
```
User selects: "Cardiology"
    ↓
System filters sft_ranked_data WHERE prompt contains cardiology keywords
    ↓
Creates temporary JSONL with only Cardiology prompts (rank-1 responses)
    ↓
Trains model on this subset
    ↓
Saves to: sft_models/Cardiology_experiment_{id}/
    ↓
Updates sft_experiments: status='completed', training_samples=6, model_output_path=...
```

### 2. Model Testing
```
User selects trained model from dropdown
    ↓
UI fetches experiment from sft_experiments WHERE status='completed'
    ↓
Loads LoRA adapter from model_output_path
    ↓
Generates response to medical question
```

---

## Indexes Explained

### `idx_sft_ranked_data_group_id`
**Purpose:** Fast retrieval of all responses for a prompt
```sql
-- Without index: Full table scan (slow on 10K+ rows)
-- With index: Direct lookup (constant time)
SELECT * FROM sft_ranked_data WHERE group_id = 'diabetes_q5';
```

**Used by:** `_build_training_data_from_db()` when constructing training datasets

---

## Common Queries

### Get all training data for a department
```sql
SELECT prompt, response_text, rank
FROM sft_ranked_data
WHERE LOWER(prompt) LIKE '%diabetes%'
  AND rank = 1
ORDER BY group_id;
```

### Count training samples by department
```sql
SELECT 
    CASE 
        WHEN LOWER(prompt) LIKE '%cardio%' THEN 'Cardiology'
        WHEN LOWER(prompt) LIKE '%diabetes%' THEN 'Diabetes'
        ELSE 'Other'
    END AS department,
    COUNT(DISTINCT group_id) AS prompts
FROM sft_ranked_data
WHERE rank = 1
GROUP BY department;
```

### Get experiment history
```sql
SELECT 
    id,
    experiment_name,
    department,
    status,
    training_samples,
    num_epochs,
    completed_at
FROM sft_experiments
ORDER BY created_at DESC;
```

### Find best model for a department
```sql
SELECT 
    id,
    experiment_name,
    metrics->>'final_loss' AS loss,
    training_samples
FROM sft_experiments
WHERE department = 'Cardiology'
  AND status = 'completed'
ORDER BY (metrics->>'final_loss')::float ASC
LIMIT 1;
```

---

## Setup Instructions

### 1. As PostgreSQL Admin (on Azure VM or local with DB access)
```bash
psql -h 4.155.102.23 -p 5432 -U <admin_user> -d pces_base \
     -f setup_sft_tables_documented.sql
```

### 2. Verify Tables Created
```sql
\dt sft_*
SELECT * FROM sft_ranked_data LIMIT 5;
SELECT * FROM sft_experiments LIMIT 5;
```

### 3. Import Training Data (from app)
```python
# Automatically imports on startup if tables are empty
# Or manually via:
from sft_experiment_manager import import_from_jsonl
result = import_from_jsonl('../pces_rlhf_experiments/medical_ranked.jsonl')
print(f"Imported {result['imported']} entries")
```

---

## Design Decisions

### Why No Foreign Keys?
- `group_id` is TEXT (not FK) for flexibility - can import external datasets without ID conflicts
- Loose coupling allows independent table management

### Why JSONB for Metrics?
- Different experiments may track different metrics (loss, perplexity, F1, etc.)
- JSONB allows flexible schema without ALTER TABLE
- Can still query: `WHERE (metrics->>'final_loss')::float < 2.0`

### Why Department is TEXT (not ENUM)?
- Easy to add new departments without schema migration
- Keyword matching on prompt text (no separate department column in sft_ranked_data)
- Flexible for multi-department prompts

### Why Separate Tables?
- **sft_ranked_data**: Long-term training data storage (changes infrequently)
- **sft_experiments**: Experiment tracking (many runs, frequent updates)
- Separation allows independent scaling and backup strategies

---

## Performance Considerations

### Current Scale
- Training data: ~100 prompts × 3 responses = 300 rows
- Experiments: ~2-10 per day = ~3,650 per year

### At 10K+ Prompts (30K+ rows)
Add these indexes:
```sql
CREATE INDEX idx_ranked_data_prompt_trgm ON sft_ranked_data 
    USING gin(prompt gin_trgm_ops);  -- For fuzzy search

CREATE INDEX idx_ranked_data_dept_composite ON sft_ranked_data
    (group_id, rank) WHERE rank = 1;  -- For training data fetch

CREATE INDEX idx_experiments_status_dept ON sft_experiments
    (status, department) WHERE status = 'completed';  -- For model selection
```

### Partition Strategy (if needed at 1M+ rows)
```sql
-- Partition sft_ranked_data by created_at year
CREATE TABLE sft_ranked_data_2026 PARTITION OF sft_ranked_data
    FOR VALUES FROM ('2026-01-01') TO ('2027-01-01');
```
