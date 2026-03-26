-- ============================================================
-- SFT Experiment Tables Setup Script
-- ============================================================
-- Run this on your hosted PostgreSQL database if pcesuser
-- lacks CREATE TABLE permission, or to manually set up tables.
--
-- Usage (from the Azure VM or any machine with DB access):
--   psql -h 4.155.102.23 -p 5432 -U <admin_user> -d pces_base -f setup_sft_tables.sql
-- ============================================================

-- 1. Ranked training data (medical Q&A with ranked responses)
CREATE TABLE IF NOT EXISTS sft_ranked_data (
    id SERIAL PRIMARY KEY,
    prompt TEXT NOT NULL,
    response_text TEXT NOT NULL,
    rank INTEGER NOT NULL CHECK (rank >= 1),
    reason TEXT,
    group_id TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    created_by INTEGER DEFAULT 1001,
    updated_by INTEGER DEFAULT 1001
);

-- 2. SFT experiment tracking
CREATE TABLE IF NOT EXISTS sft_experiments (
    id SERIAL PRIMARY KEY,
    experiment_name TEXT NOT NULL,
    department TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    model_name TEXT DEFAULT 'microsoft/phi-2',
    lora_r INTEGER DEFAULT 16,
    lora_alpha INTEGER DEFAULT 32,
    lora_dropout REAL DEFAULT 0.05,
    num_epochs INTEGER DEFAULT 10,
    batch_size INTEGER DEFAULT 2,
    gradient_accumulation_steps INTEGER DEFAULT 4,
    learning_rate REAL DEFAULT 0.0001,
    max_seq_length INTEGER DEFAULT 2048,
    training_samples INTEGER DEFAULT 0,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT,
    model_output_path TEXT,
    metrics JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    created_by INTEGER DEFAULT 1001
);

-- 3. Index for fast group lookups
CREATE INDEX IF NOT EXISTS idx_sft_ranked_data_group_id ON sft_ranked_data(group_id);

-- 4. Grant permissions to pcesuser
GRANT ALL ON sft_ranked_data TO pcesuser;
GRANT ALL ON sft_experiments TO pcesuser;
GRANT USAGE, SELECT ON SEQUENCE sft_ranked_data_id_seq TO pcesuser;
GRANT USAGE, SELECT ON SEQUENCE sft_experiments_id_seq TO pcesuser;

-- Verify
SELECT 'sft_ranked_data' AS table_name, COUNT(*) AS rows FROM sft_ranked_data
UNION ALL
SELECT 'sft_experiments', COUNT(*) FROM sft_experiments;
