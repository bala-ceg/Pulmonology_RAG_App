-- ============================================================
-- PCES Medical AI — SFT Training Schema
-- ============================================================
-- Share this file with your PostgreSQL admin.
-- Run as a superuser or a user with CREATE privilege on public schema.
--
-- Usage:
--   psql -h 4.155.102.23 -p 5432 -U <admin_user> -d pces_base -f setup_sft_schema_admin.sql
--
-- What this script creates:
--   1. sft_ranked_data    — RLHF ranked training samples + SME review fields
--   2. sft_experiments    — Model fine-tuning experiment tracking
--   3. sme_doctors        — Department doctors for SME review workflow
--   4. All required indexes
--   5. Grants pcesuser full access to all three tables
-- ============================================================


-- ============================================================
-- 1. RANKED TRAINING DATA
-- ============================================================
-- Stores prompt + ranked response triplets created by clinical users.
-- Users rank 3 response variants (1=best, 3=worst) with reasons.
-- SME doctors later score these (sme_score 1-5) for training quality.

CREATE TABLE IF NOT EXISTS sft_ranked_data (
    id                  SERIAL PRIMARY KEY,
    prompt              TEXT        NOT NULL,
    response_text       TEXT        NOT NULL,
    rank                INTEGER     NOT NULL CHECK (rank >= 1),
    reason              TEXT,
    group_id            TEXT        NOT NULL,           -- UUID linking the 3 variants of one prompt

    -- SME review fields (populated after doctor review)
    domain              TEXT,                           -- Medical department (e.g. "Cardiology")
    sme_score           INTEGER     CHECK (sme_score >= 1 AND sme_score <= 5),  -- NULL = pending review
    sme_score_reason    TEXT,                           -- Doctor's reasoning for the score
    sme_reviewed_by     TEXT,                           -- Doctor name (e.g. "Dr. Jane Smith")
    sme_reviewed_at     TIMESTAMP,                      -- When the review happened

    created_at          TIMESTAMP   DEFAULT NOW(),
    updated_at          TIMESTAMP   DEFAULT NOW(),
    created_by          INTEGER     DEFAULT 1001,
    updated_by          INTEGER     DEFAULT 1001
);

CREATE INDEX IF NOT EXISTS idx_sft_ranked_data_group_id
    ON sft_ranked_data(group_id);

CREATE INDEX IF NOT EXISTS idx_sft_ranked_data_domain
    ON sft_ranked_data(domain);

CREATE INDEX IF NOT EXISTS idx_sft_ranked_data_sme_score
    ON sft_ranked_data(sme_score);


-- ============================================================
-- 2. EXPERIMENT TRACKING
-- ============================================================
-- Each row is one fine-tuning run (LoRA/SFT on a base model).
-- Status lifecycle: pending → running → completed / failed

CREATE TABLE IF NOT EXISTS sft_experiments (
    id                          SERIAL PRIMARY KEY,
    experiment_name             TEXT        NOT NULL,
    department                  TEXT,                   -- Optional: scoped to one department
    status                      TEXT        NOT NULL DEFAULT 'pending',
                                                        -- Values: pending, running, completed, failed
    model_name                  TEXT        DEFAULT 'microsoft/phi-2',
    lora_r                      INTEGER     DEFAULT 16,
    lora_alpha                  INTEGER     DEFAULT 32,
    lora_dropout                REAL        DEFAULT 0.05,
    num_epochs                  INTEGER     DEFAULT 10,
    batch_size                  INTEGER     DEFAULT 2,
    gradient_accumulation_steps INTEGER     DEFAULT 4,
    learning_rate               REAL        DEFAULT 0.0001,
    max_seq_length              INTEGER     DEFAULT 2048,
    training_samples            INTEGER     DEFAULT 0,
    started_at                  TIMESTAMP,
    completed_at                TIMESTAMP,
    error_message               TEXT,
    model_output_path           TEXT,
    metrics                     JSONB       DEFAULT '{}',
    created_at                  TIMESTAMP   DEFAULT NOW(),
    created_by                  INTEGER     DEFAULT 1001
);

CREATE INDEX IF NOT EXISTS idx_sft_experiments_status
    ON sft_experiments(status);

CREATE INDEX IF NOT EXISTS idx_sft_experiments_department
    ON sft_experiments(department);


-- ============================================================
-- 3. SME DOCTORS
-- ============================================================
-- Doctors available for SME (Subject Matter Expert) review per department.
-- When a doctor selects their department in the UI, they see only their
-- department's prompts for review and scoring.

CREATE TABLE IF NOT EXISTS sme_doctors (
    id          SERIAL PRIMARY KEY,
    name        TEXT        NOT NULL,
    email       TEXT,
    department  TEXT        NOT NULL,
    specialty   TEXT,
    is_active   BOOLEAN     DEFAULT true,
    created_at  TIMESTAMP   DEFAULT NOW(),
    updated_at  TIMESTAMP   DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sme_doctors_department
    ON sme_doctors(department);

CREATE INDEX IF NOT EXISTS idx_sme_doctors_active
    ON sme_doctors(is_active, department);

CREATE INDEX IF NOT EXISTS idx_sme_doctors_name
    ON sme_doctors(name);


-- ============================================================
-- 4. GRANT PERMISSIONS TO APPLICATION USER
-- ============================================================
-- Replace pcesuser with your application DB user if different.

GRANT SELECT, INSERT, UPDATE, DELETE ON sft_ranked_data  TO pcesuser;
GRANT SELECT, INSERT, UPDATE, DELETE ON sft_experiments   TO pcesuser;
GRANT SELECT, INSERT, UPDATE, DELETE ON sme_doctors       TO pcesuser;

-- Grant USAGE and SELECT on sequences (needed for SERIAL / INSERT)
GRANT USAGE, SELECT ON SEQUENCE sft_ranked_data_id_seq  TO pcesuser;
GRANT USAGE, SELECT ON SEQUENCE sft_experiments_id_seq  TO pcesuser;
GRANT USAGE, SELECT ON SEQUENCE sme_doctors_id_seq      TO pcesuser;


-- ============================================================
-- 5. VERIFICATION
-- ============================================================
-- Run this to confirm all three tables exist with correct columns.

SELECT
    t.table_name,
    COUNT(c.column_name) AS column_count
FROM information_schema.tables t
JOIN information_schema.columns c USING (table_name, table_schema)
WHERE t.table_schema = 'public'
  AND t.table_name IN ('sft_ranked_data', 'sft_experiments', 'sme_doctors')
GROUP BY t.table_name
ORDER BY t.table_name;

-- Expected output:
--   sft_experiments  | 21
--   sft_ranked_data  | 16
--   sme_doctors      |  8
