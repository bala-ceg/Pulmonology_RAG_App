-- ============================================================
-- SME Review Workflow - Database Migration
-- ============================================================
-- Purpose: Add columns to support Subject Matter Expert (SME) review
--          of ranked training data created by regular users
-- 
-- Workflow:
--   1. Regular users (e.g., Cardiologists) create prompts with ranked responses
--   2. SMEs (Directors) review and score these prompts weekly
--   3. After 6 weeks, high-scoring data is used for model retraining
--
-- Usage:
--   psql -h 4.155.102.23 -p 5432 -U <admin_user> -d pces_base -f add_sme_review_columns.sql
-- ============================================================

-- Add domain/department column to categorize prompts
ALTER TABLE sft_ranked_data 
ADD COLUMN IF NOT EXISTS domain TEXT;

-- Add SME score column (1-5 rating by expert reviewer)
-- NULL = not yet reviewed
ALTER TABLE sft_ranked_data 
ADD COLUMN IF NOT EXISTS sme_score INTEGER CHECK (sme_score >= 1 AND sme_score <= 5);

-- Add SME reasoning for their score
ALTER TABLE sft_ranked_data 
ADD COLUMN IF NOT EXISTS sme_score_reason TEXT;

-- Add SME reviewer name (e.g., "Dr. Jack Nicholson")
ALTER TABLE sft_ranked_data 
ADD COLUMN IF NOT EXISTS sme_reviewed_by TEXT;

-- Add timestamp when SME review occurred
ALTER TABLE sft_ranked_data 
ADD COLUMN IF NOT EXISTS sme_reviewed_at TIMESTAMP;

-- Create index for filtering by domain
CREATE INDEX IF NOT EXISTS idx_sft_ranked_data_domain 
ON sft_ranked_data(domain);

-- Create index for filtering pending reviews (WHERE sme_score IS NULL)
CREATE INDEX IF NOT EXISTS idx_sft_ranked_data_sme_score 
ON sft_ranked_data(sme_score);

-- Grant permissions to application user
GRANT ALL ON sft_ranked_data TO pcesuser;

-- ============================================================
-- VERIFICATION QUERY
-- ============================================================
-- Run this to verify the new columns were added successfully
SELECT 
    column_name, 
    data_type, 
    is_nullable,
    column_default
FROM information_schema.columns 
WHERE table_name = 'sft_ranked_data' 
    AND column_name IN ('domain', 'sme_score', 'sme_score_reason', 'sme_reviewed_by', 'sme_reviewed_at')
ORDER BY column_name;

-- Show sample data structure
SELECT 
    id,
    domain,
    LEFT(prompt, 50) AS prompt_preview,
    rank,
    sme_score,
    LEFT(sme_score_reason, 30) AS reason_preview,
    sme_reviewed_by,
    sme_reviewed_at
FROM sft_ranked_data 
LIMIT 5;
