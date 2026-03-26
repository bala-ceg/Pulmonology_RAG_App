-- ============================================================
-- SFT Experiment Tables Setup Script (Documented)
-- ============================================================
-- Database: pces_base
-- User: pcesuser
-- Purpose: Store medical training data and track fine-tuning experiments
--
-- Usage (run as PostgreSQL admin):
--   psql -h 4.155.102.23 -p 5432 -U <admin_user> -d pces_base -f setup_sft_tables_documented.sql
-- ============================================================

-- ============================================================
-- TABLE 1: sft_ranked_data
-- Purpose: Stores ranked medical Q&A training data for RLHF/SFT
-- ============================================================
-- Each medical question (prompt) can have multiple responses,
-- ranked by quality (rank=1 is best). The system trains models
-- on the highest-ranked responses to learn from expert feedback.
-- ============================================================

CREATE TABLE IF NOT EXISTS sft_ranked_data (
    -- Primary key
    id SERIAL PRIMARY KEY,
    
    -- Medical question or scenario
    -- Example: "What are the symptoms of Type-2 Diabetes?"
    prompt TEXT NOT NULL,
    
    -- Model-generated or expert-written response
    -- Example: "Type-2 Diabetes symptoms include increased thirst..."
    response_text TEXT NOT NULL,
    
    -- Quality ranking (1 = best, 2 = second-best, etc.)
    -- Used during training to select the best response for each prompt
    -- CHECK constraint ensures rank is always >= 1
    rank INTEGER NOT NULL CHECK (rank >= 1),
    
    -- Expert's explanation for why this response received this rank
    -- Example: "Clear explanation with proper medical terminology"
    -- Can be NULL if no reasoning was provided
    -- Used in training as "### Rationale:" section
    reason TEXT,
    
    -- Groups all responses for the same prompt together
    -- Example: "group_001", "diabetes_q1", "cardio_case_5"
    -- All rows with the same group_id represent different responses
    -- to the same prompt (allowing ranking comparison)
    group_id TEXT NOT NULL,
    
    -- Audit trail: when this training data was added
    created_at TIMESTAMP DEFAULT NOW(),
    
    -- Audit trail: when this training data was last modified
    -- Useful for tracking improvements to responses
    updated_at TIMESTAMP DEFAULT NOW(),
    
    -- User ID who created this entry (default 1001 = system)
    created_by INTEGER DEFAULT 1001,
    
    -- User ID who last updated this entry
    updated_by INTEGER DEFAULT 1001
);

-- ============================================================
-- TABLE 2: sft_experiments
-- Purpose: Tracks fine-tuning experiments and their configurations
-- ============================================================
-- Each row represents one model training run with specific
-- hyperparameters. Allows comparison of different training
-- strategies and department-specific model training.
-- ============================================================

CREATE TABLE IF NOT EXISTS sft_experiments (
    -- Primary key
    id SERIAL PRIMARY KEY,
    
    -- Human-readable experiment name
    -- Example: "Cardiology_SFT_Model", "Diabetes_v2"
    experiment_name TEXT NOT NULL,
    
    -- Medical department this model specializes in
    -- Example: "Cardiology", "Neurology", "Diabetes"
    -- NULL = trained on all departments
    -- Used to filter training data by keyword matching
    department TEXT,
    
    -- Current status of the training job
    -- Values: 'pending', 'running', 'completed', 'failed'
    -- UI shows progress based on this field
    status TEXT NOT NULL DEFAULT 'pending',
    
    -- ============================================================
    -- MODEL CONFIGURATION
    -- ============================================================
    
    -- Base model from HuggingFace
    -- Default: microsoft/phi-2 (2.7B parameter medical-capable model)
    model_name TEXT DEFAULT 'microsoft/phi-2',
    
    -- LoRA rank (r): Dimensionality of the low-rank adaptation
    -- Lower = fewer trainable parameters (faster, less memory)
    -- Higher = more model capacity (better fit, more memory)
    -- Default: 16 (good balance for medical fine-tuning)
    lora_r INTEGER DEFAULT 16,
    
    -- LoRA alpha: Scaling factor for LoRA updates
    -- Typically 2x the rank for stable training
    -- Default: 32 (= 2 * lora_r)
    lora_alpha INTEGER DEFAULT 32,
    
    -- LoRA dropout: Regularization to prevent overfitting
    -- 0.05 = 5% dropout during training
    -- Helps model generalize to unseen medical questions
    lora_dropout REAL DEFAULT 0.05,
    
    -- Number of training epochs (complete passes through data)
    -- Default: 10 (sufficient for small datasets like 6 samples)
    -- Increase to 20-30 for better learning with more data
    num_epochs INTEGER DEFAULT 10,
    
    -- Batch size: Number of samples processed together
    -- Lower = less GPU memory, slower training
    -- Default: 2 (fits on most GPUs for Phi-2 + LoRA)
    batch_size INTEGER DEFAULT 2,
    
    -- Gradient accumulation: Simulates larger batches
    -- Effective batch size = batch_size * gradient_accumulation_steps
    -- Default: 4 (effective batch = 2 * 4 = 8)
    gradient_accumulation_steps INTEGER DEFAULT 4,
    
    -- Learning rate: Step size for weight updates
    -- 0.0001 = conservative, stable for medical domain
    -- Too high = unstable, too low = slow learning
    learning_rate REAL DEFAULT 0.0001,
    
    -- Maximum sequence length (tokens)
    -- Longer = can handle longer medical cases
    -- Default: 2048 (good for typical medical Q&A)
    max_seq_length INTEGER DEFAULT 2048,
    
    -- ============================================================
    -- TRAINING METRICS & STATUS
    -- ============================================================
    
    -- Actual number of training samples used
    -- Populated after filtering by department
    -- Example: 6 for Cardiology, 100 for all departments
    training_samples INTEGER DEFAULT 0,
    
    -- When training actually began (NULL if pending)
    started_at TIMESTAMP,
    
    -- When training finished (NULL if not finished)
    completed_at TIMESTAMP,
    
    -- Error message if status='failed'
    -- Contains stack trace for debugging
    error_message TEXT,
    
    -- Filesystem path to saved model adapter
    -- Example: "sft_models/Cardiology_experiment_1/"
    -- Used for loading the model during inference
    model_output_path TEXT,
    
    -- JSON object with training metrics
    -- Example: {"final_loss": 1.8780, "perplexity": 6.54}
    -- Stored as JSONB for efficient querying
    metrics JSONB DEFAULT '{}',
    
    -- Audit trail: when experiment was created
    created_at TIMESTAMP DEFAULT NOW(),
    
    -- User ID who created this experiment
    created_by INTEGER DEFAULT 1001
);

-- ============================================================
-- INDEXES
-- ============================================================

-- Index for fast group_id lookups when building training datasets
-- Without this, filtering by department would scan entire table
-- Critical for: "Get all rank-1 responses for department X"
CREATE INDEX IF NOT EXISTS idx_sft_ranked_data_group_id 
    ON sft_ranked_data(group_id);

-- Optional: Index for department filtering (add if querying slow)
-- CREATE INDEX IF NOT EXISTS idx_sft_ranked_data_prompt_lower
--     ON sft_ranked_data(LOWER(prompt));

-- Optional: Index for experiment status filtering
-- CREATE INDEX IF NOT EXISTS idx_sft_experiments_status
--     ON sft_experiments(status);

-- ============================================================
-- PERMISSIONS
-- ============================================================

-- Grant full access to the application user (pcesuser)
GRANT ALL ON sft_ranked_data TO pcesuser;
GRANT ALL ON sft_experiments TO pcesuser;

-- Grant sequence access for auto-increment IDs
GRANT USAGE, SELECT ON SEQUENCE sft_ranked_data_id_seq TO pcesuser;
GRANT USAGE, SELECT ON SEQUENCE sft_experiments_id_seq TO pcesuser;

-- ============================================================
-- VERIFICATION
-- ============================================================

-- Check that tables were created and count rows
SELECT 'sft_ranked_data' AS table_name, COUNT(*) AS rows FROM sft_ranked_data
UNION ALL
SELECT 'sft_experiments', COUNT(*) FROM sft_experiments;

-- Check permissions
SELECT 
    grantee, 
    privilege_type 
FROM information_schema.table_privileges 
WHERE table_name IN ('sft_ranked_data', 'sft_experiments')
    AND grantee = 'pcesuser';
