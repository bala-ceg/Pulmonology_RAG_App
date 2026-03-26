# SME Review Workflow - Bug Fix Applied

## Issue Found
The SME review API endpoints were failing with:
```
NameError: name 'get_db_connection' is not defined
NameError: name 'logger' is not defined
```

## Root Cause
The new API functions I created used:
1. `get_db_connection()` - which doesn't exist in your codebase
2. `logger.error()` - logger wasn't imported

Your existing code uses:
1. `psycopg.connect(**db_config)` for database connections
2. `print()` for error logging

## Fix Applied ✅

Updated all three SME API functions in `main.py`:
- `api_get_sme_review_queue()` (line ~4750)
- `api_submit_sme_reviews()` (line ~4855)
- `api_get_sme_review_stats()` (line ~4909)

**Changes made:**
```python
# BEFORE (incorrect)
conn = get_db_connection()
cur = conn.cursor()
...
logger.error(f"Error: {str(e)}")

# AFTER (correct)
with psycopg.connect(**db_config) as conn:
    with conn.cursor() as cur:
        ...
print(f"Error: {str(e)}")
```

## Next Steps Required

### 1. Run Database Migration (CRITICAL)

The new columns don't exist in your database yet. Run this:

```bash
# Option A: Using psql command line
psql -h 4.155.102.23 -p 5432 -U <your_admin_user> -d pces_base \
  -f migrations/add_sme_review_columns.sql

# Option B: Manual execution
psql -h 4.155.102.23 -p 5432 -U <your_admin_user> -d pces_base

# Then paste the following SQL:
ALTER TABLE sft_ranked_data ADD COLUMN IF NOT EXISTS domain TEXT;
ALTER TABLE sft_ranked_data ADD COLUMN IF NOT EXISTS sme_score INTEGER CHECK (sme_score >= 1 AND sme_score <= 5);
ALTER TABLE sft_ranked_data ADD COLUMN IF NOT EXISTS sme_score_reason TEXT;
ALTER TABLE sft_ranked_data ADD COLUMN IF NOT EXISTS sme_reviewed_by TEXT;
ALTER TABLE sft_ranked_data ADD COLUMN IF NOT EXISTS sme_reviewed_at TIMESTAMP;
CREATE INDEX IF NOT EXISTS idx_sft_ranked_data_domain ON sft_ranked_data(domain);
CREATE INDEX IF NOT EXISTS idx_sft_ranked_data_sme_score ON sft_ranked_data(sme_score);
```

### 2. Verify Migration

```sql
-- Check that columns were added
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'sft_ranked_data' 
  AND column_name IN ('domain', 'sme_score', 'sme_score_reason', 'sme_reviewed_by', 'sme_reviewed_at');

-- Should return 5 rows
```

### 3. Restart Application (if needed)

The code changes are already in place. If your Flask app auto-reloads on file changes, you're good. Otherwise:

```bash
# Stop the app
pkill -f "python3 main.py"

# Start it again
cd /Users/bseetharaman/Desktop/Bala/2025/AI_Medical_App/Pulmonology_RAG_App
nohup python3 main.py > nohup.out 2>&1 &
```

### 4. Test the API

```bash
# Test stats endpoint
curl "http://localhost:3000/api/rlhf/sme-review-stats"

# Should return:
{
  "success": true,
  "total": 0,
  "reviewed": 0,
  "pending": 0,
  "reviewed_this_week": 0,
  "review_percentage": 0.0,
  "score_distribution": {},
  "top_reviewers": []
}

# Test review queue
curl "http://localhost:3000/api/rlhf/sme-review-queue?status=all&page=1&per_page=50"

# Should return:
{
  "success": true,
  "items": [],
  "total": 0,
  "page": 1,
  "per_page": 50,
  "total_pages": 0
}
```

### 5. (Optional) Add Sample Data

```bash
# Update password in the script first
nano populate_sme_sample_data.py
# Change line 11: 'password': 'your_actual_password'

# Run the script
python3 populate_sme_sample_data.py
```

## Summary

**Status**: ✅ Code fixes applied, ⏳ Database migration needed

**Files Modified**:
- `main.py` - Fixed 3 API endpoints to use correct DB connection pattern

**Action Required**:
1. Run database migration (see step 1 above)
2. Restart app if needed
3. Test the SME Review Queue tab in UI

**After Migration, the SME Review Queue will work!** 🎉

---
**Updated**: March 16, 2026
**Issue**: Database connection errors
**Resolution**: Updated code to match existing patterns + migration needed
