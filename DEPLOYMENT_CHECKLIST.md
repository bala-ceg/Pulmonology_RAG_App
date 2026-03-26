# SME Review Workflow - Deployment Checklist

## ✅ Implementation Complete

All components of the SME Review Workflow have been successfully implemented.

## Pre-Deployment Checklist

### 1. Database Migration
```bash
# Connect to production database
psql -h 4.155.102.23 -p 5432 -U <admin_user> -d pces_base

# Run migration script
\i /path/to/migrations/add_sme_review_columns.sql

# Verify migration
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'sft_ranked_data' 
  AND column_name IN ('domain', 'sme_score', 'sme_score_reason', 'sme_reviewed_by', 'sme_reviewed_at');

# Expected output: 5 rows showing the new columns
```

### 2. Update Existing Data (Optional)
If you have existing data in `sft_ranked_data` without domain values:

```sql
-- Example: Assign domains based on prompt keywords
UPDATE sft_ranked_data 
SET domain = 'Cardiology' 
WHERE LOWER(prompt) LIKE '%heart%' 
   OR LOWER(prompt) LIKE '%cardio%'
   OR LOWER(prompt) LIKE '%myocardial%';

UPDATE sft_ranked_data 
SET domain = 'Pulmonology' 
WHERE LOWER(prompt) LIKE '%lung%' 
   OR LOWER(prompt) LIKE '%asthma%'
   OR LOWER(prompt) LIKE '%copd%'
   OR LOWER(prompt) LIKE '%respiratory%';

UPDATE sft_ranked_data 
SET domain = 'Diabetes' 
WHERE LOWER(prompt) LIKE '%diabetes%' 
   OR LOWER(prompt) LIKE '%insulin%'
   OR LOWER(prompt) LIKE '%glucose%';

-- Check distribution
SELECT domain, COUNT(*) 
FROM sft_ranked_data 
GROUP BY domain 
ORDER BY COUNT(*) DESC;
```

### 3. Populate Sample Data (Development/Testing Only)
```bash
# Edit the script and update database password
nano populate_sme_sample_data.py
# Update line: 'password': 'your_actual_password'

# Run the script
python3 populate_sme_sample_data.py

# When prompted, type 'yes' to proceed
```

This will create 6 sample prompts across 4 domains with 2-3 ranked responses each.

### 4. Backup Before Deployment
```bash
# Backup the database
pg_dump -h 4.155.102.23 -p 5432 -U pcesuser -d pces_base -F c -f pces_base_backup_$(date +%Y%m%d).dump

# Backup application files
tar -czf pulmonology_app_backup_$(date +%Y%m%d).tar.gz \
  main.py \
  templates/admin_rlhf.html \
  migrations/ \
  populate_sme_sample_data.py \
  SME*.md
```

### 5. Deploy Updated Files
```bash
# Stop the application
sudo systemctl stop pulmonology-app
# OR
pkill -f "python3 main.py"

# Deploy updated files (if using git)
git pull origin main

# OR manually copy files
# - main.py (modified)
# - templates/admin_rlhf.html (modified)
# - migrations/add_sme_review_columns.sql (new)
# - populate_sme_sample_data.py (new)

# Restart the application
sudo systemctl start pulmonology-app
# OR
nohup python3 main.py > nohup.out 2>&1 &
```

### 6. Verify Deployment
```bash
# Check application is running
ps aux | grep "python3 main.py"

# Test API endpoints
curl -s "http://localhost:3000/api/rlhf/sme-review-stats" | jq .

# Expected response:
# {
#   "success": true,
#   "total": <number>,
#   "reviewed": <number>,
#   "pending": <number>,
#   ...
# }

# Check application logs
tail -f nohup.out
# Should show no errors
```

### 7. UI Verification
1. Open browser: `http://your-server:3000/admin/rlhf`
2. Click **Experiments** tab
3. Verify **👨‍⚕️ SME Review Queue** button appears
4. Click the button
5. Verify:
   - Statistics dashboard displays
   - Department filter dropdown populates
   - Review Status filter shows 3 options
   - SME Name input is visible
   - Table loads (may be empty if no data)

### 8. Functional Testing
```
Test Case 1: Load Empty Queue
✓ Navigate to SME Review Queue
✓ Should show "No prompts found for review" if no data exists

Test Case 2: Filter by Department
✓ Select a department from dropdown
✓ Click should trigger loadSMEReviewQueue()
✓ Table should filter to that department

Test Case 3: Add Review
✓ Enter score (1-5) in "Score - By SME" input
✓ Enter reasoning in "Score Reason - by SME" textarea
✓ Enter SME name (e.g., "Dr. Jane Smith")
✓ Click "Save All Reviews"
✓ Should show success message
✓ "Updated Dt" and "Updated By" columns should populate

Test Case 4: View Reviewed Items
✓ Change "Review Status" to "Reviewed"
✓ Should show items with sme_score not null
✓ Score inputs should be readonly (grayed out)

Test Case 5: Statistics Update
✓ After saving reviews, stats should update
✓ "Pending Review" count should decrease
✓ "Reviewed" count should increase
✓ "Completion %" should update
```

## Post-Deployment

### Training Your Team

#### For SMEs (Directors/Department Heads):

**Access Instructions:**
1. Navigate to: `http://your-server:3000/admin/rlhf`
2. Click **Experiments** tab (6th tab)
3. Click **👨‍⚕️ SME Review Queue** (2nd sub-tab)

**Review Workflow:**
1. Select your department (e.g., "Cardiology")
2. Ensure "Pending Review" is selected
3. Enter your name in "SME Name" field
4. For each prompt:
   - Read the medical question
   - Review the LLM response
   - Assign score 1-5:
     - 5 = Excellent (evidence-based, comprehensive)
     - 4 = Good (clinically sound, detailed)
     - 3 = Acceptable (correct but basic)
     - 2 = Needs work (generic, lacks detail)
     - 1 = Poor (incorrect, misleading)
   - Provide detailed reasoning explaining your score
5. Click "Save All Reviews" when done
6. Aim for ~100 reviews per week

#### For Regular Users (Data Creators):

**When Creating Training Data:**
1. Always specify the **domain** (Cardiology, Pulmonology, etc.)
2. Provide multiple ranked responses (Rank 1 = best)
3. Include your reasoning for each rank
4. Note: Your data will be reviewed by department directors

### Monitoring

```bash
# Check review progress weekly
psql -h 4.155.102.23 -p 5432 -U pcesuser -d pces_base << EOF
SELECT 
  domain,
  COUNT(*) as total,
  COUNT(sme_score) as reviewed,
  ROUND(100.0 * COUNT(sme_score) / COUNT(*), 1) as pct_complete
FROM sft_ranked_data
GROUP BY domain
ORDER BY pct_complete DESC;
EOF
```

### Integration with Training Pipeline

After 6 weeks of reviews, filter high-quality data for SFT experiments:

```sql
-- Get high-scoring training data for a department
SELECT 
  prompt,
  response_text,
  sme_score,
  sme_score_reason
FROM sft_ranked_data
WHERE domain = 'Cardiology'
  AND rank = 1                -- Best response for each prompt
  AND sme_score >= 4          -- Good or Excellent
ORDER BY sme_score DESC, sme_reviewed_at DESC;
```

Update SFT experiment manager to filter by `sme_score >= 4` when loading training data.

## Rollback Plan

If issues arise:

```bash
# 1. Stop application
sudo systemctl stop pulmonology-app

# 2. Restore database backup
pg_restore -h 4.155.102.23 -p 5432 -U pcesuser -d pces_base -c pces_base_backup_YYYYMMDD.dump

# 3. Restore application files
tar -xzf pulmonology_app_backup_YYYYMMDD.tar.gz

# 4. Restart application
sudo systemctl start pulmonology-app
```

## Support

### Common Issues

**Issue**: Migration fails with "permission denied"  
**Solution**: Run as database admin user, then grant permissions to pcesuser

**Issue**: UI shows but no data loads  
**Solution**: Check browser console (F12) for JavaScript errors; verify API endpoints return data

**Issue**: Save fails with "SME name required"  
**Solution**: Ensure "SME Name" input field is filled before clicking Save

### Logs to Check

1. **Application logs**: `tail -f nohup.out`
2. **Database logs**: Check PostgreSQL logs on server
3. **Browser console**: F12 → Console tab

## Summary of Changes

| Component | File | Change Type | Lines Changed |
|-----------|------|-------------|---------------|
| Database | migrations/add_sme_review_columns.sql | New | +75 |
| Backend | main.py | Modified | +250 |
| Frontend | templates/admin_rlhf.html | Modified | +370 |
| Utilities | populate_sme_sample_data.py | New | +250 |
| Docs | SME_REVIEW_GUIDE.md | New | +420 |
| Docs | SME_IMPLEMENTATION_README.md | New | +120 |
| Docs | DEPLOYMENT_CHECKLIST.md | New | +280 |

**Total**: ~1,765 lines of new/modified code + comprehensive documentation

## Sign-Off

- [ ] Database migration completed successfully
- [ ] Application restarted without errors
- [ ] UI accessible and functional
- [ ] API endpoints tested and responding
- [ ] Sample data loaded (if applicable)
- [ ] Team trained on new workflow
- [ ] Monitoring in place

**Deployed By**: _________________  
**Date**: _________________  
**Version**: 1.0

---
✅ **Implementation Complete - Ready for Production Use**
