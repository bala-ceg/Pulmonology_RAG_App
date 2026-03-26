# SME Review Workflow Implementation Guide

## Overview

This document describes the new **SME (Subject Matter Expert) Review Queue** feature added to the RLHF Admin panel. This feature enables a two-phase workflow for training data curation:

1. **Phase 1**: Regular users (e.g., practicing cardiologists) create medical prompts with ranked LLM responses
2. **Phase 2**: Domain experts/Directors review these prompts weekly, providing expert scores and detailed reasoning

## Architecture

### Database Schema Changes

New columns added to `sft_ranked_data` table:

```sql
domain              TEXT                 -- Medical department (Cardiology, Pulmonology, etc.)
sme_score           INTEGER (1-5)        -- Expert's quality rating
sme_score_reason    TEXT                 -- Expert's detailed reasoning
sme_reviewed_by     TEXT                 -- Name of SME reviewer
sme_reviewed_at     TIMESTAMP            -- When review occurred
```

**Migration Script**: `migrations/add_sme_review_columns.sql`

### API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/rlhf/sme-review-queue` | GET | Fetch prompts for SME review with filtering |
| `/api/rlhf/sme-review-submit` | POST | Batch save SME reviews |
| `/api/rlhf/sme-review-stats` | GET | Get review progress statistics |

#### GET `/api/rlhf/sme-review-queue`

**Query Parameters:**
- `domain` (optional): Filter by medical department
- `status`: `pending` | `reviewed` | `all`
- `page`: Page number (default: 1)
- `per_page`: Items per page (default: 50)

**Response:**
```json
{
  "success": true,
  "items": [
    {
      "id": 123,
      "prompt": "Effect of COVID to heart patient...",
      "response_text": "There is direct effect on...",
      "rank": 1,
      "reason": "Initial user reasoning",
      "domain": "Cardiology",
      "sme_score": null,
      "sme_score_reason": null,
      "sme_reviewed_by": null,
      "sme_reviewed_at": null,
      "created_at": "2025-01-03T10:30:00",
      "created_by": 1001
    }
  ],
  "total": 150,
  "page": 1,
  "per_page": 50,
  "total_pages": 3
}
```

#### POST `/api/rlhf/sme-review-submit`

**Request Body:**
```json
{
  "sme_name": "Dr. Jack Nicholson",
  "reviews": [
    {
      "id": 123,
      "sme_score": 5,
      "sme_score_reason": "Great one - covers all aspects"
    },
    {
      "id": 124,
      "sme_score": 1,
      "sme_score_reason": "Numerous cases reported, side effects include lung disease"
    }
  ]
}
```

**Response:**
```json
{
  "success": true,
  "updated_count": 2,
  "message": "Successfully updated 2 reviews"
}
```

#### GET `/api/rlhf/sme-review-stats`

**Query Parameters:**
- `domain` (optional): Filter stats by department

**Response:**
```json
{
  "success": true,
  "total": 200,
  "reviewed": 45,
  "pending": 155,
  "reviewed_this_week": 18,
  "review_percentage": 22.5,
  "score_distribution": {
    "1": 5,
    "2": 8,
    "3": 10,
    "4": 12,
    "5": 10
  },
  "top_reviewers": [
    {"name": "Dr. Jack Nicholson", "count": 25},
    {"name": "Dr. Sarah Johnson", "count": 20}
  ]
}
```

## User Interface

### Navigation

Access via: **Admin Panel → Experiments Tab → 👨‍⚕️ SME Review Queue**

### UI Components

#### 1. Statistics Dashboard
- **Total Prompts**: Count of all prompts in database
- **Pending Review**: Prompts without SME scores
- **Reviewed**: Prompts with SME scores
- **Completion %**: Percentage of reviewed prompts
- **This Week**: Reviews completed in last 7 days

#### 2. Filters
- **Department Dropdown**: Filter by medical domain (Cardiology, Pulmonology, etc.)
- **Review Status**: 
  - ⏳ Pending Review (sme_score IS NULL)
  - ✅ Reviewed (sme_score IS NOT NULL)
  - 📊 All
- **SME Name Input**: Name to record as reviewer (e.g., "Dr. Jack Nicholson")

#### 3. Review Table

Columns match the wireframe design from idea1.jpeg and idea2.jpeg:

| Column | Description | Editable |
|--------|-------------|----------|
| ID | Database record ID | No |
| Domain | Medical department badge | No |
| Prompt | Medical question/scenario | No |
| Rank | Initial user ranking (1-3) | No |
| LLM Response | Model-generated answer | No |
| **Score - By SME** | **Expert rating (1-5)** | **Yes** |
| **Score Reason - by SME** | **Expert's detailed reasoning** | **Yes** |
| Created Date | When prompt was created | No |
| Created By | Original creator | No |
| Updated Dt | When SME reviewed | Auto-filled |
| Updated By | SME reviewer name | Auto-filled |

**Color Coding:**
- **Rank 1**: Green badge (best response)
- **Rank 2**: Orange badge (partial/acceptable)
- **Rank 3**: Red badge (poor/incorrect)

#### 4. Batch Save
- **Button**: "💾 Save All Reviews"
- Saves all rows with scores entered in current view
- Validates score range (1-5)
- Records SME name and timestamp

## Workflow

### Weekly Review Process (Typical)

1. **Login as SME** (Director/Department Head)
2. Navigate to **Experiments → SME Review Queue**
3. **Filter by Department**: Select your domain (e.g., "Cardiology")
4. **Set Status**: "Pending Review" (shows only unreviewed prompts)
5. **Enter Your Name**: In "SME Name" field (e.g., "Dr. Jack Nicholson")
6. **Review Prompts**:
   - Read each prompt and LLM response
   - Assign score 1-5 in "Score - By SME" column
   - Provide detailed reasoning in "Score Reason - by SME" textarea
7. **Save**: Click "💾 Save All Reviews" button
8. **Repeat**: Review ~100 prompts per week for 6 weeks

### Scoring Guidelines

| Score | Meaning | Example Reason |
|-------|---------|----------------|
| **5** | Excellent | "Great one - comprehensive, evidence-based, covers all aspects" |
| **4** | Good | "Not just Pneumonia but also cold and cough, high fever and arthritis" |
| **3** | Acceptable | "Correct but lacks specific clinical details" |
| **2** | Needs Improvement | "Generic advice without specific guidelines" |
| **1** | Poor | "Numerous cases reported; incorrect information, side effects include lung disease" |

### 6-Week Cycle

- **Week 1-6**: SME reviews ~100 prompts per week
- **After Week 6**: 
  - Total ~600 reviewed prompts
  - Data with `sme_score >= 4` used for model retraining
  - SFT experiment launched with high-quality curated data

## Installation & Setup

### 1. Run Database Migration

```bash
# Connect to PostgreSQL
psql -h 4.155.102.23 -p 5432 -U <admin_user> -d pces_base

# Run migration
\i migrations/add_sme_review_columns.sql

# Verify columns were added
SELECT column_name FROM information_schema.columns 
WHERE table_name = 'sft_ranked_data' 
  AND column_name LIKE 'sme_%';
```

**Expected Output:**
```
 column_name      
------------------
 sme_score
 sme_score_reason
 sme_reviewed_by
 sme_reviewed_at
```

### 2. Populate Sample Data (Optional)

```bash
# Update database password in script
nano populate_sme_sample_data.py
# Change line: 'password': 'your_password_here'

# Run population script
python3 populate_sme_sample_data.py
```

This creates 6 sample prompts across multiple domains (Cardiology, Pulmonology, Diabetes, Rheumatology) with 2-3 ranked responses each.

### 3. Restart Application

```bash
# If using systemd
sudo systemctl restart pulmonology-app

# Or run directly
python3 main.py
```

### 4. Access UI

```
http://<your-server>:3000/admin/rlhf
→ Click "Experiments" tab
→ Click "👨‍⚕️ SME Review Queue"
```

## Testing

### Manual Testing Checklist

- [ ] **Load Review Queue**: Navigate to SME Review Queue tab
- [ ] **Filter by Domain**: Select "Cardiology" - should show only cardiology prompts
- [ ] **Filter by Status**: Select "Pending" - should show only unreviewed
- [ ] **Enter Score**: Type number 1-5 in "Score - By SME" input
- [ ] **Enter Reason**: Type reasoning in textarea
- [ ] **Invalid Score**: Try entering 0 or 6 - should be rejected
- [ ] **Batch Save**: Fill multiple rows, click "Save All Reviews"
- [ ] **Verify Save**: Check "Updated Dt" and "Updated By" columns populated
- [ ] **Filter Reviewed**: Select "Reviewed" status - should show saved items
- [ ] **Statistics Update**: Verify stats dashboard updates after save
- [ ] **Pagination**: Navigate through pages if >50 items

### API Testing

```bash
# Test review queue endpoint
curl -X GET "http://localhost:3000/api/rlhf/sme-review-queue?domain=Cardiology&status=pending&page=1&per_page=10"

# Test submit reviews
curl -X POST http://localhost:3000/api/rlhf/sme-review-submit \
  -H "Content-Type: application/json" \
  -d '{
    "sme_name": "Dr. Test User",
    "reviews": [
      {"id": 1, "sme_score": 5, "sme_score_reason": "Excellent response"}
    ]
  }'

# Test stats endpoint
curl -X GET "http://localhost:3000/api/rlhf/sme-review-stats?domain=Cardiology"
```

## Data Model Workflow

```
Phase 1: User Creates Data
┌─────────────────────────────────────────────────┐
│ sft_ranked_data                                  │
├─────────────────────────────────────────────────┤
│ id: 101                                          │
│ prompt: "Effect of COVID to heart patient..."   │
│ response_text: "There is direct effect on..."   │
│ rank: 1 (user's ranking)                         │
│ reason: "Comprehensive response..." (user)       │
│ domain: "Cardiology"                             │
│ sme_score: NULL ⏳                               │
│ sme_score_reason: NULL ⏳                        │
│ sme_reviewed_by: NULL ⏳                         │
│ sme_reviewed_at: NULL ⏳                         │
│ created_by: 1001                                 │
│ created_at: 2025-01-03 10:30:00                  │
└─────────────────────────────────────────────────┘

Phase 2: SME Reviews
┌─────────────────────────────────────────────────┐
│ sft_ranked_data (AFTER REVIEW)                   │
├─────────────────────────────────────────────────┤
│ id: 101                                          │
│ prompt: "Effect of COVID to heart patient..."   │
│ response_text: "There is direct effect on..."   │
│ rank: 1                                          │
│ reason: "Comprehensive response..."              │
│ domain: "Cardiology"                             │
│ sme_score: 5 ✅                                  │
│ sme_score_reason: "Great one" ✅                 │
│ sme_reviewed_by: "Dr. Jack Nicholson" ✅         │
│ sme_reviewed_at: 2025-01-18 14:22:00 ✅          │
│ updated_by: 1001                                 │
│ updated_at: 2025-01-18 14:22:00                  │
└─────────────────────────────────────────────────┘
```

## Integration with SFT Experiments

After 6 weeks of SME reviews, use high-scoring data for training:

```python
# Example: Get training data for Cardiology with SME score >= 4
SELECT prompt, response_text, sme_score_reason
FROM sft_ranked_data
WHERE domain = 'Cardiology'
  AND rank = 1
  AND sme_score >= 4
ORDER BY sme_score DESC;
```

This curated, expert-validated data produces higher-quality fine-tuned models.

## Troubleshooting

### "No prompts found for review"
**Cause**: No data in database or filters too restrictive  
**Fix**: 
1. Check if data exists: `SELECT COUNT(*) FROM sft_ranked_data;`
2. Run `populate_sme_sample_data.py` to add sample data
3. Try "All Departments" + "All" status filter

### Reviews not saving
**Cause**: Missing SME name or invalid score range  
**Fix**: 
1. Ensure "SME Name" field is filled
2. Verify scores are 1-5 (integers)
3. Check browser console for JavaScript errors

### Migration fails
**Cause**: Table doesn't exist or permissions issue  
**Fix**:
1. Verify table exists: `\dt sft_ranked_data`
2. Check user permissions: `\du pcesuser`
3. Run as admin user if needed

## Future Enhancements

1. **Email Notifications**: Alert SMEs when new prompts need review
2. **Review Analytics**: Track average review time, score distributions by reviewer
3. **Bulk Import**: Upload CSV of prompts for review
4. **Approval Workflow**: Multi-level review (Associate → Director)
5. **Model Performance Tracking**: Compare models trained on SME score 3+ vs 4+ vs 5

## Files Modified/Created

| File | Status | Purpose |
|------|--------|---------|
| `migrations/add_sme_review_columns.sql` | ✅ Created | Database migration script |
| `main.py` | ✅ Modified | Added 3 SME API routes |
| `templates/admin_rlhf.html` | ✅ Modified | Added SME Review Queue tab + JS |
| `populate_sme_sample_data.py` | ✅ Created | Sample data generator |
| `SME_REVIEW_GUIDE.md` | ✅ Created | This documentation |

## Support

For issues or questions:
1. Check application logs: `tail -f nohup.out`
2. Verify database connectivity: `psql -h 4.155.102.23 -p 5432 -U pcesuser -d pces_base`
3. Review browser console for frontend errors

---

**Last Updated**: March 16, 2026  
**Version**: 1.0  
**Author**: AI Medical App Development Team
