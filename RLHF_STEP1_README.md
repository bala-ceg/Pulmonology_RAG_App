# RLHF Implementation - Step 1: SBERT + Logistic Regression

## Overview

This is **Step 1** of a 3-stage RLHF (Reinforcement Learning from Human Feedback) pipeline for improving medical AI responses based on Subject Matter Expert (SME) ratings.

### What This Does

This implementation uses:
- **SBERT (Sentence-BERT)** to convert text into embeddings (vectors)
- **Logistic Regression** to predict answer quality based on SME ratings (0-5 scale)
- **Re-ranking** to select the best candidate answers during inference

### Why Start Simple?

✅ **Easy to train** - No complex setup  
✅ **Requires little data** - Works with 50+ rated samples  
✅ **Very fast** - Millisecond-level inference  
✅ **Good baseline** - Provides immediate value while you collect more data

This is the "training wheels" version that bootstraps your RLHF pipeline. Later, you can upgrade to transformer-based reward models and DPO/PPO fine-tuning.

---

## Files Created

### Core Implementation
1. **`model_utils.py`** - Database utilities and model persistence
2. **`train_reward_sbert.py`** - Training script for the reward model
3. **`rlhf_reranker.py`** - Inference-time re-ranking module
4. **`test_rlhf_pipeline.py`** - Comprehensive test suite

### Configuration
- **`requirements.txt`** - Updated with new dependencies

---

## Installation

### 1. Install Dependencies

```bash
cd Pulmonology_RAG_App
pip install -r requirements.txt
```

Key new dependencies:
- `sentence-transformers` - SBERT embeddings
- `scikit-learn` - Logistic regression (already present)
- `sqlalchemy` - Database operations
- `joblib` - Model serialization

### 2. Verify Installation

```bash
python test_rlhf_pipeline.py
```

This will test:
- ✅ Database connectivity
- ✅ Model utilities
- ✅ SBERT embeddings
- ✅ Reranker functionality
- ✅ Training data availability

---

## Usage Workflow

### Stage 1: Collect Feedback (50+ samples needed)

1. **Access RLHF Admin Interface**
   ```
   http://localhost:5000/admin/rlhf
   ```

2. **Rate AI Responses**
   - Use the existing RLHF admin interface
   - Rate responses on a 1-5 scale:
     - **1-3**: Poor/mediocre quality (negative class)
     - **4-5**: Good/excellent quality (positive class)
   - Add feedback comments for context

3. **Monitor Progress**
   ```bash
   # Check how many rated samples you have
   python -c "from model_utils import engine; \
              from sqlalchemy import text; \
              with engine.connect() as c: \
                  r = c.execute(text('SELECT COUNT(*) FROM rlhf_interactions WHERE rating IS NOT NULL')); \
                  print(f'Rated samples: {r.scalar()}')"
   ```

### Stage 2: Train the Reward Model

Once you have **50+ rated interactions**:

```bash
python train_reward_sbert.py
```

**What happens:**
1. Loads rated interactions from `rlhf_interactions` table
2. Creates SBERT embeddings for (prompt, answer) pairs
3. Trains logistic regression to predict quality (rating >= 4)
4. Evaluates on test set and reports accuracy/AUC
5. Saves model to `reward_model.joblib`
6. Logs training run to `rlhf_reward_model_training` table

**Expected output:**
```
================================================================================
RLHF REWARD MODEL TRAINING - SBERT + Logistic Regression
================================================================================
📊 Configuration:
   Database: 4.155.102.23:5432/pces_base
   Embedding Model: all-MiniLM-L6-v2
   Output Model: reward_model.joblib
   Min Samples: 50
   Positive Rating Threshold: 4+
================================================================================

📥 Loading feedback from database...
   Loaded 150 interactions with ratings

🔧 Preparing dataset...
   Total samples: 150
   Positive samples (rating >= 4): 89 (59.3%)
   Negative samples (rating < 4): 61 (40.7%)

🧠 Computing SBERT embeddings...
   Embedding shape: (150, 384)

✂️ Splitting train/test...
   Train samples: 132
   Test samples: 18

🎯 Training logistic regression model...
   Training completed!

📊 Evaluating model...
   Accuracy: 0.8333
   AUC-ROC: 0.8920

💾 Saving model...
✅ Model saved to: reward_model.joblib

📝 Logging training run to database...
✅ Training run logged to database

================================================================================
✅ TRAINING COMPLETED SUCCESSFULLY!
================================================================================

📦 Model saved to: reward_model.joblib
📊 Training samples: 150
📈 Test accuracy: 0.8333
📈 Test AUC: 0.8920
```

### Stage 3: Test the Model

```bash
python rlhf_reranker.py
```

This runs a test with sample medical queries to verify re-ranking works correctly.

### Stage 4: Integrate into Flask App

Add to your inference endpoint (e.g., `/data` or `/data-html`):

```python
from rlhf_reranker import rerank_candidates, is_model_ready

# After obtaining candidates from your RAG system
if is_model_ready():
    # Re-rank candidates by predicted quality
    ranked = rerank_candidates(user_prompt, candidates)
    
    # Use the best candidate
    best_answer = ranked[0]['text']
    
    # Optional: Log score for monitoring
    print(f"Selected answer score: {ranked[0]['_score']:.4f}")
else:
    # Fallback if model not trained yet
    best_answer = candidates[0]['text']
```

**Example candidates format:**
```python
candidates = [
    {"text": "Answer from vector DB...", "source": "vector_db"},
    {"text": "Answer from Wikipedia...", "source": "wikipedia"},
    {"text": "Answer from ArXiv...", "source": "arxiv"}
]
```

---

## Configuration

### Environment Variables

Set these in your `.env` file:

```bash
# Database (uses existing DB_HOST, DB_USER, etc.)
DB_HOST=4.155.102.23
DB_PORT=5432
DB_NAME=pces_base
DB_USER=pcesuser
DB_PASSWORD=Pcesuser101

# RLHF Settings (optional - defaults shown)
EMB_MODEL=all-MiniLM-L6-v2              # SBERT model to use
REWARD_MODEL_PATH=reward_model.joblib   # Where to save/load model
MIN_SAMPLES_TO_TRAIN=50                 # Minimum samples for training
POSITIVE_RATING_THRESHOLD=4             # Rating >= 4 is "good"
TRAIN_LIMIT=0                           # Optional: limit samples (0=all)
```

### Adjusting the Rating Threshold

If your rating scale differs:
- **Conservative** (only excellent): `POSITIVE_RATING_THRESHOLD=5`
- **Default** (good and excellent): `POSITIVE_RATING_THRESHOLD=4`
- **Permissive** (average and above): `POSITIVE_RATING_THRESHOLD=3`

---

## Monitoring & Retraining

### Check Training History

```bash
python -c "from model_utils import engine; \
           from sqlalchemy import text; \
           with engine.connect() as c: \
               r = c.execute(text('SELECT * FROM rlhf_reward_model_training ORDER BY created_dt DESC LIMIT 5')); \
               for row in r: print(row)"
```

### Retrain Periodically

As you collect more feedback:

```bash
# Retrain every week/month
python train_reward_sbert.py
```

The model will automatically improve as you add more rated samples.

### Monitor Re-ranking Performance

Add logging to your Flask endpoint:

```python
if is_model_ready():
    ranked = rerank_candidates(user_prompt, candidates)
    
    # Log for analysis
    print(f"Re-ranking scores: {[c['_score'] for c in ranked]}")
    print(f"Score spread: {ranked[0]['_score'] - ranked[-1]['_score']:.4f}")
```

---

## Troubleshooting

### "Not enough labeled samples to train"

**Solution:** Collect more SME ratings via `/admin/rlhf` interface. You need at least 50 rated interactions.

### "Reward model not found"

**Solution:** Train the model first:
```bash
python train_reward_sbert.py
```

### "Database connection failed"

**Solution:** Verify your database credentials in `.env`:
```bash
python -c "from model_utils import check_database_tables; print(check_database_tables())"
```

### Low Accuracy (< 0.7)

**Possible causes:**
- Not enough training data
- Inconsistent ratings from SMEs
- Rating threshold too low/high

**Solutions:**
- Collect 200+ rated samples for better performance
- Ensure raters understand the rating scale
- Adjust `POSITIVE_RATING_THRESHOLD` if needed

### Model Not Loading in Flask

**Check:**
```python
from rlhf_reranker import get_model_info
print(get_model_info())
```

Ensure `model_path` points to an existing file.

---

## Database Schema

### Required Tables

#### `rlhf_interactions`
Stores user prompts, AI responses, and SME ratings:
```sql
- interaction_id (PK)
- session_id
- user_prompt (TEXT)
- ai_response (TEXT)
- rating (INTEGER 1-5)
- feedback_comment (TEXT)
- bias_flag (BOOLEAN)
- created_dt, updated_dt
```

#### `rlhf_reward_model_training` (optional)
Logs training runs:
```sql
- training_id (PK)
- model_version (TEXT)
- total_interactions (INTEGER)
- avg_reward (FLOAT)
- accuracy (FLOAT)
- trained_by (INTEGER)
- created_dt, updated_dt
```

---

## Next Steps (Future Enhancements)

### Step 2: Transformer-Based Reward Model (Week 2)
- Replace logistic regression with BERT/RoBERTa
- Capture more nuanced patterns
- Requires 200+ samples for best results

### Step 3: DPO Fine-Tuning (Week 3+)
- Fine-tune Llama or other open-source LLM
- Learn preferences directly in the model
- Requires 500+ preference pairs

---

## Performance Expectations

### With 50-100 Samples
- Accuracy: **70-80%**
- Useful for re-ranking among similar candidates

### With 200+ Samples
- Accuracy: **80-85%**
- Reliable quality predictions

### With 500+ Samples
- Accuracy: **85-90%**
- Ready for transformer upgrade

---

## Testing

Run the full test suite:

```bash
python test_rlhf_pipeline.py
```

Expected output:
```
================================================================================
RLHF PIPELINE TEST SUITE - Step 1: SBERT + Logistic Regression
================================================================================

🧪 TEST: Testing database connection and tables...
✅ Database connection successful
✅ rlhf_interactions table exists
⚠️  rlhf_reward_model_training table NOT found (optional)

🧪 TEST: Testing model utilities...
✅ Model saved successfully
✅ Model loaded and verified successfully

🧪 TEST: Testing SBERT embeddings...
✅ Embeddings generated successfully: shape (3, 384)
✅ Embedding similarity check passed

🧪 TEST: Testing reranker (no model)...
✅ Re-ranking returned 3 candidates
✅ All candidates have _score field

🧪 TEST: Checking training data availability...
⚠️  Insufficient training data (25 samples, need 50)

Results: 4/5 tests passed
```

---

## Questions?

### How does it improve responses?
The model learns from SME ratings what makes a "good" answer. During inference, it scores all candidate answers and picks the best one.

### How often should I retrain?
Weekly or monthly, depending on how much new feedback you collect.

### Can I use a different embedding model?
Yes! Set `EMB_MODEL=sentence-transformers/all-mpnet-base-v2` for higher quality (but slower).

### What if I have < 50 samples?
The model won't train. Collect more feedback first, or lower `MIN_SAMPLES_TO_TRAIN` (not recommended - accuracy will suffer).

---

## Contact & Support

For questions or issues with this implementation, please contact your development team or refer to the inline code documentation.

---

**Status:** ✅ Step 1 Complete - Ready for SME feedback collection!

**Next Milestone:** Collect 50+ rated samples and train your first reward model! 🎯
