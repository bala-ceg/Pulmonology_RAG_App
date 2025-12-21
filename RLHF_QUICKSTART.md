# RLHF Quick Start Guide

## 🚀 Get Started in 5 Steps

### Step 1: Install Dependencies (1 minute)

```bash
cd Pulmonology_RAG_App
pip install sentence-transformers sqlalchemy joblib
```

### Step 2: Verify Setup (30 seconds)

```bash
python -c "import model_utils; import rlhf_reranker; print('✅ Setup complete!')"
```

Expected output:
```
🔧 Initializing RLHF reranker...
⚠️ Reward model not found (this is normal - you'll train it in Step 4)
✅ Setup complete!
```

### Step 3: Collect SME Ratings (ongoing)

1. Open the RLHF admin interface:
   ```
   http://localhost:5000/admin/rlhf
   ```

2. Rate AI responses (1-5 scale):
   - **1-3**: Poor quality ❌
   - **4-5**: Good quality ✅

3. Goal: **Get 50+ rated interactions**

### Step 4: Train the Model (5 minutes)

Once you have 50+ ratings:

```bash
python train_reward_sbert.py
```

This will:
- ✅ Load your ratings from database
- ✅ Train SBERT + Logistic Regression
- ✅ Save model to `reward_model.joblib`
- ✅ Report accuracy metrics

### Step 5: Integrate into Flask (5 minutes)

Add to your inference endpoint (`/data` or `/data-html`):

```python
from rlhf_reranker import rerank_candidates, is_model_ready

# After getting candidates from RAG:
if is_model_ready():
    ranked = rerank_candidates(user_prompt, candidates)
    best_answer = ranked[0]['text']  # Highest quality
else:
    best_answer = candidates[0]['text']  # Fallback
```

---

## 📊 Monitor Progress

### Check Number of Ratings
```bash
python -c "from model_utils import engine; from sqlalchemy import text; \
with engine.connect() as c: \
    r = c.execute(text('SELECT COUNT(*) FROM rlhf_interactions WHERE rating IS NOT NULL')); \
    print(f'Rated samples: {r.scalar()}/50')"
```

### Check Training History
```bash
python -c "from model_utils import engine; from sqlalchemy import text; \
with engine.connect() as c: \
    r = c.execute(text('SELECT COUNT(*) FROM rlhf_reward_model_training')); \
    print(f'Training runs: {r.scalar()}')"
```

---

## 🎯 Success Metrics

### After 50-100 Ratings
- Accuracy: **70-80%**
- Impact: Moderate improvement in response quality

### After 200+ Ratings
- Accuracy: **80-85%**
- Impact: Significant improvement, ready for production use

### After 500+ Ratings
- Accuracy: **85-90%**
- Impact: Excellent performance, ready for Step 2 (Transformer model)

---

## 🔧 Troubleshooting

### "Not enough samples"
➡️ Collect more ratings (need 50 minimum)

### "Model not found"
➡️ Run `python train_reward_sbert.py` first

### "Database error"
➡️ Check `.env` has correct DB credentials

### Low accuracy (< 70%)
➡️ Collect more diverse ratings (200+ recommended)

---

## 📚 Full Documentation

See [`RLHF_STEP1_README.md`](RLHF_STEP1_README.md) for:
- Detailed configuration options
- Advanced usage patterns
- Database schema
- Performance tuning
- Next steps (Transformer model, DPO)

---

## ✨ That's It!

You now have a working RLHF system that learns from expert feedback to improve response quality.

**Questions?** Check the full README or contact the development team.

**Next milestone:** Collect 50 ratings and train your first model! 🎉
