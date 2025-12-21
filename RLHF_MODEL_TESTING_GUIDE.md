# RLHF Reward Model Testing Guide

## Overview
The RLHF Admin Panel now includes a **Test Model** tab where you can interact with the trained reward model in real-time. This allows you to validate model behavior, test scoring accuracy, and compare multiple candidate answers.

---

## Prerequisites

1. **Trained Model**: Ensure you have trained the model first
   - Go to "Model Training" tab
   - Click "Train Model Now" (requires ≥20 rated samples)
   - OR run: `python train_reward_sbert.py`

2. **Flask Server Running**:
   ```bash
   python main.py
   ```

3. **Access Admin Panel**:
   ```
   http://127.0.0.1:3000/admin/rlhf
   ```

---

## Testing Features

### 1. 🧪 Test Model Tab

Navigate to the **"Test Model"** tab in the admin panel to access two testing modes:

#### **Mode 1: Score Single Answer** 🎯

Test how the model scores individual prompt-response pairs.

**How to Use:**
1. Enter a medical question in the **"Medical Question / Prompt"** field
2. Enter an AI response in the **"AI Response / Answer"** field  
3. Click **"📊 Score Answer"**
4. View the quality score (0-100%) and interpretation

**Example Test Cases:**

**Test 1 - Good Quality Response:**
```
Prompt: What are the symptoms of pneumonia?
Response: Pneumonia symptoms typically include persistent cough with phlegm, high fever (101-105°F), chest pain that worsens with breathing, shortness of breath, fatigue, confusion (especially in elderly), and sometimes nausea or vomiting. Seek immediate medical attention if experiencing severe symptoms.
```
Expected: **High score (>70%)** - Comprehensive, medically accurate

**Test 2 - Poor Quality Response:**
```
Prompt: What are the symptoms of pneumonia?
Response: You might cough.
```
Expected: **Low score (<50%)** - Too brief, lacks detail

**Test 3 - Medium Quality Response:**
```
Prompt: What are the symptoms of pneumonia?
Response: Pneumonia symptoms include cough, fever, chest pain, and difficulty breathing.
```
Expected: **Medium score (50-70%)** - Adequate but not comprehensive

---

#### **Mode 2: Re-rank Multiple Answers** 🏆

Test how the model ranks multiple candidate answers against each other.

**How to Use:**
1. Enter a medical question in **"Medical Question"**
2. Enter 2-3 candidate answers in the respective fields
3. Click **"🏆 Rank Answers"**
4. View ranked results from best to worst with scores

**Example Test Case:**

**Prompt:**
```
What medications are used to treat asthma?
```

**Candidate 1 (Short/Incomplete):**
```
Inhalers like albuterol provide quick relief for asthma symptoms.
```

**Candidate 2 (Comprehensive/Best):**
```
Asthma treatment includes two main categories: long-term control medications such as inhaled corticosteroids (fluticasone, budesonide) and leukotriene modifiers (montelukast), plus quick-relief medications like short-acting beta agonists (albuterol). Treatment plans are personalized based on symptom severity and frequency.
```

**Candidate 3 (Non-medical/Poor):**
```
Try breathing exercises and avoid allergens.
```

**Expected Ranking:**
1. 🥇 Candidate 2 (highest score) - Most comprehensive
2. 🥈 Candidate 1 (medium score) - Partially complete
3. 🥉 Candidate 3 (lowest score) - Non-specific, lacks medication info

---

## Understanding Scores

### Score Interpretation:

| Score Range | Quality Level | Meaning |
|------------|--------------|---------|
| **70-100%** | ✅ High Quality | Model predicts SMEs would rate this ≥4/5 |
| **50-70%** | ⚠️ Medium Quality | Acceptable but could be improved |
| **0-50%** | ❌ Low Quality | Model predicts SMEs would rate this <4/5 |

### What Influences Scores:

The model learns from your SME ratings and looks for patterns like:
- **Comprehensiveness**: Detailed vs. brief responses
- **Medical accuracy**: Specific medical terms vs. vague language
- **Structure**: Well-organized vs. disorganized information
- **Completeness**: Addresses all aspects vs. partial answers

---

## API Testing (Developer Mode)

You can also test the model programmatically via API endpoints:

### 1. Get Model Info
```bash
curl http://127.0.0.1:3000/api/rlhf/model_info
```

**Response:**
```json
{
  "success": true,
  "model_ready": true,
  "model_path": "reward_model.joblib",
  "embedding_model": "all-MiniLM-L6-v2",
  "message": "Model loaded and ready"
}
```

### 2. Score Single Answer
```bash
curl -X POST http://127.0.0.1:3000/api/rlhf/score \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What are the symptoms of pneumonia?",
    "answer": "Pneumonia symptoms include cough, fever, chest pain, and difficulty breathing."
  }'
```

**Response:**
```json
{
  "success": true,
  "score": 0.6234,
  "prompt": "What are the symptoms of pneumonia?",
  "answer": "Pneumonia symptoms include..."
}
```

### 3. Re-rank Candidates
```bash
curl -X POST http://127.0.0.1:3000/api/rlhf/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What medications treat asthma?",
    "candidates": [
      "Inhalers like albuterol provide quick relief.",
      "Long-term control medications include inhaled corticosteroids.",
      "Try breathing exercises."
    ]
  }'
```

**Response:**
```json
{
  "success": true,
  "ranked": [
    {"answer": "Long-term control...", "score": 0.8912, "rank": 1},
    {"answer": "Inhalers like...", "score": 0.7234, "rank": 2},
    {"answer": "Try breathing...", "score": 0.4123, "rank": 3}
  ]
}
```

---

## Python Testing (Command Line)

Test the model directly from Python:

```python
from rlhf_reranker import score_text_pair, rerank_candidates, get_model_info

# Check model status
info = get_model_info()
print(f"Model: {info}")

# Score a single answer
prompt = "What are the symptoms of pneumonia?"
answer = "Pneumonia symptoms include cough, fever, chest pain, and difficulty breathing."
score = score_text_pair(prompt, answer)
print(f"Score: {score:.4f} ({score*100:.1f}%)")

# Re-rank multiple candidates
candidates = [
    "Inhalers like albuterol provide quick relief.",
    "Long-term control medications include inhaled corticosteroids like fluticasone.",
    "Try breathing exercises."
]
ranked = rerank_candidates(prompt, candidates)
for item in ranked:
    print(f"Rank {item['rank']}: {item['score']:.4f} - {item['answer'][:50]}...")
```

---

## Troubleshooting

### ❌ "Model not available" Error

**Cause**: Model hasn't been trained yet

**Solution**:
1. Go to "Model Training" tab
2. Ensure you have ≥20 rated samples
3. Click "Train Model Now"
4. Wait for training to complete (~30 seconds)

### ❌ Low Scores for All Answers

**Cause**: Insufficient training data (only 20 samples)

**Solution**:
- Collect more SME ratings (target: 100+ samples)
- Retrain the model periodically
- Accuracy improves with more diverse training data

### ❌ Inconsistent Rankings

**Cause**: Model is still learning with limited data

**Solution**:
- Current 20 samples → ~67% accuracy
- Add 30 more samples → ~75% accuracy  
- Add 80 more samples → ~85%+ accuracy

---

## Best Practices

1. **Test with Real Queries**: Use actual patient questions from your system
2. **Compare Against SME Ratings**: Test answers you've already rated to validate model behavior
3. **Test Edge Cases**: Try very short, very long, and off-topic responses
4. **Regular Retraining**: Retrain weekly as new ratings accumulate
5. **Monitor Drift**: Keep testing the same questions to detect performance changes

---

## What's Next?

After validating the model:

1. **Integrate into Production**: Use `rerank_candidates()` in your main inference pipeline
2. **A/B Testing**: Compare RLHF-reranked vs. baseline responses
3. **Monitoring**: Track user satisfaction with reranked answers
4. **Continuous Learning**: Collect feedback → retrain → deploy loop

---

## Support

For issues or questions:
- Check Flask terminal logs for error messages
- Review `RLHF_QUICKSTART.md` for setup details
- Verify model file exists: `ls -lh reward_model.joblib`
- Test model import: `python -c "from rlhf_reranker import is_model_ready; print(is_model_ready())"`
