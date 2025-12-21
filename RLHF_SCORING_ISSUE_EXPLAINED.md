# RLHF Model Scoring Issue - Root Cause & Solution

## 🔴 Problem Identified

All predictions returning **0.5 (50%)** for every answer when testing the RLHF model.

## 🔍 Root Cause Analysis

### Issue: Duplicate Training Data

Your database contains **20 identical samples** - all with the same prompt and response:

```
Prompt: "What is the latest treatment guideline for hypertension?..."
Response: "Based on AHA 2023, combination therapy with ACE inhibitors..."
```

### Why This Breaks the Model:

1. **Identical Embeddings**: SBERT creates the same 384-dimensional embedding for all samples
2. **No Class Separation**: High-quality (rating ≥4) and low-quality (rating <4) samples have identical embeddings
3. **Zero Coefficients**: Logistic regression learns coefficient norm = 0.0 (no pattern to learn)
4. **Prediction**: Model always predicts 0.5 (perfectly uncertain) because it literally cannot distinguish between classes

**Evidence:**
```
Class 0 (low quality) mean embedding: [-0.0452, -0.0051, -0.0582, ...]
Class 1 (high quality) mean embedding: [-0.0452, -0.0051, -0.0582, ...]  # IDENTICAL!
Difference norm: 0.0
Model coefficient norm: 0.0
```

## ✅ Solution: Add Diverse Training Data

The model needs **real, diverse medical Q&A pairs** with genuine quality differences.

### Option 1: Manual Entry via UI (Recommended)

Add samples through the RLHF Admin Panel:

1. Go to `http://127.0.0.1:3000/admin/rlhf`
2. Click "➕ Add Sample" tab
3. Add at least 30-40 diverse samples covering:

**High Quality Examples (Rating 5):**
```
Q: What are the symptoms of pneumonia?
A: Pneumonia symptoms typically include persistent cough with phlegm (yellow, green, or bloody), high fever (101-105°F), chest pain that worsens with breathing or coughing, shortness of breath, fatigue, confusion (especially in elderly), nausea, vomiting, and diarrhea. Severe cases may present with bluish lips or nails. Seek immediate medical attention if experiencing severe symptoms.
Rating: 5
```

**Medium Quality Examples (Rating 3):**
```
Q: What are the symptoms of pneumonia?
A: Pneumonia symptoms include cough, fever, chest pain, and difficulty breathing. You should see a doctor if symptoms are severe.
Rating: 3
```

**Low Quality Examples (Rating 1-2):**
```
Q: What are the symptoms of pneumonia?
A: You might have a cough and feel tired.
Rating: 1
```

### Key Requirements for Training Data:

1. **Diversity**: Different medical topics (pneumonia, asthma, diabetes, hypertension, COPD, stroke, etc.)
2. **Quality Variation**: Mix of excellent, good, mediocre, and poor responses
3. **Clear Differences**: High-quality responses should be noticeably more comprehensive
4. **Realistic**: Based on actual patient questions and medical responses
5. **Quantity**: Minimum 30-40 samples (target: 100+)

### Option 2: Import from Real System Data

If you have actual patient Q&A logs with SME ratings:

```python
# Extract from your production logs
# Import into rlhf_interactions table
# Ensure varied topics and quality levels
```

### Option 3: Generate Programmatically (If Allowed)

See `add_diverse_samples.py` - contains 18 diverse examples, but requires database INSERT permissions.

## 📊 Expected Results After Retraining

With diverse data, you should see:

**Before (Current):**
```
Coefficient norm: 0.0
All scores: 0.5000 (50%)
```

**After (With Diverse Data):**
```
Coefficient norm: >0.1 (model learned patterns)
High-quality responses: 70-90%
Medium-quality responses: 50-70%
Low-quality responses: 20-50%
```

## 🔧 Next Steps

1. **Add 30-40 diverse samples** via UI (different topics, varied quality)
2. **Retrain model**: `python train_reward_sbert.py`
3. **Verify learning**: Check that coefficient norm > 0
4. **Test in UI**: Scores should now vary significantly
5. **Continue**: Add more samples weekly and retrain

## 📈 Performance Expectations

| Samples | Expected Accuracy | Score Variation |
|---------|------------------|-----------------|
| 20 (current duplicates) | N/A | None (all 0.5) |
| 30-40 diverse | 70-75% | Moderate |
| 50-80 diverse | 75-85% | Good |
| 100+ diverse | 85-92% | Excellent |

## 💡 Important Notes

- **The RLHF system is working correctly** - it just needs proper training data
- Duplicate/identical samples cannot train any machine learning model
- Real-world diverse data is essential for RLHF to work
- This is a data quality issue, not a code issue

## 🎯 Temporary Workaround (Demo Only)

For demonstration purposes, you can manually test the scoring logic with this command:

```python
from rlhf_reranker import score_text_pair

# Even with broken model, the API structure works
# Just need real training data to make it learn
```

---

**Bottom Line**: Add diverse, realistic medical Q&A samples through the UI, then retrain. The model will then produce varied scores reflecting actual quality differences.
