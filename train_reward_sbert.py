# train_reward_sbert.py
"""
Train a logistic regression reward model on SBERT embeddings using SME ratings
from rlhf_interactions.rating. This is Step 1 of the RLHF pipeline.

The model learns to predict how good an answer is based on expert ratings,
and can be used to re-rank candidate answers during inference.

Usage:
    python train_reward_sbert.py
    
Environment Variables:
    DB_URI: PostgreSQL connection string (auto-constructed from DB_HOST, DB_USER, etc.)
    EMB_MODEL: SBERT model name (default: all-MiniLM-L6-v2)
    REWARD_MODEL_PATH: Output path for trained model (default: reward_model.joblib)
    MIN_SAMPLES_TO_TRAIN: Minimum samples required for training (default: 50)
    POSITIVE_RATING_THRESHOLD: Rating threshold for positive class (default: 4)
    TRAIN_LIMIT: Optional limit on number of samples to use (for testing)
"""

import os
import math
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from model_utils import save_reward_model, log_training_run

# Load environment variables
load_dotenv()

# CONFIG
DB_URI = os.getenv(
    "DB_URI",
    f"postgresql+psycopg2://{os.getenv('DB_USER', 'postgres')}:{os.getenv('DB_PASSWORD', 'password')}@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}/{os.getenv('DB_NAME', 'pces_base')}"
)
EMB_MODEL = os.getenv("EMB_MODEL", "all-MiniLM-L6-v2")
OUT_MODEL_PATH = os.getenv("REWARD_MODEL_PATH", "reward_model.joblib")
MIN_SAMPLES_TO_TRAIN = int(os.getenv("MIN_SAMPLES_TO_TRAIN", "20"))  # Lowered to 20 for initial testing
POSITIVE_RATING_THRESHOLD = int(os.getenv("POSITIVE_RATING_THRESHOLD", "4"))

print("=" * 80)
print("RLHF REWARD MODEL TRAINING - SBERT + Logistic Regression")
print("=" * 80)
print(f"📊 Configuration:")
print(f"   Database: {DB_URI.split('@')[-1] if '@' in DB_URI else 'local'}")
print(f"   Embedding Model: {EMB_MODEL}")
print(f"   Output Model: {OUT_MODEL_PATH}")
print(f"   Min Samples: {MIN_SAMPLES_TO_TRAIN}")
print(f"   Positive Rating Threshold: {POSITIVE_RATING_THRESHOLD}+")
print("=" * 80)

engine = create_engine(DB_URI)
embedder = SentenceTransformer(EMB_MODEL)


def load_feedback(limit=None):
    """
    Load rows that have rating from rlhf_interactions.
    Assumes columns: interaction_id, user_prompt, ai_response, rating
    
    Args:
        limit: Optional limit on number of rows to load (for testing)
    
    Returns:
        pandas.DataFrame with feedback data
    """
    query = """
        SELECT interaction_id, user_prompt, ai_response, rating 
        FROM rlhf_interactions 
        WHERE rating IS NOT NULL 
        ORDER BY created_dt DESC
    """
    if limit:
        query += f" LIMIT {limit}"
    
    print(f"\n📥 Loading feedback from database...")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    
    print(f"   Loaded {len(df)} interactions with ratings")
    return df


def prepare_dataset(df):
    """
    Convert to (text_pair, label) where label is binary: 
    preferred (1) if rating >= threshold else 0.
    
    Returns dataframe with columns: interaction_id, text, label
    """
    print(f"\n🔧 Preparing dataset...")
    df2 = df.copy()
    df2 = df2.dropna(subset=["user_prompt", "ai_response", "rating"])
    
    # Form a combined text for embedding: prompt + separator + candidate
    df2["text"] = df2["user_prompt"].astype(str) + " </s> " + df2["ai_response"].astype(str)
    
    # Create binary label: 1 if rating >= threshold, 0 otherwise
    df2["label"] = (df2["rating"].astype(float) >= POSITIVE_RATING_THRESHOLD).astype(int)
    
    # Show class distribution
    positive_count = df2["label"].sum()
    negative_count = len(df2) - positive_count
    positive_frac = df2["label"].mean()
    
    print(f"   Total samples: {len(df2)}")
    print(f"   Positive samples (rating >= {POSITIVE_RATING_THRESHOLD}): {positive_count} ({positive_frac:.1%})")
    print(f"   Negative samples (rating < {POSITIVE_RATING_THRESHOLD}): {negative_count} ({1-positive_frac:.1%})")
    
    return df2[["interaction_id", "text", "label"]]


def embed_texts(texts, batch_size=64):
    """
    Convert texts to SBERT embeddings.
    
    Args:
        texts: List of text strings to embed
        batch_size: Batch size for encoding (default: 64)
    
    Returns:
        numpy array of embeddings
    """
    print(f"\n🧠 Computing SBERT embeddings...")
    emb = embedder.encode(
        texts, 
        batch_size=batch_size, 
        show_progress_bar=True, 
        convert_to_numpy=True
    )
    print(f"   Embedding shape: {emb.shape}")
    return emb


def train_model(X, y):
    """
    Train logistic regression model with balanced class weights.
    
    Args:
        X: Feature matrix (embeddings)
        y: Labels (0 or 1)
    
    Returns:
        Trained sklearn LogisticRegression model
    """
    print(f"\n🎯 Training logistic regression model...")
    
    # For small datasets, use less regularization (higher C value)
    # C=100 means very weak regularization, allowing model to fit better
    clf = LogisticRegression(
        C=100.0,  # Reduced regularization for small datasets
        max_iter=2000, 
        class_weight="balanced",
        random_state=42,
        solver='lbfgs'
    )
    clf.fit(X, y)
    
    # Check if model learned anything
    coef_norm = np.linalg.norm(clf.coef_)
    print(f"   Training completed!")
    print(f"   Coefficient norm: {coef_norm:.6f}")
    print(f"   Intercept: {clf.intercept_[0]:.6f}")
    
    if coef_norm < 0.001:
        print(f"   ⚠️  Warning: Model coefficients are very small - may need more diverse data")
    
    return clf


def evaluate_model(clf, X_test, y_test):
    """
    Evaluate model performance on test set.
    
    Args:
        clf: Trained classifier
        X_test: Test feature matrix
        y_test: Test labels
    
    Returns:
        dict: Evaluation metrics
    """
    print(f"\n📊 Evaluating model...")
    
    # Predictions
    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_test)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_score)
    except Exception:
        auc = float("nan")
    
    print(f"   Accuracy: {acc:.4f}")
    print(f"   AUC-ROC: {auc:.4f}")
    
    print(f"\n   Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Low Quality', 'High Quality']))
    
    return {
        'accuracy': acc,
        'auc': auc
    }


def main(limit=None):
    """
    Main training pipeline.
    
    Args:
        limit: Optional limit on number of samples to load (for testing)
    """
    print(f"\n{'='*80}")
    print("STARTING TRAINING PIPELINE")
    print(f"{'='*80}")
    
    # 1. Load feedback
    df = load_feedback(limit=limit)
    if df.shape[0] < MIN_SAMPLES_TO_TRAIN:
        print(f"\n❌ ERROR: Not enough labeled samples to train.")
        print(f"   Found: {df.shape[0]}, Need: >= {MIN_SAMPLES_TO_TRAIN}")
        print(f"   Please collect more SME ratings in the RLHF admin interface.")
        raise SystemExit(1)
    
    # 2. Prepare dataset
    dfp = prepare_dataset(df)
    total_samples = dfp.shape[0]
    
    # 3. Embed texts
    texts = dfp["text"].tolist()
    X = embed_texts(texts)
    y = dfp["label"].values
    
    # 4. Train/test split
    print(f"\n✂️ Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.12, 
        random_state=42, 
        stratify=y
    )
    print(f"   Train samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # 5. Train model
    clf = train_model(X_train, y_train)
    
    # 6. Evaluate
    metrics = evaluate_model(clf, X_test, y_test)
    
    # 7. Save model
    print(f"\n💾 Saving model...")
    saved_path = save_reward_model(clf, OUT_MODEL_PATH)
    
    # 8. Log training run
    print(f"\n📝 Logging training run to database...")
    avg_reward = y.mean()  # fraction preferred
    log_training_run(
        model_path=saved_path, 
        total_samples=total_samples, 
        avg_reward=avg_reward, 
        accuracy=metrics['accuracy'], 
        trained_by=1001
    )
    
    print(f"\n{'='*80}")
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")
    print(f"\n📦 Model saved to: {saved_path}")
    print(f"📊 Training samples: {total_samples}")
    print(f"📈 Test accuracy: {metrics['accuracy']:.4f}")
    print(f"📈 Test AUC: {metrics['auc']:.4f}")
    print(f"\n💡 Next steps:")
    print(f"   1. Test the model with: python -c \"from rlhf_reranker import score_text_pair; print(score_text_pair('test', 'answer'))\"")
    print(f"   2. Integrate re-ranking into your Flask inference endpoint")
    print(f"   3. Collect more feedback and retrain periodically to improve performance")
    print(f"\n{'='*80}")


if __name__ == "__main__":
    # Optionally pass a numeric limit via env var TRAIN_LIMIT
    limit = int(os.getenv("TRAIN_LIMIT", "0")) or None
    
    try:
        main(limit=limit)
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ ERROR: Training failed with exception:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise SystemExit(1)
