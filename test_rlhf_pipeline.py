# test_rlhf_pipeline.py
"""
Test script for the complete RLHF pipeline (Step 1: SBERT + Logistic Regression).

This script tests:
1. Database connectivity and table existence
2. Model utilities (save/load)
3. Training pipeline (with synthetic data if needed)
4. Inference and re-ranking functionality

Usage:
    python test_rlhf_pipeline.py
"""

import os
import sys
import tempfile
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Test colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'


def print_test(message):
    print(f"{Colors.BLUE}🧪 TEST: {message}{Colors.RESET}")


def print_success(message):
    print(f"{Colors.GREEN}✅ {message}{Colors.RESET}")


def print_error(message):
    print(f"{Colors.RED}❌ {message}{Colors.RESET}")


def print_warning(message):
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.RESET}")


def test_database_connection():
    """Test 1: Database connectivity and table existence"""
    print_test("Testing database connection and tables...")
    
    try:
        from model_utils import check_database_tables, engine
        
        # Test connection
        with engine.connect() as conn:
            result = conn.execute("SELECT 1")
            result.fetchone()
        
        print_success("Database connection successful")
        
        # Check tables
        tables = check_database_tables()
        
        if tables['rlhf_interactions']:
            print_success("rlhf_interactions table exists")
        else:
            print_error("rlhf_interactions table NOT found")
            return False
        
        if tables['rlhf_reward_model_training']:
            print_success("rlhf_reward_model_training table exists")
        else:
            print_warning("rlhf_reward_model_training table NOT found (optional)")
        
        return True
        
    except Exception as e:
        print_error(f"Database test failed: {e}")
        return False


def test_model_utils():
    """Test 2: Model utilities (save/load)"""
    print_test("Testing model utilities...")
    
    try:
        from model_utils import save_reward_model
        import joblib
        from sklearn.linear_model import LogisticRegression
        import numpy as np
        
        # Create a dummy model
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        dummy_model = LogisticRegression()
        dummy_model.fit(X, y)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as tmp:
            tmp_path = tmp.name
        
        saved_path = save_reward_model(dummy_model, tmp_path)
        
        # Check file exists
        if os.path.exists(saved_path):
            print_success("Model saved successfully")
        else:
            print_error("Model file not created")
            return False
        
        # Load and verify
        loaded_model = joblib.load(saved_path)
        
        # Test prediction
        test_pred_original = dummy_model.predict(X[:5])
        test_pred_loaded = loaded_model.predict(X[:5])
        
        if (test_pred_original == test_pred_loaded).all():
            print_success("Model loaded and verified successfully")
        else:
            print_error("Loaded model produces different predictions")
            return False
        
        # Cleanup
        os.unlink(tmp_path)
        
        return True
        
    except Exception as e:
        print_error(f"Model utilities test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_embeddings():
    """Test 3: SBERT embeddings"""
    print_test("Testing SBERT embeddings...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        model_name = os.getenv("EMB_MODEL", "all-MiniLM-L6-v2")
        print(f"   Loading model: {model_name}")
        
        embedder = SentenceTransformer(model_name)
        
        # Test embedding
        test_texts = [
            "What are the symptoms of diabetes?",
            "How to treat hypertension?",
            "The weather is nice today."
        ]
        
        embeddings = embedder.encode(test_texts, convert_to_numpy=True)
        
        if embeddings.shape[0] == len(test_texts):
            print_success(f"Embeddings generated successfully: shape {embeddings.shape}")
        else:
            print_error("Embedding shape mismatch")
            return False
        
        # Check embedding similarity
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        sim = cosine_similarity(embeddings)
        
        # Medical questions should be more similar to each other than to weather
        if sim[0, 1] > sim[0, 2]:
            print_success("Embedding similarity check passed (medical queries more similar)")
        else:
            print_warning("Embedding similarity unexpected (may not indicate error)")
        
        return True
        
    except Exception as e:
        print_error(f"Embeddings test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reranker_without_model():
    """Test 4: Reranker functionality (without trained model)"""
    print_test("Testing reranker (no model)...")
    
    try:
        from rlhf_reranker import rerank_candidates, is_model_ready, get_model_info
        
        # Check model status
        info = get_model_info()
        print(f"   Model ready: {info['is_ready']}")
        print(f"   Model exists: {info['model_exists']}")
        
        # Test re-ranking (should work even without model, just return original order)
        test_prompt = "What are the symptoms of diabetes?"
        test_candidates = [
            {"text": "Answer 1", "source": "db"},
            {"text": "Answer 2", "source": "wiki"},
            {"text": "Answer 3", "source": "arxiv"}
        ]
        
        ranked = rerank_candidates(test_prompt, test_candidates)
        
        if len(ranked) == len(test_candidates):
            print_success(f"Re-ranking returned {len(ranked)} candidates")
        else:
            print_error("Re-ranking returned wrong number of candidates")
            return False
        
        # Check structure
        if all('_score' in c for c in ranked):
            print_success("All candidates have _score field")
        else:
            print_error("Missing _score field in some candidates")
            return False
        
        if not is_model_ready():
            print_warning("Model not trained yet - scores will be None")
            print_warning("This is expected. Train model with: python train_reward_sbert.py")
        
        return True
        
    except Exception as e:
        print_error(f"Reranker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_training_data_availability():
    """Check if there's enough data to train the model"""
    print_test("Checking training data availability...")
    
    try:
        from sqlalchemy import create_engine, text
        
        DB_URI = os.getenv(
            "DB_URI",
            f"postgresql+psycopg2://{os.getenv('DB_USER', 'postgres')}:{os.getenv('DB_PASSWORD', 'password')}@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}/{os.getenv('DB_NAME', 'pces_base')}"
        )
        
        engine = create_engine(DB_URI)
        
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT COUNT(*) FROM rlhf_interactions WHERE rating IS NOT NULL"
            ))
            count = result.scalar()
        
        min_samples = int(os.getenv("MIN_SAMPLES_TO_TRAIN", "50"))
        
        print(f"   Rated interactions: {count}")
        print(f"   Minimum required: {min_samples}")
        
        if count >= min_samples:
            print_success(f"Sufficient training data available ({count} samples)")
            return True
        else:
            print_warning(f"Insufficient training data ({count} samples, need {min_samples})")
            print_warning("Collect more SME ratings in the RLHF admin interface before training")
            return False
        
    except Exception as e:
        print_error(f"Training data check failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results"""
    print("\n" + "="*80)
    print("RLHF PIPELINE TEST SUITE - Step 1: SBERT + Logistic Regression")
    print("="*80)
    print()
    
    tests = [
        ("Database Connection", test_database_connection),
        ("Model Utilities", test_model_utils),
        ("SBERT Embeddings", test_embeddings),
        ("Reranker Functionality", test_reranker_without_model),
        ("Training Data Check", check_training_data_availability),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print()
        print("-" * 80)
        try:
            results[test_name] = test_func()
        except Exception as e:
            print_error(f"Test '{test_name}' crashed: {e}")
            results[test_name] = False
        print("-" * 80)
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status:12} {test_name}")
    
    print()
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print_success("All tests passed! ✨")
        print()
        print("Next steps:")
        print("  1. Collect SME ratings via the RLHF admin interface (/admin/rlhf)")
        print("  2. Once you have 50+ rated interactions, train the model:")
        print("     python train_reward_sbert.py")
        print("  3. Test the trained model:")
        print("     python rlhf_reranker.py")
        print("  4. Integrate into your Flask app (see integration example below)")
    else:
        print_error(f"Some tests failed ({total - passed} failures)")
        print()
        print("Please fix the failing tests before proceeding.")
    
    print("="*80)
    
    return passed == total


def show_integration_example():
    """Show example of how to integrate into Flask app"""
    print("\n" + "="*80)
    print("INTEGRATION EXAMPLE")
    print("="*80)
    print("""
# In your Flask inference endpoint (e.g., /data or /data-html):

from rlhf_reranker import rerank_candidates, is_model_ready

# ... after obtaining candidates from your RAG system ...

if is_model_ready():
    # candidates is list of dicts: [{"text": "...", "source": "..."}, ...]
    ranked = rerank_candidates(user_prompt, candidates)
    
    # Pick the best candidate (highest scored)
    best_candidate = ranked[0]
    
    # Or get top-3 for diversity
    top_3 = ranked[:3]
    
    # Return the best answer to the user
    final_answer = best_candidate['text']
else:
    # Fallback to original ordering if model not trained yet
    final_answer = candidates[0]['text']

# Continue with your response formatting...
""")
    print("="*80)


if __name__ == "__main__":
    success = run_all_tests()
    
    if success:
        show_integration_example()
    
    sys.exit(0 if success else 1)
