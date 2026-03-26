# rlhf_reranker.py
"""
Inference-time re-ranker using the trained SBERT reward model.
Scores candidate answers and ranks them based on predicted quality.

Usage in Flask app:
    from rlhf_reranker import rerank_candidates
    
    candidates = [
        {"text": "Answer 1", "source": "vector_db"},
        {"text": "Answer 2", "source": "wikipedia"},
        {"text": "Answer 3", "source": "arxiv"}
    ]
    
    ranked = rerank_candidates(user_prompt, candidates)
    best_answer = ranked[0]  # Highest scored answer
"""

import os
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# CONFIG
EMB_MODEL = os.getenv("EMB_MODEL", "all-MiniLM-L6-v2")
REWARD_MODEL_PATH = os.getenv("REWARD_MODEL_PATH", "reward_model.joblib")

# Initialize embedder
print(f"🔧 Initializing RLHF reranker...")
print(f"   Embedding model: {EMB_MODEL}")
print(f"   Reward model: {REWARD_MODEL_PATH}")

embedder = SentenceTransformer(EMB_MODEL)

# Load reward model if available
reward_model = None
if os.path.exists(REWARD_MODEL_PATH):
    try:
        reward_model = joblib.load(REWARD_MODEL_PATH)
        print(f"✅ Reward model loaded successfully")
    except Exception as e:
        print(f"⚠️ Failed to load reward model: {e}")
        print(f"   Re-ranking will not be available until model is trained")
else:
    print(f"⚠️ Reward model not found at {REWARD_MODEL_PATH}")
    print(f"   Please train the model first: python train_reward_sbert.py")


def score_text_pair(prompt: str, candidate: str):
    """
    Score a (prompt, candidate answer) pair using the reward model.
    Returns a score in [0,1] if reward_model is available, else None.
    
    Args:
        prompt: User's query/prompt
        candidate: Candidate answer to score
    
    Returns:
        float: Predicted quality score (0-1), or None if model not available
    """
    if reward_model is None:
        return None
    
    # Create text pair for embedding (same format as training)
    text = prompt + " </s> " + candidate
    
    # Embed the text
    emb = embedder.encode([text], convert_to_numpy=True)
    
    # Get prediction from model
    if hasattr(reward_model, "predict_proba"):
        # Probability of being high quality (class 1)
        score = float(reward_model.predict_proba(emb)[:, 1].item())
    else:
        # Decision function score (logistic regression)
        score = float(reward_model.decision_function(emb).item())
        # Normalize to [0, 1] range using sigmoid
        score = 1 / (1 + np.exp(-score))
    
    return score


def rerank_candidates(prompt: str, candidates: list):
    """
    Re-rank a list of candidate answers based on predicted quality.
    
    Args:
        prompt: User's query/prompt
        candidates: List of candidates - can be:
                   - List of dicts with 'text' key: [{"text": "...", "source": "..."}]
                   - List of strings: ["answer 1", "answer 2"]
    
    Returns:
        List of dicts with added '_score' field, sorted descending by score
        (highest quality first)
    
    Example:
        >>> candidates = [
        ...     {"text": "Answer 1", "source": "db"},
        ...     {"text": "Answer 2", "source": "wiki"}
        ... ]
        >>> ranked = rerank_candidates("What is diabetes?", candidates)
        >>> print(ranked[0])  # Best answer
        {'text': 'Answer 2', 'source': 'wiki', '_score': 0.87}
    """
    scored = []
    
    for c in candidates:
        # Extract text from candidate (handle both dict and string formats)
        if isinstance(c, dict):
            text = c.get("text", "")
            source = c.get("source")
        else:
            text = str(c)
            source = None
        
        # Score the candidate
        score = score_text_pair(prompt, text)
        
        # Build result dict
        result = {
            "text": text,
            "source": source,
            "_score": score
        }
        scored.append(result)
    
    # If no model available (all scores are None), preserve original order
    if scored and scored[0]["_score"] is None:
        print("⚠️ Reward model not available - returning candidates in original order")
        return scored
    
    # Sort by score (highest first)
    scored.sort(key=lambda x: (x["_score"] if x["_score"] is not None else -1), reverse=True)
    
    return scored


def get_top_k(prompt: str, candidates: list, k: int = 1):
    """
    Convenience function to get top-k candidates after re-ranking.
    
    Args:
        prompt: User's query/prompt
        candidates: List of candidate answers
        k: Number of top candidates to return (default: 1)
    
    Returns:
        List of top-k candidates (sorted by quality)
    """
    ranked = rerank_candidates(prompt, candidates)
    return ranked[:k]


def is_model_ready():
    """
    Check if the reward model is loaded and ready for use.
    
    Returns:
        bool: True if model is available, False otherwise
    """
    return reward_model is not None


def get_model_info():
    """
    Get information about the loaded reward model.
    
    Returns:
        dict: Model information including path, status, and metadata
    """
    info = {
        "model_path": REWARD_MODEL_PATH,
        "embedding_model": EMB_MODEL,
        "is_ready": is_model_ready(),
        "model_exists": os.path.exists(REWARD_MODEL_PATH)
    }
    
    if is_model_ready():
        # Get model type and parameters
        info["model_type"] = type(reward_model).__name__
        if hasattr(reward_model, "get_params"):
            info["model_params"] = reward_model.get_params()
    
    return info


# Test function for standalone usage
if __name__ == "__main__":
    print("\n" + "="*80)
    print("RLHF RERANKER - TEST MODE")
    print("="*80)
    
    # Show model info
    info = get_model_info()
    print(f"\n📊 Model Information:")
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    if not is_model_ready():
        print(f"\n❌ Reward model not ready. Please train it first:")
        print(f"   python train_reward_sbert.py")
        exit(1)
    
    # Test re-ranking
    print(f"\n🧪 Testing re-ranking...")
    
    test_prompt = "What are the symptoms of diabetes?"
    test_candidates = [
        {"text": "Diabetes symptoms include increased thirst, frequent urination, and fatigue.", "source": "medical_db"},
        {"text": "The weather is nice today.", "source": "irrelevant"},
        {"text": "Common signs of diabetes are excessive hunger, blurred vision, and slow healing of wounds.", "source": "clinical_kb"}
    ]
    
    print(f"\n📝 Prompt: '{test_prompt}'")
    print(f"\n📋 Original candidates:")
    for i, c in enumerate(test_candidates, 1):
        print(f"   {i}. [{c['source']}] {c['text'][:60]}...")
    
    # Re-rank
    ranked = rerank_candidates(test_prompt, test_candidates)
    
    print(f"\n🏆 Re-ranked candidates:")
    for i, c in enumerate(ranked, 1):
        score = c['_score']
        score_str = f"{score:.4f}" if score is not None else "N/A"
        print(f"   {i}. [Score: {score_str}] [{c['source']}] {c['text'][:60]}...")
    
    print(f"\n✅ Re-ranking test completed successfully!")
    print(f"\n💡 Integration example:")
    print(f"""
    from rlhf_reranker import rerank_candidates
    
    # In your Flask inference endpoint:
    candidates = [
        {{"text": answer1, "source": "vector_db"}},
        {{"text": answer2, "source": "wikipedia"}}
    ]
    
    ranked = rerank_candidates(user_prompt, candidates)
    best_answer = ranked[0]['text']  # Highest quality answer
    """)
    print("="*80)
