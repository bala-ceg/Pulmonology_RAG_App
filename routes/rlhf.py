"""
Blueprint: RLHF Admin routes.

Covers /admin/rlhf and all /api/rlhf/* endpoints that belong to the
Reinforcement-Learning-from-Human-Feedback admin section of main.py
(original lines 3658-4680).

Shared state is accessed via current_app.config:
  - LLM_INSTANCE   : ChatOpenAI
  - SCOPE_GUARD    : DomainScopeGuard (may be None)
"""

from __future__ import annotations

import os
import re
import subprocess
import threading
import traceback
from contextlib import contextmanager
from datetime import datetime
from typing import Generator

import psycopg
from flask import Blueprint, current_app, jsonify, render_template, request

from utils.error_handlers import get_logger, handle_route_errors

logger = get_logger(__name__)

rlhf_bp = Blueprint("rlhf_bp", __name__)

# ---------------------------------------------------------------------------
# Optional RLHF reranker
# ---------------------------------------------------------------------------
try:
    import rlhf_reranker as _reranker_mod  # noqa: F401
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    logger.warning("rlhf_reranker not available — score/rerank endpoints will be limited")


# ---------------------------------------------------------------------------
# DB helper — mirrors the _pg_conn() context-manager in main.py
# ---------------------------------------------------------------------------
def _db_config() -> dict:
    """Build psycopg connection kwargs from environment variables."""
    cfg = {
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT"),
        "dbname": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "connect_timeout": int(os.getenv("DB_CONNECT_TIMEOUT", "30")),
    }
    return {k: v for k, v in cfg.items() if v is not None}


@contextmanager
def _pg_conn() -> Generator:
    """Per-request psycopg connection context-manager."""
    with psycopg.connect(**_db_config()) as conn:
        yield conn


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@rlhf_bp.route("/admin/rlhf")
@handle_route_errors
def admin_rlhf():
    """Render the RLHF admin page."""
    return render_template("admin_rlhf.html")


@rlhf_bp.route("/api/rlhf/stats", methods=["GET"])
@handle_route_errors
def get_rlhf_stats():
    """Get statistics about RLHF training data."""
    with _pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM rlhf_interactions")
            total_interactions = cur.fetchone()[0]

            cur.execute(
                "SELECT AVG(rating) FROM rlhf_interactions WHERE rating IS NOT NULL"
            )
            avg_rating_result = cur.fetchone()[0]
            avg_rating = float(avg_rating_result) if avg_rating_result else 0.0

            cur.execute("SELECT COUNT(*) FROM rlhf_sessions")
            total_sessions = cur.fetchone()[0]

            cur.execute(
                "SELECT COUNT(*) FROM rlhf_interactions WHERE bias_flag = TRUE"
            )
            bias_count = cur.fetchone()[0]

    return jsonify(
        {
            "total_interactions": total_interactions,
            "avg_rating": avg_rating,
            "total_sessions": total_sessions,
            "bias_count": bias_count,
        }
    )


@rlhf_bp.route("/api/rlhf/interactions", methods=["GET"])
@handle_route_errors
def get_rlhf_interactions():
    """Get RLHF interactions with optional filters."""
    session_id = request.args.get("session_id", type=int)
    min_rating = request.args.get("min_rating", type=int)
    bias_only = request.args.get("bias_only", "false").lower() == "true"
    limit = request.args.get("limit", type=int, default=50)

    query = """
        SELECT interaction_id, session_id, user_prompt, ai_response,
               rating, feedback_comment, bias_flag, created_dt, updated_dt
        FROM rlhf_interactions
        WHERE 1=1
    """
    params: list = []

    if session_id:
        query += " AND session_id = %s"
        params.append(session_id)

    if min_rating:
        query += " AND rating >= %s"
        params.append(min_rating)

    if bias_only:
        query += " AND bias_flag = TRUE"

    query += " ORDER BY created_dt DESC LIMIT %s"
    params.append(limit)

    with _pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            columns = [desc[0] for desc in cur.description]
            results = cur.fetchall()

    interactions = []
    for row in results:
        interaction = dict(zip(columns, row))
        if interaction.get("created_dt"):
            interaction["created_dt"] = interaction["created_dt"].isoformat()
        if interaction.get("updated_dt"):
            interaction["updated_dt"] = interaction["updated_dt"].isoformat()
        interactions.append(interaction)

    return jsonify(interactions)


@rlhf_bp.route("/api/rlhf/sessions", methods=["GET"])
@handle_route_errors
def get_rlhf_sessions():
    """Get all RLHF training sessions."""
    with _pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT session_id, user_id, model_version, session_start,
                       session_end, status, notes, created_dt
                FROM rlhf_sessions
                ORDER BY session_start DESC
            """)
            columns = [desc[0] for desc in cur.description]
            results = cur.fetchall()

    sessions = []
    for row in results:
        session = dict(zip(columns, row))
        for field in ("session_start", "session_end", "created_dt"):
            if session.get(field):
                session[field] = session[field].isoformat()
        sessions.append(session)

    return jsonify(sessions)


@rlhf_bp.route("/api/rlhf/add_sample", methods=["POST"])
@handle_route_errors
def add_rlhf_sample():
    """Add a new RLHF training sample."""
    data = request.json

    for field in ("session_id", "user_prompt", "ai_response"):
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400

    with _pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO rlhf_interactions
                (session_id, user_prompt, ai_response, rating, feedback_comment,
                 bias_flag, created_by, updated_by, created_dt, updated_dt)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                RETURNING interaction_id
                """,
                (
                    data["session_id"],
                    data["user_prompt"],
                    data["ai_response"],
                    data.get("rating", 3),
                    data.get("feedback_comment", ""),
                    data.get("bias_flag", False),
                    1001,
                    1001,
                ),
            )
            interaction_id = cur.fetchone()[0]
            conn.commit()

    scope_guard = current_app.config.get("SCOPE_GUARD")
    if scope_guard is not None:
        try:
            scope_guard.refresh()
        except Exception as sg_err:
            logger.warning("Scope guard refresh failed (non-critical): %s", sg_err)

    return jsonify(
        {
            "success": True,
            "interaction_id": interaction_id,
            "message": "Training sample added successfully",
        }
    )


@rlhf_bp.route("/api/rlhf/generate_samples", methods=["POST"])
@handle_route_errors
def generate_rlhf_samples():
    """Generate multiple sample RLHF training interactions."""
    samples = [
        {
            "session_id": 1,
            "user_prompt": "What are the current guidelines for managing Type 2 Diabetes?",
            "ai_response": (
                "According to ADA 2024 guidelines, Type 2 Diabetes management includes: "
                "1) Metformin as first-line therapy, 2) HbA1c target <7% for most adults, "
                "3) SGLT2 inhibitors or GLP-1 agonists for cardiovascular benefits, "
                "4) Lifestyle modifications with diet and exercise."
            ),
            "rating": 5,
            "feedback_comment": "Accurate and comprehensive response with latest guidelines",
            "bias_flag": False,
        },
        {
            "session_id": 1,
            "user_prompt": "What is the recommended treatment for community-acquired pneumonia?",
            "ai_response": (
                "For outpatient CAP treatment: 1) Amoxicillin 1g TID or Doxycycline 100mg BID "
                "for 5-7 days, 2) Add macrolide if atypical pathogens suspected, "
                "3) Consider comorbidities and local resistance patterns."
            ),
            "rating": 4,
            "feedback_comment": "Good response but could mention severity assessment criteria",
            "bias_flag": False,
        },
        {
            "session_id": 2,
            "user_prompt": "How do I interpret elevated troponin levels?",
            "ai_response": (
                "Elevated troponin indicates myocardial injury. Consider: 1) Acute MI (most common), "
                "2) Myocarditis, 3) PE, 4) Renal failure, 5) Sepsis. "
                "Must correlate with clinical presentation, ECG changes, and cardiac imaging."
            ),
            "rating": 5,
            "feedback_comment": "Excellent differential diagnosis approach",
            "bias_flag": False,
        },
        {
            "session_id": 2,
            "user_prompt": "What are the indications for starting statin therapy?",
            "ai_response": (
                "ACC/AHA 2023 guidelines recommend statins for: 1) LDL-C ≥190 mg/dL, "
                "2) Diabetes (age 40-75) with LDL 70-189 mg/dL, "
                "3) 10-year ASCVD risk ≥7.5%, 4) Clinical ASCVD."
            ),
            "rating": 5,
            "feedback_comment": "Up-to-date guidelines accurately cited",
            "bias_flag": False,
        },
        {
            "session_id": 3,
            "user_prompt": "What is the first-line treatment for COPD exacerbation?",
            "ai_response": (
                "GOLD 2024 guidelines: 1) Short-acting bronchodilators (SABA/SAMA), "
                "2) Systemic corticosteroids (Prednisone 40mg x 5 days), "
                "3) Antibiotics if bacterial infection suspected (increased sputum purulence)."
            ),
            "rating": 4,
            "feedback_comment": "Correct but could elaborate on antibiotic selection",
            "bias_flag": False,
        },
        {
            "session_id": 3,
            "user_prompt": "When should I order a CT scan for suspected stroke?",
            "ai_response": (
                "Non-contrast CT head should be performed immediately for any suspected stroke to: "
                "1) Rule out hemorrhage, 2) Exclude stroke mimics, "
                "3) Determine eligibility for thrombolysis. "
                "CT angiography if large vessel occlusion suspected."
            ),
            "rating": 5,
            "feedback_comment": "Critical time-sensitive information accurately provided",
            "bias_flag": False,
        },
        {
            "session_id": 4,
            "user_prompt": "What are the criteria for diagnosing rheumatoid arthritis?",
            "ai_response": (
                "ACR/EULAR 2010 criteria require ≥6 points from: "
                "1) Joint involvement (0-5 points), 2) Serology (RF/anti-CCP, 0-3 points), "
                "3) Acute phase reactants (ESR/CRP, 0-1 point), 4) Duration ≥6 weeks (1 point)."
            ),
            "rating": 5,
            "feedback_comment": "Comprehensive diagnostic criteria well explained",
            "bias_flag": False,
        },
        {
            "session_id": 4,
            "user_prompt": "How do you manage acute asthma exacerbation?",
            "ai_response": (
                "GINA 2024 approach: 1) High-dose SABA via spacer/nebulizer, "
                "2) Oxygen to maintain SpO2 93-95%, 3) Systemic corticosteroids early, "
                "4) Consider ipratropium bromide, 5) Assess severity and response continuously."
            ),
            "rating": 4,
            "feedback_comment": "Good management outline, could add severity classification",
            "bias_flag": False,
        },
        {
            "session_id": 5,
            "user_prompt": "What are the red flags for low back pain?",
            "ai_response": (
                "Red flags requiring urgent evaluation: 1) Age <20 or >55, "
                "2) History of cancer, 3) Constitutional symptoms, 4) Trauma, "
                "5) Neurological deficits, 6) Saddle anesthesia, 7) Bowel/bladder dysfunction."
            ),
            "rating": 5,
            "feedback_comment": "Essential red flags comprehensively listed",
            "bias_flag": False,
        },
        {
            "session_id": 5,
            "user_prompt": "What immunizations does an adult with diabetes need?",
            "ai_response": (
                "CDC recommendations for adults with diabetes: 1) Annual influenza vaccine, "
                "2) Pneumococcal vaccines (PCV15/PCV20 or PPSV23), 3) Hepatitis B series, "
                "4) Tdap/Td boosters, 5) COVID-19 vaccines, 6) Shingles vaccine (age ≥50)."
            ),
            "rating": 5,
            "feedback_comment": "Complete immunization schedule for diabetic patients",
            "bias_flag": False,
        },
    ]

    inserted_count = 0
    with _pg_conn() as conn:
        with conn.cursor() as cur:
            for sample in samples:
                cur.execute(
                    """
                    INSERT INTO rlhf_interactions
                    (session_id, user_prompt, ai_response, rating, feedback_comment,
                     bias_flag, created_by, updated_by, created_dt, updated_dt)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    """,
                    (
                        sample["session_id"],
                        sample["user_prompt"],
                        sample["ai_response"],
                        sample["rating"],
                        sample["feedback_comment"],
                        sample["bias_flag"],
                        1001,
                        1001,
                    ),
                )
                inserted_count += 1
            conn.commit()

    scope_guard = current_app.config.get("SCOPE_GUARD")
    if scope_guard is not None:
        try:
            scope_guard.refresh()
        except Exception as sg_err:
            logger.warning("Scope guard refresh failed (non-critical): %s", sg_err)

    return jsonify(
        {
            "success": True,
            "count": inserted_count,
            "message": f"Successfully generated {inserted_count} training samples",
        }
    )


@rlhf_bp.route("/api/rlhf/update_rating", methods=["POST"])
@handle_route_errors
def update_rlhf_rating():
    """Update rating for an existing interaction."""
    data = request.json

    if "interaction_id" not in data or "rating" not in data:
        return jsonify({"error": "Missing interaction_id or rating"}), 400

    with _pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE rlhf_interactions
                SET rating = %s,
                    feedback_comment = %s,
                    bias_flag = %s,
                    updated_by = %s,
                    updated_dt = CURRENT_TIMESTAMP
                WHERE interaction_id = %s
                """,
                (
                    data["rating"],
                    data.get("feedback_comment", ""),
                    data.get("bias_flag", False),
                    1001,
                    data["interaction_id"],
                ),
            )
            conn.commit()

    return jsonify({"success": True, "message": "Rating updated successfully"})


@rlhf_bp.route("/api/rlhf/update_interaction", methods=["POST"])
@handle_route_errors
def update_rlhf_interaction():
    """Update an entire RLHF interaction (prompt, response, rating, feedback, bias)."""
    data = request.json

    if "interaction_id" not in data:
        return jsonify({"error": "Missing interaction_id"}), 400

    with _pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE rlhf_interactions
                SET user_prompt = %s,
                    ai_response = %s,
                    rating = %s,
                    feedback_comment = %s,
                    bias_flag = %s,
                    updated_by = %s,
                    updated_dt = CURRENT_TIMESTAMP
                WHERE interaction_id = %s
                """,
                (
                    data.get("user_prompt"),
                    data.get("ai_response"),
                    data.get("rating", 3),
                    data.get("feedback_comment", ""),
                    data.get("bias_flag", False),
                    1001,
                    data["interaction_id"],
                ),
            )
            conn.commit()

    return jsonify({"success": True, "message": "Interaction updated successfully"})


@rlhf_bp.route("/api/rlhf/delete_interaction", methods=["POST"])
@handle_route_errors
def delete_rlhf_interaction():
    """Delete an RLHF interaction."""
    data = request.json

    if "interaction_id" not in data:
        return jsonify({"error": "Missing interaction_id"}), 400

    with _pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM rlhf_interactions WHERE interaction_id = %s",
                (data["interaction_id"],),
            )
            conn.commit()

    return jsonify({"success": True, "message": "Interaction deleted successfully"})


@rlhf_bp.route("/api/rlhf/update_session", methods=["POST"])
@handle_route_errors
def update_rlhf_session():
    """Update an RLHF session."""
    data = request.json

    if "session_id" not in data:
        return jsonify({"error": "Missing session_id"}), 400

    with _pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE rlhf_sessions
                SET user_id = %s,
                    model_version = %s,
                    status = %s,
                    notes = %s,
                    updated_by = %s,
                    updated_dt = CURRENT_TIMESTAMP
                WHERE session_id = %s
                """,
                (
                    data.get("user_id"),
                    data.get("model_version"),
                    data.get("status"),
                    data.get("notes", ""),
                    1001,
                    data["session_id"],
                ),
            )
            conn.commit()

    return jsonify({"success": True, "message": "Session updated successfully"})


@rlhf_bp.route("/api/rlhf/train_model", methods=["POST"])
@handle_route_errors
def train_rlhf_model():
    """Train the RLHF reward model using current feedback data."""
    with _pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM rlhf_interactions WHERE rating IS NOT NULL"
            )
            rated_count = cur.fetchone()[0]

    min_samples = int(os.getenv("MIN_SAMPLES_TO_TRAIN", "20"))

    if rated_count < min_samples:
        return jsonify(
            {
                "success": False,
                "error": (
                    f"Insufficient training data. Found {rated_count} rated samples, "
                    f"need at least {min_samples}."
                ),
                "rated_count": rated_count,
                "min_required": min_samples,
            }
        ), 400

    training_output: dict = {"status": "running", "output": "", "error": None}

    def run_training() -> None:
        try:
            result = subprocess.run(
                ["python", "train_reward_sbert.py"],
                cwd=os.path.dirname(os.path.abspath(__file__ + "/../")),
                capture_output=True,
                text=True,
                timeout=300,
            )
            training_output["status"] = (
                "completed" if result.returncode == 0 else "failed"
            )
            training_output["output"] = result.stdout
            training_output["error"] = (
                result.stderr if result.returncode != 0 else None
            )
            training_output["return_code"] = result.returncode
        except subprocess.TimeoutExpired:
            training_output["status"] = "timeout"
            training_output["error"] = "Training exceeded 5 minute timeout"
        except Exception as exc:
            training_output["status"] = "error"
            training_output["error"] = str(exc)

    thread = threading.Thread(target=run_training)
    thread.start()
    thread.join(timeout=120)

    if thread.is_alive():
        return jsonify(
            {
                "success": False,
                "error": "Training is taking longer than expected. Please check server logs.",
                "status": "timeout",
            }
        ), 408

    if training_output["status"] == "completed":
        output_lines = training_output["output"]
        accuracy = None
        auc = None

        for line in output_lines.split("\n"):
            if "Accuracy:" in line:
                try:
                    accuracy = float(line.split("Accuracy:")[1].strip())
                except Exception:
                    pass
            if "AUC-ROC:" in line or "AUC:" in line:
                try:
                    auc = float(line.split(":")[1].strip())
                except Exception:
                    pass

        return jsonify(
            {
                "success": True,
                "message": "Model trained successfully!",
                "metrics": {
                    "total_samples": rated_count,
                    "accuracy": accuracy,
                    "auc": auc,
                    "trained_at": datetime.now().isoformat(),
                },
                "output": (
                    output_lines[-500:]
                    if len(output_lines) > 500
                    else output_lines
                ),
            }
        )

    return jsonify(
        {
            "success": False,
            "error": training_output.get("error", "Training failed"),
            "status": training_output["status"],
            "output": training_output.get("output", ""),
        }
    ), 500


@rlhf_bp.route("/api/rlhf/training_status", methods=["GET"])
@handle_route_errors
def get_training_status():
    """Get current training model status and statistics."""
    model_path = os.getenv("REWARD_MODEL_PATH", "reward_model.joblib")
    model_exists = os.path.exists(model_path)

    model_info = None
    if model_exists:
        stat = os.stat(model_path)
        model_info = {
            "exists": True,
            "path": model_path,
            "size_kb": round(stat.st_size / 1024, 2),
            "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        }

    with _pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM rlhf_interactions WHERE rating IS NOT NULL"
            )
            rated_count = cur.fetchone()[0]

            training_history = None
            try:
                cur.execute(
                    """
                    SELECT model_version, total_interactions, accuracy, avg_reward, created_dt
                    FROM rlhf_reward_model_training
                    ORDER BY created_dt DESC
                    LIMIT 1
                    """
                )
                latest = cur.fetchone()
                if latest:
                    training_history = {
                        "model_version": latest[0],
                        "total_interactions": latest[1],
                        "accuracy": latest[2],
                        "avg_reward": latest[3],
                        "trained_at": (
                            latest[4].isoformat() if latest[4] else None
                        ),
                    }
            except Exception as exc:
                logger.warning("Error fetching training history: %s", exc)

    min_samples = int(os.getenv("MIN_SAMPLES_TO_TRAIN", "20"))

    return jsonify(
        {
            "success": True,
            "model": model_info,
            "training_data": {
                "rated_count": rated_count,
                "min_required": min_samples,
                "ready_to_train": rated_count >= min_samples,
            },
            "latest_training": training_history,
        }
    )


@rlhf_bp.route("/api/rlhf/model_info", methods=["GET"])
@handle_route_errors
def get_model_info():
    """Get information about the RLHF reward model."""
    try:
        from rlhf_reranker import get_model_info as _get_info, is_model_ready

        model_ready = is_model_ready()
        info = _get_info()
        return jsonify({"success": True, "model_ready": model_ready, **info})
    except Exception as exc:
        return jsonify(
            {
                "success": False,
                "model_ready": False,
                "error": str(exc),
                "message": "Model not available. Please train the model first.",
            }
        )


@rlhf_bp.route("/api/rlhf/scope_guard", methods=["GET", "POST"])
@handle_route_errors
def manage_scope_guard():
    """
    Manage the RLHF Domain Scope Guard.

    GET  — Return current guard status.
    POST — Actions: ``refresh`` or ``set_threshold``.
    """
    scope_guard = current_app.config.get("SCOPE_GUARD")

    if request.method == "GET":
        if scope_guard is None:
            return jsonify(
                {
                    "success": True,
                    "available": False,
                    "message": "Domain Scope Guard is not initialized.",
                }
            )
        return jsonify(
            {"success": True, "available": True, **scope_guard.get_status()}
        )

    # POST
    data = request.json or {}
    action = data.get("action", "")

    if scope_guard is None:
        return jsonify(
            {"success": False, "error": "Domain Scope Guard is not initialized."}
        ), 503

    if action == "refresh":
        n = scope_guard.refresh()
        return jsonify(
            {
                "success": True,
                "message": f"Corpus refreshed — {n} training prompts loaded.",
                **scope_guard.get_status(),
            }
        )

    if action == "set_threshold":
        threshold = data.get("threshold")
        if threshold is None:
            return jsonify({"error": "Missing 'threshold' field"}), 400
        try:
            scope_guard.set_threshold(float(threshold))
        except ValueError as ve:
            return jsonify({"error": str(ve)}), 400
        return jsonify(
            {
                "success": True,
                "message": f"Threshold updated to {scope_guard.threshold}",
                **scope_guard.get_status(),
            }
        )

    return jsonify(
        {
            "error": (
                f"Unknown action: {action!r}. Use 'refresh' or 'set_threshold'."
            )
        }
    ), 400


@rlhf_bp.route("/api/rlhf/score", methods=["POST"])
@handle_route_errors
def score_answer():
    """Score a single prompt-answer pair using the RLHF reward model."""
    data = request.json
    prompt = data.get("prompt", "")
    answer = data.get("answer", "")

    if not prompt or not answer:
        return jsonify(
            {"success": False, "error": "Both 'prompt' and 'answer' are required"}
        ), 400

    from rlhf_reranker import is_model_ready, score_text_pair  # noqa: PLC0415

    if not is_model_ready():
        return jsonify(
            {
                "success": False,
                "error": "Model not available. Please train the model first.",
            }
        ), 503

    score = score_text_pair(prompt, answer)
    return jsonify(
        {"success": True, "score": float(score), "prompt": prompt, "answer": answer}
    )


@rlhf_bp.route("/api/rlhf/generate_candidates", methods=["POST"])
@handle_route_errors
def generate_candidates():
    """Generate multiple candidate answers for a given question using the LLM."""
    data = request.json
    prompt = data.get("prompt", "").strip()

    if not prompt:
        return jsonify({"success": False, "error": "'prompt' is required"}), 400

    from langchain_core.messages import HumanMessage, SystemMessage  # noqa: PLC0415

    llm = current_app.config.get("LLM_INSTANCE")
    if llm is None:
        return jsonify({"success": False, "error": "LLM not available"}), 503

    simple_prompt = (
        f"Question: {prompt}\n\n"
        "Generate exactly 3 answers of varying quality:\n\n"
        "Answer 1:\n"
        "[Write a detailed, comprehensive answer with specific medications, dosages, and treatment guidelines]\n\n"
        "Answer 2:\n"
        "[Write a good but less detailed answer covering main points]\n\n"
        "Answer 3:\n"
        "[Write a very brief, vague answer]"
    )

    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a medical expert writing clinical answer variations for evaluation. "
                    "IMPORTANT: Write ONLY in plain prose. Do NOT include Python code, code blocks, "
                    "programming exercises, markdown headers, or tutorial-style content. "
                    "Every answer must be a natural clinical response a doctor would give to a patient."
                )
            ),
            HumanMessage(content=simple_prompt),
        ]
    )

    response_text = response.content.strip()
    candidates: list[str] = []

    answer_pattern = r"Answer\s+(\d+)[:\s\(]"
    matches = list(re.finditer(answer_pattern, response_text, re.IGNORECASE))

    if len(matches) >= 2:
        for i in range(len(matches)):
            start_pos = matches[i].end()
            end_pos = (
                matches[i + 1].start() if i + 1 < len(matches) else len(response_text)
            )
            answer_text = response_text[start_pos:end_pos].strip()
            answer_text = re.sub(r"^\[.*?\]\s*", "", answer_text)
            answer_text = answer_text.split("\n\n")[0].strip()
            if answer_text and len(answer_text) > 10:
                candidates.append(answer_text)

    if len(candidates) < 3 and "|||" in response_text:
        candidates = [a.strip() for a in response_text.split("|||") if a.strip()]

    if len(candidates) < 3:
        paragraphs = [
            p.strip()
            for p in response_text.split("\n\n")
            if p.strip() and len(p.strip()) > 20
        ]
        filtered: list[str] = []
        for p in paragraphs:
            if not re.match(
                r"^(Answer\s+\d+|HIGH|MEDIUM|LOW|Question:)", p, re.IGNORECASE
            ):
                p = re.sub(
                    r"^(Answer\s+\d+[:\s\(].*?[\):]?\s*)", "", p, flags=re.IGNORECASE
                )
                if p.strip() and len(p.strip()) > 10:
                    filtered.append(p.strip())
        candidates = filtered[:3]

    base = prompt.lower().replace("what", "").replace("?", "").strip()
    if len(candidates) < 1:
        candidates.append(
            f"Comprehensive treatment for {base} includes multiple evidence-based approaches."
        )
    if len(candidates) < 2:
        candidates.append("Treatment typically involves standard medications and therapies.")
    if len(candidates) < 3:
        candidates.append("Use medications.")

    cleaned: list[str] = []
    for candidate in candidates[:3]:
        candidate = re.sub(r"```[\s\S]*?```", "", candidate)
        candidate = re.sub(r"`[^`]+`", "", candidate)
        candidate = re.sub(
            r"\n*(#+\s*)?(Exercise|Ideas|Solution|Example|import|def |print\()[^\n]*(\n[^\n]+)*",
            "",
            candidate,
            flags=re.IGNORECASE,
        )
        candidate = re.sub(r"^\*\*.*?\*\*\s*", "", candidate)
        candidate = re.sub(r"^#+\s+", "", candidate)
        candidate = re.sub(r"\n{3,}", "\n\n", candidate)
        candidate = candidate.strip()
        if candidate:
            cleaned.append(candidate)

    return jsonify({"success": True, "prompt": prompt, "candidates": cleaned[:3]})


@rlhf_bp.route("/api/rlhf/rerank", methods=["POST"])
@handle_route_errors
def rerank_answers():
    """Re-rank multiple candidate answers using the RLHF reward model."""
    data = request.json
    prompt = data.get("prompt", "")
    candidates = data.get("candidates", [])

    if not prompt:
        return jsonify({"success": False, "error": "'prompt' is required"}), 400

    if not candidates or len(candidates) < 2:
        return jsonify(
            {"success": False, "error": "At least 2 candidate answers are required"}
        ), 400

    from rlhf_reranker import is_model_ready, rerank_candidates  # noqa: PLC0415

    if not is_model_ready():
        return jsonify(
            {
                "success": False,
                "error": "Model not available. Please train the model first.",
            }
        ), 503

    ranked_raw = rerank_candidates(prompt, candidates)

    ranked = [
        {"answer": item.get("text", ""), "score": item.get("_score", 0.0), "rank": idx}
        for idx, item in enumerate(ranked_raw, 1)
    ]

    return jsonify({"success": True, "prompt": prompt, "ranked": ranked})
