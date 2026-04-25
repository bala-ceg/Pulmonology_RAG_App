"""
Blueprint: SFT (Supervised Fine-Tuning) Experiment routes.

Covers /api/rlhf/experiments, /api/rlhf/experiment/*, /api/rlhf/ranked-data/*,
/api/rlhf/departments, and /api/rlhf/ranked-data/by-department endpoints
(original main.py lines 4757-4985).

Delegates to sft_experiment_manager for all data operations.
Shared state: SCOPE_GUARD via current_app.config.
"""

from __future__ import annotations

from datetime import datetime

from flask import Blueprint, Response, current_app, jsonify, request

from utils.error_handlers import get_logger, handle_route_errors

logger = get_logger(__name__)

sft_bp = Blueprint("sft_bp", __name__)

# ---------------------------------------------------------------------------
# Optional SFT experiment manager
# ---------------------------------------------------------------------------
try:
    from sft_experiment_manager import (
        DEPARTMENTS,
        add_ranked_entry,
        delete_experiment,
        delete_ranked_entry,
        delete_ranked_group,
        export_to_jsonl,
        get_department_list,
        get_experiment,
        get_prompts_by_department,
        get_ranked_data,
        get_ranked_data_stats,
        get_training_status as sft_get_training_status,
        import_from_jsonl,
        list_experiments,
        start_experiment,
        test_trained_model,
        update_experiment_samples,
        update_ranked_entry,
    )

    SFT_AVAILABLE = True
    logger.info("SFT Experiment Manager loaded in sft blueprint")
except Exception as _sft_err:
    SFT_AVAILABLE = False
    logger.warning("SFT Experiment Manager not available: %s", _sft_err)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@sft_bp.route("/api/rlhf/experiments", methods=["GET"])
@handle_route_errors
def api_list_experiments():
    """List all SFT experiments."""
    if not SFT_AVAILABLE:
        return jsonify({"success": False, "error": "SFT module not available"}), 503
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)
    return jsonify(list_experiments(page=page, per_page=per_page))


@sft_bp.route("/api/rlhf/experiment/create", methods=["POST"])
@handle_route_errors
def api_create_experiment():
    """Create and start a new SFT training experiment."""
    if not SFT_AVAILABLE:
        return jsonify({"success": False, "error": "SFT module not available"}), 503

    data = request.json or {}
    department = data.get("department")
    use_sme_scores = data.get("use_sme_scores", False)
    min_sme_score = data.get("min_sme_score", 1)

    score_label = f" (SME≥{min_sme_score})" if use_sme_scores else ""
    name = data.get("name") or (
        f"{department} SFT{score_label} {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        if department
        else f"Experiment{score_label} {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )

    config = {
        "model_name": data.get("model_name", "microsoft/phi-2"),
        "lora_r": data.get("lora_r", 16),
        "lora_alpha": data.get("lora_alpha", 32),
        "lora_dropout": data.get("lora_dropout", 0.05),
        "num_epochs": data.get("num_epochs", 10),
        "batch_size": data.get("batch_size", 2),
        "gradient_accumulation_steps": data.get("gradient_accumulation_steps", 4),
        "learning_rate": data.get("learning_rate", 0.0001),
        "max_seq_length": data.get("max_seq_length", 2048),
    }

    result = start_experiment(
        name,
        config,
        department=department,
        use_sme_scores=use_sme_scores,
        min_sme_score=min_sme_score,
    )
    if result.get("success"):
        return jsonify(result)
    return jsonify(result), 400


@sft_bp.route("/api/rlhf/experiment/<int:exp_id>/status", methods=["GET"])
@handle_route_errors
def api_experiment_status(exp_id: int):
    """Get real-time training status (for polling)."""
    if not SFT_AVAILABLE:
        return jsonify({"success": False, "error": "SFT module not available"}), 503

    status = sft_get_training_status()
    exp = get_experiment(exp_id)
    return jsonify(
        {
            "success": True,
            "training": status,
            "experiment": exp.get("experiment") if exp.get("success") else None,
        }
    )


@sft_bp.route("/api/rlhf/experiment/<int:exp_id>", methods=["GET"])
@handle_route_errors
def api_get_experiment(exp_id: int):
    """Get experiment details."""
    if not SFT_AVAILABLE:
        return jsonify({"success": False, "error": "SFT module not available"}), 503
    return jsonify(get_experiment(exp_id))


@sft_bp.route("/api/rlhf/experiment/<int:exp_id>", methods=["DELETE"])
@handle_route_errors
def api_delete_experiment(exp_id: int):
    """Delete an experiment."""
    if not SFT_AVAILABLE:
        return jsonify({"success": False, "error": "SFT module not available"}), 503
    return jsonify(delete_experiment(exp_id))


@sft_bp.route("/api/rlhf/experiment/<int:exp_id>/recalc-samples", methods=["POST"])
@handle_route_errors
def api_recalc_experiment_samples(exp_id: int):
    """Recalculate and persist training_samples for an experiment."""
    if not SFT_AVAILABLE:
        return jsonify({"success": False, "error": "SFT module not available"}), 503
    return jsonify(update_experiment_samples(exp_id))


@sft_bp.route("/api/rlhf/experiment/<int:exp_id>/test", methods=["POST"])
@handle_route_errors
def api_test_experiment_model(exp_id: int):
    """Test a trained model with a medical question."""
    if not SFT_AVAILABLE:
        return jsonify({"success": False, "error": "SFT module not available"}), 503

    data = request.json or {}
    question = data.get("question", "")
    if not question:
        return jsonify({"success": False, "error": "Question is required"}), 400

    # RLHF domain scope guard — three-tier filtering
    guard_disclaimer = ""
    scope_guard = current_app.config.get("SCOPE_GUARD")
    if scope_guard is not None:
        _guard_status, similarity_score, _guard_msg = scope_guard.check(question)
        if _guard_status == "rejected":
            logger.info(
                "SFT test rejected by scope guard (score=%.3f): %r",
                similarity_score,
                question[:80],
            )
            return jsonify(
                {
                    "success": False,
                    "out_of_scope": True,
                    "similarity_score": round(similarity_score, 4),
                    "error": _guard_msg,
                }
            ), 400
        elif _guard_status == "general_medical":
            logger.info(
                "SFT general medical query (score=%.3f) — answering with disclaimer",
                similarity_score,
            )
            guard_disclaimer = _guard_msg

    max_tokens = data.get("max_tokens", 256)
    result = test_trained_model(exp_id, question, max_new_tokens=max_tokens)
    if result.get("success"):
        if guard_disclaimer:
            result["response"] = guard_disclaimer + result.get("response", "")
        return jsonify(result)
    return jsonify(result), 400


@sft_bp.route("/api/rlhf/ranked-data", methods=["GET"])
@handle_route_errors
def api_get_ranked_data():
    """Get ranked training data with optional filtering."""
    if not SFT_AVAILABLE:
        return jsonify({"success": False, "error": "SFT module not available"}), 503

    group_id = request.args.get("group_id")
    search = request.args.get("search")
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 50, type=int)
    sme_filter = request.args.get("sme_filter")
    domain = request.args.get("domain")
    reason_empty = request.args.get("reason_empty") == "1"

    return jsonify(
        get_ranked_data(
            group_id=group_id,
            search=search,
            page=page,
            per_page=per_page,
            sme_filter=sme_filter,
            domain=domain,
            reason_empty=reason_empty,
        )
    )


@sft_bp.route("/api/rlhf/ranked-data", methods=["POST"])
@handle_route_errors
def api_add_ranked_data():
    """Add a new ranked data group."""
    if not SFT_AVAILABLE:
        return jsonify({"success": False, "error": "SFT module not available"}), 503

    data = request.json or {}
    prompt = data.get("prompt", "")
    responses = data.get("responses", [])
    domain = data.get("domain", "")
    doctor_name = data.get("doctor_name", "")

    if not prompt or not responses:
        return jsonify(
            {"success": False, "error": "prompt and responses are required"}
        ), 400

    return jsonify(add_ranked_entry(prompt, responses, domain=domain, doctor_name=doctor_name))


@sft_bp.route("/api/rlhf/ranked-data/<int:entry_id>", methods=["PUT"])
@handle_route_errors
def api_update_ranked_data(entry_id: int):
    """Update a ranked data entry."""
    if not SFT_AVAILABLE:
        return jsonify({"success": False, "error": "SFT module not available"}), 503

    data = request.json or {}
    return jsonify(
        update_ranked_entry(
            entry_id,
            response_text=data.get("response_text"),
            rank=data.get("rank"),
            reason=data.get("reason"),
        )
    )


@sft_bp.route("/api/rlhf/ranked-data/<int:entry_id>", methods=["DELETE"])
@handle_route_errors
def api_delete_ranked_data(entry_id: int):
    """Delete a single ranked data entry."""
    if not SFT_AVAILABLE:
        return jsonify({"success": False, "error": "SFT module not available"}), 503
    return jsonify(delete_ranked_entry(entry_id))


@sft_bp.route("/api/rlhf/ranked-data/group/<group_id>", methods=["DELETE"])
@handle_route_errors
def api_delete_ranked_group(group_id: str):
    """Delete all entries in a ranked data group."""
    if not SFT_AVAILABLE:
        return jsonify({"success": False, "error": "SFT module not available"}), 503
    return jsonify(delete_ranked_group(group_id))


@sft_bp.route("/api/rlhf/ranked-data/import", methods=["POST"])
@handle_route_errors
def api_import_ranked_data():
    """Import ranked data from JSONL file."""
    if not SFT_AVAILABLE:
        return jsonify({"success": False, "error": "SFT module not available"}), 503

    data = request.json or {}
    file_path = data.get("file_path")
    return jsonify(import_from_jsonl(file_path))


@sft_bp.route("/api/rlhf/ranked-data/export", methods=["GET"])
@handle_route_errors
def api_export_ranked_data():
    """Export ranked data as JSONL."""
    if not SFT_AVAILABLE:
        return jsonify({"success": False, "error": "SFT module not available"}), 503

    result = export_to_jsonl()
    if result.get("success"):
        return Response(
            result["data"],
            mimetype="application/jsonl",
            headers={
                "Content-Disposition": "attachment; filename=medical_ranked_export.jsonl"
            },
        )
    return jsonify(result), 400


@sft_bp.route("/api/rlhf/ranked-data/stats", methods=["GET"])
@handle_route_errors
def api_ranked_data_stats():
    """Get ranked data statistics."""
    if not SFT_AVAILABLE:
        return jsonify({"success": False, "error": "SFT module not available"}), 503
    return jsonify(get_ranked_data_stats())


@sft_bp.route("/api/rlhf/departments", methods=["GET"])
@handle_route_errors
def api_get_departments():
    """Get list of all medical departments."""
    if not SFT_AVAILABLE:
        return jsonify({"success": False, "error": "SFT module not available"}), 503
    return jsonify(get_department_list())


@sft_bp.route("/api/rlhf/ranked-data/by-department", methods=["GET"])
@handle_route_errors
def api_get_ranked_data_by_department():
    """Get prompts filtered by medical department."""
    if not SFT_AVAILABLE:
        return jsonify({"success": False, "error": "SFT module not available"}), 503

    department = request.args.get("department", "")
    limit = request.args.get("limit", 10, type=int)
    reason_empty_only = request.args.get("reason_empty", "false").lower() == "true"

    if not department:
        return jsonify(
            {"success": False, "error": "department parameter is required"}
        ), 400

    return jsonify(
        get_prompts_by_department(department, limit=limit, reason_empty_only=reason_empty_only)
    )
