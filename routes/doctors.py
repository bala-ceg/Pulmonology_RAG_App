"""
Blueprint: Doctor management and SME review routes.

Covers /api/rlhf/doctors/*, /api/rlhf/sme-review-queue,
/api/rlhf/sme-review-submit, and /api/rlhf/sme-review-stats
(original main.py lines 4987-5355).

All data operations delegate to sft_experiment_manager.
"""

from __future__ import annotations

import traceback

from flask import Blueprint, jsonify, request

from utils.error_handlers import get_logger, handle_route_errors

logger = get_logger(__name__)

doctors_bp = Blueprint("doctors_bp", __name__)

# ---------------------------------------------------------------------------
# Optional SFT experiment manager (provides doctor & SME helpers)
# ---------------------------------------------------------------------------
try:
    from sft_experiment_manager import (
        add_doctor,
        delete_doctor,
        get_doctor_by_id,
        get_doctors,
        get_doctors_by_departments,
        update_doctor,
    )

    SFT_AVAILABLE = True
    logger.info("SFT doctor helpers loaded in doctors blueprint")
except Exception as _sft_err:
    SFT_AVAILABLE = False
    logger.warning("SFT module not available in doctors blueprint: %s", _sft_err)


# ---------------------------------------------------------------------------
# Doctor CRUD routes
# ---------------------------------------------------------------------------

@doctors_bp.route("/api/rlhf/doctors", methods=["GET"])
@handle_route_errors
def api_get_doctors():
    """Get list of doctors, optionally filtered by department."""
    if not SFT_AVAILABLE:
        return jsonify({"success": False, "error": "SFT module not available"}), 503

    department = request.args.get("department", None)
    active_only = request.args.get("active_only", "true").lower() == "true"
    return jsonify(get_doctors(department=department, active_only=active_only))


@doctors_bp.route("/api/rlhf/doctors/by-department", methods=["GET"])
@handle_route_errors
def api_get_doctors_by_department():
    """Get all doctors grouped by department."""
    if not SFT_AVAILABLE:
        return jsonify({"success": False, "error": "SFT module not available"}), 503
    return jsonify(get_doctors_by_departments())


@doctors_bp.route("/api/rlhf/doctors/<int:doctor_id>", methods=["GET"])
@handle_route_errors
def api_get_doctor(doctor_id: int):
    """Get a single doctor by ID."""
    if not SFT_AVAILABLE:
        return jsonify({"success": False, "error": "SFT module not available"}), 503
    return jsonify(get_doctor_by_id(doctor_id))


@doctors_bp.route("/api/rlhf/doctors", methods=["POST"])
@handle_route_errors
def api_add_doctor():
    """Add a new doctor."""
    if not SFT_AVAILABLE:
        return jsonify({"success": False, "error": "SFT module not available"}), 503

    data = request.json or {}
    name = data.get("name")
    department = data.get("department")
    email = data.get("email")
    specialty = data.get("specialty")

    if not name or not department:
        return jsonify(
            {"success": False, "error": "name and department are required"}
        ), 400

    result = add_doctor(name, department, email=email, specialty=specialty)
    if result.get("success"):
        return jsonify(result), 201
    return jsonify(result), 400


@doctors_bp.route("/api/rlhf/doctors/<int:doctor_id>", methods=["PUT"])
@handle_route_errors
def api_update_doctor(doctor_id: int):
    """Update an existing doctor."""
    if not SFT_AVAILABLE:
        return jsonify({"success": False, "error": "SFT module not available"}), 503

    data = request.json or {}
    result = update_doctor(
        doctor_id,
        name=data.get("name"),
        email=data.get("email"),
        department=data.get("department"),
        specialty=data.get("specialty"),
        is_active=data.get("is_active"),
    )
    if result.get("success"):
        return jsonify(result)
    return jsonify(result), 400


@doctors_bp.route("/api/rlhf/doctors/<int:doctor_id>", methods=["DELETE"])
@handle_route_errors
def api_delete_doctor(doctor_id: int):
    """Delete (deactivate) a doctor."""
    if not SFT_AVAILABLE:
        return jsonify({"success": False, "error": "SFT module not available"}), 503

    hard_delete = request.args.get("hard", "false").lower() == "true"
    result = delete_doctor(doctor_id, hard_delete=hard_delete)
    if result.get("success"):
        return jsonify(result)
    return jsonify(result), 400


# ---------------------------------------------------------------------------
# SME review routes
# ---------------------------------------------------------------------------

@doctors_bp.route("/api/rlhf/sme-review-queue", methods=["GET"])
@handle_route_errors
def api_get_sme_review_queue():
    """Get prompts for SME review with filtering options."""
    if not SFT_AVAILABLE:
        return jsonify({"success": False, "error": "SFT module not available"}), 503

    domain = request.args.get("domain", "")
    status = request.args.get("status", "pending")
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 50, type=int)

    import sft_experiment_manager as sft  # noqa: PLC0415

    # Check whether doctor_name column exists (read-only — works even without ALTER privilege)
    has_doctor_name = False
    try:
        with sft._connect() as _cc:
            with _cc.cursor() as _ccur:
                if sft._use_sqlite:
                    _ccur.execute("PRAGMA table_info(sft_ranked_data)")
                    has_doctor_name = any(r[1] == "doctor_name" for r in _ccur.fetchall())
                else:
                    _ccur.execute(
                        "SELECT 1 FROM information_schema.columns "
                        "WHERE table_name = 'sft_ranked_data' AND column_name = 'doctor_name'"
                    )
                    has_doctor_name = _ccur.fetchone() is not None
    except Exception as _ce:
        logger.warning("Could not check doctor_name column existence: %s", _ce)

    # If the column is missing, try to add it (best-effort; may fail due to permissions)
    if not has_doctor_name:
        try:
            with sft._connect(autocommit=True) as _mc:
                with _mc.cursor() as _cur:
                    _cur.execute(
                        "ALTER TABLE sft_ranked_data ADD COLUMN IF NOT EXISTS doctor_name TEXT;"
                    )
            has_doctor_name = True
        except Exception as _col_err:
            logger.warning("Could not add doctor_name column (will use NULL fallback): %s", _col_err)

    doctor_name_col = ", doctor_name" if has_doctor_name else ", NULL AS doctor_name"

    with sft._connect() as conn:
        with conn.cursor() as cur:
            query = f"""
                SELECT
                    id, prompt, response_text, rank, reason, group_id, domain,
                    sme_score, sme_score_reason, sme_reviewed_by, sme_reviewed_at,
                    created_by, created_at, updated_by, updated_at{doctor_name_col}
                FROM sft_ranked_data
                WHERE 1=1
            """
            params: list = []

            if domain:
                query += " AND LOWER(domain) = LOWER(%s)"
                params.append(domain)

            if status == "pending":
                query += " AND sme_score IS NULL"
            elif status == "reviewed":
                query += " AND sme_score IS NOT NULL"

            query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
            params.extend([per_page, (page - 1) * per_page])

            if sft._use_sqlite:
                query = sft._adapt_sql(query)

            cur.execute(query, params)
            rows = cur.fetchall()

            count_query = "SELECT COUNT(*) FROM sft_ranked_data WHERE 1=1"
            count_params: list = []
            if domain:
                count_query += " AND LOWER(domain) = LOWER(%s)"
                count_params.append(domain)
            if status == "pending":
                count_query += " AND sme_score IS NULL"
            elif status == "reviewed":
                count_query += " AND sme_score IS NOT NULL"

            if sft._use_sqlite:
                count_query = sft._adapt_sql(count_query)

            cur.execute(count_query, count_params)
            total = cur.fetchone()[0]

    items = [
        {
            "id": row[0],
            "prompt": row[1],
            "response_text": row[2],
            "rank": row[3],
            "reason": row[4],
            "group_id": row[5],
            "domain": row[6],
            "sme_score": row[7],
            "sme_score_reason": row[8],
            "sme_reviewed_by": row[9],
            "sme_reviewed_at": sft._dt(row[10]),
            "created_by": row[11],
            "created_at": sft._dt(row[12]),
            "updated_by": row[13],
            "updated_at": sft._dt(row[14]),
            "doctor_name": row[15],
        }
        for row in rows
    ]

    return jsonify(
        {
            "success": True,
            "items": items,
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": (total + per_page - 1) // per_page,
        }
    )


@doctors_bp.route("/api/rlhf/sme-review-submit", methods=["POST"])
@handle_route_errors
def api_submit_sme_reviews():
    """Batch update SME scores and reasons for multiple prompts."""
    if not SFT_AVAILABLE:
        return jsonify({"success": False, "error": "SFT module not available"}), 503

    data = request.json or {}
    reviews = data.get("reviews", [])
    sme_name = data.get("sme_name", "Unknown SME")

    if not reviews:
        return jsonify({"success": False, "error": "No reviews provided"}), 400

    import sft_experiment_manager as sft  # noqa: PLC0415

    with sft._connect() as conn:
        with conn.cursor() as cur:
            updated_count = 0
            for review in reviews:
                entry_id = review.get("id")
                sme_score = review.get("sme_score")
                sme_score_reason = review.get("sme_score_reason", "")

                if not entry_id or not sme_score:
                    continue
                if sme_score < 1 or sme_score > 5:
                    continue

                query = """
                    UPDATE sft_ranked_data
                    SET sme_score = %s,
                        sme_score_reason = %s,
                        sme_reviewed_by = %s,
                        sme_reviewed_at = CURRENT_TIMESTAMP,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                """
                if sft._use_sqlite:
                    query = sft._adapt_sql(query)

                cur.execute(query, (sme_score, sme_score_reason, sme_name, entry_id))
                updated_count += cur.rowcount

            conn.commit()

    return jsonify(
        {
            "success": True,
            "updated_count": updated_count,
            "message": f"Successfully updated {updated_count} reviews",
        }
    )


@doctors_bp.route("/api/rlhf/sme-review-stats", methods=["GET"])
@handle_route_errors
def api_get_sme_review_stats():
    """Get statistics about SME review progress."""
    if not SFT_AVAILABLE:
        return jsonify({"success": False, "error": "SFT module not available"}), 503

    domain = request.args.get("domain", "")

    import sft_experiment_manager as sft  # noqa: PLC0415

    with sft._connect() as conn:
        with conn.cursor() as cur:
            stats_query = """
                SELECT
                    COUNT(*) as total,
                    COUNT(CASE WHEN sme_score IS NOT NULL THEN 1 END) as reviewed,
                    COUNT(CASE WHEN sme_score IS NULL THEN 1 END) as pending
                FROM sft_ranked_data
            """
            stats_params: list = []
            if domain:
                stats_query += " WHERE LOWER(domain) = LOWER(%s)"
                stats_params.append(domain)

            if sft._use_sqlite:
                stats_query = sft._adapt_sql(stats_query)

            cur.execute(stats_query, stats_params)
            row = cur.fetchone()
            total, reviewed, pending = row[0], row[1], row[2]

            if sft._use_sqlite:
                week_query = """
                    SELECT COUNT(*)
                    FROM sft_ranked_data
                    WHERE sme_reviewed_at >= datetime('now', '-7 days')
                """
            else:
                week_query = """
                    SELECT COUNT(*)
                    FROM sft_ranked_data
                    WHERE sme_reviewed_at >= NOW() - INTERVAL '7 days'
                """
            week_params: list = []
            if domain:
                week_query += " AND LOWER(domain) = LOWER(%s)"
                week_params.append(domain)

            if sft._use_sqlite:
                week_query = sft._adapt_sql(week_query)

            cur.execute(week_query, week_params)
            this_week = cur.fetchone()[0]

            score_query = """
                SELECT sme_score, COUNT(*) as count
                FROM sft_ranked_data
                WHERE sme_score IS NOT NULL
            """
            score_params: list = []
            if domain:
                score_query += " AND LOWER(domain) = LOWER(%s)"
                score_params.append(domain)
            score_query += " GROUP BY sme_score ORDER BY sme_score"

            if sft._use_sqlite:
                score_query = sft._adapt_sql(score_query)

            cur.execute(score_query, score_params)
            score_dist = {r[0]: r[1] for r in cur.fetchall()}

            reviewer_query = """
                SELECT sme_reviewed_by, COUNT(*) as count
                FROM sft_ranked_data
                WHERE sme_reviewed_by IS NOT NULL
            """
            reviewer_params: list = []
            if domain:
                reviewer_query += " AND LOWER(domain) = LOWER(%s)"
                reviewer_params.append(domain)
            reviewer_query += " GROUP BY sme_reviewed_by ORDER BY count DESC LIMIT 5"

            if sft._use_sqlite:
                reviewer_query = sft._adapt_sql(reviewer_query)

            cur.execute(reviewer_query, reviewer_params)
            top_reviewers = [{"name": r[0], "count": r[1]} for r in cur.fetchall()]

    return jsonify(
        {
            "success": True,
            "total": total,
            "reviewed": reviewed,
            "pending": pending,
            "reviewed_this_week": this_week,
            "review_percentage": round(
                (reviewed / total * 100) if total > 0 else 0, 1
            ),
            "score_distribution": score_dist,
            "top_reviewers": top_reviewers,
        }
    )
