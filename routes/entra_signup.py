"""Routes for Entra self-signup and hospital Office Admin review."""
from __future__ import annotations

import secrets

from flask import Blueprint, current_app, jsonify, render_template, request, session

from services.entra_signup_service import (
    SignupError, active_hospitals, list_requests, review_request, submit_request,
)

entra_signup_bp = Blueprint("entra_signup_bp", __name__, url_prefix="/auth/entra")


def _signup_page(error=None):
    try:
        hospitals = active_hospitals()
    except Exception:
        current_app.logger.exception("Unable to load hospitals for signup")
        hospitals = []
        error = error or "Hospital list is temporarily unavailable."
    return render_template("entra_signup.html", hospitals=hospitals,
                           csrf_token=session.get("entra_signup_csrf"), error=error)


@entra_signup_bp.get("/signup")
def signup_page():
    if not isinstance(session.get("entra_pending_signup"), dict):
        return jsonify({"success": False, "error": "Start with Microsoft sign-in"}), 401
    session.setdefault("entra_signup_csrf", secrets.token_urlsafe(32))
    return _signup_page()


@entra_signup_bp.post("/signup")
def signup_submit():
    pending = session.get("entra_pending_signup")
    expected, supplied = session.get("entra_signup_csrf"), request.form.get("csrf_token", "")
    if not isinstance(pending, dict) or not expected or not supplied or not secrets.compare_digest(expected, supplied):
        session.clear()
        return jsonify({"success": False, "error": "Signup session expired; restart Microsoft sign-in"}), 400
    try:
        request_id = submit_request(pending["tenant_id"], pending["object_id"], {
            "tenant_id": request.form.get("tenant_id"),
            "party_type": request.form.get("party_type"),
            "first_name": request.form.get("first_name"),
            "middle_name": request.form.get("middle_name"),
            "last_name": request.form.get("last_name"),
            "date_of_birth": request.form.get("date_of_birth"),
            "phone": request.form.get("phone"),
            "email": pending.get("email"),
        })
    except SignupError as exc:
        return _signup_page(error=str(exc)), 400
    except Exception:
        current_app.logger.exception("Entra signup request failed")
        return _signup_page(error="Unable to submit signup request right now."), 503
    session.clear()
    return render_template("entra_signup_submitted.html", request_id=request_id), 200


def _office_admin_identity():
    identity = session.get("pces_identity")
    if not isinstance(identity, dict) or not identity.get("party_id") or not identity.get("tenant_id"):
        return None
    if str(identity.get("party_type") or "").upper() != "OFFICE_ADMIN":
        return None
    return identity


@entra_signup_bp.get("/admin")
def office_admin_page():
    identity = _office_admin_identity()
    if not identity:
        return jsonify({"success": False, "error": "Office Admin access required"}), 403
    session.setdefault("entra_admin_csrf", secrets.token_urlsafe(32))
    return render_template("entra_signup_admin.html", csrf_token=session["entra_admin_csrf"])


@entra_signup_bp.get("/admin/requests")
def office_admin_requests():
    identity = _office_admin_identity()
    if not identity:
        return jsonify({"success": False, "error": "Office Admin access required"}), 403
    try:
        return jsonify({"success": True, "requests": list_requests(identity["party_id"], identity["tenant_id"])})
    except SignupError as exc:
        return jsonify({"success": False, "error": str(exc)}), 403
    except Exception:
        current_app.logger.exception("Unable to list signup requests")
        return jsonify({"success": False, "error": "Unable to load signup requests"}), 503


@entra_signup_bp.post("/admin/requests/<request_id>/review")
def office_admin_review(request_id):
    identity = _office_admin_identity()
    if not identity:
        return jsonify({"success": False, "error": "Office Admin access required"}), 403
    expected = session.get("entra_admin_csrf")
    supplied = request.headers.get("X-CSRF-Token", "")
    if not expected or not supplied or not secrets.compare_digest(expected, supplied):
        return jsonify({"success": False, "error": "Invalid review token"}), 400
    data = request.get_json(silent=True) or {}
    try:
        result = review_request(identity["party_id"], identity["tenant_id"], request_id,
                                data.get("decision", ""), data.get("rejection_reason"))
        return jsonify({"success": True, **result})
    except SignupError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception:
        current_app.logger.exception("Unable to review signup request")
        return jsonify({"success": False, "error": "Unable to review signup request"}), 503
