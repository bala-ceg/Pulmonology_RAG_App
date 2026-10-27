"""Microsoft Entra External ID authentication and existing-account linking."""
from __future__ import annotations

import secrets
import msal
from flask import Blueprint, current_app, jsonify, redirect, render_template, request, session

from config import Config
from services.entra_identity_service import (
    EntraConfigurationError,
    EntraIdentityError,
    link_existing_pces_account,
    resolve_pces_identity,
)

entra_auth_bp = Blueprint("entra_auth_bp", __name__, url_prefix="/auth/entra")


def _configured() -> bool:
    return all((Config.ENTRA_CLIENT_ID, Config.ENTRA_CLIENT_SECRET,
                Config.ENTRA_AUTHORITY, Config.ENTRA_REDIRECT_URI))


def _client() -> msal.ConfidentialClientApplication:
    if not _configured():
        raise EntraConfigurationError("Microsoft Entra authentication is not configured")
    return msal.ConfidentialClientApplication(
        Config.ENTRA_CLIENT_ID,
        authority=Config.ENTRA_AUTHORITY,
        client_credential=Config.ENTRA_CLIENT_SECRET,
    )


def _set_authenticated_session(identity: dict, issuer_tenant: str, object_id: str) -> None:
    session.clear()
    session["pces_identity"] = {
        "party_id": str(identity["party_id"]),
        "tenant_id": str(identity["tenant_id"]),
        "party_type": identity["party_type"],
        "pces_role": identity["pces_role"],
        "auth_provider": "entra_external_id",
        "entra_tenant_id": issuer_tenant,
        "entra_object_id": object_id,
    }


@entra_auth_bp.get("/login")
def entra_login():
    """Start the authorization-code flow with an unpredictable state value."""
    if not _configured():
        return jsonify({"success": False, "error": "Entra authentication is not configured"}), 503
    state = secrets.token_urlsafe(32)
    session["entra_oauth_state"] = state
    try:
        auth_url = _client().get_authorization_request_url(
            scopes=["email"], state=state, redirect_uri=Config.ENTRA_REDIRECT_URI,
            prompt="select_account",
        )
    except Exception:
        current_app.logger.exception("Unable to start Entra authorization flow")
        session.pop("entra_oauth_state", None)
        return jsonify({"success": False, "error": "Unable to start sign-in"}), 500
    return redirect(auth_url)


@entra_auth_bp.get("/callback")
def entra_callback():
    """Validate OAuth state, then resolve or request verified linking to an existing PCES account."""
    expected_state = session.pop("entra_oauth_state", None)
    returned_state = request.args.get("state")
    if not expected_state or not returned_state or not secrets.compare_digest(expected_state, returned_state):
        session.clear()
        return jsonify({"success": False, "error": "Invalid authentication state"}), 400
    if request.args.get("error"):
        session.clear()
        return jsonify({"success": False, "error": "Microsoft sign-in was not completed"}), 401
    code = request.args.get("code")
    if not code:
        session.clear()
        return jsonify({"success": False, "error": "Authorization code is missing"}), 400

    try:
        result = _client().acquire_token_by_authorization_code(
            code, scopes=["email"], redirect_uri=Config.ENTRA_REDIRECT_URI,
        )
    except Exception:
        current_app.logger.exception("Entra authorization-code exchange failed")
        session.clear()
        return jsonify({"success": False, "error": "Sign-in failed"}), 401

    if result.get("error") or not isinstance(result.get("id_token_claims"), dict):
        current_app.logger.warning("Entra token exchange failed: %s",
                                   result.get("error") or "ID token claims missing")
        session.clear()
        return jsonify({"success": False, "error": "Microsoft identity could not be verified"}), 401

    claims = result["id_token_claims"]
    issuer_tenant, object_id = str(claims.get("tid") or ""), str(claims.get("oid") or "")
    if not issuer_tenant or not object_id:
        session.clear()
        return jsonify({"success": False, "error": "Required directory identity claims are missing"}), 403

    try:
        identity = resolve_pces_identity(issuer_tenant, object_id)
    except EntraIdentityError as exc:
        # Only an entirely unmapped OID may enter account linking. Conflicting,
        # inactive, or otherwise invalid identities must never get a link form.
        if str(exc) != "No active identity mapping exists" and str(exc) != "No active PCES identity mapping exists":
            current_app.logger.warning("Entra identity mapping denied: %s", exc)
            session.clear()
            return jsonify({"success": False, "error": "This account is not provisioned for PCES access"}), 403
        session.clear()
        session["entra_pending_link"] = {"tenant_id": issuer_tenant, "object_id": object_id}
        session["entra_link_csrf"] = secrets.token_urlsafe(32)
        return render_template("entra_link.html", csrf_token=session["entra_link_csrf"]), 200
    except EntraConfigurationError:
        current_app.logger.exception("Entra identity mapping is not configured")
        session.clear()
        return jsonify({"success": False, "error": "Identity mapping is not configured"}), 503

    _set_authenticated_session(identity, issuer_tenant, object_id)
    return redirect("/")


@entra_auth_bp.post("/link")
def entra_link_existing_account():
    """Link only after successful verification of the existing PCES credentials and tenant."""
    pending = session.get("entra_pending_link")
    expected_csrf = session.get("entra_link_csrf")
    supplied_csrf = request.form.get("csrf_token", "")
    if not pending or not expected_csrf or not supplied_csrf or not secrets.compare_digest(expected_csrf, supplied_csrf):
        session.clear()
        return jsonify({"success": False, "error": "Linking session expired; restart sign-in"}), 400

    username = (request.form.get("username") or "").strip()
    password = request.form.get("password") or ""
    pces_tenant_id = (request.form.get("pces_tenant_id") or "").strip()
    try:
        identity = link_existing_pces_account(
            pending["tenant_id"], pending["object_id"], username, password, pces_tenant_id,
        )
    except EntraIdentityError as exc:
        current_app.logger.warning("Existing PCES account linking denied: %s", exc)
        # Keep the pending identity and CSRF token so the user can retry; do not log credentials.
        return render_template("entra_link.html", csrf_token=expected_csrf,
                               error="We couldn't verify that PCES account. Check your credentials and hospital, then try again."), 403
    except EntraConfigurationError:
        current_app.logger.exception("PCES account linking is not configured")
        return jsonify({"success": False, "error": "Account linking is temporarily unavailable"}), 503

    issuer_tenant, object_id = pending["tenant_id"], pending["object_id"]
    _set_authenticated_session(identity, issuer_tenant, object_id)
    return redirect("/")


@entra_auth_bp.post("/logout")
def entra_logout():
    """Clear the local PCES session."""
    session.clear()
    return jsonify({"success": True, "message": "Signed out"})


@entra_auth_bp.get("/status")
def entra_status():
    """Return minimal authentication status without exposing token material."""
    identity = session.get("pces_identity")
    return jsonify({"authenticated": bool(identity),
                    "provider": identity.get("auth_provider") if identity else None})
