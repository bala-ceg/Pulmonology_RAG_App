"""Microsoft Entra External ID authentication endpoints.

This blueprint is additive during the phased migration. The identity resolver
uses the existing p_party.entra_object_id and validates the token tenant against
the configured single-tenant authority. Environment-backed testing and the
application-wide authorization audit are still required before production use.
"""
from __future__ import annotations

import secrets
import msal
from flask import Blueprint, current_app, jsonify, redirect, request, session

from config import Config
from services.entra_identity_service import (
    EntraConfigurationError,
    EntraIdentityError,
    resolve_pces_identity,
)

entra_auth_bp = Blueprint("entra_auth_bp", __name__, url_prefix="/auth/entra")


def _configured() -> bool:
    return all((
        Config.ENTRA_CLIENT_ID,
        Config.ENTRA_CLIENT_SECRET,
        Config.ENTRA_AUTHORITY,
        Config.ENTRA_REDIRECT_URI,
    ))


def _client() -> msal.ConfidentialClientApplication:
    if not _configured():
        raise EntraConfigurationError("Microsoft Entra authentication is not configured")
    return msal.ConfidentialClientApplication(
        Config.ENTRA_CLIENT_ID,
        authority=Config.ENTRA_AUTHORITY,
        client_credential=Config.ENTRA_CLIENT_SECRET,
    )


@entra_auth_bp.get("/login")
def entra_login():
    """Start the authorization-code flow with an unpredictable state value."""
    if not _configured():
        return jsonify({"success": False, "error": "Entra authentication is not configured"}), 503

    state = secrets.token_urlsafe(32)
    session["entra_oauth_state"] = state
    session["entra_login_next"] = "/"
    try:
        auth_url = _client().get_authorization_request_url(
            # MSAL adds the reserved OIDC scopes (openid, profile, offline_access).
            scopes=["email"],
            state=state,
            redirect_uri=Config.ENTRA_REDIRECT_URI,
            prompt="select_account",
        )
    except Exception:
        current_app.logger.exception("Unable to start Entra authorization flow")
        session.pop("entra_oauth_state", None)
        return jsonify({"success": False, "error": "Unable to start sign-in"}), 500
    return redirect(auth_url)


@entra_auth_bp.get("/callback")
def entra_callback():
    """Validate OAuth state, exchange the code, and map the trusted identity."""
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
            code,
            # Keep this aligned with the initial request; don't pass reserved OIDC scopes.
            scopes=["email"],
            redirect_uri=Config.ENTRA_REDIRECT_URI,
        )
    except Exception:
        current_app.logger.exception("Entra authorization-code exchange failed")
        session.clear()
        return jsonify({"success": False, "error": "Sign-in failed"}), 401

    if result.get("error") or not isinstance(result.get("id_token_claims"), dict):
        current_app.logger.warning(
            "Entra token exchange failed: %s",
            result.get("error_description") or result.get("error") or "ID token claims missing",
        )
        session.clear()
        return jsonify({"success": False, "error": "Microsoft identity could not be verified"}), 401

    claims = result["id_token_claims"]
    issuer_tenant = str(claims.get("tid") or "")
    object_id = str(claims.get("oid") or "")
    if not issuer_tenant or not object_id:
        session.clear()
        return jsonify({"success": False, "error": "Required directory identity claims are missing"}), 403

    try:
        identity = resolve_pces_identity(issuer_tenant, object_id)
    except EntraIdentityError as exc:
        current_app.logger.warning("Entra identity mapping denied: %s", exc)
        session.clear()
        return jsonify({"success": False, "error": "This account is not provisioned for PCES access"}), 403
    except EntraConfigurationError:
        current_app.logger.exception("Entra identity mapping is not configured")
        session.clear()
        return jsonify({"success": False, "error": "Identity mapping is not configured"}), 503

    # Regenerate session state after successful authentication.
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
    return redirect("/")


@entra_auth_bp.post("/logout")
def entra_logout():
    """Clear the local PCES session; front-channel provider logout can be added per tenant policy."""
    session.clear()
    return jsonify({"success": True, "message": "Signed out"})


@entra_auth_bp.get("/status")
def entra_status():
    """Return only a minimal authentication status; never expose token material."""
    identity = session.get("pces_identity")
    return jsonify({
        "authenticated": bool(identity),
        "provider": identity.get("auth_provider") if identity else None,
    })
