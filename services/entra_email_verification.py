"""One-time email verification for linking an Entra identity to an existing PCES account."""
from __future__ import annotations

import hashlib
import hmac
import secrets
import smtplib
import ssl
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from uuid import UUID, uuid4

import psycopg
from psycopg import sql

from config import Config
from services.entra_identity_service import (
    EntraConfigurationError,
    EntraIdentityError,
    _connect,
    _validate_tenant,
)

CODE_TTL_MINUTES = 10
MAX_VERIFY_ATTEMPTS = 5
MAX_REQUESTS_PER_15_MINUTES = 3


def _key() -> bytes:
    secret = Config.SECRET_KEY
    if not secret or len(secret) < 32:
        raise EntraConfigurationError("A strong Flask SECRET_KEY is required for email verification")
    return secret.encode("utf-8")


def _digest(value: str) -> str:
    return hmac.new(_key(), value.encode("utf-8"), hashlib.sha256).hexdigest()


def _code_digest(challenge_id: str, code: str) -> str:
    return _digest(f"pces-entra-email-link:{challenge_id}:{code}")


def _send_code(recipient: str, code: str) -> None:
    if not all((Config.SMTP_HOST, Config.SMTP_PORT, Config.SMTP_FROM_EMAIL)):
        raise EntraConfigurationError("SMTP email delivery is not configured")
    message = EmailMessage()
    message["Subject"] = "Your PCES account-linking verification code"
    message["From"] = Config.SMTP_FROM_EMAIL
    message["To"] = recipient
    message.set_content(
        "A request was made to link this email address's existing PCES account "
        "to a Microsoft sign-in.\n\n"
        f"Your PCES verification code is: {code}\n\n"
        f"This code expires in {CODE_TTL_MINUTES} minutes and can be used once. "
        "If you did not initiate this request, you can ignore this email."
    )
    timeout = 10
    if Config.SMTP_USE_SSL:
        with smtplib.SMTP_SSL(Config.SMTP_HOST, Config.SMTP_PORT, timeout=timeout,
                              context=ssl.create_default_context()) as server:
            if Config.SMTP_USERNAME:
                server.login(Config.SMTP_USERNAME, Config.SMTP_PASSWORD or "")
            server.send_message(message)
    else:
        with smtplib.SMTP(Config.SMTP_HOST, Config.SMTP_PORT, timeout=timeout) as server:
            server.ehlo()
            if Config.SMTP_USE_TLS:
                server.starttls(context=ssl.create_default_context())
                server.ehlo()
            if Config.SMTP_USERNAME:
                server.login(Config.SMTP_USERNAME, Config.SMTP_PASSWORD or "")
            server.send_message(message)


def request_email_code(
    entra_tenant_id: str,
    object_id: str,
    email: str,
    pces_tenant_id: str,
) -> str:
    """Create a generic-response OTP challenge; never select an account by Entra email."""
    _validate_tenant(entra_tenant_id)
    try:
        object_id = str(UUID(object_id))
        pces_tenant_id = str(UUID(pces_tenant_id))
    except (ValueError, TypeError) as exc:
        raise EntraIdentityError("Invalid identity or hospital identifier") from exc

    normalized_email = (email or "").strip().casefold()
    if not normalized_email or len(normalized_email) > 254 or "@" not in normalized_email:
        normalized_email = "invalid-" + secrets.token_hex(16)
    email_hash = _digest("email:" + normalized_email)
    challenge_id = str(uuid4())
    code = f"{secrets.randbelow(1_000_000):06d}"
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=CODE_TTL_MINUTES)
    recipient_to_send = None
    party_id = None

    try:
        with _connect() as conn, conn.cursor() as cur:
            # Per-address request throttling also applies to unknown addresses.
            cur.execute(
                """SELECT COUNT(*) FROM entra_email_link_challenges
                   WHERE email_hash = %s AND created_at > NOW() - INTERVAL '15 minutes'""",
                (email_hash,),
            )
            if int(cur.fetchone()[0]) >= MAX_REQUESTS_PER_15_MINUTES:
                raise EntraIdentityError("Too many verification requests; try again later")

            columns = {row[0] for row in cur.execute(
                """SELECT LOWER(column_name) FROM information_schema.columns
                   WHERE table_name = 'p_party' AND table_schema = current_schema()"""
            ).fetchall()}
            # Fail closed: the database must explicitly record that the email
            # was verified through a trusted PCES process.
            verified_col = next((c for c in ("email_verified_at",) if c in columns), None)
            if verified_col and "email" in columns and {"party_id", "party_type", "is_active", "deleted_at"} <= columns:
                cur.execute(
                    """SELECT pp.party_id, pp.email
                       FROM p_party pp
                       JOIN p_party_tenant pt ON pt.party_id = pp.party_id
                       JOIN p_tenant t ON t.tenant_id = pt.tenant_id
                       WHERE LOWER(BTRIM(pp.email)) = %s
                         AND pp.email_verified_at IS NOT NULL
                         AND COALESCE(pp.is_active, TRUE) = TRUE
                         AND pp.deleted_at IS NULL
                         AND pp.party_type IN ('DOCTOR', 'NURSE', 'OFFICE_ADMIN')
                         AND CAST(pt.tenant_id AS TEXT) = %s
                         AND COALESCE(pt.is_active, TRUE) = TRUE AND pt.status = 'ACTIVE'
                         AND COALESCE(t.is_active, TRUE) = TRUE AND t.status = 'ACTIVE'
                       LIMIT 2 FOR UPDATE OF pp""",
                    (normalized_email, pces_tenant_id),
                )
                matches = cur.fetchall()
                if len(matches) == 1:
                    party_id, recipient_to_send = matches[0]
                    # Do not silently re-link an account that already belongs to another OID.
                    cur.execute(
                        "SELECT entra_object_id FROM p_party WHERE party_id = %s",
                        (party_id,),
                    )
                    current_oid = cur.fetchone()[0]
                    if current_oid and str(current_oid).casefold() != object_id.casefold():
                        party_id, recipient_to_send = None, None
                # Multiple matching records deliberately remain indistinguishable from no match.

            cur.execute(
                """INSERT INTO entra_email_link_challenges
                   (challenge_id, party_id, tenant_id, entra_tenant_id, entra_object_id,
                    email_hash, code_hash, expires_at, attempts, consumed_at, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 0, NULL, NOW())""",
                (challenge_id, party_id, pces_tenant_id if party_id else None,
                 entra_tenant_id, object_id, email_hash,
                 _code_digest(challenge_id, code), expires_at),
            )
    except (EntraConfigurationError, EntraIdentityError):
        raise
    except psycopg.Error as exc:
        raise EntraConfigurationError("Unable to create an email verification challenge") from exc

    # The same HTTP response is used for valid and invalid addresses. Only send
    # mail when an active, email-verified PCES account and membership matched.
    if recipient_to_send:
        try:
            _send_code(recipient_to_send, code)
        except Exception as exc:
            try:
                with _connect() as conn, conn.cursor() as cur:
                    cur.execute("DELETE FROM entra_email_link_challenges WHERE challenge_id = %s",
                                (challenge_id,))
            except Exception:
                pass
            if isinstance(exc, EntraConfigurationError):
                raise
            raise EntraConfigurationError("Unable to send the verification email") from exc
    return challenge_id


def verify_email_code(challenge_id: str, code: str, entra_tenant_id: str, object_id: str) -> dict:
    """Verify an OTP and atomically link the pending Entra identity to the PCES account."""
    _validate_tenant(entra_tenant_id)
    try:
        challenge_id = str(UUID(challenge_id))
        object_id = str(UUID(object_id))
    except (ValueError, TypeError) as exc:
        raise EntraIdentityError("Invalid verification challenge") from exc
    if not code or len(code) != 6 or not code.isdigit():
        raise EntraIdentityError("Invalid or expired verification code")

    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT party_id, tenant_id, entra_tenant_id, entra_object_id, email_hash,
                          code_hash, expires_at, attempts, consumed_at
                   FROM entra_email_link_challenges
                   WHERE challenge_id = %s FOR UPDATE""",
                (challenge_id,),
            )
            row = cur.fetchone()
            if not row:
                raise EntraIdentityError("Invalid or expired verification code")
            party_id, tenant_id, saved_tid, saved_oid, email_hash, code_hash, expires_at, attempts, consumed_at = row
            if (not party_id or not tenant_id or consumed_at is not None
                    or expires_at <= datetime.now(timezone.utc)
                    or int(attempts) >= MAX_VERIFY_ATTEMPTS
                    or str(saved_tid).casefold() != entra_tenant_id.casefold()
                    or str(saved_oid).casefold() != object_id.casefold()):
                raise EntraIdentityError("Invalid or expired verification code")
            if not hmac.compare_digest(str(code_hash), _code_digest(challenge_id, code)):
                cur.execute(
                    """UPDATE entra_email_link_challenges SET attempts = attempts + 1
                       WHERE challenge_id = %s""", (challenge_id,)
                )
                # Commit the failed-attempt counter before raising; otherwise the
                # connection context would roll it back with the exception.
                conn.commit()
                raise EntraIdentityError("Invalid or expired verification code")

            cur.execute(
                """SELECT party_id, party_type, entra_object_id, is_active, deleted_at,
                          email, email_verified_at
                   FROM p_party WHERE party_id = %s FOR UPDATE""", (party_id,)
            )
            party = cur.fetchone()
            if (not party or not party[3] or party[4] is not None
                    or party[1] not in ("DOCTOR", "NURSE", "OFFICE_ADMIN")
                    or not party[6] or not party[5]
                    or not hmac.compare_digest(
                        str(email_hash), _digest("email:" + str(party[5]).strip().casefold())
                    )):
                raise EntraIdentityError("PCES account is no longer eligible for linking")
            if party[2] and str(party[2]).casefold() != object_id.casefold():
                raise EntraIdentityError("PCES account is already linked to another Microsoft identity")
            cur.execute(
                """SELECT party_id FROM p_party
                   WHERE LOWER(entra_object_id) = LOWER(%s) AND party_id <> %s LIMIT 1""",
                (object_id, party_id),
            )
            if cur.fetchone():
                raise EntraIdentityError("Microsoft identity is already linked to another PCES account")
            cur.execute(
                """SELECT 1 FROM p_party_tenant pt JOIN p_tenant t ON t.tenant_id = pt.tenant_id
                   WHERE pt.party_id = %s AND pt.tenant_id = %s
                     AND COALESCE(pt.is_active, TRUE) = TRUE AND pt.status = 'ACTIVE'
                     AND COALESCE(t.is_active, TRUE) = TRUE AND t.status = 'ACTIVE'
                   LIMIT 1""", (party_id, tenant_id)
            )
            if not cur.fetchone():
                raise EntraIdentityError("PCES hospital membership is no longer active")
            if not party[2]:
                cur.execute(
                    """UPDATE p_party SET entra_object_id = %s
                       WHERE party_id = %s AND entra_object_id IS NULL
                       RETURNING party_id""", (object_id, party_id)
                )
                if not cur.fetchone():
                    raise EntraIdentityError("PCES identity changed during linking")
            cur.execute(
                "UPDATE entra_email_link_challenges SET consumed_at = NOW() WHERE challenge_id = %s",
                (challenge_id,),
            )
            columns = {r[0] for r in cur.execute(
                """SELECT LOWER(column_name) FROM information_schema.columns
                   WHERE table_name = 'p_party' AND table_schema = current_schema()"""
            ).fetchall()}
            role_column = next((c for c in ("pces_role", "role", "specialty", "department") if c in columns), None)
            role_expr = sql.Identifier(role_column) if role_column else sql.Identifier("party_type")
            cur.execute(sql.SQL("SELECT {} FROM p_party WHERE party_id = %s").format(role_expr), (party_id,))
            role = cur.fetchone()[0]
            return {"party_id": str(party_id), "party_type": party[1],
                    "pces_role": role or party[1], "tenant_id": str(tenant_id)}
    except (EntraConfigurationError, EntraIdentityError):
        raise
    except psycopg.Error as exc:
        raise EntraConfigurationError("Unable to verify the email code") from exc
