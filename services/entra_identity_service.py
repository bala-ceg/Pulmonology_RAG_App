"""Trusted linking from Microsoft Entra External ID to an existing PCES account."""
from __future__ import annotations

from urllib.parse import urlparse
from uuid import UUID

import psycopg
from psycopg import sql

from config import Config


class EntraConfigurationError(RuntimeError):
    """Required Entra or database configuration is missing."""


class EntraIdentityError(RuntimeError):
    """The verified Entra identity has no eligible PCES mapping."""


def _connect():
    kwargs = {
        "host": Config.PG_TOOL_HOST or Config.DB_HOST,
        "port": Config.PG_TOOL_PORT or Config.DB_PORT or "5432",
        "dbname": Config.PG_TOOL_NAME or Config.DB_NAME,
        "user": Config.PG_TOOL_USER or Config.DB_USER,
        "password": Config.PG_TOOL_PASSWORD or Config.DB_PASSWORD,
        "connect_timeout": Config.DB_CONNECT_TIMEOUT,
    }
    if not all(kwargs.get(key) for key in ("host", "dbname", "user", "password")):
        raise EntraConfigurationError("EHR PostgreSQL connection is not configured")
    return psycopg.connect(**kwargs)


def _configured_tenant_id() -> str:
    authority = Config.ENTRA_AUTHORITY
    if not authority:
        raise EntraConfigurationError("Microsoft Entra authority is not configured")
    parts = [part for part in urlparse(authority).path.split("/") if part]
    if not parts:
        raise EntraConfigurationError("Entra authority must include its tenant identifier")
    return parts[-1].strip().lower()


def _validate_tenant(tenant_id: str) -> None:
    if not tenant_id or tenant_id.strip().lower() != _configured_tenant_id():
        raise EntraIdentityError("Identity was issued by an unexpected Entra tenant")


def _table_columns(cur, table_name: str) -> set[str]:
    cur.execute(
        "SELECT LOWER(column_name) FROM information_schema.columns WHERE table_name = %s",
        (table_name,),
    )
    return {row[0] for row in cur.fetchall()}


def link_existing_pces_account(
    entra_tenant_id: str,
    object_id: str,
    username: str,
    password: str,
    pces_tenant_id: str,
) -> dict:
    """Authenticate with existing PCES credentials and atomically link this Entra OID.

    Email, display name, and user-supplied party IDs are never used to select the
    account. The user must prove possession of their existing PCES credentials
    and choose a tenant membership that is active for that account.
    """
    _validate_tenant(entra_tenant_id)
    if not object_id or not username or not password or not pces_tenant_id:
        raise EntraIdentityError("PCES username, password, and hospital are required")
    try:
        object_id = str(UUID(object_id))
        pces_tenant_id = str(UUID(pces_tenant_id))
    except ValueError as exc:
        raise EntraIdentityError("Invalid identity or hospital identifier") from exc

    try:
        with _connect() as conn, conn.cursor() as cur:
            columns = _table_columns(cur, "p_party")
            login_cols = [c for c in ("username", "user_name", "email") if c in columns]
            password_col = next((c for c in ("password_hash", "password") if c in columns), None)
            if not login_cols or not password_col:
                raise EntraConfigurationError("Existing PCES credential login is not configured")
            if not {"party_id", "party_type", "entra_object_id", "is_active", "deleted_at"} <= columns:
                raise EntraConfigurationError("p_party is missing required identity columns")

            username_expr = "username" if "username" in columns else login_cols[0]
            role_col = next((c for c in ("pces_role", "role", "specialty", "department") if c in columns), None)
            role_expr = sql.Identifier(role_col) if role_col else sql.Identifier("party_type")
            predicates = sql.SQL(" OR ").join(
                sql.SQL("LOWER({}) = LOWER(%s)").format(sql.Identifier(col)) for col in login_cols
            )
            query = sql.SQL("""
                SELECT party_id, party_type, entra_object_id, is_active, deleted_at,
                       {role_col}, {password_col}
                FROM p_party
                WHERE party_type IN ('DOCTOR', 'NURSE', 'OFFICE_ADMIN')
                  AND COALESCE(is_active, TRUE) = TRUE
                  AND deleted_at IS NULL
                  AND ({predicates})
                LIMIT 2
                FOR UPDATE
            """).format(
                role_col=role_expr,
                password_col=sql.Identifier(password_col),
                predicates=predicates,
            )
            cur.execute(query, tuple(username for _ in login_cols))
            rows = cur.fetchall()
            if len(rows) != 1:
                raise EntraIdentityError("PCES credentials could not be verified")
            party_id, party_type, existing_oid, is_active, deleted_at, pces_role, stored_password = rows[0]
            # Match the existing /api/login behavior. If PCES changes to a
            # password-hash verifier, this linking verifier must change with it.
            if stored_password is None or password != stored_password:
                raise EntraIdentityError("PCES credentials could not be verified")
            if existing_oid and str(existing_oid).lower() != object_id.lower():
                raise EntraIdentityError("This PCES account is already linked to another Entra identity")

            cur.execute("""
                SELECT pt.tenant_id
                FROM p_party_tenant AS pt
                JOIN p_tenant AS t ON t.tenant_id = pt.tenant_id
                WHERE pt.party_id = %s
                  AND CAST(pt.tenant_id AS TEXT) = %s
                  AND COALESCE(pt.is_active, TRUE) = TRUE
                  AND pt.status = 'ACTIVE'
                  AND COALESCE(t.is_active, TRUE) = TRUE
                  AND t.status = 'ACTIVE'
                FOR UPDATE
            """, (party_id, pces_tenant_id))
            membership = cur.fetchone()
            if not membership:
                raise EntraIdentityError("Selected hospital is not active for this PCES account")

            cur.execute("""
                SELECT party_id FROM p_party
                WHERE LOWER(entra_object_id) = LOWER(%s) AND party_id <> %s
                LIMIT 1
            """, (object_id, party_id))
            if cur.fetchone():
                raise EntraIdentityError("This Entra identity is already linked to another PCES account")

            if not existing_oid:
                cur.execute("""
                    UPDATE p_party
                    SET entra_object_id = %s
                    WHERE party_id = %s AND entra_object_id IS NULL
                      AND COALESCE(is_active, TRUE) = TRUE AND deleted_at IS NULL
                    RETURNING party_id
                """, (object_id, party_id))
                if not cur.fetchone():
                    raise EntraIdentityError("PCES identity changed during linking")

        return {
            "party_id": str(party_id),
            "party_type": party_type,
            "pces_role": pces_role or party_type,
            "tenant_id": str(membership[0]),
        }
    except (EntraConfigurationError, EntraIdentityError):
        raise
    except psycopg.Error as exc:
        raise EntraConfigurationError("Unable to verify or link the PCES account") from exc


def resolve_pces_identity(tenant_id: str, object_id: str) -> dict:
    """Resolve a verified Entra identity to exactly one active PCES membership."""
    if not tenant_id or not object_id:
        raise EntraIdentityError("Required directory identity claims are missing")
    _validate_tenant(tenant_id)
    try:
        object_id = str(UUID(object_id))
    except ValueError as exc:
        raise EntraIdentityError("Invalid Entra Object ID") from exc

    try:
        with _connect() as conn, conn.cursor() as cur:
            columns = _table_columns(cur, "p_party")
            role_column = next((c for c in ("pces_role", "role", "specialty", "department") if c in columns), None)
            if not {"party_id", "party_type", "entra_object_id"} <= columns:
                raise EntraConfigurationError("p_party is missing required identity mapping columns")
            role_expr = sql.Identifier(role_column) if role_column else sql.Identifier("party_type")
            query = sql.SQL("""
                SELECT pp.party_id, pp.party_type, COALESCE({role_col}, pp.party_type) AS pces_role,
                       pt.tenant_id
                FROM p_party AS pp
                JOIN p_party_tenant AS pt ON pt.party_id = pp.party_id
                JOIN p_tenant AS t ON t.tenant_id = pt.tenant_id
                WHERE pp.entra_object_id = %s
                  AND COALESCE(pp.is_active, TRUE) = TRUE
                  AND pp.deleted_at IS NULL
                  AND COALESCE(pt.is_active, TRUE) = TRUE
                  AND pt.status = 'ACTIVE'
                  AND COALESCE(t.is_active, TRUE) = TRUE
                  AND t.status = 'ACTIVE'
                  AND pp.party_type IN ('DOCTOR', 'NURSE', 'OFFICE_ADMIN')
                ORDER BY pt.tenant_id
                LIMIT 2
            """).format(role_col=role_expr)
            cur.execute(query, (object_id,))
            rows = cur.fetchall()
    except EntraConfigurationError:
        raise
    except psycopg.Error as exc:
        raise EntraConfigurationError("Unable to query the PCES identity mapping") from exc

    if not rows:
        raise EntraIdentityError("No active PCES identity mapping exists")
    if len(rows) != 1:
        raise EntraIdentityError("Identity has multiple active tenant memberships; explicit tenant selection is required")
    party_id, party_type, pces_role, active_tenant_id = rows[0]
    return {"party_id": str(party_id), "party_type": party_type,
            "pces_role": pces_role, "tenant_id": str(active_tenant_id)}
