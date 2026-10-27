"""Trusted mapping from Microsoft Entra identity to an active PCES account.

A first-time mapping is permitted only when an administrator explicitly
allowlists the verified Entra object ID to an existing PCES party ID in
ENTRA_PCES_PROVISIONING_MAP. Email and display name are never used for linking.
"""
from __future__ import annotations

import json
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
    """Extract the expected directory tenant GUID from the configured authority."""
    authority = Config.ENTRA_AUTHORITY
    if not authority:
        raise EntraConfigurationError("Microsoft Entra authority is not configured")
    path_parts = [part for part in urlparse(authority).path.split("/") if part]
    if not path_parts:
        raise EntraConfigurationError(
            "The configured Entra authority must include the tenant identifier"
        )
    return path_parts[-1].strip().lower()


def _validate_tenant(tenant_id: str) -> None:
    if not tenant_id or tenant_id.strip().lower() != _configured_tenant_id():
        raise EntraIdentityError("Identity was issued by an unexpected Entra tenant")


def provision_allowlisted_identity(tenant_id: str, object_id: str) -> bool:
    """Persist a first-time mapping only for an administrator-approved OID->party ID.

    Configure ENTRA_PCES_PROVISIONING_MAP as JSON:
    {"<verified-Entra-object-id>": "<existing-PCES-party-id>"}
    This mapping must be provisioned by an administrator through a trusted channel.
    Returns True if a mapping was created, False if this OID is not allowlisted.
    """
    if not tenant_id or not object_id:
        raise EntraIdentityError("Required directory identity claims are missing")
    _validate_tenant(tenant_id)

    raw_map = getattr(Config, "ENTRA_PCES_PROVISIONING_MAP", None) or "{}"
    try:
        allowlist = json.loads(raw_map)
        if not isinstance(allowlist, dict):
            raise ValueError("expected JSON object")
        normalized = {str(k).strip().lower(): str(v).strip() for k, v in allowlist.items()}
        party_id = normalized.get(object_id.strip().lower())
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise EntraConfigurationError(
            "ENTRA_PCES_PROVISIONING_MAP must be a JSON object mapping OIDs to party IDs"
        ) from exc

    if not party_id:
        return False

    try:
        object_id = str(UUID(object_id))
        party_id = str(UUID(party_id))
    except ValueError as exc:
        raise EntraConfigurationError(
            "Provisioning map contains an invalid Object ID or party ID"
        ) from exc

    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT party_id, party_type, entra_object_id, is_active, deleted_at
                FROM p_party
                WHERE party_id = %s
                FOR UPDATE
                """,
                (party_id,),
            )
            party = cur.fetchone()
            if not party:
                raise EntraIdentityError("Allowlisted PCES party does not exist")

            existing_party_id, party_type, existing_oid, is_active, deleted_at = party
            if deleted_at is not None or is_active is False:
                raise EntraIdentityError("Allowlisted PCES party is inactive or deleted")
            if party_type not in ("DOCTOR", "NURSE", "OFFICE_ADMIN"):
                raise EntraIdentityError("PCES party type is not eligible for Entra access")
            if existing_oid and str(existing_oid).lower() != object_id.lower():
                raise EntraIdentityError("PCES party already has a different Entra mapping")

            cur.execute(
                """
                SELECT party_id
                FROM p_party
                WHERE LOWER(entra_object_id) = LOWER(%s)
                  AND party_id <> %s
                LIMIT 1
                """,
                (object_id, party_id),
            )
            if cur.fetchone():
                raise EntraIdentityError("Entra Object ID is already mapped to another PCES party")

            cur.execute(
                """
                SELECT pt.tenant_id
                FROM p_party_tenant AS pt
                JOIN p_tenant AS t ON t.tenant_id = pt.tenant_id
                WHERE pt.party_id = %s
                  AND COALESCE(pt.is_active, TRUE) = TRUE
                  AND pt.status = 'ACTIVE'
                  AND COALESCE(t.is_active, TRUE) = TRUE
                  AND t.status = 'ACTIVE'
                """,
                (party_id,),
            )
            memberships = cur.fetchall()
            if len(memberships) != 1:
                raise EntraIdentityError(
                    "Provisioning requires exactly one active PCES tenant membership"
                )

            if existing_oid and str(existing_oid).lower() == object_id.lower():
                return True

            cur.execute(
                """
                UPDATE p_party
                SET entra_object_id = %s
                WHERE party_id = %s
                  AND entra_object_id IS NULL
                  AND deleted_at IS NULL
                  AND is_active = TRUE
                RETURNING party_id
                """,
                (object_id, party_id),
            )
            if not cur.fetchone():
                raise EntraIdentityError("PCES identity mapping changed during provisioning")
        return True
    except (EntraConfigurationError, EntraIdentityError):
        raise
    except psycopg.Error as exc:
        raise EntraConfigurationError("Unable to provision the PCES identity mapping") from exc


def resolve_pces_identity(tenant_id: str, object_id: str) -> dict:
    """Resolve a verified identity to exactly one active PCES tenant membership."""
    if not tenant_id or not object_id:
        raise EntraIdentityError("Required directory identity claims are missing")
    _validate_tenant(tenant_id)

    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT LOWER(column_name)
                FROM information_schema.columns
                WHERE table_schema = current_schema() AND table_name = 'p_party'
                """
            )
            columns = {row[0] for row in cur.fetchall()}
            role_column = next(
                (name for name in ("pces_role", "role", "specialty", "department") if name in columns),
                None,
            )
            if not {"party_id", "party_type", "entra_object_id"} <= columns:
                raise EntraConfigurationError(
                    "p_party is missing required identity mapping columns"
                )
            role_expr = sql.Identifier(role_column) if role_column else sql.Identifier("party_type")
            query = sql.SQL(
                """
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
                """
            ).format(role_col=role_expr)
            cur.execute(query, (object_id,))
            rows = cur.fetchall()
    except EntraConfigurationError:
        raise
    except psycopg.Error as exc:
        raise EntraConfigurationError("Unable to query the PCES identity mapping") from exc

    if not rows:
        raise EntraIdentityError("No active PCES identity mapping exists")
    if len(rows) != 1:
        raise EntraIdentityError(
            "Identity has multiple active tenant memberships; explicit tenant selection is required"
        )

    party_id, party_type, pces_role, active_tenant_id = rows[0]
    return {
        "party_id": str(party_id),
        "party_type": party_type,
        "pces_role": pces_role,
        "tenant_id": str(active_tenant_id),
    }
