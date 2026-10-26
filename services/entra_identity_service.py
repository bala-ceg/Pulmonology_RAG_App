"""Trusted mapping from Microsoft Entra identity to an active PCES account.

This deployment uses one configured Entra External ID tenant. The token tenant
ID is checked against the configured authority, while the existing
p_party.entra_object_id column maps the user to a PCES party. Email and display
name are never used for account linking.
"""
from __future__ import annotations

from urllib.parse import urlparse

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


def resolve_pces_identity(tenant_id: str, object_id: str) -> dict:
    """Resolve a verified identity in the configured Entra tenant to one active PCES membership.

    The database stores only entra_object_id. Because this deployment is
    configured for a single Entra tenant, the incoming tid claim is checked
    against the tenant identifier in ENTRA_AUTHORITY before querying by oid.
    """
    if not tenant_id or not object_id:
        raise EntraIdentityError("Required directory identity claims are missing")

    expected_tenant_id = _configured_tenant_id()
    if tenant_id.strip().lower() != expected_tenant_id:
        raise EntraIdentityError("Identity was issued by an unexpected Entra tenant")

    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT LOWER(column_name)
                FROM information_schema.columns
                WHERE table_name = 'p_party'
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
