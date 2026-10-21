"""Trusted mapping from Microsoft Entra directory identity to an active PCES account.

The migration in docs/entra-external-id-phases-1-5.md adds the immutable
directory tenant/object identifiers. Email and display-name matching are never
used for account linking.
"""
from __future__ import annotations

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


def resolve_pces_identity(tenant_id: str, object_id: str) -> dict:
    """Resolve a directory tenant + object ID to exactly one active PCES membership."""
    if not tenant_id or not object_id:
        raise EntraIdentityError("Required directory identity claims are missing")

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
            if not {"party_id", "party_type", "entra_tenant_id", "entra_object_id"} <= columns:
                raise EntraConfigurationError(
                    "p_party is missing required Entra identity mapping columns"
                )
            role_expr = sql.Identifier(role_column) if role_column else sql.Identifier("party_type")
            query = sql.SQL(
                """
                SELECT pp.party_id, pp.party_type, COALESCE({role_col}, pp.party_type) AS pces_role,
                       pt.tenant_id
                FROM p_party AS pp
                JOIN p_party_tenant AS pt ON pt.party_id = pp.party_id
                JOIN p_tenant AS t ON t.tenant_id = pt.tenant_id
                WHERE pp.entra_tenant_id = %s
                  AND pp.entra_object_id = %s
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
            cur.execute(query, (tenant_id, object_id))
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
