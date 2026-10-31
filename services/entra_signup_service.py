"""Pending Entra signup requests and hospital-scoped approval.

New identities are not PCES users until an authorized hospital Office Admin
approves the request. This module deliberately does not mark email verified.
"""
from __future__ import annotations

from datetime import date
from uuid import UUID, uuid4

import psycopg

from config import Config


class SignupError(RuntimeError):
    """Signup request or approval could not be completed."""


def _connect():
    kwargs = {
        "host": Config.PG_TOOL_HOST or Config.DB_HOST,
        "port": Config.PG_TOOL_PORT or Config.DB_PORT or "5432",
        "dbname": Config.PG_TOOL_NAME or Config.DB_NAME,
        "user": Config.PG_TOOL_USER or Config.DB_USER,
        "password": Config.PG_TOOL_PASSWORD or Config.DB_PASSWORD,
        "connect_timeout": Config.DB_CONNECT_TIMEOUT,
        "options": "-c search_path=ehr_ccm_schema,public",
    }
    if not all(kwargs.get(k) for k in ("host", "dbname", "user", "password")):
        raise SignupError("EHR database connection is not configured")
    return psycopg.connect(**kwargs)


def _uuid(value: str, label: str) -> str:
    try:
        return str(UUID(str(value)))
    except (ValueError, TypeError, AttributeError) as exc:
        raise SignupError(f"Invalid {label}") from exc


def _validate_issuer(issuer: str) -> str:
    from services.entra_identity_service import _validate_tenant
    _validate_tenant(issuer)
    return issuer.strip().lower()


def active_hospitals() -> list[dict]:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT tenant_id, tenant_name, tenant_code
            FROM p_tenant
            WHERE COALESCE(is_active, TRUE) = TRUE
              AND status = 'ACTIVE'
            ORDER BY tenant_name
        """)
        return [
            {"tenant_id": str(r[0]), "tenant_name": r[1], "tenant_code": r[2]}
            for r in cur.fetchall()
        ]


def submit_request(issuer: str, oid: str, data: dict) -> str:
    issuer = _validate_issuer(issuer)
    oid = _uuid(oid, "Entra object identifier")
    tenant_id = _uuid(data.get("tenant_id"), "hospital")
    party_type = str(data.get("party_type") or "").upper()
    if party_type not in {"DOCTOR", "NURSE"}:
        raise SignupError("Only Doctor and Nurse self-registration is allowed")
    first = str(data.get("first_name") or "").strip()
    last = str(data.get("last_name") or "").strip()
    if not first or not last:
        raise SignupError("First name and last name are required")
    dob_raw = str(data.get("date_of_birth") or "").strip()
    try:
        dob = date.fromisoformat(dob_raw) if dob_raw else None
    except ValueError as exc:
        raise SignupError("Date of birth must be YYYY-MM-DD") from exc
    email = str(data.get("email") or "").strip() or None
    phone = str(data.get("phone") or "").strip() or None

    request_id = str(uuid4())
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT 1 FROM p_tenant
            WHERE tenant_id = %s AND COALESCE(is_active, TRUE) = TRUE
              AND status = 'ACTIVE'
        """, (tenant_id,))
        if not cur.fetchone():
            raise SignupError("Select an active hospital")
        cur.execute("""
            SELECT 1 FROM p_party
            WHERE LOWER(COALESCE(entra_object_id, '')) = LOWER(%s)
            LIMIT 1
        """, (oid,))
        if cur.fetchone():
            raise SignupError("This Microsoft account is already linked to a PCES identity")
        cur.execute("""
            INSERT INTO entra_signup_requests (
                request_id, entra_tenant_id, entra_object_id,
                requested_tenant_id, requested_party_type,
                first_name, middle_name, last_name, email, phone,
                date_of_birth, status
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'PENDING')
            ON CONFLICT DO NOTHING
            RETURNING request_id
        """, (
            request_id, issuer, oid, tenant_id, party_type, first,
            str(data.get("middle_name") or "").strip() or None,
            last, email, phone, dob,
        ))
        row = cur.fetchone()
        if not row:
            cur.execute("""
                SELECT request_id FROM entra_signup_requests
                WHERE LOWER(entra_tenant_id) = LOWER(%s)
                  AND entra_object_id = %s
                  AND requested_tenant_id = %s
                  AND status = 'PENDING'
                ORDER BY created_at DESC LIMIT 1
            """, (issuer, oid, tenant_id))
            row = cur.fetchone()
        if not row:
            raise SignupError("A signup request could not be created")
        return str(row[0])


def _require_office_admin(cur, approver_id: str, tenant_id: str) -> None:
    cur.execute("""
        SELECT 1
        FROM p_party p
        JOIN p_party_tenant pt ON pt.party_id = p.party_id
        JOIN p_tenant t ON t.tenant_id = pt.tenant_id
        WHERE p.party_id = %s
          AND UPPER(p.party_type) = 'OFFICE_ADMIN'
          AND COALESCE(p.is_active, TRUE) = TRUE
          AND p.deleted_at IS NULL
          AND pt.tenant_id = %s
          AND pt.status = 'ACTIVE'
          AND COALESCE(pt.is_active, TRUE) = TRUE
          AND t.status = 'ACTIVE'
          AND COALESCE(t.is_active, TRUE) = TRUE
        LIMIT 1
        FOR UPDATE OF p, pt
    """, (approver_id, tenant_id))
    if not cur.fetchone():
        raise SignupError("You are not an active Office Admin for this hospital")


def list_requests(approver_id: str, tenant_id: str) -> list[dict]:
    approver_id = _uuid(approver_id, "approver")
    tenant_id = _uuid(tenant_id, "hospital")
    with _connect() as conn, conn.cursor() as cur:
        _require_office_admin(cur, approver_id, tenant_id)
        cur.execute("""
            SELECT request_id, requested_party_type, first_name, middle_name,
                   last_name, email, phone, date_of_birth, status, created_at
            FROM entra_signup_requests
            WHERE requested_tenant_id = %s
            ORDER BY CASE WHEN status = 'PENDING' THEN 0 ELSE 1 END,
                     created_at DESC
        """, (tenant_id,))
        return [{
            "request_id": str(r[0]), "party_type": r[1], "first_name": r[2],
            "middle_name": r[3] or "", "last_name": r[4], "email": r[5] or "",
            "phone": r[6] or "", "date_of_birth": r[7].isoformat() if r[7] else None,
            "status": r[8], "created_at": r[9].isoformat() if r[9] else None,
        } for r in cur.fetchall()]


def review_request(approver_id: str, tenant_id: str, request_id: str,
                   decision: str, rejection_reason: str | None = None) -> dict:
    approver_id = _uuid(approver_id, "approver")
    tenant_id = _uuid(tenant_id, "hospital")
    request_id = _uuid(request_id, "request")
    decision = decision.upper()
    if decision not in {"APPROVE", "REJECT"}:
        raise SignupError("Decision must be APPROVE or REJECT")

    with _connect() as conn, conn.cursor() as cur:
        _require_office_admin(cur, approver_id, tenant_id)
        cur.execute("""
            SELECT entra_tenant_id, entra_object_id, requested_party_type,
                   first_name, middle_name, last_name, email, phone, date_of_birth
            FROM entra_signup_requests
            WHERE request_id = %s AND requested_tenant_id = %s
              AND status = 'PENDING'
            FOR UPDATE
        """, (request_id, tenant_id))
        row = cur.fetchone()
        if not row:
            raise SignupError("Pending signup request was not found")
        if decision == "REJECT":
            reason = (rejection_reason or "").strip()[:2000] or None
            cur.execute("""
                UPDATE entra_signup_requests
                SET status = 'REJECTED', reviewed_by = %s,
                    reviewed_at = NOW(), rejection_reason = %s, updated_at = NOW()
                WHERE request_id = %s AND status = 'PENDING'
            """, (approver_id, reason, request_id))
            return {"status": "REJECTED", "party_id": None}

        issuer, oid, role, first, middle, last, email, phone, dob = row
        cur.execute("""
            SELECT 1 FROM p_party
            WHERE LOWER(COALESCE(entra_object_id, '')) = LOWER(%s)
            LIMIT 1
        """, (str(oid),))
        if cur.fetchone():
            raise SignupError("This Entra identity is already linked to a PCES account")
        cur.execute("""
            SELECT 1 FROM p_tenant
            WHERE tenant_id = %s AND status = 'ACTIVE'
              AND COALESCE(is_active, TRUE) = TRUE
        """, (tenant_id,))
        if not cur.fetchone():
            raise SignupError("The selected hospital is no longer active")

        party_id, membership_id = str(uuid4()), str(uuid4())
        full_name = " ".join(x for x in (first, middle, last) if x)
        cur.execute("""
            INSERT INTO p_party (
                party_id, party_type, name, first_name, middle_name, last_name,
                date_of_birth, phone, email, entra_object_id,
                is_active, created_at, created_by, version_no
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                      TRUE, NOW(), %s, 1)
        """, (
            party_id, role, full_name, first, middle, last, dob, phone,
            email, str(oid), approver_id,
        ))
        cur.execute("""
            INSERT INTO p_party_tenant (
                party_tenant_id, party_id, tenant_id, status,
                created_at, created_by, is_active, version_no
            ) VALUES (%s, %s, %s, 'ACTIVE', NOW(), %s, TRUE, 1)
        """, (membership_id, party_id, tenant_id, approver_id))
        cur.execute("""
            UPDATE entra_signup_requests
            SET status = 'APPROVED', reviewed_by = %s,
                reviewed_at = NOW(), updated_at = NOW()
            WHERE request_id = %s AND status = 'PENDING'
        """, (approver_id, request_id))
        return {"status": "APPROVED", "party_id": party_id}
