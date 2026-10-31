-- Pending Entra self-registration and hospital-scoped review.
SET search_path TO ehr_ccm_schema, public;

CREATE TABLE IF NOT EXISTS entra_signup_requests (
    request_id UUID PRIMARY KEY,
    entra_tenant_id TEXT NOT NULL,
    entra_object_id UUID NOT NULL,
    requested_tenant_id UUID NOT NULL,
    requested_party_type VARCHAR(30) NOT NULL
        CHECK (requested_party_type IN ('DOCTOR', 'NURSE')),
    first_name VARCHAR(255) NOT NULL,
    middle_name VARCHAR(255),
    last_name VARCHAR(255) NOT NULL,
    email VARCHAR(320),
    phone VARCHAR(50),
    date_of_birth DATE,
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING'
        CHECK (status IN ('PENDING', 'APPROVED', 'REJECTED')),
    reviewed_by UUID,
    reviewed_at TIMESTAMP WITHOUT TIME ZONE,
    rejection_reason TEXT,
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITHOUT TIME ZONE
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_entra_signup_pending_identity_hospital
    ON entra_signup_requests (LOWER(entra_tenant_id), entra_object_id, requested_tenant_id)
    WHERE status = 'PENDING';

CREATE INDEX IF NOT EXISTS ix_entra_signup_requests_hospital_status
    ON entra_signup_requests (requested_tenant_id, status, created_at DESC);

CREATE INDEX IF NOT EXISTS ix_entra_signup_requests_identity
    ON entra_signup_requests (LOWER(entra_tenant_id), entra_object_id, created_at DESC);
