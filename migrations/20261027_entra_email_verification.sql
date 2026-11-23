-- Apply to the PCES EHR database as a migration owner.
-- The schema is explicit so unqualified service queries resolve consistently.
SET search_path TO ehr_ccm_schema, public;

-- Existing email addresses are NOT presumed verified. A trusted existing
-- onboarding/verification process must populate email_verified_at per account.
ALTER TABLE p_party
    ADD COLUMN IF NOT EXISTS email_verified_at TIMESTAMPTZ NULL;

CREATE TABLE IF NOT EXISTS entra_email_link_challenges (
    challenge_id UUID PRIMARY KEY,
    party_id UUID NULL,
    tenant_id UUID NULL,
    entra_tenant_id TEXT NOT NULL,
    entra_object_id UUID NOT NULL,
    email_hash TEXT NOT NULL,
    code_hash TEXT NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    consumed_at TIMESTAMPTZ NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_entra_email_link_challenges_email_created
    ON entra_email_link_challenges (email_hash, created_at DESC);

CREATE INDEX IF NOT EXISTS ix_entra_email_link_challenges_expiry
    ON entra_email_link_challenges (expires_at);

-- Enforce one Entra OID per PCES party even when case differs.
CREATE UNIQUE INDEX IF NOT EXISTS uq_p_party_entra_object_id_lower
    ON p_party (LOWER(entra_object_id))
    WHERE entra_object_id IS NOT NULL AND BTRIM(entra_object_id) <> '';
