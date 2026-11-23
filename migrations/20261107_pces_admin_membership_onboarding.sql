-- Forward-compatible correction for databases that already ran the original
-- onboarding migration, which placed these flags on p_party by mistake.
SET search_path TO ehr_ccm_schema, public;

ALTER TABLE p_party_tenant
    ADD COLUMN IF NOT EXISTS on_board BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS on_board_date DATE;

CREATE INDEX IF NOT EXISTS ix_p_party_tenant_onboarding
    ON p_party_tenant (tenant_id, party_id);
