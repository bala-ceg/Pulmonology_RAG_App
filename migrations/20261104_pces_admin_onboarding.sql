-- PCES Admin onboarding tracking belongs to each hospital membership.
-- The service reads and updates these fields on p_party_tenant, not p_party.
SET search_path TO ehr_ccm_schema, public;

ALTER TABLE p_party_tenant
    ADD COLUMN IF NOT EXISTS on_board BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS on_board_date DATE;

CREATE INDEX IF NOT EXISTS ix_p_party_tenant_onboarding
    ON p_party_tenant (tenant_id, party_id);
