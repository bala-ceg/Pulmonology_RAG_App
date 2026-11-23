-- Run against the PCES EHR database with ehr_ccm_schema on search_path.
-- If this fails because duplicate non-empty Object IDs already exist, resolve
-- the duplicate mappings manually before rerunning; do not delete identities blindly.
CREATE UNIQUE INDEX IF NOT EXISTS uq_p_party_entra_object_id_lower
    ON p_party (LOWER(entra_object_id))
    WHERE entra_object_id IS NOT NULL AND BTRIM(entra_object_id) <> '';
