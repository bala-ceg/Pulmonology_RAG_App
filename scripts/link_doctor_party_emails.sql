-- =============================================================
-- Link pces_users doctors to p_party UUIDs via email match
-- Run as: postgres or any superuser on pces_ehr_ccm
-- Command: psql -h 4.155.102.23 -U postgres -d pces_ehr_ccm -f link_doctor_party_emails.sql
-- =============================================================

BEGIN;

-- dr.shruti.cardio  →  545dc7ed-04e0-4abf-afa1-8e54e56b45e1
UPDATE public.p_party
SET email = 'Shruti.Malee.cardio@example.com', updated_at = NOW()
WHERE party_id = '545dc7ed-04e0-4abf-afa1-8e54e56b45e1' AND party_type = 'DOCTOR';

-- dr.avinash.family  →  95e4eaec-453a-487d-9aa4-f45edcf25021
UPDATE public.p_party
SET email = 'Avinash.Javvaji.family@example.com', updated_at = NOW()
WHERE party_id = '95e4eaec-453a-487d-9aa4-f45edcf25021' AND party_type = 'DOCTOR';

-- dr.brown.surgeon  →  4b3f9c1c-334e-4298-b901-abfaafc639b5
UPDATE public.p_party
SET email = 'olivia.brown.surgeon@example.com', updated_at = NOW()
WHERE party_id = '4b3f9c1c-334e-4298-b901-abfaafc639b5' AND party_type = 'DOCTOR';

-- dr.shalini.dent  →  19a5314e-f17e-45be-b204-6cdfe9e8ea28
UPDATE public.p_party
SET email = 'Shalini.Tharimana.dent@example.com', updated_at = NOW()
WHERE party_id = '19a5314e-f17e-45be-b204-6cdfe9e8ea28' AND party_type = 'DOCTOR';

-- dr.wilson.neuro  →  ef04a141-a6ad-452b-a20a-ca1cf18aefef
UPDATE public.p_party
SET email = 'david.wilson.neuro@example.com', updated_at = NOW()
WHERE party_id = 'ef04a141-a6ad-452b-a20a-ca1cf18aefef' AND party_type = 'DOCTOR';

-- dr.moore.eye  →  414b37b3-9c01-4899-8723-64de6a96317f
UPDATE public.p_party
SET email = 'james.moore.eye@example.com', updated_at = NOW()
WHERE party_id = '414b37b3-9c01-4899-8723-64de6a96317f' AND party_type = 'DOCTOR';

-- dr. Kim Naidu  →  37839ffd-802e-4a87-966d-0822c8100536
UPDATE public.p_party
SET email = 'kimnaidu@gamil.com', updated_at = NOW()
WHERE party_id = '37839ffd-802e-4a87-966d-0822c8100536' AND party_type = 'DOCTOR';

-- Allow pcesuser to update email/updated_at going forward
GRANT UPDATE (email, updated_at) ON public.p_party TO pcesuser;

COMMIT;

-- Verify: show all 7 doctors now have matching emails
SELECT party_id, first_name, last_name, email, updated_at
FROM public.p_party
WHERE party_type = 'DOCTOR'
  AND email ILIKE '%example.com'
   OR (party_type = 'DOCTOR' AND email = 'kimnaidu@gamil.com')
ORDER BY last_name, first_name;
