-- Migration: Add per-user allowed_departments restriction to pces_users
-- NULL means no restriction (user can see all departments).
-- A comma-separated value restricts the user to those departments only.

ALTER TABLE pces_users ADD COLUMN IF NOT EXISTS allowed_departments TEXT DEFAULT NULL;

-- Upsert Shruti: allowed to access Cardiology and General Surgery only,
-- with Cardiology as the default (first entry).
INSERT INTO pces_users (username, password_hash, pces_role, first_name, last_name, allowed_departments)
VALUES ('shruti', 'shruti123', 'CARDIOLOGIST', 'Shruti', '', 'Cardiology,General Surgery')
ON CONFLICT (username) DO UPDATE
    SET allowed_departments = EXCLUDED.allowed_departments;
