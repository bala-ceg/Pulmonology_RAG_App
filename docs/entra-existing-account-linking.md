# PCES Microsoft Entra External ID account linking

## First sign-in for an existing PCES user

1. The user signs in with Microsoft Entra External ID. PCES uses the ID-token `tid` and `oid`; it does not use Entra email or display name alone to select a PCES account.
2. If that Object ID is not mapped, PCES asks for the email address already verified in PCES and the user's active hospital.
3. PCES creates a random six-digit code, stores only an HMAC digest, and emails the code only when exactly one eligible PCES party has that previously verified email and active hospital membership.
4. The user submits the code. PCES validates the session-bound challenge, Entra tenant/object IDs, expiry, one-time-use status, attempt limit, and current account/hospital status.
5. In a database transaction, PCES links the Entra Object ID to that party. Future sign-ins resolve by Object ID and continue to enforce active party and tenant-membership checks.

Unknown, unverified, ambiguous, inactive, or already-conflicting account matches do not receive an email code. The request page returns a generic confirmation to reduce account enumeration. The code expires after 10 minutes and permits at most five verification attempts. Requests are limited to three per email-hash in a rolling 15-minute window. SMTP delivery must be configured.

## Required configuration

Configure these environment variables through your deployment's secret manager; do not commit credentials:

- `SMTP_HOST`
- `SMTP_PORT` (defaults to 587)
- `SMTP_FROM_EMAIL`
- `SMTP_USERNAME` (optional if relay authentication is not required)
- `SMTP_PASSWORD` (optional if relay authentication is not required)
- `SMTP_USE_TLS` (defaults to true for STARTTLS)
- `SMTP_USE_SSL` (defaults to false; use SSL on the port expected by your provider)

Use a strong, persistent `SECRET_KEY` across application restarts. It protects the HMAC used for email and code digests. Configure the SMTP relay to require TLS and restrict sending to the approved sender.

## Database migration and trusted email enrollment

Apply `migrations/20261027_entra_email_verification.sql` to the PCES EHR database. It explicitly selects `ehr_ccm_schema`, adds `p_party.email_verified_at`, creates the challenge table, and enforces uniqueness of non-empty Entra Object IDs.

**The migration deliberately leaves existing email addresses unverified.** Do not mark all current rows as verified automatically. Before enabling this flow, populate `email_verified_at` only for addresses confirmed through a trusted pre-existing onboarding process, administrator verification, or a separate verified-email process. If PCES has no trustworthy source for existing email verification, use an administrator-assisted enrollment process to verify those accounts first. An email merely being present in `p_party` is not sufficient.

Apply the migration with the intended database role and inspect the schema before production. Check for duplicate Entra OIDs before creating the unique index; resolve any duplicates manually rather than deleting mappings blindly.

## Deployment and tests

1. Review changes on branch `bala/2026-10-20`.
2. Configure SMTP secrets and a persistent `SECRET_KEY`.
3. Apply the migration to a test database first.
4. Test with a non-production Entra identity and PCES account whose email has been verified through a trusted process.
5. Verify success, wrong code, expired code, exhausted attempts, request throttling, unknown/unverified email, inactive membership, conflicting OID, and repeat sign-in.
6. Confirm no raw verification code or email is written to application logs. Remove expired challenge rows with a scheduled cleanup job after confirming the retention policy.

## Security notes and limitations

- OAuth state and a session-bound CSRF token protect the flow. Use HTTPS and secure session-cookie settings in deployment.
- Codes are random, short-lived, one-time, HMAC-digested, and attempt-limited. Challenge rows should be periodically cleaned up.
- Account matching uses the email in PCES only after `email_verified_at` is populated by a trusted process; Entra email alone is not used as proof of account ownership.
- This feature does not fix PCES's existing password storage/authentication behavior. The legacy `/api/login` code currently compares submitted passwords directly with stored values; migrate to a standard password hash (for example Argon2id or scrypt) separately before production.
- The application has not been deployed to the VM or integration-tested against the live database as part of these code changes.
