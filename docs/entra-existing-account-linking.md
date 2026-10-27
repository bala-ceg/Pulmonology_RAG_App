# PCES Microsoft Entra External ID account linking

## First sign-in for an existing PCES user

1. The user signs in with Microsoft Entra External ID. PCES uses the validated ID-token `tid` and `oid`; it does not use email or display name to choose a PCES account.
2. If that Object ID is not mapped, PCES presents the existing-account linking form.
3. The user supplies their existing PCES username/password and selects the hospital assigned to that PCES account.
4. PCES validates those credentials against the existing `p_party` login columns and verifies an active membership in `p_party_tenant` / `p_tenant`.
5. Within a database transaction, PCES links the verified Entra Object ID to that party. Future sign-ins resolve by Object ID and continue to enforce active party, role, and tenant-membership checks.

No per-user environment variable is required. The identity is linked once in the database.

## Deployment

1. Review the account-linking changes on branch `bala/2026-10-20`.
2. Apply `migrations/20261027_entra_identity_linking.sql` to the EHR database with `ehr_ccm_schema` on the PostgreSQL search path. If the unique index fails, investigate existing duplicate Entra Object IDs before retrying.
3. Ensure the app's normal `/api/tenants` endpoint is available to the link page.
4. Deploy/restart the Flask application and test using a non-production PCES account first.
5. Verify that a first sign-in links exactly one account; a second sign-in skips the linking form; wrong PCES credentials, inactive memberships, and already-linked conflicting identities are rejected.

## Security notes and limitations

- The OAuth callback's state is validated and the linking form uses a session-bound CSRF token.
- The account is selected by existing PCES credentials, not by a client-supplied party ID or email match.
- The current PCES `/api/login` implementation compares the supplied password directly with the database value. The linking flow intentionally mirrors that existing behavior for compatibility. This is not a safe password-storage design. Before production, migrate to a standard password-hashing scheme (for example, Argon2id or scrypt) and use one shared password-verification function for both PCES login and Entra linking.
- Add rate limiting / lockout controls to both password-verification endpoints before exposing them publicly.
- The flow proves possession of the existing PCES credentials; it is not a substitute for administrator onboarding policy, MFA, or recovery controls. Do not enable it for accounts whose credentials are shared or unverified.
- The application has not been deployed to the VM or integration-tested against the live database as part of this change.
