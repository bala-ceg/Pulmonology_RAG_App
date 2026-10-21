# Microsoft Entra External ID integration — Phases 1–5

## Scope and current migration status

This branch adds a first-pass server-side authorization-code flow and trusted identity lookup. It is intentionally **not production-ready until the app registration, database migration, tenant-selection design, and route-by-route authorization audit are completed**. The existing password login remains in place during the staged migration; do not assume that adding Entra routes disables or secures legacy login.

## Phase 1 — Discovery and preparation

Reviewed on branch `bala/2026-10-20` (base: `PREPROD`):

- Flask application bootstraps in `main.py`; blueprints are registered in `routes/__init__.py`.
- Existing PCES login is `POST /api/login` in `routes/disciplines.py`; it checks username/password against `p_party`, then validates `p_party_tenant` and `p_tenant`.
- Current session identity is `session["pces_identity"]` with `party_id` and `tenant_id`.
- `GET /api/tenants` has overlapping route definitions in `routes/disciplines.py`; review and resolve before using it for identity/tenant selection.
- `login_page/server.js` is a separate Express login endpoint. Deployment topology must be confirmed; it must not be considered protected by Flask session middleware.
- The repository does not include a verified live database schema or Entra app registration details. This implementation does not infer account linkage from email or display name.

## Phase 2 — Entra app configuration

Set these environment variables in the deployment secret store (never commit real values):

- `ENTRA_CLIENT_ID`
- `ENTRA_CLIENT_SECRET`
- `ENTRA_AUTHORITY` — authority URL for the configured External ID tenant/user flow
- `ENTRA_REDIRECT_URI` — exact HTTPS callback URI, ending in `/auth/entra/callback`

Register the same redirect URI in the Entra application. The authority format and user-flow policy must match the External ID setup chosen by the identity administrator. Use HTTPS outside local development.

Install the dependency with `pip install -r requirements.txt`. This branch adds `msal`.

## Phase 3 — Sign-in/callback/logout

Routes in `routes/entra_auth.py`:

- `GET /auth/entra/login` begins the MSAL authorization-code flow.
- `GET /auth/entra/callback` validates one-time OAuth state, exchanges the code, requires `tid` and `oid` claims, and establishes `pces_identity` only after trusted database mapping succeeds.
- `POST /auth/entra/logout` clears the local session.
- `GET /auth/entra/status` returns minimal status only.

Provider-wide logout, PKCE/state behavior against the actual External ID user flow, session cookie settings, CSRF controls, and end-to-end callback behavior still require environment-backed testing. No access or ID tokens are returned to the browser by these endpoints.

## Phase 4 — Identity mapping

A database administrator must review the schema and apply an approved migration equivalent to:

```sql
ALTER TABLE p_party
  ADD COLUMN IF NOT EXISTS entra_tenant_id TEXT,
  ADD COLUMN IF NOT EXISTS entra_object_id TEXT;

CREATE UNIQUE INDEX IF NOT EXISTS uq_p_party_entra_identity
  ON p_party (entra_tenant_id, entra_object_id)
  WHERE entra_tenant_id IS NOT NULL AND entra_object_id IS NOT NULL;
```

Before applying, confirm the actual schema/search path and whether the table is schema-qualified in this deployment. Populate the mapping only from an independently verified directory object ID. Never automatically link by email, UPN, or display name. Accounts without an explicit mapping are denied.

The current resolver requires exactly one active hospital membership. Users with multiple active memberships are denied until a deliberate hospital-selection step is implemented and validated against the server-side membership table.

## Phase 5 — Hospital and role authorization

The resolver requires an active `p_party` record and exactly one active `p_party_tenant` / `p_tenant` membership. It stores the resolved party and tenant in the Flask session, not from untrusted callback parameters.

This phase does not complete the application-wide authorization audit. Before rollout:

1. Inventory every route that reads/writes patients, notes, uploads, vector stores, admin functions, training data, and clinical summaries.
2. Add server-side authentication and tenant/role checks to every protected route; UI flags and `sessionStorage` are not security boundaries.
3. Remove or separately secure the Express login service if it remains deployed.
4. Resolve the duplicate `/api/tenants` endpoint definitions and define a tenant-selection flow for multi-hospital users.
5. Test anonymous access, unmapped identities, inactive accounts, inactive memberships, cross-tenant patient IDs, role escalation, forged state, missing callback code, and expired sessions.

## Current limitations / rollout blockers

- No Entra tenant, client ID, secret, authority, redirect URI, or live EHR schema was provided; sign-in cannot be exercised here.
- The legacy `/api/login` route remains enabled for backwards compatibility during phased migration.
- The resolver assumes `p_party` has `pces_role` or `role`; verify the deployed schema and adjust the query if neither column exists.
- Route-level tenant authorization is not globally enforced by this change.
- No automated tests or production sign-in test have been run from this connector.
