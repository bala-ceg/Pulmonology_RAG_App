# PCES Admin onboarding API

This API implements the PCES Admin onboarding workflow in the approved onboarding specification.

## Authentication and authorization

- Sign in through the existing /api/login endpoint using the centrally provisioned PCES Admin username and password.
- A PCES_ADMIN account is global and does not require a hospital membership at login.
- Every onboarding operation checks the PCES Admin role in the signed Flask session and revalidates the active PCES Admin record in p_party.
- The development administrator record supplied for testing is PCES_ADMIN_001. Do not place its password or hash in source control or tests.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | /api/pces-admin/hospitals | List active hospitals; read-only |
| GET | /api/pces-admin/users?tenant_id=<uuid> | List users; default page size 20 |
| GET | /api/pces-admin/users/search?tenant_id=<uuid>&last_name=&party_type=&email= | Search hospital users |
| POST | /api/pces-admin/users | Insert Doctor/Nurse and hospital membership |
| PATCH | /api/pces-admin/users/<party_id> | Update only email and/or username |
| POST | /api/pces-admin/users/<party_id>/onboard | Provision Entra identity, save OID, mark onboarded, email instructions |

## Create user request

~~~json
{
  "tenant_id": "hospital-uuid",
  "first_name": "Cody",
  "middle_name": "Gerry",
  "last_name": "Pearson",
  "party_type": "DOCTOR",
  "pces_role": "Cardiology",
  "email": "cody@example.org",
  "username": "Dr.Cody.Cardio"
}
~~~

The create operation inserts p_party and p_party_tenant in one database transaction. New records have on_board=false until Entra provisioning completes.

## Database migration

Apply migrations/20261104_pces_admin_onboarding.sql to pces_ehr_ccm before using list/create/onboard operations. It adds p_party.on_board and p_party.on_board_date.

## Microsoft Graph configuration

Configure these server-side environment variables; never commit secrets:

- PCES_PROVISIONING_CLIENT_ID
- PCES_PROVISIONING_CLIENT_SECRET
- PCES_DIRECTORY_TENANT_ID
- PCES_DIRECTORY_DOMAIN
- PCES_SIGNIN_URL

The provisioning application needs Microsoft Graph application permissions and admin consent to create External ID customer users. Validate the identities issuer/domain against the actual External ID directory configuration.

## SMTP configuration

The onboarding email uses the existing SMTP settings: SMTP_HOST, SMTP_PORT, SMTP_FROM_EMAIL, SMTP_USERNAME, SMTP_PASSWORD, SMTP_USE_TLS.

## Tests

Run the focused tests in the project environment:

~~~bash
pytest -q tests/test_pces_admin_onboarding.py
~~~

Route tests mock service boundaries and do not create Microsoft accounts. A separate development-tenant integration test is still required to validate Graph permissions, External ID customer creation, first sign-in, and forced password change.

## Operational caveat

Entra user creation and PostgreSQL updates cannot be one distributed transaction. If Graph creates the user but saving the returned OID fails, reconcile the Entra object manually before retrying; do not blindly retry or create a duplicate. The current service reports this partial-failure state.

The requested workflow emails a temporary password. This is security-sensitive: before production, consider a one-time activation/password-reset link instead of sending a reusable credential by email. Never log the password, Graph access token, or full Graph request payload.
