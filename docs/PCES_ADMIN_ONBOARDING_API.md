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
| POST | /api/pces-admin/users/<party_id>/onboard | Provision Entra identity for the selected hospital, save OID, mark that hospital membership onboarded, email instructions |

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

The create operation inserts p_party and p_party_tenant in one database transaction. New hospital membership rows have p_party_tenant.on_board=false until Entra provisioning completes. The onboarding endpoint requires tenant_id in its JSON body (or query string) so the correct hospital membership is updated.

## Database migration

The database administrator has added p_party_tenant.on_board (BOOLEAN, default FALSE) and p_party_tenant.on_board_date (DATE). The service reads and updates these membership-level fields; no p_party onboarding columns are required.

## Microsoft Graph configuration

Configure these server-side environment variables; never commit secrets:

- PCES_PROVISIONING_CLIENT_ID
- PCES_PROVISIONING_CLIENT_SECRET
- PCES_DIRECTORY_TENANT_ID
- PCES_DIRECTORY_DOMAIN
- PCES_SIGNIN_URL

The provisioning application needs Microsoft Graph application permissions and admin consent to create External ID customer users. Validate the identities issuer/domain against the actual External ID directory configuration.

## Microsoft Graph onboarding email

Onboarding notifications are sent with Microsoft Graph using the same confidential client application configured for Entra user provisioning. Configure these server-side environment variables; never commit secrets:

- PCES_PROVISIONING_CLIENT_ID
- PCES_PROVISIONING_CLIENT_SECRET
- PCES_DIRECTORY_TENANT_ID
- PCES_ONBOARDING_FROM_EMAIL (the approved, provisioned sender mailbox)
- PCES_SIGNIN_URL

The application requires Microsoft Graph application permission Mail.Send with administrator consent. The sender mailbox must be approved for application sending; scope mailbox access with Exchange Online Application RBAC where supported by the tenant. The VM does not need a local SMTP service for this flow.

The current development flow emails a temporary password. This is security-sensitive and must not be used in production; replace it with a one-time activation/password-reset link before release. Never log the password, Graph access token, or full Graph request payload.

## Tests

Run the focused tests in the project environment:

~~~bash
pytest -q tests/test_pces_admin_onboarding.py
~~~

Route tests mock service boundaries and do not create Microsoft accounts. A separate development-tenant integration test is still required to validate Graph permissions, External ID customer creation, first sign-in, and forced password change.

## Operational caveat

Entra user creation and PostgreSQL updates cannot be one distributed transaction. If Graph creates the user but saving the returned OID fails, reconcile the Entra object manually before retrying; do not blindly retry or create a duplicate. The current service reports this partial-failure state.

The current development flow emails a temporary password. This is security-sensitive and must not be used in production; replace it with a one-time activation/password-reset link before release. Never log the password, Graph access token, or full Graph request payload.
