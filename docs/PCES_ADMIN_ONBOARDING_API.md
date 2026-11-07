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

## Bluehost SMTP onboarding email

Microsoft Entra External ID remains responsible for customer-user creation and sign-in. Flask sends onboarding notifications through the Bluehost mailbox `noreply@pcesglobal.ai`; Microsoft Graph `Mail.Send` is not used and is not required for this mailbox.

Configure these server-side environment variables on the Azure VM running Flask. Never commit secrets:

- `PCES_SMTP_HOST=pcesglobal.ai`
- `PCES_SMTP_PORT=465` (implicit TLS / SMTP_SSL)
- `PCES_SMTP_USERNAME=noreply@pcesglobal.ai`
- `PCES_SMTP_PASSWORD` (Bluehost mailbox password)
- `PCES_SMTP_TIMEOUT=30` (optional; defaults to 30 seconds)
- `PCES_SIGNIN_URL` (the actual PCES sign-in URL)

The SMTP sender address is the configured `PCES_SMTP_USERNAME`. These `PCES_SMTP_*` settings are deliberately separate from the existing `SMTP_*` settings used by Entra account-link verification.

The standalone VM smoke test may additionally use `PCES_TEST_RECIPIENT`; this variable is not needed by the Flask application. Run the test from the Azure VM, then confirm that the message arrives in the recipient inbox or spam folder. An SMTP success response means the server accepted the message; it does not by itself prove inbox delivery.

The onboarding sequence remains: create the Entra user, persist its Entra Object ID in `p_party`, mark the selected `p_party_tenant` membership onboarded, and then attempt SMTP delivery. Email delivery failure is non-fatal: the API reports `email_sent=false` while retaining the successful Entra and database provisioning. The temporary password is generated during provisioning and is not persisted for a later resend, so an email failure requires an administrative recovery/reset process rather than blindly repeating user creation.

The current development flow emails a temporary password. This is security-sensitive and must not be used in production; replace it with a one-time activation/password-reset link before release. Never log the password or SMTP credentials.

## Tests

Run the focused tests in the project environment:

~~~bash
pytest -q tests/test_pces_admin_onboarding.py
~~~

Route tests mock service boundaries and do not create Microsoft accounts. A separate development-tenant integration test is still required to validate Graph permissions for External ID customer creation, first sign-in, forced password change, and actual Bluehost SMTP delivery from the Azure VM.

## Operational caveat

Entra user creation and PostgreSQL updates cannot be one distributed transaction. If Graph creates the user but saving the returned OID fails, reconcile the Entra object manually before retrying; do not blindly retry or create a duplicate. The current service reports this partial-failure state. Graph is used for Entra user provisioning only; onboarding email delivery uses Bluehost SMTP.

The current development flow emails a temporary password. This is security-sensitive and must not be used in production; replace it with a one-time activation/password-reset link before release. Never log the password, Graph access token, or full Graph request payload.
