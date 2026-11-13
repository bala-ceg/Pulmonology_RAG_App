"""PCES Admin onboarding: hospital users, Microsoft Graph provisioning and notification."""
from __future__ import annotations
import secrets, string, smtplib, ssl
from email.message import EmailMessage
from uuid import UUID, uuid4
import msal, psycopg, requests
from config import Config

class OnboardingError(RuntimeError):
    pass

def _connect():
    """Use the same canonical EHR connection as routes.disciplines._ehr_conn.

    User creation/update/onboarding must never silently fall back to DB_*,
    because list/search endpoints read the EHR database via PG_TOOL_*.
    """
    cfg = Config
    kw = {
        "host": cfg.PG_TOOL_HOST,
        "port": cfg.PG_TOOL_PORT,
        "dbname": cfg.PG_TOOL_NAME,
        "user": cfg.PG_TOOL_USER,
        "password": cfg.PG_TOOL_PASSWORD,
        "connect_timeout": 10,
        "options": "-c search_path=ehr_ccm_schema,public",
    }
    if not all(kw.get(k) for k in ("host", "dbname", "user", "password")):
        raise OnboardingError(
            "EHR database connection is not configured: PG_TOOL_HOST, "
            "PG_TOOL_NAME, PG_TOOL_USER and PG_TOOL_PASSWORD are required"
        )
    return psycopg.connect(**kw)

def _uuid(value, label):
    try: return str(UUID(str(value)))
    except (TypeError, ValueError, AttributeError) as exc: raise OnboardingError("Invalid " + label) from exc

def _admin(cur, party_id):
    cur.execute("""SELECT 1 FROM p_party WHERE party_id=%s AND party_type='PCES_ADMIN'
                   AND UPPER(COALESCE(pces_role,''))='PCES_ADMIN'
                   AND COALESCE(is_active,TRUE)=TRUE AND deleted_at IS NULL""", (party_id,))
    if not cur.fetchone(): raise OnboardingError("PCES_ADMIN access required")

def list_hospitals(admin_id):
    admin_id=_uuid(admin_id, "PCES Admin party ID")
    with _connect() as conn, conn.cursor() as cur:
        _admin(cur, admin_id)
        # Use the canonical hospital registry relationship. A tenant alone is
        # not necessarily a hospital; the organization party supplies the
        # hospital identity and location shown by the existing UI.
        cur.execute("""SELECT p.party_id,p.name,p.hospital_location,t.tenant_id,t.tenant_code
                       FROM ehr_ccm_schema.p_party p
                       JOIN ehr_ccm_schema.p_tenant t
                         ON t.organization_party_id=p.party_id
                       WHERE p.party_type='ORGANIZATION'
                         AND COALESCE(p.is_active,TRUE)=TRUE
                         AND COALESCE(t.is_active,TRUE)=TRUE
                         AND t.status='ACTIVE'
                       ORDER BY p.name ASC NULLS LAST,p.party_id""")
        return [{"party_id":str(r[0]),"hospital_name":r[1] or "",
                 "location_id":r[2] or "","tenant_id":str(r[3]),
                 "tenant_code":r[4] or ""} for r in cur.fetchall()]

def list_users(admin_id, tenant_id, limit=20, offset=0):
    tenant_id=_uuid(tenant_id,"hospital ID"); limit=max(1,min(int(limit),100)); offset=max(0,int(offset))
    with _connect() as conn, conn.cursor() as cur:
        _admin(cur, _uuid(admin_id, 'PCES Admin party ID'))
        cur.execute("""SELECT p.party_id,p.first_name,p.middle_name,p.last_name,p.party_type,p.pces_role,
                              p.email,p.username,p.entra_object_id,pt.on_board,pt.on_board_date,p.created_at
                       FROM p_party p JOIN p_party_tenant pt ON pt.party_id=p.party_id
                       WHERE pt.tenant_id=%s AND pt.status='ACTIVE' AND COALESCE(pt.is_active,TRUE)=TRUE
                         AND COALESCE(p.is_active,TRUE)=TRUE
                         AND p.party_type IN ('DOCTOR','NURSE','OFFICE_ADMIN','PCES_ADMIN')
                       ORDER BY pt.on_board_date DESC NULLS LAST,p.created_at DESC LIMIT %s OFFSET %s""",
                    (tenant_id,limit,offset))
        rows=cur.fetchall()
    return [{"party_id":str(r[0]),"first_name":r[1] or "","middle_name":r[2] or "",
             "last_name":r[3] or "","party_type":r[4],"pces_role":r[5] or "",
             "email":r[6] or "","username":r[7] or "","entra_id":r[8] or "",
             "on_boarded":bool(r[9]),"on_board_date":r[10].isoformat() if r[10] else None,
             "created_at":r[11].isoformat() if r[11] else None} for r in rows]

def search_users(admin_id,tenant_id,last_name="",party_type="",email=""):
    tenant_id=_uuid(tenant_id,"hospital ID"); party_type=party_type.strip().upper()
    if party_type and party_type not in {"DOCTOR","NURSE","OFFICE_ADMIN","PCES_ADMIN"}: raise OnboardingError("Invalid party type")
    clauses=["pt.tenant_id=%s","pt.status='ACTIVE'","COALESCE(pt.is_active,TRUE)=TRUE",
             "COALESCE(p.is_active,TRUE)=TRUE","p.deleted_at IS NULL","p.party_type IN ('DOCTOR','NURSE','OFFICE_ADMIN','PCES_ADMIN')"]
    params=[tenant_id]
    for col,val in (("p.last_name",last_name),("p.party_type",party_type),("p.email",email)):
        if val.strip(): clauses.append("LOWER(COALESCE("+col+",'')) LIKE LOWER(%s)"); params.append("%"+val.strip()+"%")
    with _connect() as conn, conn.cursor() as cur:
        _admin(cur, _uuid(admin_id, "PCES Admin party ID"))
        cur.execute("""SELECT p.party_id,p.first_name,p.middle_name,p.last_name,p.party_type,p.pces_role,
                              p.email,p.username,p.entra_object_id,pt.on_board,pt.on_board_date
                       FROM p_party p JOIN p_party_tenant pt ON pt.party_id=p.party_id
                       WHERE """+" AND ".join(clauses)+" ORDER BY pt.on_board_date DESC NULLS LAST,p.created_at DESC LIMIT 100",tuple(params))
        rows=cur.fetchall()
    return [{"party_id":str(r[0]),"first_name":r[1] or "","middle_name":r[2] or "",
             "last_name":r[3] or "","party_type":r[4],"pces_role":r[5] or "",
             "email":r[6] or "","username":r[7] or "","entra_id":r[8] or "",
             "on_boarded":bool(r[9]),"on_board_date":r[10].isoformat() if r[10] else None} for r in rows]

def create_user(admin_id,tenant_id,data):
    admin_id=_uuid(admin_id,"PCES Admin party ID"); tenant_id=_uuid(tenant_id,"hospital ID")
    first=str(data.get("first_name") or "").strip(); middle=str(data.get("middle_name") or "").strip()
    last=str(data.get("last_name") or "").strip(); role=str(data.get("pces_role") or "").strip()
    email=str(data.get("email") or "").strip(); username=str(data.get("username") or "").strip()
    kind=str(data.get("party_type") or "").strip().upper()
    if not all((first,last,role,email,username)): raise OnboardingError("First name, last name, PCES role, email and username are required")
    if kind not in {"DOCTOR","NURSE"}: raise OnboardingError("Only DOCTOR and NURSE users can be onboarded")
    party_id=str(uuid4()); membership_id=str(uuid4())
    with _connect() as conn, conn.cursor() as cur:
        _admin(cur,admin_id)
        cur.execute("SELECT 1 FROM p_tenant WHERE tenant_id=%s AND status='ACTIVE' AND COALESCE(is_active,TRUE)=TRUE",(tenant_id,))
        if not cur.fetchone(): raise OnboardingError("Selected hospital is not active")
        cur.execute("""INSERT INTO p_party
          (party_id,party_type,name,first_name,middle_name,last_name,email,username,pces_role,
           is_active,created_at,created_by,version_no)
          VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,TRUE,NOW(),%s,1)""",
          (party_id,kind," ".join(x for x in (first,middle,last) if x),first,middle or None,last,email,username,role,admin_id))
        cur.execute("""INSERT INTO p_party_tenant
          (party_tenant_id,party_id,tenant_id,status,created_at,created_by,is_active,version_no,on_board,on_board_date)
          VALUES (%s,%s,%s,'ACTIVE',NOW(),%s,TRUE,1,FALSE,NULL)""",(membership_id,party_id,tenant_id,admin_id))
    return {"party_id":party_id,"party_tenant_id":membership_id,"tenant_id":tenant_id,"on_boarded":False}

def update_user(admin_id,party_id,tenant_id,data):
    admin_id=_uuid(admin_id,"PCES Admin party ID")
    party_id=_uuid(party_id,"party ID")
    tenant_id=_uuid(tenant_id,"hospital ID")

    allowed={"first_name","middle_name","last_name","party_type","pces_role","email","username"}
    if not data or set(data)-allowed:
        raise OnboardingError("Unsupported user fields were supplied")

    values={}
    for key in allowed:
        if key in data:
            values[key]=str(data[key] or "").strip()
    for key in ("first_name","last_name","party_type","pces_role","email","username"):
        if key in values and not values[key]:
            raise OnboardingError(key.replace("_"," ").title()+" is required")
    if "party_type" in values and values["party_type"] not in {"DOCTOR","NURSE"}:
        raise OnboardingError("Only DOCTOR and NURSE users can be updated")

    columns={"first_name":"first_name","middle_name":"middle_name","last_name":"last_name",
             "party_type":"party_type","pces_role":"pces_role","email":"email","username":"username"}
    sets=[]
    vals=[]
    for key in columns:
        if key in values:
            sets.append(columns[key]+"=%s")
            vals.append(values[key] or None)

    with _connect() as conn, conn.cursor() as cur:
        _admin(cur,admin_id)
        cur.execute("""SELECT 1 FROM p_party_tenant
                       WHERE party_id=%s AND tenant_id=%s AND status='ACTIVE'
                         AND COALESCE(is_active,TRUE)=TRUE""",(party_id,tenant_id))
        if not cur.fetchone():
            raise OnboardingError("Doctor/Nurse is not a member of the selected hospital")

        for field, label in (("username", "User Name"), ("email", "Email")):
            if field in values:
                cur.execute("SELECT 1 FROM p_party WHERE party_id<>%s AND LOWER("+
                            field+")=LOWER(%s) LIMIT 1",(party_id,values[field]))
                if cur.fetchone():
                    raise OnboardingError(label+" already exists")

        cur.execute("UPDATE p_party SET "+",".join(sets)+
                    ",updated_at=NOW(),updated_by=%s,version_no=COALESCE(version_no,1)+1 "+
                    "WHERE party_id=%s AND party_type IN ('DOCTOR','NURSE') RETURNING party_id",
                    tuple(vals+[admin_id,party_id]))
        if not cur.fetchone():
            raise OnboardingError("Doctor/Nurse record was not found")
    return {"party_id":party_id,"updated_fields":sorted(values)}

def _temp_password():
    alphabet=string.ascii_letters+string.digits
    while True:
        value="".join(secrets.choice(alphabet) for _ in range(16))
        if any(c.isupper() for c in value) and any(c.islower() for c in value) and any(c.isdigit() for c in value): return value

def _graph_headers():
    cfg=Config
    if not all((cfg.PCES_PROVISIONING_CLIENT_ID,cfg.PCES_PROVISIONING_CLIENT_SECRET,cfg.PCES_DIRECTORY_TENANT_ID,cfg.PCES_DIRECTORY_DOMAIN)):
        raise OnboardingError("Microsoft Graph provisioning configuration is incomplete")
    app=msal.ConfidentialClientApplication(cfg.PCES_PROVISIONING_CLIENT_ID,
        authority="https://login.microsoftonline.com/"+cfg.PCES_DIRECTORY_TENANT_ID,
        client_credential=cfg.PCES_PROVISIONING_CLIENT_SECRET)
    token=app.acquire_token_for_client(scopes=["https://graph.microsoft.com/.default"])
    access=token.get("access_token")
    if not access: raise OnboardingError("Unable to obtain Microsoft Graph provisioning token")
    return {"Authorization":"Bearer "+access,"Content-Type":"application/json"}

def _graph_error(response, action):
    if response.status_code == 403:
        return OnboardingError(
            action + " was denied by Microsoft Graph. Check the provisioning application's "
            "permissions and admin consent in the External ID directory"
        )
    if response.status_code == 404:
        return OnboardingError("The linked Entra account was not found in the configured directory")
    return OnboardingError(action + " failed: HTTP " + str(response.status_code))

def provision_entra_user(user):
    cfg=Config
    headers=_graph_headers()
    password=_temp_password(); email=user["email"]
    payload={"accountEnabled":True,"displayName":" ".join(x for x in (user["first_name"],user.get("middle_name"),user["last_name"]) if x),
      "givenName":user["first_name"],"surname":user["last_name"],"mail":email,
      "identities":[{"signInType":"emailAddress","issuer":cfg.PCES_DIRECTORY_DOMAIN,"issuerAssignedId":email}],
      "passwordProfile":{"password":password,"forceChangePasswordNextSignIn":True},
      "passwordPolicies":"DisablePasswordExpiration"}
    response=requests.post("https://graph.microsoft.com/v1.0/users",
      headers=headers,json=payload,timeout=30)
    if response.status_code!=201: raise _graph_error(response,"Microsoft Entra provisioning")
    oid=response.json().get("id")
    if not oid: raise OnboardingError("Microsoft Entra did not return an object ID")
    return str(UUID(oid)),password

def reset_entra_password(object_id, email):
    headers = _graph_headers()
    url = "https://graph.microsoft.com/v1.0/users/" + _uuid(object_id, "Entra object ID")
    response = requests.get(url, headers=headers, params={"$select":"id,identities"}, timeout=30)
    if response.status_code != 200:
        raise _graph_error(response, "Microsoft Entra account lookup")
    identities = response.json().get("identities") or []
    if not any(
        identity.get("signInType") == "emailAddress"
        and str(identity.get("issuer") or "").lower() == Config.PCES_DIRECTORY_DOMAIN.lower()
        and str(identity.get("issuerAssignedId") or "").lower() == email.lower()
        for identity in identities
    ):
        raise OnboardingError(
            "The linked account is not an email/password customer account for this email "
            "in the configured directory. Verify its Microsoft sign-in identity before resetting"
        )
    password = _temp_password()
    response = requests.patch(
        url, headers=headers,
        json={"passwordProfile":{"password":password,"forceChangePasswordNextSignIn":True}},
        timeout=30,
    )
    if response.status_code != 204:
        raise _graph_error(response, "Microsoft Entra password reset")
    return password

def _email_settings():
    cfg = Config
    host = (getattr(cfg, "PCES_SMTP_HOST", None) or "").strip()
    username_smtp = (getattr(cfg, "PCES_SMTP_USERNAME", None) or "").strip()
    password_smtp = getattr(cfg, "PCES_SMTP_PASSWORD", None)
    port = int(getattr(cfg, "PCES_SMTP_PORT", 465))
    timeout = int(getattr(cfg, "PCES_SMTP_TIMEOUT", 30))

    if not all((host, username_smtp, password_smtp)):
        raise OnboardingError(
            "Bluehost SMTP configuration is incomplete; set PCES_SMTP_HOST, "
            "PCES_SMTP_USERNAME, and PCES_SMTP_PASSWORD"
        )
    if not Config.PCES_SIGNIN_URL or Config.PCES_SIGNIN_URL == "https://pces.example.com":
        raise OnboardingError("Set PCES_SIGNIN_URL to the actual PCES sign-in URL before onboarding")
    return host, username_smtp, password_smtp, port, timeout

def _send_email(email, username, password):
    """Send PCES onboarding instructions through the configured Bluehost mailbox."""
    cfg = Config
    host, username_smtp, password_smtp, port, timeout = _email_settings()
    if not email or not username or not password:
        raise OnboardingError("Recipient email and onboarding credentials are required")

    message = EmailMessage()
    message["From"] = username_smtp
    message["To"] = email
    message["Subject"] = "Your PCES account is ready"
    message.set_content(
        "Your PCES account has been created.\n\n"
        "Microsoft sign-in email: " + email + "\n"
        "PCES username: " + username + "\n"
        "Temporary password: " + password + "\n"
        "Sign-in URL: " + cfg.PCES_SIGNIN_URL + "\n\n"
        "Choose Microsoft sign-in in PCES and use the Microsoft sign-in email above, "
        "not your PCES username. Change your temporary password when prompted."
    )

    try:
        with smtplib.SMTP_SSL(
            host=host,
            port=port,
            context=ssl.create_default_context(),
            timeout=timeout,
        ) as smtp:
            smtp.login(username_smtp, password_smtp)
            refused = smtp.send_message(message)
            if refused:
                raise OnboardingError("Bluehost SMTP rejected one or more recipients")
    except OnboardingError:
        raise
    except (OSError, smtplib.SMTPException) as exc:
        # Do not include credentials, message bodies, or SMTP authentication data.
        raise OnboardingError("Unable to send PCES onboarding email through Bluehost SMTP") from exc

def onboard_user(admin_id,party_id,tenant_id,reset_password=False):
    admin_id=_uuid(admin_id,"PCES Admin party ID"); party_id=_uuid(party_id,"party ID"); tenant_id=_uuid(tenant_id,"hospital ID")
    with _connect() as conn, conn.cursor() as cur:
        _admin(cur,admin_id)
        cur.execute("""SELECT p.first_name,p.middle_name,p.last_name,p.email,p.username,p.entra_object_id,p.party_type,pt.party_tenant_id
                       FROM p_party p JOIN p_party_tenant pt ON pt.party_id=p.party_id
                       WHERE p.party_id=%s AND pt.tenant_id=%s AND pt.status='ACTIVE'
                         AND COALESCE(pt.is_active,TRUE)=TRUE AND p.party_type IN ('DOCTOR','NURSE','OFFICE_ADMIN','PCES_ADMIN')
                         AND COALESCE(p.is_active,TRUE)=TRUE FOR UPDATE OF p,pt""",(party_id,tenant_id))
        row=cur.fetchone()
        if not row: raise OnboardingError("Doctor/Nurse record was not found")
        first,middle,last,email,username,oid,kind,party_tenant_id=row
        if not email or not username: raise OnboardingError("Email and username are required before onboarding")
        if oid and not reset_password:
            cur.execute("""UPDATE p_party_tenant
                           SET on_board=TRUE,updated_at=NOW(),updated_by=%s,
                               version_no=COALESCE(version_no,1)+1
                           WHERE party_tenant_id=%s
                           RETURNING on_board_date""",(admin_id,party_tenant_id))
            repaired=cur.fetchone()
            if not repaired:
                raise OnboardingError("Entra ID is linked, but the hospital membership could not be reconciled")
            return {"party_id":party_id,"party_tenant_id":str(party_tenant_id),
                    "tenant_id":tenant_id,"entra_object_id":str(oid),
                    "on_boarded":True,"on_board_date":repaired[0].isoformat() if repaired[0] else None,"email_sent":None,
                    "message":"Existing Entra account linked; PCES onboarding status reconciled. No password was reset or emailed."}
        _email_settings()
        existing_account=bool(oid)
        # Keep the row lock until the mapping is saved so concurrent clicks cannot
        # create multiple Microsoft accounts for the same PCES user.
        if existing_account:
            password=reset_entra_password(oid,email)
        else:
            oid,password=provision_entra_user({"first_name":first,"middle_name":middle,"last_name":last,"email":email})
        try:
            if not existing_account:
                cur.execute("""UPDATE p_party SET entra_object_id=%s,updated_at=NOW(),updated_by=%s,
                               version_no=COALESCE(version_no,1)+1
                               WHERE party_id=%s AND entra_object_id IS NULL RETURNING party_id""",(oid,admin_id,party_id))
                if not cur.fetchone(): raise OnboardingError("Entra account created but PCES mapping save failed; reconcile before retrying")
            cur.execute("""UPDATE p_party_tenant SET on_board=TRUE,on_board_date=COALESCE(on_board_date,CURRENT_DATE),
                           updated_at=NOW(),updated_by=%s,version_no=COALESCE(version_no,1)+1
                           WHERE party_tenant_id=%s RETURNING on_board_date""",(admin_id,party_tenant_id))
            membership=cur.fetchone()
            if not membership:
                raise OnboardingError("Entra credentials changed but PCES membership save failed; reconcile before retrying")
            conn.commit()
        except psycopg.Error as exc:
            raise OnboardingError(
                "Microsoft credentials were created or reset, but the PCES database save failed. "
                "Reconcile the Microsoft account before retrying"
            ) from exc
    email_error=None
    try:
        _send_email(email,username,password)
        email_sent=True
    except OnboardingError as exc:
        email_sent=False
        email_error=str(exc)
    except Exception:
        email_sent=False
        email_error="Unable to deliver the onboarding email"
    action="Existing Entra account password reset" if existing_account else "Entra account provisioned"
    return {"party_id":party_id,"party_tenant_id":str(party_tenant_id),"tenant_id":tenant_id,"entra_object_id":oid,"on_boarded":True,
            "on_board_date":membership[0].isoformat() if membership[0] else None,
            "email_sent":email_sent,"email_error":email_error,
            "message":action+"; temporary-password email sent" if email_sent else
                      action+"; "+email_error+". After correcting email delivery, use On Board again "
                      "to reset the password and send a new temporary password. The old password will be invalidated."}
