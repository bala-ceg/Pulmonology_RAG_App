"""PCES Admin-only onboarding API."""
from __future__ import annotations
from flask import Blueprint, current_app, jsonify, request, session
from services.pces_admin_onboarding import OnboardingError, UsernameConflict, check_username, create_user, delete_user, list_hospitals, list_users, onboard_user, pending_onboarding_users, search_users, update_user

pces_admin_bp=Blueprint("pces_admin_bp",__name__,url_prefix="/api/pces-admin")

def _admin_id():
    identity=session.get("pces_identity")
    if not isinstance(identity,dict): return None
    if str(identity.get("party_type") or "").upper()!="PCES_ADMIN": return None
    if str(identity.get("pces_role") or "").upper()!="PCES_ADMIN": return None
    return identity.get("party_id")

def _denied(): return jsonify({"success":False,"error":"PCES_ADMIN access required"}),403

@pces_admin_bp.get("/hospitals")
def hospitals():
    if not _admin_id(): return _denied()
    try: return jsonify({"success":True,"hospitals":list_hospitals(_admin_id())})
    except Exception:
        current_app.logger.exception("Unable to list hospitals")
        return jsonify({"success":False,"error":"Unable to load hospitals"}),503

@pces_admin_bp.get("/users")
def users():
    if not _admin_id(): return _denied()
    try:
        items=list_users(_admin_id(),request.args.get("tenant_id",""),request.args.get("limit",20,type=int),request.args.get("offset",0,type=int))
        return jsonify({"success":True,"users":items,"count":len(items)})
    except OnboardingError as exc: return jsonify({"success":False,"error":str(exc)}),400
    except Exception:
        current_app.logger.exception("Unable to list users")
        return jsonify({"success":False,"error":"Unable to load users"}),503

@pces_admin_bp.get("/users/search")
def users_search():
    if not _admin_id(): return _denied()
    try:
        items=search_users(_admin_id(),request.args.get("tenant_id",""),request.args.get("last_name",""),request.args.get("party_type",""),request.args.get("email",""))
        return jsonify({"success":True,"users":items,"count":len(items)})
    except OnboardingError as exc: return jsonify({"success":False,"error":str(exc)}),400
    except Exception:
        current_app.logger.exception("Unable to search users")
        return jsonify({"success":False,"error":"Unable to search users"}),503

@pces_admin_bp.get("/users/check-username")
def users_check_username():
    if not _admin_id(): return _denied()
    a=request.args
    try: return jsonify({"success":True,**check_username(_admin_id(),a.get("tenant_id",""),a.get("username",""),a.get("party_id") or None,a.get("first_name",""),a.get("last_name",""),a.get("party_type",""))})
    except OnboardingError as exc: return jsonify({"success":False,"error":str(exc)}),400
    except Exception:
        current_app.logger.exception("Unable to check username")
        return jsonify({"success":False,"error":"Unable to check username"}),503

@pces_admin_bp.post("/users")
def users_create():
    if not _admin_id(): return _denied()
    data=request.get_json(silent=True) or {}
    tenant_id=data.pop("tenant_id","")
    try: return jsonify({"success":True,**create_user(_admin_id(),tenant_id,data)}),201
    except UsernameConflict as exc: return jsonify({"success":False,"error":str(exc),"suggestions":exc.suggestions}),409
    except OnboardingError as exc: return jsonify({"success":False,"error":str(exc)}),400
    except Exception:
        current_app.logger.exception("Unable to create user")
        return jsonify({"success":False,"error":"Unable to create user"}),503

@pces_admin_bp.patch("/users/<party_id>")
def users_update(party_id):
    if not _admin_id(): return _denied()
    data=request.get_json(silent=True) or {}
    tenant_id=data.pop("tenant_id","")
    try: return jsonify({"success":True,**update_user(_admin_id(),party_id,tenant_id,data)})
    except UsernameConflict as exc: return jsonify({"success":False,"error":str(exc),"suggestions":exc.suggestions}),409
    except OnboardingError as exc: return jsonify({"success":False,"error":str(exc)}),400
    except Exception:
        current_app.logger.exception("Unable to update user")
        return jsonify({"success":False,"error":"Unable to update user"}),503

@pces_admin_bp.post("/users/<party_id>/onboard")
def users_onboard(party_id):
    if not _admin_id(): return _denied()
    data=request.get_json(silent=True) or {}
    tenant_id=data.get("tenant_id") or request.args.get("tenant_id","")
    reset_password=data.get("reset_password",False)
    if not isinstance(reset_password,bool):
        return jsonify({"success":False,"error":"reset_password must be a boolean"}),400
    only_pending=data.get("only_pending",False)
    if not isinstance(only_pending,bool):
        return jsonify({"success":False,"error":"only_pending must be a boolean"}),400
    try:
        options={"reset_password":reset_password}
        if only_pending: options["only_pending"]=True
        return jsonify({"success":True,**onboard_user(_admin_id(),party_id,tenant_id,**options)})
    except OnboardingError as exc: return jsonify({"success":False,"error":str(exc)}),400
    except Exception:
        current_app.logger.exception("Unable to onboard user")
        return jsonify({"success":False,"error":"Unable to complete onboarding"}),503

@pces_admin_bp.get("/users/pending-onboarding")
def users_pending_onboarding():
    if not _admin_id(): return _denied()
    try:
        return jsonify({"success":True,"users":pending_onboarding_users(_admin_id(),request.args.get("tenant_id",""))})
    except OnboardingError as exc:
        return jsonify({"success":False,"error":str(exc)}),400
    except Exception:
        current_app.logger.exception("Unable to load pending onboarding users")
        return jsonify({"success":False,"error":"Unable to load pending users"}),503

@pces_admin_bp.delete("/users/<party_id>")
def users_delete(party_id):
    if not _admin_id(): return _denied()
    tenant_id=request.args.get("tenant_id") or (request.get_json(silent=True) or {}).get("tenant_id","")
    try: return jsonify({"success":True,**delete_user(_admin_id(),party_id,tenant_id)})
    except OnboardingError as exc: return jsonify({"success":False,"error":str(exc)}),400
    except Exception:
        current_app.logger.exception("Unable to delete user")
        return jsonify({"success":False,"error":"Unable to delete user"}),503
