"""PCES Admin-only onboarding API."""
from __future__ import annotations
from flask import Blueprint, current_app, jsonify, request, session
from services.pces_admin_onboarding import OnboardingError, create_user, list_hospitals, list_users, onboard_user, search_users, update_user

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
    try: return jsonify({"success":True,"hospitals":list_hospitals()})
    except Exception:
        current_app.logger.exception("Unable to list hospitals")
        return jsonify({"success":False,"error":"Unable to load hospitals"}),503

@pces_admin_bp.get("/users")
def users():
    if not _admin_id(): return _denied()
    try:
        items=list_users(request.args.get("tenant_id",""),request.args.get("limit",20,type=int),request.args.get("offset",0,type=int))
        return jsonify({"success":True,"users":items,"count":len(items)})
    except OnboardingError as exc: return jsonify({"success":False,"error":str(exc)}),400
    except Exception:
        current_app.logger.exception("Unable to list users")
        return jsonify({"success":False,"error":"Unable to load users"}),503

@pces_admin_bp.get("/users/search")
def users_search():
    if not _admin_id(): return _denied()
    try:
        items=search_users(request.args.get("tenant_id",""),request.args.get("last_name",""),request.args.get("party_type",""),request.args.get("email",""))
        return jsonify({"success":True,"users":items,"count":len(items)})
    except OnboardingError as exc: return jsonify({"success":False,"error":str(exc)}),400
    except Exception:
        current_app.logger.exception("Unable to search users")
        return jsonify({"success":False,"error":"Unable to search users"}),503

@pces_admin_bp.post("/users")
def users_create():
    if not _admin_id(): return _denied()
    data=request.get_json(silent=True) or {}
    tenant_id=data.pop("tenant_id","")
    try: return jsonify({"success":True,**create_user(_admin_id(),tenant_id,data)}),201
    except OnboardingError as exc: return jsonify({"success":False,"error":str(exc)}),400
    except Exception:
        current_app.logger.exception("Unable to create user")
        return jsonify({"success":False,"error":"Unable to create user"}),503

@pces_admin_bp.patch("/users/<party_id>")
def users_update(party_id):
    if not _admin_id(): return _denied()
    data=request.get_json(silent=True) or {}
    try: return jsonify({"success":True,**update_user(_admin_id(),party_id,data)})
    except OnboardingError as exc: return jsonify({"success":False,"error":str(exc)}),400
    except Exception:
        current_app.logger.exception("Unable to update user")
        return jsonify({"success":False,"error":"Unable to update user"}),503

@pces_admin_bp.post("/users/<party_id>/onboard")
def users_onboard(party_id):
    if not _admin_id(): return _denied()
    try: return jsonify({"success":True,**onboard_user(_admin_id(),party_id)})
    except OnboardingError as exc: return jsonify({"success":False,"error":str(exc)}),400
    except Exception:
        current_app.logger.exception("Unable to onboard user")
        return jsonify({"success":False,"error":"Unable to complete onboarding"}),503
