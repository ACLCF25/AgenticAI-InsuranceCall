"""
Supabase-backed authentication helpers and profile management routes.

The app now treats Supabase Auth as the credential source of truth while
retaining direct Postgres access for application data and role metadata.
"""

import os
from functools import wraps
from typing import Any, Optional

import psycopg2
import psycopg2.extras
import psycopg2.pool
import requests
from flask import Blueprint, g, jsonify, request

auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")

SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").rstrip("/")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

ROLE_RANK = {
    "agent": 1,
    "admin": 2,
    "super_admin": 3,
}
VALID_ROLES = tuple(ROLE_RANK.keys())
VALID_APPROVAL_STATUSES = ("pending", "approved", "rejected")

_auth_pool = psycopg2.pool.ThreadedConnectionPool(
    minconn=1,
    maxconn=10,
    host=os.getenv("SUPABASE_HOST"),
    database="postgres",
    user=os.getenv("SUPABASE_USER", "postgres"),
    password=os.getenv("SUPABASE_PASSWORD"),
    port=5432,
)


def _get_conn():
    return _auth_pool.getconn()


def _query_one(sql: Any, params: tuple):
    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        _auth_pool.putconn(conn)


def _query_all(sql: Any, params: tuple = ()):
    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            return [dict(r) for r in rows]
    finally:
        _auth_pool.putconn(conn)


def _execute(sql: Any, params: tuple = ()):
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
        conn.commit()
    finally:
        _auth_pool.putconn(conn)


def _execute_returning(sql: Any, params: tuple):
    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
        conn.commit()
        return dict(row) if row else None
    finally:
        _auth_pool.putconn(conn)


def _serialize_user(row: dict) -> dict:
    user = dict(row)
    if user.get("approved_at"):
        user["approved_at"] = user["approved_at"].isoformat()
    if user.get("created_at"):
        user["created_at"] = user["created_at"].isoformat()
    if user.get("updated_at"):
        user["updated_at"] = user["updated_at"].isoformat()
    user["email_confirmed"] = bool(user.get("email_confirmed"))
    return user


def _get_profile(user_id: str) -> Optional[dict]:
    row = _query_one(
        """
        SELECT
            p.user_id::text AS id,
            p.username,
            p.role,
            p.approval_status,
            p.approved_by::text AS approved_by,
            p.approved_at,
            p.created_at,
            p.updated_at,
            au.email,
            (au.email_confirmed_at IS NOT NULL) AS email_confirmed
        FROM user_profiles p
        JOIN auth.users au ON au.id = p.user_id
        WHERE p.user_id = %s
        """,
        (user_id,),
    )
    return _serialize_user(row) if row else None


def _get_bearer_token() -> Optional[str]:
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None
    return auth_header.split(" ", 1)[1].strip() or None


def _fetch_supabase_user(access_token: str) -> Optional[dict]:
    if not SUPABASE_URL:
        return None

    api_key = SUPABASE_ANON_KEY or SUPABASE_SERVICE_ROLE_KEY
    if not api_key:
        return None

    try:
        response = requests.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={
                "Authorization": f"Bearer {access_token}",
                "apikey": api_key,
            },
            timeout=10,
        )
    except requests.RequestException:
        return None
    if response.status_code != 200:
        return None
    return response.json()


def get_current_user(force_refresh: bool = False) -> Optional[dict]:
    if not force_refresh and hasattr(g, "current_user"):
        return g.current_user

    access_token = _get_bearer_token()
    if not access_token:
        g.current_user = None
        return None

    supabase_user = _fetch_supabase_user(access_token)
    if not supabase_user or not supabase_user.get("id"):
        g.current_user = None
        return None

    profile = _get_profile(supabase_user["id"])
    if profile:
        g.current_user = profile
        return profile

    user = {
        "id": supabase_user["id"],
        "email": supabase_user.get("email"),
        "username": None,
        "role": None,
        "approval_status": "pending",
        "approved_by": None,
        "approved_at": None,
        "created_at": None,
        "updated_at": None,
        "email_confirmed": bool(supabase_user.get("email_confirmed_at")),
        "profile_missing": True,
    }
    g.current_user = user
    return user


def get_current_user_id() -> Optional[str]:
    user = get_current_user()
    return user.get("id") if user else None


def _require_role(min_role: Optional[str] = None):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            user = get_current_user()
            if not user:
                return jsonify({"error": "Authentication required"}), 401
            if user.get("profile_missing"):
                return jsonify({"error": "User profile not found"}), 403
            if not user.get("email_confirmed"):
                return jsonify({"error": "Email confirmation required"}), 403
            if user.get("approval_status") != "approved":
                return jsonify({"error": "Account approval required"}), 403
            if min_role:
                if ROLE_RANK.get(user.get("role"), 0) < ROLE_RANK[min_role]:
                    return jsonify({"error": f"{min_role.replace('_', ' ').title()} access required"}), 403
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def authenticated_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        user = get_current_user()
        if not user:
            return jsonify({"error": "Authentication required"}), 401
        return fn(*args, **kwargs)

    return wrapper


def agent_or_above(fn):
    return _require_role("agent")(fn)


def admin_or_above(fn):
    return _require_role("admin")(fn)


def super_admin_required(fn):
    return _require_role("super_admin")(fn)


# Backward-compatible alias for legacy route decorators.
admin_required = admin_or_above


def _log_auth_audit(action: str, resource_id: Optional[str], details: dict) -> None:
    user_id = get_current_user_id()
    try:
        _execute(
            """
            INSERT INTO audit_log (user_id, action, resource_type, resource_id, details, ip_address)
            VALUES (%s, %s, 'user_profile', %s, %s::jsonb, %s)
            """,
            (
                user_id,
                action,
                resource_id,
                psycopg2.extras.Json(details),
                request.remote_addr,
            ),
        )
    except Exception:
        # Audit logging should not block the business action.
        pass


@auth_bp.route("/me", methods=["GET"])
@authenticated_required
def me():
    user = get_current_user(force_refresh=True)
    return jsonify({"user": user}), 200


@auth_bp.route("/users", methods=["GET"])
@admin_or_above
def list_users():
    current_user = get_current_user()
    params = []
    where_clause = ""
    if current_user["role"] == "admin":
        where_clause = "WHERE p.role = 'agent'"

    rows = _query_all(
        f"""
        SELECT
            p.user_id::text AS id,
            p.username,
            au.email,
            p.role,
            p.approval_status,
            p.approved_by::text AS approved_by,
            approver.username AS approved_by_username,
            p.approved_at,
            p.created_at,
            p.updated_at,
            (au.email_confirmed_at IS NOT NULL) AS email_confirmed
        FROM user_profiles p
        JOIN auth.users au ON au.id = p.user_id
        LEFT JOIN user_profiles approver ON approver.user_id = p.approved_by
        {where_clause}
        ORDER BY
            CASE WHEN p.approval_status = 'pending' THEN 0 ELSE 1 END,
            p.created_at DESC
        """,
        tuple(params),
    )
    return jsonify({"users": [_serialize_user(row) for row in rows]}), 200


@auth_bp.route("/users/<user_id>", methods=["PATCH"])
@admin_or_above
def update_user(user_id: str):
    current_user = get_current_user()
    data = request.get_json(silent=True) or {}

    target = _query_one(
        """
        SELECT
            p.user_id::text AS id,
            p.username,
            p.role,
            p.approval_status,
            au.email,
            (au.email_confirmed_at IS NOT NULL) AS email_confirmed
        FROM user_profiles p
        JOIN auth.users au ON au.id = p.user_id
        WHERE p.user_id = %s
        """,
        (user_id,),
    )
    if not target:
        return jsonify({"error": "User not found"}), 404

    if current_user["role"] == "admin" and target["role"] != "agent":
        return jsonify({"error": "Admins can only manage agent accounts"}), 403

    set_clauses = []
    params: list[Any] = []
    audit_details: dict[str, Any] = {}

    if "approval_status" in data:
        approval_status = data["approval_status"]
        if approval_status not in VALID_APPROVAL_STATUSES:
            return jsonify({"error": "Invalid approval_status"}), 400

        set_clauses.append("approval_status = %s")
        params.append(approval_status)
        audit_details["approval_status"] = approval_status

        set_clauses.append("approved_by = %s")
        params.append(current_user["id"])
        set_clauses.append(
            "approved_at = CASE WHEN %s = 'approved' THEN NOW() ELSE NULL END"
        )
        params.append(approval_status)

    if "role" in data:
        if current_user["role"] != "super_admin":
            return jsonify({"error": "Only super admins can change roles"}), 403
        role = data["role"]
        if role not in VALID_ROLES:
            return jsonify({"error": "Invalid role"}), 400
        set_clauses.append("role = %s")
        params.append(role)
        audit_details["role"] = role

    if not set_clauses:
        return jsonify({"error": "No valid fields to update"}), 400

    set_clauses.append("updated_at = NOW()")
    params.append(user_id)

    user = _execute_returning(
        f"""
        UPDATE user_profiles
        SET {", ".join(set_clauses)}
        WHERE user_id = %s
        RETURNING
            user_id::text AS id,
            username,
            role,
            approval_status,
            approved_by::text AS approved_by,
            approved_at,
            created_at,
            updated_at
        """,
        tuple(params),
    )
    if not user:
        return jsonify({"error": "User not found"}), 404

    merged = _get_profile(user["id"]) or {**user, "email": target["email"], "email_confirmed": target["email_confirmed"]}
    _log_auth_audit("USER_PROFILE_UPDATED", user_id, audit_details)
    return jsonify({"user": merged}), 200


@auth_bp.route("/login", methods=["POST"])
def deprecated_login():
    return jsonify({"error": "Deprecated endpoint. Use Supabase Auth on the frontend."}), 410


@auth_bp.route("/logout", methods=["POST"])
def deprecated_logout():
    return jsonify({"error": "Deprecated endpoint. Use Supabase Auth on the frontend."}), 410


@auth_bp.route("/refresh", methods=["POST"])
def deprecated_refresh():
    return jsonify({"error": "Deprecated endpoint. Use Supabase Auth on the frontend."}), 410


@auth_bp.route("/register", methods=["POST"])
def deprecated_register():
    return jsonify({"error": "Deprecated endpoint. Public registration now uses Supabase Auth."}), 410
