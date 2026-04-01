"""
Authentication Blueprint for the Credentialing Agent API.
Provides JWT-based login/logout/refresh endpoints and role-based decorators.
"""

import os
import re
from functools import wraps
from datetime import datetime, timezone

import bcrypt
import psycopg2
import psycopg2.extras
import psycopg2.pool
import psycopg2.sql
from flask import Blueprint, request, jsonify
from flask_jwt_extended import (
    create_access_token,
    create_refresh_token,
    jwt_required,
    get_jwt_identity,
    get_jwt,
)

auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")


# ---------------------------------------------------------------------------
# Module-level connection pool — created once at import time, shared across
# all auth requests.  minconn=2 is sufficient for auth (low concurrency);
# maxconn=10 caps memory on constrained deployments.
# ---------------------------------------------------------------------------

_auth_pool = psycopg2.pool.ThreadedConnectionPool(
    minconn=2,
    maxconn=10,
    host=os.getenv("SUPABASE_HOST"),
    database="postgres",
    user=os.getenv("SUPABASE_USER", "postgres"),
    password=os.getenv("SUPABASE_PASSWORD"),
    port=5432,
)


# ---------------------------------------------------------------------------
# DB helper – acquires from pool instead of creating a new connection
# ---------------------------------------------------------------------------

def _get_conn():
    return _auth_pool.getconn()


def _query_one(sql: str, params: tuple):
    """Execute a SELECT and return the first row as a dict, or None."""
    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        # Return to pool instead of closing so the underlying TCP connection
        # is reused by the next request.
        _auth_pool.putconn(conn)


def _query_all(sql: str, params: tuple = ()):
    """Execute a SELECT and return all rows as a list of dicts."""
    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            return [dict(r) for r in rows]
    finally:
        _auth_pool.putconn(conn)


def _execute(sql: str, params: tuple):
    """Execute a write statement and commit."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
        conn.commit()
    finally:
        _auth_pool.putconn(conn)


def _execute_returning(sql: str, params: tuple):
    """Execute a write statement with RETURNING and return the row as a dict."""
    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
        conn.commit()
        return dict(row) if row else None
    finally:
        _auth_pool.putconn(conn)


# ---------------------------------------------------------------------------
# Token blacklist callback (registered on JWTManager in api_server.py)
# ---------------------------------------------------------------------------

def check_token_blacklist(jwt_payload: dict) -> bool:
    """Return True if the token's jti has been blacklisted (revoked)."""
    jti = jwt_payload.get("jti")
    if not jti:
        return False
    row = _query_one(
        "SELECT id FROM token_blacklist WHERE jti = %s AND expires_at > NOW()",
        (jti,),
    )
    return row is not None


# ---------------------------------------------------------------------------
# Role-based access decorators
# ---------------------------------------------------------------------------

def admin_required(fn):
    """Decorator: requires a valid JWT with role='admin'."""
    @wraps(fn)
    @jwt_required()
    def wrapper(*args, **kwargs):
        claims = get_jwt()
        if claims.get("role") != "admin":
            return jsonify({"error": "Admin access required"}), 403
        return fn(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------

@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400

    user = _query_one(
        "SELECT id, username, email, password_hash, role, is_active "
        "FROM users WHERE username = %s",
        (username,),
    )

    if not user or not user["is_active"]:
        return jsonify({"error": "Invalid credentials"}), 401

    if not bcrypt.checkpw(password.encode(), user["password_hash"].encode()):
        return jsonify({"error": "Invalid credentials"}), 401

    # Update last_login timestamp
    _execute(
        "UPDATE users SET last_login = NOW() WHERE id = %s",
        (user["id"],),
    )

    identity = str(user["id"])
    claims = {"role": user["role"]}
    access_token = create_access_token(identity=identity, additional_claims=claims)
    refresh_token = create_refresh_token(identity=identity, additional_claims=claims)

    return jsonify({
        "access_token": access_token,
        "refresh_token": refresh_token,
        "user": {
            "id": str(user["id"]),
            "username": user["username"],
            "email": user["email"],
            "role": user["role"],
        },
    }), 200


@auth_bp.route("/logout", methods=["POST"])
@jwt_required()
def logout():
    claims = get_jwt()
    jti = claims["jti"]
    exp = claims["exp"]
    expires_at = datetime.fromtimestamp(exp, tz=timezone.utc)

    _execute(
        "INSERT INTO token_blacklist (jti, expires_at) VALUES (%s, %s) "
        "ON CONFLICT (jti) DO NOTHING",
        (jti, expires_at),
    )
    return jsonify({"message": "Logged out successfully"}), 200


@auth_bp.route("/refresh", methods=["POST"])
@jwt_required(refresh=True)
def refresh():
    identity = get_jwt_identity()
    claims = get_jwt()
    role = claims.get("role")
    access_claims = {"role": role} if role else None
    if access_claims:
        access_token = create_access_token(identity=identity, additional_claims=access_claims)
    else:
        access_token = create_access_token(identity=identity)
    return jsonify({"access_token": access_token}), 200


@auth_bp.route("/me", methods=["GET"])
@jwt_required()
def me():
    identity = get_jwt_identity()
    user = _query_one(
        "SELECT id, username, email, role, created_at, last_login "
        "FROM users WHERE id = %s",
        (identity,),
    )
    if not user:
        return jsonify({"error": "User not found"}), 404

    # Serialize datetime fields for JSON
    user["id"] = str(user["id"])
    if user.get("created_at"):
        user["created_at"] = user["created_at"].isoformat()
    if user.get("last_login"):
        user["last_login"] = user["last_login"].isoformat()

    return jsonify({"user": user}), 200


# ---------------------------------------------------------------------------
# User management routes (admin only)
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@auth_bp.route("/register", methods=["POST"])
@admin_required
def register():
    """Create a new user (admin only). User starts as inactive (pending approval)."""
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    role = data.get("role", "user")

    # Validation
    if not username:
        return jsonify({"error": "Username is required"}), 400
    if not email or not _EMAIL_RE.match(email):
        return jsonify({"error": "A valid email is required"}), 400
    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters"}), 400
    if role not in ("admin", "user"):
        return jsonify({"error": "Role must be 'admin' or 'user'"}), 400

    password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    try:
        user = _execute_returning(
            "INSERT INTO users (username, email, password_hash, role, is_active) "
            "VALUES (%s, %s, %s, %s, FALSE) "
            "RETURNING id, username, email, role, is_active, created_at",
            (username, email, password_hash, role),
        )
    except psycopg2.errors.UniqueViolation:
        return jsonify({"error": "Username or email already exists"}), 409

    user["id"] = str(user["id"])
    if user.get("created_at"):
        user["created_at"] = user["created_at"].isoformat()

    return jsonify({"user": user}), 201


@auth_bp.route("/users", methods=["GET"])
@admin_required
def list_users():
    """List all users (admin only)."""
    rows = _query_all(
        "SELECT id, username, email, role, is_active, created_at, last_login "
        "FROM users ORDER BY created_at DESC"
    )
    for row in rows:
        row["id"] = str(row["id"])
        if row.get("created_at"):
            row["created_at"] = row["created_at"].isoformat()
        if row.get("last_login"):
            row["last_login"] = row["last_login"].isoformat()

    return jsonify({"users": rows}), 200


@auth_bp.route("/users/<user_id>", methods=["PATCH"])
@admin_required
def update_user(user_id):
    """Update a user's is_active or role (admin only)."""
    current_user_id = get_jwt_identity()
    data = request.get_json(silent=True) or {}

    # Prevent self-deactivation
    if str(user_id) == str(current_user_id) and data.get("is_active") is False:
        return jsonify({"error": "Cannot deactivate your own account"}), 400

    set_clauses = []
    params = []

    if "is_active" in data:
        set_clauses.append(psycopg2.sql.SQL("{} = %s").format(psycopg2.sql.Identifier("is_active")))
        params.append(bool(data["is_active"]))

    if "role" in data:
        if data["role"] not in ("admin", "user"):
            return jsonify({"error": "Role must be 'admin' or 'user'"}), 400
        # Prevent changing own role
        if str(user_id) == str(current_user_id):
            return jsonify({"error": "Cannot change your own role"}), 400
        set_clauses.append(psycopg2.sql.SQL("{} = %s").format(psycopg2.sql.Identifier("role")))
        params.append(data["role"])

    if not set_clauses:
        return jsonify({"error": "No valid fields to update"}), 400

    params.append(user_id)
    sql = psycopg2.sql.SQL(
        "UPDATE users SET {} WHERE id = %s "
        "RETURNING id, username, email, role, is_active, created_at, last_login"
    ).format(psycopg2.sql.SQL(", ").join(set_clauses))

    user = _execute_returning(sql, tuple(params))
    if not user:
        return jsonify({"error": "User not found"}), 404

    user["id"] = str(user["id"])
    if user.get("created_at"):
        user["created_at"] = user["created_at"].isoformat()
    if user.get("last_login"):
        user["last_login"] = user["last_login"].isoformat()

    return jsonify({"user": user}), 200
