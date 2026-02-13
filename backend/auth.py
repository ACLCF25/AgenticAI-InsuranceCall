"""
Authentication Blueprint for the Credentialing Agent API.
Provides JWT-based login/logout/refresh endpoints and role-based decorators.
"""

import os
from functools import wraps
from datetime import datetime, timezone

import bcrypt
import psycopg2
import psycopg2.extras
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
# DB helper – raw psycopg2 (mirrors DatabaseManager connection pattern)
# ---------------------------------------------------------------------------

def _get_conn():
    return psycopg2.connect(
        host=os.getenv("SUPABASE_HOST"),
        database="postgres",
        user=os.getenv("SUPABASE_USER", "postgres"),
        password=os.getenv("SUPABASE_PASSWORD"),
        port=5432,
    )


def _query_one(sql: str, params: tuple):
    """Execute a SELECT and return the first row as a dict, or None."""
    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        conn.close()


def _execute(sql: str, params: tuple):
    """Execute a write statement and commit."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
        conn.commit()
    finally:
        conn.close()


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
