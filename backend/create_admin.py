#!/usr/bin/env python3
"""
Bootstrap a Supabase-authenticated application user.

Examples:
    python create_admin.py --username founder --email founder@example.com --password strongpass123
    python create_admin.py --username ops-admin --email ops@example.com --password strongpass123 --role admin
"""

import argparse
import os
import sys

import psycopg2
import requests
from dotenv import load_dotenv

load_dotenv()


def create_or_get_auth_user(email: str, password: str, username: str) -> str:
    supabase_url = (os.getenv("SUPABASE_URL") or "").rstrip("/")
    service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not supabase_url or not service_role_key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required")

    response = requests.post(
        f"{supabase_url}/auth/v1/admin/users",
        headers={
            "apikey": service_role_key,
            "Authorization": f"Bearer {service_role_key}",
            "Content-Type": "application/json",
        },
        json={
            "email": email,
            "password": password,
            "email_confirm": True,
            "user_metadata": {"username": username},
        },
        timeout=20,
    )

    if response.status_code in (200, 201):
        user = response.json().get("user", {})
        return user["id"]

    if response.status_code == 422 and "already been registered" in response.text.lower():
        raise ValueError("EMAIL_EXISTS")

    raise RuntimeError(f"Supabase admin user creation failed: {response.status_code} {response.text}")


def main():
    parser = argparse.ArgumentParser(description="Create or promote a Supabase-authenticated application user")
    parser.add_argument("--username", required=True, help="Display username stored in public.user_profiles")
    parser.add_argument("--email", required=True, help="User email address")
    parser.add_argument("--password", required=True, help="Initial password for Supabase Auth")
    parser.add_argument(
        "--role",
        default="super_admin",
        choices=["super_admin", "admin", "agent"],
        help="Application role to assign (default: super_admin)",
    )
    args = parser.parse_args()

    if len(args.password) < 8:
        print("Error: Password must be at least 8 characters.", file=sys.stderr)
        sys.exit(1)

    try:
        conn = psycopg2.connect(
            host=os.getenv("SUPABASE_HOST"),
            database="postgres",
            user=os.getenv("SUPABASE_USER", "postgres"),
            password=os.getenv("SUPABASE_PASSWORD"),
            port=5432,
        )
    except Exception as exc:
        print(f"Database connection failed: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM auth.users WHERE email = %s", (args.email.lower(),))
            existing = cur.fetchone()
            if existing:
                user_id = str(existing[0])
                print(f"Auth user already exists for {args.email}; updating profile role/status.")
                cur.execute(
                    "UPDATE auth.users SET email_confirmed_at = COALESCE(email_confirmed_at, NOW()) WHERE id = %s",
                    (user_id,),
                )
            else:
                try:
                    user_id = create_or_get_auth_user(args.email.lower(), args.password, args.username)
                except ValueError as exc:
                    if str(exc) != "EMAIL_EXISTS":
                        raise
                    cur.execute("SELECT id FROM auth.users WHERE email = %s", (args.email.lower(),))
                    existing = cur.fetchone()
                    if not existing:
                        raise RuntimeError("User reported as existing but could not be found in auth.users")
                    user_id = str(existing[0])

            cur.execute(
                """
                INSERT INTO public.user_profiles (user_id, username, role, approval_status, approved_at)
                VALUES (%s, %s, %s, 'approved', NOW())
                ON CONFLICT (user_id) DO UPDATE SET
                    username = EXCLUDED.username,
                    role = EXCLUDED.role,
                    approval_status = 'approved',
                    approved_at = NOW(),
                    updated_at = NOW()
                """,
                (user_id, args.username, args.role),
            )
        conn.commit()
    except Exception as exc:
        conn.rollback()
        print(f"Failed to create or update user: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        conn.close()

    print(f"User '{args.email}' configured successfully with role '{args.role}'. ID: {user_id}")


if __name__ == "__main__":
    main()
