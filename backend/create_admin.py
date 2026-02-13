#!/usr/bin/env python3
"""
One-time script to create application users (admin or user role).

Usage:
    python create_admin.py --username admin --email admin@example.com --password yourpassword
    python create_admin.py --username staff1 --email staff@example.com --password pass123 --role user
"""

import argparse
import os
import sys

import bcrypt
import psycopg2
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Create an application user")
    parser.add_argument("--username", required=True, help="Login username")
    parser.add_argument("--email", required=True, help="User email address")
    parser.add_argument("--password", required=True, help="Plain-text password (will be hashed)")
    parser.add_argument("--role", default="admin", choices=["admin", "user"],
                        help="User role (default: admin)")
    args = parser.parse_args()

    if len(args.password) < 8:
        print("Error: Password must be at least 8 characters.", file=sys.stderr)
        sys.exit(1)

    password_hash = bcrypt.hashpw(args.password.encode(), bcrypt.gensalt(12)).decode()

    try:
        conn = psycopg2.connect(
            host=os.getenv("SUPABASE_HOST"),
            database="postgres",
            user=os.getenv("SUPABASE_USER", "postgres"),
            password=os.getenv("SUPABASE_PASSWORD"),
            port=5432,
        )
    except Exception as e:
        print(f"Database connection failed: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO users (username, email, password_hash, role)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (username) DO NOTHING
                RETURNING id
                """,
                (args.username, args.email, password_hash, args.role),
            )
            result = cur.fetchone()
        conn.commit()
    except Exception as e:
        print(f"Failed to create user: {e}", file=sys.stderr)
        conn.close()
        sys.exit(1)
    finally:
        conn.close()

    if result:
        print(f"User '{args.username}' ({args.role}) created successfully. ID: {result[0]}")
    else:
        print(f"User '{args.username}' already exists — no changes made.")


if __name__ == "__main__":
    main()
