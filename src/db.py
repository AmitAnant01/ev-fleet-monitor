"""
db.py
-----
SQLite-backed user store for sign up / sign in.

v2 additions (pro upgrade):
- Richer profile: full_name, email, assigned_car_id, last_login
- Driver accounts are linked to a real Car_ID from the fleet, so the
  driver dashboard can auto-select "their" car instead of a bare list.
- Password change (requires the current password).
- Lightweight admin-facing helpers (list_users, set_role, delete_user,
  deactivate/reactivate) so the admin dashboard can manage accounts.
- Migration-safe: ALTER TABLE is attempted for existing DBs created by
  the old schema, so nobody has to delete data/users.db to upgrade.

Passwords are never stored in plain text: PBKDF2-HMAC-SHA256 with a
random salt per user (100,000 iterations).
"""

import sqlite3
import hashlib
import os
import re
from datetime import datetime, timezone

DB_PATH = os.path.join("data", "users.db")

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# ---------------------------------------------------------------------
# low-level helpers
# ---------------------------------------------------------------------

def _connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _hash_password(password: str, salt: str = None):
    if salt is None:
        salt = os.urandom(16).hex()
    pwd_hash = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), bytes.fromhex(salt), 100_000
    ).hex()
    return salt, pwd_hash


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _column_exists(c, table, column):
    c.execute(f"PRAGMA table_info({table})")
    return column in {row[1] for row in c.fetchall()}


def _migrate(conn):
    """Add any columns missing from an older users.db so existing
    installs don't break when this file is upgraded."""
    c = conn.cursor()
    new_columns = {
        "full_name": "TEXT DEFAULT ''",
        "email": "TEXT DEFAULT ''",
        "assigned_car_id": "TEXT DEFAULT ''",
        "last_login": "TEXT DEFAULT ''",
        "is_active": "INTEGER DEFAULT 1",
    }
    for col, ddl in new_columns.items():
        if not _column_exists(c, "users", col):
            c.execute(f"ALTER TABLE users ADD COLUMN {col} {ddl}")
    conn.commit()


# ---------------------------------------------------------------------
# schema / seed
# ---------------------------------------------------------------------

def init_db():
    """Create the users table if it doesn't exist, migrate older DBs,
    and seed two default accounts (admin/admin123, driver/driver123) so
    the team has a working login on a fresh clone before anyone signs
    up."""
    conn = _connect()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            salt TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('Admin','Driver')),
            full_name TEXT DEFAULT '',
            email TEXT DEFAULT '',
            assigned_car_id TEXT DEFAULT '',
            last_login TEXT DEFAULT '',
            is_active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            actor TEXT NOT NULL,
            action TEXT NOT NULL,
            target TEXT DEFAULT '',
            details TEXT DEFAULT '',
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    _migrate(conn)

    seed_accounts = [
        ("admin", "admin123", "Admin", "Fleet Administrator", "admin@evfleet.io", ""),
        ("driver", "driver123", "Driver", "Demo Driver", "driver@evfleet.io", "HYU-01"),
    ]
    for username, password, role, full_name, email, car_id in seed_accounts:
        salt, pwd_hash = _hash_password(password)
        try:
            c.execute(
                """INSERT INTO users
                   (username, salt, password_hash, role, full_name, email, assigned_car_id)
                   VALUES (?,?,?,?,?,?,?)""",
                (username, salt, pwd_hash, role, full_name, email, car_id),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            pass  # already exists, fine

    conn.close()


# ---------------------------------------------------------------------
# auth
# ---------------------------------------------------------------------

def username_exists(username: str) -> bool:
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT 1 FROM users WHERE username=?", (username.lower(),))
    exists = c.fetchone() is not None
    conn.close()
    return exists


def email_exists(email: str) -> bool:
    email = (email or "").strip().lower()
    if not email:
        return False
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT 1 FROM users WHERE lower(email)=?", (email,))
    exists = c.fetchone() is not None
    conn.close()
    return exists


def register_user(username: str, password: str, confirm_password: str, role: str,
                   full_name: str = "", email: str = "", assigned_car_id: str = ""):
    """Validates and creates a new account. Returns (success: bool, message: str)."""
    username = (username or "").strip().lower()
    full_name = (full_name or "").strip()
    email = (email or "").strip().lower()

    if not username or not password:
        return False, "Username and password can't be empty."
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if not re.match(r"^[a-zA-Z0-9_.]+$", username):
        return False, "Username can only contain letters, numbers, dots and underscores."
    if not full_name:
        return False, "Please enter your full name."
    if password != confirm_password:
        return False, "Passwords don't match."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    if not re.search(r"[A-Za-z]", password) or not re.search(r"[0-9]", password):
        return False, "Password should include at least one letter and one number."
    if email and not EMAIL_RE.match(email):
        return False, "That doesn't look like a valid email address."
    if email and email_exists(email):
        return False, "An account with that email already exists."
    if username_exists(username):
        return False, "That username is already taken."
    if role == "Driver" and not assigned_car_id:
        return False, "Please select the car you drive."

    salt, pwd_hash = _hash_password(password)
    conn = _connect()
    c = conn.cursor()
    c.execute(
        """INSERT INTO users
           (username, salt, password_hash, role, full_name, email, assigned_car_id)
           VALUES (?,?,?,?,?,?,?)""",
        (username, salt, pwd_hash, role, full_name, email, assigned_car_id),
    )
    conn.commit()
    conn.close()
    return True, "Account created successfully. You can sign in now."


def verify_user(username: str, password: str):
    """Returns a user dict if credentials are correct and the account is
    active, else None (or {'error': 'disabled'} for a disabled account).
    Also stamps last_login."""
    username = (username or "").strip().lower()
    conn = _connect()
    c = conn.cursor()
    c.execute(
        """SELECT salt, password_hash, role, full_name, email,
                  assigned_car_id, is_active
           FROM users WHERE username=?""",
        (username,),
    )
    row = c.fetchone()

    if row is None:
        conn.close()
        return None

    salt, stored_hash, role, full_name, email, car_id, is_active = row
    _, test_hash = _hash_password(password, salt)

    if test_hash != stored_hash:
        conn.close()
        return None
    if not is_active:
        conn.close()
        return {"error": "disabled"}

    c.execute("UPDATE users SET last_login=? WHERE username=?", (_now(), username))
    conn.commit()
    conn.close()

    return {
        "username": username,
        "role": role,
        "full_name": full_name,
        "email": email,
        "assigned_car_id": car_id,
    }


# ---------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------

def get_user(username: str):
    username = (username or "").strip().lower()
    conn = _connect()
    c = conn.cursor()
    c.execute(
        """SELECT username, role, full_name, email, assigned_car_id,
                  last_login, created_at, is_active
           FROM users WHERE username=?""",
        (username,),
    )
    row = c.fetchone()
    conn.close()
    if row is None:
        return None
    keys = ["username", "role", "full_name", "email", "assigned_car_id",
            "last_login", "created_at", "is_active"]
    return dict(zip(keys, row))


def update_profile(username: str, full_name: str = None, email: str = None,
                    assigned_car_id: str = None):
    username = (username or "").strip().lower()
    fields, values = [], []
    if full_name is not None:
        fields.append("full_name=?")
        values.append(full_name.strip())
    if email is not None:
        email = email.strip().lower()
        if email and not EMAIL_RE.match(email):
            return False, "That doesn't look like a valid email address."
        fields.append("email=?")
        values.append(email)
    if assigned_car_id is not None:
        fields.append("assigned_car_id=?")
        values.append(assigned_car_id)
    if not fields:
        return False, "Nothing to update."

    values.append(username)
    conn = _connect()
    c = conn.cursor()
    c.execute(f"UPDATE users SET {', '.join(fields)} WHERE username=?", values)
    conn.commit()
    conn.close()
    return True, "Profile updated."


def change_password(username: str, current_password: str, new_password: str,
                     confirm_password: str):
    username = (username or "").strip().lower()
    if new_password != confirm_password:
        return False, "New passwords don't match."
    if len(new_password) < 6:
        return False, "New password must be at least 6 characters."
    if not re.search(r"[A-Za-z]", new_password) or not re.search(r"[0-9]", new_password):
        return False, "Password should include at least one letter and one number."

    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT salt, password_hash FROM users WHERE username=?", (username,))
    row = c.fetchone()
    if row is None:
        conn.close()
        return False, "User not found."

    salt, stored_hash = row
    _, test_hash = _hash_password(current_password, salt)
    if test_hash != stored_hash:
        conn.close()
        return False, "Current password is incorrect."

    new_salt, new_hash = _hash_password(new_password)
    c.execute("UPDATE users SET salt=?, password_hash=? WHERE username=?",
              (new_salt, new_hash, username))
    conn.commit()
    conn.close()
    return True, "Password changed successfully."


# ---------------------------------------------------------------------
# admin-facing helpers (used by admin_dashboard.py)
# ---------------------------------------------------------------------

def list_users(role: str = None):
    conn = _connect()
    c = conn.cursor()
    if role:
        c.execute(
            """SELECT username, role, full_name, email, assigned_car_id,
                      last_login, created_at, is_active
               FROM users WHERE role=? ORDER BY created_at DESC""",
            (role,),
        )
    else:
        c.execute(
            """SELECT username, role, full_name, email, assigned_car_id,
                      last_login, created_at, is_active
               FROM users ORDER BY created_at DESC"""
        )
    rows = c.fetchall()
    conn.close()
    keys = ["username", "role", "full_name", "email", "assigned_car_id",
            "last_login", "created_at", "is_active"]
    return [dict(zip(keys, row)) for row in rows]


def set_active(username: str, is_active: bool):
    conn = _connect()
    c = conn.cursor()
    c.execute("UPDATE users SET is_active=? WHERE username=?",
              (1 if is_active else 0, username.strip().lower()))
    conn.commit()
    conn.close()


def set_role(username: str, role: str):
    conn = _connect()
    c = conn.cursor()
    c.execute("UPDATE users SET role=? WHERE username=?", (role, username.strip().lower()))
    conn.commit()
    conn.close()


def admin_reset_password(username: str, new_password: str):
    """Admin-triggered reset — no current-password check."""
    salt, pwd_hash = _hash_password(new_password)
    conn = _connect()
    c = conn.cursor()
    c.execute("UPDATE users SET salt=?, password_hash=? WHERE username=?",
              (salt, pwd_hash, username.strip().lower()))
    conn.commit()
    conn.close()


def delete_user(username: str):
    conn = _connect()
    c = conn.cursor()
    c.execute("DELETE FROM users WHERE username=?", (username.strip().lower(),))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------
# audit log (used by auth.py for logins/signups and admin_dashboard.py
# for account-management actions)
# ---------------------------------------------------------------------

def log_audit(actor: str, action: str, target: str = "", details: str = ""):
    """Appends one row to the audit trail. Never raises — a logging
    failure should never break the action it's logging."""
    try:
        conn = _connect()
        c = conn.cursor()
        c.execute(
            "INSERT INTO audit_logs (actor, action, target, details, created_at) VALUES (?,?,?,?,?)",
            (actor or "system", action, target, details, _now()),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def list_audit_logs(limit: int = 200, actor: str = None, action_contains: str = None):
    conn = _connect()
    c = conn.cursor()
    query = "SELECT actor, action, target, details, created_at FROM audit_logs"
    clauses, params = [], []
    if actor:
        clauses.append("actor=?")
        params.append(actor.strip().lower())
    if action_contains:
        clauses.append("action LIKE ?")
        params.append(f"%{action_contains}%")
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    c.execute(query, params)
    rows = c.fetchall()
    conn.close()
    keys = ["actor", "action", "target", "details", "created_at"]
    return [dict(zip(keys, row)) for row in rows]
