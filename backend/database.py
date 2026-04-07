"""
database.py
-----------
SQLite database layer for ResolveAI.

Tables:
  users       – user accounts (email, password_hash, created_at)
  complaints  – every complaint submitted (text, category, confidence, status, user_id)
  tickets     – support tickets created when confidence is low
  analytics   – aggregated category counts (updated on each complaint)
"""

import sqlite3
import os
import uuid
from datetime import datetime
from typing import List, Optional, Dict
import bcrypt

# Database configuration - supports both SQLite (dev) and PostgreSQL (prod)
DATABASE_URL = os.environ.get('DATABASE_URL')

if DATABASE_URL:
    # Production: PostgreSQL
    import psycopg2
    import psycopg2.extras
    DB_TYPE = 'postgresql'
else:
    # Development: SQLite
    DB_TYPE = 'sqlite'
    DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resolveai.db")


# ─────────────────────────────────────────────────────────────────────────────
# DB INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def init_db():
    """Create tables if they do not exist."""
    conn = get_connection()
    cursor = conn.cursor()

    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id          TEXT PRIMARY KEY,
            email       TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at  TEXT NOT NULL
        )
    """)

    # Complaints table (updated with user_id)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS complaints (
            id          TEXT PRIMARY KEY,
            user_id     TEXT NOT NULL,
            text        TEXT NOT NULL,
            category    TEXT NOT NULL,
            confidence  REAL NOT NULL,
            status      TEXT NOT NULL DEFAULT 'resolved',
            response    TEXT,
            ticket_id   TEXT,
            created_at  TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # Tickets table (created when confidence < threshold)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            id              TEXT PRIMARY KEY,
            user_id         TEXT NOT NULL,
            complaint_id    TEXT NOT NULL,
            complaint_text  TEXT NOT NULL,
            category        TEXT NOT NULL,
            confidence      REAL NOT NULL,
            priority        TEXT NOT NULL DEFAULT 'medium',
            status          TEXT NOT NULL DEFAULT 'open',
            created_at      TEXT NOT NULL,
            FOREIGN KEY (complaint_id) REFERENCES complaints(id),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # Analytics table (now per user)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analytics (
            user_id     TEXT NOT NULL,
            category    TEXT NOT NULL,
            count       INTEGER NOT NULL DEFAULT 0,
            avg_conf    REAL NOT NULL DEFAULT 0.0,
            resolved    INTEGER NOT NULL DEFAULT 0,
            tickets     INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (user_id, category),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()


def get_connection():
    """Return a configured database connection (SQLite or PostgreSQL)."""
    if DB_TYPE == 'postgresql':
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    else:
        # SQLite
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn


# ─────────────────────────────────────────────────────────────────────────────
# COMPLAINTS
# ─────────────────────────────────────────────────────────────────────────────

def save_complaint(
    text: str,
    category: str,
    confidence: float,
    status: str,
    user_id: str,
    response: Optional[str] = None,
    ticket_id: Optional[str] = None
) -> str:
    """Insert a complaint record and update analytics. Returns complaint ID."""
    complaint_id = f"CMP-{uuid.uuid4().hex[:8].upper()}"
    now = datetime.utcnow().isoformat()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO complaints (id, user_id, text, category, confidence, status, response, ticket_id, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (complaint_id, user_id, text, category, confidence, status, response, ticket_id, now))

    # Update analytics
    _update_analytics(cursor, user_id, category, confidence, status)

    conn.commit()
    conn.close()
    return complaint_id


def get_all_complaints(user_id: str, limit: int = 50) -> List[dict]:
    """Fetch most recent complaints for a specific user."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM complaints WHERE user_id = ? ORDER BY created_at DESC LIMIT ?", (user_id, limit)
    )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# TICKETS
# ─────────────────────────────────────────────────────────────────────────────

def create_ticket(
    user_id: str,
    complaint_id: str,
    complaint_text: str,
    category: str,
    confidence: float
) -> str:
    """Create a support ticket and return its ID."""
    ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"
    now = datetime.utcnow().isoformat()

    # Priority based on confidence: very low → high priority
    if confidence < 0.35:
        priority = "high"
    elif confidence < 0.45:
        priority = "medium"
    else:
        priority = "low"

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO tickets (id, user_id, complaint_id, complaint_text, category, confidence, priority, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'open', ?)
    """, (ticket_id, user_id, complaint_id, complaint_text, category, confidence, priority, now))

    # Update analytics
    cursor.execute(
        "UPDATE analytics SET tickets = tickets + 1 WHERE user_id = ? AND category = ?", (user_id, category)
    )

    conn.commit()
    conn.close()
    return ticket_id


def get_all_tickets(user_id: str, limit: int = 50) -> List[dict]:
    """Fetch most recent open tickets for a specific user."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM tickets WHERE user_id = ? AND status = 'open' ORDER BY created_at DESC LIMIT ?", (user_id, limit)
    )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# USER MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def create_user(email: str, password: str) -> Optional[str]:
    """Create a new user account. Returns user_id on success, None if email exists."""
    # Hash the password
    password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    user_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()

    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO users (id, email, password_hash, created_at)
            VALUES (?, ?, ?, ?)
        """, (user_id, email.lower(), password_hash, now))

        # Initialize analytics for this user
        for cat in ["Billing", "Technical", "Service", "General"]:
            cursor.execute("""
                INSERT INTO analytics (user_id, category, count, avg_conf, resolved, tickets)
                VALUES (?, ?, 0, 0.0, 0, 0)
            """, (user_id, cat))

        conn.commit()
        conn.close()
        return user_id
    except sqlite3.IntegrityError:
        # Email already exists
        return None


def authenticate_user(email: str, password: str) -> Optional[str]:
    """Authenticate user. Returns user_id on success, None on failure."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, password_hash FROM users WHERE email = ?",
        (email.lower(),)
    )
    row = cursor.fetchone()
    conn.close()

    if row and bcrypt.checkpw(password.encode('utf-8'), row['password_hash'].encode('utf-8')):
        return row['id']
    return None


def get_user_by_id(user_id: str) -> Optional[dict]:
    """Get user info by ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, email, created_at FROM users WHERE id = ?",
        (user_id,)
    )
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_email(email: str) -> Optional[dict]:
    """Get user info by email."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, email, created_at FROM users WHERE email = ?",
        (email.lower(),)
    )
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


# ─────────────────────────────────────────────────────────────────────────────
# ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────

def get_analytics(user_id: str) -> Dict[str, dict]:
    """Return analytics summary per category for a specific user."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT category, count, avg_conf, resolved, tickets FROM analytics WHERE user_id = ?", (user_id,))
    rows = {row["category"]: dict(row) for row in cursor.fetchall()}
    conn.close()
    return rows


def _update_analytics(cursor, user_id: str, category: str, confidence: float, status: str):
    """Internal helper – update rolling analytics for a user-category pair."""
    cursor.execute("SELECT count, avg_conf, resolved FROM analytics WHERE user_id = ? AND category = ?", (user_id, category))
    row = cursor.fetchone()

    if row:
        old_count = row["count"]
        old_avg = row["avg_conf"]
        new_count = old_count + 1
        new_avg = ((old_avg * old_count) + confidence) / new_count
        new_resolved = row["resolved"] + (1 if status == "resolved" else 0)

        cursor.execute("""
            UPDATE analytics
            SET count = ?, avg_conf = ?, resolved = ?
            WHERE user_id = ? AND category = ?
        """, (new_count, round(new_avg, 4), new_resolved, user_id, category))
    else:
        cursor.execute("""
            INSERT INTO analytics (user_id, category, count, avg_conf, resolved, tickets)
            VALUES (?, ?, 1, ?, ?, 0)
        """, (user_id, category, round(confidence, 4), 1 if status == "resolved" else 0))


# Initialise on import
init_db()
