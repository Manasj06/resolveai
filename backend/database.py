"""
database.py
-----------
SQLite database layer for ResolveAI.

Tables:
  complaints  – every complaint submitted (text, category, confidence, status)
  tickets     – support tickets created when confidence is low
  analytics   – aggregated category counts (updated on each complaint)
"""

import sqlite3
import os
import uuid
from datetime import datetime
from typing import List, Optional, Dict

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resolveai.db")


# ─────────────────────────────────────────────────────────────────────────────
# DB INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def init_db():
    """Create tables if they do not exist."""
    conn = get_connection()
    cursor = conn.cursor()

    # Complaints table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS complaints (
            id          TEXT PRIMARY KEY,
            text        TEXT NOT NULL,
            category    TEXT NOT NULL,
            confidence  REAL NOT NULL,
            status      TEXT NOT NULL DEFAULT 'resolved',
            response    TEXT,
            ticket_id   TEXT,
            created_at  TEXT NOT NULL
        )
    """)

    # Tickets table (created when confidence < threshold)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            id              TEXT PRIMARY KEY,
            complaint_id    TEXT NOT NULL,
            complaint_text  TEXT NOT NULL,
            category        TEXT NOT NULL,
            confidence      REAL NOT NULL,
            priority        TEXT NOT NULL DEFAULT 'medium',
            status          TEXT NOT NULL DEFAULT 'open',
            created_at      TEXT NOT NULL,
            FOREIGN KEY (complaint_id) REFERENCES complaints(id)
        )
    """)

    # Analytics table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analytics (
            category    TEXT PRIMARY KEY,
            count       INTEGER NOT NULL DEFAULT 0,
            avg_conf    REAL NOT NULL DEFAULT 0.0,
            resolved    INTEGER NOT NULL DEFAULT 0,
            tickets     INTEGER NOT NULL DEFAULT 0
        )
    """)

    # Seed analytics rows if empty
    cursor.execute("SELECT COUNT(*) FROM analytics")
    if cursor.fetchone()[0] == 0:
        for cat in ["Billing", "Technical", "Service", "General"]:
            cursor.execute(
                "INSERT INTO analytics (category, count, avg_conf, resolved, tickets) VALUES (?,0,0.0,0,0)",
                (cat,)
            )

    conn.commit()
    conn.close()


def get_connection() -> sqlite3.Connection:
    """Return a configured SQLite connection."""
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
    response: Optional[str] = None,
    ticket_id: Optional[str] = None
) -> str:
    """Insert a complaint record and update analytics. Returns complaint ID."""
    complaint_id = f"CMP-{uuid.uuid4().hex[:8].upper()}"
    now = datetime.utcnow().isoformat()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO complaints (id, text, category, confidence, status, response, ticket_id, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (complaint_id, text, category, confidence, status, response, ticket_id, now))

    # Update analytics
    _update_analytics(cursor, category, confidence, status)

    conn.commit()
    conn.close()
    return complaint_id


def get_all_complaints(limit: int = 50) -> List[dict]:
    """Fetch most recent complaints."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM complaints ORDER BY created_at DESC LIMIT ?", (limit,)
    )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# TICKETS
# ─────────────────────────────────────────────────────────────────────────────

def create_ticket(
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
        INSERT INTO tickets (id, complaint_id, complaint_text, category, confidence, priority, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, 'open', ?)
    """, (ticket_id, complaint_id, complaint_text, category, confidence, priority, now))

    # Update analytics
    cursor.execute(
        "UPDATE analytics SET tickets = tickets + 1 WHERE category = ?", (category,)
    )

    conn.commit()
    conn.close()
    return ticket_id


def get_all_tickets(limit: int = 50) -> List[dict]:
    """Fetch most recent open tickets."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM tickets ORDER BY created_at DESC LIMIT ?", (limit,)
    )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────

def get_analytics() -> Dict[str, dict]:
    """Return analytics summary per category."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM analytics")
    rows = {row["category"]: dict(row) for row in cursor.fetchall()}
    conn.close()
    return rows


def _update_analytics(cursor, category: str, confidence: float, status: str):
    """Internal helper – update rolling analytics for a category."""
    cursor.execute("SELECT count, avg_conf, resolved FROM analytics WHERE category = ?", (category,))
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
            WHERE category = ?
        """, (new_count, round(new_avg, 4), new_resolved, category))
    else:
        cursor.execute("""
            INSERT INTO analytics (category, count, avg_conf, resolved, tickets)
            VALUES (?, 1, ?, ?, 0)
        """, (category, round(confidence, 4), 1 if status == "resolved" else 0))


# Initialise on import
init_db()
