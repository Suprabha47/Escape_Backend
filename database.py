"""
database.py — Async Supabase client for Escape
===============================================
All DB I/O lives here. main.py never imports supabase directly.

Pattern:
  - Every public method is async.
  - Supabase-py's execute() is synchronous; we run it in a thread pool
    via asyncio.to_thread() so we never block the event loop.
  - Raises DatabaseError (a plain RuntimeError subclass) on failures
    so callers don't need to know supabase internals.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID
import uuid
import sqlite3
from pathlib import Path
import logging

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("escape.database")

_SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
_SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

DB_PATH = Path(__file__).resolve().parent / "escape_local.db"


class DatabaseError(RuntimeError):
    """Raised when a database operation fails."""


def _use_supabase() -> bool:
    return bool(_SUPABASE_URL.strip() and _SUPABASE_KEY.strip())


def _get_client():
    from supabase import create_client
    return create_client(_SUPABASE_URL, _SUPABASE_KEY)


def _run(fn):
    """Run a synchronous supabase-py call in a thread pool."""
    return asyncio.to_thread(fn)


# ── SQLite Setup & Helpers ──────────────────────────────────────────────────

def _init_sqlite():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                goal TEXT NOT NULL,
                steps_json TEXT NOT NULL,
                energy_level INTEGER DEFAULT 5,
                low_power_mode INTEGER NOT NULL DEFAULT 0,
                five_second_start TEXT DEFAULT '',
                stuck_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS streaks (
                user_id TEXT PRIMARY KEY,
                current_streak INTEGER NOT NULL DEFAULT 0,
                longest_streak INTEGER NOT NULL DEFAULT 0,
                last_completed TEXT,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
        """)
        conn.commit()


_init_sqlite()


# ── Users ──────────────────────────────────────────────────────────────────

async def create_user() -> dict:
    """Insert a new anonymous user row. Returns the created row."""
    if _use_supabase():
        try:
            sb = _get_client()
            result = await _run(lambda: sb.table("users").insert({}).execute())
            _check(result, "create_user")
            return result.data[0]
        except Exception as e:
            logger.warning(f"Supabase create_user failed, falling back to SQLite: {e}")

    # SQLite fallback
    user_id = str(uuid.uuid4())
    now_iso = datetime.now(timezone.utc).isoformat()
    def _insert():
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("INSERT INTO users (id, created_at) VALUES (?, ?)", (user_id, now_iso))
            conn.commit()
    await asyncio.to_thread(_insert)
    return {"id": user_id, "created_at": now_iso}


async def get_user(user_id: str) -> Optional[dict]:
    if _use_supabase():
        try:
            sb = _get_client()
            result = await _run(
                lambda: sb.table("users").select("*").eq("id", user_id).maybe_single().execute()
            )
            return result.data
        except Exception as e:
            logger.warning(f"Supabase get_user failed, falling back to SQLite: {e}")

    # SQLite fallback
    def _query():
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, created_at FROM users WHERE id = ?", (user_id,))
            row = cursor.fetchone()
            if row:
                return {"id": row[0], "created_at": row[1]}
            return None
    return await asyncio.to_thread(_query)


# ── Sessions ───────────────────────────────────────────────────────────────

async def create_session(
    *,
    user_id: str,
    goal: str,
    steps_json: list[dict],
    energy_level: int,
    low_power_mode: bool,
    five_second_start: str,
) -> dict:
    """Insert a new in-progress session row. Returns the created row."""
    if _use_supabase():
        try:
            sb = _get_client()
            payload = {
                "user_id": user_id,
                "goal": goal,
                "steps_json": steps_json,
                "energy_level": energy_level,
                "low_power_mode": low_power_mode,
                "five_second_start": five_second_start,
            }
            result = await _run(lambda: sb.table("sessions").insert(payload).execute())
            _check(result, "create_session")
            return result.data[0]
        except Exception as e:
            logger.warning(f"Supabase create_session failed, falling back to SQLite: {e}")

    # SQLite fallback
    session_id = str(uuid.uuid4())
    now_iso = datetime.now(timezone.utc).isoformat()
    def _insert():
        with sqlite3.connect(DB_PATH) as conn:
            # Ensure user exists in SQLite
            conn.execute(
                "INSERT OR IGNORE INTO users (id, created_at) VALUES (?, ?)",
                (user_id, now_iso),
            )
            conn.execute(
                """
                INSERT INTO sessions (
                    id, user_id, goal, steps_json, energy_level,
                    low_power_mode, five_second_start, stuck_count, created_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, NULL)
                """,
                (
                    session_id,
                    user_id,
                    goal,
                    json.dumps(steps_json),
                    energy_level,
                    1 if low_power_mode else 0,
                    five_second_start,
                    now_iso,
                ),
            )
            conn.commit()

    await asyncio.to_thread(_insert)
    return {
        "id": session_id,
        "user_id": user_id,
        "goal": goal,
        "steps_json": steps_json,
        "energy_level": energy_level,
        "low_power_mode": low_power_mode,
        "five_second_start": five_second_start,
        "stuck_count": 0,
        "created_at": now_iso,
        "completed_at": None,
    }


async def complete_session(session_id: str, stuck_count: int = 0) -> dict:
    """Stamp completed_at on a session. Returns the updated row."""
    now_iso = datetime.now(timezone.utc).isoformat()
    if _use_supabase():
        try:
            sb = _get_client()
            payload = {
                "completed_at": now_iso,
                "stuck_count": stuck_count,
            }
            result = await _run(
                lambda: sb.table("sessions")
                    .update(payload)
                    .eq("id", session_id)
                    .execute()
            )
            _check(result, "complete_session")
            if result.data:
                return result.data[0]
        except Exception as e:
            logger.warning(f"Supabase complete_session failed, falling back to SQLite: {e}")

    # SQLite fallback
    def _update():
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE sessions
                SET completed_at = ?, stuck_count = ?
                WHERE id = ?
                """,
                (now_iso, stuck_count, session_id),
            )
            conn.commit()
            cursor.execute(
                """
                SELECT id, user_id, goal, steps_json, energy_level,
                       low_power_mode, five_second_start, stuck_count, created_at, completed_at
                FROM sessions WHERE id = ?
                """,
                (session_id,),
            )
            row = cursor.fetchone()
            if not row:
                raise DatabaseError(f"Session '{session_id}' not found")
            return {
                "id": row[0],
                "user_id": row[1],
                "goal": row[2],
                "steps_json": json.loads(row[3]) if row[3] else [],
                "energy_level": row[4],
                "low_power_mode": bool(row[5]),
                "five_second_start": row[6],
                "stuck_count": row[7],
                "created_at": row[8],
                "completed_at": row[9],
            }

    return await asyncio.to_thread(_update)


async def get_session(session_id: str) -> Optional[dict]:
    if _use_supabase():
        try:
            sb = _get_client()
            result = await _run(
                lambda: sb.table("sessions")
                    .select("*")
                    .eq("id", session_id)
                    .maybe_single()
                    .execute()
            )
            return result.data
        except Exception as e:
            logger.warning(f"Supabase get_session failed, falling back to SQLite: {e}")

    # SQLite fallback
    def _query():
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, user_id, goal, steps_json, energy_level,
                       low_power_mode, five_second_start, stuck_count, created_at, completed_at
                FROM sessions WHERE id = ?
                """,
                (session_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return {
                "id": row[0],
                "user_id": row[1],
                "goal": row[2],
                "steps_json": json.loads(row[3]) if row[3] else [],
                "energy_level": row[4],
                "low_power_mode": bool(row[5]),
                "five_second_start": row[6],
                "stuck_count": row[7],
                "created_at": row[8],
                "completed_at": row[9],
            }
    return await asyncio.to_thread(_query)


async def get_user_sessions(
    user_id: str,
    limit: int = 20,
    completed_only: bool = False,
) -> list[dict]:
    """Return recent sessions for a user, newest first."""
    if _use_supabase():
        try:
            sb = _get_client()
            def _query_sb():
                q = (
                    sb.table("sessions")
                      .select("id, goal, energy_level, low_power_mode, stuck_count, created_at, completed_at")
                      .eq("user_id", user_id)
                      .order("created_at", desc=True)
                      .limit(limit)
                )
                if completed_only:
                    q = q.not_.is_("completed_at", "null")
                return q.execute()

            result = await _run(_query_sb)
            _check(result, "get_user_sessions")
            return result.data
        except Exception as e:
            logger.warning(f"Supabase get_user_sessions failed, falling back to SQLite: {e}")

    # SQLite fallback
    def _query():
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            sql = """
                SELECT id, goal, energy_level, low_power_mode, stuck_count, created_at, completed_at
                FROM sessions
                WHERE user_id = ?
            """
            if completed_only:
                sql += " AND completed_at IS NOT NULL"
            sql += " ORDER BY created_at DESC LIMIT ?"
            cursor.execute(sql, (user_id, limit))
            rows = cursor.fetchall()
            return [
                {
                    "id": r[0],
                    "goal": r[1],
                    "energy_level": r[2],
                    "low_power_mode": bool(r[3]),
                    "stuck_count": r[4],
                    "created_at": r[5],
                    "completed_at": r[6],
                }
                for r in rows
            ]
    return await asyncio.to_thread(_query)


# ── Streaks ────────────────────────────────────────────────────────────────

async def recalculate_streak(user_id: str) -> dict:
    """
    Recalculate streak for user based on completed sessions.
    Returns: {"current_streak": int, "longest_streak": int, "last_completed": str | None}
    """
    if _use_supabase():
        try:
            sb = _get_client()
            result = await _run(
                lambda: sb.rpc("recalculate_streak", {"p_user_id": user_id}).execute()
            )
            _check(result, "recalculate_streak")
            if result.data:
                row = result.data[0]
                return {
                    "current_streak": row["current_streak"],
                    "longest_streak": row["longest_streak"],
                    "last_completed": row["last_completed"],
                }
        except Exception as e:
            logger.warning(f"Supabase recalculate_streak failed, falling back to SQLite: {e}")

    # SQLite streak engine
    now_utc = datetime.now(timezone.utc)
    now_iso = now_utc.isoformat()
    today = now_utc.date()

    def _calc():
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT DISTINCT substr(completed_at, 1, 10) as comp_date
                FROM sessions
                WHERE user_id = ? AND completed_at IS NOT NULL
                ORDER BY comp_date ASC
                """,
                (user_id,),
            )
            rows = cursor.fetchall()
            dates = [datetime.strptime(r[0], "%Y-%m-%d").date() for r in rows if r[0]]

            if not dates:
                cursor.execute(
                    """
                    INSERT INTO streaks (user_id, current_streak, longest_streak, last_completed, updated_at)
                    VALUES (?, 0, 0, NULL, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        current_streak = 0,
                        updated_at = excluded.updated_at
                    """,
                    (user_id, now_iso),
                )
                conn.commit()
                return {"current_streak": 0, "longest_streak": 0, "last_completed": None}

            # 1. Longest streak
            longest = 1
            running = 1
            for i in range(1, len(dates)):
                if (dates[i] - dates[i-1]).days == 1:
                    running += 1
                elif (dates[i] - dates[i-1]).days > 1:
                    running = 1
                if running > longest:
                    longest = running

            # 2. Current streak
            last_date = dates[-1]
            days_since_last = (today - last_date).days

            if days_since_last > 1:
                current = 0  # Streak broke
            else:
                current = 1
                for i in range(len(dates) - 1, 0, -1):
                    if (dates[i] - dates[i-1]).days == 1:
                        current += 1
                    else:
                        break

            if current > longest:
                longest = current

            last_comp_str = last_date.strftime("%Y-%m-%d")

            cursor.execute(
                """
                INSERT INTO streaks (user_id, current_streak, longest_streak, last_completed, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    current_streak = excluded.current_streak,
                    longest_streak = max(streaks.longest_streak, excluded.longest_streak),
                    last_completed = excluded.last_completed,
                    updated_at = excluded.updated_at
                """,
                (user_id, current, longest, last_comp_str, now_iso),
            )
            conn.commit()

            return {
                "current_streak": current,
                "longest_streak": longest,
                "last_completed": last_comp_str,
            }

    return await asyncio.to_thread(_calc)


async def get_streak(user_id: str) -> Optional[dict]:
    """Read current streaks row without recalculating."""
    if _use_supabase():
        try:
            sb = _get_client()
            result = await _run(
                lambda: sb.table("streaks")
                    .select("*")
                    .eq("user_id", user_id)
                    .maybe_single()
                    .execute()
            )
            if result.data:
                return result.data
        except Exception as e:
            logger.warning(f"Supabase get_streak failed, falling back to SQLite: {e}")

    # SQLite fallback
    def _query():
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT user_id, current_streak, longest_streak, last_completed, updated_at
                FROM streaks
                WHERE user_id = ?
                """,
                (user_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return {
                "user_id": row[0],
                "current_streak": row[1],
                "longest_streak": row[2],
                "last_completed": row[3],
                "updated_at": row[4],
            }
    return await asyncio.to_thread(_query)


# ── Helper ─────────────────────────────────────────────────────────────────

def _check(result, operation: str) -> None:
    """Raise DatabaseError if supabase-py returned an error."""
    if hasattr(result, "error") and result.error:
        raise DatabaseError(f"[{operation}] Supabase error: {result.error}")

