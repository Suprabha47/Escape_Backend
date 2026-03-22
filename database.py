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

from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

_SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
_SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")


class DatabaseError(RuntimeError):
    """Raised when a Supabase operation fails."""


def _get_client() -> Client:
    """Create a fresh Supabase client. Cheap to call — no long-lived connection."""
    return create_client(_SUPABASE_URL, _SUPABASE_KEY)


def _run(fn):
    """Run a synchronous supabase-py call in a thread pool."""
    return asyncio.to_thread(fn)


# ── Users ──────────────────────────────────────────────────────────────────

async def create_user() -> dict:
    """
    Insert a new anonymous user row. Returns the created row.
    In production, replace this with a trigger on auth.users.
    """
    sb = _get_client()
    result = await _run(lambda: sb.table("users").insert({}).execute())
    _check(result, "create_user")
    return result.data[0]


async def get_user(user_id: str) -> Optional[dict]:
    sb = _get_client()
    result = await _run(
        lambda: sb.table("users").select("*").eq("id", user_id).maybe_single().execute()
    )
    return result.data  # None if not found


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
    """
    Insert a new in-progress session row. Returns the created row.
    Called at the end of a successful /intake stream.
    """
    sb = _get_client()
    payload = {
        "user_id": user_id,
        "goal": goal,
        "steps_json": steps_json,           # stored as JSONB
        "energy_level": energy_level,
        "low_power_mode": low_power_mode,
        "five_second_start": five_second_start,
    }
    result = await _run(lambda: sb.table("sessions").insert(payload).execute())
    _check(result, "create_session")
    return result.data[0]


async def complete_session(session_id: str, stuck_count: int = 0) -> dict:
    """
    Stamp completed_at on a session. Returns the updated row.
    Call this when the user marks the last step done.
    """
    sb = _get_client()
    payload = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "stuck_count": stuck_count,
    }
    result = await _run(
        lambda: sb.table("sessions")
            .update(payload)
            .eq("id", session_id)
            .execute()
    )
    _check(result, "complete_session")
    if not result.data:
        raise DatabaseError(f"Session '{session_id}' not found")
    return result.data[0]


async def get_session(session_id: str) -> Optional[dict]:
    sb = _get_client()
    result = await _run(
        lambda: sb.table("sessions")
            .select("*")
            .eq("id", session_id)
            .maybe_single()
            .execute()
    )
    return result.data


async def get_user_sessions(
    user_id: str,
    limit: int = 20,
    completed_only: bool = False,
) -> list[dict]:
    """Return recent sessions for a user, newest first."""
    sb = _get_client()

    def _query():
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

    result = await _run(_query)
    _check(result, "get_user_sessions")
    return result.data


# ── Streaks ────────────────────────────────────────────────────────────────

async def recalculate_streak(user_id: str) -> dict:
    """
    Call the recalculate_streak() Postgres function and return the result.
    Shape: {"current_streak": int, "longest_streak": int, "last_completed": str | None}
    """
    sb = _get_client()
    result = await _run(
        lambda: sb.rpc("recalculate_streak", {"p_user_id": user_id}).execute()
    )
    _check(result, "recalculate_streak")

    if not result.data:
        return {"current_streak": 0, "longest_streak": 0, "last_completed": None}

    row = result.data[0]
    return {
        "current_streak": row["current_streak"],
        "longest_streak": row["longest_streak"],
        "last_completed": row["last_completed"],
    }


async def get_streak(user_id: str) -> Optional[dict]:
    """Read the current streaks row without recalculating."""
    sb = _get_client()
    result = await _run(
        lambda: sb.table("streaks")
            .select("*")
            .eq("user_id", user_id)
            .maybe_single()
            .execute()
    )
    return result.data


# ── Helper ─────────────────────────────────────────────────────────────────

def _check(result, operation: str) -> None:
    """Raise DatabaseError if supabase-py returned an error."""
    # supabase-py raises on HTTP errors, but be defensive about empty responses
    if hasattr(result, "error") and result.error:
        raise DatabaseError(f"[{operation}] Supabase error: {result.error}")
