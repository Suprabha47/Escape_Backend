"""
store.py — In-memory session store for Escape
=============================================
Default: pure in-memory dict (zero dependencies, great for dev).
Upgrade path: swap SessionStore for RedisSessionStore (stub included)
when you add Supabase/Redis in production.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Optional

from models import MicroStep, SessionRecord, SubStep


class SessionStore:
    """
    Thread-safe in-memory store.
    Sessions are lost on process restart — add persistence (Redis/Postgres)
    for production by implementing the same interface in a subclass.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, SessionRecord] = {}
        self._lock = threading.Lock()

    # ── Write ──────────────────────────────────────────────────────────────

    def save(
        self,
        session_id: str,
        goal: str,
        answers: list[str],
        blueprint: dict,
    ) -> SessionRecord:
        """
        Persist a freshly generated blueprint as a new session record.
        Called at the end of a successful /intake stream.
        """
        raw_steps = blueprint.get("micro_steps", [])
        steps = [
            MicroStep(
                id=s.get("id", i + 1),
                text=s["text"],
                duration=s.get("duration", "5 min"),
                low_power_substitute=s.get("low_power_substitute"),
            )
            for i, s in enumerate(raw_steps)
        ]

        record = SessionRecord(
            session_id=session_id,
            goal=goal,
            answers=answers,
            energy_level=blueprint.get("energy_level", 5),
            low_power_mode=blueprint.get("low_power_mode", False),
            five_second_start=blueprint.get("five_second_start", ""),
            micro_steps=steps,
            created_at=datetime.now(timezone.utc),
        )

        with self._lock:
            self._sessions[session_id] = record

        return record

    def mark_step_complete(self, session_id: str, step_id: int) -> bool:
        """Stamp a completion timestamp on a step. Returns False if not found."""
        with self._lock:
            record = self._sessions.get(session_id)
            if not record:
                return False
            for step in record.micro_steps:
                if step.id == step_id:
                    step.completed_at = datetime.now(timezone.utc)
                    # If all steps done, stamp session completion
                    if all(s.completed_at for s in record.micro_steps):
                        record.completed_at = datetime.now(timezone.utc)
                    return True
        return False

    def append_stuck(
        self,
        session_id: str,
        parent_step_id: int,
        substeps: list[SubStep],
    ) -> None:
        """
        Record a stuck event — the sub-steps generated for a stuck moment —
        against both the session-level log and the specific step.
        """
        with self._lock:
            record = self._sessions.get(session_id)
            if not record:
                return
            event = {
                "parent_step_id": parent_step_id,
                "substeps": [s.model_dump() for s in substeps],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            record.stuck_events.append(event)
            for step in record.micro_steps:
                if step.id == parent_step_id:
                    step.stuck_events.append(substeps)
                    break

    # ── Read ───────────────────────────────────────────────────────────────

    def get(self, session_id: str) -> Optional[SessionRecord]:
        with self._lock:
            return self._sessions.get(session_id)

    def all_sessions(self) -> list[SessionRecord]:
        with self._lock:
            return list(self._sessions.values())

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None


# ── Singleton used by main.py ──────────────────────────────────────────────
session_store = SessionStore()


# ── Redis upgrade stub ─────────────────────────────────────────────────────
# Uncomment and flesh out when moving to production:
#
# import redis.asyncio as redis
# import json
#
# class RedisSessionStore(SessionStore):
#     def __init__(self, url: str = "redis://localhost:6379"):
#         self._redis = redis.from_url(url)
#         self._ttl = 60 * 60 * 24  # 24h TTL
#
#     async def save(self, session_id, goal, answers, blueprint):
#         record = super().save(session_id, goal, answers, blueprint)
#         await self._redis.setex(
#             f"session:{session_id}",
#             self._ttl,
#             record.model_dump_json(),
#         )
#         return record
#
#     async def get(self, session_id):
#         raw = await self._redis.get(f"session:{session_id}")
#         if raw is None:
#             return None
#         return SessionRecord.model_validate_json(raw)
