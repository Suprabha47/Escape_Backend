"""
models.py — Pydantic request/response schemas for Escape API
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ── Intake ─────────────────────────────────────────────────────────────────

class IntakeRequest(BaseModel):
    goal: str = Field(..., min_length=3, max_length=1000, description="The user's stated goal")
    answers: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="Answers to the 2-3 clarifying questions, in order",
    )
    energy_level: Optional[int] = Field(
        default=None,
        ge=1,
        le=10,
        description="Explicit energy level 1-10 if already collected; otherwise inferred by Claude",
    )

    model_config = {"json_schema_extra": {
        "example": {
            "goal": "Write the introduction for my research paper",
            "answers": [
                "I keep skipping the opening hook — I don't know how to grab attention",
                "4",
                "Yes, I have my notes and outline open",
            ],
            "energy_level": 4,
        }
    }}


class IntakeStreamChunk(BaseModel):
    """Shape of each SSE data payload sent to the client."""
    type: Literal["text_delta", "blueprint", "error", "done"]
    text: Optional[str] = None        # present on text_delta
    payload: Optional[dict] = None    # present on blueprint
    message: Optional[str] = None     # present on error


# ── Stuck ──────────────────────────────────────────────────────────────────

class StuckRequest(BaseModel):
    step_id: int = Field(..., description="ID of the step the user is stuck on")
    step_text: str = Field(..., min_length=3, description="Full text of the stuck step")
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for persisting the stuck event to history",
    )

    model_config = {"json_schema_extra": {
        "example": {
            "step_id": 3,
            "step_text": "Write the opening sentence of the introduction",
            "session_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
        }
    }}


class SubStep(BaseModel):
    id: int
    text: str
    duration: Literal["< 1 min", "2 min", "5 min", "8 min"] = "< 1 min"


class StuckResponse(BaseModel):
    parent_step_id: int
    substeps: list[SubStep] = Field(..., min_length=3, max_length=3)


# ── Micro step (used in session history) ──────────────────────────────────

class MicroStep(BaseModel):
    id: int
    text: str
    duration: str
    low_power_substitute: Optional[str] = None
    completed_at: Optional[datetime] = None
    stuck_events: list[list[SubStep]] = Field(default_factory=list)


# ── Session ────────────────────────────────────────────────────────────────

class SessionRecord(BaseModel):
    session_id: str
    goal: str
    answers: list[str]
    energy_level: int
    low_power_mode: bool
    five_second_start: str
    micro_steps: list[MicroStep]
    created_at: datetime
    completed_at: Optional[datetime] = None
    stuck_events: list[dict] = Field(default_factory=list)


class SessionResponse(SessionRecord):
    """What GET /session/{id} returns — same as SessionRecord."""
    pass


# ── Supabase user ──────────────────────────────────────────────────────────

class UserResponse(BaseModel):
    id: str
    created_at: datetime


# ── Supabase session (DB row shape, distinct from in-memory SessionRecord) ─

class DbSessionRow(BaseModel):
    """Maps 1:1 to the sessions table row returned by Supabase."""
    id: str
    user_id: str
    goal: str
    steps_json: list[dict]
    energy_level: int
    low_power_mode: bool
    five_second_start: Optional[str] = None
    stuck_count: int = 0
    created_at: datetime
    completed_at: Optional[datetime] = None


class DbSessionSummary(BaseModel):
    """Lightweight row for list endpoints — no steps_json payload."""
    id: str
    goal: str
    energy_level: int
    low_power_mode: bool
    stuck_count: int
    created_at: datetime
    completed_at: Optional[datetime] = None


# ── Complete session request ───────────────────────────────────────────────

class CompleteSessionRequest(BaseModel):
    user_id: str = Field(..., description="UUID of the user completing the session")
    stuck_count: int = Field(default=0, ge=0, description="How many times the user clicked 'I'm stuck'")

    model_config = {"json_schema_extra": {
        "example": {
            "user_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
            "stuck_count": 2,
        }
    }}


class CompleteSessionResponse(BaseModel):
    session: DbSessionRow
    streak: "StreakResponse"


# ── Streak ─────────────────────────────────────────────────────────────────

class StreakResponse(BaseModel):
    user_id: str
    current_streak: int
    longest_streak: int
    last_completed: Optional[str] = None  # ISO date string "YYYY-MM-DD"
    updated_at: Optional[datetime] = None


# ── User sessions list ─────────────────────────────────────────────────────

class UserSessionsResponse(BaseModel):
    user_id: str
    sessions: list[DbSessionSummary]
    total: int
