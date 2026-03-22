"""
Escape — FastAPI Backend (Azure OpenAI edition)
===============================================
Endpoints:
  POST /intake       — streams micro_steps via Azure OpenAI SSE
  POST /stuck        — returns 3 sub-steps for a stuck step
  GET  /session/{id} — in-memory session history
  GET  /health       — health check

Run:
  uvicorn main:app --reload --port 8000
"""

import json
import uuid
import os
from datetime import datetime, timezone
from typing import AsyncIterator

from openai import AsyncAzureOpenAI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

load_dotenv()

from models import (
    IntakeRequest,
    IntakeStreamChunk,
    StuckRequest,
    StuckResponse,
    SessionResponse,
    SubStep,
)
from store import session_store
from prompts import SYSTEM_PROMPT, build_intake_messages, build_stuck_messages

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(title="Escape API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = AsyncAzureOpenAI(
    api_key=os.environ.get("AZURE_OPENAI_KEY", ""),
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
    api_version="2024-12-01-preview",
)

DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4-nano")


# ── /intake — streaming SSE ────────────────────────────────────────────────
async def stream_intake(request: IntakeRequest, session_id: str) -> AsyncIterator[str]:
    messages = build_intake_messages(request)
    accumulated = ""

    try:
        async with client.chat.completions.stream(
            model=DEPLOYMENT,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            max_tokens=2048,
        ) as stream:
            async for text in stream.text_deltas:
                accumulated += text
                chunk = IntakeStreamChunk(type="text_delta", text=text)
                yield f"data: {chunk.model_dump_json()}\n\n"

        blueprint = _parse_blueprint(accumulated)

        session_store.save(
            session_id=session_id,
            goal=request.goal,
            answers=request.answers,
            blueprint=blueprint,
        )

        payload_chunk = IntakeStreamChunk(type="blueprint", payload=blueprint)
        yield f"data: {payload_chunk.model_dump_json()}\n\n"

    except Exception as e:
        yield f"data: {IntakeStreamChunk(type='error', message=str(e)).model_dump_json()}\n\n"
    finally:
        yield f"data: {IntakeStreamChunk(type='done').model_dump_json()}\n\n"


@app.post("/intake")
async def intake(request: IntakeRequest):
    session_id = str(uuid.uuid4())
    return StreamingResponse(
        stream_intake(request, session_id),
        media_type="text/event-stream",
        headers={
            "X-Session-Id": session_id,
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── /stuck ─────────────────────────────────────────────────────────────────
@app.post("/stuck", response_model=StuckResponse)
async def stuck(request: StuckRequest):
    messages = build_stuck_messages(request)

    try:
        response = await client.chat.completions.create(
            model=DEPLOYMENT,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            max_tokens=512,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"API error: {str(e)}")

    raw = response.choices[0].message.content or ""

    try:
        data = _parse_json_from_text(raw)
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse sub-steps: {e}")

    substeps_raw = data.get("substeps", [])
    if len(substeps_raw) != 3:
        raise HTTPException(status_code=422, detail=f"Expected 3 substeps, got {len(substeps_raw)}")

    substeps = [
        SubStep(id=i + 1, text=s["text"], duration=s.get("duration", "< 1 min"))
        for i, s in enumerate(substeps_raw)
    ]

    if request.session_id:
        session_store.append_stuck(
            session_id=request.session_id,
            parent_step_id=request.step_id,
            substeps=substeps,
        )

    return StuckResponse(parent_step_id=request.step_id, substeps=substeps)


# ── GET /session/{id} ──────────────────────────────────────────────────────
@app.get("/session/{session_id}", response_model=SessionResponse)
async def get_session_memory(session_id: str):
    record = session_store.get(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return record


# ── Health ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


# ── Supabase stubs ─────────────────────────────────────────────────────────
_SUPABASE_MSG = "Supabase not configured yet."

@app.post("/users", status_code=503)
async def create_user():
    raise HTTPException(status_code=503, detail=_SUPABASE_MSG)

@app.get("/users/{user_id}/sessions", status_code=503)
async def list_user_sessions(user_id: str):
    raise HTTPException(status_code=503, detail=_SUPABASE_MSG)

@app.get("/sessions/{session_id}", status_code=503)
async def get_session_db(session_id: str):
    raise HTTPException(status_code=503, detail=_SUPABASE_MSG)

@app.post("/sessions/{session_id}/complete", status_code=503)
async def complete_session(session_id: str):
    raise HTTPException(status_code=503, detail=_SUPABASE_MSG)

@app.get("/users/{user_id}/streak", status_code=503)
async def get_streak(user_id: str):
    raise HTTPException(status_code=503, detail=_SUPABASE_MSG)


# ── Helpers ────────────────────────────────────────────────────────────────
def _parse_json_from_text(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(line for line in lines if not line.startswith("```")).strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object found in response")
    return json.loads(text[start:end])


def _parse_blueprint(raw: str) -> dict:
    data = _parse_json_from_text(raw)
    required = {"micro_steps", "five_second_start", "energy_level", "low_power_mode"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"Blueprint missing fields: {missing}")
    steps_count = len(data["micro_steps"])
    if not (10 <= steps_count <= 15):
        data["_warning"] = f"Expected 10-15 steps, got {steps_count}"
    return data