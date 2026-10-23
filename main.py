"""
Escape — FastAPI Backend (Azure OpenAI edition)
===============================================
Endpoints:
  POST /intake                        — streams micro_steps via Azure OpenAI SSE
  POST /stuck                         — returns 3 sub-steps for a stuck step
  GET  /session/{id}                  — in-memory session history
  POST /users                         — create anonymous user (Supabase)
  GET  /users/{user_id}/sessions      — list user sessions (Supabase)
  GET  /sessions/{session_id}         — get session by DB id (Supabase)
  POST /sessions/{session_id}/complete — mark session complete + recalc streak
  GET  /users/{user_id}/streak        — get streak data
  GET  /health                        — health check

Run:
  uvicorn main:app --reload --port 8000
"""

import asyncio
import json
import uuid
import os
import re
from datetime import datetime, timezone
from typing import AsyncIterator

import httpx
from openai import AsyncOpenAI, AsyncAzureOpenAI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from models import (
    IntakeRequest,
    IntakeStreamChunk,
    StuckRequest,
    StuckResponse,
    SessionResponse,
    SubStep,
    UserResponse,
    DbSessionRow,
    DbSessionSummary,
    CompleteSessionRequest,
    CompleteSessionResponse,
    StreakResponse,
    UserSessionsResponse,
)
from store import session_store
from prompts import SYSTEM_PROMPT, CHAT_SYSTEM_PROMPT, build_intake_messages, build_stuck_messages
import database as db
from database import DatabaseError

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(title="Escape API", version="2.0.0")

# Support comma-separated origins via env var for Render + local dev
_raw_origins = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000",
)
_allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if os.environ.get("APP_ENV") == "development" else _allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Multi-Provider AI Client Setup ─────────────────────────────────────────
# Automatically detects:
# 1. Google Gemini (free via Google AI Studio)
# 2. Groq (free via console.groq.com)
# 3. OpenAI
# 4. Azure OpenAI
_gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
_groq_key = os.environ.get("GROQ_API_KEY", "").strip()
_openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
_azure_key = os.environ.get("AZURE_OPENAI_KEY", "").strip()
_azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()

if _gemini_key:
    client = AsyncOpenAI(
        api_key=_gemini_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        timeout=4.0,
        max_retries=0,
    )
    DEPLOYMENT = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")
elif _groq_key:
    client = AsyncOpenAI(
        api_key=_groq_key,
        base_url="https://api.groq.com/openai/v1",
        timeout=4.0,
        max_retries=0,
    )
    DEPLOYMENT = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
elif _openai_key:
    client = AsyncOpenAI(
        api_key=_openai_key,
        timeout=4.0,
        max_retries=0,
    )
    DEPLOYMENT = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
elif _azure_key and _azure_endpoint:
    client = AsyncAzureOpenAI(
        api_key=_azure_key,
        azure_endpoint=_azure_endpoint,
        api_version="2024-12-01-preview",
        timeout=4.0,
        max_retries=0,
    )
    DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4-nano")
else:
    client = None
    DEPLOYMENT = "gemini-flash-lite-latest"


# ── /chat — Interactive Chat Endpoint ──────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]

class ChatResponse(BaseModel):
    reply: str


def _generate_fallback_chat(messages: list[ChatMessage]) -> str:
    """Smart behavioral fallback when external AI key is in high-demand or unavailable."""
    user_msgs = [m.content for m in messages if m.role == "user"]
    if not user_msgs:
        return json.dumps({
            "phase": "question",
            "question": "What task or project are you avoiding right now?",
            "options": ["Writing / Content", "Coding / Bug fix", "Cleaning / Chore", "Study / Research"]
        })

    first_msg = user_msgs[0].lower()
    last_msg = user_msgs[-1].lower()

    # Step generation triggered (after 6 questions or early bypass trigger)
    is_generate_intent = bool(
        re.search(r"\b(generate (the )?steps|atomic steps|generate now|ready to begin)\b", last_msg)
    )
    if len(user_msgs) >= 7 or is_generate_intent:
        raw_goal = user_msgs[0]
        for prefix in ["my goal:", "goal:"]:
            if raw_goal.lower().startswith(prefix):
                raw_goal = raw_goal[len(prefix):].strip()
        goal_text = raw_goal or "Your task"
        goal_lower = goal_text.lower()

        # 1. Gym / Workout / Fitness
        if any(k in goal_lower for k in ["gym", "workout", "exercise", "run", "fitness", "training"]):
            steps = [
                {"id": 1, "text": "Put on your workout shoes or socks right now", "duration": "< 1 min"},
                {"id": 2, "text": "Fill up your water bottle", "duration": "< 1 min"},
                {"id": 3, "text": "Put your workout clothes or gym bag by the front door", "duration": "< 1 min"},
                {"id": 4, "text": "Head out the door and start your commute", "duration": "5 min"},
                {"id": 5, "text": "Walk inside and check in at the gym", "duration": "2 min"},
                {"id": 6, "text": "Do a gentle 3-minute warm-up on treadmill or stretching", "duration": "5 min"},
                {"id": 7, "text": "Complete your first comfortable exercise set", "duration": "5 min"},
                {"id": 8, "text": "Work through your main routine without rushing", "duration": "8 min"},
                {"id": 9, "text": "Take a 2-minute cool down and hydrate", "duration": "2 min"},
                {"id": 10, "text": "Mark this workout complete — you broke the inertia", "duration": "< 1 min"},
            ]
            return json.dumps({
                "phase": "blueprint",
                "steps": steps,
                "energy_level": 6,
                "low_power_mode": False,
                "five_second_start": "Put on workout shoes"
            })

        # 2. Dishes / Kitchen / Sink
        if any(k in goal_lower for k in ["dish", "dishes", "sink", "kitchen", "plates"]):
            steps = [
                {"id": 1, "text": "Clear one empty space on counter or drying rack next to sink", "duration": "< 1 min"},
                {"id": 2, "text": "Scrape leftover food scraps from dishes into trash", "duration": "< 1 min"},
                {"id": 3, "text": "Stack plates, bowls, and mugs into separate piles", "duration": "2 min"},
                {"id": 4, "text": "Turn on hot water and fill sink or soapy sponge", "duration": "< 1 min"},
                {"id": 5, "text": "Wash all glassware and coffee mugs first", "duration": "2 min"},
                {"id": 6, "text": "Wash silverware and cooking utensils", "duration": "2 min"},
                {"id": 7, "text": "Wash plates and bowls with sponge", "duration": "5 min"},
                {"id": 8, "text": "Scrub any remaining pots or greasy pans", "duration": "5 min"},
                {"id": 9, "text": "Rinse all soap away and place items on drying rack", "duration": "2 min"},
                {"id": 10, "text": "Wipe down sink basin and hang dish towel — kitchen is reset", "duration": "< 1 min"},
            ]
            return json.dumps({
                "phase": "blueprint",
                "steps": steps,
                "energy_level": 6,
                "low_power_mode": False,
                "five_second_start": "Clear drying rack space"
            })

        # 3. Room Cleaning / Laundry / Declutter
        if any(k in goal_lower for k in ["clean", "room", "laundry", "tidy", "organize", "trash", "bed"]):
            steps = [
                {"id": 1, "text": "Grab one trash bag and throw away 5 obvious pieces of garbage", "duration": "< 1 min"},
                {"id": 2, "text": "Gather all dirty clothes and drop them into laundry hamper", "duration": "2 min"},
                {"id": 3, "text": "Clear everything off the bed or primary desk surface", "duration": "2 min"},
                {"id": 4, "text": "Put away books, mugs, or items that belong in other rooms", "duration": "3 min"},
                {"id": 5, "text": "Wipe down the main table or desk surface with a damp cloth", "duration": "2 min"},
                {"id": 6, "text": "Make the bed or straighten the sheets and pillows", "duration": "2 min"},
                {"id": 7, "text": "Do a quick 2-minute sweep or vacuum of the open floor space", "duration": "3 min"},
                {"id": 8, "text": "Take the trash bag out to the main bin", "duration": "2 min"},
                {"id": 9, "text": "Put fresh water or open a window for fresh air", "duration": "< 1 min"},
                {"id": 10, "text": "Step back and take a breath — your space is reset", "duration": "< 1 min"},
            ]
            return json.dumps({
                "phase": "blueprint",
                "steps": steps,
                "energy_level": 6,
                "low_power_mode": False,
                "five_second_start": "Grab a trash bag"
            })

        # 4. Coding / Software Development / Bug Fixing
        if any(k in goal_lower for k in ["code", "bug", "program", "app", "dev", "api", "database"]):
            steps = [
                {"id": 1, "text": "Open your code editor and terminal to the project folder", "duration": "< 1 min"},
                {"id": 2, "text": "Run the dev server or test command to confirm current state", "duration": "< 1 min"},
                {"id": 3, "text": "Open the single source file where the change is needed", "duration": "2 min"},
                {"id": 4, "text": "Add a log statement or breakpoint to observe the exact state", "duration": "2 min"},
                {"id": 5, "text": "Write the first 5 lines of the core implementation or fix", "duration": "5 min"},
                {"id": 6, "text": "Save and verify output in the terminal or browser", "duration": "2 min"},
                {"id": 7, "text": "Handle the edge case or error handling branch", "duration": "5 min"},
                {"id": 8, "text": "Remove temporary debug logs and format the code", "duration": "2 min"},
                {"id": 9, "text": "Run tests or verify full workflow from end to end", "duration": "5 min"},
                {"id": 10, "text": "Commit the change with a descriptive message", "duration": "< 1 min"},
            ]
            return json.dumps({
                "phase": "blueprint",
                "steps": steps,
                "energy_level": 6,
                "low_power_mode": False,
                "five_second_start": "Open code editor"
            })

        # 5. Writing / Content / Email / Documentation
        if any(k in goal_lower for k in ["write", "essay", "report", "paper", "email", "post", "blog", "draft"]):
            steps = [
                {"id": 1, "text": f"Open blank document and type working title for: {goal_text}", "duration": "< 1 min"},
                {"id": 2, "text": "Write 3 quick bullet points of what you must communicate", "duration": "2 min"},
                {"id": 3, "text": "Draft a messy, imperfect first opening sentence", "duration": "< 1 min"},
                {"id": 4, "text": "Write 5 uninterrupted minutes without backspacing or judging", "duration": "5 min"},
                {"id": 5, "text": "Review what you've drafted and highlight the strongest point", "duration": "2 min"},
                {"id": 6, "text": "Flesh out the second main section with supporting details", "duration": "5 min"},
                {"id": 7, "text": "Draft the concluding thought or call to action", "duration": "3 min"},
                {"id": 8, "text": "Do a quick read-through to fix awkward wording and typos", "duration": "5 min"},
                {"id": 9, "text": "Format headings, spacing, and final layout", "duration": "2 min"},
                {"id": 10, "text": "Save or send the completed draft", "duration": "< 1 min"},
            ]
            return json.dumps({
                "phase": "blueprint",
                "steps": steps,
                "energy_level": 6,
                "low_power_mode": False,
                "five_second_start": "Open blank document"
            })

        # 6. General Fallback
        steps = [
            {"id": 1, "text": f"Set up the immediate physical or digital space for: {goal_text}", "duration": "< 1 min"},
            {"id": 2, "text": "Put away anything unrelated to this task for the next 15 minutes", "duration": "< 1 min"},
            {"id": 3, "text": "Complete the absolute easiest 60-second mechanical starter action", "duration": "< 1 min"},
            {"id": 4, "text": "Spend 5 uninterrupted minutes on the core piece without judging", "duration": "5 min"},
            {"id": 5, "text": "Check off the first tangible progress milestone", "duration": "2 min"},
            {"id": 6, "text": "Address the hardest remaining point with a quick rough attempt", "duration": "5 min"},
            {"id": 7, "text": "Review what you've accomplished so far", "duration": "2 min"},
            {"id": 8, "text": "Polish and complete the remaining rough edges", "duration": "5 min"},
            {"id": 9, "text": "Do a final quick pass and save or wrap up", "duration": "2 min"},
            {"id": 10, "text": f"Mark {goal_text} complete — you broke the paralysis", "duration": "< 1 min"},
        ]
        return json.dumps({
            "phase": "blueprint",
            "steps": steps,
            "energy_level": 6,
            "low_power_mode": False,
            "five_second_start": f"Set up space for {goal_text[:20]}"
        })

    # Question 6: Energy Level (Anchor)
    if len(user_msgs) == 6:
        if any(k in first_msg for k in ["gym", "workout", "exercise", "run", "fitness"]):
            return json.dumps({
                "phase": "question",
                "question": "Where is your physical energy battery right now (1 to 10)?",
                "options": ["Exhausted / Low battery (1-3)", "Moderate / Steady (4-6)", "Hyped / High energy (7-10)"]
            })
        return json.dumps({
            "phase": "question",
            "question": "Where is your mental battery right now (1 to 10)?",
            "options": ["Drained / Foggy (1-3)", "Neutral / Steady (4-6)", "High Energy / Focused (7-10)"]
        })

    # Question 5: Available Time (Anchor)
    if len(user_msgs) == 5:
        if any(k in first_msg for k in ["gym", "workout", "exercise", "run", "fitness"]):
            return json.dumps({
                "phase": "question",
                "question": "How much time do you realistically want to spend at the gym today?",
                "options": ["Quick 20-min session", "Solid 40-45 minutes", "Full 60-75 min workout", "Just 10 mins to break inertia"]
            })
        if any(k in first_msg for k in ["clean", "room", "laundry", "dishes", "organize"]):
            return json.dumps({
                "phase": "question",
                "question": "How long do you want to tidy or clean right now?",
                "options": ["10-min speed clean", "20-minute tidy blitz", "30-45 min deep clean", "Just 5 minutes to start"]
            })
        return json.dumps({
            "phase": "question",
            "question": "How much time do you realistically want to spend on this right now?",
            "options": ["15-minute quick sprint", "30 minutes solid", "45 to 60 minutes", "Open-ended deep focus"]
        })

    # Question 4: 30-Second Momentum Seed
    if len(user_msgs) == 4:
        if any(k in first_msg for k in ["gym", "workout", "exercise", "run", "fitness"]):
            return json.dumps({
                "phase": "question",
                "question": "If you only had to do one 30-second mechanical action right now, what would it be?",
                "options": ["Put on gym shoes & socks", "Fill up water bottle", "Grab headphones & keys", "Put gym bag by the front door"]
            })
        if any(k in first_msg for k in ["clean", "room", "laundry", "dishes", "organize"]):
            return json.dumps({
                "phase": "question",
                "question": "If you only had to do one 30-second mechanical action right now, what would it be?",
                "options": ["Pick up 5 items off the floor", "Put clothes in the hamper", "Clear one corner of desk/counter", "Take out the trash bag"]
            })
        if any(k in first_msg for k in ["code", "bug", "program", "app", "dev"]):
            return json.dumps({
                "phase": "question",
                "question": "If you only had to do one 30-second mechanical action right now, what would it be?",
                "options": ["Run the dev server / app", "Open the target source file", "Read the last error traceback", "Add a console.log or breakpoint"]
            })
        if any(k in first_msg for k in ["write", "essay", "report", "paper", "email"]):
            return json.dumps({
                "phase": "question",
                "question": "If you only had to do one 30-second mechanical action right now, what would it be?",
                "options": ["Write 1 messy rough sentence", "Type document title & date", "List 3 bullet headings", "Open document in full screen"]
            })
        return json.dumps({
            "phase": "question",
            "question": "If you only had to do one 30-second mechanical action right now, what would it be?",
            "options": ["Open the needed app / space", "Write down the first micro-step", "Gather the 1-2 items needed", "Set a 5-minute timer"]
        })

    # Question 3: Core Friction / Blocker
    if len(user_msgs) == 3:
        if any(k in first_msg for k in ["gym", "workout", "exercise", "run"]):
            return json.dumps({
                "phase": "question",
                "question": "What part of going feels like the biggest mental hurdle right now?",
                "options": ["Leaving the house", "Feeling physically tired", "Dreading the crowd/noise", "Not sure what workout to do"]
            })
        if any(k in first_msg for k in ["code", "bug", "program", "app", "dev"]):
            return json.dumps({
                "phase": "question",
                "question": "What part of this bug or feature feels most overwhelming?",
                "options": ["Finding where the error starts", "Dealing with messy legacy code", "Writing the logic from scratch", "Just lack of focus"]
            })
        return json.dumps({
            "phase": "question",
            "question": "What specific part of this are you most tempted to put off or avoid?",
            "options": ["Starting from a blank page", "Too many messy thoughts to organize", "Fear of doing it poorly", "Pure boredom & fatigue"]
        })

    # Question 2: Starting State & Environment
    if len(user_msgs) == 2:
        if any(k in first_msg for k in ["gym", "workout", "exercise", "run", "fitness"]):
            return json.dumps({
                "phase": "question",
                "question": "Where are you physically right now?",
                "options": ["On couch or bed", "At desk in work clothes", "Already in workout gear", "Out running errands"]
            })
        if any(k in first_msg for k in ["clean", "room", "laundry", "dishes", "organize"]):
            return json.dumps({
                "phase": "question",
                "question": "What does the immediate space around you look like?",
                "options": ["Clothes scattered on bed/floor", "Desk covered with clutter", "Dishes piled in sink", "Overall disorganized room"]
            })
        if any(k in first_msg for k in ["code", "bug", "program", "app", "dev"]):
            return json.dumps({
                "phase": "question",
                "question": "What is the state of your development environment?",
                "options": ["IDE & terminal already open", "Staring at a blank screen", "Need to pull code & set up DB", "Stuck on an error traceback"]
            })
        return json.dumps({
            "phase": "question",
            "question": "What does your current setup or workspace look like right now?",
            "options": ["Files / tabs already open", "Staring at a blank screen", "Need to log in / locate files", "Messy desk / cluttered space"]
        })

    # Question 1: Scope & Boundary
    if any(k in first_msg for k in ["gym", "workout", "exercise", "run", "fitness"]):
        return json.dumps({
            "phase": "question",
            "question": "What kind of session are we aiming for today?",
            "options": ["Light 20-min maintenance", "Standard weight/cardio routine", "Heavy intense workout", "Just stretching & moving"]
        })
    if any(k in first_msg for k in ["write", "essay", "report", "paper", "email"]):
        return json.dumps({
            "phase": "question",
            "question": "Which specific section or stage are we tackling today?",
            "options": ["Outline & initial hook", "Writing the messy first draft", "Polishing & citations", "Just organizing reference notes"]
        })
    if any(k in first_msg for k in ["clean", "room", "laundry", "dishes", "organize"]):
        return json.dumps({
            "phase": "question",
            "question": "Which specific area needs attention first?",
            "options": ["Floor & laundry pile", "Desk / work surface", "Kitchen sink / dishes", "Quick general declutter"]
        })
    if any(k in first_msg for k in ["code", "bug", "program", "app", "dev"]):
        return json.dumps({
            "phase": "question",
            "question": "What is the primary deliverable for this coding session?",
            "options": ["Fix specific bug / crash", "Scaffold new component/route", "Write tests or refactor", "Set up project / packages"]
        })

    return json.dumps({
        "phase": "question",
        "question": "What specific part of this goal are we aiming to finish today?",
        "options": ["Just getting the rough start down", "Completing the hardest section", "Organizing & preparing everything", "Polishing an existing draft"]
    })


async def _call_gemini_native(messages: list[ChatMessage]) -> str:
    """Call Google Gemini native generateContent API directly with JSON response mode."""
    contents = []
    for m in messages:
        role = "model" if m.role == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": m.content}]})

    model_name = os.environ.get("GEMINI_MODEL", "models/gemini-flash-lite-latest")
    if not model_name.startswith("models/"):
        model_name = f"models/{model_name}"

    url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={_gemini_key}"
    payload = {
        "system_instruction": {"parts": [{"text": CHAT_SYSTEM_PROMPT}]},
        "contents": contents,
        "generationConfig": {
            "response_mime_type": "application/json"
        }
    }

    async with httpx.AsyncClient(timeout=18.0) as http_client:
        resp = await http_client.post(url, json=payload)
        if resp.status_code == 200:
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        elif resp.status_code in (404, 429) and "gemini-flash-lite-latest" not in model_name:
            # Automatic fallback to flash-lite if the configured model hits 429 quota or 404
            fb_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-lite-latest:generateContent?key={_gemini_key}"
            resp_fb = await http_client.post(fb_url, json=payload)
            if resp_fb.status_code == 200:
                data = resp_fb.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
        raise Exception(f"Gemini API returned status {resp.status_code}: {resp.text[:120]}")


async def _call_gemini_stuck(step_id: int, step_text: str) -> list[SubStep]:
    """Call Google Gemini native generateContent API to get 3 micro-substeps for a stuck step."""
    user_content = (
        f"The user is stuck on step {step_id}: \"{step_text}\"\n\n"
        "Break this into exactly 3 smaller sub-steps. "
        "Each sub-step must be simpler, shorter, and lower cognitive load than the parent. "
        "Output only raw JSON with format: {\"substeps\": [{\"id\": 1, \"text\": \"...\", \"duration\": \"< 1 min\"}, {\"id\": 2, \"text\": \"...\", \"duration\": \"< 1 min\"}, {\"id\": 3, \"text\": \"...\", \"duration\": \"2 min\"}]}"
    )
    contents = [{"role": "user", "parts": [{"text": user_content}]}]

    model_name = os.environ.get("GEMINI_MODEL", "models/gemini-flash-lite-latest")
    if not model_name.startswith("models/"):
        model_name = f"models/{model_name}"

    url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={_gemini_key}"
    payload = {
        "system_instruction": {"parts": [{"text": "You are Escape's AI coach. Break down stuck tasks into 3 microscopic sub-steps. Return ONLY valid JSON with key 'substeps'."}]},
        "contents": contents,
        "generationConfig": {
            "response_mime_type": "application/json"
        }
    }

    async with httpx.AsyncClient(timeout=18.0) as http_client:
        resp = await http_client.post(url, json=payload)
        if resp.status_code == 200:
            data = resp.json()
            raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
            parsed = _parse_json_from_text(raw_text)
            substeps_raw = parsed.get("substeps", [])
            if len(substeps_raw) == 3:
                return [
                    SubStep(id=i + 1, text=s["text"], duration=s.get("duration", "< 1 min"))
                    for i, s in enumerate(substeps_raw)
                ]
    return []


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(body: ChatRequest):
    reply = None
    # 1. Primary: If GEMINI_API_KEY is configured, call native Gemini REST API
    if _gemini_key:
        try:
            reply = await _call_gemini_native(body.messages)
        except Exception as e:
            print("Escape: Gemini native call error or timeout, falling back:", e)

    # 2. Secondary: If OpenAI / Azure / Groq client is configured
    if (not reply or not reply.strip()) and client:
        try:
            openai_msgs = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}] + [
                {"role": m.role, "content": m.content} for m in body.messages
            ]
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=DEPLOYMENT,
                    messages=openai_msgs,
                    max_tokens=2048,
                    timeout=15.0,
                ),
                timeout=15.0,
            )
            content = response.choices[0].message.content or ""
            if content.strip():
                reply = content
        except Exception as e:
            print("Escape: OpenAI/Azure call error, falling back:", e)

    if reply and reply.strip():
        return ChatResponse(reply=reply)

    # 3. Fallback: Return domain-contextualized questions with options
    return ChatResponse(reply=_generate_fallback_chat(body.messages))


# ── /intake — streaming SSE ────────────────────────────────────────────────
async def stream_intake(request: IntakeRequest, session_id: str) -> AsyncIterator[str]:
    messages = build_intake_messages(request)
    accumulated = ""

    if client:
        try:
            stream_coro = client.chat.completions.stream(
                model=DEPLOYMENT,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
                max_tokens=2048,
                timeout=6.0,
            )
            async with stream_coro as stream:
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
            yield f"data: {IntakeStreamChunk(type='done').model_dump_json()}\n\n"
            return
        except Exception:
            pass

    # Fallback streaming
    fallback_steps = [
        {"id": 1, "text": f"Open workspace for {request.goal}", "duration": "< 1 min"},
        {"id": 2, "text": "Close unrelated tabs and distractions", "duration": "< 1 min"},
        {"id": 3, "text": "Set up the first draft outline", "duration": "2 min"},
        {"id": 4, "text": "Work on the initial section for 5 minutes", "duration": "5 min"},
        {"id": 5, "text": "Review draft and polish to completion", "duration": "5 min"},
    ]
    blueprint = {
        "phase": "blueprint",
        "energy_level": request.energy_level or 6,
        "low_power_mode": False,
        "five_second_start": f"Open workspace for {request.goal[:20]}",
        "micro_steps": fallback_steps,
    }
    session_store.save(
        session_id=session_id,
        goal=request.goal,
        answers=request.answers,
        blueprint=blueprint,
    )
    yield f"data: {IntakeStreamChunk(type='blueprint', payload=blueprint).model_dump_json()}\n\n"
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
    substeps = []

    # 1. Primary: Native Gemini if GEMINI_API_KEY is configured
    if _gemini_key:
        try:
            substeps = await _call_gemini_stuck(request.step_id, request.step_text)
        except Exception as e:
            print("Escape: Gemini stuck call error, falling back:", e)

    # 2. Secondary: Client (OpenAI, Azure, Groq)
    if not substeps and client:
        try:
            messages = build_stuck_messages(request)
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=DEPLOYMENT,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
                    max_tokens=512,
                    timeout=15.0,
                ),
                timeout=15.0,
            )
            raw = response.choices[0].message.content or ""
            data = _parse_json_from_text(raw)
            substeps_raw = data.get("substeps", [])
            if len(substeps_raw) == 3:
                substeps = [
                    SubStep(id=i + 1, text=s["text"], duration=s.get("duration", "< 1 min"))
                    for i, s in enumerate(substeps_raw)
                ]
        except Exception as e:
            print("Escape: Client stuck call error, falling back:", e)

    # 3. Fallback
    if not substeps:
        substeps = [
            SubStep(id=1, text=f"Focus on just the first tiny mechanical part of: {request.step_text[:35]}", duration="< 1 min"),
            SubStep(id=2, text="Spend 60 seconds on a rough, imperfect start without judging", duration="< 1 min"),
            SubStep(id=3, text="Take 2 minutes to bring this specific piece to a close", duration="2 min"),
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


# ── Users ──────────────────────────────────────────────────────────────────

@app.post("/users", response_model=UserResponse, status_code=201)
async def create_user():
    try:
        row = await db.create_user()
        return UserResponse(**row)
    except DatabaseError as e:
        raise HTTPException(status_code=502, detail=str(e))


# ── Sessions ───────────────────────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    user_id: str
    goal: str
    steps_json: list[dict]
    energy_level: int = 5
    low_power_mode: bool = False
    five_second_start: str = ""

    model_config = {"arbitrary_types_allowed": True}


@app.post("/sessions", response_model=DbSessionRow, status_code=201)
async def create_session_endpoint(body: CreateSessionRequest):
    try:
        row = await db.create_session(
            user_id=body.user_id,
            goal=body.goal,
            steps_json=body.steps_json,
            energy_level=body.energy_level,
            low_power_mode=body.low_power_mode,
            five_second_start=body.five_second_start,
        )
        return DbSessionRow(**row)
    except DatabaseError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/users/{user_id}/sessions", response_model=UserSessionsResponse)
async def list_user_sessions(user_id: str, completed_only: bool = False, limit: int = 20):
    try:
        rows = await db.get_user_sessions(user_id, limit=limit, completed_only=completed_only)
        return UserSessionsResponse(
            user_id=user_id,
            sessions=[DbSessionSummary(**r) for r in rows],
            total=len(rows),
        )
    except DatabaseError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/sessions/{session_id}", response_model=DbSessionRow)
async def get_session_db(session_id: str):
    try:
        row = await db.get_session(session_id)
    except DatabaseError as e:
        raise HTTPException(status_code=502, detail=str(e))
    if row is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return DbSessionRow(**row)


class UpdateSessionStepsRequest(BaseModel):
    steps_json: list[dict]


@app.patch("/sessions/{session_id}/steps", response_model=DbSessionRow)
async def update_session_steps_endpoint(session_id: str, body: UpdateSessionStepsRequest):
    try:
        row = await db.update_session_steps(session_id, body.steps_json)
        return DbSessionRow(**row)
    except DatabaseError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.delete("/sessions/{session_id}")
async def delete_session_endpoint(session_id: str):
    try:
        deleted = await db.delete_session(session_id)
        return {"ok": True, "deleted": deleted}
    except DatabaseError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/sessions/{session_id}/complete", response_model=CompleteSessionResponse)
async def complete_session(session_id: str, body: CompleteSessionRequest):
    try:
        session_row = await db.complete_session(session_id, stuck_count=body.stuck_count)
        streak_data = await db.recalculate_streak(body.user_id)
    except DatabaseError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return CompleteSessionResponse(
        session=DbSessionRow(**session_row),
        streak=StreakResponse(
            user_id=body.user_id,
            current_streak=streak_data["current_streak"],
            longest_streak=streak_data["longest_streak"],
            last_completed=streak_data.get("last_completed"),
        ),
    )


# ── Streaks ────────────────────────────────────────────────────────────────

@app.get("/users/{user_id}/streak", response_model=StreakResponse)
async def get_streak(user_id: str):
    try:
        row = await db.get_streak(user_id)
    except DatabaseError as e:
        raise HTTPException(status_code=502, detail=str(e))
    if row is None:
        # User exists but has no streak yet — return zeroed response
        return StreakResponse(user_id=user_id, current_streak=0, longest_streak=0)
    return StreakResponse(
        user_id=user_id,
        current_streak=row["current_streak"],
        longest_streak=row["longest_streak"],
        last_completed=row.get("last_completed"),
        updated_at=row.get("updated_at"),
    )


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
    # Ensure both steps and micro_steps keys exist for interoperability
    if "steps" in data and "micro_steps" not in data:
        data["micro_steps"] = data["steps"]
    elif "micro_steps" in data and "steps" not in data:
        data["steps"] = data["micro_steps"]

    required = {"five_second_start", "energy_level", "low_power_mode"}
    missing = required - set(data.keys())
    if missing or ("steps" not in data and "micro_steps" not in data):
        raise ValueError(f"Blueprint missing fields: {missing}")
    steps_list = data.get("steps", data.get("micro_steps", []))
    steps_count = len(steps_list)
    if not (10 <= steps_count <= 15):
        data["_warning"] = f"Expected 10-15 steps, got {steps_count}"
    return data