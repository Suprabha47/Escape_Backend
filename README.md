# Escape — FastAPI Backend v2

## Setup

```bash
cd escape_backend
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Fill in ANTHROPIC_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY

uvicorn main:app --reload --port 8000
```

Interactive docs: http://localhost:8000/docs

---

## Supabase Setup (one-time)

1. Create a project at [supabase.com](https://supabase.com)
2. Go to **SQL Editor** and run the contents of `supabase/schema.sql`
3. Copy **Project URL** and **service_role key** from **Settings → API** into your `.env`

---

## File Structure

```
main.py          — FastAPI routes (Claude + Supabase endpoints)
models.py        — Pydantic schemas
store.py         — In-memory session store (fast local reads)
database.py      — All Supabase I/O (async wrapper)
prompts.py       — System prompt + message builders
supabase/
  schema.sql     — Run once in Supabase SQL Editor
```

---

## Endpoints

### Claude (streaming)

#### POST /intake
Streams a blueprint from Claude as SSE. Returns `X-Session-Id` header.

```bash
curl -N -X POST http://localhost:8000/intake \
  -H "Content-Type: application/json" \
  -d '{"goal": "Finish the quarterly report", "answers": ["The summary intimidates me", "3"]}'
```

SSE events: `text_delta` → `blueprint` → `done`

#### POST /stuck
Breaks a stuck step into 3 sub-steps (synchronous).

```bash
curl -X POST http://localhost:8000/stuck \
  -H "Content-Type: application/json" \
  -d '{"step_id": 3, "step_text": "Write the opening sentence", "session_id": "..."}'
```

#### GET /session/{id}
In-memory session record (rich, fast, lost on restart).

---

### Supabase

#### POST /users
Create an anonymous user. Call once on first launch, persist the `id`.

```bash
curl -X POST http://localhost:8000/users
# → {"id": "uuid", "created_at": "..."}
```

#### GET /users/{user_id}/sessions
List past sessions, newest first.

```bash
curl "http://localhost:8000/users/{user_id}/sessions?limit=10&completed_only=true"
```

#### GET /sessions/{id}
Fetch a full session row from Supabase (includes `steps_json`).

```bash
curl http://localhost:8000/sessions/{session_id}
```

#### POST /sessions/{id}/complete
Mark a session done and recalculate the streak in one call.
Returns both the updated session row and the new streak.

```bash
curl -X POST http://localhost:8000/sessions/{session_id}/complete \
  -H "Content-Type: application/json" \
  -d '{"user_id": "...", "stuck_count": 2}'
# → {"session": {...}, "streak": {"current_streak": 4, "longest_streak": 7, ...}}
```

#### GET /users/{user_id}/streak
Read the pre-computed streak row (O(1) — no date calculation).

```bash
curl http://localhost:8000/users/{user_id}/streak
# → {"user_id": "...", "current_streak": 4, "longest_streak": 7, "last_completed": "2025-01-15"}
```

---

## Frontend Integration

### Typical flow

```
1. On first launch:
   POST /users  →  store user_id in localStorage

2. User submits goal:
   POST /intake  →  stream blueprint, get X-Session-Id

3. Simultaneously, persist session to Supabase:
   (call from backend during intake, or POST /sessions with the blueprint)

4. User completes all steps:
   POST /sessions/{id}/complete  →  get updated streak

5. Show streak on home screen:
   GET /users/{user_id}/streak
```

### Consuming SSE in React

```typescript
async function runIntake(goal: string, answers: string[]) {
  const res = await fetch("http://localhost:8000/intake", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ goal, answers }),
  });
  const sessionId = res.headers.get("X-Session-Id")!;
  const reader = res.body!.getReader();
  const dec = new TextDecoder();
  let buf = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const lines = buf.split("\n\n");
    buf = lines.pop() ?? "";
    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const chunk = JSON.parse(line.slice(6));
      if (chunk.type === "blueprint") enterActionMode(chunk.payload, sessionId);
    }
  }
}
```

### Completing a session

```typescript
async function completeSession(sessionId: string, userId: string, stuckCount: number) {
  const res = await fetch(`http://localhost:8000/sessions/${sessionId}/complete`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId, stuck_count: stuckCount }),
  });
  const { session, streak } = await res.json();
  showStreak(streak.current_streak); // 🔥 4-day streak
}
```

---

## Streak Logic

Streaks are calculated by a pure SQL function `recalculate_streak()` in Postgres:

- One "streak day" = at least one completed session that calendar day (UTC)
- Multiple completions in one day = still 1 streak day
- Streak is **live** if last completion was today or yesterday
- Streak **breaks** on a gap of ≥ 2 calendar days
- `longest_streak` never decreases — it's the all-time record
- Result is written to the `streaks` table for O(1) reads
