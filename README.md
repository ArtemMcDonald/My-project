# pathfinder.py
# Project "Pathfinder" — single-file MVP
# FastAPI + SQLite with constraints:
# 1) learner + memory sections
# 2) Canvas pairing stubs
# 3) random problems (no manual picking)
# 4) 10-min break after 5 misses
# 5) 100 questions/day cap

from __future__ import annotations
from fastapi import FastAPI, HTTPException, Depends, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta, timezone, date
import random
import json
import sqlite3
import os

DB_PATH = os.environ.get("PATHFINDER_DB", "pathfinder.db")

# ---------------------------
# DB utilities (sqlite3)
# ---------------------------
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = db()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS students (
        id TEXT PRIMARY KEY,
        name TEXT,
        canvas_user_id TEXT,
        pace_level TEXT DEFAULT 'med',
        interests TEXT DEFAULT '[]',
        prior_knowledge TEXT DEFAULT '[]',
        created_at TEXT
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS question_bank (
        id TEXT PRIMARY KEY,
        prompt TEXT,
        solution TEXT,
        tags TEXT,              -- JSON array
        difficulty INTEGER,     -- 1..5
        source TEXT,
        is_active INTEGER DEFAULT 1
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT,
        question_id TEXT,
        timestamp_start TEXT,
        timestamp_end TEXT,
        result TEXT,          -- correct/incorrect/partial/skip
        score REAL,           -- 0..1
        time_spent REAL       -- seconds
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS policy_state (
        student_id TEXT PRIMARY KEY,
        date_key TEXT,
        questions_answered_today INTEGER,
        consecutive_incorrect INTEGER,
        cooldown_until TEXT
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS memory_queue (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT,
        question_id TEXT,
        box INTEGER DEFAULT 1,           -- Leitner box
        due_date TEXT                    -- ISO date
    )""")

    conn.commit()
    conn.close()

init_db()

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(title="Pathfinder MVP", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# ---------------------------
# Schemas
# ---------------------------
class RegisterStudent(BaseModel):
    id: str = Field(..., description="Your internal student id")
    name: str
    canvas_user_id: Optional[str] = None
    pace_level: str = Field("med", description="slow | med | fast")
    interests: List[str] = []
    prior_knowledge: List[str] = []

class NextQuestionReq(BaseModel):
    student_id: str
    # Optional explicit guidance (usually chosen by recommender)
    target_tags: List[str] = []
    target_difficulty: Optional[int] = None

class SubmitReq(BaseModel):
    student_id: str
    question_id: str
    correct: bool
    time_spent: Optional[float] = 0.0

# ---------------------------
# Helpers: policy + memory
# ---------------------------
def today_key() -> str:
    return datetime.now(timezone.utc).date().isoformat()

def get_policy(student_id: str) -> Dict[str, Any]:
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM policy_state WHERE student_id=?", (student_id,))
    row = cur.fetchone()

    if not row:
        cur.execute(
            "INSERT INTO policy_state(student_id, date_key, questions_answered_today, consecutive_incorrect, cooldown_until) VALUES (?, ?, ?, ?, ?)",
            (student_id, today_key(), 0, 0, None)
        )
        conn.commit()
        cur.execute("SELECT * FROM policy_state WHERE student_id=?", (student_id,))
        row = cur.fetchone()

    # Reset if day changed
    if row["date_key"] != today_key():
        cur.execute(
            "UPDATE policy_state SET date_key=?, questions_answered_today=?, consecutive_incorrect=?, cooldown_until=? WHERE student_id=?",
            (today_key(), 0, 0, None, student_id)
        )
        conn.commit()
        cur.execute("SELECT * FROM policy_state WHERE student_id=?", (student_id,))
        row = cur.fetchone()

    conn.close()
    return dict(row)

def update_policy(student_id: str, **patch):
    conn = db()
    sets = ", ".join([f"{k}=?" for k in patch.keys()])
    values = list(patch.values()) + [student_id]
    conn.execute(f"UPDATE policy_state SET {sets} WHERE student_id=?", values)
    conn.commit()
    conn.close()

def enforce_caps(student_id: str):
    p = get_policy(student_id)
    # cooldown
    if p["cooldown_until"]:
        cu = datetime.fromisoformat(p["cooldown_until"])
        if datetime.now(timezone.utc) < cu:
            remaining = int((cu - datetime.now(timezone.utc)).total_seconds() // 60) + 1
            raise HTTPException(429, detail=f"Cooldown active. Try again in ~{remaining} minutes.")
    # daily cap
    if p["questions_answered_today"] >= 100:
        raise HTTPException(429, detail="Daily question cap (100) reached. Come back tomorrow!")

def add_to_memory(student_id: str, question_id: str, correct: bool):
    """Leitner-style: correct → promote box; incorrect → box=1"""
    conn = db()
    cur = conn.cursor()
    cur.execute("""SELECT id, box FROM memory_queue WHERE student_id=? AND question_id=?""",
                (student_id, question_id))
    row = cur.fetchone()

    next_box = 1
    if row:
        box = row["box"]
        next_box = min(5, box + 1) if correct else 1
        cur.execute("DELETE FROM memory_queue WHERE id=?", (row["id"],))
    else:
        next_box = 2 if correct else 1

    # schedule due date based on box
    days = {1: 1, 2: 3, 3: 7, 4: 14, 5: 30}[next_box]
    due = (date.today() + timedelta(days=days)).isoformat()
    cur.execute(
        "INSERT INTO memory_queue(student_id, question_id, box, due_date) VALUES (?, ?, ?, ?)",
        (student_id, question_id, next_box, due)
    )
    conn.commit()
    conn.close()

def pull_memory_due(student_id: str, limit: int = 3) -> List[str]:
    """Return question_ids that are due today or earlier."""
    conn = db()
    cur = conn.cursor()
    cur.execute("""SELECT question_id FROM memory_queue 
                   WHERE student_id=? AND due_date <= ? 
                   ORDER BY box DESC LIMIT ?""",
                (student_id, date.today().isoformat(), limit))
    rows = [r["question_id"] for r in cur.fetchall()]
    conn.close()
    return rows

# ---------------------------
# Recommender stubs
# ---------------------------
def load_student(student_id: str) -> Dict[str, Any] | None:
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM students WHERE id=?", (student_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    d["interests"] = json.loads(d["interests"] or "[]")
    d["prior_knowledge"] = json.loads(d["prior_knowledge"] or "[]")
    return d

def last_k_accuracy(student_id: str, k: int = 20) -> float:
    conn = db()
    cur = conn.cursor()
    cur.execute("""SELECT result FROM interactions 
                   WHERE student_id=? ORDER BY id DESC LIMIT ?""",
                (student_id, k))
    res = cur.fetchall()
    conn.close()
    if not res:
        return 0.7
    correct = sum(1 for r in res if r["result"] == "correct")
    return correct / len(res)

def pick_target(student_id: str) -> tuple[List[str], int]:
    """Choose tags + base difficulty from profile & recent accuracy."""
    s = load_student(student_id)
    if not s:
        return ([], 3)
    acc = last_k_accuracy(student_id)
    base = {"slow": 2, "med": 3, "fast": 4}.get(s["pace_level"], 3)
    if acc > 0.85:
        base = min(5, base + 1)
    elif acc < 0.55:
        base = max(1, base - 1)

    # tag scoring: favor interests, then prior knowledge
    tags = list(dict.fromkeys(s["interests"] + s["prior_knowledge"]))[:3]
    if not tags:
        tags = []  # means "any"
    return tags, base

# ---------------------------
# Randomized question sampling
# ---------------------------
def get_questions_by_ids(ids: List[str]) -> List[Dict[str, Any]]:
    if not ids:
        return []
    conn = db()
    cur = conn.cursor()
    qmarks = ",".join("?" * len(ids))
    cur.execute(f"SELECT * FROM question_bank WHERE id IN ({qmarks}) AND is_active=1", ids)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    for r in rows:
        r["tags"] = json.loads(r["tags"] or "[]")
    return rows

def sample_question(target_tags: List[str] | None, target_difficulty: int | None) -> Dict[str, Any]:
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM question_bank WHERE is_active=1")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    if target_tags:
        # tag filter (any overlap)
        cur.execute("SELECT * FROM question_bank WHERE is_active=1")
        rows = [dict(r) for r in cur.fetchall()]
        rows = [r for r in rows if set(json.loads(r["tags"])).intersection(set(target_tags))]
    else:
        cur.execute("SELECT * FROM question_bank WHERE is_active=1")
        rows = [dict(r) for r in cur.fetchall()]

    if not rows:
        conn.close()
        raise HTTPException(404, "No active questions available that match filters.")

    for r in rows:
        r["tags"] = json.loads(r["tags"])

    # Difficulty stratification
    if target_difficulty:
        below = [r for r in rows if r["difficulty"] == max(1, target_difficulty - 1)]
        target = [r for r in rows if r["difficulty"] == target_difficulty]
        above = [r for r in rows if r["difficulty"] == min(5, target_difficulty + 1)]
        pools = [target, below, above]
        weights = [0.4, 0.3, 0.3]
        pool = random.choices(pools, weights=weights, k=1)[0] or rows
        choice = random.choice(pool)
    else:
        choice = random.choice(rows)

    return {
        "id": choice["id"],
        "prompt": choice["prompt"],
        "tags": choice["tags"],
        "difficulty": choice["difficulty"],
        "source": choice.get("source"),
    }

# ---------------------------
# Endpoints
# ---------------------------
@app.get("/", response_class=HTMLResponse)
def root():
    return """<h2>Pathfinder MVP</h2>
<p>POST <code>/admin/seed</code> to load sample questions.</p>
<p>POST <code>/students/register</code> to register a student.</p>
<p>POST <code>/next-question</code> to get a question.</p>
<p>POST <code>/submit</code> to submit an answer.</p>
"""

@app.post("/students/register")
def register_student(req: RegisterStudent):
    conn = db()
    try:
        conn.execute("""INSERT OR REPLACE INTO students(id, name, canvas_user_id, pace_level, interests, prior_knowledge, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)""",
                     (req.id, req.name, req.canvas_user_id, req.pace_level,
                      json.dumps(req.interests), json.dumps(req.prior_knowledge),
                      datetime.now(timezone.utc).isoformat()))
        conn.commit()
    finally:
        conn.close()
    # initialize policy
    get_policy(req.id)
    return {"ok": True, "student_id": req.id}

@app.get("/status/{student_id}")
def status(student_id: str):
    p = get_policy(student_id)
    due = pull_memory_due(student_id, limit=10)
    return {"policy": p, "memory_due": due}

@app.post("/next-question")
def next_question(req: NextQuestionReq):
    # enforce caps and cooldown
    enforce_caps(req.student_id)

    # Memory injection: 20% chance (and if anything due)
    due_ids = pull_memory_due(req.student_id, limit=3)
    use_memory = bool(due_ids) and random.random() < 0.2
    if use_memory:
        qs = get_questions_by_ids(due_ids)
        if qs:
            q = random.choice(qs)
            return {"from_memory": True, "question": {
                "id": q["id"], "prompt": q["prompt"], "tags": q["tags"],
                "difficulty": q["difficulty"], "source": q.get("source")
            }}

    # If caller didn’t specify targets, pick via recommender
    t_tags = req.target_tags
    t_diff = req.target_difficulty
    if not t_tags and not t_diff:
        t_tags, t_diff = pick_target(req.student_id)

    q = sample_question(t_tags, t_diff)
    return {"from_memory": False, "question": q, "target_tags": t_tags, "target_difficulty": t_diff}

@app.post("/submit")
def submit(req: SubmitReq):
    p = get_policy(req.student_id)

    # Update daily count
    new_count = p["questions_answered_today"] + 1
    # Update streaks
    if req.correct:
        new_streak = 0
    else:
        new_streak = p["consecutive_incorrect"] + 1

    cooldown_until = p["cooldown_until"]

    forced_break = False

    if new_streak >= 5:
        cooldown_until = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
        new_streak = 0
        forced_break = True

    update_policy(
        req.student_id,
        questions_answered_today=new_count,
        consecutive_incorrect=new_streak,
        cooldown_until=cooldown_until
    )

    # record interaction
    conn = db()
    conn.execute("""INSERT INTO interactions(student_id, question_id, timestamp_start, timestamp_end, result, score, time_spent)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                 (req.student_id, req.question_id,
                  datetime.now(timezone.utc).isoformat(),
                  datetime.now(timezone.utc).isoformat(),
                  "correct" if req.correct else "incorrect",
                  1.0 if req.correct else 0.0,
                  req.time_spent or 0.0))
    conn.commit()
    conn.close()

    # Update memory queue
    add_to_memory(req.student_id, req.question_id, req.correct)

    return {"ok": True, "forced_break": forced_break, "cooldown_until": cooldown_until}

# ---------------------------
# Admin: seeding & quick ops
# ---------------------------
@app.post("/admin/seed")
def seed():
    """Seed a tiny question bank for demo."""
    rows = [
        # difficulty 1-5, tags mix across math/programming/memory
        ("Q1", "2 + 3 = ?", "5", ["math", "arithmetic"], 1, "demo"),
        ("Q2", "What is 7 - 4?", "3", ["math", "arithmetic"], 1, "demo"),
        ("Q3", "Find 6 * 7", "42", ["math", "multiplication"], 2, "demo"),
        ("Q4", "Square of 12?", "144", ["math", "squares"], 2, "demo"),
        ("Q5", "What does list[::-1] do in Python?", "Reverses list", ["python", "lists"], 3, "demo"),
        ("Q6", "len({'a','b','a'}) == ?", "2", ["python", "sets"], 3, "demo"),
        ("Q7", "Time complexity of binary search?", "O(log n)", ["cs", "algorithms"], 4, "demo"),
        ("Q8", "Derivative of sin(x)?", "cos(x)", ["calculus"], 4, "demo"),
        ("Q9", "What is a hash map?", "Key-value structure", ["cs", "data_structures"], 5, "demo"),
        ("Q10","What is RL exploration?", "Trying unknown actions", ["ml", "reinforcement"], 5, "demo"),
    ]
    conn = db()
    cur = conn.cursor()
    for qid, prompt, sol, tags, diff, src in rows:
        cur.execute("""INSERT OR REPLACE INTO question_bank(id, prompt, solution, tags, difficulty, source, is_active)
                       VALUES (?, ?, ?, ?, ?, ?, 1)""",
                    (qid, prompt, sol, json.dumps(tags), diff, src))
    conn.commit()
    conn.close()
    return {"ok": True, "seeded": len(rows)}

# ---------------------------
# Canvas stubs (LTI/REST placeholders)
# ---------------------------
@app.post("/canvas/lti/launch")
async def canvas_lti_launch(request: Request):
    """
    Placeholder for LTI 1.3 launch.
    In production: validate JWT ID token, map sub->canvas_user_id->students.id.
    """
    form = await request.form()
    return JSONResponse({"ok": True, "message": "LTI launch stub accepted", "form_keys": list(form.keys())})

@app.post("/canvas/gradepassback")
def canvas_gradepassback(student_id: str = Form(...), score: float = Form(...)):
    """
    Placeholder for grade passback.
    In production: call Canvas LineItems & Scores API.
    """
    return {"ok": True, "student_id": student_id, "score": score, "note": "Stub only"}

# ---------------------------
# Simple utilities for manual checks
# ---------------------------
@app.get("/questions")
def list_questions():
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT id, tags, difficulty FROM question_bank WHERE is_active=1")
    rows = [{"id": r["id"], "tags": json.loads(r["tags"]), "difficulty": r["difficulty"]} for r in cur.fetchall()]
    conn.close()
    return rows

@app.get("/students/{student_id}")
def get_student(student_id: str):
    s = load_student(student_id)
    if not s:
        raise HTTPException(404, "Student not found")
    return s

# ---------------------------
# Entry point for direct run
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("pathfinder:app", host="0.0.0.0", port=8000, reload=True)


