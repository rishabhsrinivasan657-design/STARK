# app.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from pathlib import Path
from datetime import datetime
import sqlite3
import tempfile
import os
import uuid
import subprocess
import sys

from dotenv import load_dotenv
from openai import OpenAI

# ─────────────────────────────────────────
# Load env + OpenAI client (optional)
# ─────────────────────────────────────────
load_dotenv()
openai_client = OpenAI()
print("[OpenAI] API key present:", bool(os.getenv("OPENAI_API_KEY")))
last_llm_error: Optional[str] = None

def _call_llm(prompt: str) -> Optional[str]:
    """
    Simple wrapper around OpenAI Responses API.
    Returns text or None and records error in last_llm_error.
    """
    global last_llm_error
    try:
        resp = openai_client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
        )
        # Adjust to your Responses API structure if needed
        text = resp.output[0].content[0].text
        last_llm_error = None
        return str(text).strip()
    except Exception as e:
        last_llm_error = repr(e)
        print("[OpenAI][ERROR]", repr(e))
        return None

# ─────────────────────────────────────────
# PDF → text (pdftotext→pdfplumber)
# ─────────────────────────────────────────
def convert_pdf_to_text(pdf_path: str) -> str:
    """Convert a PDF file to plain text using `pdftotext` or `pdfplumber`."""
    try:
        result = subprocess.run(
            ["pdftotext", pdf_path, "-"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return result.stdout
    except FileNotFoundError:
        pass
    except subprocess.CalledProcessError as e:
        sys.stderr.write(
            f"Warning: pdftotext failed ({e.stderr.strip()}); trying pdfplumber...\n"
        )

    try:
        import pdfplumber  # type: ignore
    except ImportError:
        raise RuntimeError(
            "Neither pdftotext nor pdfplumber is available. "
            "Install poppler-utils or pdfplumber."
        )
    text_chunks: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_chunks.append(page_text)
    return "\n".join(text_chunks)

# ─────────────────────────────────────────
# Import your project modules
# ─────────────────────────────────────────
from summary_module import (
    SummaryService,
    StudentProfile as CoreStudentProfile,
    ONBOARDING_QUESTIONS,
    StudentRepository,
)

# Domains/roles and ATS checker wrapper
from resume_analyzer_integrated import (
    DOMAINS,            # list of domains with roles/skills
    run_heuristic_analysis,
)

from student_faq_chatbot_llm import (
    KnowledgeBase,
    StudentProfile as FAQStudentProfile,
)

# ─────────────────────────────────────────
# DB setup
# ─────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "stark.db"

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

# ─────────────────────────────────────────
# FastAPI app + CORS
# ─────────────────────────────────────────
app = FastAPI(title="STARK Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory fallback repo (rarely used because we persist to DB)
student_repo = StudentRepository()

# ─────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────
class StudentSummaryOut(BaseModel):
    uid: str
    name: str
    program: str
    level: str
    term: str
    gpa_current: Optional[float]
    gpa_cumulative: Optional[float]
    credits_earned: Optional[int]
    credits_required: Optional[int]
    risk_flags: List[str] = []
    notes: List[str] = []
    advisor_actions: Optional[str] = None
    ai_overview: Optional[str] = None
    ai_meeting_outline: Optional[str] = None
    ai_error: Optional[str] = None

class OnboardingAnswersIn(BaseModel):
    uid: Optional[str] = Field(None, alias="uid")
    student_id: Optional[str] = Field(None, alias="studentId")
    answers: List[str]
    class Config:
        allow_population_by_field_name = True
        allow_population_by_alias = True

class ATSJobOut(BaseModel):
    job_id: str
    job_title: str
    company: str
    location: Optional[str]
    seniority: Optional[str]
    must_have_skills: Optional[str]
    nice_to_have_skills: Optional[str]
    job_link: Optional[str]

class ATSEventOut(BaseModel):
    event_id: str
    event_name: str
    category: Optional[str]
    description: Optional[str]
    skills_covered: Optional[str]
    date: Optional[str]
    location: Optional[str]
    delivery_mode: Optional[str]
    link: Optional[str]
    # NEW: backend-served ICS link for this DB event
    calendar_ics_url: Optional[str] = None

class AIResumeInsights(BaseModel):
    overview: Optional[str] = None
    strengths: Optional[str] = None
    weaknesses: Optional[str] = None
    recommended_next_steps: Optional[str] = None
    advisor_notes: Optional[str] = None
    llm_error: Optional[str] = None

class ATSAnalysisResult(BaseModel):
    job: ATSJobOut
    domain: str
    role_index: int
    resume_text: str
    heuristic_feedback: dict
    suggested_events: List[ATSEventOut]
    ai_insights: Optional[AIResumeInsights] = None

class FAQAskRequest(BaseModel):
    student_id: Optional[str] = None
    question: str
    use_llm: bool = True

class FAQAnswerOut(BaseModel):
    answer: str
    used_llm: bool
    sources: List[str] = []

# ─────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────
def fetch_student_from_db(db: sqlite3.Connection, uid: str) -> Optional[sqlite3.Row]:
    return db.execute("SELECT * FROM students_summary WHERE uid = ?", (uid,)).fetchone()

def fetch_advising_notes(db: sqlite3.Connection, uid: str) -> List[str]:
    rows = db.execute(
        "SELECT note_text FROM advising_notes WHERE uid = ? ORDER BY note_date DESC", (uid,)
    ).fetchall()
    return [r["note_text"] for r in rows]

def fetch_all_students(db: sqlite3.Connection) -> List[sqlite3.Row]:
    return db.execute("SELECT * FROM students_summary ORDER BY name").fetchall()

def fetch_jobs_for_domain(db: sqlite3.Connection, domain_name: str) -> List[sqlite3.Row]:
    like = f"%{domain_name}%"
    return db.execute(
        "SELECT * FROM ats_jobs WHERE job_title LIKE ? OR job_description LIKE ?", (like, like)
    ).fetchall()

def fetch_events_by_skills(db: sqlite3.Connection, skills: List[str]) -> List[sqlite3.Row]:
    cleaned = [s.strip().lower() for s in skills if isinstance(s, str) and s.strip()]
    if not cleaned:
        return db.execute("SELECT * FROM ats_events LIMIT 5").fetchall()
    placeholders = " OR ".join(["LOWER(skills_covered) LIKE ?"] * len(cleaned))
    params = [f"%{s}%" for s in cleaned]
    query = f"SELECT * FROM ats_events WHERE {placeholders} LIMIT 10"
    rows = db.execute(query, params).fetchall()
    if rows:
        return rows
    return db.execute("SELECT * FROM ats_events LIMIT 5").fetchall()

def yes_no_to_bool(value) -> bool:
    if value is None:
        return False
    s = str(value).strip().lower()
    return s in ("yes", "true", "1", "y", "t")

def row_to_core_profile(row: sqlite3.Row) -> CoreStudentProfile:
    keys = set(row.keys())
    return CoreStudentProfile(
        uid=row["uid"],
        name=row["name"],
        program=row["program"],
        level=row["level"],
        term=row["term"],
        gpa_current=row["gpa_current"],
        gpa_cumulative=row["gpa_cumulative"],
        credits_earned=row["credits_earned"],
        credits_required=row["credits_required"],
        canvas_logins_7d=row["canvas_logins_7d"],
        late_assignments_30d=row["late_assignments_30d"],
        holds_on_account=yes_no_to_bool(row["holds_on_account"]) if "holds_on_account" in keys else False,
        financial_risk_flag=yes_no_to_bool(row["financial_risk_flag"]) if "financial_risk_flag" in keys else False,
        international_student=yes_no_to_bool(row["international_student"]) if "international_student" in keys else False,
        employment_hours_per_week=row["employment_hours_per_week"] if "employment_hours_per_week" in keys else None,
        reported_challenges=row["reported_challenges"] if "reported_challenges" in keys else None,
        career_interest=row["career_interest"] if "career_interest" in keys else None,
        involvement_notes=row["involvement_notes"] if "involvement_notes" in keys else None,
        desired_role=row["desired_role"] if "desired_role" in keys else None,
        desired_industry=row["desired_industry"] if "desired_industry" in keys else None,
        work_experience=row["work_experience"] if "work_experience" in keys else None,
        goals=row["goals"] if "goals" in keys else None,
        challenges=row["challenges"] if "challenges" in keys else None,
        involvement=row["involvement"] if "involvement" in keys else None,
    )

def upsert_onboarding_into_db(db: sqlite3.Connection, uid: str, answers: List[str]) -> None:
    from summary_module import ONBOARDING_QUESTIONS
    padded = (answers + [""] * len(ONBOARDING_QUESTIONS))[: len(ONBOARDING_QUESTIONS)]

    preferred_name = padded[0].strip() or "Student"
    program = padded[2].strip() or "MS in Information Systems"
    level = padded[3].strip() or "Graduate"
    term = padded[4].strip() or "Fall 2025"

    desired_role = padded[5].strip() or None
    desired_industry = padded[6].strip() or None
    work_experience = padded[7].strip() or None
    goals = padded[8].strip() or None
    challenges = padded[9].strip() or None
    involvement = padded[10].strip() or None

    existing = fetch_student_from_db(db, uid)

    if existing:
        db.execute(
            """
            UPDATE students_summary
            SET name = ?,
                program = ?,
                level = ?,
                term = ?,
                desired_role = ?,
                desired_industry = ?,
                work_experience = ?,
                goals = ?,
                challenges = ?,
                involvement = ?
            WHERE uid = ?
            """,
            (
                preferred_name,
                program,
                level,
                term,
                desired_role,
                desired_industry,
                work_experience,
                goals,
                challenges,
                involvement,
                uid,
            ),
        )
    else:
        db.execute(
            """
            INSERT INTO students_summary (
                uid, name, program, level, term,
                desired_role, desired_industry, work_experience,
                goals, challenges, involvement
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                uid,
                preferred_name,
                program,
                level,
                term,
                desired_role,
                desired_industry,
                work_experience,
                goals,
                challenges,
                involvement,
            ),
        )
    db.commit()

# Dummy Testudo-like table
def _is_blank(val) -> bool:
    if val is None:
        return True
    if isinstance(val, str) and val.strip().lower() in {"", "n/a", "na", "not specified", "not specified in report"}:
        return True
    return False

def fetch_tested_record(db: sqlite3.Connection, uid: str) -> Optional[sqlite3.Row]:
    return db.execute(
        """
        SELECT
            uid,
            gpa_current,
            gpa_cumulative,
            credits_earned,
            credits_required,
            canvas_logins_7d,
            late_assignments_30d,
            holds_on_account,
            financial_risk_flag,
            international_student,
            employment_hours_per_week
        FROM tested
        WHERE uid = ?
        """,
        (uid,),
    ).fetchone()

def backfill_profile_from_tested(profile: CoreStudentProfile, tested_row: Optional[sqlite3.Row]) -> CoreStudentProfile:
    if not tested_row:
        return profile
    t = dict(tested_row)
    p = profile

    def fill(attr_name: str, tested_key: str, cast=None):
        if getattr(p, attr_name, None) in (None, ""):
            val = t.get(tested_key)
            if cast is not None and val is not None:
                try:
                    val = cast(val)
                except Exception:
                    pass
            setattr(p, attr_name, val)

    fill("gpa_current", "gpa_current", float)
    fill("gpa_cumulative", "gpa_cumulative", float)
    fill("credits_earned", "credits_earned", int)
    fill("credits_required", "credits_required", int)
    fill("canvas_logins_7d", "canvas_logins_7d", int)
    fill("late_assignments_30d", "late_assignments_30d", int)

    holds_text = t.get("holds_on_account")
    if _is_blank(getattr(p, "holds_on_account", None)) or getattr(p, "holds_on_account", None) is False:
        setattr(p, "holds_on_account", bool(holds_text and str(holds_text).strip().lower() != "none"))

    if _is_blank(getattr(p, "financial_risk_flag", None)) or getattr(p, "financial_risk_flag", None) is False:
        setattr(p, "financial_risk_flag", bool(t.get("financial_risk_flag")))

    if _is_blank(getattr(p, "international_student", None)) or getattr(p, "international_student", None) is False:
        setattr(p, "international_student", bool(t.get("international_student")))

    fill("employment_hours_per_week", "employment_hours_per_week", int)
    return p

# Optional: Knowledge base builder (not used in simple FAQ flow)
class SimpleFAQItem:
    def __init__(self, faq_id, category, question, answer, tags):
        self.id = faq_id
        self.category = category
        self.question = question
        self.answer = answer
        self.tags = tags or []
    def full_text(self) -> str:
        parts = [self.category or "", self.question or "", self.answer or "", " ".join(self.tags or [])]
        return " ".join(p for p in parts if p).strip()

def build_knowledge_base_from_db(db: sqlite3.Connection) -> KnowledgeBase:
    faq_rows = db.execute("SELECT * FROM faq").fetchall()
    items: List[SimpleFAQItem] = []
    for r in faq_rows:
        tags = (r["tags"] or "").split(";") if r["tags"] else []
        items.append(SimpleFAQItem(
            faq_id=r["faq_id"],
            category=r["category"],
            question=r["question"],
            answer=r["answer"],
            tags=tags,
        ))
    return KnowledgeBase(items=items)

def log_chat(db: sqlite3.Connection, student_id: Optional[str], question: str, answer: str):
    log_id = str(uuid.uuid4())
    ts = datetime.utcnow().isoformat()
    db.execute(
        """
        INSERT INTO chat_logs (log_id, student_id, timestamp, user_query, bot_response)
        VALUES (?, ?, ?, ?, ?)
        """,
        (log_id, student_id or "", ts, question, answer),
    )
    db.commit()

# ─────────────────────────────────────────
# AI helpers (advisor + resume insights)
# ─────────────────────────────────────────
def generate_ai_summaries(profile: CoreStudentProfile, risk_flags: List[str], advisor_actions: str):
    lines: List[str] = [
        f"Student name: {profile.name}",
        f"UID: {profile.uid}",
        f"Program: {profile.program}",
        f"Level: {profile.level}",
        f"Term: {profile.term}",
    ]
    if profile.gpa_current is not None:
        lines.append(f"Current GPA: {profile.gpa_current}")
    if profile.gpa_cumulative is not None:
        lines.append(f"Cumulative GPA: {profile.gpa_cumulative}")
    if profile.credits_earned is not None and profile.credits_required is not None:
        lines.append(f"Credits: {profile.credits_earned}/{profile.credits_required}")
    if profile.desired_role or profile.desired_industry or profile.career_interest:
        lines.append("Career interest: " + ", ".join([s for s in [profile.desired_role, profile.desired_industry, profile.career_interest] if s]))
    if profile.goals:
        lines.append(f"Goals: {profile.goals}")
    if profile.challenges:
        lines.append(f"Challenges: {profile.challenges}")
    if profile.involvement:
        lines.append(f"Involvement: {profile.involvement}")
    elif profile.involvement_notes:
        lines.append(f"Involvement notes: {profile.involvement_notes}")
    if risk_flags:
        lines.append("Risk flags: " + "; ".join(risk_flags))

    context = "\n".join(lines)

    overview_prompt = (
        "You are helping an academic advisor prepare for a short 1:1 meeting.\n\n"
        f"{context}\n\n"
        "Write a concise, advisor-facing overview (3–5 sentences) that:\n"
        "- Summarizes who this student is and where they are in the program.\n"
        "- Highlights key goals, challenges, and any risk indicators.\n"
        "- Suggests the overall tone the advisor should take (reassuring, challenging, exploratory, etc.).\n"
    )

    outline_prompt = (
        "You are helping an academic advisor structure a 25–30 minute meeting.\n\n"
        f"{context}\n\n"
        "Here is a checklist of suggested actions from a rule-based engine:\n"
        f"{advisor_actions}\n\n"
        "Using this information, write a short, numbered meeting outline (5–7 items) "
        "the advisor can follow. Focus on rapport, clarifying goals, highest-risk issues, "
        "and concrete next steps.\n"
    )

    overview = _call_llm(overview_prompt)
    outline = _call_llm(outline_prompt)
    error = last_llm_error
    return overview, outline, error

def generate_ai_resume_insights(resume_text: str, heuristic_feedback: dict, domain: str, role_title: str, missing_skills: List[str]) -> dict:
    scores = heuristic_feedback.get("scores", {}) or {}
    issues = heuristic_feedback.get("issues", []) or []
    summary = heuristic_feedback.get("summary", "")

    high_issues = [i for i in issues if str(i.get("severity", "")).lower() == "high"]
    medium_issues = [i for i in issues if str(i.get("severity", "")).lower() == "medium"]

    issue_lines: List[str] = []
    for issue in high_issues[:5] + medium_issues[:5]:
        issue_lines.append(
            f"- [{issue.get('severity', '').upper()} / {issue.get('category', '')}] "
            f"{issue.get('message', '')} Hint: {issue.get('hint', '')}"
        )

    context_lines: List[str] = [
        f"Target domain: {domain}",
        f"Target role: {role_title}",
        "",
        "ATS scores:",
        f"- General: {scores.get('general', 'N/A')}",
        f"- Formatting: {scores.get('formatting', 'N/A')}",
        f"- Content: {scores.get('content', 'N/A')}",
        "",
        f"ATS overall summary: {summary}",
    ]

    if issue_lines:
        context_lines.append("")
        context_lines.append("Key ATS issues:")
        context_lines.extend(issue_lines)

    if missing_skills:
        context_lines.append("")
        context_lines.append("Role-aligned skills that appear missing or underemphasized in the résumé:")
        context_lines.append(", ".join(missing_skills))

    context_lines.append("")
    context_lines.append("Résumé text (for reference):")
    context_lines.append(resume_text[:6000])

    context = "\n".join(context_lines)

    prompt = (
        "You are an AI career coach helping a graduate student tailor their résumé "
        "for a specific job. Provide the following sections exactly:\n\n"
        "OVERVIEW:\n"
        "STRENGTHS:\n"
        "WEAKNESSES:\n"
        "RECOMMENDED_NEXT_STEPS:\n"
        "ADVISOR_NOTES:\n\n"
        f"{context}\n"
    )

    raw = _call_llm(prompt)
    if raw is None:
        return {
            "overview": "AI résumé insights are unavailable right now (LLM error). You can still use the ATS scores and issues for guidance.",
            "strengths": None,
            "weaknesses": None,
            "recommended_next_steps": None,
            "advisor_notes": None,
            "llm_error": last_llm_error,
        }

    section_keys = ["OVERVIEW", "STRENGTHS", "WEAKNESSES", "RECOMMENDED_NEXT_STEPS", "ADVISOR_NOTES"]
    sections = {k: "" for k in section_keys}
    current: Optional[str] = None

    for line in raw.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        matched_key = None
        for key in section_keys:
            prefix = key + ":"
            if upper.startswith(prefix):
                matched_key = key
                content_after = stripped[len(prefix):].strip()
                current = key
                if content_after:
                    sections[key] += content_after + "\n"
                break
        if matched_key is None and current:
            if stripped:
                sections[current] += stripped + "\n"

    for key in section_keys:
        sections[key] = sections[key].strip() or None

    return {
        "overview": sections["OVERVIEW"],
        "strengths": sections["STRENGTHS"],
        "weaknesses": sections["WEAKNESSES"],
        "recommended_next_steps": sections["RECOMMENDED_NEXT_STEPS"],
        "advisor_notes": sections["ADVISOR_NOTES"],
        "llm_error": last_llm_error,
    }

# ─────────────────────────────────────────
# META
# ─────────────────────────────────────────
@app.get("/api/meta")
def get_meta():
    return {"app": "STARK Backend", "version": "0.1.0", "db_path": str(DB_PATH), "modules": ["summary", "ats", "faq"]}

# ─────────────────────────────────────────
# SUMMARY ENDPOINTS
# ─────────────────────────────────────────
@app.get("/api/summary/students", response_model=List[StudentSummaryOut])
def list_students(db: sqlite3.Connection = Depends(get_db)):
    rows = fetch_all_students(db)
    return [
        StudentSummaryOut(
            uid=r["uid"],
            name=r["name"],
            program=r["program"],
            level=r["level"],
            term=r["term"],
            gpa_current=r["gpa_current"],
            gpa_cumulative=r["gpa_cumulative"],
            credits_earned=r["credits_earned"],
            credits_required=r["credits_required"],
            risk_flags=[],
            notes=[],
            advisor_actions=None,
            ai_overview=None,
            ai_meeting_outline=None,
            ai_error=None,
        ) for r in rows
    ]

@app.get("/api/summary/student/{uid}", response_model=StudentSummaryOut)
def get_student_summary(uid: str, db: sqlite3.Connection = Depends(get_db)):
    row = fetch_student_from_db(db, uid)
    if not row:
        raise HTTPException(status_code=404, detail="Student not found")

    notes = fetch_advising_notes(db, uid)
    profile = row_to_core_profile(row)
    tested_row = fetch_tested_record(db, uid)
    profile = backfill_profile_from_tested(profile, tested_row)

    service = SummaryService()
    summary_result = service.generate_summary(profile)
    advisor_actions = service.generate_advisor_actions(profile)
    ai_overview, ai_meeting_outline, ai_error = generate_ai_summaries(profile, summary_result.risk_flags, advisor_actions)

    return StudentSummaryOut(
        uid=profile.uid,
        name=profile.name,
        program=profile.program,
        level=profile.level,
        term=profile.term,
        gpa_current=profile.gpa_current,
        gpa_cumulative=profile.gpa_cumulative,
        credits_earned=profile.credits_earned,
        credits_required=profile.credits_required,
        risk_flags=summary_result.risk_flags,
        notes=notes,
        advisor_actions=advisor_actions,
        ai_overview=ai_overview,
        ai_meeting_outline=ai_meeting_outline,
        ai_error=ai_error,
    )

@app.post("/api/summary/from_answers")
def summary_from_answers(payload: OnboardingAnswersIn, db: sqlite3.Connection = Depends(get_db)):
    canonical_uid = payload.uid or payload.student_id
    if not canonical_uid:
        if payload.answers and len(payload.answers) > 1:
            typed_uid = (payload.answers[1] or "").strip()
            if typed_uid:
                canonical_uid = typed_uid
    if not canonical_uid:
        canonical_uid = "anonymous"

    upsert_onboarding_into_db(db, canonical_uid, payload.answers)
    row = fetch_student_from_db(db, canonical_uid)
    if row:
        profile = row_to_core_profile(row)
    else:
        profile = student_repo.upsert_from_onboarding(canonical_uid, payload.answers)

    tested_row = fetch_tested_record(db, canonical_uid)
    profile = backfill_profile_from_tested(profile, tested_row)

    service = SummaryService()
    summary_result = service.generate_summary(profile)
    advisor_actions = service.generate_advisor_actions(profile)
    ai_overview, ai_meeting_outline, ai_error = generate_ai_summaries(profile, summary_result.risk_flags, advisor_actions)

    if summary_result.risk_flags:
        summary_bullets = (
            "Here are the main points the assistant noticed about your current situation:"
            + "<ul>"
            + "".join(f"<li>{flag}</li>" for flag in summary_result.risk_flags)
            + "</ul>"
        )
    else:
        summary_bullets = "No major risk flags detected. Focus on your goals and next steps with your advisor."

    return {
        "summary": {
            "uid": profile.uid,
            "name": profile.name,
            "risk_flags": summary_result.risk_flags,
            "summary_bullets": summary_bullets,
            "advisor_actions": advisor_actions,
            "ai_overview": ai_overview,
            "ai_meeting_outline": ai_meeting_outline,
            "ai_error": ai_error,
        }
    }

@app.get("/api/summary/onboarding_questions")
def get_onboarding_questions():
    return {"questions": ONBOARDING_QUESTIONS}

# ─────────────────────────────────────────
# ATS / RESUME ENDPOINTS
# ─────────────────────────────────────────
@app.get("/api/ats/domains")
def get_ats_domains():
    return {"domains": DOMAINS}

@app.get("/api/ats/jobs", response_model=List[ATSJobOut])
def get_ats_jobs(domain: Optional[str] = None, db: sqlite3.Connection = Depends(get_db)):
    rows = fetch_jobs_for_domain(db, domain) if domain else db.execute("SELECT * FROM ats_jobs").fetchall()
    return [
        ATSJobOut(
            job_id=r["job_id"],
            job_title=r["job_title"],
            company=r["company"],
            location=r["location"],
            seniority=r["seniority"],
            must_have_skills=r["must_have_skills"],
            nice_to_have_skills=r["nice_to_have_skills"],
            job_link=r["job_link"],
        ) for r in rows
    ]

def _event_calendar_url(event_id: Optional[str]) -> Optional[str]:
    """Return relative path to ICS download for a DB event"""
    if not event_id:
        return None
    return f"/api/events/{event_id}.ics"

@app.post("/api/ats/analyze_upload", response_model=ATSAnalysisResult)
async def analyze_resume_upload(
    domain_index: int = Form(...),
    role_index: int = Form(...),
    use_llm: bool = Form(False),
    file: UploadFile = File(...),
    db: sqlite3.Connection = Depends(get_db),
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        resume_text = convert_pdf_to_text(tmp_path)

        # Resolve domain and role
        try:
            domain = DOMAINS[domain_index]
        except IndexError:
            raise HTTPException(status_code=400, detail="Invalid domain_index")

        try:
            role = domain["roles"][role_index]
        except IndexError:
            raise HTTPException(status_code=400, detail="Invalid role_index")

        domain_name = domain.get("name", f"Domain {domain_index}") if isinstance(domain, dict) else str(domain)
        role_title = (role.get("job_title") or role.get("name") or "Selected Role") if isinstance(role, dict) else str(role)
        job_desc = ""
        if isinstance(role, dict):
            job_desc = role.get("job_description") or role.get("description") or role.get("job_title") or role.get("name") or ""

        heuristic_feedback = run_heuristic_analysis(resume_text, job_desc)

        raw_missing = (
            heuristic_feedback.get("missing_keywords")
            or heuristic_feedback.get("missing_skills")
            or []
        )
        missing_skills: List[str] = []
        for item in raw_missing:
            if isinstance(item, str):
                if item not in missing_skills:
                    missing_skills.append(item)
            elif isinstance(item, dict):
                for key in ("keyword", "skill", "name", "label"):
                    val = item.get(key)
                    if isinstance(val, str) and val.strip() and val not in missing_skills:
                        missing_skills.append(val.strip())
                        break

        role_skills = role.get("skills", []) if isinstance(role, dict) else []
        resume_lower = resume_text.lower()
        for skill in role_skills:
            if not isinstance(skill, str):
                continue
            s_clean = skill.strip()
            if not s_clean:
                continue
            if s_clean.lower() not in resume_lower and s_clean not in missing_skills:
                missing_skills.append(s_clean)

        # Use DB-only events (ats_events)
        event_rows = fetch_events_by_skills(db, missing_skills)
        suggested_events: List[ATSEventOut] = []
        for r in event_rows:
            ev_id = r["event_id"]
            suggested_events.append(
                ATSEventOut(
                    event_id=ev_id,
                    event_name=r["event_name"],
                    category=r["category"],
                    description=r["description"],
                    skills_covered=r["skills_covered"],
                    date=r["date"],
                    location=r["location"],
                    delivery_mode=r["delivery_mode"],
                    link=r["link"],
                    calendar_ics_url=_event_calendar_url(ev_id) if ev_id else None,
                )
            )

        if isinstance(role, dict):
            job_out = ATSJobOut(
                job_id=str(role.get("id", "")),
                job_title=role_title,
                company=role.get("company") or "",
                location=role.get("location"),
                seniority=role.get("seniority"),
                must_have_skills=", ".join(role.get("must_have_skills", [])) or None,
                nice_to_have_skills=", ".join(role.get("nice_to_have_skills", [])) or None,
                job_link=role.get("job_link"),
            )
        else:
            job_out = ATSJobOut(
                job_id="",
                job_title=role_title,
                company="",
                location=None,
                seniority=None,
                must_have_skills=None,
                nice_to_have_skills=None,
                job_link=None,
            )

        ai_insights_obj: Optional[AIResumeInsights] = None
        if use_llm:
            insights_dict = generate_ai_resume_insights(
                resume_text=resume_text,
                heuristic_feedback=heuristic_feedback,
                domain=domain_name,
                role_title=role_title,
                missing_skills=missing_skills,
            )
            ai_insights_obj = AIResumeInsights(**insights_dict)

        return ATSAnalysisResult(
            job=job_out,
            domain=domain_name,
            role_index=role_index,
            resume_text=resume_text,
            heuristic_feedback=heuristic_feedback,
            suggested_events=suggested_events,
            ai_insights=ai_insights_obj,
        )
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

# ─────────────────────────────────────────
# FAQ CHATBOT ENDPOINT
# ─────────────────────────────────────────
@app.post("/api/faq/ask", response_model=FAQAnswerOut)
def ask_faq(req: FAQAskRequest, db: sqlite3.Connection = Depends(get_db)):
    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        pattern = f"%{q}%"
        row = db.execute(
            """
            SELECT * FROM faq
            WHERE question LIKE ?
               OR category LIKE ?
               OR tags LIKE ?
            ORDER BY faq_id
            LIMIT 1
            """,
            (pattern, pattern, pattern),
        ).fetchone()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FAQ DB error: {e}")

    if row:
        answer = row["answer"]
        sources = [f"faq:{row['faq_id']}"]
    else:
        answer = "I'm not sure about that yet. Please contact your academic advisor or check the university website for more details."
        sources = []

    log_chat(db, req.student_id, q, answer)
    return FAQAnswerOut(answer=answer, used_llm=False, sources=sources)

# ─────────────────────────────────────────
# ICS HELPERS & ENDPOINTS
# ─────────────────────────────────────────
def _format_dt_for_ics(date_str: Optional[str]) -> Optional[str]:
    """Convert 'YYYY-MM-DD' → 'YYYYMMDD' (all-day)."""
    if not date_str:
        return None
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%Y%m%d")
    except Exception:
        return None

def _make_ics(
    *,
    uid: str,
    summary: str,
    description: str = "",
    location: str = "",
    url: str = "",
    date_ymd: Optional[str] = None
) -> str:
    dtstamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    dtstart = _format_dt_for_ics(date_ymd) or datetime.utcnow().strftime("%Y%m%d")
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//STARK//Events//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        "BEGIN:VEVENT",
        f"UID:{uid}@stark.local",
        f"DTSTAMP:{dtstamp}",
        f"DTSTART;VALUE=DATE:{dtstart}",
        f"DTEND;VALUE=DATE:{dtstart}",
        f"SUMMARY:{summary}",
    ]
    if description:
        clean_desc = description.replace("\n", " ").replace("\r", " ")
        lines.append(f"DESCRIPTION:{clean_desc}")
    if location:
        lines.append(f"LOCATION:{location}")
    if url:
        lines.append(f"URL:{url}")
    lines += ["END:VEVENT", "END:VCALENDAR"]
    return "\r\n".join(lines)

@app.post("/api/events/ics", response_class=PlainTextResponse)
def create_ics_from_payload(payload: dict):
    """
    Build an ICS from payload:
    { event_name, description, location, link, date: 'YYYY-MM-DD' }
    """
    event_name = (payload.get("event_name") or "Event").strip()
    description = (payload.get("description") or "").strip()
    location = (payload.get("location") or "").strip()
    link = (payload.get("link") or "").strip()
    date_str = (payload.get("date") or "").strip() or None

    uid = f"adhoc-{uuid.uuid4().hex}"
    ics = _make_ics(
        uid=uid,
        summary=event_name,
        description=description,
        location=location,
        url=link,
        date_ymd=date_str,
    )
    return PlainTextResponse(content=ics, media_type="text/calendar; charset=utf-8")

@app.get("/api/events/{event_id}.ics", response_class=PlainTextResponse)
def create_ics_for_event(event_id: str, db: sqlite3.Connection = Depends(get_db)):
    """Serve an ICS for a DB event (ats_events.event_id)."""
    row = db.execute("SELECT * FROM ats_events WHERE event_id = ?", (event_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Event not found")

    summary = row["event_name"] or "Event"
    description = row["description"] or ""
    location = row["location"] or ""
    link = row["link"] or ""
    date_str = row["date"] or None

    ics = _make_ics(
        uid=f"db-{event_id}",
        summary=summary,
        description=description,
        location=location,
        url=link,
        date_ymd=date_str,
    )
    return PlainTextResponse(content=ics, media_type="text/calendar; charset=utf-8")
