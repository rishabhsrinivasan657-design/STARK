"""
summary_module.py

Summary + onboarding core logic for STARK.

Exports:
- StudentProfile
- StudentRepository
- SummaryService
- SummaryResult
- ONBOARDING_QUESTIONS
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


# ─────────────────────────────────────────
# Core data model
# ─────────────────────────────────────────


@dataclass
class StudentProfile:
    """
    Canonical view of a student used by the summary / advisor logic.

    app.py currently constructs this with roughly:
      uid, name, program, level, term,
      gpa_current, gpa_cumulative,
      credits_earned, credits_required,
      canvas_logins_7d, late_assignments_30d,
      holds_on_account, financial_risk_flag,
      international_student, employment_hours_per_week,
      reported_challenges, career_interest, involvement_notes,
      desired_role, desired_industry, work_experience,
      goals, challenges, involvement
    """

    uid: str
    name: str
    program: str               # e.g. "MS in Information Systems"
    level: str                 # e.g. "Graduate"
    term: str                  # e.g. "Fall 2025"

    # Academic metrics
    gpa_current: Optional[float] = None
    gpa_cumulative: Optional[float] = None
    credits_earned: Optional[int] = None
    credits_required: Optional[int] = None

    # Engagement & coursework behaviour
    canvas_logins_7d: Optional[int] = None
    late_assignments_30d: Optional[int] = 0  # may come back as NULL/None from DB

    # Administrative / financial status
    holds_on_account: bool = False
    financial_risk_flag: bool = False

    # Context flags
    international_student: bool = False
    employment_hours_per_week: Optional[int] = None

    # Free-text fields from SIS / notes
    reported_challenges: Optional[str] = None
    career_interest: Optional[str] = None
    involvement_notes: Optional[str] = None

    # Onboarding questionnaire (optional)
    desired_role: Optional[str] = None
    desired_industry: Optional[str] = None
    work_experience: Optional[str] = None
    goals: Optional[str] = None
    challenges: Optional[str] = None
    involvement: Optional[str] = None


# ─────────────────────────────────────────
# Onboarding questions
# ─────────────────────────────────────────

# 11 questions – UID is its own question (index 1).
ONBOARDING_QUESTIONS: List[str] = [
    "What name do you prefer the assistant to use for you?",  # 0
    "What is your university ID / UID (e.g., S101)?",         # 1
    "What is your program (e.g., MS in Information Systems)?",  # 2
    "What is your current level (e.g., Graduate, Undergraduate)?",  # 3
    "What term are you currently in (e.g., Fall 2025)?",      # 4
    "What roles are you most interested in after graduation?",  # 5
    "Which industries are you most interested in (e.g., consulting, finance, tech)?",  # 6
    "Briefly describe your prior work experience (if any).",  # 7
    "What are your top 1–2 academic or career goals for this year?",  # 8
    "What challenges are you currently facing (academic, personal, or logistical)?",  # 9
    "How are you currently involved on campus (clubs, jobs, leadership, etc.)?",  # 10
]


# ─────────────────────────────────────────
# Simple in-memory repository
# ─────────────────────────────────────────


class StudentRepository:
    """
    Very lightweight in-memory store mainly for prototyping onboarding.

    In the current FastAPI app, most student data comes from SQLite,
    but we keep this here in case we want to stash onboarding answers
    before a dedicated DB table exists (or for non-DB demos).
    """

    def __init__(self) -> None:
        self._profiles: Dict[str, StudentProfile] = {}

    def get(self, uid: str) -> Optional[StudentProfile]:
        return self._profiles.get(uid)

    def upsert_from_onboarding(self, uid: str, answers: List[str]) -> StudentProfile:
        """
        Build or update a StudentProfile using the ordered answers to
        ONBOARDING_QUESTIONS.

        We treat the UID provided to this function (`uid` parameter) as
        the authoritative ID; the answer to the "UID" question (index 1)
        is not used for matching right now.
        """
        padded = (answers + [""] * len(ONBOARDING_QUESTIONS))[: len(ONBOARDING_QUESTIONS)]

        # Index map (now that UID is Q2):
        # 0: preferred name
        # 1: UID (ignored; we use the uid parameter)
        # 2: program
        # 3: level
        # 4: term
        # 5: desired_role
        # 6: desired_industry
        # 7: work_experience
        # 8: goals
        # 9: challenges
        # 10: involvement

        name = padded[0].strip() or "Student"
        program = padded[2].strip() or "MS in Information Systems"
        level = padded[3].strip() or "Graduate"
        term = padded[4].strip() or "Fall 2025"

        desired_role = padded[5].strip() or None
        desired_industry = padded[6].strip() or None
        work_experience = padded[7].strip() or None
        goals = padded[8].strip() or None
        challenges = padded[9].strip() or None
        involvement = padded[10].strip() or None

        existing = self._profiles.get(uid)

        profile = StudentProfile(
            uid=uid,
            name=name,
            program=program,
            level=level,
            term=term,
            gpa_current=existing.gpa_current if existing else None,
            gpa_cumulative=existing.gpa_cumulative if existing else None,
            credits_earned=existing.credits_earned if existing else None,
            credits_required=existing.credits_required if existing else None,
            canvas_logins_7d=existing.canvas_logins_7d if existing else None,
            late_assignments_30d=existing.late_assignments_30d if existing else 0,
            holds_on_account=existing.holds_on_account if existing else False,
            financial_risk_flag=existing.financial_risk_flag if existing else False,
            international_student=existing.international_student if existing else False,
            employment_hours_per_week=existing.employment_hours_per_week if existing else None,
            reported_challenges=existing.reported_challenges if existing else None,
            career_interest=existing.career_interest if existing else None,
            involvement_notes=existing.involvement_notes if existing else None,
            desired_role=desired_role,
            desired_industry=desired_industry,
            work_experience=work_experience,
            goals=goals,
            challenges=challenges,
            involvement=involvement,
        )

        self._profiles[uid] = profile
        return profile


# ─────────────────────────────────────────
# Summary service
# ─────────────────────────────────────────


@dataclass
class SummaryResult:
    """
    Output from SummaryService.generate_summary.
    """
    risk_flags: List[str]


class SummaryService:
    """
    Encapsulates rule-based logic for interpreting a StudentProfile.
    """

    def generate_summary(self, profile: StudentProfile) -> SummaryResult:
        risk_flags = self._derive_risk_flags(profile)
        return SummaryResult(risk_flags=risk_flags)

    # Internal helpers
    # ----------------

    def _derive_risk_flags(self, p: StudentProfile) -> List[str]:
        flags: List[str] = []

        # GPA / academic performance
        if p.gpa_current is not None and p.gpa_current < 3.0:
            flags.append("GPA below 3.0 (current term)")

        if p.gpa_cumulative is not None and p.gpa_cumulative < 3.0:
            flags.append("Cumulative GPA below 3.0")

        # Progress toward degree
        if (
            p.credits_earned is not None
            and p.credits_required is not None
            and p.credits_required > 0
        ):
            ratio = p.credits_earned / float(p.credits_required)
            if ratio < 0.25:
                flags.append("Early in program – may need extra onboarding support")
            elif ratio > 0.90:
                flags.append("Close to graduation – prioritize career planning")

        # Engagement & coursework
        if p.canvas_logins_7d is not None and p.canvas_logins_7d < 3:
            flags.append("Low LMS activity in last 7 days")

        # Guard against None before comparing
        if p.late_assignments_30d is not None and p.late_assignments_30d > 2:
            flags.append("Multiple late assignments in last 30 days")

        # Administrative / financial
        if p.holds_on_account:
            flags.append("Active hold on student account")

        if p.financial_risk_flag:
            flags.append("Financial risk indicator present")

        # Work / time constraints
        if p.employment_hours_per_week is not None:
            if p.employment_hours_per_week >= 20:
                flags.append("Working 20+ hours per week while studying")

        # Contextual factors
        if p.international_student:
            flags.append("International student – may need immigration / cultural support")

        # Self-reported challenges
        if p.reported_challenges:
            flags.append("Student reported specific challenges")

        # Career focus
        if not p.career_interest and not p.desired_role:
            flags.append("Career interests not yet specified")

        return flags

    def generate_advisor_actions(self, profile: StudentProfile) -> str:
        """
        Convert risk flags + onboarding info into a short checklist
        an advisor could use during a meeting.
        """
        p = profile
        actions: List[str] = []
        flags = self._derive_risk_flags(p)

        if any("GPA" in f for f in flags):
            actions.append(
                "- Ask about study strategies and time management; discuss tutoring or academic resources."
            )

        if any("late assignments" in f.lower() for f in flags):
            actions.append(
                "- Review upcoming assignments and deadlines; explore causes of late submissions."
            )

        if any("hold on student account" in f.lower() for f in flags):
            actions.append(
                "- Clarify the nature of the account hold and provide steps to resolve it (e.g., billing, paperwork)."
            )

        if any("financial risk" in f.lower() for f in flags):
            actions.append(
                "- Connect the student with financial aid, scholarships, or emergency funding resources."
            )

        if any("working 20+" in f for f in flags):
            actions.append(
                "- Discuss workload and burnout; help the student plan a sustainable course load with work hours."
            )

        if any("international student" in f.lower() for f in flags):
            actions.append(
                "- Check in on immigration, housing, and cultural adjustment; refer to international student services."
            )

        if p.desired_role or p.desired_industry or p.career_interest:
            actions.append(
                "- Explore concrete next steps toward the student's target roles/industries (projects, internships, networking)."
            )
        else:
            actions.append(
                "- Help the student clarify short-term career goals and possible roles of interest."
            )

        if p.challenges:
            actions.append(
                "- Ask open-ended questions about reported challenges and co-create an action plan."
            )

        if p.involvement:
            actions.append(
                "- Reinforce beneficial involvement and suggest additional relevant opportunities."
            )
        else:
            actions.append(
                "- Recommend at least one club, event, or workshop aligned with the student's interests."
            )

        if not actions:
            actions.append(
                "- Schedule a brief check-in to set goals and identify any hidden challenges."
            )

        return "\n".join(actions)
