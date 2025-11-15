"""
resume_analyzer_integrated.py

Provides:
- DOMAINS: list of domain/role definitions + target skills
- run_heuristic_analysis(resume_text, jd_text=None) -> dict (ATS analysis)
"""

from typing import Optional, List
from ats_checker_updated import AtsChecker

# ---------------------------
# Domain and Role Definitions
# ---------------------------

DOMAINS = [
    {
        "name": "Data & Analytics",
        "roles": [
            {
                "name": "Data Analyst",
                "skills": [
                    "SQL", "Excel", "Python", "Tableau", "Power BI",
                    "data cleaning", "data visualization", "descriptive statistics",
                    "A/B testing", "business storytelling",
                ],
            },
            {
                "name": "Business Data Analyst",
                "skills": [
                    "SQL", "Excel", "requirements gathering", "stakeholder communication",
                    "KPIs", "dashboarding", "Power BI", "Tableau", "storytelling",
                ],
            },
            {
                "name": "Product Data Analyst",
                "skills": [
                    "SQL", "experimentation", "product metrics", "funnel analysis",
                    "A/B testing", "Python", "Tableau", "stakeholder communication",
                ],
            },
        ],
    },
    {
        "name": "Product Management",
        "roles": [
            {
                "name": "Associate Product Manager",
                "skills": [
                    "roadmapping", "requirements gathering", "JIRA", "backlog management",
                    "user research", "A/B testing", "prioritization", "stakeholder communication",
                ],
            },
            {
                "name": "Product Analyst",
                "skills": [
                    "SQL", "product metrics", "feature experimentation",
                    "dashboarding", "user behavior analysis", "storytelling with data",
                ],
            },
        ],
    },
    {
        "name": "Software Engineering",
        "roles": [
            {
                "name": "Backend Engineer",
                "skills": [
                    "Java", "Python", "APIs", "databases", "Git",
                    "unit testing", "system design", "cloud basics",
                ],
            },
            {
                "name": "Full-Stack Engineer",
                "skills": [
                    "JavaScript", "React", "HTML/CSS", "APIs",
                    "SQL", "Git", "UI implementation", "debugging",
                ],
            },
        ],
    },
    {
        "name": "Consulting / Strategy",
        "roles": [
            {
                "name": "Management Consultant (Analytics Focus)",
                "skills": [
                    "Excel", "PowerPoint", "structured problem solving",
                    "case cracking", "storytelling", "client communication",
                    "market research", "basic statistics",
                ],
            },
            {
                "name": "Business Strategy Analyst",
                "skills": [
                    "financial modeling", "market sizing", "Excel",
                    "PowerPoint", "stakeholder communication", "KPI design",
                ],
            },
        ],
    },
    {
        "name": "Finance / Quant",
        "roles": [
            {
                "name": "Financial/Data Analyst",
                "skills": [
                    "Excel", "financial modeling", "SQL",
                    "Tableau", "valuation", "forecasting", "presentation skills",
                ],
            },
            {
                "name": "Risk/Quant Analyst",
                "skills": [
                    "probability", "statistics", "R", "Python",
                    "risk modeling", "mathematics", "data visualization",
                ],
            },
        ],
    },
]

# ---------------------------
# Heuristic ATS wrapper
# ---------------------------

def run_heuristic_analysis(resume_text: str, jd_text: Optional[str] = None) -> dict:
    """
    Run the AtsChecker on the given resume text (and optional JD text)
    and return a structured analysis dict:
    {
        "scores": {...},
        "summary": "...",
        "issues": [...]
    }
    """
    checker = AtsChecker(job_description=jd_text)
    analysis = checker.analyze(resume_text)
    return analysis
