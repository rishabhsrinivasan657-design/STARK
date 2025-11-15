"""
ats_checker_updated.py

Heuristic ATS-style résumé checker used by resume_analyzer_integrated.py.

Checks:
- contact info
- sections
- formatting
- content + keywords (if JD provided)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import re


@dataclass
class AtsIssue:
    category: str   # "general", "formatting", "content", "keywords"
    severity: str   # "low", "medium", "high"
    message: str
    hint: str


class AtsChecker:
    def __init__(self, job_description: Optional[str] = None) -> None:
        self.job_description = job_description or ""

    def analyze(self, resume_text: str) -> Dict:
        issues: List[AtsIssue] = []

        general_score, gen_issues = self._check_general(resume_text)
        issues.extend(gen_issues)

        formatting_score, fmt_issues = self._check_formatting(resume_text)
        issues.extend(fmt_issues)

        content_score, content_issues = self._check_content(resume_text)
        issues.extend(content_issues)

        if self.job_description.strip():
            keyword_score, kw_issues = self._check_keywords(resume_text, self.job_description)
            issues.extend(kw_issues)
        else:
            keyword_score = 80  # neutral if no JD is provided

        combined_content_score = int((content_score + keyword_score) / 2)

        scores = {
            "general": max(0, min(100, general_score)),
            "formatting": max(0, min(100, formatting_score)),
            "content": max(0, min(100, combined_content_score)),
        }

        summary = self._build_summary(scores, issues)

        return {
            "scores": scores,
            "summary": summary,
            "issues": [issue.__dict__ for issue in issues],
        }

    def _check_general(self, text: str) -> Tuple[int, List[AtsIssue]]:
        issues: List[AtsIssue] = []
        lower = text.lower()
        score = 100

        has_email = bool(re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text))
        has_phone = bool(re.search(r"\+?\d[\d\s\-().]{7,}", text))

        if not has_email:
            score -= 15
            issues.append(
                AtsIssue(
                    category="general",
                    severity="high",
                    message="Email address not detected.",
                    hint="Add a professional email address near the top of the résumé.",
                )
            )
        if not has_phone:
            score -= 10
            issues.append(
                AtsIssue(
                    category="general",
                    severity="medium",
                    message="Phone number not detected.",
                    hint="Include a reachable phone number in the header.",
                )
            )

        for sec in ["education", "experience", "skills"]:
            if sec not in lower:
                score -= 10
                issues.append(
                    AtsIssue(
                        category="general",
                        severity="medium",
                        message=f"'{sec.title()}' section not clearly found.",
                        hint=f"Add a clear '{sec.title()}' heading.",
                    )
                )

        num_words = len(text.split())
        if num_words < 150:
            score -= 10
            issues.append(
                AtsIssue(
                    category="general",
                    severity="medium",
                    message="Résumé appears very short.",
                    hint="Consider adding more detail on your experiences and impact.",
                )
            )
        elif num_words > 900:
            score -= 10
            issues.append(
                AtsIssue(
                    category="general",
                    severity="low",
                    message="Résumé appears very long.",
                    hint="Try to keep it to 1–2 pages with the most relevant information.",
                )
            )

        return max(0, score), issues

    def _check_formatting(self, text: str) -> Tuple[int, List[AtsIssue]]:
        issues: List[AtsIssue] = []
        score = 100

        lines = [line for line in text.splitlines() if line.strip()]
        bullet_chars = ["-", "•", "●", "*"]
        bullet_count = sum(
            1 for line in lines if any(line.strip().startswith(ch) for ch in bullet_chars)
        )

        if bullet_count < 5:
            score -= 15
            issues.append(
                AtsIssue(
                    category="formatting",
                    severity="medium",
                    message="Few bullet points detected.",
                    hint="Use bullet points to structure responsibilities and achievements.",
                )
            )

        caps_headings = [
            line.strip()
            for line in lines
            if line.strip().isupper() and len(line.strip()) > 3
        ]
        if len(caps_headings) < 2:
            score -= 10
            issues.append(
                AtsIssue(
                    category="formatting",
                    severity="low",
                    message="Clear section headings not strongly detected.",
                    hint="Use consistent, clear headings (e.g., EDUCATION, EXPERIENCE, SKILLS).",
                )
            )

        if "│" in text or "┼" in text or "table" in text.lower():
            score -= 10
            issues.append(
                AtsIssue(
                    category="formatting",
                    severity="medium",
                    message="Table-like formatting or complex layout detected.",
                    hint="Use a simple, single-column layout without tables for ATS compatibility.",
                )
            )

        return max(0, score), issues

    def _check_content(self, text: str) -> Tuple[int, List[AtsIssue]]:
        issues: List[AtsIssue] = []
        score = 100

        has_metrics = bool(re.search(r"[%$]|[0-9]{2,}", text))
        if not has_metrics:
            score -= 20
            issues.append(
                AtsIssue(
                    category="content",
                    severity="medium",
                    message="Few or no quantified achievements detected.",
                    hint="Include metrics (%, $, or numbers) to show impact.",
                )
            )

        weak_verbs = ["helped", "worked on", "responsible for"]
        weak_hits = sum(text.lower().count(v) for v in weak_verbs)
        if weak_hits > 3:
            score -= 10
            issues.append(
                AtsIssue(
                    category="content",
                    severity="low",
                    message="Multiple weak or vague action phrases found.",
                    hint="Use strong action verbs (e.g., Led, Built, Analyzed, Designed, Implemented).",
                )
            )

        return max(0, score), issues

    def _check_keywords(self, resume_text: str, jd_text: str) -> Tuple[int, List[AtsIssue]]:
        issues: List[AtsIssue] = []

        resume_lower = resume_text.lower()
        jd_lower = jd_text.lower()

        jd_words = re.findall(r"[a-zA-Z]{4,}", jd_lower)
        stopwords = {
            "with", "from", "this", "that", "have", "will", "your", "about",
            "which", "their", "they", "them", "into", "such", "than", "then",
            "after", "before", "also", "more", "most", "some", "other",
            "when", "where", "while", "shall"
        }
        jd_keywords = {w for w in jd_words if w not in stopwords}
        freq = {w: jd_lower.count(w) for w in jd_keywords}
        jd_top_keywords = sorted(freq, key=freq.get, reverse=True)[:40]

        if not jd_top_keywords:
            return 80, issues

        matched = sum(1 for kw in jd_top_keywords if kw in resume_lower)
        coverage = matched / len(jd_top_keywords)

        if coverage >= 0.6:
            score = 90
        elif coverage >= 0.4:
            score = 80
        elif coverage >= 0.25:
            score = 70
        else:
            score = 55

        if coverage < 0.4:
            issues.append(
                AtsIssue(
                    category="keywords",
                    severity="high",
                    message="Low keyword overlap between résumé and job description.",
                    hint=(
                        "Mirror important keywords from the job description "
                        "in your résumé where they honestly reflect your experience."
                    ),
                )
            )

        return score, issues

    def _build_summary(self, scores: Dict[str, int], issues: List[AtsIssue]) -> str:
        avg_score = (scores["general"] + scores["formatting"] + scores["content"]) / 3

        if avg_score >= 85:
            level = "strong and likely to pass many ATS filters."
        elif avg_score >= 70:
            level = "decent but would benefit from improvements."
        else:
            level = "at risk of being filtered out by ATS or recruiters."

        high_issues = [i for i in issues if i.severity == "high"]
        medium_issues = [i for i in issues if i.severity == "medium"]

        return (
            f"Overall, this résumé appears {level} "
            f"Key concerns: {len(high_issues)} high-severity and {len(medium_issues)} medium-severity issues detected."
        )
