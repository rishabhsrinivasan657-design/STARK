# stark_context.py
from dataclasses import dataclass, field
from typing import Set, List, Dict

@dataclass
class StudentContext:
    uid: str
    name: str
    program: str
    level: str
    term: str

    # Filled/updated by different modules
    skills: Set[str] = field(default_factory=set)
    missing_skills: Set[str] = field(default_factory=set)
    resume_notes: List[str] = field(default_factory=list)
    faq_history: List[str] = field(default_factory=list)
    summary_history: List[str] = field(default_factory=list)
    extra: Dict[str, str] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"StudentContext(uid={self.uid}, name={self.name}, program={self.program}, "
            f"level={self.level}, term={self.term}, "
            f"skills={sorted(self.skills)}, missing_skills={sorted(self.missing_skills)}, "
            f"faq_qs={len(self.faq_history)}, resume_notes={len(self.resume_notes)}, "
            f"summaries={len(self.summary_history)})"
        )
