"""
student_faq_chatbot_llm.py

Prototype of a Student FAQ + Recommendation chatbot using:
- TF-IDF retrieval over dummy knowledge items
- Optional LLM integration for natural, contextual answers

Usage (CLI demo):

    python student_faq_chatbot_llm.py

Make sure you have:
- A .env file in the same folder with:  OPENAI_API_KEY=sk-...
- Dependencies installed:
    pip install scikit-learn "openai>=1.0.0" python-dotenv
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables from .env (including OPENAI_API_KEY)
load_dotenv()

# Try importing OpenAI (new SDK)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# ─────────────────────────────────────────
# 1. Data models
# ─────────────────────────────────────────

@dataclass
class StudentProfile:
    uid: str
    name: str
    program: str
    level: str
    term: str
    interests: List[str]


@dataclass
class KnowledgeItem:
    item_id: str
    source_type: str     # e.g. "email", "announcement", "website"
    title: str
    content: str
    url: Optional[str]
    tags: List[str]
    audience: List[str]
    action_hint: str

    def full_text(self) -> str:
        """
        Text used for TF-IDF similarity.
        """
        tag_text = " ".join(self.tags)
        return f"{self.title} {self.content} {tag_text}"


# ─────────────────────────────────────────
# 2. Dummy data
# ─────────────────────────────────────────

def build_dummy_knowledge_items() -> List[KnowledgeItem]:
    """
    Build a small in-memory knowledge base of emails, announcements, and web pages.
    In the real project this will be replaced by DB / scraped content.
    """
    return [
        KnowledgeItem(
            "email_1",
            "email",
            "Spring 2026 Course Registration Opens",
            "Registration for Spring 2026 opens on November 11, 2025 for graduate programs including MSIS and MBA.",
            None,
            ["registration", "courses", "spring 2026"],
            ["msis", "mba", "graduate"],
            "Check Testudo and register starting November 11."
        ),
        KnowledgeItem(
            "email_2",
            "email",
            "Graduate Career Fair – Data & Analytics Focus",
            "The Smith School is hosting a Graduate Career Fair focused on data, analytics, and consulting roles.",
            None,
            ["career", "analytics", "networking"],
            ["msis", "msba", "graduate"],
            "Register on the career portal and prepare your resume."
        ),
        KnowledgeItem(
            "ann_1",
            "announcement",
            "MSIS Python Bootcamp for New Students",
            "An intensive weekend Python bootcamp covering data processing, APIs, and basic machine learning.",
            None,
            ["python", "bootcamp", "skills", "analytics"],
            ["msis", "msba", "graduate"],
            "Register for the bootcamp to strengthen Python skills."
        ),
        KnowledgeItem(
            "web_1",
            "website",
            "When can I register for classes?",
            "Graduate students begin registration in early November for the Spring term. Dates are on the registrar’s calendar.",
            "https://dummy.university.edu/registrar/calendar",
            ["registration", "calendar"],
            ["graduate"],
            "Visit the academic calendar for your date."
        ),
    ]


# ─────────────────────────────────────────
# 3. Knowledge Base with TF-IDF retrieval
# ─────────────────────────────────────────

class KnowledgeBase:
    """
    Simple TF-IDF based retrieval over the knowledge items.
    """

    def __init__(self, items: List[KnowledgeItem]):
        self.items = items
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_matrix = self.vectorizer.fit_transform(
            [item.full_text() for item in self.items]
        )

    def search(self, query: str, top_k: int = 5) -> List[Tuple[KnowledgeItem, float]]:
        """
        Return a list of (KnowledgeItem, similarity_score) sorted by relevance.
        """
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.doc_matrix)[0]
        scored = list(zip(self.items, sims))
        scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
        return scored_sorted[:top_k]


# ─────────────────────────────────────────
# 4. Personalization + simple (non-LLM) answer
# ─────────────────────────────────────────

def personalize_results(
    student: StudentProfile,
    results: List[Tuple[KnowledgeItem, float]]
) -> List[Tuple[KnowledgeItem, float]]:
    """
    Boost items that match program/level/interests.
    """
    boosted: List[Tuple[KnowledgeItem, float]] = []

    for item, base_score in results:
        score = base_score
        tags = [t.lower() for t in item.tags]
        audience = [a.lower() for a in item.audience]

        # Boost for program / level match
        if student.program.lower() in audience:
            score += 0.25
        if student.level.lower() in audience:
            score += 0.15

        # Boost for interests
        for interest in student.interests:
            if interest.lower() in tags:
                score += 0.10

        boosted.append((item, score))

    boosted_sorted = sorted(boosted, key=lambda x: x[1], reverse=True)
    return boosted_sorted


def truncate(text: str, maxlen: int = 250) -> str:
    """
    Truncate long content.
    """
    if len(text) <= maxlen:
        return text
    return text[:maxlen - 3] + "..."


def build_simple_answer(
    query: str,
    student: StudentProfile,
    ranked_results: List[Tuple[KnowledgeItem, float]]
) -> str:
    """
    Simple rule-based answer using the top ranked knowledge item.
    """
    if not ranked_results:
        return f"Sorry {student.name}, I found nothing relevant for: '{query}'"

    top_item = ranked_results[0][0]

    lines: List[str] = []
    lines.append(f"Here’s what I found for you, {student.name}:")
    lines.append(f"**{top_item.title} ({top_item.source_type})**")
    lines.append(truncate(top_item.content))
    lines.append(f"→ {top_item.action_hint}")

    # Simple personalization message
    lines.append("\n🎯 Personalized Suggestion:")
    tags_text = " ".join(top_item.tags).lower()
    if "python" in tags_text:
        lines.append("Since you’re interested in Python, consider joining the MSIS Python bootcamp.")
    if "career" in tags_text or "analytics" in tags_text:
        lines.append("You should attend the analytics career fair to network with employers.")

    return "\n".join(lines)


def answer_query_simple(
    query: str,
    student: StudentProfile,
    kb: KnowledgeBase
) -> str:
    """
    End-to-end simple answer without LLM.
    """
    results = kb.search(query)
    ranked = personalize_results(student, results)
    answer = build_simple_answer(query, student, ranked)
    return answer


# ─────────────────────────────────────────
# 5. LLM client wrapper (OpenAI GPT)
# ─────────────────────────────────────────

class LLMClient:
    """
    Thin wrapper over an LLM API (OpenAI GPT for now).

    - If OPENAI_API_KEY is not set, or OpenAI is not installed, it falls back to dry_run
      and returns the raw prompts instead of calling the API.
    """

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        dry_run: bool = False
    ):
        self.model = model
        self.dry_run = dry_run

        api_key = os.getenv("OPENAI_API_KEY")

        if api_key and OPENAI_AVAILABLE and not dry_run:
            # OpenAI client will read the API key from the environment
            self.client = OpenAI()
        else:
            self.client = None
            # If not properly configured, force dry_run so we don't crash
            self.dry_run = True

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """
        Call the LLM or return the combined prompt in dry_run mode.
        """
        if self.dry_run or self.client is None:
            combined = (
                "========== SYSTEM PROMPT ==========\n"
                f"{system_prompt}\n\n"
                "========== USER MESSAGE ==========\n"
                f"{user_prompt}\n"
            )
            return combined

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )

        return response.choices[0].message.content.strip()


# ─────────────────────────────────────────
# 6. Building the LLM prompt from retrieved items
# ─────────────────────────────────────────

def build_llm_prompt_from_context(
    query: str,
    student: StudentProfile,
    ranked_items: List[Tuple[KnowledgeItem, float]],
    top_k: int = 3,
) -> Tuple[str, str]:
    """
    Build (system_prompt, user_prompt) to send to the LLM.
    """
    system_prompt = (
        "You are a helpful academic advisor chatbot for graduate students at a business school. "
        "You answer questions using the provided context from emails, announcements, and websites. "
        "If the context does not fully answer the question, you say so and give your best guess, "
        "but you never invent specific dates or policies that are not in the context.\n\n"
        "Always speak in a friendly, concise tone and give clear next steps."
    )

    context_lines: List[str] = []
    for item, score in ranked_items[:top_k]:
        item_block = (
            f"[{item.item_id}] {item.title} "
            f"(type={item.source_type}, tags={','.join(item.tags)})\n"
            f"{item.content}\n"
            f"Action hint: {item.action_hint}\n"
        )
        context_lines.append(item_block)

    if context_lines:
        context_text = "\n---\n".join(context_lines)
    else:
        context_text = "No matching context items."

    user_prompt = (
        f"Student profile:\n"
        f"- Name: {student.name}\n"
        f"- Program: {student.program}\n"
        f"- Level: {student.level}\n"
        f"- Term: {student.term}\n"
        f"- Interests: {', '.join(student.interests) if student.interests else 'None'}\n\n"
        f"Student question:\n"
        f"\"{query}\"\n\n"
        f"Relevant context items:\n"
        f"{context_text}\n\n"
        f"Using only the information above, answer the student's question. "
        f"Also, if appropriate, include one short bullet list of recommended next steps."
    )

    return system_prompt, user_prompt


# ─────────────────────────────────────────
# 7. High-level LLM-based answer function
# ─────────────────────────────────────────

def answer_query_llm(
    query: str,
    student: StudentProfile,
    kb: KnowledgeBase,
    llm: Optional[LLMClient] = None,
    top_k: int = 3,
) -> str:
    """
    Main entry point for GenAI-powered FAQ:
    - Retrieve knowledge
    - Personalize ranking
    - Build LLM prompt
    - Call LLM (or print prompts if dry_run)
    """
    if llm is None:
        # Default: dry_run=True so you can see prompts without paying for API calls
        llm = LLMClient(dry_run=True)

    results = kb.search(query, top_k=top_k)
    ranked = personalize_results(student, results)

    if not ranked:
        return f"Sorry {student.name}, I couldn't find anything related to: '{query}'."

    system_prompt, user_prompt = build_llm_prompt_from_context(
        query=query,
        student=student,
        ranked_items=ranked,
        top_k=top_k,
    )

    answer = llm.chat(system_prompt, user_prompt)
    return answer


# ─────────────────────────────────────────
# 8. CLI demo (for testing outside UI)
# ─────────────────────────────────────────

if __name__ == "__main__":
    # Build knowledge base once
    items = build_dummy_knowledge_items()
    kb = KnowledgeBase(items)

    # Hardcoded test student
    student = StudentProfile(
        uid="123",
        name="Rishabh",
        program="MSIS",
        level="Graduate",
        term="Fall 2025",
        interests=["python", "analytics"],
    )

    # Set dry_run=False to call real GPT (requires OPENAI_API_KEY in .env)
    # Set dry_run=True to just print the prompts instead.
    llm_client = LLMClient(dry_run=False)

    print("=== Student FAQ Chatbot (CLI) ===")
    print(f"Logged in as: {student.name} ({student.program}, {student.term})")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Bye! 👋")
            break

        if not query:
            continue

        if query.lower() in {"exit", "quit"}:
            print("Bot: Bye! 👋")
            break

        # LLM-based answer
        answer = answer_query_llm(
            query=query,
            student=student,
            kb=kb,
            llm=llm_client,
        )

        print("\nBot:\n")
        print(answer)
        print("\n" + "-" * 60 + "\n")
