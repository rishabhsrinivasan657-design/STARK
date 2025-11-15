# main_chatbot.py

from stark_context import StudentContext

print("🔧 Loading STARK modules...")

# ATS / résumé analyzer
try:
    from resume_analyzer_integrated import run_ats_flow
    print("  ✅ resume_analyzer_integrated.run_ats_flow loaded")
except Exception as e:
    print("  ⚠️ Could not import run_ats_flow:", e)

    def run_ats_flow(context: StudentContext) -> str:
        return "ATS flow not available (import error)."


# FAQ chatbot
try:
    from student_faq_chatbot_llm import run_faq_session
    print("  ✅ student_faq_chatbot_llm.run_faq_session loaded")
except Exception as e:
    print("  ⚠️ Could not import run_faq_session:", e)

    def run_faq_session(context: StudentContext) -> None:
        print("FAQ module not available (import error).")


# Summary module
try:
    from summary_module import generate_summary
    print("  ✅ summary_module.generate_summary loaded")
except Exception as e:
    print("  ⚠️ Could not import summary_module.generate_summary:", e)

    def generate_summary(context: StudentContext) -> str:
        text = (
            f"(Stub summary)\n"
            f"Student: {context.name} ({context.uid})\n"
            f"Program: {context.program}, Term: {context.term}\n"
            f"Skills: {', '.join(sorted(context.skills)) or 'None'}\n"
            f"Missing: {', '.join(sorted(context.missing_skills)) or 'None'}\n"
        )
        context.summary_history.append(text)
        return text


def main():
    print("\n🤖 Welcome to STARK – Unified Chatbot")
    print("=====================================")

    uid = input("Enter student ID: ").strip() or "0001"
    name = input("Enter your name: ").strip() or "Student"
    program = input("Program (e.g., MSIS, MBA): ").strip() or "MSIS"
    level = input("Level (Graduate/Undergrad): ").strip() or "Graduate"
    term = input("Term (e.g., Fall 2025): ").strip() or "Fall 2025"

    context = StudentContext(
        uid=uid,
        name=name,
        program=program,
        level=level,
        term=term,
    )

    while True:
        print("\nWhat would you like to do?")
        print("1) Resume & ATS checker")
        print("2) Student FAQ chatbot")
        print("3) Generate advisor summary")
        print("4) View current context snapshot")
        print("0) Exit")

        choice = input("> ").strip()

        if choice == "1":
            print("\n-------------------- ATS CHECKER --------------------")
            result = run_ats_flow(context)
            print("\n" + result + "\n")

        elif choice == "2":
            print("\n-------------------- FAQ CHATBOT --------------------")
            run_faq_session(context)

        elif choice == "3":
            print("\n------------------ ADVISOR SUMMARY ------------------")
            summary_text = generate_summary(context)
            print(summary_text + "\n")

        elif choice == "4":
            print("\n----------------- CONTEXT SNAPSHOT ------------------")
            print(context)

        elif choice == "0":
            print("\n👋 Goodbye from STARK")
            break

        else:
            print("Invalid choice, try again.")


if __name__ == "__main__":
    print("🚀 Starting main_chatbot.py")
    main()
