import sqlite3
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = BASE_DIR / "stark.db"


def create_schema(conn: sqlite3.Connection):
    cur = conn.cursor()

    cur.executescript(
        """
        DROP TABLE IF EXISTS ats_jobs;
        DROP TABLE IF EXISTS ats_skill_aliases;
        DROP TABLE IF EXISTS ats_events;
        DROP TABLE IF EXISTS resumes_index;

        DROP TABLE IF EXISTS students_summary;
        DROP TABLE IF EXISTS advising_notes;

        DROP TABLE IF EXISTS student_profiles;
        DROP TABLE IF EXISTS faq;
        DROP TABLE IF EXISTS announcements;
        DROP TABLE IF EXISTS student_emails;
        DROP TABLE IF EXISTS events;
        DROP TABLE IF EXISTS course_catalog;
        DROP TABLE IF EXISTS support_contacts;
        DROP TABLE IF EXISTS career_roles;
        DROP TABLE IF EXISTS clubs;
        DROP TABLE IF EXISTS chat_logs;

        -- NEW: dummy Testudo-like table for backfilling summaries
        DROP TABLE IF EXISTS tested;

        CREATE TABLE ats_jobs (
            job_id TEXT PRIMARY KEY,
            job_title TEXT,
            company TEXT,
            location TEXT,
            job_description TEXT,
            must_have_skills TEXT,
            nice_to_have_skills TEXT,
            seniority TEXT,
            job_link TEXT
        );

        CREATE TABLE ats_skill_aliases (
            canonical_skill TEXT,
            alias_1 TEXT,
            alias_2 TEXT,
            alias_3 TEXT
        );

        CREATE TABLE ats_events (
            event_id TEXT PRIMARY KEY,
            event_name TEXT,
            category TEXT,
            description TEXT,
            skills_covered TEXT,
            date TEXT,             -- ISO 'YYYY-MM-DD'
            location TEXT,
            delivery_mode TEXT,
            link TEXT
        );

        CREATE TABLE resumes_index (
            resume_id TEXT PRIMARY KEY,
            candidate_name TEXT,
            file_path TEXT,
            notes TEXT
        );

        CREATE TABLE students_summary (
            uid TEXT PRIMARY KEY,
            name TEXT,
            program TEXT,
            level TEXT,
            term TEXT,
            gpa_current REAL,
            gpa_cumulative REAL,
            credits_earned INTEGER,
            credits_required INTEGER,
            canvas_logins_7d INTEGER,
            late_assignments_30d INTEGER,
            holds_on_account TEXT,
            financial_risk_flag TEXT,
            international_student TEXT,
            employment_hours_per_week INTEGER,
            reported_challenges TEXT,
            career_interest TEXT,
            involvement_notes TEXT,
            -- Onboarding-style fields linked to UID
            desired_role TEXT,
            desired_industry TEXT,
            work_experience TEXT,
            goals TEXT,
            challenges TEXT,
            involvement TEXT
        );

        CREATE TABLE advising_notes (
            uid TEXT,
            note_date TEXT,
            advisor_name TEXT,
            note_text TEXT
        );

        CREATE TABLE student_profiles (
            student_id TEXT PRIMARY KEY,
            name TEXT,
            program TEXT,
            level TEXT,
            term TEXT,
            gpa_current REAL,
            interests TEXT,
            career_goal TEXT,
            clubs_joined TEXT,
            issues_reported TEXT,
            preferred_learning_style TEXT
        );

        CREATE TABLE faq (
            faq_id TEXT PRIMARY KEY,
            category TEXT,
            question TEXT,
            answer TEXT,
            tags TEXT
        );

        CREATE TABLE announcements (
            announcement_id TEXT PRIMARY KEY,
            date TEXT,
            title TEXT,
            description TEXT,
            category TEXT,
            link TEXT
        );

        CREATE TABLE student_emails (
            email_id TEXT PRIMARY KEY,
            student_id TEXT,
            timestamp TEXT,
            subject TEXT,
            body TEXT,
            category TEXT
        );

        CREATE TABLE events (
            event_id TEXT PRIMARY KEY,
            title TEXT,
            category TEXT,
            description TEXT,
            date TEXT,
            time TEXT,
            location TEXT,
            skills_tags TEXT,
            audience TEXT,
            link TEXT
        );

        CREATE TABLE course_catalog (
            course_id TEXT PRIMARY KEY,
            course_name TEXT,
            credits INTEGER,
            description TEXT,
            prerequisites TEXT,
            tags TEXT
        );

        CREATE TABLE support_contacts (
            support_id TEXT PRIMARY KEY,
            category TEXT,
            name TEXT,
            email TEXT,
            phone TEXT,
            office_hours TEXT,
            notes TEXT
        );

        CREATE TABLE career_roles (
            role_id TEXT PRIMARY KEY,
            role_name TEXT,
            description TEXT,
            skills_required TEXT,
            recommended_courses TEXT,
            recommended_events TEXT
        );

        CREATE TABLE clubs (
            club_id TEXT PRIMARY KEY,
            name TEXT,
            category TEXT,
            description TEXT,
            skills_tags TEXT,
            meeting_pattern TEXT,
            link TEXT
        );

        CREATE TABLE chat_logs (
            log_id TEXT PRIMARY KEY,
            student_id TEXT,
            timestamp TEXT,
            user_query TEXT,
            bot_response TEXT
        );

        -- Dummy Testudo backfill source
        CREATE TABLE tested (
            uid TEXT PRIMARY KEY,
            gpa_current REAL,
            gpa_cumulative REAL,
            credits_earned INTEGER,
            credits_required INTEGER,
            canvas_logins_7d INTEGER,
            late_assignments_30d INTEGER,
            holds_on_account TEXT,          -- "None", "Immunization Hold", etc.
            financial_risk_flag INTEGER,    -- 0/1
            international_student INTEGER,  -- 0/1
            employment_hours_per_week INTEGER
        );
        """
    )
    conn.commit()


def load_csv_to_table(conn, csv_name, table_name):
    path = DATA_DIR / csv_name
    df = pd.read_csv(path)

    # Normalize skills_covered for ats_events to improve LIKE matching
    if table_name == "ats_events" and "skills_covered" in df.columns:
        df["skills_covered"] = df["skills_covered"].fillna("").astype(str).str.lower()

    df.to_sql(table_name, conn, if_exists="append", index=False)
    print(f"Loaded {len(df)} rows into {table_name}")


def seed_tested(conn: sqlite3.Connection):
    """Seed the dummy Testudo-like table with a few demo students."""
    cur = conn.cursor()
    rows = [
        ("U001", 3.70, 3.62, 38, 48, 12, 0, "None", 0, 0, 8),
        ("U002", 3.48, 3.41, 56, 120, 8,  1, "None", 0, 0, 10),
        ("U003", 3.25, 3.18, 24, 120, 15, 2, "Immunization Hold", 1, 1, 5),
        ("U004", 3.92, 3.88, 72, 120, 20, 0, "None", 0, 0, 12),
    ]
    cur.executemany(
        """
        INSERT OR REPLACE INTO tested
        (uid, gpa_current, gpa_cumulative, credits_earned, credits_required,
         canvas_logins_7d, late_assignments_30d, holds_on_account,
         financial_risk_flag, international_student, employment_hours_per_week)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    print(f"Loaded {len(rows)} rows into tested")


def main():
    conn = sqlite3.connect(DB_PATH)
    create_schema(conn)

    load_csv_to_table(conn, "ats_jobs.csv", "ats_jobs")
    load_csv_to_table(conn, "ats_skill_aliases.csv", "ats_skill_aliases")
    load_csv_to_table(conn, "ats_events.csv", "ats_events")
    load_csv_to_table(conn, "resumes_index.csv", "resumes_index")

    load_csv_to_table(conn, "students_summary_input.csv", "students_summary")
    load_csv_to_table(conn, "advising_notes.csv", "advising_notes")

    load_csv_to_table(conn, "student_profiles.csv", "student_profiles")
    load_csv_to_table(conn, "faq_dataset.csv", "faq")
    load_csv_to_table(conn, "announcements.csv", "announcements")
    load_csv_to_table(conn, "student_emails.csv", "student_emails")
    load_csv_to_table(conn, "events_data.csv", "events")
    load_csv_to_table(conn, "course_catalog.csv", "course_catalog")
    load_csv_to_table(conn, "support_contacts.csv", "support_contacts")
    load_csv_to_table(conn, "career_data.csv", "career_roles")
    load_csv_to_table(conn, "clubs_data.csv", "clubs")
    load_csv_to_table(conn, "chat_logs.csv", "chat_logs")

    # Seed the dummy Testudo layer last
    seed_tested(conn)
    conn.close()


if __name__ == "__main__":
    main()
