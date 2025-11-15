# backend/seed_more_data.py
import sqlite3
from pathlib import Path
import uuid
from datetime import datetime

DB_PATH = Path(__file__).resolve().parent / "stark.db"

def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # --- ATS EXTRA DATA ---
    cur.execute(
        """
        INSERT INTO ats_jobs (job_id, job_title, company, location,
                              job_description, must_have_skills,
                              nice_to_have_skills, seniority, job_link)
        VALUES
        ('J006','Data Science Intern','Meta','Remote',
         'Support experiments and analyze user behavior data using Python and SQL.',
         'python;sql;statistics',
         'pandas;machine learning','Internship',
         'https://careers.example.com/jobs/J006'),
        ('J007','Operations Research Intern','United Airlines','Hybrid - Chicago, IL',
         'Work on optimization models to improve scheduling and operations.',
         'optimization;python;excel',
         'sql;simulation','Internship',
         'https://careers.example.com/jobs/J007')
        """
    )

    cur.execute(
        """
        INSERT INTO ats_events (event_id, event_name, category, description,
                                skills_covered, date, location, delivery_mode, link)
        VALUES
        ('E005','Intro to Power BI for Terps','Workshop',
         'Hands-on workshop building dashboards in Power BI.',
         'power bi;data visualization','2025-04-10','VMH 1505','In-person',
         'https://umd-events.example.com/E005'),
        ('E006','Advanced SQL Queries','Academic',
         'Practice complex joins, window functions and performance tuning.',
         'sql;advanced sql','2025-04-15','Online','Online',
         'https://umd-events.example.com/E006')
        """
    )

    # --- STUDENT EXTRA DATA ---
    cur.execute(
        """
        INSERT INTO students_summary
            (uid, name, program, level, term,
             gpa_current, gpa_cumulative, credits_earned, credits_required,
             canvas_logins_7d, late_assignments_30d, holds_on_account,
             financial_risk_flag, international_student,
             employment_hours_per_week, reported_challenges,
             career_interest, involvement_notes)
        VALUES
        ('S106','Rohan Mehta','MSIS','Graduate','Spring 2026',
         3.5,3.45,9,36,8,0,'No','No','Yes',10,
         'Balancing TA work and classes','Data Engineering',
         'Active in AI Society; exploring research opportunities'),
        ('S107','Sarah Kim','MS Business Analytics','Graduate','Fall 2025',
         3.8,3.75,21,36,15,0,'No','No','No',0,
         'Wants to publish a data science project','Data Science',
         'Leads a small study group; interested in hackathons')
        """
    )

    cur.execute(
        """
        INSERT INTO advising_notes (uid, note_date, advisor_name, note_text)
        VALUES
        ('S106','2025-03-01','Dr. Emily Rogers',
         'Discussed potential research with a faculty member and course planning.'),
        ('S107','2025-03-02','Dr. Mark Davis',
         'Student wants to build a portfolio project; suggested joining a hackathon.')
        """
    )

    # --- STUDENT PROFILES EXTRA ---
    cur.execute(
        """
        INSERT INTO student_profiles
            (student_id, name, program, level, term,
             gpa_current, interests, career_goal, clubs_joined,
             issues_reported, preferred_learning_style)
        VALUES
        ('S106','Rohan Mehta','MSIS','Graduate','Spring 2026',
         3.5,'data engineering;cloud;python','Data Engineer',
         'AI Society;Cloud Club','Time management with TA duties','Hands-on'),
        ('S107','Sarah Kim','MS Business Analytics','Graduate','Fall 2025',
         3.8,'data science;ml;competitions','ML Engineer',
         'Data Science Club;Analytics Society','Looking for project teammates','Visual')
        """
    )

    # --- CLUBS EXTRA ---
    cur.execute(
        """
        INSERT INTO clubs
            (club_id, name, category, description,
             skills_tags, meeting_pattern, link)
        VALUES
        ('C006','Cloud Computing Club','Tech',
         'Hands-on labs and talks about AWS, Azure, and GCP.',
         'cloud;aws;gcp','Bi-weekly, Mondays 7 PM',
         'https://umd-clubs.example.com/cloud'),
        ('C007','Analytics Society','Tech/Business',
         'Talks and projects around business analytics and BI tools.',
         'analytics;bi;data','Weekly, Thursdays 5 PM',
         'https://umd-clubs.example.com/analytics')
        """
    )

    # --- CAREER ROLES EXTRA ---
    cur.execute(
        """
        INSERT INTO career_roles
            (role_id, role_name, description, skills_required,
             recommended_courses, recommended_events)
        VALUES
        ('R005','Data Engineer Intern',
         'Build and maintain data pipelines and ETL processes.',
         'python;sql;cloud',
         'BUDT702;BUDT704','E002;E003'),
        ('R006','ML Engineer Intern',
         'Train and deploy machine learning models for business problems.',
         'python;ml;statistics',
         'BUDT704;BUDT731','E002')
        """
    )

    conn.commit()
    conn.close()
    print("Extra dummy data inserted ✅")

if __name__ == "__main__":
    main()
