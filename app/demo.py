from __future__ import annotations

from datetime import date, datetime, time

from werkzeug.security import generate_password_hash

from .extensions import db
from .models import ClassRoom, ClassSession, Student, Teacher, User


def seed_demo_records() -> None:
    if User.query.first() is not None:
        return

    teacher = Teacher(
        teacher_id="TEA001",
        name="Dr. Sana Rahman",
        email="sana.rahman@example.edu",
        phone="03001234567",
        department="Computer Science",
        qualification="MS Computer Science",
        designation="Assistant Professor",
        date_of_joining=date(2024, 1, 15),
        status="Active",
    )
    db.session.add(teacher)

    classroom = ClassRoom(
        class_id="CLS001",
        class_name="BSCS 6A",
        department="Computer Science",
        course="Artificial Intelligence",
        year=3,
        semester=6,
        division="A",
        teacher=teacher,
        status="Active",
    )
    db.session.add(classroom)

    students = [
        Student(
            student_id="STD001",
            enrollment_no="ENR-2026-001",
            name="Areeba Khan",
            department="Computer Science",
            course="Artificial Intelligence",
            year=3,
            semester=6,
            division="A",
            roll_no="01",
            gender="Female",
            dob=date(2004, 5, 12),
            email="areeba.khan@example.edu",
            phone="03004561234",
            address="Lahore, Pakistan",
            teacher_name=teacher.name,
            class_id="CLS001",
            date_of_enrollment=date(2026, 1, 10),
            status="Active",
        ),
        Student(
            student_id="STD002",
            enrollment_no="ENR-2026-002",
            name="Hassan Ali",
            department="Computer Science",
            course="Artificial Intelligence",
            year=3,
            semester=6,
            division="A",
            roll_no="02",
            gender="Male",
            dob=date(2004, 7, 2),
            email="hassan.ali@example.edu",
            phone="03007894561",
            address="Islamabad, Pakistan",
            teacher_name=teacher.name,
            class_id="CLS001",
            date_of_enrollment=date(2026, 1, 10),
            status="Active",
        ),
        Student(
            student_id="STD003",
            enrollment_no="ENR-2026-003",
            name="Zainab Noor",
            department="Computer Science",
            course="Artificial Intelligence",
            year=3,
            semester=6,
            division="A",
            roll_no="03",
            gender="Female",
            dob=date(2004, 11, 18),
            email="zainab.noor@example.edu",
            phone="03009993333",
            address="Karachi, Pakistan",
            teacher_name=teacher.name,
            class_id="CLS001",
            date_of_enrollment=date(2026, 1, 10),
            status="Active",
        ),
    ]
    db.session.add_all(students)

    session_today = ClassSession(
        session_id="SES001",
        class_id="CLS001",
        session_date=date.today(),
        session_time=time(9, 0),
        end_time=time(10, 30),
        meeting_link="https://meet.example.com/bscs-6a",
        status="Scheduled",
    )
    session_next = ClassSession(
        session_id="SES002",
        class_id="CLS001",
        session_date=date.today(),
        session_time=time(13, 0),
        end_time=time(14, 30),
        meeting_link="https://meet.example.com/bscs-6a-afternoon",
        status="Scheduled",
    )
    db.session.add_all([session_today, session_next])
    db.session.flush()

    users = [
        User(
            login_id="admin",
            full_name="System Administrator",
            email="admin@virtualclassroom.local",
            password_hash=generate_password_hash("admin123"),
            role="Admin",
            status="Active",
            last_login=datetime.utcnow(),
        ),
        User(
            login_id="teacher1",
            full_name=teacher.name,
            email="teacher1@virtualclassroom.local",
            password_hash=generate_password_hash("teacher123"),
            role="Teacher",
            teacher_id=teacher.teacher_id,
            status="Active",
        ),
        User(
            login_id="student1",
            full_name=students[0].name,
            email="student1@virtualclassroom.local",
            password_hash=generate_password_hash("student123"),
            role="Student",
            student_id=students[0].student_id,
            status="Active",
        ),
    ]
    db.session.add_all(users)
    db.session.commit()
