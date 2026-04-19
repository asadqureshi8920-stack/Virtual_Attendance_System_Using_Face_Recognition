from __future__ import annotations

from datetime import datetime

from sqlalchemy import UniqueConstraint

from .extensions import db


class TimestampMixin:
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated_at = db.Column(
        db.DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )


class Teacher(TimestampMixin, db.Model):
    __tablename__ = "teacher"

    teacher_id = db.Column(db.String(20), primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), nullable=False, unique=True)
    phone = db.Column(db.String(20), nullable=False)
    department = db.Column(db.String(80), nullable=False)
    qualification = db.Column(db.String(120), nullable=False)
    designation = db.Column(db.String(80), nullable=False, default="Lecturer")
    date_of_joining = db.Column(db.Date, nullable=False)
    status = db.Column(db.String(20), nullable=False, default="Active")

    classes = db.relationship("ClassRoom", back_populates="teacher", lazy="dynamic")
    user_account = db.relationship("User", back_populates="teacher_profile", uselist=False)


class ClassRoom(TimestampMixin, db.Model):
    __tablename__ = "class"

    class_id = db.Column(db.String(20), primary_key=True)
    class_name = db.Column(db.String(120), nullable=False)
    department = db.Column(db.String(80), nullable=False)
    course = db.Column(db.String(120), nullable=False)
    year = db.Column(db.Integer, nullable=False)
    semester = db.Column(db.Integer, nullable=False)
    division = db.Column(db.String(20), nullable=False)
    teacher_id = db.Column(db.String(20), db.ForeignKey("teacher.teacher_id"), nullable=True)
    status = db.Column(db.String(20), nullable=False, default="Active")

    teacher = db.relationship("Teacher", back_populates="classes")
    students = db.relationship("Student", back_populates="classroom", lazy="dynamic")
    sessions = db.relationship("ClassSession", back_populates="classroom", lazy="dynamic")

    @property
    def label(self) -> str:
        return f"{self.class_name} - Sem {self.semester} {self.division}"


class Student(TimestampMixin, db.Model):
    __tablename__ = "student"

    student_id = db.Column(db.String(20), primary_key=True)
    enrollment_no = db.Column(db.String(30), nullable=False, unique=True)
    name = db.Column(db.String(120), nullable=False)
    department = db.Column(db.String(80), nullable=False)
    course = db.Column(db.String(120), nullable=False)
    year = db.Column(db.Integer, nullable=False)
    semester = db.Column(db.Integer, nullable=False)
    division = db.Column(db.String(20), nullable=False)
    roll_no = db.Column(db.String(20), nullable=False)
    gender = db.Column(db.String(20), nullable=False)
    dob = db.Column(db.Date, nullable=False)
    email = db.Column(db.String(120), nullable=False, unique=True)
    phone = db.Column(db.String(20), nullable=False)
    address = db.Column(db.String(255), nullable=False)
    teacher_name = db.Column(db.String(120), nullable=False)
    class_id = db.Column(db.String(20), db.ForeignKey("class.class_id"), nullable=False)
    photo_path = db.Column(db.String(255), nullable=False, default="")
    dataset_path = db.Column(db.String(255), nullable=True)
    face_samples_count = db.Column(db.Integer, nullable=False, default=0)
    face_model_label = db.Column(db.Integer, nullable=True)
    status = db.Column(db.String(20), nullable=False, default="Active")
    date_of_enrollment = db.Column(db.Date, nullable=False)

    classroom = db.relationship("ClassRoom", back_populates="students")
    attendance_records = db.relationship(
        "Attendance",
        back_populates="student",
        lazy="dynamic",
        cascade="all, delete-orphan",
    )
    user_account = db.relationship("User", back_populates="student_profile", uselist=False)


class ClassSession(TimestampMixin, db.Model):
    __tablename__ = "session"

    session_id = db.Column(db.String(20), primary_key=True)
    class_id = db.Column(db.String(20), db.ForeignKey("class.class_id"), nullable=False)
    session_date = db.Column(db.Date, nullable=False)
    session_time = db.Column(db.Time, nullable=False)
    end_time = db.Column(db.Time, nullable=True)
    meeting_link = db.Column(db.String(255), nullable=True)
    status = db.Column(db.String(20), nullable=False, default="Scheduled")
    recognition_started_at = db.Column(db.DateTime, nullable=True)
    recognition_ended_at = db.Column(db.DateTime, nullable=True)

    classroom = db.relationship("ClassRoom", back_populates="sessions")
    attendance_records = db.relationship(
        "Attendance",
        back_populates="session",
        lazy="dynamic",
        cascade="all, delete-orphan",
    )

    @property
    def display_time(self) -> str:
        end_segment = f" - {self.end_time.strftime('%H:%M')}" if self.end_time else ""
        return f"{self.session_time.strftime('%H:%M')}{end_segment}"


class Attendance(TimestampMixin, db.Model):
    __tablename__ = "attendance"
    __table_args__ = (UniqueConstraint("student_id", "session_id", name="uq_attendance_student_session"),)

    attendance_id = db.Column(db.String(20), primary_key=True)
    student_id = db.Column(db.String(20), db.ForeignKey("student.student_id"), nullable=False)
    session_id = db.Column(db.String(20), db.ForeignKey("session.session_id"), nullable=False)
    student_name = db.Column(db.String(120), nullable=False)
    attendance_date = db.Column(db.Date, nullable=False)
    attendance_time = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    status = db.Column(db.String(20), nullable=False, default="Absent")
    confidence_score = db.Column(db.Float, nullable=False, default=0.0)
    source = db.Column(db.String(40), nullable=False, default="Face Recognition")
    capture_path = db.Column(db.String(255), nullable=True)
    remarks = db.Column(db.String(255), nullable=True)

    student = db.relationship("Student", back_populates="attendance_records")
    session = db.relationship("ClassSession", back_populates="attendance_records")


class User(TimestampMixin, db.Model):
    __tablename__ = "user"

    login_id = db.Column(db.String(50), primary_key=True)
    full_name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), nullable=False, unique=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False)
    teacher_id = db.Column(db.String(20), db.ForeignKey("teacher.teacher_id"), nullable=True)
    student_id = db.Column(db.String(20), db.ForeignKey("student.student_id"), nullable=True)
    status = db.Column(db.String(20), nullable=False, default="Active")
    last_login = db.Column(db.DateTime, nullable=True)

    teacher_profile = db.relationship("Teacher", back_populates="user_account", uselist=False)
    student_profile = db.relationship("Student", back_populates="user_account", uselist=False)


def next_identifier(prefix: str, model, width: int = 3) -> str:
    next_value = model.query.count() + 1
    while True:
        candidate = f"{prefix}{next_value:0{width}d}"
        if db.session.get(model, candidate) is None:
            return candidate
        next_value += 1
