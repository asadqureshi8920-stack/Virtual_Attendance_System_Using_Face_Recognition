from flask import Blueprint, g, render_template

from ..models import Attendance, ClassRoom, ClassSession, Student, Teacher
from ..security import login_required


bp = Blueprint("dashboard", __name__)


@bp.route("/")
@bp.route("/dashboard")
@login_required
def index():
    metrics = {
        "students": Student.query.count(),
        "teachers": Teacher.query.count(),
        "classes": ClassRoom.query.count(),
        "sessions": ClassSession.query.count(),
        "attendance": Attendance.query.count(),
    }
    recent_sessions = (
        ClassSession.query.order_by(ClassSession.session_date.desc(), ClassSession.session_time.desc())
        .limit(6)
        .all()
    )

    student_summary = None
    if g.user and g.user.role == "Student" and g.user.student_profile is not None:
        student = g.user.student_profile
        student_records = student.attendance_records.order_by(Attendance.attendance_time.desc()).limit(6).all()
        student_summary = {
            "student": student,
            "records": student_records,
            "present": sum(1 for record in student_records if record.status == "Present"),
        }

    teacher_summary = None
    if g.user and g.user.role == "Teacher" and g.user.teacher_profile is not None:
        teacher = g.user.teacher_profile
        teacher_summary = {
            "teacher": teacher,
            "classes": teacher.classes.order_by(ClassRoom.class_name.asc()).all(),
        }

    return render_template(
        "dashboard/index.html",
        metrics=metrics,
        recent_sessions=recent_sessions,
        student_summary=student_summary,
        teacher_summary=teacher_summary,
    )
