from datetime import date, time

from flask import Blueprint, flash, redirect, render_template, request, url_for
from sqlalchemy.exc import IntegrityError

from ..extensions import db
from ..models import Attendance, ClassRoom, ClassSession, Student, next_identifier
from ..security import login_required, roles_required, validate_csrf
from ..services.attendance_service import sync_session_attendance


bp = Blueprint("sessions", __name__, url_prefix="/sessions")


@bp.get("/")
@login_required
def index():
    sessions = (
        ClassSession.query.order_by(ClassSession.session_date.desc(), ClassSession.session_time.desc()).all()
    )
    return render_template("sessions/list.html", sessions=sessions)


@bp.route("/new", methods=["GET", "POST"])
@roles_required("Admin", "Teacher")
def create():
    classes = ClassRoom.query.order_by(ClassRoom.class_name.asc(), ClassRoom.division.asc()).all()

    if request.method == "POST":
        validate_csrf()
        try:
            session_record = ClassSession(
                session_id=request.form.get("session_id", "").strip() or next_identifier("SES", ClassSession),
                class_id=request.form["class_id"],
                session_date=date.fromisoformat(request.form["session_date"]),
                session_time=time.fromisoformat(request.form["session_time"]),
                end_time=time.fromisoformat(request.form["end_time"]) if request.form.get("end_time") else None,
                meeting_link=request.form.get("meeting_link", "").strip() or None,
                status=request.form["status"],
            )
            db.session.add(session_record)
            db.session.commit()
        except (IntegrityError, ValueError):
            db.session.rollback()
            flash("Session data could not be saved. Check for duplicate session IDs or invalid dates and times.", "danger")
            return render_template("sessions/form.html", classes=classes)

        sync_session_attendance(session_record)
        flash("Session created successfully.", "success")
        return redirect(url_for("sessions.index"))

    return render_template("sessions/form.html", classes=classes)


@bp.route("/<session_id>/start", methods=["GET", "POST"])
@roles_required("Admin", "Teacher")
def start(session_id: str):
    session_record = db.session.get(ClassSession, session_id)
    if session_record is None:
        return redirect(url_for("sessions.index"))

    if request.method == "POST":
        validate_csrf()
        sync_session_attendance(session_record)
        flash("Recognition started. Students can now use the attendance portal.", "success")
        return redirect(url_for("sessions.recognize", session_id=session_id))

    sync_session_attendance(session_record)
    return render_template("sessions/start.html", session_record=session_record)


@bp.route("/<session_id>/recognize", methods=["GET"])
@roles_required("Admin", "Teacher")
def recognize(session_id: str):
    session_record = db.session.get(ClassSession, session_id)
    if session_record is None:
        return redirect(url_for("sessions.index"))

    sync_session_attendance(session_record)
    students = session_record.classroom.students.order_by(Student.roll_no.asc(), Student.name.asc()).all()
    attendance_records = (
        Attendance.query.filter_by(session_id=session_id)
        .order_by(Attendance.status.desc(), Attendance.attendance_time.desc(), Attendance.student_name.asc())
        .all()
    )
    metrics = {
        "present": sum(1 for record in attendance_records if record.status == "Present"),
        "absent": sum(1 for record in attendance_records if record.status == "Absent"),
        "unknown": sum(1 for record in attendance_records if record.status == "Unknown"),
    }
    enrolled_count = sum(1 for student in students if student.face_samples_count > 0)
    return render_template(
        "sessions/recognize.html",
        session_record=session_record,
        students=students,
        attendance_records=attendance_records,
        metrics=metrics,
        enrolled_count=enrolled_count,
    )
