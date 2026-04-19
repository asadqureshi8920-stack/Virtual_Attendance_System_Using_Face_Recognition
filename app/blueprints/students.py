import json
from datetime import date

from flask import Blueprint, flash, redirect, render_template, request, url_for
from sqlalchemy.exc import IntegrityError
from werkzeug.security import generate_password_hash

from ..extensions import db
from ..models import ClassRoom, Student, User, next_identifier
from ..security import login_required, roles_required, validate_csrf, validate_password_format
from ..services.face_recognition_service import FaceRecognitionService


bp = Blueprint("students", __name__, url_prefix="/students")


def _camera_samples_from_request() -> list[str]:
    raw_value = request.form.get("camera_image_samples", "").strip()
    if not raw_value:
        legacy_value = request.form.get("camera_image_data", "").strip()
        return [legacy_value] if legacy_value else []

    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        return []
    return [sample for sample in parsed if isinstance(sample, str) and sample.strip()]


@bp.get("/")
@login_required
def index():
    students = Student.query.order_by(Student.class_id.asc(), Student.roll_no.asc(), Student.name.asc()).all()
    return render_template("students/list.html", students=students)


@bp.route("/new", methods=["GET", "POST"])
@roles_required("Admin", "Teacher")
def create():
    classes = ClassRoom.query.order_by(ClassRoom.class_name.asc(), ClassRoom.division.asc()).all()

    if request.method == "POST":
        validate_csrf()
        classroom = db.session.get(ClassRoom, request.form["class_id"])
        if classroom is None:
            flash("Select a valid class.", "danger")
            return render_template("students/form.html", classes=classes)

        teacher_name = request.form.get("teacher_name", "").strip() or (
            classroom.teacher.name if classroom.teacher is not None else "Unassigned"
        )
        try:
            student = Student(
                student_id=request.form.get("student_id", "").strip() or next_identifier("STD", Student),
                enrollment_no=request.form["enrollment_no"].strip(),
                name=request.form["name"].strip(),
                department=request.form["department"].strip(),
                course=request.form["course"].strip(),
                year=int(request.form["year"]),
                semester=int(request.form["semester"]),
                division=request.form["division"].strip(),
                roll_no=request.form["roll_no"].strip(),
                gender=request.form["gender"].strip(),
                dob=date.fromisoformat(request.form["dob"]),
                email=request.form["email"].strip(),
                phone=request.form["phone"].strip(),
                address=request.form["address"].strip(),
                teacher_name=teacher_name,
                class_id=classroom.class_id,
                status=request.form["status"],
                date_of_enrollment=date.fromisoformat(request.form["date_of_enrollment"]),
            )
            db.session.add(student)
            db.session.flush()

            login_id = request.form.get("login_id", "").strip() or student.student_id.lower()
            password = request.form.get("password", "").strip() or "Student@123"
            is_valid_password, password_message = validate_password_format(password)
            if not is_valid_password:
                flash(password_message, "danger")
                db.session.rollback()
                return render_template("students/form.html", classes=classes)
            user = User(
                login_id=login_id,
                full_name=student.name,
                email=request.form.get("account_email", "").strip() or f"{login_id}@virtualclassroom.local",
                password_hash=generate_password_hash(password),
                role="Student",
                student_id=student.student_id,
                status="Active",
            )
            db.session.add(user)
            db.session.commit()
        except (IntegrityError, ValueError):
            db.session.rollback()
            flash("Student data could not be saved. Check IDs, email addresses, and dates.", "danger")
            return render_template("students/form.html", classes=classes)

        flash("Student created successfully. Capture the face dataset next.", "success")
        return redirect(url_for("students.capture", student_id=student.student_id))

    return render_template("students/form.html", classes=classes)


@bp.route("/train-model", methods=["GET", "POST"])
@roles_required("Admin", "Teacher")
def train_model():
    students = Student.query.order_by(Student.name.asc()).all()
    if request.method == "POST":
        validate_csrf()
        try:
            result = FaceRecognitionService().train_model()
            db.session.commit()
            flash(result.message, "success")
        except ValueError as error:
            db.session.rollback()
            flash(str(error), "danger")
        return redirect(url_for("students.train_model"))

    return render_template("students/train_model.html", students=students)


@bp.route("/<student_id>/capture", methods=["GET", "POST"])
@bp.route("/<student_id>/enroll", methods=["GET", "POST"])
@roles_required("Admin", "Teacher")
def capture(student_id: str):
    student = db.session.get(Student, student_id)
    if student is None:
        return redirect(url_for("students.index"))

    if request.method == "POST":
        validate_csrf()
        camera_samples = _camera_samples_from_request()
        try:
            result = FaceRecognitionService().enroll_student_face_from_camera(student, camera_samples)
            db.session.commit()
            flash(result.message, "success")
            return redirect(url_for("students.index"))
        except ValueError as error:
            db.session.rollback()
            flash(str(error), "danger")

    return render_template("students/capture.html", student=student)
