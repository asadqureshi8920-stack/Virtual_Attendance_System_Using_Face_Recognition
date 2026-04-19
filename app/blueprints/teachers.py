from datetime import date

from flask import Blueprint, flash, redirect, render_template, request, url_for
from sqlalchemy.exc import IntegrityError
from werkzeug.security import generate_password_hash

from ..extensions import db
from ..models import Teacher, User, next_identifier
from ..security import login_required, roles_required, validate_csrf, validate_password_format


bp = Blueprint("teachers", __name__, url_prefix="/teachers")


@bp.get("/")
@login_required
def index():
    teachers = Teacher.query.order_by(Teacher.name.asc()).all()
    return render_template("teachers/list.html", teachers=teachers)


@bp.route("/new", methods=["GET", "POST"])
@roles_required("Admin")
def create():
    if request.method == "POST":
        validate_csrf()
        try:
            teacher = Teacher(
                teacher_id=next_identifier("TEA", Teacher),
                name=request.form["name"].strip(),
                email=request.form["email"].strip(),
                phone=request.form["phone"].strip(),
                department=request.form["department"].strip(),
                qualification=request.form["qualification"].strip(),
                designation=request.form["designation"].strip(),
                date_of_joining=date.fromisoformat(request.form["date_of_joining"]),
                status=request.form["status"],
            )
            db.session.add(teacher)
            db.session.flush()

            login_id = request.form.get("login_id", "").strip() or f"teacher{Teacher.query.count()}"
            password = request.form.get("password", "").strip() or "Teacher@123"
            is_valid_password, password_message = validate_password_format(password)
            if not is_valid_password:
                flash(password_message, "danger")
                db.session.rollback()
                return render_template("teachers/form.html")
            user = User(
                login_id=login_id,
                full_name=teacher.name,
                email=request.form.get("account_email", "").strip() or f"{login_id}@virtualclassroom.local",
                password_hash=generate_password_hash(password),
                role="Teacher",
                teacher_id=teacher.teacher_id,
                status="Active",
            )
            db.session.add(user)
            db.session.commit()
        except (IntegrityError, ValueError):
            db.session.rollback()
            flash("Teacher data could not be saved. Check for duplicate email, login ID, or invalid dates.", "danger")
            return render_template("teachers/form.html")

        flash("Teacher profile and teacher login were created successfully.", "success")
        return redirect(url_for("teachers.index"))

    return render_template("teachers/form.html")
