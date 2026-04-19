from flask import Blueprint, flash, redirect, render_template, request, url_for
from sqlalchemy.exc import IntegrityError

from ..extensions import db
from ..models import ClassRoom, Teacher, next_identifier
from ..security import login_required, roles_required, validate_csrf


bp = Blueprint("classes", __name__, url_prefix="/classes")


@bp.get("/")
@login_required
def index():
    class_rows = ClassRoom.query.order_by(ClassRoom.class_name.asc(), ClassRoom.division.asc()).all()
    return render_template("classes/list.html", classes=class_rows)


@bp.route("/new", methods=["GET", "POST"])
@roles_required("Admin", "Teacher")
def create():
    teachers = Teacher.query.order_by(Teacher.name.asc()).all()

    if request.method == "POST":
        validate_csrf()
        try:
            classroom = ClassRoom(
                class_id=next_identifier("CLS", ClassRoom),
                class_name=request.form["class_name"].strip(),
                department=request.form["department"].strip(),
                course=request.form["course"].strip(),
                year=int(request.form["year"]),
                semester=int(request.form["semester"]),
                division=request.form["division"].strip(),
                teacher_id=request.form.get("teacher_id") or None,
                status=request.form["status"],
            )
            db.session.add(classroom)
            db.session.commit()
        except (IntegrityError, ValueError):
            db.session.rollback()
            flash("Class data could not be saved. Check the values and try again.", "danger")
            return render_template("classes/form.html", teachers=teachers)

        flash("Class created successfully.", "success")
        return redirect(url_for("classes.index"))

    return render_template("classes/form.html", teachers=teachers)
