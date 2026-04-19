from datetime import datetime

from flask import Blueprint, flash, g, redirect, render_template, request, url_for
from sqlalchemy.exc import SQLAlchemyError
from werkzeug.security import check_password_hash, generate_password_hash

from ..extensions import db
from ..models import User
from ..security import login_user, logout_user, validate_csrf, validate_password_format


bp = Blueprint("auth", __name__)


@bp.route("/login", methods=["GET", "POST"])
def login():
    if g.user is not None:
        return redirect(url_for("dashboard.index"))

    if request.method == "POST":
        validate_csrf()
        login_id = request.form.get("login_id", "").strip()
        password = request.form.get("password", "").strip()

        user = User.query.filter_by(login_id=login_id, status="Active").first()
        if user is not None and check_password_hash(user.password_hash, password):
            user.last_login = datetime.utcnow()
            try:
                db.session.commit()
            except SQLAlchemyError:
                db.session.rollback()
            login_user(user)
            flash("Login successful.", "success")
            return redirect(url_for("dashboard.index"))

        flash("Invalid username or password.", "danger")

    return render_template("auth/login.html")


@bp.route("/change-credentials", methods=["GET", "POST"])
def change_credentials():
    if request.method == "POST":
        validate_csrf()
        admin_login_id = request.form.get("admin_login_id", "").strip()
        admin_password = request.form.get("admin_password", "").strip()
        target_role = request.form.get("target_role", "").strip()
        current_login_id = request.form.get("current_login_id", "").strip()
        account_email = request.form.get("account_email", "").strip().lower()
        new_login_id = request.form.get("new_login_id", "").strip()
        new_password = request.form.get("new_password", "").strip()
        confirm_password = request.form.get("confirm_password", "").strip()

        admin_user = User.query.filter_by(login_id=admin_login_id, role="Admin", status="Active").first()
        if admin_user is None or not check_password_hash(admin_user.password_hash, admin_password):
            flash("Admin verification failed. Enter valid admin credentials.", "danger")
            return render_template("auth/change_credentials.html")

        if target_role not in {"Admin", "Teacher", "Student"}:
            flash("Only admin, teacher, and student credentials can be changed from this page.", "danger")
            return render_template("auth/change_credentials.html")

        target_user = User.query.filter_by(login_id=current_login_id, role=target_role).first()
        if target_user is None:
            flash(f"{target_role} account was not found.", "danger")
            return render_template("auth/change_credentials.html")

        if target_user.email.lower() != account_email:
            flash("Registered email does not match the selected account.", "danger")
            return render_template("auth/change_credentials.html")

        if not new_login_id:
            flash("New login ID is required.", "danger")
            return render_template("auth/change_credentials.html")

        existing_user = db.session.get(User, new_login_id)
        if existing_user is not None and existing_user.login_id != target_user.login_id:
            flash("That new login ID is already in use.", "danger")
            return render_template("auth/change_credentials.html")

        if new_password:
            is_valid_password, password_message = validate_password_format(new_password)
            if not is_valid_password:
                flash(password_message, "danger")
                return render_template("auth/change_credentials.html")
            if new_password != confirm_password:
                flash("Password confirmation does not match.", "danger")
                return render_template("auth/change_credentials.html")
            target_user.password_hash = generate_password_hash(new_password)

        target_user.login_id = new_login_id
        db.session.commit()
        flash(f"{target_role} login details updated successfully after admin verification.", "success")
        return redirect(url_for("auth.login"))

    return render_template("auth/change_credentials.html")


@bp.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        validate_csrf()
        target_role = request.form.get("target_role", "").strip()
        login_id = request.form.get("login_id", "").strip()
        account_email = request.form.get("account_email", "").strip().lower()
        new_password = request.form.get("new_password", "").strip()
        confirm_password = request.form.get("confirm_password", "").strip()

        if target_role not in {"Admin", "Teacher", "Student"}:
            flash("Select a valid account type.", "danger")
            return render_template("auth/forgot_password.html")

        user = User.query.filter_by(login_id=login_id, role=target_role, status="Active").first()
        if user is None or user.email.lower() != account_email:
            flash("Account details could not be verified.", "danger")
            return render_template("auth/forgot_password.html")

        is_valid_password, password_message = validate_password_format(new_password)
        if not is_valid_password:
            flash(password_message, "danger")
            return render_template("auth/forgot_password.html")

        if new_password != confirm_password:
            flash("Password confirmation does not match.", "danger")
            return render_template("auth/forgot_password.html")

        user.password_hash = generate_password_hash(new_password)
        db.session.commit()
        flash("Password reset successfully. You can sign in with the new password.", "success")
        return redirect(url_for("auth.login"))

    return render_template("auth/forgot_password.html")


@bp.post("/logout")
def logout():
    validate_csrf()
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("auth.login"))
