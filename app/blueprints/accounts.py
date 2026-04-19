from __future__ import annotations

from flask import Blueprint, flash, redirect, render_template, request, session, url_for
from werkzeug.security import generate_password_hash

from ..extensions import db
from ..models import User
from ..security import roles_required, validate_csrf, validate_password_format


bp = Blueprint("accounts", __name__, url_prefix="/accounts")


@bp.get("/")
@roles_required("Admin")
def index():
    users = User.query.order_by(User.role.asc(), User.full_name.asc()).all()
    return render_template("accounts/index.html", users=users)


@bp.route("/<path:login_id>/edit", methods=["GET", "POST"])
@roles_required("Admin")
def edit(login_id: str):
    user = db.session.get(User, login_id)
    if user is None:
        flash("User account was not found.", "danger")
        return redirect(url_for("accounts.index"))

    if request.method == "POST":
        validate_csrf()
        new_login_id = request.form.get("login_id", "").strip()
        new_password = request.form.get("password", "").strip()
        confirm_password = request.form.get("confirm_password", "").strip()

        if not new_login_id:
            flash("Login ID is required.", "danger")
            return render_template("accounts/edit.html", user=user)

        if new_login_id != user.login_id and db.session.get(User, new_login_id) is not None:
            flash("That login ID is already in use.", "danger")
            return render_template("accounts/edit.html", user=user)

        if new_password:
            is_valid_password, password_message = validate_password_format(new_password)
            if not is_valid_password:
                flash(password_message, "danger")
                return render_template("accounts/edit.html", user=user)
            if new_password != confirm_password:
                flash("Password confirmation does not match.", "danger")
                return render_template("accounts/edit.html", user=user)
            user.password_hash = generate_password_hash(new_password)

        old_login_id = user.login_id
        user.login_id = new_login_id
        db.session.commit()

        if session.get("user_id") == old_login_id:
            session["user_id"] = new_login_id

        flash(f"Login details updated for {user.full_name}.", "success")
        return redirect(url_for("accounts.index"))

    return render_template("accounts/edit.html", user=user)
