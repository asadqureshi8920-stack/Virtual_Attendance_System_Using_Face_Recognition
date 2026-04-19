from __future__ import annotations

from datetime import datetime, timedelta
from functools import wraps
from secrets import token_hex

from flask import abort, current_app, flash, g, redirect, request, session, url_for

from .extensions import db
from .models import User


PASSWORD_POLICY_TEXT = (
    "Password must be at least 8 characters and include uppercase, lowercase, number, and special character."
)


def validate_password_format(password: str) -> tuple[bool, str]:
    if len(password) < 8:
        return False, PASSWORD_POLICY_TEXT
    if not any(character.isupper() for character in password):
        return False, PASSWORD_POLICY_TEXT
    if not any(character.islower() for character in password):
        return False, PASSWORD_POLICY_TEXT
    if not any(character.isdigit() for character in password):
        return False, PASSWORD_POLICY_TEXT
    if not any(not character.isalnum() for character in password):
        return False, PASSWORD_POLICY_TEXT
    return True, ""


def login_user(user: User) -> None:
    session.clear()
    session.permanent = True
    session["user_id"] = user.login_id
    session["user_role"] = user.role
    session["last_seen_at"] = datetime.utcnow().isoformat()


def logout_user() -> None:
    session.clear()


def get_current_user() -> User | None:
    user_id = session.get("user_id")
    if not user_id:
        return None
    return db.session.get(User, user_id)


def refresh_active_session() -> None:
    user_id = session.get("user_id")
    if not user_id:
        return

    last_seen_raw = session.get("last_seen_at")
    timeout_minutes = int(current_app.config.get("SESSION_TIMEOUT_MINUTES", 30))
    timeout_delta = timedelta(minutes=timeout_minutes)
    if last_seen_raw:
        try:
            last_seen_at = datetime.fromisoformat(last_seen_raw)
        except ValueError:
            last_seen_at = datetime.utcnow() - timeout_delta - timedelta(seconds=1)
        if datetime.utcnow() - last_seen_at > timeout_delta:
            logout_user()
            return

    session["last_seen_at"] = datetime.utcnow().isoformat()


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if g.user is None:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("auth.login"))
        return view(*args, **kwargs)

    return wrapped


def roles_required(*roles):
    def decorator(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            if g.user is None:
                flash("Please log in to continue.", "warning")
                return redirect(url_for("auth.login"))
            if g.user.role not in roles:
                abort(403)
            return view(*args, **kwargs)

        return wrapped

    return decorator


def generate_csrf_token() -> str:
    token = session.get("_csrf_token")
    if not token:
        token = token_hex(16)
        session["_csrf_token"] = token
    return token


def validate_csrf() -> None:
    if request.method != "POST":
        return
    expected = session.get("_csrf_token")
    received = request.form.get("csrf_token") or request.headers.get("X-CSRFToken")
    if not expected or not received or expected != received:
        abort(400, "Invalid form token. Refresh the page and try again.")
