from __future__ import annotations

from pathlib import Path

from flask import Flask, g, jsonify, render_template, request

from .blueprints import accounts, auth, classes, dashboard, public, reports, sessions, students, teachers
from .config import Config
from .demo import seed_demo_records
from .extensions import db
from .security import PASSWORD_POLICY_TEXT, generate_csrf_token, get_current_user, refresh_active_session


def create_app(config_class: type[Config] = Config) -> Flask:
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(config_class)

    _ensure_directories(app)
    db.init_app(app)

    @app.before_request
    def load_user() -> None:
        refresh_active_session()
        g.user = get_current_user()

    @app.context_processor
    def inject_template_helpers() -> dict[str, object]:
        return {
            "asset_version": "2026.04.17-change-email",
            "password_policy_text": PASSWORD_POLICY_TEXT,
            "csrf_token": generate_csrf_token,
        }

    @app.get("/healthz")
    def healthz():
        return jsonify({"status": "ok", "service": "virtual-classroom-attendance"})

    @app.errorhandler(400)
    def bad_request(error):
        if request.path.startswith("/attendance/api/"):
            return jsonify({"status": "Error", "message": str(error)}), 400
        return render_template("errors/error.html", code=400, title="Bad Request", message=str(error)), 400

    @app.errorhandler(413)
    def request_too_large(_error):
        message = "Captured camera samples were too large. Refresh the page and try again."
        if request.path.startswith("/attendance/api/"):
            return jsonify({"status": "Error", "message": message}), 413
        return render_template("errors/error.html", code=413, title="Request Too Large", message=message), 413

    @app.errorhandler(403)
    def forbidden(_error):
        if request.path.startswith("/attendance/api/"):
            return jsonify({"status": "Error", "message": "Forbidden"}), 403
        return render_template(
            "errors/error.html",
            code=403,
            title="Access Restricted",
            message="You do not have permission to view this page.",
        ), 403

    @app.errorhandler(404)
    def not_found(_error):
        if request.path.startswith("/attendance/api/"):
            return jsonify({"status": "Error", "message": "Not found"}), 404
        return render_template(
            "errors/error.html",
            code=404,
            title="Page Not Found",
            message="The requested page was not found.",
        ), 404

    @app.errorhandler(500)
    def internal_error(_error):
        db.session.rollback()
        if request.path.startswith("/attendance/api/"):
            return jsonify({"status": "Error", "message": "Internal server error"}), 500
        return render_template(
            "errors/error.html",
            code=500,
            title="System Error",
            message="Something went wrong while processing the request.",
        ), 500

    app.register_blueprint(auth.bp)
    app.register_blueprint(accounts.bp)
    app.register_blueprint(public.bp)
    app.register_blueprint(dashboard.bp)
    app.register_blueprint(teachers.bp)
    app.register_blueprint(classes.bp)
    app.register_blueprint(students.bp)
    app.register_blueprint(sessions.bp)
    app.register_blueprint(reports.bp)

    with app.app_context():
        db.create_all()
        seed_demo_records()

    return app


def _ensure_directories(app: Flask) -> None:
    Path(app.instance_path).mkdir(parents=True, exist_ok=True)
    if "DATABASE_FILE" in app.config:
        Path(app.config["DATABASE_FILE"]).parent.mkdir(parents=True, exist_ok=True)
    for key in (
        "DATA_ROOT",
        "DATASET_FOLDER",
        "CAPTURE_FOLDER",
        "MODEL_FOLDER",
        "REPORT_FOLDER",
        "UPLOAD_FOLDER",
    ):
        Path(app.config[key]).mkdir(parents=True, exist_ok=True)
