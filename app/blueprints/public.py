from __future__ import annotations

from flask import Blueprint, jsonify, render_template, request

from ..extensions import db
from ..models import Student
from ..security import validate_csrf
from ..services.attendance_service import find_session_for_student, process_student_check_in
from ..services.face_recognition_service import FaceRecognitionService


bp = Blueprint("public", __name__)


def _payload_samples(payload: dict) -> list[str]:
    samples = payload.get("camera_image_samples") or []
    if isinstance(samples, list):
        return [sample for sample in samples if isinstance(sample, str) and sample.strip()]

    single_sample = payload.get("camera_image_data")
    return [single_sample] if isinstance(single_sample, str) and single_sample.strip() else []


def _student_from_enrollment(enrollment_no: str) -> Student | None:
    return Student.query.filter_by(enrollment_no=enrollment_no, status="Active").first()


@bp.get("/attendance")
def attendance():
    requested_session_id = request.args.get("session_id", "").strip() or None
    return render_template("public/attendance.html", requested_session_id=requested_session_id)


@bp.post("/attendance/api/identify")
def identify():
    validate_csrf()
    payload = request.get_json(silent=True) or {}
    enrollment_no = str(payload.get("enrollment_no", "")).strip()
    requested_session_id = str(payload.get("session_id", "")).strip() or None
    camera_samples = _payload_samples(payload)

    student = _student_from_enrollment(enrollment_no)
    if student is None:
        return jsonify(
            {
                "status": "Error",
                "label": "Unknown",
                "confidence_percent": 0.0,
                "message": "Enrollment number not found.",
                "face_detected": False,
            }
        ), 404

    session_record = find_session_for_student(student, requested_session_id)
    if session_record is None:
        return jsonify(
            {
                "status": "Error",
                "label": "Unknown",
                "confidence_percent": 0.0,
                "message": "No active session is available for this student.",
                "face_detected": False,
            }
        ), 404

    result = FaceRecognitionService().preview_student_match(student, camera_samples)
    return jsonify(
        {
            "status": result.status,
            "label": result.label,
            "confidence_percent": result.confidence_percent,
            "threshold_score": result.threshold_score,
            "message": result.remarks,
            "face_detected": result.face_detected,
            "session_id": session_record.session_id,
            "class_name": session_record.classroom.class_name,
        }
    )


@bp.post("/attendance/api/check-in")
def check_in():
    validate_csrf()
    payload = request.get_json(silent=True) or {}
    enrollment_no = str(payload.get("enrollment_no", "")).strip()
    requested_session_id = str(payload.get("session_id", "")).strip() or None
    camera_samples = _payload_samples(payload)

    student = _student_from_enrollment(enrollment_no)
    if student is None:
        return jsonify(
            {
                "status": "Error",
                "label": "Unknown",
                "confidence_percent": 0.0,
                "message": "Enrollment number not found.",
            }
        ), 404

    session_record = find_session_for_student(student, requested_session_id)
    if session_record is None:
        return jsonify(
            {
                "status": "Error",
                "label": "Unknown",
                "confidence_percent": 0.0,
                "message": "No active session is available for this student.",
            }
        ), 404

    try:
        outcome = process_student_check_in(session_record, student, camera_samples)
        db.session.commit()
    except ValueError as error:
        db.session.rollback()
        return jsonify(
            {
                "status": "Error",
                "label": "Unknown",
                "confidence_percent": 0.0,
                "message": str(error),
            }
        ), 400

    attendance_record = outcome.attendance_record
    return jsonify(
        {
            "status": outcome.status,
            "label": outcome.label,
            "confidence_percent": outcome.confidence_percent,
            "message": outcome.remarks,
            "duplicate_blocked": outcome.duplicate_blocked,
            "auto_enrolled": outcome.auto_enrolled,
            "attendance_marked": outcome.status == "Present",
            "session_id": session_record.session_id,
            "class_name": session_record.classroom.class_name,
            "attendance_id": attendance_record.attendance_id if attendance_record else "",
        }
    )
