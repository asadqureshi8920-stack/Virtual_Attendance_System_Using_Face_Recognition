from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from ..extensions import db
from ..models import Attendance, ClassSession, Student, next_identifier
from .face_recognition_service import FaceRecognitionService, RecognitionResult


@dataclass
class AttendanceCheckInResult:
    status: str
    label: str
    confidence_percent: float
    remarks: str
    duplicate_blocked: bool
    auto_enrolled: bool
    attendance_record: Attendance | None
    recognition: RecognitionResult | None


def sync_session_attendance(session_record: ClassSession) -> list[Attendance]:
    roster = session_record.classroom.students.order_by(Student.roll_no.asc(), Student.name.asc()).all()
    created_records: list[Attendance] = []
    for student in roster:
        attendance = Attendance.query.filter_by(
            session_id=session_record.session_id,
            student_id=student.student_id,
        ).first()
        if attendance is not None:
            continue

        attendance = Attendance(
            attendance_id=next_identifier("ATT", Attendance),
            student_id=student.student_id,
            session_id=session_record.session_id,
            student_name=student.name,
            attendance_date=session_record.session_date,
            attendance_time=datetime.combine(session_record.session_date, session_record.session_time),
            status="Absent",
            confidence_score=0.0,
            remarks="Waiting for face recognition.",
        )
        db.session.add(attendance)
        created_records.append(attendance)

    if session_record.status == "Scheduled":
        session_record.status = "Ongoing"
        session_record.recognition_started_at = datetime.utcnow()

    db.session.commit()
    return created_records


def find_session_for_student(student: Student, session_id: str | None = None) -> ClassSession | None:
    query = ClassSession.query.filter_by(class_id=student.class_id)
    if session_id:
        return query.filter_by(session_id=session_id).first()

    today = date.today()
    ongoing = (
        query.filter(ClassSession.session_date == today, ClassSession.status.in_(["Ongoing", "Scheduled"]))
        .order_by(ClassSession.status.asc(), ClassSession.session_time.asc())
        .first()
    )
    if ongoing is not None:
        return ongoing

    return (
        query.filter(ClassSession.session_date >= today)
        .order_by(ClassSession.session_date.asc(), ClassSession.session_time.asc())
        .first()
    )


def process_student_check_in(
    session_record: ClassSession,
    student: Student,
    camera_image_samples: list[str],
) -> AttendanceCheckInResult:
    sync_session_attendance(session_record)
    attendance = Attendance.query.filter_by(
        session_id=session_record.session_id,
        student_id=student.student_id,
    ).first()

    if attendance is not None and attendance.status == "Present":
        return AttendanceCheckInResult(
            status="Duplicate",
            label=student.name,
            confidence_percent=attendance.confidence_score,
            remarks="Duplicate attendance blocked for this session.",
            duplicate_blocked=True,
            auto_enrolled=False,
            attendance_record=attendance,
            recognition=None,
        )

    face_service = FaceRecognitionService()
    if student.face_samples_count == 0:
        face_service.enroll_student_face_from_camera(student, camera_image_samples)
        recognition = RecognitionResult(
            status="Present",
            label=student.name,
            student_id=student.student_id,
            confidence_percent=100.0,
            threshold_score=0.0,
            remarks="Face enrolled automatically and attendance marked.",
            face_detected=True,
        )
        _mark_present(attendance, session_record, student, recognition)
        db.session.commit()
        return AttendanceCheckInResult(
            status="Present",
            label=student.name,
            confidence_percent=100.0,
            remarks=recognition.remarks,
            duplicate_blocked=False,
            auto_enrolled=True,
            attendance_record=attendance,
            recognition=recognition,
        )

    recognition = face_service.verify_student_for_samples(
        student,
        camera_image_samples,
        capture_group="session_checkins",
        capture_identifier=session_record.session_id,
    )

    if recognition.status == "Present":
        _mark_present(attendance, session_record, student, recognition)
        db.session.commit()
        return AttendanceCheckInResult(
            status="Present",
            label=recognition.label,
            confidence_percent=recognition.confidence_percent,
            remarks=recognition.remarks,
            duplicate_blocked=False,
            auto_enrolled=False,
            attendance_record=attendance,
            recognition=recognition,
        )

    attendance.attendance_time = datetime.utcnow()
    attendance.confidence_score = recognition.confidence_percent
    attendance.capture_path = recognition.capture_path or attendance.capture_path
    attendance.remarks = recognition.remarks
    db.session.commit()
    return AttendanceCheckInResult(
        status=recognition.status,
        label=recognition.label,
        confidence_percent=recognition.confidence_percent,
        remarks=recognition.remarks,
        duplicate_blocked=False,
        auto_enrolled=False,
        attendance_record=attendance,
        recognition=recognition,
    )


def _mark_present(
    attendance: Attendance | None,
    session_record: ClassSession,
    student: Student,
    recognition: RecognitionResult,
) -> Attendance:
    if attendance is None:
        attendance = Attendance(
            attendance_id=next_identifier("ATT", Attendance),
            student_id=student.student_id,
            session_id=session_record.session_id,
            student_name=student.name,
            attendance_date=session_record.session_date,
        )
        db.session.add(attendance)

    attendance.student_name = student.name
    attendance.attendance_date = session_record.session_date
    attendance.attendance_time = datetime.utcnow()
    attendance.status = "Present"
    attendance.confidence_score = recognition.confidence_percent
    attendance.capture_path = recognition.capture_path or attendance.capture_path
    attendance.remarks = recognition.remarks
    session_record.status = "Ongoing"
    if session_record.recognition_started_at is None:
        session_record.recognition_started_at = datetime.utcnow()
    return attendance
