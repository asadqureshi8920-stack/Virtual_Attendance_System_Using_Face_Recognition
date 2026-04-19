from __future__ import annotations

from datetime import date, timedelta
from io import BytesIO

import pandas as pd

from ..extensions import db
from ..models import Attendance, ClassRoom, ClassSession, Student


def resolve_period(period: str, anchor_date: date) -> tuple[date, date]:
    normalized_period = period.lower()
    if normalized_period == "weekly":
        start = anchor_date - timedelta(days=anchor_date.weekday())
        end = start + timedelta(days=6)
        return start, end
    if normalized_period == "monthly":
        start = anchor_date.replace(day=1)
        if start.month == 12:
            next_month = start.replace(year=start.year + 1, month=1, day=1)
        else:
            next_month = start.replace(month=start.month + 1, day=1)
        return start, next_month - timedelta(days=1)
    return anchor_date, anchor_date


def attendance_dataframe(period: str, anchor_date: date) -> tuple[pd.DataFrame, tuple[date, date]]:
    date_range = resolve_period(period, anchor_date)
    start_date, end_date = date_range

    rows = (
        db.session.query(Attendance, Student, ClassSession, ClassRoom)
        .join(Student, Student.student_id == Attendance.student_id)
        .join(ClassSession, ClassSession.session_id == Attendance.session_id)
        .join(ClassRoom, ClassRoom.class_id == ClassSession.class_id)
        .filter(ClassSession.session_date >= start_date, ClassSession.session_date <= end_date)
        .order_by(ClassSession.session_date.desc(), ClassSession.session_time.desc(), Student.roll_no.asc())
        .all()
    )

    records = [
        {
            "Attendance ID": attendance.attendance_id,
            "Session ID": class_session.session_id,
            "Class ID": classroom.class_id,
            "Class Name": classroom.class_name,
            "Student ID": student.student_id,
            "Enrollment No": student.enrollment_no,
            "Name": attendance.student_name,
            "Department": student.department,
            "Course": student.course,
            "Year": student.year,
            "Semester": student.semester,
            "Division": student.division,
            "Roll No": student.roll_no,
            "Date": class_session.session_date.isoformat(),
            "Time": class_session.session_time.strftime("%H:%M"),
            "Status": attendance.status,
            "Confidence %": attendance.confidence_score,
            "Remarks": attendance.remarks or "",
        }
        for attendance, student, class_session, classroom in rows
    ]
    return pd.DataFrame(records), date_range


def summary_from_dataframe(dataframe: pd.DataFrame) -> dict[str, int]:
    if dataframe.empty:
        return {"present": 0, "absent": 0, "unknown": 0, "sessions": 0}
    return {
        "present": int((dataframe["Status"] == "Present").sum()),
        "absent": int((dataframe["Status"] == "Absent").sum()),
        "unknown": int((dataframe["Status"] == "Unknown").sum()),
        "sessions": int(dataframe["Session ID"].nunique()),
    }


def dataframe_to_excel_bytes(dataframe: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        dataframe.to_excel(writer, index=False, sheet_name="Attendance")
    output.seek(0)
    return output.read()
