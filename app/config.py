from __future__ import annotations

import os
import tempfile
from datetime import timedelta
from pathlib import Path

from sqlalchemy.pool import StaticPool


BASE_DIR = Path(__file__).resolve().parent.parent
INSTANCE_DIR = BASE_DIR / "instance"
APP_DATA_DIR = BASE_DIR / "app_data"
LOCAL_DATA_DIR = Path(os.getenv("LOCALAPPDATA", tempfile.gettempdir())) / "VirtualClassroomAttendance"
DEFAULT_DATABASE_PATH = LOCAL_DATA_DIR / "virtual_classroom_attendance.db"
DEFAULT_DATABASE_URI = f"sqlite:///{DEFAULT_DATABASE_PATH.as_posix()}"


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-me")
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL",
        DEFAULT_DATABASE_URI,
    )
    DATABASE_FILE = Path(os.getenv("DATABASE_FILE", DEFAULT_DATABASE_PATH.as_posix()))
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = (
        {
            "connect_args": {"check_same_thread": False},
            "poolclass": StaticPool,
        }
        if SQLALCHEMY_DATABASE_URI.endswith(":memory:")
        else (
            {"connect_args": {"check_same_thread": False}}
            if SQLALCHEMY_DATABASE_URI.startswith("sqlite")
            else {}
        )
    )

    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = False
    PERMANENT_SESSION_LIFETIME = timedelta(minutes=int(os.getenv("SESSION_TIMEOUT_MINUTES", "30")))
    SESSION_TIMEOUT_MINUTES = int(os.getenv("SESSION_TIMEOUT_MINUTES", "30"))

    FACE_IMAGE_SIZE = int(os.getenv("FACE_IMAGE_SIZE", "224"))
    FACE_MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.55"))
    FACE_DISTANCE_CEILING = float(os.getenv("FACE_DISTANCE_CEILING", "100.0"))
    FACE_MIN_QUALITY = float(os.getenv("FACE_MIN_QUALITY", "0.34"))
    FACE_MATCH_DOMINANCE = float(os.getenv("FACE_MATCH_DOMINANCE", "0.55"))
    FACE_MIN_CONSENSUS_FRAMES = int(os.getenv("FACE_MIN_CONSENSUS_FRAMES", "2"))
    FACE_MAX_ENROLLMENT_SAMPLES = int(os.getenv("FACE_MAX_ENROLLMENT_SAMPLES", "8"))
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", str(32 * 1024 * 1024)))

    DATA_ROOT = APP_DATA_DIR
    DATASET_FOLDER = APP_DATA_DIR / "dataset"
    CAPTURE_FOLDER = APP_DATA_DIR / "captures"
    MODEL_FOLDER = APP_DATA_DIR / "models"
    TRAINER_FILE = APP_DATA_DIR / "models" / "trainer.yml"
    LABELS_FILE = APP_DATA_DIR / "models" / "labels.json"
    REPORT_FOLDER = APP_DATA_DIR / "reports"
    UPLOAD_FOLDER = APP_DATA_DIR / "uploads"
