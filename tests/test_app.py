import unittest
from pathlib import Path

from app import create_app
from werkzeug.security import check_password_hash

from app.models import Attendance, User


class TestConfig:
    TESTING = True
    SECRET_KEY = "test-secret"
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {"connect_args": {"check_same_thread": False}}
    SESSION_TIMEOUT_MINUTES = 30
    FACE_IMAGE_SIZE = 160
    FACE_MATCH_THRESHOLD = 0.55
    FACE_DISTANCE_CEILING = 100.0
    FACE_MIN_QUALITY = 0.34
    FACE_MATCH_DOMINANCE = 0.55
    FACE_MIN_CONSENSUS_FRAMES = 2
    FACE_MAX_ENROLLMENT_SAMPLES = 8
    MAX_CONTENT_LENGTH = 8 * 1024 * 1024
    DATA_ROOT = Path("test_data")
    DATASET_FOLDER = Path("test_data/dataset")
    CAPTURE_FOLDER = Path("test_data/captures")
    MODEL_FOLDER = Path("test_data/models")
    TRAINER_FILE = Path("test_data/models/trainer.yml")
    LABELS_FILE = Path("test_data/models/labels.json")
    REPORT_FOLDER = Path("test_data/reports")
    UPLOAD_FOLDER = Path("test_data/uploads")
    DATABASE_FILE = Path("test_data/test.db")


class AppTestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app(TestConfig)
        self.client = self.app.test_client()

    def _set_csrf(self, token: str = "test-token") -> str:
        with self.client.session_transaction() as session:
            session["_csrf_token"] = token
        return token

    def _login(self) -> None:
        token = self._set_csrf()
        response = self.client.post(
            "/login",
            data={"csrf_token": token, "login_id": "admin", "password": "admin123"},
            follow_redirects=True,
        )
        self.assertEqual(response.status_code, 200)

    def test_login_page_loads(self):
        response = self.client.get("/login")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Login", response.data)
        self.assertNotIn(b"admin123", response.data)
        self.assertNotIn(b"teacher123", response.data)
        self.assertNotIn(b"student123", response.data)
        self.assertIn(b"Change Username/Password", response.data)
        self.assertIn(b"Forgot Password", response.data)

    def test_seeded_roles_exist(self):
        with self.app.app_context():
            self.assertIsNotNone(User.query.filter_by(login_id="admin", role="Admin").first())
            self.assertIsNotNone(User.query.filter_by(login_id="teacher1", role="Teacher").first())
            self.assertIsNotNone(User.query.filter_by(login_id="student1", role="Student").first())

    def test_management_pages_load_after_login(self):
        self._login()
        for path in (
            "/dashboard",
            "/teachers/",
            "/accounts/",
            "/classes/",
            "/students/",
            "/students/train-model",
            "/sessions/",
            "/reports/",
        ):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200, path)

    def test_start_session_creates_absent_records(self):
        self._login()
        token = self._set_csrf()
        response = self.client.post(
            "/sessions/SES001/start",
            data={"csrf_token": token},
            follow_redirects=True,
        )
        self.assertEqual(response.status_code, 200)
        with self.app.app_context():
            self.assertEqual(Attendance.query.filter_by(session_id="SES001").count(), 3)

    def test_report_exports_work(self):
        self._login()
        csv_response = self.client.get("/reports/export/csv?period=daily")
        excel_response = self.client.get("/reports/export/excel?period=daily")
        self.assertEqual(csv_response.status_code, 200)
        self.assertIn("text/csv", csv_response.headers.get("Content-Type", ""))
        self.assertEqual(excel_response.status_code, 200)
        self.assertIn(
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            excel_response.headers.get("Content-Type", ""),
        )

    def test_public_identify_rejects_unknown_enrollment(self):
        token = self._set_csrf()
        response = self.client.post(
            "/attendance/api/identify",
            json={"enrollment_no": "BAD", "camera_image_samples": []},
            headers={"X-CSRFToken": token},
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.get_json()["status"], "Error")

    def test_admin_can_update_user_login_and_password(self):
        self._login()
        token = self._set_csrf()
        response = self.client.post(
            "/accounts/student1/edit",
            data={
                "csrf_token": token,
                "login_id": "student-new",
                "password": "Newpass@123",
                "confirm_password": "Newpass@123",
            },
            follow_redirects=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Login details updated", response.data)

        with self.app.app_context():
            self.assertIsNone(User.query.filter_by(login_id="student1").first())
            updated_user = User.query.filter_by(login_id="student-new", role="Student").first()
            self.assertIsNotNone(updated_user)
            self.assertTrue(check_password_hash(updated_user.password_hash, "Newpass@123"))

    def test_login_page_credential_change_requires_admin_verification(self):
        token = self._set_csrf()
        response = self.client.post(
            "/change-credentials",
            data={
                "csrf_token": token,
                "admin_login_id": "wrong",
                "admin_password": "wrong",
                "target_role": "Student",
                "current_login_id": "student1",
                "account_email": "student1@virtualclassroom.local",
                "new_login_id": "student-from-login",
                "new_password": "weakpass",
                "confirm_password": "weakpass",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Admin verification failed", response.data)

        token = self._set_csrf()
        response = self.client.post(
            "/change-credentials",
            data={
                "csrf_token": token,
                "admin_login_id": "admin",
                "admin_password": "admin123",
                "target_role": "Student",
                "current_login_id": "student1",
                "account_email": "student1@virtualclassroom.local",
                "new_login_id": "student-from-login",
                "new_password": "Newpass@123",
                "confirm_password": "Newpass@123",
            },
            follow_redirects=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"updated successfully", response.data)

        with self.app.app_context():
            updated_user = User.query.filter_by(login_id="student-from-login", role="Student").first()
            self.assertIsNotNone(updated_user)
            self.assertTrue(check_password_hash(updated_user.password_hash, "Newpass@123"))

    def test_login_page_credential_change_can_update_admin_accounts(self):
        token = self._set_csrf()
        response = self.client.post(
            "/change-credentials",
            data={
                "csrf_token": token,
                "admin_login_id": "admin",
                "admin_password": "admin123",
                "target_role": "Admin",
                "current_login_id": "admin",
                "account_email": "admin@virtualclassroom.local",
                "new_login_id": "admin-new",
                "new_password": "Adminnew@123",
                "confirm_password": "Adminnew@123",
            },
            follow_redirects=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"updated successfully", response.data)

        with self.app.app_context():
            self.assertIsNone(User.query.filter_by(login_id="admin").first())
            updated_user = User.query.filter_by(login_id="admin-new", role="Admin").first()
            self.assertIsNotNone(updated_user)
            self.assertTrue(check_password_hash(updated_user.password_hash, "Adminnew@123"))

    def test_credential_change_requires_registered_email_match(self):
        token = self._set_csrf()
        response = self.client.post(
            "/change-credentials",
            data={
                "csrf_token": token,
                "admin_login_id": "admin",
                "admin_password": "admin123",
                "target_role": "Teacher",
                "current_login_id": "teacher1",
                "account_email": "wrong@example.com",
                "new_login_id": "teacher-new",
                "new_password": "Teachernew@123",
                "confirm_password": "Teachernew@123",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Registered email does not match", response.data)

    def test_forgot_password_resets_verified_account(self):
        token = self._set_csrf()
        weak_response = self.client.post(
            "/forgot-password",
            data={
                "csrf_token": token,
                "target_role": "Student",
                "login_id": "student1",
                "account_email": "student1@virtualclassroom.local",
                "new_password": "weakpass",
                "confirm_password": "weakpass",
            },
        )
        self.assertEqual(weak_response.status_code, 200)
        self.assertIn(b"Password must be at least 8 characters", weak_response.data)

        token = self._set_csrf()
        response = self.client.post(
            "/forgot-password",
            data={
                "csrf_token": token,
                "target_role": "Student",
                "login_id": "student1",
                "account_email": "student1@virtualclassroom.local",
                "new_password": "Reset@123",
                "confirm_password": "Reset@123",
            },
            follow_redirects=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Password reset successfully", response.data)

        with self.app.app_context():
            updated_user = User.query.filter_by(login_id="student1", role="Student").first()
            self.assertIsNotNone(updated_user)
            self.assertTrue(check_password_hash(updated_user.password_hash, "Reset@123"))


if __name__ == "__main__":
    unittest.main()
