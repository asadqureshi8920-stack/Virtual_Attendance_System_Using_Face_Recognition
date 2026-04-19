<<<<<<< HEAD
# Virtual Classroom Attendance Using Face Recognition

This is a complete Flask-based virtual classroom attendance system with:

- Role-based authentication for `Admin`, `Teacher`, and `Student`
- Student, class, session, and attendance management
- OpenCV Haarcascade face detection
- LBPH face recognition with dataset storage and `trainer.yml` generation
- Public webcam attendance portal using enrollment number only
- Automatic attendance marking with duplicate prevention
- Daily, weekly, and monthly reports with CSV and Excel export

## Default Accounts

- `admin` / `admin123`
- `teacher1` / `teacher123`
- `student1` / `student123`

## Run

```powershell
python run.py
```

Open:

- `http://127.0.0.1:5000/login`
- `http://127.0.0.1:5000/attendance`

## Core Workflow

1. Login as admin or teacher.
2. Add teacher, class, and student records.
3. Open `Capture Face` for each student to save dataset images.
4. Open `Train Model` to generate the LBPH trainer file.
5. Create and start a class session.
6. Students open `/attendance`, enter enrollment number, and use webcam recognition.
7. Attendance is stored automatically in the database.
8. Reports can be filtered and exported to CSV or Excel.

## Storage

- Default SQLite database: `%LOCALAPPDATA%\\VirtualClassroomAttendance\\virtual_classroom_attendance.db`
- Dataset images: `app_data/dataset/`
- Captures: `app_data/captures/`
- Model files: `app_data/models/trainer.yml` and `app_data/models/labels.json`

## Tests

```powershell
python -m unittest discover -s tests -v
```
=======
# Virtual_Attendance_System_Using_Face_Recognitions
>>>>>>> fbfb7b34b1123c1204c169bd94d889d9220f71cf
