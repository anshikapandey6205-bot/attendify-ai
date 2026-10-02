from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    Response,
    jsonify,
    send_file
)

import cv2
import os
import datetime
import pandas as pd

from database import (
    init_db,
    get_db,
    verify_admin
)

from face_recognition_service import (
    save_face_samples,
    train_recognizer,
    recognize_face,
    detect_faces
)


app = Flask(__name__)

app.secret_key = "change-this-secret-key"

init_db()

camera = cv2.VideoCapture(0)


# --------------------------------------------------
# LOGIN
# --------------------------------------------------

@app.route("/")
def index():

    if "admin" in session:
        return redirect(url_for("dashboard"))

    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        if verify_admin(username, password):

            session["admin"] = username

            return redirect(
                url_for("dashboard")
            )

        return render_template(
            "login.html",
            error="Invalid username or password"
        )

    return render_template("login.html")


@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        url_for("login")
    )


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

@app.route("/dashboard")
def dashboard():

    if "admin" not in session:
        return redirect(url_for("login"))

    conn = get_db()

    total_students = conn.execute(
        "SELECT COUNT(*) FROM students"
    ).fetchone()[0]

    today = datetime.date.today().isoformat()

    present_today = conn.execute(
        """
        SELECT COUNT(*)
        FROM attendance
        WHERE date = ?
        AND status = 'Present'
        """,
        (today,)
    ).fetchone()[0]

    conn.close()

    absent_today = total_students - present_today

    percentage = 0

    if total_students > 0:
        percentage = round(
            (present_today / total_students) * 100,
            2
        )

    return render_template(
        "dashboard.html",
        total_students=total_students,
        present_today=present_today,
        absent_today=absent_today,
        percentage=percentage
    )


# --------------------------------------------------
# STUDENTS
# --------------------------------------------------

@app.route("/students")
def students():

    if "admin" not in session:
        return redirect(url_for("login"))

    conn = get_db()

    students = conn.execute(
        "SELECT * FROM students ORDER BY id DESC"
    ).fetchall()

    conn.close()

    return render_template(
        "students.html",
        students=students
    )


@app.route("/students/add", methods=["GET", "POST"])
def add_student():

    if "admin" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":

        name = request.form["name"]
        roll_number = request.form["roll_number"]
        course = request.form["course"]
        semester = request.form["semester"]
        section = request.form["section"]
        email = request.form["email"]
        phone = request.form["phone"]

        conn = get_db()

        try:

            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO students
                (name, roll_number, course, semester,
                 section, email, phone)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    name,
                    roll_number,
                    course,
                    semester,
                    section,
                    email,
                    phone
                )
            )

            conn.commit()

            student_id = cursor.lastrowid

            conn.close()

            return redirect(
                url_for(
                    "capture_face",
                    student_id=student_id
                )
            )

        except Exception:

            conn.close()

            return render_template(
                "add_student.html",
                error="Roll number already exists."
            )

    return render_template(
        "add_student.html"
    )


# --------------------------------------------------
# FACE CAPTURE
# --------------------------------------------------

@app.route("/capture/<int:student_id>")
def capture_face(student_id):

    if "admin" not in session:
        return redirect(url_for("login"))

    return render_template(
        "attendance.html",
        enrollment=True,
        student_id=student_id
    )


@app.route("/capture-face/<int:student_id>",
           methods=["POST"])
def capture_face_api(student_id):

    if "admin" not in session:
        return jsonify({
            "success": False
        })

    frame_data = request.files.get("frame")

    if not frame_data:
        return jsonify({
            "success": False,
            "message": "No image received"
        })

    temp_path = "uploads/temp.jpg"

    frame_data.save(temp_path)

    frame = cv2.imread(temp_path)

    if frame is None:
        return jsonify({
            "success": False,
            "message": "Invalid image"
        })

    success = save_face_samples(
        student_id,
        frame
    )

    if success:

        conn = get_db()

        conn.execute(
            """
            UPDATE students
            SET face_id = ?
            WHERE id = ?
            """,
            (
                student_id,
                student_id
            )
        )

        conn.commit()
        conn.close()

        return jsonify({
            "success": True,
            "message": "Face registered successfully"
        })

    return jsonify({
        "success": False,
        "message": "No face detected"
    })


# --------------------------------------------------
# CAMERA
# --------------------------------------------------

def generate_frames():

    while True:

        success, frame = camera.read()

        if not success:
            break

        ret, buffer = cv2.imencode(
            ".jpg",
            frame
        )

        frame = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame +
            b"\r\n"
        )


@app.route("/video_feed")
def video_feed():

    if "admin" not in session:
        return redirect(url_for("login"))

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# --------------------------------------------------
# ATTENDANCE
# --------------------------------------------------

@app.route("/attendance")
def attendance():

    if "admin" not in session:
        return redirect(url_for("login"))

    conn = get_db()

    records = conn.execute(
        """
        SELECT
            attendance.id,
            students.name,
            students.roll_number,
            students.course,
            attendance.date,
            attendance.time,
            attendance.status
        FROM attendance
        JOIN students
        ON students.id = attendance.student_id
        ORDER BY attendance.date DESC,
                 attendance.time DESC
        """
    ).fetchall()

    conn.close()

    return render_template(
        "attendance.html",
        records=records,
        enrollment=False
    )


@app.route("/recognize", methods=["POST"])
def recognize():

    if "admin" not in session:
        return jsonify({
            "success": False
        })

    frame_data = request.files.get("frame")

    if not frame_data:
        return jsonify({
            "success": False
        })

    temp_path = "uploads/recognition.jpg"

    frame_data.save(temp_path)

    frame = cv2.imread(temp_path)

    if frame is None:
        return jsonify({
            "success": False,
            "message": "Invalid frame"
        })

    recognizer = train_recognizer()

    if recognizer is None:
        return jsonify({
            "success": False,
            "message": "No registered faces"
        })

    student_id, confidence, box = recognize_face(
        frame,
        recognizer
    )

    if student_id is None:

        return jsonify({
            "success": False,
            "message": "Unknown face"
        })

    conn = get_db()

    student = conn.execute(
        "SELECT * FROM students WHERE id = ?",
        (student_id,)
    ).fetchone()

    if not student:

        conn.close()

        return jsonify({
            "success": False,
            "message": "Student not found"
        })

    today = datetime.date.today().isoformat()
    current_time = datetime.datetime.now().strftime(
        "%H:%M:%S"
    )

    existing = conn.execute(
        """
        SELECT * FROM attendance
        WHERE student_id = ?
        AND date = ?
        """,
        (
            student_id,
            today
        )
    ).fetchone()

    if existing:

        conn.close()

        return jsonify({
            "success": True,
            "already_marked": True,
            "name": student["name"],
            "message": "Attendance already marked today"
        })

    conn.execute(
        """
        INSERT INTO attendance
        (student_id, date, time, status)
        VALUES (?, ?, ?, ?)
        """,
        (
            student_id,
            today,
            current_time,
            "Present"
        )
    )

    conn.commit()
    conn.close()

    return jsonify({
        "success": True,
        "already_marked": False,
        "name": student["name"],
        "roll_number": student["roll_number"],
        "time": current_time,
        "message": "Attendance marked successfully"
    })


# --------------------------------------------------
# REPORTS
# --------------------------------------------------

@app.route("/reports")
def reports():

    if "admin" not in session:
        return redirect(url_for("login"))

    conn = get_db()

    records = conn.execute(
        """
        SELECT
            students.name,
            students.roll_number,
            students.course,
            attendance.date,
            attendance.time,
            attendance.status
        FROM attendance
        JOIN students
        ON students.id = attendance.student_id
        ORDER BY attendance.date DESC
        """
    ).fetchall()

    conn.close()

    return render_template(
        "reports.html",
        records=records
    )


@app.route("/export")
def export():

    if "admin" not in session:
        return redirect(url_for("login"))

    conn = get_db()

    rows = conn.execute(
        """
        SELECT
            students.name,
            students.roll_number,
            students.course,
            attendance.date,
            attendance.time,
            attendance.status
        FROM attendance
        JOIN students
        ON students.id = attendance.student_id
        ORDER BY attendance.date DESC
        """
    ).fetchall()

    conn.close()

    data = [dict(row) for row in rows]

    df = pd.DataFrame(data)

    path = "attendance_report.xlsx"

    df.to_excel(
        path,
        index=False
    )

    return send_file(
        path,
        as_attachment=True
    )


if __name__ == "__main__":
    init_db()
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )
    