import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

DATABASE = "attendify.db"


def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    cursor = conn.cursor()

    # Admin table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS admin (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)

    # Students table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            roll_number TEXT UNIQUE NOT NULL,
            course TEXT,
            semester TEXT,
            section TEXT,
            email TEXT,
            phone TEXT,
            face_id INTEGER
        )
    """)

    # Attendance table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            status TEXT NOT NULL,
            FOREIGN KEY(student_id) REFERENCES students(id),
            UNIQUE(student_id, date)
        )
    """)

    # Create default admin account
    existing = cursor.execute(
        "SELECT * FROM admin WHERE username = ?",
        ("admin",)
    ).fetchone()

    if not existing:
        password = generate_password_hash("admin123")

        cursor.execute(
            "INSERT INTO admin (username, password) VALUES (?, ?)",
            ("admin", password)
        )

    conn.commit()
    conn.close()


def verify_admin(username, password):
    conn = get_db()

    admin = conn.execute(
        "SELECT * FROM admin WHERE username = ?",
        (username,)
    ).fetchone()

    conn.close()

    if admin and check_password_hash(admin["password"], password):
        return True

    return False