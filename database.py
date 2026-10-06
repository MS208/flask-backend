import sqlite3
import hashlib

DB_NAME = 'smart_messaging.db'

def get_db():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            full_name TEXT NOT NULL,
            password_hash TEXT NOT NULL
        )
    ''')
    
    # User settings table with custom active hours columns
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            active_hours_enabled INTEGER DEFAULT 1,
            work_start_time TEXT DEFAULT '09:00',
            work_end_time TEXT DEFAULT '17:00',
            allow_auto_reply INTEGER DEFAULT 0,
            dnd_bypass_enabled INTEGER DEFAULT 1,
            default_persona TEXT DEFAULT 'casual'
        )
    ''')
    
    # Stored style samples for TF-IDF adaptation
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_style_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            persona TEXT NOT NULL,
            sample_reply TEXT NOT NULL
        )
    ''')

    # Tracked tasks / reminders table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reminders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            task_title TEXT NOT NULL,
            deadline_text TEXT,
            buffer_lead TEXT
        )
    ''')

    # Meetings table for extracted meeting links
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS meetings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            platform TEXT NOT NULL,
            meeting_url TEXT NOT NULL,
            scheduled_time TEXT
        )
    ''')

    # Analysis logs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            message TEXT NOT NULL,
            sentiment TEXT,
            urgency TEXT,
            priority TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, email, full_name, password):
    conn = get_db()
    cursor = conn.cursor()
    try:
        pw_hash = hash_password(password)
        cursor.execute(
            "INSERT INTO users (username, email, full_name, password_hash) VALUES (?, ?, ?, ?)",
            (username, email, full_name, pw_hash)
        )
        cursor.execute(
            "INSERT INTO user_settings (username, work_start_time, work_end_time) VALUES (?, '09:00', '17:00')",
            (username,)
        )
        conn.commit()
        return True, "Registration successful."
    except sqlite3.IntegrityError:
        return False, "Username or email already exists."
    finally:
        conn.close()

def verify_user(identifier, password):
    conn = get_db()
    cursor = conn.cursor()
    pw_hash = hash_password(password)
    cursor.execute(
        "SELECT * FROM users WHERE (username = ? OR email = ?) AND password_hash = ?",
        (identifier, identifier, pw_hash)
    )
    user = cursor.fetchone()
    conn.close()
    return user

def log_analysis(username, message, sentiment, urgency, priority):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO analysis_logs (username, message, sentiment, urgency, priority) VALUES (?, ?, ?, ?, ?)",
        (username, message, sentiment, urgency, priority)
    )
    conn.commit()
    conn.close()