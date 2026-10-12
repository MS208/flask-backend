import os
import sqlite3
import hashlib
import time
from datetime import datetime
from urllib.parse import urlparse
from temporal_engine import calculate_reminder_timestamp

DATABASE_URL = os.environ.get('DATABASE_URL')

def is_postgres():
    return bool(DATABASE_URL)

def get_db_connection():
    if is_postgres():
        import pg8000.dbapi
        url = DATABASE_URL
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        parsed = urlparse(url)
        conn = pg8000.dbapi.connect(
            user=parsed.username,
            password=parsed.password,
            host=parsed.hostname,
            port=parsed.port or 5432,
            database=parsed.path.lstrip('/')
        )
        return conn
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(base_dir, 'smart_messaging.db')
        conn = sqlite3.connect(db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

def get_db():
    return get_db_connection()

def query_db(query, args=(), one=False, commit=False):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        last_id = None
        
        if is_postgres():
            pg_query = query.replace('?', '%s')
            is_insert = pg_query.strip().upper().startswith('INSERT INTO')
            if is_insert and 'RETURNING' not in pg_query.upper():
                pg_query = pg_query.rstrip(';') + ' RETURNING id;'
                cursor.execute(pg_query, args)
                ret = cursor.fetchone()
                if ret:
                    last_id = ret[0]
            else:
                cursor.execute(pg_query, args)
            
            if commit:
                conn.commit()
                
            if cursor.description:
                cols = [d[0] for d in cursor.description]
                rows = [dict(zip(cols, r)) for r in cursor.fetchall()]
            else:
                rows = []
        else:
            cursor.execute(query, args)
            if commit:
                conn.commit()
                last_id = cursor.lastrowid
            
            if cursor.description:
                rows = [dict(r) for r in cursor.fetchall()]
            else:
                rows = []
        
        if commit and last_id is not None:
            return last_id
        if one:
            return rows[0] if rows else None
        return rows
    finally:
        try:
            conn.close()
        except Exception:
            pass

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    postgres = is_postgres()
    
    id_type = "SERIAL PRIMARY KEY" if postgres else "INTEGER PRIMARY KEY AUTOINCREMENT"
    
    # 1. Users table
    query = f'''
        CREATE TABLE IF NOT EXISTS users (
            id {id_type},
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            full_name TEXT NOT NULL,
            password_hash TEXT NOT NULL
        );
    '''
    cursor.execute(query)

    # 2. User settings table
    query = f'''
        CREATE TABLE IF NOT EXISTS user_settings (
            id {id_type},
            username TEXT UNIQUE NOT NULL,
            active_hours_enabled INTEGER DEFAULT 1,
            work_start_time TEXT DEFAULT '09:00',
            work_end_time TEXT DEFAULT '17:00',
            allow_auto_reply INTEGER DEFAULT 0,
            dnd_bypass_enabled INTEGER DEFAULT 1,
            default_persona TEXT DEFAULT 'casual',
            custom_topics TEXT DEFAULT '',
            handling_stage TEXT DEFAULT 'immediate',
            spam_handling TEXT DEFAULT 'delete'
        );
    '''
    cursor.execute(query)

    # 3. User style samples table
    query = f'''
        CREATE TABLE IF NOT EXISTS user_style_samples (
            id {id_type},
            username TEXT NOT NULL,
            persona TEXT NOT NULL,
            sample_reply TEXT NOT NULL
        );
    '''
    cursor.execute(query)

    # 4. Reminders / Tasks table
    query = f'''
        CREATE TABLE IF NOT EXISTS reminders (
            id {id_type},
            username TEXT NOT NULL,
            task_title TEXT NOT NULL,
            description TEXT DEFAULT '',
            deadline_text TEXT,
            buffer_lead TEXT,
            status TEXT DEFAULT 'Pending',
            due_date TEXT,
            due_time TEXT,
            reminder_offset INTEGER DEFAULT 30,
            reminder_timestamp BIGINT,
            source_message TEXT,
            sender TEXT,
            linked_meeting_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    '''
    cursor.execute(query)

    # 5. Meetings table
    query = f'''
        CREATE TABLE IF NOT EXISTS meetings (
            id {id_type},
            username TEXT NOT NULL,
            platform TEXT NOT NULL,
            meeting_url TEXT NOT NULL,
            scheduled_time TEXT,
            meeting_date TEXT,
            start_time TEXT,
            reminder_offset INTEGER DEFAULT 30,
            reminder_timestamp BIGINT,
            status TEXT DEFAULT 'Scheduled',
            title TEXT DEFAULT '',
            source_message TEXT,
            sender TEXT,
            linked_task_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    '''
    cursor.execute(query)

    # 6. Analysis logs table
    query = f'''
        CREATE TABLE IF NOT EXISTS analysis_logs (
            id {id_type},
            username TEXT NOT NULL,
            message TEXT NOT NULL,
            sentiment TEXT,
            urgency TEXT,
            priority TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    '''
    cursor.execute(query)

    # Safe Schema Migrations for existing databases
    if postgres:
        conn.commit()
        cursor.execute("ALTER TABLE reminders ADD COLUMN IF NOT EXISTS description TEXT DEFAULT '';")
        cursor.execute("ALTER TABLE reminders ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'Pending';")
        cursor.execute("ALTER TABLE reminders ADD COLUMN IF NOT EXISTS due_date TEXT;")
        cursor.execute("ALTER TABLE reminders ADD COLUMN IF NOT EXISTS due_time TEXT;")
        cursor.execute("ALTER TABLE reminders ADD COLUMN IF NOT EXISTS reminder_offset INTEGER DEFAULT 30;")
        cursor.execute("ALTER TABLE reminders ADD COLUMN IF NOT EXISTS reminder_timestamp BIGINT;")
        cursor.execute("ALTER TABLE reminders ADD COLUMN IF NOT EXISTS source_message TEXT;")
        cursor.execute("ALTER TABLE reminders ADD COLUMN IF NOT EXISTS sender TEXT;")
        cursor.execute("ALTER TABLE reminders ADD COLUMN IF NOT EXISTS linked_meeting_id INTEGER;")
        cursor.execute("ALTER TABLE reminders ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;")
        
        cursor.execute("ALTER TABLE meetings ADD COLUMN IF NOT EXISTS title TEXT DEFAULT '';")
        cursor.execute("ALTER TABLE meetings ADD COLUMN IF NOT EXISTS meeting_date TEXT;")
        cursor.execute("ALTER TABLE meetings ADD COLUMN IF NOT EXISTS start_time TEXT;")
        cursor.execute("ALTER TABLE meetings ADD COLUMN IF NOT EXISTS reminder_offset INTEGER DEFAULT 30;")
        cursor.execute("ALTER TABLE meetings ADD COLUMN IF NOT EXISTS reminder_timestamp BIGINT;")
        cursor.execute("ALTER TABLE meetings ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'Scheduled';")
        cursor.execute("ALTER TABLE meetings ADD COLUMN IF NOT EXISTS source_message TEXT;")
        cursor.execute("ALTER TABLE meetings ADD COLUMN IF NOT EXISTS sender TEXT;")
        cursor.execute("ALTER TABLE meetings ADD COLUMN IF NOT EXISTS linked_task_id INTEGER;")
        cursor.execute("ALTER TABLE meetings ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;")
        cursor.execute("ALTER TABLE user_settings ADD COLUMN IF NOT EXISTS work_mode_enabled INTEGER DEFAULT 1;")
        cursor.execute("ALTER TABLE user_settings ADD COLUMN IF NOT EXISTS custom_topics TEXT DEFAULT '';")
        cursor.execute("ALTER TABLE user_settings ADD COLUMN IF NOT EXISTS handling_stage TEXT DEFAULT 'immediate';")
        cursor.execute("ALTER TABLE user_settings ADD COLUMN IF NOT EXISTS spam_handling TEXT DEFAULT 'delete';")
        conn.commit()
    else:
        cursor.execute("PRAGMA table_info(user_settings)")
        user_settings_cols = [col[1] for col in cursor.fetchall()]
        if 'work_mode_enabled' not in user_settings_cols:
            cursor.execute("ALTER TABLE user_settings ADD COLUMN work_mode_enabled INTEGER DEFAULT 1")
        if 'custom_topics' not in user_settings_cols:
            cursor.execute("ALTER TABLE user_settings ADD COLUMN custom_topics TEXT DEFAULT ''")
        if 'handling_stage' not in user_settings_cols:
            cursor.execute("ALTER TABLE user_settings ADD COLUMN handling_stage TEXT DEFAULT 'immediate'")
        if 'spam_handling' not in user_settings_cols:
            cursor.execute("ALTER TABLE user_settings ADD COLUMN spam_handling TEXT DEFAULT 'delete'")
        cursor.execute("PRAGMA table_info(reminders)")
        reminder_cols = [col[1] for col in cursor.fetchall()]
        if 'description' not in reminder_cols:
            cursor.execute("ALTER TABLE reminders ADD COLUMN description TEXT DEFAULT ''")
        if 'status' not in reminder_cols:
            cursor.execute("ALTER TABLE reminders ADD COLUMN status TEXT DEFAULT 'Pending'")
        if 'due_date' not in reminder_cols:
            cursor.execute("ALTER TABLE reminders ADD COLUMN due_date TEXT")
        if 'due_time' not in reminder_cols:
            cursor.execute("ALTER TABLE reminders ADD COLUMN due_time TEXT")
        if 'reminder_offset' not in reminder_cols:
            cursor.execute("ALTER TABLE reminders ADD COLUMN reminder_offset INTEGER DEFAULT 30")
        if 'reminder_timestamp' not in reminder_cols:
            cursor.execute("ALTER TABLE reminders ADD COLUMN reminder_timestamp BIGINT")
        if 'source_message' not in reminder_cols:
            cursor.execute("ALTER TABLE reminders ADD COLUMN source_message TEXT")
        if 'sender' not in reminder_cols:
            cursor.execute("ALTER TABLE reminders ADD COLUMN sender TEXT")
        if 'linked_meeting_id' not in reminder_cols:
            cursor.execute("ALTER TABLE reminders ADD COLUMN linked_meeting_id INTEGER")
        if 'created_at' not in reminder_cols:
            cursor.execute("ALTER TABLE reminders ADD COLUMN created_at DATETIME DEFAULT CURRENT_TIMESTAMP")

        cursor.execute("PRAGMA table_info(meetings)")
        meeting_cols = [col[1] for col in cursor.fetchall()]
        if 'title' not in meeting_cols:
            cursor.execute("ALTER TABLE meetings ADD COLUMN title TEXT DEFAULT ''")
        if 'meeting_date' not in meeting_cols:
            cursor.execute("ALTER TABLE meetings ADD COLUMN meeting_date TEXT")
        if 'start_time' not in meeting_cols:
            cursor.execute("ALTER TABLE meetings ADD COLUMN start_time TEXT")
        if 'reminder_offset' not in meeting_cols:
            cursor.execute("ALTER TABLE meetings ADD COLUMN reminder_offset INTEGER DEFAULT 30")
        if 'reminder_timestamp' not in meeting_cols:
            cursor.execute("ALTER TABLE meetings ADD COLUMN reminder_timestamp BIGINT")
        if 'status' not in meeting_cols:
            cursor.execute("ALTER TABLE meetings ADD COLUMN status TEXT DEFAULT 'Scheduled'")
        if 'source_message' not in meeting_cols:
            cursor.execute("ALTER TABLE meetings ADD COLUMN source_message TEXT")
        if 'sender' not in meeting_cols:
            cursor.execute("ALTER TABLE meetings ADD COLUMN sender TEXT")
        if 'linked_task_id' not in meeting_cols:
            cursor.execute("ALTER TABLE meetings ADD COLUMN linked_task_id INTEGER")
        if 'created_at' not in meeting_cols:
            cursor.execute("ALTER TABLE meetings ADD COLUMN created_at DATETIME DEFAULT CURRENT_TIMESTAMP")

        conn.commit()

    # Pre-populate default admin account if none exists
    cursor.execute("SELECT COUNT(*) FROM users")
    count = cursor.fetchone()[0]
    if count == 0:
        default_pwd = hashlib.sha256("Admin@123".encode()).hexdigest()
        if postgres:
            cursor.execute("""
                INSERT INTO users (username, email, full_name, password_hash)
                VALUES (%s, %s, %s, %s)
            """, ('admin', 'admin@gateway.com', 'System Administrator', default_pwd))
            cursor.execute("""
                INSERT INTO user_settings (username, active_hours_enabled, work_start_time, work_end_time, allow_auto_reply, dnd_bypass_enabled, default_persona)
                VALUES (%s, 1, '09:00', '17:00', 0, 1, 'casual')
            """, ('admin',))
        else:
            cursor.execute("""
                INSERT INTO users (username, email, full_name, password_hash)
                VALUES (?, ?, ?, ?)
            """, ('admin', 'admin@gateway.com', 'System Administrator', default_pwd))
            cursor.execute("""
                INSERT INTO user_settings (username, active_hours_enabled, work_start_time, work_end_time, allow_auto_reply, dnd_bypass_enabled, default_persona)
                VALUES (?, 1, '09:00', '17:00', 0, 1, 'casual')
            """, ('admin',))
        conn.commit()

    conn.close()

# --- User Auth Operations ---

def register_user(username, email, full_name, password):
    username = username.strip().lower()
    email = email.strip().lower()
    full_name = full_name.strip()
    
    if not username or not email or not password or not full_name:
        return False, "All fields are required."
        
    pwd_hash = hashlib.sha256(password.encode()).hexdigest()
    
    try:
        query_db(
            "INSERT INTO users (username, email, full_name, password_hash) VALUES (?, ?, ?, ?)",
            (username, email, full_name, pwd_hash),
            commit=True
        )
        query_db(
            "INSERT INTO user_settings (username, active_hours_enabled, work_start_time, work_end_time, allow_auto_reply, dnd_bypass_enabled, default_persona) VALUES (?, 1, '09:00', '17:00', 0, 1, 'casual')",
            (username,),
            commit=True
        )
        return True, "Registration successful! You may now sign in."
    except Exception as e:
        err_msg = str(e).lower()
        if "unique constraint" in err_msg or "duplicate key" in err_msg:
            return False, "Username or email is already registered."
        return False, f"Registration failed: {str(e)}"

def verify_user(identifier, password):
    ident = identifier.strip().lower()
    pwd_hash = hashlib.sha256(password.encode()).hexdigest()
    user = query_db(
        "SELECT * FROM users WHERE (LOWER(username) = ? OR LOWER(email) = ?) AND password_hash = ?",
        (ident, ident, pwd_hash),
        one=True
    )
    return user

def get_user_profile(username):
    uname = (username or 'admin').strip().lower()
    user = query_db("SELECT id, username, email, full_name FROM users WHERE LOWER(username) = ?", (uname,), one=True)
    if not user:
        return {
            'username': uname,
            'email': f"{uname}@gateway.com",
            'full_name': uname.capitalize(),
            'role': 'System Administrator'
        }
    return {
        'id': user['id'],
        'username': user['username'],
        'email': user['email'],
        'full_name': user['full_name'],
        'role': 'System Administrator'
    }

def update_user_profile(username, full_name, email):
    uname = (username or 'admin').strip().lower()
    full_name = full_name.strip()
    email = email.strip().lower()
    if not full_name or not email:
        return False, "Full name and email cannot be empty."

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        if is_postgres():
            cursor.execute("""
                UPDATE users SET full_name = %s, email = %s WHERE LOWER(username) = %s
            """, (full_name, email, uname))
        else:
            cursor.execute("""
                UPDATE users SET full_name = ?, email = ? WHERE LOWER(username) = ?
            """, (full_name, email, uname))
        conn.commit()
        return True, "Profile updated successfully."
    except Exception as e:
        err_msg = str(e).lower()
        if "unique constraint" in err_msg or "duplicate key" in err_msg:
            return False, "Email address is already in use by another account."
        return False, f"Failed to update profile: {str(e)}"
    finally:
        try:
            conn.close()
        except Exception:
            pass

def log_analysis(username, message, sentiment, urgency, priority):
    username = (username or 'admin').strip().lower()
    # Deduplicate recent logs to prevent duplicate alerts
    existing = query_db("""
        SELECT id FROM analysis_logs 
        WHERE LOWER(username) = ? AND message = ?
        ORDER BY id DESC LIMIT 1
    """, (username, message), one=True)
    if existing:
        return existing['id']

    row_id = query_db(
        "INSERT INTO analysis_logs (username, message, sentiment, urgency, priority) VALUES (?, ?, ?, ?, ?)",
        (username, message, sentiment, urgency, priority),
        commit=True
    )
    return row_id

def get_p1_alert_count(username):
    uname = (username or 'admin').strip().lower()
    row = query_db("""
        SELECT COUNT(*) as count 
        FROM analysis_logs 
        WHERE LOWER(username) = ? 
          AND (LOWER(priority) LIKE '%p1%' OR LOWER(priority) LIKE '%urgent%' OR LOWER(urgency) = 'high')
    """, (uname,), one=True)
    return row['count'] if row else 0

def get_total_message_count(username):
    uname = (username or 'admin').strip().lower()
    row = query_db("""
        SELECT COUNT(*) as count 
        FROM analysis_logs 
        WHERE LOWER(username) = ?
    """, (uname,), one=True)
    return row['count'] if row else 0

def get_quarantined_messages(username):
    return []

# --- Meeting Database Operations ---

def compute_item_status(status_val, date_val, time_val, default_active="Upcoming"):
    st = (status_val or default_active).strip()
    if st in ("Completed", "Cancelled", "Dismissed"):
        return st
    if not date_val:
        return "Upcoming"
    try:
        t_str = time_val.strip() if time_val and time_val.strip() else "23:59:59"
        dt_str = f"{date_val.strip()} {t_str[:5]}"
        dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
        if dt < datetime.now():
            return "Overdue"
        return "Upcoming"
    except Exception:
        return "Upcoming"

def get_meetings_for_user(username):
    uname = (username or 'admin').strip().lower()
    rows = query_db("""
        SELECT id, platform, meeting_url, scheduled_time, meeting_date, start_time, 
               reminder_offset, reminder_timestamp, status, title, source_message, sender, created_at, linked_task_id 
        FROM meetings 
        WHERE LOWER(username) = ? 
        ORDER BY id DESC
    """, (uname,))
    return [
        {
            'id': row['id'],
            'platform': row.get('platform') or 'Meeting',
            'meeting_url': row['meeting_url'],
            'scheduled_time': row.get('scheduled_time') or 'Unknown',
            'meeting_date': row.get('meeting_date') or '',
            'start_time': row.get('start_time') or '',
            'reminder_offset': row.get('reminder_offset') if row.get('reminder_offset') is not None else 30,
            'reminder_timestamp': row.get('reminder_timestamp'),
            'status': row.get('status') or 'Scheduled',
            'computed_status': compute_item_status(row.get('status'), row.get('meeting_date'), row.get('start_time'), 'Upcoming'),
            'title': row.get('title') or f"{row.get('platform', 'Meeting')} Meeting",
            'source_message': row.get('source_message') or '',
            'sender': row.get('sender') or '',
            'created_at': str(row.get('created_at') or ''),
            'linked_task_id': row.get('linked_task_id')
        } for row in rows
    ]

def get_meeting_by_url(username, meeting_url):
    uname = (username or 'admin').strip().lower()
    url = meeting_url.strip()
    row = query_db(
        "SELECT * FROM meetings WHERE LOWER(username) = ? AND LOWER(meeting_url) = LOWER(?) AND status != 'Cancelled'",
        (uname, url),
        one=True
    )
    return row

def add_meeting(username, platform, meeting_url, scheduled_time="Unknown", title="", source_message="", sender="", 
                meeting_date="", start_time="", reminder_offset=30, reminder_timestamp=None, status="Scheduled", return_tuple=False):
    uname = (username or 'admin').strip().lower()
    url = meeting_url.strip()
    platform = platform.strip() or 'Meeting'
    title = title.strip() or f"{platform} Meeting"
    scheduled_time = (scheduled_time or 'Unknown').strip()
    meeting_date = (meeting_date or '').strip()
    start_time = (start_time or '').strip()
    status = (status or 'Scheduled').strip()

    try:
        reminder_offset = int(reminder_offset) if reminder_offset is not None else 30
    except Exception:
        reminder_offset = 30

    if reminder_timestamp is None and reminder_offset > 0:
        reminder_timestamp = calculate_reminder_timestamp(meeting_date or scheduled_time, start_time, reminder_offset)

    # Deduplication check
    existing = get_meeting_by_url(uname, url)
    if existing:
        if (existing.get('scheduled_time') in ('Unknown', '', None) and scheduled_time != 'Unknown') or meeting_date or start_time:
            update_meeting(
                meeting_id=existing['id'],
                username=uname,
                platform=platform,
                meeting_url=url,
                scheduled_time=scheduled_time if scheduled_time != 'Unknown' else existing.get('scheduled_time', 'Unknown'),
                title=title or existing.get('title'),
                meeting_date=meeting_date or existing.get('meeting_date', ''),
                start_time=start_time or existing.get('start_time', ''),
                reminder_offset=reminder_offset,
                reminder_timestamp=reminder_timestamp or existing.get('reminder_timestamp'),
                status=status
            )
        existing_id = int(existing['id'])
        return (existing_id, False) if return_tuple else existing_id

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        if is_postgres():
            cursor.execute("""
                INSERT INTO meetings (username, platform, meeting_url, scheduled_time, meeting_date, start_time, 
                                      reminder_offset, reminder_timestamp, status, title, source_message, sender)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (uname, platform, url, scheduled_time, meeting_date, start_time, reminder_offset, reminder_timestamp, status, title, source_message, sender))
            row = cursor.fetchone()
            new_id = int(row[0]) if row else None
        else:
            cursor.execute("""
                INSERT INTO meetings (username, platform, meeting_url, scheduled_time, meeting_date, start_time, 
                                      reminder_offset, reminder_timestamp, status, title, source_message, sender)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (uname, platform, url, scheduled_time, meeting_date, start_time, reminder_offset, reminder_timestamp, status, title, source_message, sender))
            new_id = int(cursor.lastrowid) if cursor.lastrowid else None

        # Auto-generate linked Upcoming Task
        if new_id:
            task_deadline = scheduled_time if scheduled_time != 'Unknown' else f"{meeting_date} {start_time}".strip()
            task_desc = f"Meeting Link: {url}"
            task_status = 'Completed' if status.lower() == 'completed' else 'Pending'
            buffer_lead = f"{reminder_offset}m prior" if reminder_offset > 0 else "No reminder"

            if is_postgres():
                cursor.execute("""
                    INSERT INTO reminders (username, task_title, description, deadline_text, buffer_lead, status, 
                                           due_date, due_time, reminder_offset, reminder_timestamp, source_message, sender, linked_meeting_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (uname, title, task_desc, task_deadline, buffer_lead, task_status, meeting_date, start_time, reminder_offset, reminder_timestamp, source_message, sender, new_id))
                t_row = cursor.fetchone()
                task_id = int(t_row[0]) if t_row else None
                if task_id:
                    cursor.execute("UPDATE meetings SET linked_task_id = %s WHERE id = %s", (task_id, new_id))
            else:
                cursor.execute("""
                    INSERT INTO reminders (username, task_title, description, deadline_text, buffer_lead, status, 
                                           due_date, due_time, reminder_offset, reminder_timestamp, source_message, sender, linked_meeting_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (uname, title, task_desc, task_deadline, buffer_lead, task_status, meeting_date, start_time, reminder_offset, reminder_timestamp, source_message, sender, new_id))
                task_id = int(cursor.lastrowid) if cursor.lastrowid else None
                if task_id:
                    cursor.execute("UPDATE meetings SET linked_task_id = ? WHERE id = ?", (task_id, new_id))

        conn.commit()
        return (new_id, True) if return_tuple else new_id
    finally:
        try:
            conn.close()
        except Exception:
            pass

def update_meeting(meeting_id, username, platform, meeting_url, scheduled_time="Unknown", title="", 
                   meeting_date="", start_time="", reminder_offset=30, reminder_timestamp=None, status="Scheduled"):
    try:
        m_id = int(meeting_id)
    except (ValueError, TypeError):
        return False

    uname = (username or '').strip().lower() if username else None
    url = meeting_url.strip()
    platform = platform.strip() or 'Meeting'
    title = title.strip() or f"{platform} Meeting"
    scheduled_time = (scheduled_time or 'Unknown').strip()
    meeting_date = (meeting_date or '').strip()
    start_time = (start_time or '').strip()
    status = (status or 'Scheduled').strip()

    try:
        reminder_offset = int(reminder_offset) if reminder_offset is not None else 30
    except Exception:
        reminder_offset = 30

    if reminder_timestamp is None and reminder_offset > 0:
        reminder_timestamp = calculate_reminder_timestamp(meeting_date or scheduled_time, start_time, reminder_offset)

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        if is_postgres():
            cursor.execute("SELECT id, username, linked_task_id FROM meetings WHERE id = %s", (m_id,))
        else:
            cursor.execute("SELECT id, username, linked_task_id FROM meetings WHERE id = ?", (m_id,))
        row = cursor.fetchone()
        if not row:
            return False

        row_user = (row[1] if isinstance(row, tuple) else row['username']) or ''
        row_user = row_user.strip().lower()
        authorized = (not uname or uname == 'admin' or row_user == uname or not row_user or row_user == 'admin')
        if not authorized:
            return False

        if is_postgres():
            cursor.execute("""
                UPDATE meetings 
                SET platform = %s, meeting_url = %s, scheduled_time = %s, title = %s,
                    meeting_date = %s, start_time = %s, reminder_offset = %s, reminder_timestamp = %s, status = %s
                WHERE id = %s
            """, (platform, url, scheduled_time, title, meeting_date, start_time, reminder_offset, reminder_timestamp, status, m_id))
        else:
            cursor.execute("""
                UPDATE meetings 
                SET platform = ?, meeting_url = ?, scheduled_time = ?, title = ?,
                    meeting_date = ?, start_time = ?, reminder_offset = ?, reminder_timestamp = ?, status = ?
                WHERE id = ?
            """, (platform, url, scheduled_time, title, meeting_date, start_time, reminder_offset, reminder_timestamp, status, m_id))

        # Synchronize linked Upcoming Task
        task_status = 'Completed' if status.lower() == 'completed' else 'Pending'
        task_deadline = scheduled_time if scheduled_time != 'Unknown' else f"{meeting_date} {start_time}".strip()
        task_desc = f"Meeting Link: {url}"
        buffer_lead = f"{reminder_offset}m prior" if reminder_offset > 0 else "No reminder"

        if is_postgres():
            cursor.execute("SELECT id FROM reminders WHERE linked_meeting_id = %s", (m_id,))
            t_row = cursor.fetchone()
            if t_row:
                cursor.execute("""
                    UPDATE reminders
                    SET task_title = %s, description = %s, deadline_text = %s, buffer_lead = %s, status = %s,
                        due_date = %s, due_time = %s, reminder_offset = %s, reminder_timestamp = %s
                    WHERE linked_meeting_id = %s
                """, (title, task_desc, task_deadline, buffer_lead, task_status, meeting_date, start_time, reminder_offset, reminder_timestamp, m_id))
            else:
                cursor.execute("""
                    INSERT INTO reminders (username, task_title, description, deadline_text, buffer_lead, status, 
                                           due_date, due_time, reminder_offset, reminder_timestamp, linked_meeting_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (row_user or uname or 'admin', title, task_desc, task_deadline, buffer_lead, task_status, meeting_date, start_time, reminder_offset, reminder_timestamp, m_id))
                new_t = cursor.fetchone()
                if new_t:
                    cursor.execute("UPDATE meetings SET linked_task_id = %s WHERE id = %s", (new_t[0], m_id))
        else:
            cursor.execute("SELECT id FROM reminders WHERE linked_meeting_id = ?", (m_id,))
            t_row = cursor.fetchone()
            if t_row:
                cursor.execute("""
                    UPDATE reminders
                    SET task_title = ?, description = ?, deadline_text = ?, buffer_lead = ?, status = ?, 
                        due_date = ?, due_time = ?, reminder_offset = ?, reminder_timestamp = ?
                    WHERE linked_meeting_id = ?
                """, (title, task_desc, task_deadline, buffer_lead, task_status, meeting_date, start_time, reminder_offset, reminder_timestamp, m_id))
            else:
                cursor.execute("""
                    INSERT INTO reminders (username, task_title, description, deadline_text, buffer_lead, status, 
                                           due_date, due_time, reminder_offset, reminder_timestamp, linked_meeting_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (row_user or uname or 'admin', title, task_desc, task_deadline, buffer_lead, task_status, meeting_date, start_time, reminder_offset, reminder_timestamp, m_id))
                new_tid = cursor.lastrowid
                if new_tid:
                    cursor.execute("UPDATE meetings SET linked_task_id = ? WHERE id = ?", (new_tid, m_id))

        conn.commit()
        return True
    finally:
        try:
            conn.close()
        except Exception:
            pass

def update_meeting_status(meeting_id, username, status):
    try:
        m_id = int(meeting_id)
    except (ValueError, TypeError):
        return False

    uname = (username or '').strip().lower() if username else None
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        if is_postgres():
            cursor.execute("SELECT id, username, linked_task_id FROM meetings WHERE id = %s", (m_id,))
        else:
            cursor.execute("SELECT id, username, linked_task_id FROM meetings WHERE id = ?", (m_id,))
        row = cursor.fetchone()
        if not row:
            return False

        row_user = (row[1] if isinstance(row, tuple) else row['username']) or ''
        row_user = row_user.strip().lower()
        authorized = (not uname or uname == 'admin' or row_user == uname or not row_user or row_user == 'admin')
        if not authorized:
            return False

        if is_postgres():
            cursor.execute("UPDATE meetings SET status = %s WHERE id = %s", (status, m_id))
        else:
            cursor.execute("UPDATE meetings SET status = ? WHERE id = ?", (status, m_id))

        # Bi-directional sync with linked Upcoming Task
        task_status = 'Completed' if status.lower() == 'completed' else 'Pending'
        if is_postgres():
            cursor.execute("UPDATE reminders SET status = %s WHERE linked_meeting_id = %s", (task_status, m_id))
        else:
            cursor.execute("UPDATE reminders SET status = ? WHERE linked_meeting_id = ?", (task_status, m_id))

        conn.commit()
        return True
    finally:
        try:
            conn.close()
        except Exception:
            pass

def delete_meeting(meeting_id, username=None):
    try:
        m_id = int(meeting_id)
    except (ValueError, TypeError):
        return False

    uname = (username or '').strip().lower() if username else None
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        if is_postgres():
            cursor.execute("SELECT id, username, linked_task_id FROM meetings WHERE id = %s", (m_id,))
        else:
            cursor.execute("SELECT id, username, linked_task_id FROM meetings WHERE id = ?", (m_id,))
        row = cursor.fetchone()
        if not row:
            return False

        row_user = (row[1] if isinstance(row, tuple) else row['username']) or ''
        row_user = row_user.strip().lower()

        authorized = (
            not uname or
            uname == 'admin' or
            row_user == uname or
            not row_user or
            row_user == 'admin'
        )
        if not authorized:
            return False

        # Deleting a meeting also removes its generated task in Upcoming Tasks
        if is_postgres():
            cursor.execute("DELETE FROM reminders WHERE linked_meeting_id = %s", (m_id,))
            cursor.execute("DELETE FROM meetings WHERE id = %s", (m_id,))
        else:
            cursor.execute("DELETE FROM reminders WHERE linked_meeting_id = ?", (m_id,))
            cursor.execute("DELETE FROM meetings WHERE id = ?", (m_id,))
        conn.commit()
        return True
    finally:
        try:
            conn.close()
        except Exception:
            pass

# --- Task Database Operations ---

def get_tasks_for_user(username):
    uname = (username or 'admin').strip().lower()
    rows = query_db("""
        SELECT id, task_title, description, deadline_text, buffer_lead, status, due_date, due_time, 
               reminder_offset, reminder_timestamp, source_message, sender, created_at, linked_meeting_id 
        FROM reminders 
        WHERE LOWER(username) = ? 
        ORDER BY id DESC
    """, (uname,))
    return [
        {
            'id': row['id'],
            'task_title': row['task_title'],
            'description': row.get('description') or '',
            'deadline_text': row.get('deadline_text') or 'No deadline specified',
            'buffer_lead': row.get('buffer_lead') or '30m prior',
            'status': row.get('status') or 'Pending',
            'computed_status': compute_item_status(row.get('status'), row.get('due_date'), row.get('due_time'), 'Upcoming'),
            'due_date': row.get('due_date') or '',
            'due_time': row.get('due_time') or '',
            'reminder_offset': row.get('reminder_offset') if row.get('reminder_offset') is not None else 30,
            'reminder_timestamp': row.get('reminder_timestamp'),
            'source_message': row.get('source_message') or '',
            'sender': row.get('sender') or '',
            'created_at': str(row.get('created_at') or ''),
            'linked_meeting_id': row.get('linked_meeting_id')
        } for row in rows
    ]

def get_task_by_title(username, task_title):
    uname = (username or 'admin').strip().lower()
    title = task_title.strip()
    row = query_db(
        "SELECT * FROM reminders WHERE LOWER(username) = ? AND LOWER(task_title) = ? AND status != 'Completed'",
        (uname, title.lower()),
        one=True
    )
    return row

def add_task(username, task_title, deadline_text="No deadline specified", buffer_lead="30m prior", status="Pending", 
             due_date="", due_time="", description="", reminder_offset=30, reminder_timestamp=None, 
             source_message="", sender="", linked_meeting_id=None, return_tuple=False):
    uname = (username or 'admin').strip().lower()
    title = task_title.strip()
    deadline_text = (deadline_text or 'No deadline specified').strip()
    buffer_lead = (buffer_lead or '30m prior').strip()
    status = (status or 'Pending').strip()
    due_date = (due_date or '').strip()
    due_time = (due_time or '').strip()
    description = (description or '').strip()

    try:
        reminder_offset = int(reminder_offset) if reminder_offset is not None else 30
    except Exception:
        reminder_offset = 30

    if reminder_timestamp is None and reminder_offset > 0:
        reminder_timestamp = calculate_reminder_timestamp(due_date or deadline_text, due_time, reminder_offset)

    # Deduplication check
    existing = get_task_by_title(uname, title)
    if existing:
        if (due_date or due_time or description) and (not existing.get('due_date') and not existing.get('due_time')):
            update_task(
                task_id=existing['id'],
                username=uname,
                task_title=title,
                deadline_text=deadline_text if deadline_text != 'No deadline specified' else existing.get('deadline_text', 'No deadline specified'),
                buffer_lead=buffer_lead,
                status=status or existing.get('status', 'Pending'),
                due_date=due_date or existing.get('due_date', ''),
                due_time=due_time or existing.get('due_time', ''),
                description=description or existing.get('description', ''),
                reminder_offset=reminder_offset,
                reminder_timestamp=reminder_timestamp or existing.get('reminder_timestamp')
            )
        existing_id = int(existing['id'])
        return (existing_id, False) if return_tuple else existing_id

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        if is_postgres():
            cursor.execute("""
                INSERT INTO reminders (username, task_title, description, deadline_text, buffer_lead, status, 
                                       due_date, due_time, reminder_offset, reminder_timestamp, source_message, sender, linked_meeting_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (uname, title, description, deadline_text, buffer_lead, status, due_date, due_time, reminder_offset, reminder_timestamp, source_message, sender, linked_meeting_id))
            row = cursor.fetchone()
            new_id = int(row[0]) if row else None
        else:
            cursor.execute("""
                INSERT INTO reminders (username, task_title, description, deadline_text, buffer_lead, status, 
                                       due_date, due_time, reminder_offset, reminder_timestamp, source_message, sender, linked_meeting_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (uname, title, description, deadline_text, buffer_lead, status, due_date, due_time, reminder_offset, reminder_timestamp, source_message, sender, linked_meeting_id))
            new_id = int(cursor.lastrowid) if cursor.lastrowid else None

        conn.commit()
        return (new_id, True) if return_tuple else new_id
    finally:
        try:
            conn.close()
        except Exception:
            pass

def update_task(task_id, username, task_title, deadline_text="No deadline specified", buffer_lead="30m prior", 
                status="Pending", due_date="", due_time="", description="", reminder_offset=30, reminder_timestamp=None):
    try:
        t_id = int(task_id)
    except (ValueError, TypeError):
        return False

    uname = (username or '').strip().lower() if username else None
    title = task_title.strip()
    deadline_text = (deadline_text or 'No deadline specified').strip()
    buffer_lead = (buffer_lead or '30m prior').strip()
    status = (status or 'Pending').strip()
    due_date = (due_date or '').strip()
    due_time = (due_time or '').strip()
    description = (description or '').strip()

    try:
        reminder_offset = int(reminder_offset) if reminder_offset is not None else 30
    except Exception:
        reminder_offset = 30

    if reminder_timestamp is None and reminder_offset > 0:
        reminder_timestamp = calculate_reminder_timestamp(due_date or deadline_text, due_time, reminder_offset)

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        if is_postgres():
            cursor.execute("SELECT id, username, linked_meeting_id FROM reminders WHERE id = %s", (t_id,))
        else:
            cursor.execute("SELECT id, username, linked_meeting_id FROM reminders WHERE id = ?", (t_id,))
        row = cursor.fetchone()
        if not row:
            return False

        row_user = (row[1] if isinstance(row, tuple) else row['username']) or ''
        row_user = row_user.strip().lower()
        authorized = (not uname or uname == 'admin' or row_user == uname or not row_user or row_user == 'admin')
        if not authorized:
            return False

        if is_postgres():
            cursor.execute("""
                UPDATE reminders 
                SET task_title = %s, description = %s, deadline_text = %s, buffer_lead = %s, status = %s, 
                    due_date = %s, due_time = %s, reminder_offset = %s, reminder_timestamp = %s
                WHERE id = %s
            """, (title, description, deadline_text, buffer_lead, status, due_date, due_time, reminder_offset, reminder_timestamp, t_id))
        else:
            cursor.execute("""
                UPDATE reminders 
                SET task_title = ?, description = ?, deadline_text = ?, buffer_lead = ?, status = ?, 
                    due_date = ?, due_time = ?, reminder_offset = ?, reminder_timestamp = ?
                WHERE id = ?
            """, (title, description, deadline_text, buffer_lead, status, due_date, due_time, reminder_offset, reminder_timestamp, t_id))

        # If linked to a meeting, sync title, date, time, reminder and status to meeting
        linked_m_id = row['linked_meeting_id'] if hasattr(row, 'keys') else (row[2] if len(row) > 2 else None)
        if linked_m_id:
            m_status = 'Completed' if status.lower() == 'completed' else 'Scheduled'
            sched_time = f"{due_date} {due_time}".strip() or deadline_text
            if is_postgres():
                cursor.execute("""
                    UPDATE meetings 
                    SET title = %s, meeting_date = %s, start_time = %s, scheduled_time = %s,
                        reminder_offset = %s, reminder_timestamp = %s, status = %s
                    WHERE id = %s
                """, (title, due_date, due_time, sched_time, reminder_offset, reminder_timestamp, m_status, linked_m_id))
            else:
                cursor.execute("""
                    UPDATE meetings 
                    SET title = ?, meeting_date = ?, start_time = ?, scheduled_time = ?,
                        reminder_offset = ?, reminder_timestamp = ?, status = ?
                    WHERE id = ?
                """, (title, due_date, due_time, sched_time, reminder_offset, reminder_timestamp, m_status, linked_m_id))

        conn.commit()
        return True
    finally:
        try:
            conn.close()
        except Exception:
            pass

def update_task_status(task_id, username, status):
    try:
        t_id = int(task_id)
    except (ValueError, TypeError):
        return False

    uname = (username or '').strip().lower() if username else None
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        if is_postgres():
            cursor.execute("SELECT id, username, linked_meeting_id FROM reminders WHERE id = %s", (t_id,))
        else:
            cursor.execute("SELECT id, username, linked_meeting_id FROM reminders WHERE id = ?", (t_id,))
        row = cursor.fetchone()
        if not row:
            return False

        row_user = (row[1] if isinstance(row, tuple) else row['username']) or ''
        row_user = row_user.strip().lower()
        authorized = (not uname or uname == 'admin' or row_user == uname or not row_user or row_user == 'admin')
        if not authorized:
            return False

        if is_postgres():
            cursor.execute("UPDATE reminders SET status = %s WHERE id = %s", (status, t_id))
        else:
            cursor.execute("UPDATE reminders SET status = ? WHERE id = ?", (status, t_id))

        # Bi-directional sync with linked Meeting
        linked_m_id = row['linked_meeting_id'] if hasattr(row, 'keys') else (row[2] if len(row) > 2 else None)
        if linked_m_id:
            m_status = 'Completed' if status.lower() == 'completed' else 'Scheduled'
            if is_postgres():
                cursor.execute("UPDATE meetings SET status = %s WHERE id = %s", (m_status, linked_m_id))
            else:
                cursor.execute("UPDATE meetings SET status = ? WHERE id = ?", (m_status, linked_m_id))

        conn.commit()
        return True
    finally:
        try:
            conn.close()
        except Exception:
            pass

def delete_task(task_id, username=None):
    try:
        t_id = int(task_id)
    except (ValueError, TypeError):
        return False

    uname = (username or '').strip().lower() if username else None
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        if is_postgres():
            cursor.execute("SELECT id, username, linked_meeting_id FROM reminders WHERE id = %s", (t_id,))
        else:
            cursor.execute("SELECT id, username, linked_meeting_id FROM reminders WHERE id = ?", (t_id,))
        row = cursor.fetchone()
        if not row:
            return False

        row_user = (row[1] if isinstance(row, tuple) else row['username']) or ''
        row_user = row_user.strip().lower()

        authorized = (
            not uname or
            uname == 'admin' or
            row_user == uname or
            not row_user or
            row_user == 'admin'
        )
        if not authorized:
            return False

        # If this task was generated from a meeting, deleting the task removes it from Upcoming Tasks
        # without deleting the original meeting record.
        linked_m_id = row['linked_meeting_id'] if hasattr(row, 'keys') else (row[2] if len(row) > 2 else None)
        if linked_m_id:
            if is_postgres():
                cursor.execute("UPDATE meetings SET linked_task_id = NULL WHERE id = %s", (linked_m_id,))
            else:
                cursor.execute("UPDATE meetings SET linked_task_id = NULL WHERE id = ?", (linked_m_id,))

        if is_postgres():
            cursor.execute("DELETE FROM reminders WHERE id = %s", (t_id,))
        else:
            cursor.execute("DELETE FROM reminders WHERE id = ?", (t_id,))
        conn.commit()
        return True
    finally:
        try:
            conn.close()
        except Exception:
            pass

def get_upcoming_reminders(username):
    """
    Retrieves all future scheduled meetings and non-completed tasks that have active reminder timestamps.
    Completed or cancelled meetings/tasks are excluded.
    Tasks generated from meetings are excluded to prevent duplicate alarms (meeting alarm handles them).
    """
    uname = (username or 'admin').strip().lower()
    now_ms = int(time.time() * 1000)

    meeting_rows = query_db("""
        SELECT id, platform, meeting_url, scheduled_time, meeting_date, start_time, 
               reminder_offset, reminder_timestamp, status, title 
        FROM meetings 
        WHERE LOWER(username) = ? AND reminder_timestamp IS NOT NULL AND reminder_timestamp > ? 
          AND status != 'Cancelled' AND status != 'Completed'
        ORDER BY reminder_timestamp ASC
    """, (uname, now_ms))

    task_rows = query_db("""
        SELECT id, task_title, description, deadline_text, due_date, due_time, 
               reminder_offset, reminder_timestamp, status 
        FROM reminders 
        WHERE LOWER(username) = ? AND reminder_timestamp IS NOT NULL AND reminder_timestamp > ? 
          AND status != 'Completed' AND status != 'Cancelled'
          AND (linked_meeting_id IS NULL OR linked_meeting_id = 0)
        ORDER BY reminder_timestamp ASC
    """, (uname, now_ms))

    return {
        'status': 'success',
        'meetings': meeting_rows,
        'tasks': task_rows,
        'server_time_ms': now_ms
    }