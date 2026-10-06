import time
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from predict import analyze_message
from temporal_engine import extract_meeting_links, extract_deadline_and_buffer, is_within_working_hours
from database import (
    get_db,
    init_db,
    log_analysis,
    register_user,
    verify_user
)

app = Flask(__name__)
app.secret_key = 'smart_messaging_gateway_secure_2026'

init_db()

STAGED_REPLIES = {}
LIVE_FEED_ALERTS = []

@app.route('/')
@app.route('/index.html')
def home():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template(
        'index.html',
        username=session.get('username', 'Admin'),
        full_name=session.get('full_name', ''),
        email=session.get('email', '')
    )

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        identifier = request.form.get('identifier', '').strip()
        password = request.form.get('password', '').strip()
        user = verify_user(identifier, password)
        if user:
            session['logged_in'] = True
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['full_name'] = user['full_name']
            session['email'] = user['email']
            return redirect(url_for('home'))
        error = 'Invalid username/email or password.'
    return render_template('login.html', error=error)

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == 'POST':
        full_name = request.form.get('full_name', '').strip()
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()

        if password != confirm_password:
            return render_template('register.html', error='Passwords do not match.')

        ok, msg = register_user(username, email, full_name, password)
        if ok:
            return render_template('login.html', success=msg)
        error = msg
    return render_template('register.html', error=error)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/get_settings', methods=['GET'])
def get_settings():
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    username = session.get('username')
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM user_settings WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return jsonify({
            'active_hours_enabled': 1,
            'work_start_time': '09:00',
            'work_end_time': '17:00',
            'allow_auto_reply': 0,
            'dnd_bypass_enabled': 1,
            'default_persona': 'casual',
            'preferences_configured': 0
        })
    
    return jsonify({
        'active_hours_enabled': row['active_hours_enabled'],
        'work_start_time': row['work_start_time'] or '09:00',
        'work_end_time': row['work_end_time'] or '17:00',
        'allow_auto_reply': row['allow_auto_reply'],
        'dnd_bypass_enabled': row['dnd_bypass_enabled'],
        'default_persona': row['default_persona'],
        'preferences_configured': 1
    })

@app.route('/update_settings', methods=['POST'])
def update_settings():
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    username = session.get('username')
    data = request.json or {}
    
    active_enabled = 1 if data.get('active_hours_enabled') else 0
    work_start = data.get('work_start_time', '09:00')
    work_end = data.get('work_end_time', '17:00')
    auto_reply = 1 if data.get('allow_auto_reply') else 0
    dnd_bypass = 1 if data.get('dnd_bypass_enabled', True) else 0
    persona = data.get('persona', 'casual')

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO user_settings (username, active_hours_enabled, work_start_time, work_end_time, allow_auto_reply, dnd_bypass_enabled, default_persona)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(username) DO UPDATE SET
            active_hours_enabled = excluded.active_hours_enabled,
            work_start_time = excluded.work_start_time,
            work_end_time = excluded.work_end_time,
            allow_auto_reply = excluded.allow_auto_reply,
            dnd_bypass_enabled = excluded.dnd_bypass_enabled,
            default_persona = excluded.default_persona
    ''', (username, active_enabled, work_start, work_end, auto_reply, dnd_bypass, persona))
    conn.commit()
    conn.close()

    return jsonify({'status': 'success'})

@app.route('/get_tasks', methods=['GET'])
def get_tasks():
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    username = session.get('username')
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, task_title, deadline_text, buffer_lead, status FROM reminders WHERE username = ? ORDER BY id DESC", (username,))
    rows = cursor.fetchall()
    conn.close()

    tasks = [
        {
            'id': row['id'],
            'task_title': row['task_title'],
            'deadline_text': row['deadline_text'],
            'buffer_lead': row['buffer_lead'],
            'status': row['status']
        } for row in rows
    ]
    return jsonify({'tasks': tasks})

@app.route('/update_task_status', methods=['POST'])
def update_task_status():
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.json or {}
    task_id = data.get('task_id')
    new_status = data.get('status', 'Completed')
    username = session.get('username')

    if not task_id:
        return jsonify({'error': 'Task ID missing'}), 400

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE reminders 
        SET status = ? 
        WHERE id = ? AND username = ?
    """, (new_status, task_id, username))
    conn.commit()
    conn.close()

    return jsonify({'status': 'success', 'updated_status': new_status})

@app.route('/get_meetings', methods=['GET'])
def get_meetings():
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    username = session.get('username')
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT platform, meeting_url, scheduled_time FROM meetings WHERE username = ? ORDER BY id DESC", (username,))
    rows = cursor.fetchall()
    conn.close()

    meetings = [
        {
            'platform': row['platform'],
            'meeting_url': row['meeting_url'],
            'scheduled_time': row['scheduled_time']
        } for row in rows
    ]
    return jsonify({'meetings': meetings})

@app.route('/ingest_notification', methods=['POST'])
def ingest_notification():
    data = request.json or {}
    message = data.get('message', '').strip()
    sender = data.get('sender', 'Unknown')
    app_source = data.get('app_source', 'Notification')

    if not message:
        return jsonify({'error': 'No message content provided'}), 400

    results = analyze_message(message, persona="casual")
    meeting_info = extract_meeting_links(message)
    deadline_info = extract_deadline_and_buffer(message)

    priority = results.get('Priority', 'STANDARD')
    username = session.get('username', 'Admin')
    
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM user_settings WHERE username = ?", (username,))
    settings = cursor.fetchone()
    conn.close()

    is_urgent = "URGENT" in priority
    in_work_hours = True
    if settings and settings['active_hours_enabled']:
        in_work_hours = is_within_working_hours(settings['work_start_time'], settings['work_end_time'])

    status_tag = "Live"
    if not is_urgent and not in_work_hours:
        status_tag = "Deferred (1-Hour Buffer / Off-Hours)"

    auto_reply_allowed = settings and settings['allow_auto_reply'] == 1
    if auto_reply_allowed and not is_urgent:
        status_tag = "Autonomous Auto-Replied 🤖"

    alert_item = {
        'id': int(time.time() * 1000),
        'app_source': app_source,
        'sender': sender,
        'message': message,
        'priority': priority,
        'badge_color': results.get('Badge Color', '#10b981'),
        'directives': results.get('Relay Directives', []),
        'meeting_url': meeting_info.get('join_url') if meeting_info else None,
        'meeting_platform': meeting_info.get('platform') if meeting_info else None,
        'status_tag': status_tag,
        'timestamp': time.strftime("%I:%M %p")
    }

    LIVE_FEED_ALERTS.insert(0, alert_item)
    if len(LIVE_FEED_ALERTS) > 50:
        LIVE_FEED_ALERTS.pop()

    log_analysis(
        username=username,
        message=f"[{app_source}] {sender}: {message}",
        sentiment=results['Sentiment'][0],
        urgency=results['Urgency'][0],
        priority=priority
    )

    return jsonify({'status': 'ingested', 'alert': alert_item, 'delivery_mode': status_tag})

@app.route('/get_feed', methods=['GET'])
def get_feed():
    return jsonify({'feed': LIVE_FEED_ALERTS})

@app.route('/stage_reply', methods=['POST'])
def stage_reply():
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    data = request.json or {}
    reply_id = f"reply_{int(time.time() * 1000)}"
    STAGED_REPLIES[reply_id] = {
        "username": session.get('username'),
        "reply_text": data.get('reply_text', ''),
        "status": "pending"
    }
    return jsonify({"status": "staged", "reply_id": reply_id, "cooloff_seconds": 7})

@app.route('/cancel_reply', methods=['POST'])
def cancel_reply():
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    data = request.json or {}
    reply_id = data.get('reply_id')
    if reply_id in STAGED_REPLIES:
        STAGED_REPLIES[reply_id]['status'] = 'cancelled'
        return jsonify({"status": "cancelled", "message": "Draft recalled successfully."})
    return jsonify({"error": "Draft ID not found"}), 404

@app.route('/dispatch_reply', methods=['POST'])
def dispatch_reply():
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    data = request.json or {}
    reply_id = data.get('reply_id')
    if reply_id in STAGED_REPLIES:
        if STAGED_REPLIES[reply_id]['status'] != 'cancelled':
            STAGED_REPLIES[reply_id]['status'] = 'dispatched'
            return jsonify({"status": "sent", "message": "Draft dispatched successfully."})
        return jsonify({"status": "aborted", "message": "Draft was already recalled."})
    return jsonify({"error": "Draft ID not found"}), 404

@app.route('/analyze', methods=['POST'])
def analyze():
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401

    data = request.json or {}
    message = data.get('message', '').strip()
    persona = data.get('persona', 'casual')
    thread_count = int(data.get('thread_count', 1))

    if not message:
        return jsonify({'error': 'No message provided'}), 400

    results = analyze_message(message, persona=persona, thread_count=thread_count)
    meeting_info = extract_meeting_links(message)
    deadline_info = extract_deadline_and_buffer(message)

    if meeting_info and "Relay Directives" in results:
        results["Relay Directives"].insert(0, meeting_info["directive"])

    username = session.get('username', 'Admin')
    log_analysis(
        username=username,
        message=message,
        sentiment=results['Sentiment'][0],
        urgency=results['Urgency'][0],
        priority=results['Priority']
    )

    if deadline_info:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO reminders (username, task_title, deadline_text, buffer_lead, status)
            VALUES (?, ?, ?, ?, 'Pending')
        """, (username, message[:50], deadline_info['detected_deadline'], deadline_info['lead_buffer']))
        conn.commit()
        conn.close()

    formatted = {
        'analysis': {
            'Sentiment': {
                'prediction': results['Sentiment'][0],
                'confidence': results['Sentiment'][1]
            },
            'Message Type': {
                'prediction': results['Message Type'][0],
                'confidence': results['Message Type'][1]
            },
            'Urgency': {
                'prediction': results['Urgency'][0],
                'confidence': results['Urgency'][1]
            },
            'Tone': {
                'prediction': results['Tone'][0],
                'confidence': results['Tone'][1]
            },
            'Suggested Action': {
                'prediction': results['Suggested Action'][0],
                'confidence': results['Suggested Action'][1]
            }
        },
        'priority': results['Priority'],
        'badge_color': results['Badge Color'],
        'relay_directives': results['Relay Directives'],
        'meeting_info': meeting_info,
        'deadline_info': deadline_info
    }
    return jsonify(formatted)

if __name__ == '__main__':
    app.run(debug=True, port=5000)