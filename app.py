import time
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from predict import analyze_message
from temporal_engine import (
    extract_meeting_links,
    extract_all_meeting_links,
    extract_task_from_message,
    extract_deadline_and_buffer,
    calculate_reminder_timestamp,
    is_within_working_hours,
    detect_meeting_platform,
    validate_meeting_url
)
from database import (
    get_db,
    init_db,
    log_analysis,
    register_user,
    verify_user,
    get_meetings_for_user,
    add_meeting,
    update_meeting,
    delete_meeting,
    get_tasks_for_user,
    add_task,
    update_task,
    update_task_status,
    delete_task,
    get_upcoming_reminders,
    query_db
)

app = Flask(__name__)
app.secret_key = 'smart_messaging_gateway_secure_2026'

init_db()

STAGED_REPLIES = {}
LIVE_FEED_ALERTS = []

def resolve_username_or_default(explicit_user=None):
    """
    Resolves the active user. Checks explicit parameter, session, JSON payload,
    headers, and query params. If none found, safely falls back to the registered user or 'admin'.
    """
    if explicit_user and isinstance(explicit_user, str) and explicit_user.strip():
        return explicit_user.strip()
    if session.get('logged_in') and session.get('username'):
        return session.get('username').strip()
    if request.is_json and request.json:
        uname = request.json.get('username')
        if uname and isinstance(uname, str) and uname.strip():
            return uname.strip()
    header_user = request.headers.get('X-Username') or request.args.get('username')
    if header_user and header_user.strip():
        return header_user.strip()
    try:
        first_user = query_db("SELECT username FROM users ORDER BY id ASC LIMIT 1", one=True)
        if first_user and first_user.get('username'):
            return first_user['username']
    except Exception:
        pass
    return 'admin'

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

# --- Task Routes ---

@app.route('/get_tasks', methods=['GET'])
@app.route('/api/tasks', methods=['GET'])
def get_tasks_route():
    username = resolve_username_or_default(session.get('username') or request.args.get('username'))
    tasks = get_tasks_for_user(username)
    return jsonify({'tasks': tasks})

@app.route('/add_task', methods=['POST'])
@app.route('/api/add_task', methods=['POST'])
def add_task_route():
    data = request.json or {}
    username = resolve_username_or_default(data.get('username') or session.get('username'))

    task_title = data.get('task_title', '').strip()
    if not task_title:
        return jsonify({'error': 'Task title is required.'}), 400

    deadline_text = data.get('deadline_text', '').strip()
    due_date = data.get('due_date', '').strip()
    due_time = data.get('due_time', '').strip()
    description = data.get('description', '').strip()

    if not deadline_text and (due_date or due_time):
        deadline_text = f"{due_date} {due_time}".strip()
    elif not deadline_text:
        deadline_text = 'No deadline specified'

    buffer_lead = data.get('buffer_lead', '').strip() or '30m prior'
    status = data.get('status', 'Pending').strip() or 'Pending'
    reminder_offset = data.get('reminder_offset', 30)
    source_message = data.get('source_message', '').strip()
    sender = data.get('sender', '').strip()

    try:
        new_id = add_task(
            username=username,
            task_title=task_title,
            deadline_text=deadline_text,
            buffer_lead=buffer_lead,
            status=status,
            due_date=due_date,
            due_time=due_time,
            description=description,
            reminder_offset=reminder_offset,
            source_message=source_message,
            sender=sender
        )
        task_obj = next((t for t in get_tasks_for_user(username) if t['id'] == new_id), {
            'id': new_id,
            'task_title': task_title,
            'description': description,
            'deadline_text': deadline_text,
            'buffer_lead': buffer_lead,
            'status': status,
            'due_date': due_date,
            'due_time': due_time,
            'reminder_offset': reminder_offset
        })
        return jsonify({
            'status': 'success',
            'message': 'Task added successfully',
            'task': task_obj
        }), 201
    except Exception as e:
        return jsonify({'error': f'Failed to add task: {str(e)}'}), 500

@app.route('/update_task', methods=['POST'])
@app.route('/api/update_task', methods=['POST'])
def update_task_route():
    data = request.json or {}
    username = resolve_username_or_default(data.get('username') or session.get('username'))

    task_id = data.get('id') or data.get('task_id')
    if not task_id:
        return jsonify({'error': 'Task ID is required.'}), 400

    task_title = data.get('task_title', '').strip()
    if not task_title:
        return jsonify({'error': 'Task title cannot be empty.'}), 400

    deadline_text = data.get('deadline_text', '').strip()
    due_date = data.get('due_date', '').strip()
    due_time = data.get('due_time', '').strip()
    description = data.get('description', '').strip()

    if not deadline_text and (due_date or due_time):
        deadline_text = f"{due_date} {due_time}".strip()
    elif not deadline_text:
        deadline_text = 'No deadline specified'

    buffer_lead = data.get('buffer_lead', '').strip() or '30m prior'
    status = data.get('status', 'Pending').strip() or 'Pending'
    reminder_offset = data.get('reminder_offset', 30)

    try:
        updated = update_task(
            task_id=task_id,
            username=username,
            task_title=task_title,
            deadline_text=deadline_text,
            buffer_lead=buffer_lead,
            status=status,
            due_date=due_date,
            due_time=due_time,
            description=description,
            reminder_offset=reminder_offset
        )
        if not updated:
            return jsonify({'error': 'Task not found or unauthorized.'}), 404

        task_obj = next((t for t in get_tasks_for_user(username) if t['id'] == int(task_id)), {
            'id': int(task_id),
            'task_title': task_title,
            'description': description,
            'deadline_text': deadline_text,
            'buffer_lead': buffer_lead,
            'status': status,
            'due_date': due_date,
            'due_time': due_time,
            'reminder_offset': reminder_offset
        })
        return jsonify({
            'status': 'success',
            'message': 'Task updated successfully',
            'task': task_obj
        })
    except Exception as e:
        return jsonify({'error': f'Failed to update task: {str(e)}'}), 500

@app.route('/update_task_status', methods=['POST'])
@app.route('/api/update_task_status', methods=['POST'])
def update_task_status_route():
    data = request.json or {}
    task_id = data.get('task_id') or data.get('id')
    new_status = data.get('status', 'Completed').strip() or 'Completed'
    username = resolve_username_or_default(data.get('username') or session.get('username'))

    if not task_id:
        return jsonify({'error': 'Task ID missing.'}), 400

    try:
        updated = update_task_status(task_id=task_id, username=username, status=new_status)
        if not updated:
            return jsonify({'error': 'Task not found or unauthorized.'}), 404

        return jsonify({
            'status': 'success',
            'message': f'Task status updated to {new_status}',
            'updated_status': new_status,
            'task_id': task_id
        })
    except Exception as e:
        return jsonify({'error': f'Failed to update task status: {str(e)}'}), 500

@app.route('/delete_task', methods=['POST'])
def delete_task_route():
    data = request.json or {}
    username = resolve_username_or_default(data.get('username') or session.get('username'))

    task_id = data.get('task_id') or data.get('id')
    if not task_id:
        return jsonify({'error': 'Task ID is required.'}), 400

    try:
        deleted = delete_task(task_id=task_id, username=username)
        if not deleted:
            return jsonify({'error': 'Task not found or unauthorized.'}), 404

        return jsonify({
            'status': 'success',
            'message': 'Task deleted successfully',
            'task_id': task_id
        })
    except Exception as e:
        return jsonify({'error': f'Failed to delete task: {str(e)}'}), 500

# --- Meeting Routes ---

@app.route('/get_meetings', methods=['GET'])
@app.route('/api/meetings', methods=['GET'])
def get_meetings_route():
    username = resolve_username_or_default(session.get('username') or request.args.get('username'))
    meetings = get_meetings_for_user(username)
    return jsonify({'meetings': meetings})

@app.route('/add_meeting', methods=['POST'])
@app.route('/api/add_meeting', methods=['POST'])
def add_meeting_route():
    data = request.json or {}
    username = resolve_username_or_default(data.get('username') or session.get('username'))

    raw_url = data.get('meeting_url', '').strip()
    is_valid, validated_url_or_err = validate_meeting_url(raw_url)
    if not is_valid:
        return jsonify({'error': validated_url_or_err}), 400

    url = validated_url_or_err
    platform = data.get('platform', '').strip()
    if not platform or platform.lower() == 'auto-detect':
        platform = detect_meeting_platform(url)

    title = data.get('title', '').strip() or f"{platform} Meeting"
    scheduled_time = data.get('scheduled_time', '').strip()
    meeting_date = data.get('meeting_date', '').strip()
    start_time = data.get('start_time', '').strip()
    reminder_offset = data.get('reminder_offset', 30)
    status = data.get('status', 'Scheduled').strip() or 'Scheduled'

    if not scheduled_time and (meeting_date or start_time):
        scheduled_time = f"{meeting_date} {start_time}".strip()
    elif not scheduled_time:
        scheduled_time = 'Unknown'

    source_message = data.get('source_message', '').strip()
    sender = data.get('sender', '').strip()

    try:
        new_id = add_meeting(
            username=username,
            platform=platform,
            meeting_url=url,
            scheduled_time=scheduled_time,
            title=title,
            source_message=source_message,
            sender=sender,
            meeting_date=meeting_date,
            start_time=start_time,
            reminder_offset=reminder_offset,
            status=status
        )
        meeting_obj = next((m for m in get_meetings_for_user(username) if m['id'] == new_id), {
            'id': new_id,
            'platform': platform,
            'meeting_url': url,
            'scheduled_time': scheduled_time,
            'title': title,
            'meeting_date': meeting_date,
            'start_time': start_time,
            'reminder_offset': reminder_offset,
            'status': status
        })
        return jsonify({
            'status': 'success',
            'message': 'Meeting link saved successfully',
            'meeting': meeting_obj
        }), 201
    except Exception as e:
        return jsonify({'error': f'Failed to save meeting link: {str(e)}'}), 500

@app.route('/update_meeting', methods=['POST'])
def update_meeting_route():
    data = request.json or {}
    username = resolve_username_or_default(data.get('username') or session.get('username'))

    meeting_id = data.get('id') or data.get('meeting_id')
    if not meeting_id:
        return jsonify({'error': 'Meeting ID is required.'}), 400

    raw_url = data.get('meeting_url', '').strip()
    is_valid, validated_url_or_err = validate_meeting_url(raw_url)
    if not is_valid:
        return jsonify({'error': validated_url_or_err}), 400

    url = validated_url_or_err
    platform = data.get('platform', '').strip()
    if not platform or platform.lower() == 'auto-detect':
        platform = detect_meeting_platform(url)

    title = data.get('title', '').strip() or f"{platform} Meeting"
    scheduled_time = data.get('scheduled_time', '').strip()
    meeting_date = data.get('meeting_date', '').strip()
    start_time = data.get('start_time', '').strip()
    reminder_offset = data.get('reminder_offset', 30)
    status = data.get('status', 'Scheduled').strip() or 'Scheduled'

    if not scheduled_time and (meeting_date or start_time):
        scheduled_time = f"{meeting_date} {start_time}".strip()
    elif not scheduled_time:
        scheduled_time = 'Unknown'

    try:
        updated = update_meeting(
            meeting_id=meeting_id,
            username=username,
            platform=platform,
            meeting_url=url,
            scheduled_time=scheduled_time,
            title=title,
            meeting_date=meeting_date,
            start_time=start_time,
            reminder_offset=reminder_offset,
            status=status
        )
        if not updated:
            return jsonify({'error': 'Meeting not found or unauthorized.'}), 404

        meeting_obj = next((m for m in get_meetings_for_user(username) if m['id'] == int(meeting_id)), {
            'id': int(meeting_id),
            'platform': platform,
            'meeting_url': url,
            'scheduled_time': scheduled_time,
            'title': title,
            'meeting_date': meeting_date,
            'start_time': start_time,
            'reminder_offset': reminder_offset,
            'status': status
        })
        return jsonify({
            'status': 'success',
            'message': 'Meeting link updated successfully',
            'meeting': meeting_obj
        })
    except Exception as e:
        return jsonify({'error': f'Failed to update meeting link: {str(e)}'}), 500

@app.route('/delete_meeting', methods=['POST'])
def delete_meeting_route():
    data = request.json or {}
    username = resolve_username_or_default(data.get('username') or session.get('username'))

    meeting_id = data.get('id') or data.get('meeting_id')
    if not meeting_id:
        return jsonify({'error': 'Meeting ID is required.'}), 400

    try:
        deleted = delete_meeting(meeting_id=meeting_id, username=username)
        if not deleted:
            return jsonify({'error': 'Meeting not found or unauthorized.'}), 404

        return jsonify({
            'status': 'success',
            'message': 'Meeting link deleted successfully',
            'meeting_id': meeting_id
        })
    except Exception as e:
        return jsonify({'error': f'Failed to delete meeting: {str(e)}'}), 500

# --- Quick Action Endpoints for Notifications & Mobile ---

@app.route('/quick_add_task', methods=['POST'])
@app.route('/api/quick_add_task', methods=['POST'])
def quick_add_task_route():
    """
    Action A: Quick Add Task from Android Notification or API.
    Saves message or extracted action item as a persistent task with deduplication and reminder offset.
    """
    data = request.json or {}
    message = data.get('message', '').strip()
    sender = data.get('sender', 'Unknown')
    username = resolve_username_or_default(data.get('username'))
    reminder_offset = data.get('reminder_offset', 30)

    explicit_title = data.get('task_title', '').strip()
    explicit_deadline = data.get('deadline_text', '').strip()

    if explicit_title:
        title = explicit_title
        deadline = explicit_deadline or "No deadline specified"
        buffer_lead = data.get('buffer_lead', '30m prior')
    elif message:
        task_info = extract_task_from_message(message, sender=sender)
        if task_info:
            title = task_info['task_title']
            deadline = explicit_deadline or task_info['deadline_text']
            buffer_lead = task_info['buffer_lead']
        else:
            title = message[:80].strip()
            deadline = explicit_deadline or "No deadline specified"
            buffer_lead = "30m prior"
    else:
        return jsonify({'error': 'Either message or task_title must be provided.'}), 400

    try:
        task_id = add_task(
            username=username,
            task_title=title,
            deadline_text=deadline,
            buffer_lead=buffer_lead,
            status='Pending',
            reminder_offset=reminder_offset,
            source_message=message,
            sender=sender
        )
        task_obj = next((t for t in get_tasks_for_user(username) if t['id'] == task_id), {
            'id': task_id,
            'task_title': title,
            'deadline_text': deadline,
            'buffer_lead': buffer_lead,
            'reminder_offset': reminder_offset,
            'status': 'Pending'
        })
        return jsonify({
            'status': 'success',
            'message': 'Task saved successfully',
            'task_id': task_id,
            'task': task_obj
        }), 201
    except Exception as e:
        return jsonify({'error': f'Failed to quick-add task: {str(e)}'}), 500

@app.route('/quick_add_meeting', methods=['POST'])
@app.route('/api/quick_add_meeting', methods=['POST'])
def quick_add_meeting_route():
    """
    Action B: Quick Add Meeting from Android Notification or API.
    Detects and saves all valid meeting links with deduplication and reminder offset.
    """
    data = request.json or {}
    message = data.get('message', '').strip()
    sender = data.get('sender', 'Unknown')
    username = resolve_username_or_default(data.get('username'))
    explicit_url = data.get('meeting_url', '').strip()
    reminder_offset = data.get('reminder_offset', 30)

    saved = []
    if explicit_url:
        is_valid, validated_url = validate_meeting_url(explicit_url)
        if not is_valid:
            return jsonify({'error': validated_url}), 400
        platform = data.get('platform') or detect_meeting_platform(validated_url)
        sched_time = data.get('scheduled_time', 'Unknown')
        title = data.get('title') or f"{platform} Meeting"
        m_id = add_meeting(
            username=username,
            platform=platform,
            meeting_url=validated_url,
            scheduled_time=sched_time,
            title=title,
            reminder_offset=reminder_offset,
            source_message=message,
            sender=sender
        )
        m_obj = next((m for m in get_meetings_for_user(username) if m['id'] == m_id), {
            'id': m_id,
            'platform': platform,
            'meeting_url': validated_url,
            'scheduled_time': sched_time,
            'title': title,
            'reminder_offset': reminder_offset
        })
        saved.append(m_obj)
    elif message:
        links = extract_all_meeting_links(message)
        if not links:
            return jsonify({
                'status': 'no_link_found',
                'message': 'No valid meeting link found in the message.'
            }), 200
        for m in links:
            m_id = add_meeting(
                username=username,
                platform=m['platform'],
                meeting_url=m['meeting_url'],
                scheduled_time=m.get('scheduled_time', 'Unknown'),
                title=m.get('title') or f"{m['platform']} Meeting",
                reminder_offset=reminder_offset,
                source_message=message,
                sender=sender
            )
            m_obj = next((x for x in get_meetings_for_user(username) if x['id'] == m_id), {
                **m,
                'id': m_id,
                'reminder_offset': reminder_offset
            })
            saved.append(m_obj)
    else:
        return jsonify({'error': 'Either message or meeting_url must be provided.'}), 400

    return jsonify({
        'status': 'success',
        'message': f'{len(saved)} meeting link(s) saved successfully',
        'saved_count': len(saved),
        'meetings': saved
    }), 201

# --- Reminders Synchronization Endpoint ---

@app.route('/api/sync_reminders', methods=['GET'])
@app.route('/sync_reminders', methods=['GET'])
def sync_reminders_route():
    """
    Returns all upcoming active meetings and tasks that have future reminder timestamps.
    Used by Android AlarmManager and BootReceiver.
    """
    username = resolve_username_or_default(request.args.get('username') or session.get('username'))
    reminders = get_upcoming_reminders(username)
    return jsonify(reminders)

# --- Notification Ingestion & Analysis ---

@app.route('/ingest_notification', methods=['POST'])
def ingest_notification():
    data = request.json or {}
    message = data.get('message', '').strip()
    sender = data.get('sender', 'Unknown')
    app_source = data.get('app_source', 'Notification')

    msg_lower = message.lower()
    if ("data limit" in msg_lower or 
        "staying up to date" in msg_lower or 
        "storage" in msg_lower or 
        "battery" in msg_lower or 
        "screenshot saved" in msg_lower):
        return jsonify({'status': 'ignored_system_noise'}), 200

    if not message:
        return jsonify({'error': 'No message content provided'}), 400

    username = resolve_username_or_default(data.get('username'))

    all_meetings = extract_all_meeting_links(message)
    task_info = extract_task_from_message(message, sender=sender)

    saved_meetings = []
    for m in all_meetings:
        try:
            m_id = add_meeting(
                username=username,
                platform=m['platform'],
                meeting_url=m['meeting_url'],
                scheduled_time=m.get('scheduled_time', 'Unknown'),
                title=m.get('title') or f"{m['platform']} Meeting",
                reminder_offset=30,
                source_message=message,
                sender=sender
            )
            saved_meetings.append({**m, 'id': m_id})
        except Exception as e:
            print("Meeting save error:", e)

    saved_task = None
    if task_info:
        try:
            t_id = add_task(
                username=username,
                task_title=task_info['task_title'],
                deadline_text=task_info['deadline_text'],
                buffer_lead=task_info['buffer_lead'],
                status=task_info.get('status', 'Pending'),
                reminder_offset=30,
                source_message=message,
                sender=sender
            )
            saved_task = {**task_info, 'id': t_id}
        except Exception as e:
            print("Task save error:", e)

    try:
        results = analyze_message(message, persona="casual")
        priority = results.get('Priority', 'STANDARD')
        badge_color = results.get('Badge Color', '#10B981')
        urgency = results.get('Urgency', ['Medium'])[0]
        sentiment = results.get('Sentiment', ['Neutral'])[0]
    except Exception:
        priority = "STANDARD"
        badge_color = "#10B981"
        urgency = "Medium"
        sentiment = "Neutral"

    directives = [m['directive'] for m in all_meetings]

    alert_item = {
        'id': int(time.time() * 1000),
        'app_source': app_source,
        'sender': sender,
        'message': message,
        'priority': priority,
        'badge_color': badge_color,
        'directives': directives,
        'meetings': saved_meetings,
        'meeting_url': saved_meetings[0]['meeting_url'] if saved_meetings else None,
        'meeting_platform': saved_meetings[0]['platform'] if saved_meetings else None,
        'task': saved_task,
        'status_tag': "Live",
        'timestamp': time.strftime("%I:%M %p")
    }

    LIVE_FEED_ALERTS.insert(0, alert_item)
    if len(LIVE_FEED_ALERTS) > 50:
        LIVE_FEED_ALERTS.pop()

    log_analysis(
        username=username,
        message=f"[{app_source}] {sender}: {message}",
        sentiment=sentiment,
        urgency=urgency,
        priority=priority
    )

    return jsonify({
        'status': 'ingested',
        'alert': alert_item,
        'meetings_saved': len(saved_meetings),
        'task_saved': bool(saved_task),
        'delivery_mode': 'Live'
    })

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
    username = resolve_username_or_default(session.get('username'))
    data = request.json or {}
    message = data.get('message', '').strip()
    persona = data.get('persona', 'casual')
    thread_count = int(data.get('thread_count', 1))

    if not message:
        return jsonify({'error': 'No message provided'}), 400

    results = analyze_message(message, persona=persona, thread_count=thread_count)
    meeting_links = extract_all_meeting_links(message)
    task_info = extract_task_from_message(message)

    directives = [m['directive'] for m in meeting_links]
    if "Relay Directives" in results:
        results["Relay Directives"] = directives + results["Relay Directives"]

    log_analysis(
        username=username,
        message=message,
        sentiment=results['Sentiment'][0],
        urgency=results['Urgency'][0],
        priority=results['Priority']
    )

    saved_meetings = []
    for m in meeting_links:
        try:
            m_id = add_meeting(
                username=username,
                platform=m['platform'],
                meeting_url=m['meeting_url'],
                scheduled_time=m.get('scheduled_time', 'Unknown'),
                title=f"{m['platform']} Meeting",
                reminder_offset=30,
                source_message=message
            )
            saved_meetings.append({**m, 'id': m_id})
        except Exception as e:
            print("Meeting save error in analyze:", e)

    saved_task = None
    if task_info:
        try:
            t_id = add_task(
                username=username,
                task_title=task_info['task_title'],
                deadline_text=task_info['deadline_text'],
                buffer_lead=task_info['buffer_lead'],
                status='Pending',
                reminder_offset=30,
                source_message=message
            )
            saved_task = {**task_info, 'id': t_id}
        except Exception as e:
            print("Task save error in analyze:", e)

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
        'meeting_links': saved_meetings,
        'meeting_info': extract_meeting_links(message),
        'task_info': saved_task
    }
    return jsonify(formatted)

if __name__ == '__main__':
    app.run(debug=True, port=5000)