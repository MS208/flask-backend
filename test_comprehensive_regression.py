import os
import sys
import unittest
import json
from datetime import datetime, timedelta

# Ensure Mishthi directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from predict import classify_message_context
from app import extract_all_meeting_links, extract_meeting_links, extract_task_from_message
from database import (
    init_db, get_db, add_meeting, get_meetings_for_user, update_meeting, delete_meeting,
    update_meeting_status, add_task, get_tasks_for_user, update_task, delete_task,
    update_task_status, get_upcoming_reminders,
    log_analysis, get_p1_alert_count, get_total_message_count, get_quarantined_messages
)
import app as flask_app_module

class TestComprehensiveRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()
        flask_app_module.app.config['TESTING'] = True
        flask_app_module.app.config['SECRET_KEY'] = 'test-secret-key'
        cls.client = flask_app_module.app.test_client()

    # ==========================================
    # 1. MESSAGE CLASSIFICATION OVERHAUL TESTS
    # ==========================================
    def test_p1_urgent_life_emergencies(self):
        life_msgs = [
            "Emergency! Car accident on highway, need an ambulance immediately!",
            "Dad had a stroke and is in the ICU at city hospital.",
            "House is on fire, calling emergency services right now!"
        ]
        for msg in life_msgs:
            res = classify_message_context(msg, 'Ham', 0.95)
            self.assertEqual(res['priority'], "P1 - URGENT", f"Expected P1 for: {msg}")
            self.assertEqual(res['badge_color'], "#ef4444")
            self.assertEqual(res['classification'], "Urgent")

    def test_p1_urgent_infra_emergencies(self):
        infra_msgs = [
            "Production server is down, database corrupted after deployment!",
            "Critical security breach detected, credentials compromised in prod cluster.",
            "The main database corrupted, customer data unreachable, severe outage!"
        ]
        for msg in infra_msgs:
            res = classify_message_context(msg, 'Ham', 0.95)
            self.assertEqual(res['priority'], "P1 - URGENT", f"Expected P1 for: {msg}")
            self.assertEqual(res['badge_color'], "#ef4444")
            self.assertEqual(res['classification'], "Urgent")

    def test_expanded_p1_urgent_expressions(self):
        expanded_p1_msgs = [
            "Emergency!",
            "It's an emergency, someone is hurt!",
            "Help me immediately, I'm in danger",
            "Call me now",
            "Please respond ASAP",
            "Need help urgently",
            "Can't breathe",
            "SOS! Send an ambulance"
        ]
        for msg in expanded_p1_msgs:
            res = classify_message_context(msg, 'Ham', 0.95)
            self.assertEqual(res['priority'], "P1 - URGENT", f"Expected P1 for: {msg}")
            self.assertEqual(res['classification'], "Urgent")

    def test_expanded_p2_important_expressions(self):
        expanded_p2_msgs = [
            "We need to talk about the project.",
            "u need to see that new design update",
            "Don't forget the sprint review tomorrow at 3pm",
            "Deadline tomorrow for the marketing deck",
            "pls check the attached invoice and confirm",
            "ur report is due before 5pm today"
        ]
        for msg in expanded_p2_msgs:
            res = classify_message_context(msg, 'Ham', 0.95)
            self.assertEqual(res['priority'], "P2 - IMPORTANT", f"Expected P2 for: {msg}")
            self.assertEqual(res['classification'], "Important")

    def test_non_emergencies_disqualified_from_p1(self):
        non_emergencies = [
            "Can you help me with this homework assignment tonight?",
            "Please call me when you have a free minute.",
            "Not urgent: remember to buy milk on your way home!",
            "Help me choose which shirt to wear to dinner."
        ]
        for msg in non_emergencies:
            res = classify_message_context(msg, 'Ham', 0.95)
            self.assertNotEqual(res['priority'], "P1 - URGENT", f"Should NOT be P1: {msg}")

    def test_word_first_p1_urgent_detection(self):
        # Test individual words, distress cries, capitalization, spelling variations, and longer messages
        p1_test_cases = [
            "Help",
            "HELP!",
            "hlp",
            "I need help",
            "Need help",
            "SOS",
            "s.o.s",
            "URGENT",
            "Urgent: please reply immediately",
            "Accident on highway",
            "He is bleeding",
            "Trapped inside the room",
            "Danger, please stay back",
            "Unable to breathe, calling an ambulance",
            "Car crash near the signal",
            "Building is on fire",
            "Server down, critical outage",
            "Help me, someone is hurt"
        ]
        for msg in p1_test_cases:
            res = classify_message_context(msg, 'Ham', 0.95)
            self.assertEqual(res['priority'], "P1 - URGENT", f"Expected P1 for: {msg}")
            self.assertEqual(res['badge_color'], "#ef4444")
            self.assertEqual(res['classification'], "Urgent")

    def test_p1_overrides_p2_conflicting_signals(self):
        # When both P1 (emergency/urgent) and P2 (meeting/deadline/review) appear, P1 takes absolute priority
        conflicting_cases = [
            "Urgent: review the contract, there was an accident",
            "I need help with this emergency before the meeting",
            "SOS! Deadline tomorrow but server down in prod",
            "Help! Check the status of the injured driver"
        ]
        for msg in conflicting_cases:
            res = classify_message_context(msg, 'Ham', 0.95)
            self.assertEqual(res['priority'], "P1 - URGENT", f"Expected P1 override for: {msg}")

    def test_contextual_p2_and_standard_differentiation(self):
        # 'Need to talk' and 'u need to see this' are P2
        p2_cases = [
            "Need to talk",
            "We need to talk",
            "u need to see this",
            "you need to check this",
            "Please confirm your attendance",
            "Remember to submit your timesheet",
            "Action required: sign document"
        ]
        for msg in p2_cases:
            res = classify_message_context(msg, 'Ham', 0.95)
            self.assertEqual(res['priority'], "P2 - IMPORTANT", f"Expected P2 for: {msg}")

        # Ordinary everyday conversation should be STANDARD (silence, shade only)
        standard_cases = [
            "I need a coffee",
            "I need some sleep",
            "Call me when you get free",
            "Did you check the score?",
            "Hey how was your day?",
            "Help me buy groceries",
            "Help me pick an outfit",
            "Not urgent, just checking in",
            "Coffee emergency haha"
        ]
        for msg in standard_cases:
            res = classify_message_context(msg, 'Ham', 0.95)
            self.assertEqual(res['priority'], "STANDARD", f"Expected STANDARD for: {msg}")
            self.assertEqual(res['badge_color'], "#10b981")
            self.assertEqual(res['classification'], "Standard")

    def test_ingest_notification_end_to_end_p1_classification(self):
        # Test full API pipeline: /ingest_notification with 'Help' and 'I need help'
        res1 = self.client.post('/ingest_notification', json={
            'app_source': 'WhatsApp',
            'sender': 'Alice',
            'message': 'Help',
            'username': 'admin'
        })
        self.assertEqual(res1.status_code, 200)
        feed1 = self.client.get('/get_feed')
        self.assertEqual(feed1.status_code, 200)
        items1 = feed1.get_json()['feed']
        help_alert = next((a for a in items1 if a['message'] == 'Help'), None)
        self.assertIsNotNone(help_alert)
        self.assertEqual(help_alert['priority'], "P1 - URGENT")
        self.assertEqual(help_alert['badge_color'], "#ef4444")

        res2 = self.client.post('/ingest_notification', json={
            'app_source': 'SMS',
            'sender': 'Bob',
            'message': 'I need help',
            'username': 'admin'
        })
        self.assertEqual(res2.status_code, 200)
        feed2 = self.client.get('/get_feed')
        items2 = feed2.get_json()['feed']
        need_help_alert = next((a for a in items2 if a['message'] == 'I need help'), None)
        self.assertIsNotNone(need_help_alert)
        self.assertEqual(need_help_alert['priority'], "P1 - URGENT")
        self.assertEqual(need_help_alert['badge_color'], "#ef4444")

    def test_p2_important_work_and_meetings(self):
        important_msgs = [
            "Please review the PR and client contract before the 3pm deadline.",
            "Client presentation scheduled for tomorrow, submit deliverables by 5pm.",
            "Join our sprint sync at https://meet.google.com/abc-defg-hij",
            "Zoom architecture review meeting https://zoom.us/j/9876543210 tomorrow at 10am"
        ]
        for msg in important_msgs:
            res = classify_message_context(msg, 'Ham', 0.95)
            self.assertEqual(res['priority'], "P2 - IMPORTANT", f"Expected P2 for: {msg}")
            self.assertEqual(res['badge_color'], "#f59e0b")
            self.assertEqual(res['classification'], "Important")

    def test_spam_quarantine_classification(self):
        spam_msgs = [
            "Congratulations! You won $10,000 cash in the state lottery! Claim your prize now.",
            "Pre-approved loan for $50,000 with 0% interest, click here to claim funds immediately.",
            "Dear beneficiary, wire transfer of $5,000,000 waiting for your bank details."
        ]
        for msg in spam_msgs:
            res = classify_message_context(msg, 'Spam', 0.99)
            self.assertEqual(res['priority'], "LOW (SPAM)", f"Expected Spam for: {msg}")
            self.assertEqual(res['badge_color'], "#94a3b8")
            self.assertEqual(res['classification'], "Spam")

    def test_standard_casual_chat(self):
        standard_msgs = [
            "Hey, how are you doing today?",
            "Good morning! Hope you have a wonderful weekend.",
            "Are you free for coffee later this afternoon?"
        ]
        for msg in standard_msgs:
            res = classify_message_context(msg, 'Ham', 0.95)
            self.assertEqual(res['priority'], "STANDARD", f"Expected Standard for: {msg}")
            self.assertEqual(res['badge_color'], "#10b981")
            self.assertEqual(res['classification'], "Standard")

    # ==========================================
    # 2. MEETING LINKS EXTRACTION & CRUD TESTS
    # ==========================================
    def test_meeting_links_extraction(self):
        text = "Join our product sync https://meet.google.com/abc-defg-hij and backup https://zoom.us/j/123456789"
        meetings = extract_all_meeting_links(text)
        self.assertEqual(len(meetings), 2)
        self.assertEqual(meetings[0]['platform'], "Google Meet")
        self.assertEqual(meetings[1]['platform'], "Zoom")

    def test_meeting_crud_persistence(self):
        user = "test_user_meetings"
        # 1. Add
        m_id = add_meeting(
            username=user,
            platform="Google Meet",
            meeting_url="https://meet.google.com/xyz-uvw-rst",
            title="Sprint Planning",
            meeting_date="2026-10-15",
            start_time="10:00",
            reminder_offset=30
        )
        self.assertIsNotNone(m_id)

        # 2. Retrieve
        meetings = get_meetings_for_user(user)
        self.assertTrue(any(m['id'] == m_id for m in meetings))
        saved = next(m for m in meetings if m['id'] == m_id)
        self.assertEqual(saved['title'], "Sprint Planning")
        self.assertEqual(saved['platform'], "Google Meet")

        # 3. Update
        updated = update_meeting(
            meeting_id=m_id,
            username=user,
            platform="Google Meet",
            meeting_url="https://meet.google.com/xyz-uvw-rst",
            title="Sprint Planning - Revised",
            meeting_date="2026-10-15",
            start_time="11:00",
            reminder_offset=15,
            status="Scheduled"
        )
        self.assertTrue(updated)
        meetings = get_meetings_for_user(user)
        saved = next(m for m in meetings if m['id'] == m_id)
        self.assertEqual(saved['title'], "Sprint Planning - Revised")
        self.assertEqual(saved['start_time'], "11:00")
        self.assertEqual(saved['reminder_offset'], 15)

        # 4. Delete
        deleted = delete_meeting(m_id, user)
        self.assertTrue(deleted)
        meetings = get_meetings_for_user(user)
        self.assertFalse(any(m['id'] == m_id for m in meetings))

    # ==========================================
    # 3. TASKS CRUD & COMMITMENT COUNT TESTS
    # ==========================================
    def test_tasks_crud_persistence(self):
        user = "test_user_tasks"
        # 1. Add
        t_id = add_task(
            username=user,
            task_title="Deliver Quarterly Budget Report",
            due_date="2026-10-20",
            due_time="17:00",
            description="Submit financial sheet to executive committee",
            reminder_offset=30,
            status="Pending"
        )
        self.assertIsNotNone(t_id)

        # 2. Retrieve
        tasks = get_tasks_for_user(user)
        self.assertTrue(any(t['id'] == t_id for t in tasks))
        saved = next(t for t in tasks if t['id'] == t_id)
        self.assertEqual(saved['task_title'], "Deliver Quarterly Budget Report")

        # 3. Update
        updated = update_task(
            task_id=t_id,
            username=user,
            task_title="Deliver Final Budget Report",
            due_date="2026-10-21",
            due_time="18:00",
            description="Updated version",
            reminder_offset=15,
            status="Pending"
        )
        self.assertTrue(updated)

        # 4. Update status
        st_updated = update_task_status(t_id, user, "Completed")
        self.assertTrue(st_updated)
        tasks = get_tasks_for_user(user)
        saved = next(t for t in tasks if t['id'] == t_id)
        self.assertEqual(saved['status'], "Completed")

        # 5. Delete
        del_ok = delete_task(t_id, user)
        self.assertTrue(del_ok)
        tasks = get_tasks_for_user(user)
        self.assertFalse(any(t['id'] == t_id for t in tasks))

    # ==========================================
    # 4. INGESTION, FEED, & QUARANTINE ROUTE TESTS
    # ==========================================
    def test_ingest_p1_alert_and_feed_counter(self):
        payload = {
            "app_source": "WhatsApp",
            "sender": "DevOps Lead",
            "message": "Production server is down, database corrupted after latest deployment! Immediate emergency!",
            "username": "admin"
        }
        res = self.client.post('/ingest_notification', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data['classification'], "Urgent")
        self.assertEqual(data['priority'], "P1 - URGENT")
        self.assertEqual(data['delivery_mode'], "Live")

        # Check /get_feed
        feed_res = self.client.get('/get_feed')
        self.assertEqual(feed_res.status_code, 200)
        feed_data = feed_res.get_json()
        self.assertGreaterEqual(feed_data['p1_count'], 1)
        self.assertGreaterEqual(feed_data['total_read'], 1)

    def test_ingest_spam_to_quarantine(self):
        payload = {
            "app_source": "SMS",
            "sender": "1800-PRIZE",
            "message": "Congratulations! You won $10,000 cash in the international lottery! Click here to claim your prize now.",
            "username": "admin"
        }
        res = self.client.post('/ingest_notification', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data['classification'], "Spam")
        self.assertEqual(data['delivery_mode'], "Quarantine")

        # Check /get_quarantine
        q_res = self.client.get('/get_quarantine')
        self.assertEqual(q_res.status_code, 200)
        q_data = q_res.get_json()
        self.assertGreaterEqual(len(q_data['quarantine']), 1)

    # ==========================================
    # 5. SETTINGS ENDPOINTS & PERSISTENCE
    # ==========================================
    def test_settings_persistence(self):
        # Update settings with work mode enabled and formal persona
        payload = {
            "active_hours_enabled": True,
            "work_start_time": "08:30",
            "work_end_time": "18:00",
            "work_mode_enabled": True,
            "dnd_bypass_enabled": True,
            "persona": "formal"
        }
        with self.client.session_transaction() as sess:
            sess['logged_in'] = True
            sess['username'] = 'admin'

        res = self.client.post('/update_settings', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(res.status_code, 200)

        # Get settings
        get_res = self.client.get('/get_settings')
        self.assertEqual(get_res.status_code, 200)
        settings = get_res.get_json()
        self.assertEqual(settings['work_start_time'], "08:30")
        self.assertEqual(settings['work_end_time'], "18:00")
        self.assertEqual(settings['work_mode_enabled'], 1)
        self.assertEqual(settings['default_persona'], "formal")

    # ==========================================
    # 6. TEMPLATE RENDERING & STRUCTURE TESTS
    # ==========================================
    def test_dashboard_template_renders_properly(self):
        with self.client.session_transaction() as sess:
            sess['logged_in'] = True
            sess['username'] = 'admin'
            sess['full_name'] = 'Administrator'
            sess['email'] = 'admin@gateway.com'

        res = self.client.get('/')
        self.assertEqual(res.status_code, 200)
        html = res.data.decode('utf-8')

        # Check key structural sections
        self.assertIn('id="topNavBar"', html)
        self.assertIn('id="threeDotsBtn"', html)
        self.assertIn('id="quickDropdownMenu"', html)
        self.assertIn('id="overviewView"', html)
        self.assertIn('id="meetingsView"', html)
        self.assertIn('id="tasksView"', html)
        self.assertIn('id="settingsView"', html)
        self.assertIn('id="profileView"', html)
        self.assertIn('id="assistantModal"', html)
        self.assertIn('id="questionnaireModal"', html)

    # ==========================================
    # 7. USER PROFILE & PREFERENCES TESTS
    # ==========================================
    def test_user_profile_crud(self):
        with self.client.session_transaction() as sess:
            sess['logged_in'] = True
            sess['username'] = 'admin'

        # Get initial profile
        res = self.client.get('/get_profile')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['username'], 'admin')

        # Update profile
        update_res = self.client.post('/update_profile', data=json.dumps({
            'full_name': 'Chief System Architect',
            'email': 'architect@gateway.com'
        }), content_type='application/json')
        self.assertEqual(update_res.status_code, 200)

        # Verify profile updated
        verify_res = self.client.get('/get_profile')
        self.assertEqual(verify_res.status_code, 200)
        updated = verify_res.get_json()
        self.assertEqual(updated['full_name'], 'Chief System Architect')
        self.assertEqual(updated['email'], 'architect@gateway.com')

    def test_message_preferences_persistence(self):
        with self.client.session_transaction() as sess:
            sess['logged_in'] = True
            sess['username'] = 'admin'

        payload = {
            "active_hours_enabled": True,
            "work_start_time": "09:00",
            "work_end_time": "17:00",
            "work_mode_enabled": True,
            "dnd_bypass_enabled": True,
            "persona": "short",
            "custom_topics": "crypto, bank otp, sprint deliverables",
            "handling_stage": "staged",
            "spam_handling": "archive"
        }
        res = self.client.post('/update_settings', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(res.status_code, 200)

        # Fetch settings and verify all preferences are persistent
        get_res = self.client.get('/get_settings')
        self.assertEqual(get_res.status_code, 200)
        data = get_res.get_json()
        self.assertEqual(data['custom_topics'], "crypto, bank otp, sprint deliverables")
        self.assertEqual(data['handling_stage'], "staged")
        self.assertEqual(data['spam_handling'], "archive")
        self.assertEqual(data['default_persona'], "short")
        self.assertEqual(data['preferences_configured'], 1)

    def test_auto_save_and_deduplication_in_analyze(self):
        with self.client.session_transaction() as sess:
            sess['logged_in'] = True
            sess['username'] = 'admin'

        msg = "Team sprint sync meeting today at 4pm at https://meet.google.com/prj-sync-2026. Please submit final report before tomorrow 5pm."

        # First analyze call: should auto-extract and auto-save
        res1 = self.client.post('/analyze', data=json.dumps({'message': msg}), content_type='application/json')
        self.assertEqual(res1.status_code, 200)
        data1 = res1.get_json()
        self.assertGreaterEqual(len(data1['meeting_links']), 1)
        self.assertIsNotNone(data1['task_info'])

        # Check meetings list
        m_res = self.client.get('/get_meetings')
        self.assertEqual(m_res.status_code, 200)
        meetings = [m for m in m_res.get_json()['meetings'] if 'prj-sync-2026' in m['meeting_url']]
        self.assertEqual(len(meetings), 1)

        # Second analyze call with exact same message: must NOT duplicate
        res2 = self.client.post('/analyze', data=json.dumps({'message': msg}), content_type='application/json')
        self.assertEqual(res2.status_code, 200)

        m_res2 = self.client.get('/get_meetings')
        self.assertEqual(m_res2.status_code, 200)
        meetings2 = [m for m in m_res2.get_json()['meetings'] if 'prj-sync-2026' in m['meeting_url']]
        self.assertEqual(len(meetings2), 1, "Meeting should not be duplicated")

    def test_active_persona_and_thread_turns_removed_from_ui(self):
        with self.client.session_transaction() as sess:
            sess['logged_in'] = True
            sess['username'] = 'admin'

        res = self.client.get('/')
        self.assertEqual(res.status_code, 200)
        html = res.data.decode('utf-8')

        # Active Persona selector and Thread Turns input should be removed
        self.assertNotIn('id="userPersona"', html, "userPersona selector should be removed from assistant")
        self.assertNotIn('id="threadCountInput"', html, "threadCountInput should be removed from assistant")

        # Editable profile inputs should be present
        self.assertIn('id="profileFullName"', html, "profileFullName input should be in profileView")
        self.assertIn('id="profileEmail"', html, "profileEmail input should be in profileView")

    def test_tone_settings_influences_analyze_directives(self):
        with self.client.session_transaction() as sess:
            sess['logged_in'] = True
            sess['username'] = 'admin'

        # Set tone to formal
        self.client.post('/update_settings', data=json.dumps({"persona": "formal"}), content_type='application/json')
        res_formal = self.client.post('/analyze', data=json.dumps({'message': 'Can you share the project status report?'}), content_type='application/json')
        directives_formal = res_formal.get_json()['relay_directives']
        self.assertTrue(any("review" in d.lower() or "deliverable" in d.lower() or "acknowledged" in d.lower() for d in directives_formal), "Expected formal phrasing in directives")

        # Set tone to short
        self.client.post('/update_settings', data=json.dumps({"persona": "short"}), content_type='application/json')
        res_short = self.client.post('/analyze', data=json.dumps({'message': 'Can you share the project status report?'}), content_type='application/json')
        directives_short = res_short.get_json()['relay_directives']
        self.assertTrue(any("sending shortly" in d.lower() or "done" in d.lower() or "working on it" in d.lower() for d in directives_short), "Expected short phrasing in directives")

    # ==========================================
    # 7. MEETING AND TASK DELETION TESTS
    # ==========================================
    def test_delete_meeting_functionality(self):
        with self.client.session_transaction() as sess:
            sess['logged_in'] = True
            sess['username'] = 'admin'

        # 1. Create two test meetings
        m1_id = add_meeting("admin", "Google Meet", "https://meet.google.com/del-test-1", "Tomorrow 10am", "Del Test 1")
        m2_id = add_meeting("admin", "Google Meet", "https://meet.google.com/del-test-2", "Tomorrow 11am", "Del Test 2")
        self.assertIsNotNone(m1_id)
        self.assertIsNotNone(m2_id)

        # 2. Delete m1 via POST /delete_meeting with json {"id": m1_id}
        res1 = self.client.post('/delete_meeting', data=json.dumps({'id': m1_id}), content_type='application/json')
        self.assertEqual(res1.status_code, 200)
        self.assertEqual(res1.get_json()['status'], 'success')

        # Verify m1 is deleted and m2 is still present
        meetings_after = get_meetings_for_user('admin')
        meeting_ids = [m['id'] for m in meetings_after]
        self.assertNotIn(m1_id, meeting_ids, "Meeting 1 should be deleted")
        self.assertIn(m2_id, meeting_ids, "Meeting 2 should still exist")

        # 3. Delete m2 via DELETE /api/delete_meeting with query params ?meeting_id=...
        res2 = self.client.delete(f'/api/delete_meeting?meeting_id={m2_id}')
        self.assertEqual(res2.status_code, 200)
        self.assertEqual(res2.get_json()['status'], 'success')

        # Verify m2 is also deleted
        meetings_after2 = get_meetings_for_user('admin')
        meeting_ids2 = [m['id'] for m in meetings_after2]
        self.assertNotIn(m2_id, meeting_ids2, "Meeting 2 should be deleted")

        # 4. Deleting non-existent meeting returns 404
        res_nonexistent = self.client.post('/delete_meeting', data=json.dumps({'id': 99999999}), content_type='application/json')
        self.assertEqual(res_nonexistent.status_code, 404)

        # 5. Missing or invalid ID returns 400
        res_bad = self.client.post('/delete_meeting', data=json.dumps({'id': 'invalid'}), content_type='application/json')
        self.assertEqual(res_bad.status_code, 400)

    def test_delete_task_functionality(self):
        with self.client.session_transaction() as sess:
            sess['logged_in'] = True
            sess['username'] = 'admin'

        # 1. Create two test tasks
        t1_id = add_task("admin", "Del Task 1", "Tomorrow 2pm")
        t2_id = add_task("admin", "Del Task 2", "Tomorrow 3pm")
        self.assertIsNotNone(t1_id)
        self.assertIsNotNone(t2_id)

        # 2. Delete t1 via POST /delete_task with json {"task_id": t1_id}
        res1 = self.client.post('/delete_task', data=json.dumps({'task_id': t1_id}), content_type='application/json')
        self.assertEqual(res1.status_code, 200)
        self.assertEqual(res1.get_json()['status'], 'success')

        # Verify t1 is deleted and t2 is still present
        tasks_after = get_tasks_for_user('admin')
        task_ids = [t['id'] for t in tasks_after]
        self.assertNotIn(t1_id, task_ids, "Task 1 should be deleted")
        self.assertIn(t2_id, task_ids, "Task 2 should still exist")

        # 3. Delete t2 via DELETE /api/delete_task with form data taskId=...
        res2 = self.client.delete('/api/delete_task', data={'taskId': str(t2_id)})
        self.assertEqual(res2.status_code, 200)
        self.assertEqual(res2.get_json()['status'], 'success')

        # Verify t2 is also deleted
        tasks_after2 = get_tasks_for_user('admin')
        task_ids2 = [t['id'] for t in tasks_after2]
        self.assertNotIn(t2_id, task_ids2, "Task 2 should be deleted")

        # 4. Deleting non-existent task returns 404
        res_nonexistent = self.client.post('/delete_task', data=json.dumps({'task_id': 99999999}), content_type='application/json')
        self.assertEqual(res_nonexistent.status_code, 404)

        # 5. Missing or invalid ID returns 400
        res_bad = self.client.post('/delete_task', data=json.dumps({}), content_type='application/json')
        self.assertEqual(res_bad.status_code, 400)

    # ==========================================
    # 8. MEETING & TASK LINKAGE, BI-DIRECTIONAL SYNC & DEDUPLICATION
    # ==========================================
    def test_meeting_auto_generates_linked_upcoming_task(self):
        user = "test_link_user"
        m_id = add_meeting(
            username=user,
            platform="Google Meet",
            meeting_url="https://meet.google.com/link-sync-test",
            scheduled_time="2026-10-25 15:00",
            title="Q3 Strategy Review",
            meeting_date="2026-10-25",
            start_time="15:00",
            reminder_offset=15
        )
        self.assertIsNotNone(m_id)

        # Verify meeting has linked_task_id
        meetings = get_meetings_for_user(user)
        meeting = next((m for m in meetings if m['id'] == m_id), None)
        self.assertIsNotNone(meeting)
        linked_tid = meeting.get('linked_task_id')
        self.assertIsNotNone(linked_tid)

        # Verify task exists in upcoming tasks with matching fields
        tasks = get_tasks_for_user(user)
        task = next((t for t in tasks if t['id'] == linked_tid), None)
        self.assertIsNotNone(task)
        self.assertEqual(task['task_title'], "Q3 Strategy Review")
        self.assertEqual(task['due_date'], "2026-10-25")
        self.assertEqual(task['due_time'], "15:00")
        self.assertEqual(task['status'], "Pending")
        self.assertEqual(task['linked_meeting_id'], m_id)

        # Clean up
        delete_meeting(m_id, user)

    def test_update_meeting_syncs_to_linked_task(self):
        user = "test_link_update_user"
        m_id = add_meeting(
            username=user,
            platform="Zoom",
            meeting_url="https://zoom.us/j/99887766",
            scheduled_time="2026-10-26 10:00",
            title="Design Sync",
            meeting_date="2026-10-26",
            start_time="10:00",
            reminder_offset=30
        )
        meetings = get_meetings_for_user(user)
        m = next(m for m in meetings if m['id'] == m_id)
        linked_tid = m['linked_task_id']

        # Update meeting details
        update_meeting(
            meeting_id=m_id,
            username=user,
            platform="Zoom",
            meeting_url="https://zoom.us/j/99887766",
            scheduled_time="2026-10-26 11:30",
            title="Updated Design Sync",
            meeting_date="2026-10-26",
            start_time="11:30",
            reminder_offset=10
        )

        tasks = get_tasks_for_user(user)
        task = next(t for t in tasks if t['id'] == linked_tid)
        self.assertEqual(task['task_title'], "Updated Design Sync")
        self.assertEqual(task['due_time'], "11:30")
        self.assertEqual(task['reminder_offset'], 10)

        # Clean up
        delete_meeting(m_id, user)

    def test_bidirectional_completion_status_sync(self):
        user = "test_sync_status_user"
        m_id = add_meeting(
            username=user,
            platform="Teams",
            meeting_url="https://teams.microsoft.com/l/meetup-join/123",
            title="All Hands",
            meeting_date="2026-10-27",
            start_time="09:00"
        )
        m = next(m for m in get_meetings_for_user(user) if m['id'] == m_id)
        t_id = m['linked_task_id']

        # 1. Complete meeting -> linked task should become Completed
        update_meeting_status(m_id, user, "Completed")
        m_after = next(m for m in get_meetings_for_user(user) if m['id'] == m_id)
        t_after = next(t for t in get_tasks_for_user(user) if t['id'] == t_id)
        self.assertEqual(m_after['status'], "Completed")
        self.assertEqual(t_after['status'], "Completed")

        # 2. Reopen task -> linked meeting should become Scheduled
        update_task_status(t_id, user, "Pending")
        m_reopened = next(m for m in get_meetings_for_user(user) if m['id'] == m_id)
        t_reopened = next(t for t in get_tasks_for_user(user) if t['id'] == t_id)
        self.assertEqual(m_reopened['status'], "Scheduled")
        self.assertEqual(t_reopened['status'], "Pending")

        # Clean up
        delete_meeting(m_id, user)

    def test_deleting_meeting_removes_linked_task(self):
        user = "test_del_link_user"
        m_id = add_meeting(
            username=user,
            platform="Google Meet",
            meeting_url="https://meet.google.com/del-cascade-test",
            title="Temporary Meeting"
        )
        m = next(m for m in get_meetings_for_user(user) if m['id'] == m_id)
        t_id = m['linked_task_id']

        # Delete meeting
        delete_meeting(m_id, user)

        # Both meeting and linked task must be gone
        meetings = [m['id'] for m in get_meetings_for_user(user)]
        tasks = [t['id'] for t in get_tasks_for_user(user)]
        self.assertNotIn(m_id, meetings)
        self.assertNotIn(t_id, tasks)

    def test_deleting_linked_task_preserves_original_meeting(self):
        user = "test_del_task_only_user"
        m_id = add_meeting(
            username=user,
            platform="Google Meet",
            meeting_url="https://meet.google.com/del-task-only",
            title="Preserved Meeting"
        )
        m = next(m for m in get_meetings_for_user(user) if m['id'] == m_id)
        t_id = m['linked_task_id']

        # Delete task only
        delete_task(t_id, user)

        # Task should be gone, but meeting must remain with linked_task_id cleared
        tasks = [t['id'] for t in get_tasks_for_user(user)]
        self.assertNotIn(t_id, tasks)

        meetings = get_meetings_for_user(user)
        m_still_there = next((m for m in meetings if m['id'] == m_id), None)
        self.assertIsNotNone(m_still_there)
        self.assertIsNone(m_still_there.get('linked_task_id'))

        # Clean up
        delete_meeting(m_id, user)

    def test_upcoming_reminders_deduplication_and_completion_exclusion(self):
        user = "test_reminders_dedup_user"
        # Create a meeting with reminder 60 minutes in future
        future_dt = datetime.now() + timedelta(minutes=60)
        mtg_time = future_dt.strftime("%H:%M")
        mtg_date = future_dt.strftime("%Y-%m-%d")
        m_id = add_meeting(
            username=user,
            platform="Google Meet",
            meeting_url="https://meet.google.com/alarm-test-1",
            title="Alarm Dedup Meeting",
            meeting_date=mtg_date,
            start_time=mtg_time,
            reminder_offset=15
        )

        reminders_data = get_upcoming_reminders(user)
        meetings = reminders_data.get('meetings', [])
        tasks = reminders_data.get('tasks', [])

        # Should have meeting alarm
        meeting_reminders = [r for r in meetings if r.get('id') == m_id]
        self.assertEqual(len(meeting_reminders), 1, "Meeting reminder should be present")

        # Auto-generated linked task should NOT produce a duplicate alarm in tasks
        linked_tasks = [t for t in tasks if t.get('task_title') == 'Alarm Dedup Meeting']
        self.assertEqual(len(linked_tasks), 0, "Linked task duplicate reminder must be suppressed")

        # Now mark meeting as Completed
        update_meeting_status(m_id, user, "Completed")
        reminders_after_complete = get_upcoming_reminders(user)
        meetings_after = [r for r in reminders_after_complete.get('meetings', []) if r.get('id') == m_id]
        self.assertEqual(len(meetings_after), 0, "Completed items must be excluded from active reminders")

        # Clean up
        delete_meeting(m_id, user)

    # ==========================================
    # 9. TONE & BOUNDARIES PERSISTENCE TESTS
    # ==========================================
    def test_tone_and_boundaries_persistence(self):
        with self.client.session_transaction() as sess:
            sess['logged_in'] = True
            sess['username'] = 'admin'

        payload = {
            'active_hours_enabled': 1,
            'work_start_time': '08:30',
            'work_end_time': '18:15',
            'work_mode_enabled': 1,
            'dnd_bypass_enabled': 1,
            'persona': 'professional',
            'custom_topics': 'Server Outage, Client Pitch, Budget Approval',
            'handling_stage': 'immediate',
            'spam_handling': 'digest'
        }
        res = self.client.post('/update_settings', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(res.status_code, 200)

        # Verify via GET /get_settings
        res_get = self.client.get('/get_settings')
        self.assertEqual(res_get.status_code, 200)
        data = res_get.get_json()
        self.assertEqual(data.get('work_start_time'), '08:30')
        self.assertEqual(data.get('work_end_time'), '18:15')
        self.assertEqual(data.get('work_mode_enabled'), 1)
        self.assertEqual(data.get('dnd_bypass_enabled'), 1)
        self.assertEqual(data.get('custom_topics'), 'Server Outage, Client Pitch, Budget Approval')
        self.assertEqual(data.get('handling_stage'), 'immediate')
        self.assertEqual(data.get('spam_handling'), 'digest')

if __name__ == '__main__':
    unittest.main()


