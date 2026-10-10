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
    add_task, get_tasks_for_user, update_task, delete_task, update_task_status,
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

    def test_non_emergencies_disqualified_from_p1(self):
        non_emergencies = [
            "Can you help me with this homework assignment tonight?",
            "Please call me when you have a free minute.",
            "Urgent: remember to buy milk on your way home!",
            "Help me choose which shirt to wear to dinner."
        ]
        for msg in non_emergencies:
            res = classify_message_context(msg, 'Ham', 0.95)
            self.assertNotEqual(res['priority'], "P1 - URGENT", f"Should NOT be P1: {msg}")

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

        # Check required changes
        self.assertNotIn('header-user-pill', html, "Header user chip should be removed")
        self.assertIn('id="workModeToggle"', html, "Work mode toggle should be in settingsView")
        self.assertIn('id="settingsToneSelect"', html, "Tone selector should be in settingsView")
        self.assertNotIn('allowAutoReplyToggle', html, "Autonomous auto-reply toggle should be completely removed")
        self.assertIn('id="urgentStat"', html)
        self.assertIn('id="tasksStat"', html)
        self.assertIn('id="totalStat"', html)

if __name__ == '__main__':
    unittest.main()
