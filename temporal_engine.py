import re
from datetime import datetime, timedelta

def extract_deadline_and_buffer(message_text):
    """
    Scans incoming text for natural language time anchors (e.g., 'by 4 PM', 'at 11:30 AM')
    and calculates a mandatory 30-minute advance alert lead time.
    """
    time_pattern = r'\b(?:by|at|before|until)?\s*(\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM))\b'
    match = re.search(time_pattern, message_text)
    
    if not match:
        return None

    raw_time_str = match.group(1).strip()
    
    try:
        cleaned_time = raw_time_str.upper().replace(" ", "")
        if ":" in cleaned_time:
            parsed_time = datetime.strptime(cleaned_time, "%I:%M%p")
        else:
            parsed_time = datetime.strptime(cleaned_time, "%I%p")
            
        lead_buffer_time = parsed_time - timedelta(minutes=30)
        lead_time_str = lead_buffer_time.strftime("%I:%M %p").lstrip("0")
        
        return {
            "detected_deadline": raw_time_str,
            "lead_buffer": f"Alert at {lead_time_str} (30 mins prior)",
            "is_commitment": True
        }
    except Exception:
        return {
            "detected_deadline": raw_time_str,
            "lead_buffer": "Alert 30 mins prior",
            "is_commitment": True
        }

def extract_deadline_and_alert(message_text):
    return extract_deadline_and_buffer(message_text)

def extract_meeting_links(message_text):
    """
    Regex parser that detects Zoom, Google Meet, Microsoft Teams, and Webex URLs.
    Extracts whatever dynamic time is present in the background.
    """
    meeting_patterns = [
        r'(https?://[\w\.-]*zoom\.us/[j|my]/[\w\?=&-]+)',
        r'(https?://meet\.google\.com/[a-z]{3}-[a-z]{4}-[a-z]{3}[\w\?=&-]*)',
        r'(https?://teams\.microsoft\.com/l/meetup-join/[^\s]+)',
        r'(https?://[\w\.-]*webex\.com/[^\s]+)'
    ]
    
    found_url = None
    platform = "Meeting"

    for pattern in meeting_patterns:
        match = re.search(pattern, message_text, re.IGNORECASE)
        if match:
            found_url = match.group(1).rstrip('.,;:)')
            if "zoom.us" in found_url: platform = "Zoom"
            elif "meet.google.com" in found_url: platform = "Google Meet"
            elif "teams.microsoft.com" in found_url: platform = "MS Teams"
            elif "webex.com" in found_url: platform = "Webex"
            break

    if not found_url:
        return None

    deadline_data = extract_deadline_and_buffer(message_text)
    scheduled_time = deadline_data["detected_deadline"] if deadline_data else "Scheduled"
    
    notify_lead = "Lead alert active"
    if deadline_data:
        try:
            cleaned_time = scheduled_time.upper().replace(" ", "")
            if ":" in cleaned_time:
                parsed_time = datetime.strptime(cleaned_time, "%I:%M%p")
            else:
                parsed_time = datetime.strptime(cleaned_time, "%I%p")
            notify_time = parsed_time - timedelta(minutes=10)
            notify_lead = f"Alert scheduled for {notify_time.strftime('%I:%M %p').lstrip('0')}"
        except:
            pass

    return {
        "platform": platform,
        "join_url": found_url,
        "scheduled_time": scheduled_time,
        "notify_lead": notify_lead,
        "directive": f"📹 Join {platform} Call"
    }

def is_within_working_hours(work_start="09:00", work_end="17:00"):
    """
    Checks if current local time falls within the user's active custom work window.
    """
    try:
        now = datetime.now().time()
        start = datetime.strptime(work_start, "%H:%M").time()
        end = datetime.strptime(work_end, "%H:%M").time()
        return start <= now <= end
    except Exception:
        return True