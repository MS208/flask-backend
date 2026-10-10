import re
from datetime import datetime, timedelta, time
from urllib.parse import urlparse

MEETING_DOMAINS = {
    'zoom.us': 'Zoom',
    'meet.google.com': 'Google Meet',
    'teams.microsoft.com': 'MS Teams',
    'teams.live.com': 'MS Teams',
    'webex.com': 'Cisco Webex',
    'join.skype.com': 'Skype',
    'skype.com': 'Skype',
    'jit.si': 'Jitsi Meet',
    'whereby.com': 'Whereby',
    'gotomeeting.com': 'GoToMeeting',
    'gotomeet.me': 'GoToMeeting',
    'app.chime.aws': 'Amazon Chime',
    'discord.gg': 'Discord',
    'discord.com': 'Discord',
    'bluejeans.com': 'BlueJeans',
    'ringcentral.com': 'RingCentral'
}

PATH_INDICATORS = ['/meet', '/meeting', '/join', '/room', '/conf', '/conference', '/webinar', '/call']
MEETING_CONTEXT_WORDS = ['meeting', 'conference', 'video call', 'sync', 'webinar', 'zoom', 'google meet', 'teams link', 'join here', 'join call']

def detect_meeting_platform(url):
    """
    Detects the meeting provider from a given URL.
    Supports Google Meet, Zoom, MS Teams, Webex, Skype, Jitsi, and custom conference links.
    """
    if not url:
        return "Meeting Link"
    url_lower = url.lower()
    for domain, name in MEETING_DOMAINS.items():
        if domain in url_lower:
            return name
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.replace("www.", "")
        if domain:
            return domain.split('.')[0].capitalize()
    except Exception:
        pass
    return "Meeting Link"

def is_meeting_url(url, context_text=""):
    """
    Validates whether a URL is a video-conferencing or meeting link,
    checking known providers, path patterns, or contextual surrounding keywords.
    Avoids treating generic websites (like github.com or google.com search) as meeting links.
    """
    url_lower = url.lower()
    for domain in MEETING_DOMAINS:
        if domain in url_lower:
            return True
            
    # Check meeting-specific URL paths
    if any(ind in url_lower for ind in PATH_INDICATORS):
        return True
        
    # Check context words in the message
    ctx_lower = context_text.lower() if context_text else ""
    if any(mw in ctx_lower for mw in MEETING_CONTEXT_WORDS):
        return True

    return False

def validate_meeting_url(url):
    """
    Validates and normalizes a meeting URL. Preserves query parameters and invitation tokens.
    """
    if not url or not isinstance(url, str):
        return False, "Meeting link URL cannot be empty."
    url = url.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        if re.match(r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/.*)?$', url):
            url = "https://" + url
        else:
            return False, "Invalid URL format. Please include a valid URL (e.g. https://meet.google.com/xyz or https://zoom.us/j/123)."
    
    try:
        parsed = urlparse(url)
        if not parsed.netloc or "." not in parsed.netloc:
            return False, "Invalid URL format. Please provide a valid domain name."
        return True, url
    except Exception:
        return False, "Invalid URL format."

def extract_datetime_from_text(text):
    """
    Extracts time/date mentions from message text without inventing dates.
    Returns formatted date/time string, or 'Unknown' if none found.
    """
    time_date_pattern = re.compile(
        r'\b(?:at|by|before|on)?\s*'
        r'(?:(tomorrow|today|tonight|next\s+[a-z]+|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s*(?:at|by)?\s*)?'
        r'(\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM))\b'
        r'|\b(tomorrow|tonight|(?:next\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b',
        re.IGNORECASE
    )
    match = time_date_pattern.search(text)
    if match:
        matched_str = match.group(0).strip()
        cleaned = re.sub(r'^(?:at|by|before|on)\s+', '', matched_str, flags=re.IGNORECASE)
        return cleaned.capitalize()
    return "Unknown"

def parse_date_and_time(date_str, time_str=None):
    """
    Parses date and time strings into a datetime object.
    Supports relative dates ('today', 'tomorrow', 'friday') and absolute dates ('2026-10-10', '10/15/2026').
    Supports time formats ('4 PM', '16:00', '4:30pm', '11:00 AM').
    """
    now = datetime.now()
    target_date = None
    target_time = None

    combined_text = f"{date_str or ''} {time_str or ''}".strip()
    if not combined_text:
        return None

    lower_text = combined_text.lower()
    if "tomorrow" in lower_text:
        target_date = (now + timedelta(days=1)).date()
    elif "today" in lower_text or "tonight" in lower_text:
        target_date = now.date()
    else:
        weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        for idx, w in enumerate(weekdays):
            if w in lower_text:
                days_ahead = (idx - now.weekday() + 7) % 7
                if days_ahead == 0:
                    days_ahead = 7
                target_date = (now + timedelta(days=days_ahead)).date()
                break

    if not target_date:
        for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%d-%m-%Y", "%B %d, %Y", "%b %d, %Y", "%B %d", "%b %d"]:
            try:
                date_match = re.search(r'\b(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4}|\d{1,2}-\d{1,2}-\d{2,4}|[A-Za-z]{3,9}\s+\d{1,2}(?:,\s*\d{4})?)\b', combined_text)
                if date_match:
                    dt_parsed = datetime.strptime(date_match.group(0), fmt)
                    if dt_parsed.year == 1900:
                        dt_parsed = dt_parsed.replace(year=now.year)
                    target_date = dt_parsed.date()
                    break
            except Exception:
                continue

    if not target_date:
        target_date = now.date()

    time_pattern = r'\b(\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM)|\d{1,2}:\d{2})\b'
    m_time = re.search(time_pattern, combined_text)
    if m_time:
        raw_t = m_time.group(1).strip()
        for t_fmt in ["%I:%M %p", "%I:%M%p", "%I %p", "%I%p", "%H:%M"]:
            try:
                parsed_t = datetime.strptime(raw_t.upper().replace(" ", " ").strip(), t_fmt).time()
                target_time = parsed_t
                break
            except Exception:
                try:
                    parsed_t = datetime.strptime(raw_t.upper().replace(" ", "").strip(), t_fmt).time()
                    target_time = parsed_t
                    break
                except Exception:
                    continue

    if not target_time:
        return None

    return datetime.combine(target_date, target_time)

def calculate_reminder_timestamp(date_str, time_str=None, offset_minutes=30):
    """
    Calculates the exact epoch timestamp in milliseconds when a reminder alarm should trigger.
    Returns None if date/time cannot be parsed or if offset_minutes is 0 (No reminder).
    """
    if offset_minutes is None:
        offset_minutes = 30
    try:
        offset_minutes = int(offset_minutes)
    except Exception:
        offset_minutes = 30

    if offset_minutes <= 0:
        return None

    dt = parse_date_and_time(date_str, time_str)
    if not dt:
        return None
    reminder_dt = dt - timedelta(minutes=offset_minutes)
    return int(reminder_dt.timestamp() * 1000)

def extract_deadline_and_buffer(message_text):
    """
    Scans incoming text for natural language time anchors (e.g., 'by 4 PM', 'at 11:30 AM')
    and calculates a 30-minute advance alert lead time.
    """
    if not message_text:
        return None
        
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

def extract_all_meeting_links(message_text):
    """
    Multi-provider, multi-link meeting extractor.
    Extracts every valid meeting URL from a message, preserving query parameters and invitation codes.
    Returns list of meeting dicts: platform, meeting_url, scheduled_time, title, directive, reminder_offset, reminder_timestamp.
    """
    if not message_text or not isinstance(message_text, str):
        return []

    url_pattern = r'(?:https?://|www\.)[^\s<>\x22\x27{}|\\^`\[\]]+'
    found_raw = re.findall(url_pattern, message_text)
    
    standalone_provider_pattern = r'\b(?:meet\.google\.com/[a-zA-Z0-9_-]+|[\w\.-]*zoom\.us/[j|my|w]/[a-zA-Z0-9_?=&-]+)\b'
    for m in re.finditer(standalone_provider_pattern, message_text, re.IGNORECASE):
        found_raw.append(m.group(0))

    results = []
    seen = set()

    detected_time = extract_datetime_from_text(message_text)
    reminder_ts = calculate_reminder_timestamp(detected_time, offset_minutes=30) if detected_time != "Unknown" else None

    for raw in found_raw:
        cleaned = re.sub(r'[.,;!?:)]+$', '', raw.strip())
        if not cleaned:
            continue
        if cleaned.startswith('www.'):
            cleaned = 'https://' + cleaned
        elif not (cleaned.startswith('http://') or cleaned.startswith('https://')):
            cleaned = 'https://' + cleaned

        if cleaned in seen:
            continue
        seen.add(cleaned)

        if is_meeting_url(cleaned, message_text):
            platform = detect_meeting_platform(cleaned)
            results.append({
                'platform': platform,
                'meeting_url': cleaned,
                'join_url': cleaned,
                'scheduled_time': detected_time,
                'title': f'{platform} Meeting',
                'directive': f"📹 Open {platform} Link",
                'reminder_offset': 30,
                'reminder_timestamp': reminder_ts,
                'status': 'Scheduled'
            })

    return results

def extract_meeting_links(message_text):
    all_links = extract_all_meeting_links(message_text)
    if not all_links:
        return None
    primary = all_links[0]
    return {
        "platform": primary["platform"],
        "join_url": primary["meeting_url"],
        "meeting_url": primary["meeting_url"],
        "scheduled_time": primary["scheduled_time"],
        "reminder_offset": primary.get("reminder_offset", 30),
        "reminder_timestamp": primary.get("reminder_timestamp"),
        "notify_lead": "Lead alert active" if primary["scheduled_time"] != "Unknown" else "No time specified",
        "directive": primary["directive"]
    }

# --- Task / Commitment Extractor ---

TASK_TRIGGER_PATTERNS = [
    r'\b(?:remind me to|don\'t forget to|dont forget to|remember to|make sure to|be sure to|ensure to)\s+([^.!?\n]+)',
    r'\b(?:please|kindly|could you|can you|i need you to|you need to|we need to|have to|must)\s+([^.!?\n]+)',
    r'^(?:hey|hi|hello)?\s*(?:[a-zA-Z]+,\s*)?(submit|complete|finish|call|send|attend|prepare|review|update|draft|schedule|check|email|pay|buy|organize|share|verify|follow up on|write|fix|upload|deliver)\b\s+([^.!?\n]+)'
]

DEADLINE_PATTERNS = [
    r'\b(?:by|before|until|due(?:\s+on|\s+by)?|at)\s+((?:tomorrow|today|tonight|next\s+[a-z]+|monday|tuesday|wednesday|thursday|friday|saturday|sunday)(?:\s+(?:at|by)\s+\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM))?|\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM)|\d{1,2}/\d{1,2}(?:/\d{2,4})?)\b',
    r'\b(tomorrow|tonight|next\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b'
]

def extract_deadline_from_text(text):
    for pat in DEADLINE_PATTERNS:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            found = match.group(1) if match.group(1) else match.group(0)
            return found.strip().capitalize()
    return None

def extract_task_from_message(message_text, sender="Unknown"):
    """
    Inspects incoming messages for actual tasks, reminders, commitments, deadlines, and action items.
    Returns task dict with reminder timestamps or None if no action item is present.
    Does not invent deadlines when none are mentioned.
    """
    if not message_text or not isinstance(message_text, str):
        return None
    
    clean_text = message_text.strip()
    if len(clean_text) < 5:
        return None

    noise_patterns = [
        r'^(hi|hello|hey|good morning|good evening|good afternoon|thanks|thank you|ok|okay|cool|great|yes|no)\.?$',
        r'^(how are you|what\'s up|whats up|how is it going)\??$'
    ]
    for np in noise_patterns:
        if re.match(np, clean_text, re.IGNORECASE):
            return None

    is_task = False
    task_desc = ""

    for pat in TASK_TRIGGER_PATTERNS:
        m = re.search(pat, clean_text, re.IGNORECASE)
        if m:
            is_task = True
            groups = [g for g in m.groups() if g]
            if len(groups) == 2:
                task_desc = f"{groups[0]} {groups[1]}".strip()
            elif len(groups) == 1:
                task_desc = groups[0].strip()
            break

    deadline = extract_deadline_from_text(clean_text)

    if not is_task and deadline:
        action_verb_match = re.search(r'\b(submit|call|complete|finish|send|attend|prepare|review|update|draft|schedule|check|email|pay|buy|organize|share|verify|write|fix|upload|deliver|meet)\b', clean_text, re.IGNORECASE)
        if action_verb_match:
            is_task = True
            task_desc = clean_text

    if not is_task:
        return None

    if not task_desc:
        task_desc = clean_text

    clean_desc = re.sub(r'^(?:hey|hi|hello)\s+[a-zA-Z]+,\s*', '', task_desc, flags=re.IGNORECASE)
    clean_desc = re.sub(r'^(?:please|kindly)\s+', '', clean_desc, flags=re.IGNORECASE).strip()
    if clean_desc:
        clean_desc = clean_desc[0].upper() + clean_desc[1:]

    title = clean_desc[:80].strip()
    if len(clean_desc) > 80:
        title += "..."

    buffer_lead = "Alert 30 mins prior" if deadline and ("am" in deadline.lower() or "pm" in deadline.lower()) else "Standard Alert"
    reminder_ts = calculate_reminder_timestamp(deadline, offset_minutes=30) if deadline else None

    return {
        "task_title": title,
        "deadline_text": deadline if deadline else "No deadline specified",
        "buffer_lead": buffer_lead,
        "reminder_offset": 30 if deadline else 0,
        "reminder_timestamp": reminder_ts,
        "source_message": clean_text,
        "sender": sender,
        "status": "Pending"
    }

def is_within_working_hours(work_start="09:00", work_end="17:00"):
    try:
        now = datetime.now().time()
        start = datetime.strptime(work_start, "%H:%M").time()
        end = datetime.strptime(work_end, "%H:%M").time()
        return start <= now <= end
    except Exception:
        return True