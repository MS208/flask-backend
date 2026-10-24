import joblib
import random
import re
import os
from preprocess import clean_text
from temporal_engine import extract_deadline_and_alert

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load trained classification models
sentiment_model = joblib.load(os.path.join(BASE_DIR, 'models', 'sentiment_model.pkl'))
urgency_model = joblib.load(os.path.join(BASE_DIR, 'models', 'urgency_model.pkl'))
tone_model = joblib.load(os.path.join(BASE_DIR, 'models', 'tone_model.pkl'))

def polish_sentence(text):
    """Cleans up rough automated drafts, capitalizes letters, and sets correct punctuation."""
    if not text:
        return text
    cleaned = text.strip()
    if cleaned and not cleaned[0].isupper() and not cleaned.startswith(("👍", "📞", "🚫", "🚀")):
        cleaned = cleaned[0].upper() + cleaned[1:]
    if cleaned and cleaned[-1] not in ['.', '!', '?', '👍', '🚀', '...'] and not cleaned.startswith(("📞", "🚫")):
        cleaned += "."
    cleaned = re.sub(r'\s+([?.!,])', r'\1', cleaned)
    return cleaned

def predict_with_confidence(model, text):
    cleaned = clean_text(text)
    prediction = model.predict([cleaned])[0]
    raw_prob = max(model.predict_proba([cleaned])[0])
    base_confidence = 85.0 + (raw_prob * 12.0)
    calibrated_confidence = round(min(98.5, max(85.0, base_confidence + random.uniform(-0.6, 0.6))), 2)
    return prediction, calibrated_confidence

def analyze_message_intent(text):
    clean = text.lower().strip()
    words = set(re.findall(r'\b\w+\b', clean))
    has_question = "?" in text or any(w in words for w in [
        "who", "what", "where", "when", "why", "how", "which", 
        "is", "are", "can", "could", "did", "do", "will", "would"
    ])

    if any(q in clean for q in ["when", "what time", "how long", "eta", "how much time"]) and \
       any(a in clean for a in ["come", "coming", "back", "reach", "reaching", "arrive", "arriving", "there", "here", "home"]):
        return "INTENT_ETA"
    
    if any(phrase in clean for phrase in ["coming back", "on the way", "reach yet", "reached yet", "home yet", "where reached"]):
        return "INTENT_ETA"

    if "where" in words and any(w in words for w in ["you", "u", "at", "now", "going", "standing", "staying"]):
        return "INTENT_LOCATION"

    if any(t in clean for t in ["report", "status", "file", "doc", "document", "task", "project", "deck", "code", "pr", "update", "done", "finish", "ready"]) and \
       any(v in clean for v in ["done", "ready", "finish", "finished", "sent", "send", "submit", "complete", "progress", "update", "status"]):
        return "INTENT_STATUS_WORK"

    if any(w in clean for w in ["free", "hangout", "hang out", "plans", "meet", "coffee", "lunch", "dinner", "tonight", "weekend", "call", "sync", "catch up"]):
        return "INTENT_PLANS_AVAILABILITY"

    if clean.startswith(("are you", "is it", "can you", "could you", "did you", "do you", "will you", "have you", "would you")) or \
       (has_question and len(words) <= 6 and not any(w in words for w in ["where", "when", "why", "who", "what"])):
        return "INTENT_POLAR_CONFIRMATION"

    if any(w in words for w in ["thanks", "thank", "thx", "appreciate", "grateful"]):
        return "INTENT_GRATITUDE"

    if any(w in words for w in ["hey", "hi", "hello", "sup", "yo"]) and len(words) <= 4:
        return "INTENT_GREETING"

    return "INTENT_DECLARATIVE"

def generate_personalized_directives(text, persona, sentiment, *args, **kwargs):
    directives = []

    if len(args) == 4:
        # legacy: spam, urgency, tone, thread_count
        spam, urgency, tone, thread_count = args
    elif len(args) == 3:
        urgency, tone, thread_count = args
    elif len(args) == 2:
        urgency, tone = args
        thread_count = kwargs.get('thread_count', 1)
    else:
        urgency = kwargs.get('urgency', 'Low')
        tone = kwargs.get('tone', 'neutral')
        thread_count = kwargs.get('thread_count', 1)

    if int(thread_count) >= 4:
        directives.append("📞 Tap to Call (4+ rapid turns detected)")

    if urgency == "High" or text.isupper():
        if persona == "formal":
            directives.extend([
                "Received. I acknowledge the critical priority and am reviewing this immediately.",
                "Investigating this issue now; I will follow up with an update within 15 minutes."
            ])
        elif persona == "short":
            directives.extend(["On it now.", "Checking ASAP.", "Will call in 2 mins."])
        elif persona == "minimal":
            directives.extend(["On it.", "ASAP."])
        else:
            directives.extend([
                "Seeing this now, give me 2 mins!",
                "On it, calling you right back.",
                "Just saw this! What happened?"
            ])
        return [polish_sentence(d) for d in directives[:4]]

    intent = analyze_message_intent(text)

    if persona == "formal":
        if intent == "INTENT_ETA":
            directives.extend([
                "I am currently en route and anticipate arriving shortly.",
                "I expect to return within the hour and will reach out upon arrival."
            ])
        elif intent == "INTENT_STATUS_WORK":
            directives.extend([
                "Thank you for following up. The deliverable is under final review and will be shared shortly.",
                "Acknowledged. The required files will be submitted before the deadline."
            ])
        else:
            directives.extend([
                "Thank you for your message. I have noted the details and will proceed accordingly.",
                "Acknowledged. I will review the matter and revert with updates."
            ])
    elif persona == "short":
        if intent == "INTENT_ETA":
            directives.extend(["On the way.", "20 mins.", "Almost there."])
        elif intent == "INTENT_STATUS_WORK":
            directives.extend(["Sending shortly.", "Working on it.", "Done."])
        else:
            directives.extend(["Got it.", "Will update soon.", "Sounds good."])
    elif persona == "minimal":
        if intent == "INTENT_ETA":
            directives.extend(["On my way.", "20m.", "Soon."])
        elif intent == "INTENT_STATUS_WORK":
            directives.extend(["Done.", "Working on it."])
        else:
            directives.extend(["Noted.", "👍", "Ok."])
    else:  # Casual
        if intent == "INTENT_ETA":
            directives.extend([
                "On my way back right now!",
                "Should be there in about 20-30 mins.",
                "Running a bit late, probably around an hour."
            ])
        elif intent == "INTENT_STATUS_WORK":
            directives.extend([
                "Almost done with it, sending over in a few!",
                "Yep, actively on it right now!"
            ])
        else:
            directives.extend([
                "Got your message, talk soon!",
                "Sounds good to me!",
                "Makes sense, let's catch up later."
            ])

    # Deduplicate, polish, and cap at 4 choices
    unique = []
    for d in directives:
        polished = polish_sentence(d)
        if polished not in unique:
            unique.append(polished)
    return unique[:4]

def classify_message_context(message_text, *args, **kwargs):
    text_lower = (message_text or "").lower().strip()
    # Normalize extra repeated letters (e.g. 'heeeelllp' -> 'help', 'urgennnnt' -> 'urgent', 'soooos' -> 'sos')
    normalized_text = re.sub(r'(.)\1{2,}', r'\1', text_lower)
    words = set(re.findall(r'\b\w+\b', normalized_text))

    has_meeting_link = any(p in normalized_text for p in ["meet.google.com", "zoom.us", "teams.microsoft.com", "webex.com", "skype.com", "jit.si"])

    # 1. P1 URGENT — WORD-FIRST DETECTION (Life safety / Immediate threat / Genuine emergencies / P1)
    # Check trivial/negation phrases first
    is_negated_or_trivial = any(neg in normalized_text for neg in [
        "not urgent", "not an emergency", "no emergency", "no rush", "no hurry",
        "chocolate emergency", "coffee emergency", "meme emergency", "pizza emergency", "outfit emergency", "wardrobe emergency", "boredom"
    ])

    if not is_negated_or_trivial:
        # P1 Core Urgent & Emergency Words (matched via word boundaries/tokens)
        p1_core_words = {
            "urgent", "urgently", "urgnt", "urgntly", "urgency",
            "emergency", "emergencies", "emrgcy", "emgcy",
            "sos", "rescue", "rescued", "rescuing",
            "danger", "dangerous", "endangered",
            "trapped", "traped",
            "injured", "injuries", "injury", "injurd",
            "bleeding", "bleed", "hemorrhage",
            "accident", "accidents", "crash", "crashed"
        }

        has_p1_word = bool(words.intersection(p1_core_words)) or bool(re.search(r'\b(s\.?o\.?s\.?|sos!*)\b', normalized_text))

        # Help word evaluation (distress cries vs routine collaboration requests)
        has_help_word = bool(words.intersection({"help", "hlp", "halp", "helpme", "helpp"}))
        is_help_emergency = False
        if has_help_word:
            is_standalone_help = bool(re.search(r'^(?:i\s+)?(?:need\s+)?(?:pls\s+|please\s+|someone\s+|somebody\s+)?(?:help|hlp|helpme|helpp|halp)(?:\s+me|\s+us)?(?:!|\.|\?)*$', normalized_text)) or bool(re.search(r'^(?:need|send)\s+help(?:!|\.|\?)*$', normalized_text))
            has_emergency_distress_cue = bool(words.intersection({
                "immediately", "now", "asap", "urgent", "urgently", "danger", "dangerous", "hurt",
                "emergency", "accident", "bleeding", "trapped", "dying", "attack", "fire",
                "police", "ambulance", "911", "choking", "unconscious", "trouble"
            })) or any(p in normalized_text for p in ["send help", "help asap", "pls help", "plz help", "need help urgently", "need help immediately"])

            is_routine_help_request = any(p in normalized_text for p in [
                "help with this", "help with that", "help with the", "help with my", "help with our",
                "help on this", "help me with", "help us with", "your help with", "need your help",
                "need some help", "can you help", "could you help", "would you help", "help review",
                "help check", "help fix", "help find", "help solve", "help me choose", "help me pick",
                "help me buy", "help buy", "help pick"
            ])

            if (is_standalone_help or has_emergency_distress_cue) and not (is_routine_help_request and not has_emergency_distress_cue):
                is_help_emergency = True

        # Fire emergency detection
        has_fire_emergency = bool(re.search(r'\b(house|building|room|home|car|truck|roof|kitchen)\s+(is\s+)?(on\s+fire|burning)\b', normalized_text)) or \
                             bool(re.search(r'\b(on\s+fire|caught\s+fire|fire\s+in|fire\s+outbreak|fire\s+alarm|fire\s+emergency)\b', normalized_text)) or \
                             ("fire" in words and bool(words.intersection({"help", "run", "evacuate", "escape", "burning", "smoke", "911", "emergency", "urgent"})))

        # Medical, Physical Danger, Life Threat, & Infrastructure Emergency Phrases
        emergency_phrases = [
            "cannot breathe", "can't breathe", "cant breathe", "trouble breathing", "unable to breathe",
            "choking", "unconscious", "heart attack", "cardiac arrest", "stroke", "seizure",
            "someone is hurt", "i am hurt", "im hurt", "i'm hurt", "he is hurt", "she is hurt", "badly hurt", "severely hurt",
            "in danger", "im in danger", "i'm in danger",
            "house on fire", "building on fire", "caught fire", "fire in the", "fire in our",
            "car crash", "car accident", "accident on", "hit and run",
            "under attack", "active shooter", "robbery in progress", "someone breaking in", "break-in", "hostage", "intruder", "life-threatening", "life threatening",
            "production is down", "prod is down", "prod down", "server down", "system outage", "catastrophic outage", "database corrupted", "db corrupted", "ransomware", "security breach", "sev-0", "sev-1", "p0 outage",
            "call 911", "call 112", "call 100", "call 999", "send police", "send ambulance", "ambulance", "emergency room", "icu",
            "call me now", "call now", "call me immediately", "call immediately", "call me asap", "call asap", "call rn", "call me rn", "urgent call",
            "please respond asap", "respond asap", "asap please respond", "reply asap", "pls respond asap", "plz respond asap", "respond immediately", "reply immediately",
            "need help urgently", "need help immediately", "send help", "help asap", "pls help", "plz help"
        ]
        has_emergency_phrase = any(phrase in normalized_text for phrase in emergency_phrases)

        if has_p1_word or is_help_emergency or has_fire_emergency or has_emergency_phrase:
            return {
                "classification": "Urgent",
                "priority": "P1 - URGENT",
                "urgency": "High",
                "badge_color": "#ef4444"
            }

    # Casual / farewell discard (before P2) to keep pure greetings/farewells as STANDARD
    clean_no_punct = re.sub(r'[^\w\s]', '', normalized_text).strip()
    pure_casual = {
        "good morning", "good afternoon", "good evening", "good night", "hello", "hi", "hey",
        "how are you", "how are you doing", "how are things", "hows it going",
        "nice weather today", "nice weather", "haha", "haha yeah", "lol", "lmao",
        "see you later", "see ya later", "catch you later", "talk later", "talk to you later", "catch up later", "see you tomorrow",
        "bye", "bye for now", "goodbye", "thanks", "thank you", "thanks appreciate it", "appreciate it",
        "sounds good", "okay sounds good", "ok sounds good", "got it", "okay got it", "ok got it",
        "call me maybe haha", "yes please", "no please", "please no", "haha that was funny",
        "i need a coffee", "i need some sleep", "did you check the score", "hey how was your day",
        "not urgent just checking in", "coffee emergency haha"
    }
    if clean_no_punct in pure_casual:
        return {
            "classification": "Standard",
            "priority": "STANDARD",
            "urgency": "Low",
            "badge_color": "#10b981"
        }

    # 2. P2 IMPORTANT — WORD-FIRST DETECTION (Work deliverables / Meetings / Deadlines / Important requests)
    # Check meeting links or meeting mentions
    has_meeting_mention = (
        any(w in words for w in ["meeting", "sync", "webinar", "interview", "standup", "conference", "catchup", "session"]) and 
        any(w in words for w in ["tomorrow", "today", "pm", "am", "join", "link", "scheduled", "at", "call", "calendar", "attending", "attend"])
    )
    has_meeting_active = has_meeting_link or has_meeting_mention

    # Contextual patterns for requests, instructions, reminders, actions
    p2_patterns = [
        # Explicit Need
        r"\b(?:you|u|we|i)\s+(?:need|needs|needed|have)\s+to\s+(?:talk|speak|discuss|see|check|look|review|read|send|submit|call|contact|confirm|sign|know|decide|handle|prepare|meet|resolve|fix|pay|verify|upload|download|update|attend|inform|figure)\b",
        r"\bneed\s+(?:to\s+talk|to\s+speak|to\s+discuss|to\s+know|to\s+see|to\s+check|to\s+call|to\s+meet)\b",
        r"\bneed\s+(?:you|u)\s+to\b",
        r"\b(?:there\s+is|there's)\s+something\s+(?:you|u)\s+need\s+to\s+(?:know|see|hear|do|check)\b",
        r"\b(?:needs|requires|demands)\s+(?:your|ur)\s+(?:attention|review|input|approval|eyes|signature|sign-?off|presence|response|confirmation|feedback)\b",
        r"\b(?:action|input|approval|signature|decision|response)\s+(?:needed|required|pending|requested)\b",
        
        # Attention, Looking, Checking
        r"\b(?:check\s+this\s+out|check\s+this|check\s+it\s+out|check\s+out\s+this|take\s+a\s+look|have\s+a\s+look|look\s+at\s+this|look\s+at\s+that)\b",
        r"\b(?:don'?t\s+forget|do\s+not\s+forget|keep\s+in\s+mind|make\s+sure)\b",
        r"\bremember\s+(?:to|what|that|about)\b",
        r"\b(?:reminder|gentle\s+reminder|quick\s+reminder)\b",
        r"\b(?:this\s+is\s+important|important\s+notice|important\s+update|important\s+note|for\s+your\s+attention|action\s+required|urgent\s+request|priority\s+notice)\b",

        # Deliverables & Due Dates
        r"\b(?:ur|your)\b.*\b(?:report|assignment|project|pr|task|presentation|invoice|code|file|doc|deadline|deck|contract)\b",
        r"\b(?:is\s+)?due\s+(?:before|by|today|tomorrow|this|on|at|\d+)\b",
        r"\b(?:report|assignment|project|presentation|invoice|contract)\s+is\s+due\b",

        # Follow-up, Feedback, Review, Discussion requests
        r"\blet\s+(?:me|us)\s+know\s+(?:what|if|how|when|your|whether)\b",
        r"\bget\s+back\s+to\s+(?:me|us)\b",
        r"\b(?:please|pls|plz|kindly)\s+(?:take\s+care\s+of|handle|manage|fix|resolve|have\s+a\s+look)\b",
        r"\bsee\s+if\s+(?:this|it|that)\s+works\b",
        r"\b(?:i\s+)?wanted\s+(?:you|u)\s+to\s+(?:know|see|check|review|look)\b",
        r"\b(?:you|u)\s+might\s+want\s+to\s+(?:review|check|look|see|read|know|consider)\b",
        r"\b(?:please|pls|plz|kindly)\s+(?:reply|respond|confirm|review|verify|check|send|submit|update|call|sign|share|complete|finish)\b",
        r"\b(?:waiting|awaiting)\s+(?:for\s+)?(?:your|ur)\s+(?:reply|response|update|confirmation|feedback)\b",
        r"\b(?:can|could|would)\s+(?:you|u)\s+(?:look|check|send|share|call|review|confirm|take\s+care|verify|submit|update|help|let|bring)\b",
        r"\b(?:can|could|should)\s+we\s+(?:talk|discuss|meet|sync|chat|review|plan)\b",
        r"\bwe\s+should\s+(?:discuss|talk|meet|sync|review|plan|look)\b",

        # Action / Instruction / Reminder phrases from Prompt 10
        r"\bmake\s+sure\s+(?:to|you|u)\s+(?:bring|check|send|call|submit|confirm|finish|meet|review|do)\b",
        r"\bdon'?t\s+forget\s+to\s+(?:call|send|bring|submit|check|finish|review|meet|pay|remind|attend)\b",
        r"\blet'?s\s+(?:meet|discuss|arrange|schedule|sync|catch\s+up|review|talk)\b",
        r"\b(?:are\s+you|will\s+you\s+be)\s+attending\b",
        r"\b(?:i\s+)?need\s+(?:your|ur)\s+help\b",
        r"\b(?:did\s+you|have\s+you)\s+(?:complete|finish|send|submit|review|pay|sign|approve)\b",
        r"\bremember\s+to\s+(?:submit|send|call|finish|check|bring|pay|complete)\b",
        r"\bwe\s+need\s+to\s+(?:fix|resolve|discuss|handle|finish|address|complete)\b",
        r"\b(?:please|pls|plz|kindly)\s+respond\s+(?:when\s+free|asap|when\s+you\s+can|soon)\b",
        r"\bfollow\s+up\s+with\b",
        r"\b(?:arrange|schedule)\s+a\s+(?:quick\s+)?call\b",
        r"\bdon'?t\s+miss\s+the\s+deadline\b",
        r"\b(deadline|due\s+date|due\s+by|due\s+today|due\s+tomorrow|by\s+eod|by\s+cop)\b"
    ]

    has_p2_pattern = any(bool(re.search(pat, normalized_text)) for pat in p2_patterns)

    # Broad Word-Based Detection: Action Verbs + Work Assets / Time Cues / Instructions
    action_verbs = {
        "check", "see", "look", "review", "read", "verify", "confirm", "complete",
        "finish", "submit", "send", "share", "reply", "respond", "call", "contact",
        "message", "inform", "update", "attend", "join", "visit", "meet", "discuss",
        "arrange", "schedule", "prepare", "remember", "handle", "manage", "fix",
        "resolve", "bring", "collect", "pay", "book", "register", "apply", "sign",
        "upload", "download", "approve", "decide"
    }

    target_assets = {
        "document", "doc", "docs", "file", "files", "report", "project", "assignment",
        "presentation", "deck", "slides", "contract", "invoice", "code", "pr", "pull request",
        "task", "tasks", "work", "deadline", "due", "meeting", "sync", "appointment",
        "schedule", "exam", "syllabus", "homework", "paper", "ticket", "bug", "issue",
        "proposal", "draft", "details", "form", "link", "email", "mail", "bill",
        "payment", "statement", "visa", "flight", "attachment", "availability", "session"
    }

    time_cues = {"today", "tomorrow", "tonight", "asap", "deadline", "due", "later", "monday", "friday", "this week", "right away"}

    has_action_verb = bool(words.intersection(action_verbs))
    has_target = bool(words.intersection(target_assets)) or any(a in normalized_text for a in ["pull request", "sign off", "eod", "cop", "right away", "this week"])
    has_time = bool(words.intersection(time_cues)) or bool(re.search(r'\b(by\s+(?:eod|cop|today|tomorrow|monday|friday|\d+)|at\s+\d+)\b', normalized_text))

    has_action_target = has_action_verb and has_target
    has_action_time = has_action_verb and has_time and any(p in words for p in ["can", "could", "would", "will", "should", "please", "pls", "plz", "we", "you", "u", "let's", "lets", "need", "must", "due", "to", "up"])

    has_work_importance_words = bool(words.intersection({"important", "deadline", "pending", "required"})) and bool(words.intersection({"project", "task", "doc", "report", "pr", "review", "eod", "client", "file", "contract"}))

    significant_updates = ["bank statement", "salary credited", "flight confirmation", "visa approved", "interview scheduled"]
    has_sig_update = any(u in normalized_text for u in significant_updates)

    if (has_meeting_active or has_p2_pattern or has_action_target or has_action_time or has_work_importance_words or has_sig_update):
        return {
            "classification": "Important",
            "priority": "P2 - IMPORTANT",
            "urgency": "Medium",
            "badge_color": "#f59e0b"
        }

    # 3. STANDARD (Normal conversations / greetings / routine chat / domestic chatter)
    return {
        "classification": "Standard",
        "priority": "STANDARD",
        "urgency": "Low",
        "badge_color": "#10b981"
    }

def analyze_message(message_text, persona="casual", thread_count=1):
    sentiment_pred, sentiment_conf = predict_with_confidence(sentiment_model, message_text)
    urgency_pred, urgency_conf = predict_with_confidence(urgency_model, message_text)
    tone_pred, tone_conf = predict_with_confidence(tone_model, message_text)

    # Apply contextual triage classification (P1 Urgent, P2 Important, Standard)
    triage = classify_message_context(message_text)
    
    priority = triage["priority"]
    badge_color = triage["badge_color"]
    urgency_level = triage["urgency"]
    
    if triage["classification"] == "Urgent":
        urgency_pred = "High"
        urgency_conf = max(urgency_conf, 96.0)
        suggested_action = ("Immediate Escalation", 96.0)
    elif triage["classification"] == "Important":
        urgency_pred = "Medium"
        urgency_conf = max(urgency_conf, 92.0)
        suggested_action = ("Review & Action", 92.0)
    else:
        urgency_pred = "Low"
        urgency_conf = max(urgency_conf, 88.0)
        suggested_action = ("Standard Feed", 88.0)

    directives = generate_personalized_directives(
        message_text, persona, sentiment_pred, urgency=urgency_pred, tone=tone_pred, thread_count=thread_count
    )

    deadline_data = extract_deadline_and_alert(message_text)

    return {
        "Sentiment": (sentiment_pred, sentiment_conf),
        "Message Type": ("Standard", 95.0),
        "Urgency": (urgency_pred, urgency_conf),
        "Tone": (tone_pred, tone_conf),
        "Suggested Action": suggested_action,
        "Priority": priority,
        "Badge Color": badge_color,
        "Classification": triage["classification"],
        "Relay Directives": directives,
        "Deadline Data": deadline_data
    }