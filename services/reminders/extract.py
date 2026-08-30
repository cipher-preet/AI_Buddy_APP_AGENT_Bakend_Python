"""Rule-first reminder extraction. LLM is only a fallback for messy speech."""

from __future__ import annotations

import re
from datetime import datetime

from services.reminders.dates import parse_date, parse_repeat, parse_time
from services.reminders.language import detect_language
from services.reminders.schemas import ReminderCollected, ReminderExtractorResponse, ReminderIntent

_REMINDER_HINTS = re.compile(
    r"\b(remind|reminder|remember|alarm|notify|notification|wake me|don't forget|do not forget|"
    r"set (a |an )?reminder|create (a |an )?reminder|"
    r"yaad|dilao|dilana|dila dena|set kar|rakh dena|rakh do)\b|"
    r"याद|रिमाइंडर|याद दिल",
    re.IGNORECASE,
)

_OUT_OF_SCOPE = re.compile(
    r"\b(weather|joke|news|recipe|song|music|capital of|who is|what is|how are you|"
    r"tell me a story|play|cricket score|stock price|mausam|mausam kaisa)\b|"
    r"मौसम|चुटकुला|गाना|क्रिकेट",
    re.IGNORECASE,
)

_TITLE_PREFIXES = (
    r"^(please\s+)?(can you\s+|could you\s+|i want (you )?to\s+|i('d| would) like (you )?to\s+)?",
    r"(set|create|add|make|schedule)\s+(me\s+)?(a\s+|an\s+|the\s+)?reminder\s+(for me\s+)?(for|to|about)?\s*",
    r"remind\s+me\s+(to|about|for)?\s*",
    r"^(?:a\s+|an\s+|the\s+)?reminder\s+(to|for|about)\s*",
    r"(don'?t|do not)\s+forget\s+to\s*",
    r"wake\s+me(\s+up)?\s*(to|for)?\s*",
    r"^(please\s+)?(mujhe\s+)?(yaad\s+dila(?:o|na|iye)?(?:\s+dena|\s+do)?|"
    r"reminder\s+(set|laga|rakh)\s+(kar(?:o|na)?|do|dena)?)\s*(ki|ke|ko)?\s*",
    r"^मुझे\s+याद\s+दिला(?:ओ|ना|इए)?(?:\s+देना|\s+दो)?\s*",
    r"^(?:एक\s+)?रिमाइंडर\s+(?:सेट\s+)?(?:करो|कर\s+दो|कर\s+दीजिए)?\s*",
)

_WHEN_WORDS = (
    r"tomorrow|tommrow|tommorow|today|tonight|kal|aaj|parson|parso|"
    r"कल|आज|परसों|"
    r"(?:this|next)\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|week)|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"somwar|mangalwar|budhwar|guruwar|shukrawar|shanivar|raviwar|"
    r"सोमवार|मंगलवार|बुधवार|गुरुवार|शुक्रवार|शनिवार|रविवार|"
    r"day after tomorrow|"
    r"subah|shaam|sham|raat|dopahar|सुबह|शाम|रात|दोपहर|"
    r"\d{1,2}\s*(?:baje|bajkar|बजे)|"
    r"\d{1,2}(?::|\.)\d{2}\s*(?:a\.?m\.?|p\.?m\.?)?|"
    r"\d{1,2}\s*(?:a\.?m\.?|p\.?m\.?)|"
    r"noon|midnight|"
    r"(?:every|each|roz|har)\s+(?:day|week|month|din|hafte)|daily|weekly|monthly|weekdays"
)

_TRAILING_WHEN = re.compile(
    rf"\s+(?:on|at|by|for|this|next|ko|ke|ki|को)?\s*(?:{_WHEN_WORDS}).*$",
    re.IGNORECASE,
)

_LEADING_WHEN = re.compile(
    r"^(?:"
    r"(?:kal|aaj|parson|parso|tomorrow|today|tonight|कल|आज|परसों)"
    r"(?:\s+(?:subah|shaam|sham|raat|dopahar|सुबह|शाम|रात|दोपहर))?"
    r"(?:\s+(?:at|on|by|ko|ke|par|pe|को))?"
    r"(?:\s+\d{1,2}(?:[:.][0-5]\d)?\s*(?:baje|bajkar|बजे|a\.?m\.?|p\.?m\.?)?)?"
    r"|"
    r"(?:subah|shaam|sham|raat|dopahar|सुबह|शाम|रात|दोपहर)"
    r"(?:\s+(?:ke|ki|at|को))?"
    r"(?:\s+\d{1,2}(?:[:.][0-5]\d)?\s*(?:baje|bajkar|बजे|a\.?m\.?|p\.?m\.?)?)?"
    r"|"
    r"\d{1,2}(?:[:.][0-5]\d)?\s*(?:baje|bajkar|बजे)"
    r")"
    r"(?:\s+(?:ko|ke|par|pe|को))?(?:\s+|$)",
    re.IGNORECASE,
)

_LEADING_FILLER = re.compile(
    r"^(?:please|mujhe|humein|hume|humko|मुझे|हमें|हमको)\s+",
    re.IGNORECASE,
)

_HINDI_REMINDER_TAIL = re.compile(
    r"\s+(?:mujhe\s+)?(?:yaad\s+dila(?:o|na|iye)?(?:\s+dena|\s+do)?|"
    r"reminder\s+(?:set|laga)\s*(?:kar(?:o|na| do| dena)?)?|"
    r"याद\s+दिला(?:ओ|ना)?(?:\s+देना)?)\s*$",
    re.IGNORECASE,
)

_SETUP_LEFTOVER = re.compile(
    r"^(?:please\s+)?(?:can you\s+|could you\s+)?"
    r"(?:set|create|add|make|schedule|put)?\s*"
    r"(?:me\s+)?(?:a |an |the )?"
    r"(?:reminder|alarm)s?\s*"
    r"(?:for me\s*)?(?:for|to|about)?\s*$",
    re.IGNORECASE,
)

_GENERIC_TITLES = {
    "",
    "it",
    "this",
    "that",
    "me",
    "a reminder",
    "reminder",
    "something",
    "for",
    "to",
    "yaad",
    "mujhe",
    "hume",
    "humein",
    "please",
    "at",
    "on",
    "by",
    "ko",
    "ke",
    "baje",
    "bajkar",
    "रिमाइंडर",
    "याद",
    "मुझे",
    "बजे",
}

_FILLER_SPEECH = re.compile(
    r"^(hmm+|uh+|um+|ah+|okay|ok|maybe|later|yes|yeah|yep|no|nope|nothing|"
    r"please|hello|hi|hey)(?:\s+\w+){0,4}$",
    re.IGNORECASE,
)

_DATE_ONLY_TITLE = re.compile(
    r"^(?:on|at|by|for|this|next|ko|ke)?\s*"
    r"(?:tomorrow|tommrow|tommorow|today|tonight|kal|aaj|parson|"
    r"कल|आज|परसों|"
    r"(?:this|next)\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|week)|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"day after tomorrow|"
    r"subah|shaam|raat|सुबह|शाम|रात|"
    r"\d{1,2}\s*(?:baje|बजे)|"
    r"\d{1,2}(?::|\.)\d{2}\s*(?:a\.?m\.?|p\.?m\.?)?|"
    r"\d{1,2}\s*(?:a\.?m\.?|p\.?m\.?)|"
    r"noon|midnight).*$",
    re.IGNORECASE,
)


_SPEAKER_LABEL = re.compile(
    r"\[?\s*speaker\s+\d+\s*\]?\s*:?\s*",
    re.IGNORECASE,
)


def clean_reminder_transcript(transcript: str) -> str:
    lines = []
    for raw_line in (transcript or "").splitlines():
        line = _SPEAKER_LABEL.sub(" ", raw_line)
        line = " ".join(line.split())
        if line:
            lines.append(line)
    return " ".join(lines)


def looks_like_reminder(transcript: str, collected: ReminderCollected) -> bool:
    text = (transcript or "").strip()
    if not text:
        return False
    if _REMINDER_HINTS.search(text):
        return True
    if collected.title or collected.dateKey or collected.timeLabel:
        return bool(parse_date(text, datetime.now()) or parse_time(text) or extract_title(text))
    title = extract_title(text)
    return bool(title and (parse_date(text, datetime.now()) or parse_time(text)))


def looks_out_of_context(transcript: str, collected: ReminderCollected) -> bool:
    text = (transcript or "").strip()
    if not text:
        return False
    if _REMINDER_HINTS.search(text):
        return False
    return bool(_OUT_OF_SCOPE.search(text))


def extract_title(transcript: str) -> str | None:
    text = clean_reminder_transcript(transcript)
    if not text:
        return None

    cleaned = text
    for prefix in _TITLE_PREFIXES:
        cleaned = re.sub(prefix, "", cleaned, count=1, flags=re.IGNORECASE)

    cleaned = _LEADING_FILLER.sub("", cleaned).strip(" .,!?")
    cleaned = _HINDI_REMINDER_TAIL.sub("", cleaned).strip(" .,!?")
    cleaned = _LEADING_WHEN.sub("", cleaned).strip(" .,!?")
    cleaned = _TRAILING_WHEN.sub("", cleaned).strip(" .,!?")
    cleaned = re.sub(r"^(to|for|about|ko|ke)\s+", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = _HINDI_REMINDER_TAIL.sub("", cleaned).strip(" .,!?")
    cleaned = _DATE_ONLY_TITLE.sub("", cleaned).strip(" .,!?")

    if (
        not is_usable_title(cleaned)
        or _FILLER_SPEECH.match(text)
        or _DATE_ONLY_TITLE.match(cleaned)
    ):
        if re.search(r"\bwake me(\s+up)?\b", text, re.IGNORECASE):
            return "Wake up"
        if re.search(r"जगाना|जगा देना|wake", text, re.IGNORECASE) and not is_usable_title(cleaned):
            return "जगा देना" if "जग" in text else None
        return None

    if len(cleaned) > 80:
        cleaned = cleaned[:80].rsplit(" ", 1)[0]

    return cleaned or None


def rule_extract(transcript: str, now: datetime) -> ReminderCollected:
    transcript = clean_reminder_transcript(transcript)
    title = extract_title(transcript)
    date = parse_date(transcript, now)
    time_label = parse_time(transcript)
    repeat = parse_repeat(transcript)
    description = title
    return ReminderCollected(
        title=title,
        description=description,
        dateKey=date[0] if date else None,
        dateLabel=date[1] if date else None,
        timeLabel=time_label,
        repeat=repeat,
        language=detect_language(transcript),
    )


def merge_collected(
    previous: ReminderCollected,
    incoming: ReminderCollected,
) -> ReminderCollected:
    title = _prefer_title(incoming.title, previous.title)
    return ReminderCollected(
        title=title,
        description=_prefer_title(incoming.description, previous.description) or title,
        dateKey=_prefer_text(incoming.dateKey, previous.dateKey),
        dateLabel=_prefer_text(incoming.dateLabel, previous.dateLabel),
        timeLabel=_prefer_text(incoming.timeLabel, previous.timeLabel),
        repeat=incoming.repeat or previous.repeat,
        language=incoming.language or previous.language,
    )


def apply_llm_extract(
    collected: ReminderCollected,
    llm: ReminderExtractorResponse,
    now: datetime,
) -> ReminderCollected:
    date = parse_date(llm.dateExpression or "", now) if llm.dateExpression else None
    time_label = parse_time(llm.timeExpression) if llm.timeExpression else None
    incoming = ReminderCollected(
        title=_prefer_title(llm.title, None),
        description=_prefer_title(llm.description, llm.title),
        dateKey=date[0] if date else None,
        dateLabel=date[1] if date else None,
        timeLabel=time_label,
        repeat=llm.repeat,
    )
    return merge_collected(collected, incoming)


def infer_intent(
    transcript: str,
    collected: ReminderCollected,
    llm_intent: ReminderIntent | None = None,
) -> ReminderIntent:
    if looks_out_of_context(transcript, collected):
        return "out_of_context"
    if looks_like_reminder(transcript, collected) or collected.title:
        return "set_reminder"
    if llm_intent in {"set_reminder", "out_of_context", "unclear"}:
        return llm_intent
    return "unclear"


def is_usable_title(value: str | None) -> bool:
    cleaned = " ".join((value or "").strip().split())
    if not cleaned or cleaned.casefold() in _GENERIC_TITLES:
        return False
    if _SETUP_LEFTOVER.match(cleaned) or _DATE_ONLY_TITLE.match(cleaned) or _FILLER_SPEECH.match(cleaned):
        return False
    if re.search(r"\bspeaker\s+\d+\b", cleaned, re.IGNORECASE):
        return False
    if re.match(r"^(set|create|add|make|schedule)\s+(a\s+|an\s+|the\s+)?reminder\b", cleaned, re.IGNORECASE):
        return False
    return True


def _prefer_title(incoming: str | None, previous: str | None) -> str | None:
    value = incoming if is_usable_title(incoming) else None
    if value:
        return " ".join(value.strip().split())
    if is_usable_title(previous):
        return " ".join((previous or "").strip().split())
    return None


def _prefer_text(incoming: str | None, previous: str | None) -> str | None:
    value = " ".join((incoming or "").strip().split())
    if value:
        return value
    value = " ".join((previous or "").strip().split())
    return value or None
