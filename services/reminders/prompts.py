"""Hardcoded reminder-voice replies. Questions never go through the LLM."""

from services.reminders.language import ReplyLanguage

PROMPT_KIND_GREETING = "greeting"
PROMPT_KIND_ASK_TITLE = "ask_title"
PROMPT_KIND_ASK_DATE = "ask_date"
PROMPT_KIND_ASK_TIME = "ask_time"
PROMPT_KIND_OUT_OF_CONTEXT = "out_of_context"
PROMPT_KIND_UNCLEAR = "unclear"
PROMPT_KIND_EMPTY = "empty"
PROMPT_KIND_SAVED = "saved"

_PROMPTS = {
    "en": {
        PROMPT_KIND_GREETING: "Please tell me what reminder I should set.",
        PROMPT_KIND_ASK_TITLE: "What should I remind you about?",
        PROMPT_KIND_ASK_DATE: "On which date should I set this reminder?",
        PROMPT_KIND_ASK_TIME: "What time should I set this for?",
        PROMPT_KIND_OUT_OF_CONTEXT: (
            "I can only help you set a reminder. Please tell me what to remind you about, and when."
        ),
        PROMPT_KIND_UNCLEAR: "I didn't catch that. Please say the reminder again.",
        PROMPT_KIND_EMPTY: "I didn't hear anything. Please say the reminder again.",
        PROMPT_KIND_SAVED: "I've set your reminder.",
    },
    "hi": {
        PROMPT_KIND_GREETING: "कृपया बताइए, आपको कौन सा रिमाइंडर सेट करना है?",
        PROMPT_KIND_ASK_TITLE: "आपको किस चीज़ की याद दिलानी है?",
        PROMPT_KIND_ASK_DATE: "यह रिमाइंडर किस तारीख के लिए सेट करूँ?",
        PROMPT_KIND_ASK_TIME: "यह रिमाइंडर किस समय के लिए सेट करूँ?",
        PROMPT_KIND_OUT_OF_CONTEXT: (
            "मैं केवल रिमाइंडर सेट कर सकती हूँ। कृपया बताइए क्या याद दिलाना है, और कब।"
        ),
        PROMPT_KIND_UNCLEAR: "मैं समझ नहीं पाई। कृपया रिमाइंडर दोबारा बोलिए।",
        PROMPT_KIND_EMPTY: "कुछ सुनाई नहीं दिया। कृपया रिमाइंडर दोबारा बोलिए।",
        PROMPT_KIND_SAVED: "मैंने आपका रिमाइंडर सेट कर दिया है।",
    },
}

# Back-compat English aliases used by tests.
GREETING = _PROMPTS["en"][PROMPT_KIND_GREETING]
ASK_TITLE = _PROMPTS["en"][PROMPT_KIND_ASK_TITLE]
ASK_DATE = _PROMPTS["en"][PROMPT_KIND_ASK_DATE]
ASK_TIME = _PROMPTS["en"][PROMPT_KIND_ASK_TIME]
OUT_OF_CONTEXT = _PROMPTS["en"][PROMPT_KIND_OUT_OF_CONTEXT]
UNCLEAR = _PROMPTS["en"][PROMPT_KIND_UNCLEAR]
EMPTY_AUDIO = _PROMPTS["en"][PROMPT_KIND_EMPTY]
SAVED = _PROMPTS["en"][PROMPT_KIND_SAVED]

STATIC_PROMPTS = _PROMPTS["en"]


def prompt_text(kind: str, language: ReplyLanguage = "en") -> str:
    pack = _PROMPTS["hi"] if language == "hi" else _PROMPTS["en"]
    return pack.get(kind) or _PROMPTS["en"].get(kind) or UNCLEAR


def missing_field_prompt(field: str | None, language: ReplyLanguage = "en") -> tuple[str, str]:
    if field == "title":
        return PROMPT_KIND_ASK_TITLE, prompt_text(PROMPT_KIND_ASK_TITLE, language)
    if field == "date":
        return PROMPT_KIND_ASK_DATE, prompt_text(PROMPT_KIND_ASK_DATE, language)
    if field == "time":
        return PROMPT_KIND_ASK_TIME, prompt_text(PROMPT_KIND_ASK_TIME, language)
    return PROMPT_KIND_UNCLEAR, prompt_text(PROMPT_KIND_UNCLEAR, language)


def confirmation_text(
    title: str,
    date_label: str,
    time_label: str,
    language: ReplyLanguage = "en",
) -> str:
    clean_title = (title or "this").strip().rstrip(".")
    if language == "hi":
        return f"मैंने {clean_title} का रिमाइंडर {date_label} को {time_label} पर सेट कर दिया है।"
    return f"I've set a reminder to {clean_title} on {date_label} at {time_label}."
