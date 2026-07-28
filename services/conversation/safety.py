from __future__ import annotations

import re


ACTION_PATTERNS = [
    r"\bwill\b",
    r"\bmust\b",
    r"\bneed to\b",
    r"\bshould\b",
    r"\bplease\b",
    r"\bremind\b",
    r"\bfollow up\b",
    r"\bassigned\b",
    r"\bresponsible\b",
    r"\bcompleted\b",
    r"\bcancelled\b",
    r"\bblocked\b",
    r"\bpending\b",
    r"\btomorrow\b",
    r"\bfriday\b",
    r"\bnext week\b",
    r"\bdeadline\b",
    r"\bdecided\b",
    r"\bagreed\b",
    r"\bkarna hai\b",
    r"\bkal\b",
    r"\bzimmedar\b",
    r"\bpoora ho gaya\b",
    r"\bफॉलो अप\b",
    r"\bकल\b",
    r"\bपूरा\b",
    r"\bअटका\b",
]

ENTITY_PATTERNS = {
    "email": r"[\w.+-]+@[\w-]+\.[\w.-]+",
    "url": r"https?://\S+",
    "phone": r"\+?\d[\d\s-]{7,}\d",
    "currency": r"(?:₹|rs\.?|inr|\$)\s?\d+(?:[\d,]*)(?:\.\d+)?",
    "percentage": r"\b\d+(?:\.\d+)?%",
}


def detect_rule_signals(text: str) -> dict[str, list[str]]:
    lowered = text.lower()
    signals = [pattern for pattern in ACTION_PATTERNS if re.search(pattern, lowered, flags=re.IGNORECASE)]
    entities = {
        name: re.findall(pattern, text, flags=re.IGNORECASE)
        for name, pattern in ENTITY_PATTERNS.items()
    }
    return {"actionSignals": signals, "entities": [item for values in entities.values() for item in values]}
