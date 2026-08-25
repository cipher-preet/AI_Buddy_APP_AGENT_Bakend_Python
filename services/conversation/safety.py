from __future__ import annotations

import re


# Safety/entity inspection intentionally does not infer conversational intent.
ENTITY_PATTERNS = {
    "email": r"[\w.+-]+@[\w-]+\.[\w.-]+",
    "url": r"https?://\S+",
    "phone": r"\+?\d[\d\s-]{7,}\d",
    "currency": r"(?:₹|rs\.?|inr|\$)\s?\d+(?:[\d,]*)(?:\.\d+)?",
    "percentage": r"\b\d+(?:\.\d+)?%",
}


def detect_rule_signals(text: str) -> dict[str, list[str]]:
    entities = {name: re.findall(pattern, text or "", flags=re.IGNORECASE) for name, pattern in ENTITY_PATTERNS.items()}
    return {"actionSignals": [], "entities": [item for values in entities.values() for item in values]}
