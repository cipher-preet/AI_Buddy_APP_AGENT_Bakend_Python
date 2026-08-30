"""Detect English vs Hindi/Hinglish for reminder voice replies."""

from __future__ import annotations

import re

ReplyLanguage = str  # "en" | "hi"

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_HINDI_ROMAN = re.compile(
    r"\b(aaj|kal|parso|parson|subah|shaam|sham|raat|dopahar|baje|bajkar|"
    r"yaad|dilao|dilana|dila|mujhe|mera|meri|karo|karna|kar do|set kar|"
    r"phone lagana|call karna|dawai|dawa)\b",
    re.IGNORECASE,
)


def detect_language(text: str, fallback: str | None = None) -> ReplyLanguage:
    sample = text or ""
    if _DEVANAGARI.search(sample) or _HINDI_ROMAN.search(sample):
        return "hi"
    if fallback in {"hi", "en"}:
        return fallback
    return "en"


def is_hindi_text(text: str) -> bool:
    return bool(_DEVANAGARI.search(text or ""))
