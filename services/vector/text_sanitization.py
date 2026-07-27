"""Text cleanup helpers used before vector storage."""

from __future__ import annotations

import unicodedata


def _text_quality_score(text: str) -> int:
    score = 0
    for char in text:
        codepoint = ord(char)
        category = unicodedata.category(char)

        if char == "\ufffd":
            score -= 12
        elif category.startswith("L"):
            score += 3 if not char.isascii() else 1
        elif category.startswith("N"):
            score += 1
        elif category.startswith("P") or category.startswith("Z"):
            score += 0
        elif category.startswith("C"):
            score -= 8

        if 0x00A0 <= codepoint <= 0x00FF and category != "Ll" and category != "Lu":
            score -= 4
        elif 0x0080 <= codepoint <= 0x009F:
            score -= 8

    return score


def repair_transcript_text(text: str) -> str:
    """Repair common UTF-8 text that was accidentally decoded as Latin-1/CP1252."""
    if not text:
        return text

    candidates = [text]
    for encoding in ("latin-1", "cp1252"):
        try:
            repaired = text.encode(encoding).decode("utf-8")
        except UnicodeError:
            continue
        if repaired not in candidates:
            candidates.append(repaired)

    return max(candidates, key=_text_quality_score)
