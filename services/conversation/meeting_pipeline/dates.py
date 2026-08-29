"""Deterministic relative-date normalization after the verifier supports dueDate.

Python must not decide whether a deadline was assigned. It only converts a
verifier-supported temporal expression when the meeting timestamp makes one
calendar date unambiguous.
"""

from __future__ import annotations

import re
from calendar import monthrange
from datetime import datetime, timedelta


_WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def normalize_supported_due_date(text: str | None, meeting_at: datetime | None) -> str | None:
    if not text or meeting_at is None:
        return None
    folded = " ".join(str(text).casefold().split())
    if not folded:
        return None
    today = meeting_at.date()
    if _is_today(folded):
        return today.isoformat()
    if _is_tomorrow(folded):
        return (today + timedelta(days=1)).isoformat()
    if "yesterday" in folded:
        return (today - timedelta(days=1)).isoformat()
    if "end of month" in folded or "end of the month" in folded:
        last = monthrange(today.year, today.month)[1]
        return today.replace(day=last).isoformat()
    weekday = _weekday(folded)
    if weekday is None:
        return None
    delta = (weekday - today.weekday()) % 7
    if "next " in folded and delta == 0:
        delta = 7
    elif delta == 0 and "next " not in folded:
        return today.isoformat()
    return (today + timedelta(days=delta)).isoformat()


def _is_today(folded: str) -> bool:
    return folded in {"today", "tonight", "this evening", "aaj"} or "tonight" in folded


def _is_tomorrow(folded: str) -> bool:
    if folded in {"tomorrow", "kal", "कल"}:
        return True
    return bool(re.search(r"\b(tomorrow|kal)\b", folded)) or "कल" in folded


def _weekday(folded: str) -> int | None:
    for name, value in _WEEKDAYS.items():
        if re.search(rf"\b{name}\b", folded):
            return value
    return None
