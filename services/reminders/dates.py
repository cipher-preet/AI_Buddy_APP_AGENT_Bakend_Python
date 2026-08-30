"""Timezone-aware date and time parsing for reminder voice turns."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

IST = timezone(timedelta(hours=5, minutes=30))
UTC = timezone.utc
_FIXED_OFFSETS = {
    "asia/kolkata": IST,
    "asia/calcutta": IST,
    "ist": IST,
    "india": IST,
    "utc": UTC,
    "etc/utc": UTC,
    "gmt": UTC,
}
_OFFSET_PATTERN = re.compile(r"^(?:utc)?([+-])(\d{1,2})(?::?(\d{2}))?$")

_WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
    "somwar": 0,
    "somvaar": 0,
    "mangalwar": 1,
    "mangalvaar": 1,
    "budhwar": 2,
    "budhvaar": 2,
    "guruwar": 3,
    "guruvaar": 3,
    "shukrawar": 4,
    "shukravaar": 4,
    "shanivar": 5,
    "shanivaar": 5,
    "raviwar": 6,
    "ravivaar": 6,
    "सोमवार": 0,
    "मंगलवार": 1,
    "बुधवार": 2,
    "गुरुवार": 3,
    "शुक्रवार": 4,
    "शनिवार": 5,
    "रविवार": 6,
}

_HINDI_CLOCK = re.compile(
    r"(?:(?P<period_before>subah|shaam|sham|raat|dopahar|सुबह|शाम|रात|दोपहर)\s*(?:ke|ki|को)?\s*)?"
    r"(?P<hour>2[0-3]|1[0-9]|0?[1-9])"
    r"(?:[:.](?P<minute>[0-5]\d))?"
    r"\s*(?:baje|bajkar|बजे)"
    r"(?:\s*(?P<period_after>subah|shaam|sham|raat|dopahar|सुबह|शाम|रात|दोपहर))?",
    re.IGNORECASE,
)

_MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

_TIME_PATTERN = re.compile(
    r"(?<!\d)\b(?:at\s+)?(?P<hour>2[0-3]|1[0-9]|0?[1-9]|00)"
    r"(?:[:.](?P<minute>[0-5]\d))?"
    r"\s*(?P<meridiem>a\.?m\.?|p\.?m\.?)?(?!\d)\b",
    re.IGNORECASE,
)

_NAMED_TIME = {
    "noon": (12, 0, "PM"),
    "midnight": (12, 0, "AM"),
}

_ORDINAL_DATE = re.compile(
    r"\b(?P<day>0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?"
    r"(?:\s+(?:of\s+)?)?(?P<month>january|jan|february|feb|march|mar|april|apr|"
    r"may|june|jun|july|jul|august|aug|september|sep|sept|october|oct|"
    r"november|nov|december|dec)"
    r"(?:\s+(?P<year>20\d{2}))?\b",
    re.IGNORECASE,
)

_MONTH_FIRST_DATE = re.compile(
    r"\b(?P<month>january|jan|february|feb|march|mar|april|apr|may|june|jun|"
    r"july|jul|august|aug|september|sep|sept|october|oct|november|nov|"
    r"december|dec)\s+(?P<day>0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?"
    r"(?:,?\s+(?P<year>20\d{2}))?\b",
    re.IGNORECASE,
)

_NUMERIC_DATE = re.compile(
    r"\b(?P<day>0?[1-9]|[12]\d|3[01])[/-](?P<month>0?[1-9]|1[0-2])(?:[/-](?P<year>20\d{2}|\d{2}))?\b"
)

_ISO_DATE = re.compile(r"\b(?P<year>20\d{2})-(?P<month>0[1-9]|1[0-2])-(?P<day>0[1-9]|[12]\d|3[01])\b")


def now_in_timezone(timezone_name: str | None) -> datetime:
    return datetime.now(_resolve_zone(timezone_name))


def _resolve_zone(timezone_name: str | None):
    name = (timezone_name or "").strip() or "Asia/Kolkata"
    try:
        return ZoneInfo(name)
    except Exception:
        pass

    key = name.casefold().replace(" ", "")
    if key in _FIXED_OFFSETS:
        return _FIXED_OFFSETS[key]

    offset = _OFFSET_PATTERN.fullmatch(key)
    if offset:
        sign = 1 if offset.group(1) == "+" else -1
        hours = int(offset.group(2))
        minutes = int(offset.group(3) or 0)
        return timezone(timedelta(hours=sign * hours, minutes=sign * minutes))

    return IST


def format_date_label(value: datetime) -> str:
    return f"{value.day} {value.strftime('%b')} {value.year}"


def format_time_label(hour24: int, minute: int) -> str:
    period = "AM" if hour24 < 12 else "PM"
    hour12 = hour24 % 12
    if hour12 == 0:
        hour12 = 12
    return f"{hour12}:{minute:02d} {period}"


def to_date_key(value: datetime) -> str:
    return value.strftime("%Y-%m-%d")


def parse_date(text: str, now: datetime) -> tuple[str, str] | None:
    folded = " ".join((text or "").casefold().split())
    if not folded:
        return None

    if _has_word(
        folded,
        ("today", "tonight", "this evening", "this morning", "aaj", "आज"),
    ):
        return to_date_key(now), format_date_label(now)

    if _has_word(
        folded,
        ("day after tomorrow", "parson", "parso", "parso", "परसों"),
    ):
        value = now + timedelta(days=2)
        return to_date_key(value), format_date_label(value)

    if _has_word(
        folded,
        ("tomorrow", "tommrow", "tommorow", "tomorow", "kal", "कल"),
    ):
        value = now + timedelta(days=1)
        return to_date_key(value), format_date_label(value)

    in_days = re.search(r"\bin\s+(\d+)\s+days?\b", folded)
    if in_days:
        value = now + timedelta(days=int(in_days.group(1)))
        return to_date_key(value), format_date_label(value)

    if "next week" in folded:
        value = now + timedelta(days=7)
        return to_date_key(value), format_date_label(value)

    weekday = _parse_weekday(folded, now)
    if weekday is not None:
        return to_date_key(weekday), format_date_label(weekday)

    for pattern in (_ISO_DATE, _MONTH_FIRST_DATE, _ORDINAL_DATE, _NUMERIC_DATE):
        match = pattern.search(text or "")
        if not match:
            continue
        parsed = _calendar_date(match, now)
        if parsed is not None:
            return to_date_key(parsed), format_date_label(parsed)

    return None


def parse_time(text: str) -> str | None:
    folded = " ".join((text or "").casefold().split())
    if not folded:
        return None

    hindi_clock = _HINDI_CLOCK.search(text or "") or _HINDI_CLOCK.search(folded)
    if hindi_clock:
        hour = int(hindi_clock.group("hour"))
        minute = int(hindi_clock.group("minute") or 0)
        period = (
            hindi_clock.group("period_before")
            or hindi_clock.group("period_after")
            or ""
        ).casefold()
        hour24 = _hour_for_period(hour, period)
        if hour24 is not None:
            return format_time_label(hour24, minute)

    for name, (hour12, minute, period) in _NAMED_TIME.items():
        if re.search(rf"\b{name}\b", folded):
            hour24 = 0 if period == "AM" and hour12 == 12 else hour12
            if period == "PM" and hour12 != 12:
                hour24 = hour12 + 12
            if period == "AM" and hour12 == 12:
                hour24 = 0
            return format_time_label(hour24, minute)

    spoken_clock = re.search(
        r"\b(?P<hour>1[0-2]|0?[1-9])(?:[:.](?P<minute>[0-5]\d))?\s+"
        r"(?:in the\s+)?(?P<period>morning|evening|afternoon|night|subah|shaam|sham|raat)\b",
        folded,
    )
    if spoken_clock:
        hour = int(spoken_clock.group("hour"))
        minute = int(spoken_clock.group("minute") or 0)
        period = spoken_clock.group("period")
        if period in {"morning", "subah", "सुबह"}:
            hour24 = 0 if hour == 12 else hour
        else:
            hour24 = hour if hour == 12 else hour + 12
        return format_time_label(hour24, minute)

    if re.search(r"\b(morning|evening|night|afternoon)\b", folded) and not _TIME_PATTERN.search(
        text or ""
    ):
        return None

    match = None
    for candidate in _TIME_PATTERN.finditer(text or ""):
        has_meridiem = bool(candidate.group("meridiem"))
        has_minutes = candidate.group("minute") is not None
        has_at = bool(re.search(r"\bat\s+$", (text or "")[: candidate.start()], re.IGNORECASE))
        if has_meridiem or has_minutes or has_at:
            match = candidate
            break
    if not match:
        return None

    hour = int(match.group("hour"))
    minute = int(match.group("minute") or 0)
    meridiem = (match.group("meridiem") or "").lower().replace(".", "")

    if hour > 23 or minute > 59:
        return None

    if meridiem:
        if hour > 12:
            return None
        if meridiem.startswith("a"):
            hour24 = 0 if hour == 12 else hour
        else:
            hour24 = hour if hour == 12 else hour + 12
        return format_time_label(hour24, minute)

    if 0 <= hour <= 23 and (hour > 12 or hour == 0):
        return format_time_label(hour, minute)

    return format_time_label(hour, minute) if hour <= 12 else None


def parse_repeat(text: str) -> str | None:
    folded = " ".join((text or "").casefold().split())
    if re.search(
        r"\b(every day|everyday|daily|each day|roz|roj|har din)\b",
        folded,
    ) or "रोज़" in (text or "") or "रोज" in (text or "") or "हर दिन" in (text or ""):
        return "daily"
    if re.search(r"\b(weekdays|every weekday|each weekday)\b", folded):
        return "weekdays"
    if re.search(r"\b(every week|weekly|each week|har hafte|har week)\b", folded) or "हर हफ्ते" in (text or ""):
        return "weekly"
    if re.search(r"\b(every month|monthly|each month|har mahine)\b", folded) or "हर महीने" in (text or ""):
        return "monthly"
    return None


def _has_token(folded: str, tokens: tuple[str, ...]) -> bool:
    return any(token in folded for token in tokens)


def _has_word(folded: str, tokens: tuple[str, ...]) -> bool:
    for token in tokens:
        if not token:
            continue
        if re.search(r"[a-z]", token):
            if re.search(rf"\b{re.escape(token)}\b", folded):
                return True
        elif token in folded:
            return True
    return False


def _hour_for_period(hour: int, period: str) -> int | None:
    if hour > 23:
        return None
    if period in {"subah", "सुबह"}:
        return 0 if hour == 12 else hour
    if period in {"shaam", "sham", "raat", "dopahar", "शाम", "रात", "दोपहर"}:
        if hour == 12:
            return 12
        if hour < 12:
            return hour + 12
        return hour
    if 0 <= hour <= 23 and (hour > 12 or hour == 0):
        return hour
    return hour if hour <= 12 else None


def _parse_weekday(folded: str, now: datetime) -> datetime | None:
    for name, weekday in _WEEKDAYS.items():
        if re.search(r"[a-z]", name):
            if not re.search(rf"\b(?:this |next )?{name}\b", folded):
                continue
            next_word = f"next {name}" in folded
        else:
            if name not in folded:
                continue
            next_word = f"अगले {name}" in folded or f"next {name}" in folded
        delta = (weekday - now.weekday()) % 7
        if next_word:
            delta = 7 if delta == 0 else delta
        elif delta == 0 and f"this {name}" not in folded:
            delta = 7
        return now + timedelta(days=delta)
    return None


def _calendar_date(match: re.Match[str], now: datetime) -> datetime | None:
    groups = match.groupdict()
    try:
        day = int(groups["day"])
        month_raw = groups["month"]
        month = int(month_raw) if month_raw.isdigit() else _MONTHS[month_raw.casefold()]
        year_raw = groups.get("year")
        year = now.year if not year_raw else int(year_raw)
        if year < 100:
            year += 2000
        value = now.replace(year=year, month=month, day=day, hour=0, minute=0, second=0, microsecond=0)
        if value.date() < now.date() and not year_raw:
            value = value.replace(year=year + 1)
        return value
    except (KeyError, ValueError, TypeError):
        return None
