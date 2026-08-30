from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

UTC = timezone.utc
DEFAULT_TIMEZONE = "Asia/Kolkata"

DELIVERY_TYPES = (
    "NORMAL_NOTIFICATION",
    "ALARM_NOTIFICATION",
    "AI_CALL",
)

CLAIMABLE_STATUSES = ("SCHEDULED", "RETRY_PENDING", "TRIGGERING")


def resolve_zone(name: str | None) -> ZoneInfo | timezone:
    raw = (name or "").strip() or DEFAULT_TIMEZONE
    try:
        return ZoneInfo(raw)
    except Exception:
        if raw.upper() in {"UTC", "GMT", "ETC/UTC"}:
            return UTC
        return ZoneInfo(DEFAULT_TIMEZONE)


def parse_time_label(time_label: str) -> tuple[int, int] | None:
    import re

    match = re.match(r"^(1[0-2]|[1-9]):([0-5]\d)\s?(AM|PM)$", (time_label or "").strip(), re.I)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2))
    period = match.group(3).upper()
    if period == "PM" and hour < 12:
        hour += 12
    if period == "AM" and hour == 12:
        hour = 0
    return hour, minute


def zoned_local_to_utc(date_key: str, time_label: str, timezone_name: str | None) -> datetime | None:
    parsed = parse_time_label(time_label)
    try:
        year, month, day = [int(part) for part in date_key.split("-")]
    except (TypeError, ValueError, AttributeError):
        return None
    if not parsed:
        return None
    hour, minute = parsed
    zone = resolve_zone(timezone_name)
    try:
        local = datetime(year, month, day, hour, minute, 0, tzinfo=zone)
    except ValueError:
        return None
    return local.astimezone(UTC)


def to_occurrence_key(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def occurrence_id(reminder_id: str, occurrence_at_utc: datetime) -> str:
    return f"{reminder_id}:{to_occurrence_key(occurrence_at_utc)}"


def parse_occurrence_id(value: str) -> tuple[str, str] | None:
    if ":" not in (value or ""):
        return None
    reminder_id, stamp = value.split(":", 1)
    if not reminder_id or not stamp:
        return None
    return reminder_id, stamp


def parse_occurrence_stamp(stamp: str) -> datetime | None:
    try:
        return datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError:
        return None


def add_repeat(occurrence_at_utc: datetime, repeat: str, timezone_name: str | None) -> datetime | None:
    zone = resolve_zone(timezone_name)
    local = occurrence_at_utc.astimezone(zone)
    if repeat == "daily":
        nxt = local + timedelta(days=1)
    elif repeat == "weekly":
        nxt = local + timedelta(days=7)
    elif repeat == "weekdays":
        nxt = local + timedelta(days=1)
        while nxt.weekday() >= 5:
            nxt = nxt + timedelta(days=1)
    elif repeat == "monthly":
        month = local.month + 1
        year = local.year
        if month > 12:
            month = 1
            year += 1
        day = min(local.day, _days_in_month(year, month))
        nxt = local.replace(year=year, month=month, day=day)
    else:
        return None
    return nxt.astimezone(UTC)


def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        nxt = datetime(year + 1, 1, 1)
    else:
        nxt = datetime(year, month + 1, 1)
    prev = nxt - timedelta(days=1)
    return prev.day


def compute_next_trigger(
    date_key: str,
    time_label: str,
    timezone_name: str | None,
    repeat: str,
    after: datetime,
) -> datetime | None:
    candidate = zoned_local_to_utc(date_key, time_label, timezone_name)
    if candidate is None:
        return None
    if candidate > after or repeat == "once":
        return candidate
    for _ in range(400):
        nxt = add_repeat(candidate, repeat, timezone_name)
        if nxt is None or nxt <= candidate:
            return None
        candidate = nxt
        if candidate > after:
            return candidate
    return None


def delivery_type_from_flags(ai_calling: bool, beeping: bool, notification: bool) -> str | None:
    if ai_calling:
        return "AI_CALL"
    if beeping:
        return "ALARM_NOTIFICATION"
    if notification:
        return "NORMAL_NOTIFICATION"
    return None


@dataclass(frozen=True)
class TriggerEvent:
    version: int
    event_id: str
    reminder_id: str
    user_id: str
    occurrence_at_utc: datetime
    timezone: str
    type: str
    title: str
    message: str
    created_at: str
    attempt: int = 1
