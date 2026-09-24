from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

try:
    # Windows Python installs often need the tzdata package for IANA names.
    import tzdata  # noqa: F401
except ImportError:
    pass

DEFAULT_TIMEZONE = "Asia/Kolkata"
DATE_KEY_FORMAT = "%Y-%m-%d"
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


class InvalidTimezoneError(ValueError):
    pass


def resolve_zone(timezone_name: str | None):
    """Return a tzinfo for the given IANA name, with IST/UTC fallbacks on Windows."""
    name = (timezone_name or DEFAULT_TIMEZONE).strip() or DEFAULT_TIMEZONE
    try:
        return ZoneInfo(name)
    except Exception:
        pass

    key = name.casefold().replace(" ", "")
    if key in _FIXED_OFFSETS:
        return _FIXED_OFFSETS[key]

    # Keep chat/day tools usable even when tzdata is missing.
    if name == DEFAULT_TIMEZONE or key in {"asia/kolkata", "asia/calcutta", "ist"}:
        return IST

    raise InvalidTimezoneError(f"Unsupported IANA timezone: {name}")


def ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def to_local(value: datetime, timezone_name: str) -> datetime:
    return ensure_utc(value).astimezone(resolve_zone(timezone_name))


def date_key_for(value: datetime, timezone_name: str) -> str:
    return to_local(value, timezone_name).strftime(DATE_KEY_FORMAT)


def previous_date_key(now_utc: datetime, timezone_name: str) -> str:
    local = to_local(now_utc, timezone_name)
    return (local.date() - timedelta(days=1)).isoformat()


def local_day_bounds_utc(date_key: str, timezone_name: str) -> tuple[datetime, datetime]:
    zone = resolve_zone(timezone_name)
    start_local = datetime.fromisoformat(date_key).replace(tzinfo=zone)
    end_local = start_local + timedelta(days=1)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


def local_midnight(now_utc: datetime, timezone_name: str) -> datetime:
    local = to_local(now_utc, timezone_name)
    return local.replace(hour=0, minute=0, second=0, microsecond=0)


def is_after_trigger(
    now_utc: datetime,
    timezone_name: str,
    hour: int,
    minute: int,
    grace_minutes: int = 0,
) -> bool:
    local = to_local(now_utc, timezone_name)
    midnight = local.replace(hour=0, minute=0, second=0, microsecond=0)
    trigger = midnight.replace(hour=hour, minute=minute)
    grace_ready = local >= midnight + timedelta(minutes=max(grace_minutes, 0))
    return local >= trigger and grace_ready
