from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from services.daily_briefing.timezones import (
    DEFAULT_TIMEZONE,
    InvalidTimezoneError,
    local_day_bounds_utc,
    resolve_zone,
)

DATE_KEY_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
MAX_SCOPE_DAYS = 62
Window = tuple[datetime, datetime]


@dataclass(frozen=True)
class SourceScope:
    """What a mind map or document is allowed to read from a space."""

    mode: str = "space"
    dates: tuple[str, ...] = ()
    time_zone: str = DEFAULT_TIMEZONE
    windows: tuple[Window, ...] = ()

    def as_document(self) -> dict[str, Any]:
        if self.mode != "date" or not self.dates:
            return {"mode": "space"}
        return {"mode": "date", "dates": list(self.dates), "timeZone": self.time_zone}

    @property
    def day_label(self) -> str:
        if len(self.dates) == 1:
            return self.dates[0]
        if len(self.dates) > 1:
            return f"{len(self.dates)} days"
        return ""

    @property
    def coverage_label(self) -> str | None:
        if self.mode != "date" or not self.dates:
            return None
        return ", ".join(self.dates)

    def note_filter(self) -> dict[str, Any] | None:
        return _field_windows("createdAt", self.windows)

    def task_filter(self) -> dict[str, Any] | None:
        created = _field_windows("createdAt", self.windows)
        if created is None or not self.dates:
            return None
        return {
            "$or": [
                {"dueDate": {"$in": list(self.dates)}},
                {
                    "$and": [
                        {
                            "$or": [
                                {"dueDate": None},
                                {"dueDate": {"$exists": False}},
                                {"dueDate": ""},
                            ]
                        },
                        created,
                    ]
                },
            ]
        }

    def transcript_filter(self) -> dict[str, Any] | None:
        captured = _field_windows("capturedAt", self.windows)
        created = _field_windows("createdAt", self.windows)
        if captured is None or created is None:
            return None
        return {
            "$or": [
                captured,
                {
                    "$and": [
                        {
                            "$or": [
                                {"capturedAt": None},
                                {"capturedAt": {"$exists": False}},
                            ]
                        },
                        created,
                    ]
                },
            ]
        }


def _field_windows(field: str, windows: tuple[Window, ...]) -> dict[str, Any] | None:
    if not windows:
        return None
    clauses = [{field: {"$gte": start, "$lt": end}} for start, end in windows]
    if len(clauses) == 1:
        return clauses[0]
    return {"$or": clauses}


def _merged_windows(dates: list[str], time_zone: str) -> tuple[Window, ...]:
    """Neighboring days become one range so the query stays small."""
    merged: list[Window] = []
    for day in dates:
        start, end_exclusive = local_day_bounds_utc(day, time_zone)
        if merged and merged[-1][1] == start:
            merged[-1] = (merged[-1][0], end_exclusive)
            continue
        merged.append((start, end_exclusive))
    return tuple(merged)


def _clean_dates(date: str | None, dates: list[str] | tuple[str, ...] | None) -> list[str]:
    raw: list[Any] = list(dates or [])
    if date:
        raw.append(date)
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in raw:
        day = str(item or "").strip()
        if not day or day in seen:
            continue
        if not DATE_KEY_PATTERN.fullmatch(day):
            raise ValueError("Each date must be YYYY-MM-DD.")
        seen.add(day)
        cleaned.append(day)
    cleaned.sort()
    return cleaned


def parse_source_scope(
    scope: str | None,
    date: str | None = None,
    time_zone: str | None = None,
    dates: list[str] | tuple[str, ...] | None = None,
) -> SourceScope:
    """Whole space is the default so older clients keep their current behavior."""
    mode = str(scope or "space").strip().lower() or "space"
    if mode in {"space", "all"}:
        return SourceScope(mode="space", time_zone=DEFAULT_TIMEZONE)
    if mode != "date":
        raise ValueError("scope must be 'space' or 'date'.")

    selected = _clean_dates(date, dates)
    if not selected:
        raise ValueError("Select at least one date.")
    if len(selected) > MAX_SCOPE_DAYS:
        raise ValueError(f"Select at most {MAX_SCOPE_DAYS} days.")

    zone_name = str(time_zone or DEFAULT_TIMEZONE).strip() or DEFAULT_TIMEZONE
    try:
        resolve_zone(zone_name)
    except InvalidTimezoneError as error:
        raise ValueError(str(error)) from error

    return SourceScope(
        mode="date",
        dates=tuple(selected),
        time_zone=zone_name,
        windows=_merged_windows(selected, zone_name),
    )


def scope_from_stored(raw: Any) -> SourceScope:
    if not isinstance(raw, dict) or str(raw.get("mode") or "space") != "date":
        return SourceScope(mode="space")
    stored_dates = raw.get("dates")
    dates = stored_dates if isinstance(stored_dates, list) else None
    return parse_source_scope("date", raw.get("date"), raw.get("timeZone"), dates=dates)


def active_scope_filter(scope: SourceScope) -> dict[str, Any]:
    if scope.mode == "date" and scope.dates:
        days = list(scope.dates)
        return {
            "sourceScope.mode": "date",
            "sourceScope.dates": {"$all": days, "$size": len(days)},
        }
    return {
        "$or": [
            {"sourceScope.mode": "space"},
            {"sourceScope": {"$exists": False}},
            {"sourceScope": None},
        ]
    }


def empty_context_message(scope: SourceScope) -> str:
    if scope.mode == "date" and scope.dates:
        if len(scope.dates) == 1:
            return f"No notes, tasks, or meeting transcripts found on {scope.dates[0]}."
        return "No notes, tasks, or meeting transcripts found on the selected days."
    return "No notes, tasks, or meeting transcripts found for this space yet."


def combine_query(base: dict[str, Any], extra: dict[str, Any] | None) -> dict[str, Any]:
    if not extra:
        return base
    return {"$and": [base, extra]}
