"""Schedule fields the goal workspace reads: dates, categories, coach note, metric label."""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any


def _text(value: Any) -> str:
    return str(value or "").strip()


def _horizon(goal: dict[str, Any], plan: dict[str, Any], max_index: int) -> int:
    source = f"{_text(goal.get('timeframe'))} {_text((plan.get('successDefinition') or {}).get('deadlineLabel'))}".lower()
    if re.search(r"\b(a year|1 year|12 months|within a year)\b", source):
        return max(365, max_index)
    years = re.search(r"(\d+)\s*year", source)
    if years:
        return max(int(years.group(1)) * 365, max_index, 365)
    months = re.search(r"(\d+)\s*month", source)
    if months:
        return max(int(months.group(1)) * 30, max_index, 30)
    weeks = re.search(r"(\d+)\s*week", source)
    if weeks:
        return max(int(weeks.group(1)) * 7, max_index, 7)
    days = re.search(r"(\d+)\s*day", source)
    if days:
        return max(int(days.group(1)), max_index, 1)
    if "no fixed" in source:
        return max(90, max_index)
    return max(max_index, 90, 1)


def _category(title: str, focus: str) -> str:
    source = f"{title} {focus}".lower()
    if any(word in source for word in ("outreach", "message", "email")):
        return "Outreach"
    if any(word in source for word in ("product hunt", "twitter", "thread", "launch")):
        return "Launch prep"
    if any(word in source for word in ("landing", "headline", "webflow", "carrd")):
        return "Landing page"
    if any(word in source for word in ("onboard", "demo")):
        return "Onboarding"
    if any(word in source for word in ("interview", "persona", "audience", "research")):
        return "Research"
    return (focus or "Step")[:28]


def annotate_execution(plan: dict[str, Any], goal: dict[str, Any]) -> dict[str, Any]:
    """Fill schedule metadata without wiping progress the user already logged."""
    days = list(plan.get("days") or [])
    max_index = max([int(day.get("dayIndex") or 0) for day in days] or [len(days)])
    horizon = _horizon(goal, plan, max_index)
    created = _text(goal.get("createdAt"))[:10]
    try:
        start = date.fromisoformat(created) if created else date.today()
    except ValueError:
        start = date.today()
    if not plan.get("startedOn"):
        plan["startedOn"] = start.isoformat()
    else:
        try:
            start = date.fromisoformat(_text(plan.get("startedOn"))[:10])
        except ValueError:
            start = date.today()
    for index, day in enumerate(days, start=1):
        day_index = int(day.get("dayIndex") or index)
        if not day.get("id"):
            day["id"] = f"day-{day_index}"
        if not _text(day.get("scheduledDate")):
            day["scheduledDate"] = (start + timedelta(days=max(day_index - 1, 0))).isoformat()
        if not _text(day.get("category")):
            day["category"] = _category(_text(day.get("title")), _text(day.get("focus")))
        if not day.get("minutes"):
            hours = day.get("hours")
            day["minutes"] = int(round(float(hours) * 60)) if isinstance(hours, (int, float)) and hours else 45
        if not _text(day.get("status")):
            day["status"] = "open"
        day["partOf"] = day.get("partOf") or day.get("category") or ""
    if not _text(plan.get("coachNote")):
        plan["coachNote"] = _text(plan.get("immediateNextStep") or plan.get("recommendation"))[:320]
    plan["horizonDays"] = horizon
    return plan
