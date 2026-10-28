"""Scheduler agent.

The strategist, advisor, execution writer, and critic decide the work.
This agent places that work on the deadline and writes what each step
actually involves. Step 2 stays step 2. Its date is a separate fact.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any

_DAY_PREFIX = re.compile(r"^(?:day|d)\s*\d+\s*[-–—:]\s*", re.I)


def _text(value: Any) -> str:
    return str(value or "").strip()


def clean_step_title(title: str) -> str:
    cleaned = _DAY_PREFIX.sub("", _text(title)).strip(" -–—:")
    return cleaned or _text(title) or "Next step"


def explain_step(title: str, tasks: list[str], done: str) -> str:
    """Use the actions the model wrote. Do not replace them with a template."""
    useful = [item.strip().rstrip(".") for item in tasks if item and item.strip().lower() != title.lower()]
    if len(useful) >= 2:
        second = useful[1][0].lower() + useful[1][1:]
        lead = f"{useful[0]}. Then {second}."
    elif useful:
        lead = f"{useful[0]}."
    else:
        lead = title
    finish = _text(done).rstrip(".")
    if finish and finish.lower() not in lead.lower() and finish.lower() != title.lower():
        return f"{lead} Done when {finish[0].lower()}{finish[1:]}."
    return lead


def calendar_slots(count: int, span: int) -> list[int]:
    """Even positions across the deadline. The last step lands on the last day."""
    if count <= 0:
        return []
    span = max(span, count)
    if count == 1:
        return [1]
    slots: list[int] = []
    last = 0
    for index in range(count):
        slot = 1 + round(index * (span - 1) / (count - 1))
        slot = min(span, max(slot, last + 1))
        slots.append(slot)
        last = slot
    slots[-1] = span
    return slots


def assign_step_schedule(plan: dict[str, Any], goal: dict[str, Any], span: int, label: str, start: date) -> None:
    """Number steps in order and put each one on its date inside the deadline."""
    del goal
    days = [day for day in (plan.get("days") or []) if isinstance(day, dict)]
    slots = calendar_slots(len(days), span)
    for index, (day, slot) in enumerate(zip(days, slots), start=1):
        title = clean_step_title(_text(day.get("title")) or f"Step {index}")
        day["title"] = title
        tasks = [
            _text(task.get("text"))
            for task in (day.get("tasks") or [])
            if isinstance(task, dict) and _text(task.get("text"))
        ]
        done = _text(day.get("definitionOfDone"))
        day["dayIndex"] = index
        day["scheduleDay"] = slot
        day["dateLabel"] = (start + timedelta(days=slot - 1)).strftime("%b %d").replace(" 0", " ")
        day["scheduledDate"] = (start + timedelta(days=slot - 1)).isoformat()
        written = _text(day.get("focus"))
        day["focus"] = written or explain_step(title, tasks, done)
        if not done:
            day["definitionOfDone"] = explain_step(title, tasks, "")
    if days:
        first = _text(days[0].get("focus"))
        if first and not _text(plan.get("coachNote")):
            plan["coachNote"] = first
        if first and not _text(plan.get("immediateNextStep")):
            plan["immediateNextStep"] = first
    trace = [str(item) for item in (plan.get("agentTrace") or [])]
    if "scheduler" not in trace:
        trace.append("scheduler")
    plan["agentTrace"] = trace
    plan["deadlineLabel"] = label
