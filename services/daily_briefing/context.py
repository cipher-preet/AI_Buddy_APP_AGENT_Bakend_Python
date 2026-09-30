"""Deterministic day context for the v2 briefing: refs, ranked backlog, agenda, timeline."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from services.daily_briefing.prepare import filter_and_order_transcripts
from services.daily_briefing.schemas import ActivityBundle, BriefingStats
from services.daily_briefing.timezones import ensure_utc, local_day_bounds_utc, to_local

TRANSCRIPT_CHAR_LIMIT = 2000
USER_CHAT_CHAR_LIMIT = 700
ASSISTANT_CHAT_CHAR_LIMIT = 400
NOTE_CHAR_LIMIT = 280
TASK_DETAIL_CHAR_LIMIT = 200
NOTE_LIMIT = 30
_TIME_LABEL = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*([AaPp][Mm])?\s*$")
_DATE_PREFIX = re.compile(r"^(\d{4}-\d{2}-\d{2})")


@dataclass(frozen=True)
class RefEntry:
    sourceType: str
    sourceId: str
    text: str
    timestamp: datetime | None = None
    label: str = ""


@dataclass
class BriefingContext:
    date_key: str
    plan_date_key: str
    timezone_name: str
    refs: dict[str, RefEntry] = field(default_factory=dict)
    timeline: list[dict[str, Any]] = field(default_factory=list)
    open_tasks: list[dict[str, Any]] = field(default_factory=list)
    completed_tasks: list[dict[str, Any]] = field(default_factory=list)
    agenda: list[dict[str, Any]] = field(default_factory=list)
    day_agenda: list[dict[str, Any]] = field(default_factory=list)
    notes: list[dict[str, Any]] = field(default_factory=list)
    stats: BriefingStats = field(default_factory=BriefingStats)
    corpus: str = ""

    @property
    def empty(self) -> bool:
        return not (
            self.timeline
            or self.notes
            or self.completed_tasks
            or self.agenda
            or self.day_agenda
            or any(task["touched"] for task in self.open_tasks)
        )

    def facts(self) -> dict[str, Any]:
        return {
            "dateKey": self.date_key,
            "planDateKey": self.plan_date_key,
            "planWeekday": date.fromisoformat(self.plan_date_key).strftime("%A"),
            "timezone": self.timezone_name,
            "stats": self.stats.model_dump(),
            "planAgenda": [
                {key: item[key] for key in ("ref", "kind", "time", "endTime", "title", "location", "detail") if item.get(key)}
                for item in self.agenda
            ],
            "openTasks": [
                {
                    key: value
                    for key, value in {
                        "ref": task["ref"],
                        "title": task["title"],
                        "detail": task["detail"][:TASK_DETAIL_CHAR_LIMIT],
                        "due": task["due"],
                        "dueStatus": task["dueStatus"],
                        "priority": task["priority"],
                        "space": task["space"],
                        "touchedOnDate": task["touched"],
                    }.items()
                    if value not in ("", None, False)
                }
                for task in self.open_tasks
            ],
            "completedTasks": [{"ref": item["ref"], "title": item["title"]} for item in self.completed_tasks],
            "dayAgenda": [
                {key: item[key] for key in ("ref", "kind", "time", "title") if item.get(key)}
                for item in self.day_agenda
            ],
            "notes": [
                {"ref": note["ref"], "title": note["title"], "detail": note["detail"]}
                for note in self.notes
            ],
        }


def time_label_minutes(label: str | None) -> int | None:
    match = _TIME_LABEL.match(label or "")
    if not match:
        return None
    hour, minute = int(match.group(1)), int(match.group(2))
    period = (match.group(3) or "").upper()
    if period == "PM" and hour < 12:
        hour += 12
    if period == "AM" and hour == 12:
        hour = 0
    return hour * 60 + minute


def _local_time_label(value: datetime | None, timezone_name: str) -> str:
    if not isinstance(value, datetime):
        return ""
    return to_local(value, timezone_name).strftime("%I:%M %p").lstrip("0")


def _parse_due(value: Any, timezone_name: str) -> date | None:
    if isinstance(value, datetime):
        return to_local(value, timezone_name).date()
    if isinstance(value, date):
        return value
    match = _DATE_PREFIX.match(str(value or ""))
    if not match:
        return None
    try:
        return date.fromisoformat(match.group(1))
    except ValueError:
        return None


def normalize_priority(value: str) -> str:
    text = (value or "").casefold()
    if any(token in text for token in ("high", "urgent", "critical", "p1")):
        return "high"
    if any(token in text for token in ("medium", "normal", "p2")):
        return "medium"
    if any(token in text for token in ("low", "p3")):
        return "low"
    return ""


def _in_period(value: Any, start: datetime, end: datetime) -> bool:
    return isinstance(value, datetime) and start <= ensure_utc(value) < end


def _due_status(due: date | None, plan_day: date) -> str:
    if due is None:
        return "none"
    days = (due - plan_day).days
    if days < 0:
        return "overdue"
    if days == 0:
        return "today"
    if days <= 2:
        return "soon"
    return "later"


_DUE_SCORE = {"overdue": 40, "today": 35, "soon": 20, "later": 5, "none": 0}
_PRIORITY_SCORE = {"high": 15, "medium": 6, "low": 0, "": 0}


def _task_score(task: dict[str, Any], plan_day: date) -> float:
    score = _DUE_SCORE[task["dueStatus"]] + _PRIORITY_SCORE[task["priority"]]
    if task["dueStatus"] == "overdue" and task["dueDate"] is not None:
        score += min((plan_day - task["dueDate"]).days, 14)
    if task["touched"]:
        score += 10
    return score


class _RefAllocator:
    def __init__(self, refs: dict[str, RefEntry]):
        self.refs = refs
        self.counters: dict[str, int] = {}
        self.by_source: dict[tuple[str, str], str] = {}

    def allocate(self, prefix: str, entry: RefEntry) -> str:
        key = (entry.sourceType, entry.sourceId)
        existing = self.by_source.get(key)
        if existing:
            return existing
        self.counters[prefix] = self.counters.get(prefix, 0) + 1
        ref = f"{prefix}{self.counters[prefix]}"
        self.refs[ref] = entry
        self.by_source[key] = ref
        return ref


def build_context(
    bundle: ActivityBundle,
    date_key: str,
    timezone_name: str,
    open_task_limit: int,
) -> BriefingContext:
    period_start, period_end = local_day_bounds_utc(date_key, timezone_name)
    plan_key = bundle.planDateKey or date_key
    plan_day = date.fromisoformat(plan_key)
    ctx = BriefingContext(date_key=date_key, plan_date_key=plan_key, timezone_name=timezone_name)
    alloc = _RefAllocator(ctx.refs)
    corpus: list[str] = []

    ranked: list[dict[str, Any]] = []
    for item in bundle.openTasks:
        due = _parse_due(item.get("dueDate"), timezone_name)
        task = {
            "id": item["id"],
            "title": item["title"],
            "detail": item.get("detail") or "",
            "dueDate": due,
            "due": due.isoformat() if due else (item.get("dueText") or ""),
            "dueStatus": _due_status(due, plan_day),
            "priority": normalize_priority(item.get("priority") or ""),
            "space": item.get("space") or "",
            "touched": _in_period(item.get("updatedAt"), period_start, period_end)
            or _in_period(item.get("createdAt"), period_start, period_end),
            "updatedAt": item.get("updatedAt"),
        }
        task["score"] = _task_score(task, plan_day)
        ranked.append(task)
    ranked.sort(
        key=lambda task: (task["score"], ensure_utc(task["updatedAt"]).timestamp() if isinstance(task["updatedAt"], datetime) else 0),
        reverse=True,
    )
    for task in ranked[:open_task_limit]:
        task["ref"] = alloc.allocate("T", RefEntry("task", task["id"], task["title"], label="Task"))
        corpus.append(f"{task['title']} {task['detail']}")
        ctx.open_tasks.append(task)

    open_ids = {task["id"] for task in bundle.openTasks}
    for item in bundle.tasks:
        if item["id"] in open_ids or not item["title"]:
            continue
        status = str(item.get("status") or "").lower()
        operation = str(item.get("operation") or "").upper()
        if status in {"completed", "done"} or operation == "DONE":
            ref = alloc.allocate("K", RefEntry("task", item["id"], item["title"], label="Completed task"))
            ctx.completed_tasks.append({"ref": ref, "id": item["id"], "title": item["title"]})
            corpus.append(item["title"])

    for kind, items, prefix in (("meeting", bundle.planEvents, "E"), ("reminder", bundle.planReminders, "R")):
        for item in items:
            if not item.get("title"):
                continue
            source_type = "calendar_event" if kind == "meeting" else "reminder"
            ref = alloc.allocate(prefix, RefEntry(source_type, item["id"], item["title"], label=kind.title()))
            ctx.agenda.append(
                {
                    "ref": ref,
                    "id": item["id"],
                    "kind": kind,
                    "time": item.get("timeLabel") or "",
                    "endTime": item.get("endTimeLabel") or "",
                    "title": item["title"],
                    "location": item.get("location") or "",
                    "detail": (item.get("detail") or "")[:TASK_DETAIL_CHAR_LIMIT],
                }
            )
            corpus.append(f"{item['title']} {item.get('detail') or ''}")
    ctx.agenda.sort(key=lambda item: time_label_minutes(item["time"]) if time_label_minutes(item["time"]) is not None else 24 * 60)

    for kind, items in (("meeting", bundle.events), ("reminder", bundle.reminders)):
        for item in items:
            if not item.get("title"):
                continue
            source_type = "calendar_event" if kind == "meeting" else "reminder"
            ref = alloc.allocate("P", RefEntry(source_type, item["id"], item["title"], label=kind.title()))
            ctx.day_agenda.append({"ref": ref, "kind": kind, "time": item.get("timeLabel") or "", "title": item["title"]})
            corpus.append(item["title"])

    for item in bundle.notes[:NOTE_LIMIT]:
        if not (item.get("title") or item.get("detail")):
            continue
        detail = (item.get("detail") or "")[:NOTE_CHAR_LIMIT]
        ref = alloc.allocate("N", RefEntry("note", item["id"], f"{item.get('title') or ''} {detail}".strip(), label="Note"))
        ctx.notes.append({"ref": ref, "title": item.get("title") or "", "detail": detail})
        corpus.append(f"{item.get('title') or ''} {item.get('detail') or ''}")

    stream: list[dict[str, Any]] = []
    for item in bundle.transcripts:
        stream.append(
            {
                "sourceType": "transcript",
                "sourceId": item["id"],
                "text": item["text"][:TRANSCRIPT_CHAR_LIMIT],
                "createdAt": ensure_utc(item["createdAt"]) if isinstance(item.get("createdAt"), datetime) else period_start,
                "source": "voice",
            }
        )
    for item in bundle.chats:
        limit = USER_CHAT_CHAR_LIMIT if item["role"] == "user" else ASSISTANT_CHAT_CHAR_LIMIT
        stream.append(
            {
                "sourceType": "chat",
                "sourceId": item["id"],
                "text": item["text"][:limit],
                "createdAt": ensure_utc(item["createdAt"]) if isinstance(item.get("createdAt"), datetime) else period_start,
                "source": "chat",
                "role": item["role"],
            }
        )
    for item in filter_and_order_transcripts(stream):
        prefix = "S" if item["sourceType"] == "transcript" else "C"
        label = "Voice capture" if prefix == "S" else "Buddy chat"
        ref = alloc.allocate(prefix, RefEntry(item["sourceType"], item["sourceId"], item["text"], item["createdAt"], label))
        entry = {
            "ref": ref,
            "source": item["source"],
            "time": _local_time_label(item["createdAt"], timezone_name),
            "text": item["text"],
            "createdAt": item["createdAt"],
        }
        if item.get("role"):
            entry["role"] = item["role"]
        ctx.timeline.append(entry)
        corpus.append(item["text"])

    ctx.stats = BriefingStats(
        completedTasks=len(ctx.completed_tasks),
        openTasks=len(bundle.openTasks),
        overdueTasks=sum(1 for task in ranked if task["dueStatus"] == "overdue"),
        dueToday=sum(1 for task in ranked if task["dueStatus"] == "today"),
        meetings=sum(1 for item in ctx.agenda if item["kind"] == "meeting"),
        reminders=sum(1 for item in ctx.agenda if item["kind"] == "reminder"),
        conversations=len({item.get("conversationId") for item in bundle.transcripts if item.get("conversationId")}),
        chatMessages=sum(1 for item in bundle.chats if item["role"] == "user"),
        notes=len(bundle.notes),
    )
    ctx.corpus = " ".join(corpus).casefold()
    return ctx


def timeline_payload(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in item.items() if key != "createdAt"} for item in items]
