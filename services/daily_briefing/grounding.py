"""Deterministic grounding: turn an LLM draft (or a fallback) into a safe, evidence-backed briefing."""

from __future__ import annotations

import re
from datetime import date
from typing import Iterable

from services.daily_briefing.context import BriefingContext, RefEntry
from services.daily_briefing.schemas import (
    AgendaCard,
    BriefingDraft,
    BriefingItem,
    DailyBriefingSynthesis,
    DraftFocus,
    DraftFollowUp,
    DraftInsight,
    DraftItem,
    EvidenceRef,
    FocusItem,
    InsightCard,
    MeetingCard,
    TaskCard,
    WindowDigest,
)
from services.daily_briefing.timezones import to_local

LIMITS = {
    "focus": 5,
    "tasks": 12,
    "followUps": 8,
    "risks": 5,
    "highlights": 6,
    "decisions": 6,
    "completed": 10,
    "moments": 6,
    "insights": 4,
    "missedCandidates": 6,
    "people": 12,
    "topics": 10,
}
_SPEAKER = re.compile(r"\bspeaker[\s_-]*\d+\b", re.IGNORECASE)
_REF_GROUP = re.compile(r"\s*[\(\[]\s*(?:refs?:?\s*)?[A-Z]\d{1,4}(?:\s*(?:,|;|/|&|and)\s*[A-Z]\d{1,4})*\s*[\)\]]")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")
_WHITESPACE = re.compile(r"\s+")
_REF = re.compile(r"^[A-Z]\d+$")
_SELF_NAMES = {"you", "user", "me", "i", "myself", "buddy", "assistant", "ai"}
_PRIORITIES = {"high", "medium", "low"}
_EXCERPT_LIMIT = 220


def clean_text(value: str | None, limit: int) -> str:
    text = _REF_GROUP.sub("", str(value or ""))
    text = _WHITESPACE.sub(" ", text).strip()
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _SPEAKER.sub("a participant", text)
    if text.startswith("a participant"):
        text = "A" + text[1:]
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip(",;:-")
    return f"{cut}…"


def _key(value: str) -> str:
    return re.sub(r"[^\w]+", " ", value.casefold()).strip()


def _evidence(refs: Iterable[str], ctx: BriefingContext) -> list[EvidenceRef]:
    seen: set[str] = set()
    evidence: list[EvidenceRef] = []
    for raw in refs or []:
        ref = str(raw or "").strip().upper()
        entry = ctx.refs.get(ref) if _REF.match(ref) else None
        if entry is None or ref in seen:
            continue
        seen.add(ref)
        evidence.append(EvidenceRef(sourceType=entry.sourceType, sourceId=entry.sourceId, timestamp=entry.timestamp))
    return evidence


def _first_entry(refs: Iterable[str], ctx: BriefingContext) -> RefEntry | None:
    for raw in refs or []:
        entry = ctx.refs.get(str(raw or "").strip().upper())
        if entry is not None:
            return entry
    return None


def _items(drafts: Iterable[DraftItem], ctx: BriefingContext, kind: str, seen: set[str] | None = None) -> list[BriefingItem]:
    seen = seen if seen is not None else set()
    result: list[BriefingItem] = []
    for draft in drafts:
        title = clean_text(draft.title, 140)
        evidence = _evidence(draft.refs, ctx)
        if not title or not evidence or _key(title) in seen:
            continue
        seen.add(_key(title))
        detail = clean_text(draft.detail, 360)
        person = clean_text(getattr(draft, "person", ""), 60)
        if person and _is_known_person(person, ctx) and person.casefold() not in title.casefold():
            detail = f"{person}: {detail}" if detail else person
        result.append(BriefingItem(id=f"{kind}-{len(result) + 1}", title=title, detail=detail, evidence=evidence))
        if len(result) >= LIMITS[kind]:
            break
    return result


def _is_known_person(name: str, ctx: BriefingContext) -> bool:
    folded = name.casefold().strip()
    if not folded or folded in _SELF_NAMES or _SPEAKER.search(name):
        return False
    return folded in ctx.corpus


def _names(values: Iterable[str], ctx: BriefingContext, limit: int, require_known: bool) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        name = clean_text(value, 60)
        if not name or _key(name) in seen or _SPEAKER.search(name) or name.casefold() in _SELF_NAMES:
            continue
        if require_known and not _is_known_person(name, ctx):
            continue
        seen.add(_key(name))
        result.append(name)
        if len(result) >= limit:
            break
    return result


def _task_reason(task: dict) -> str:
    status = task["dueStatus"]
    if status == "overdue":
        return f"Overdue since {task['due']}"
    if status == "today":
        return "Due today"
    if status == "soon":
        return f"Due {task['due']}"
    if task["priority"] == "high":
        return "High priority"
    if task["touched"]:
        return "You worked on this yesterday"
    return ""


def _due_label(task: dict, plan_day: date) -> str:
    status = task["dueStatus"]
    if status == "overdue":
        return "Overdue"
    if status == "today":
        return "Due today"
    if task["dueDate"] is not None:
        days = (task["dueDate"] - plan_day).days
        return "Due tomorrow" if days == 1 else f"Due {task['dueDate'].strftime('%b %d')}"
    return task["due"]


def _task_cards(draft: BriefingDraft, ctx: BriefingContext) -> tuple[list[TaskCard], list[BriefingItem]]:
    by_ref = {task["ref"]: task for task in ctx.open_tasks}
    llm: dict[str, tuple[str, str]] = {}
    order: list[str] = []
    for item in draft.taskPriorities:
        ref = item.ref.strip().upper()
        if ref in by_ref and ref not in llm:
            priority = item.priority.strip().lower()
            llm[ref] = (priority if priority in _PRIORITIES else "", clean_text(item.reason, 160))
            order.append(ref)
    order += [task["ref"] for task in ctx.open_tasks if task["ref"] not in llm]
    plan_day = date.fromisoformat(ctx.plan_date_key)
    cards: list[TaskCard] = []
    pending: list[BriefingItem] = []
    for ref in order[: LIMITS["tasks"]]:
        task = by_ref[ref]
        llm_priority, llm_reason = llm.get(ref, ("", ""))
        priority = llm_priority or task["priority"] or ("high" if task["dueStatus"] in {"overdue", "today"} else "medium")
        reason = llm_reason or _task_reason(task)
        due = _due_label(task, plan_day)
        meta = " · ".join(part for part in (priority.title(), due, task["space"]) if part)
        cards.append(
            TaskCard(
                id=task["id"],
                title=clean_text(task["title"], 200),
                meta=meta,
                priority=priority,
                due=due,
                dueStatus=task["dueStatus"],
                space=task["space"],
                reason=reason,
            )
        )
        pending.append(
            BriefingItem(
                id=task["id"],
                title=clean_text(task["title"], 200),
                detail=reason,
                evidence=[EvidenceRef(sourceType="task", sourceId=task["id"])],
            )
        )
    return cards, pending


def _with_task_reasons(focus: list[FocusItem], tasks: list[TaskCard]) -> list[FocusItem]:
    reasons = {card.id: card.reason for card in tasks}
    return [
        item.model_copy(update={"why": reasons.get(item.taskId, "")}) if not item.why and item.taskId else item
        for item in focus
    ]


def _agenda(draft: BriefingDraft, ctx: BriefingContext) -> tuple[list[AgendaCard], list[MeetingCard]]:
    prep = {note.ref.strip().upper(): clean_text(note.prep, 200) for note in draft.agendaNotes if note.prep}
    agenda: list[AgendaCard] = []
    meetings: list[MeetingCard] = []
    for item in ctx.agenda:
        agenda.append(
            AgendaCard(
                id=item["id"],
                kind=item["kind"],
                time=item["time"],
                endTime=item["endTime"],
                title=item["title"],
                location=item["location"],
                prep=prep.get(item["ref"], ""),
            )
        )
        if item["kind"] == "meeting":
            meta = item["location"] or (f"Until {item['endTime']}" if item["endTime"] else "Calendar")
            meetings.append(MeetingCard(id=item["id"], time=item["time"], title=item["title"], meta=meta))
    return agenda, meetings


def _focus(drafts: list[DraftFocus], ctx: BriefingContext) -> list[FocusItem]:
    by_ref = {task["ref"]: task for task in ctx.open_tasks}
    seen: set[str] = set()
    result: list[FocusItem] = []
    for draft in drafts:
        task = by_ref.get(draft.taskRef.strip().upper())
        title = clean_text(draft.title, 120) or (clean_text(task["title"], 120) if task else "")
        if not title or _key(title) in seen or (task and task["id"] in seen):
            continue
        seen.add(_key(title))
        if task:
            seen.add(task["id"])
        result.append(
            FocusItem(
                id=f"focus-{len(result) + 1}",
                title=title,
                why=clean_text(draft.why, 200),
                timeHint=clean_text(draft.timeHint, 40),
                taskId=task["id"] if task else "",
            )
        )
        if len(result) >= LIMITS["focus"]:
            break
    return result


def _insights(drafts: list[DraftInsight], ctx: BriefingContext) -> list[InsightCard]:
    result: list[InsightCard] = []
    seen: set[str] = set()
    for draft in drafts:
        title = clean_text(draft.title, 120)
        body = clean_text(draft.body, 400)
        entry = _first_entry(draft.refs, ctx)
        if not title or not body or entry is None or _key(title) in seen:
            continue
        seen.add(_key(title))
        result.append(
            InsightCard(
                id=f"insight-{len(result) + 1}",
                source=entry.label,
                sourceType=entry.sourceType,
                title=title,
                body=body,
                excerpt=clean_text(entry.text, _EXCERPT_LIMIT),
                whyItMatters=clean_text(draft.whyItMatters, 240),
                capturedAt=to_local(entry.timestamp, ctx.timezone_name).strftime("%I:%M %p").lstrip("0") if entry.timestamp else "",
            )
        )
        if len(result) >= LIMITS["insights"]:
            break
    return result


def fallback_draft(ctx: BriefingContext, digests: list[WindowDigest] | None = None) -> BriefingDraft:
    digests = digests or []
    weekday = date.fromisoformat(ctx.plan_date_key).strftime("%A")
    stats = ctx.stats
    urgent = [task for task in ctx.open_tasks if task["dueStatus"] in {"overdue", "today"}]
    top = ctx.open_tasks[:3]

    plan_bits = []
    if stats.meetings:
        plan_bits.append(f"{stats.meetings} meeting{'s' if stats.meetings != 1 else ''}")
    if urgent:
        plan_bits.append(f"{len(urgent)} urgent task{'s' if len(urgent) != 1 else ''}")
    if stats.reminders:
        plan_bits.append(f"{stats.reminders} reminder{'s' if stats.reminders != 1 else ''}")
    headline = f"{weekday}: {', '.join(plan_bits)}" if plan_bits else f"Your plan for {weekday}"

    overview_parts: list[str] = []
    summaries = [digest.summary for digest in digests if digest.summary]
    if summaries:
        overview_parts.append(" ".join(summaries)[:500])
    done_bits = []
    if stats.completedTasks:
        done_bits.append(f"closed {stats.completedTasks} task{'s' if stats.completedTasks != 1 else ''}")
    if stats.conversations:
        done_bits.append(f"captured {stats.conversations} conversation{'s' if stats.conversations != 1 else ''}")
    if stats.notes:
        done_bits.append(f"saved {stats.notes} note{'s' if stats.notes != 1 else ''}")
    if done_bits:
        overview_parts.append(f"Yesterday you {', '.join(done_bits)}.")
    if top:
        overview_parts.append(f"Start with {top[0]['title']}" + (f", then {top[1]['title']}." if len(top) > 1 else "."))
    if stats.overdueTasks:
        overview_parts.append(f"{stats.overdueTasks} task{'s are' if stats.overdueTasks != 1 else ' is'} overdue.")

    work_items = [item for digest in digests for item in digest.workItems]
    return BriefingDraft(
        headline=headline,
        overview=" ".join(overview_parts),
        focus=[DraftFocus(title=task["title"], why=_task_reason(task), taskRef=task["ref"]) for task in top],
        risks=[
            DraftItem(title=f"Overdue: {task['title']}", detail=f"Was due {task['due']}", refs=[task["ref"]])
            for task in ctx.open_tasks
            if task["dueStatus"] == "overdue"
        ],
        followUps=[item for digest in digests for item in digest.followUps],
        decisions=[item for digest in digests for item in digest.decisions],
        moments=[item for digest in digests for item in digest.moments],
        completed=[
            DraftItem(title=item.title, detail=item.detail, refs=item.refs)
            for item in work_items
            if item.status.lower() in {"done", "completed"}
        ],
        missedCandidates=[
            DraftItem(title=item.title, detail=item.detail, refs=item.refs)
            for item in work_items
            if item.status.lower() not in {"done", "completed"}
        ],
        highlights=[DraftItem(title=note["title"] or note["detail"][:80], detail=note["detail"], refs=[note["ref"]]) for note in ctx.notes[:3]],
        people=[person for digest in digests for person in digest.people],
        topics=[topic for digest in digests for topic in digest.topics],
    )


def ground_briefing(draft: BriefingDraft, ctx: BriefingContext, digests: list[WindowDigest] | None = None) -> DailyBriefingSynthesis:
    backup = fallback_draft(ctx, digests)
    tasks, pending = _task_cards(draft, ctx)
    agenda, meetings = _agenda(draft, ctx)
    focus = _with_task_reasons(_focus(draft.focus, ctx) or _focus(backup.focus, ctx), tasks)

    completed_seen: set[str] = set()
    completed = _items(draft.completed, ctx, "completed", completed_seen)
    completed += _items(
        [DraftItem(title=item["title"], refs=[item["ref"]]) for item in ctx.completed_tasks],
        ctx,
        "completed",
        completed_seen,
    )[: max(0, LIMITS["completed"] - len(completed))]

    risks = _items(draft.risks, ctx, "risks") or _items(backup.risks, ctx, "risks")
    follow_ups = _items(draft.followUps, ctx, "followUps")
    decisions = _items(draft.decisions, ctx, "decisions")
    moments = _items(draft.moments, ctx, "moments")
    highlights = _items(draft.highlights, ctx, "highlights") or _items(backup.highlights, ctx, "highlights")

    return DailyBriefingSynthesis(
        headline=clean_text(draft.headline, 140) or clean_text(backup.headline, 140),
        overview=clean_text(draft.overview, 900) or clean_text(backup.overview, 900),
        focus=focus,
        agenda=agenda,
        meetings=meetings,
        tasks=tasks,
        pendingTasks=pending,
        risks=risks,
        followUps=follow_ups,
        decisions=decisions,
        importantMoments=moments,
        highlights=highlights,
        completed=completed,
        tomorrowFocus=[
            BriefingItem(
                id=item.id,
                title=item.title,
                detail=item.why,
                evidence=[EvidenceRef(sourceType="task", sourceId=item.taskId)] if item.taskId else [],
            )
            for item in focus
        ],
        insights=_insights(draft.insights, ctx),
        missedCandidates=_items(draft.missedCandidates, ctx, "missedCandidates"),
        people=_names(draft.people or backup.people, ctx, LIMITS["people"], require_known=True),
        topics=_names(draft.topics or backup.topics, ctx, LIMITS["topics"], require_known=False),
        stats=ctx.stats,
        planDateKey=ctx.plan_date_key,
    )
