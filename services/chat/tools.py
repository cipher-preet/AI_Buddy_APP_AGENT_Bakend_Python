from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any

from services.chat.planner import ChatQueryPlan
from services.conversation.repository import ConversationRepository
from services.db.mongo import get_database


class ChatToolRunner:
    def __init__(self, repository: ConversationRepository | None = None):
        self.repository = repository or ConversationRepository(get_database())

    async def build_context(self, question: str, user_id: str, space_id: str | None, plan: ChatQueryPlan | None = None) -> str:
        result = await self.run(question, user_id, space_id, plan)
        return str(result["context"])

    async def run(
        self,
        question: str,
        user_id: str,
        space_id: str | None,
        plan: ChatQueryPlan | None = None,
    ) -> dict[str, Any]:
        plan = plan or ChatQueryPlan(understoodRequest=question, searchQueries=[question])
        if plan.optionKind == "spaces" and (plan.responseMode == "list_options" or plan.requiresSpace):
            spaces = await self.repository.list_user_spaces(user_id)
            answer = _space_options_answer(spaces)
            return {
                "context": _format_spaces(spaces),
                "answer": answer,
                "direct": True,
                "pending_action": _space_pending_action(question, spaces, plan),
            }

        if not plan.useStructuredTools:
            return {
                "context": "Structured tools skipped by query plan.",
                "answer": None,
                "direct": False,
                "pending_action": None,
            }

        if plan.responseMode == "ask_clarifying_question" and plan.missingInfoQuestion and not _plan_needs_space(plan):
            answer = _english_missing_info_answer(plan)
            return {
                "context": f"Clarifying question requested by query plan: {answer}",
                "answer": answer,
                "direct": True,
                "pending_action": None,
            }

        if not space_id:
            if not _plan_needs_space(plan):
                return {
                    "context": "Structured tools skipped: no spaceId was provided and this query does not require workspace tools.",
                    "answer": None,
                    "direct": False,
                    "pending_action": None,
                }
            spaces = await self.repository.list_user_spaces(user_id)
            return {
                "context": "\n\n".join(
                    [
                        "Structured tools need a selected space before reading saved workspace data.",
                        _format_spaces(spaces),
                    ]
                ),
                "answer": _space_options_answer(spaces),
                "direct": True,
                "pending_action": _space_pending_action(question, spaces, plan),
            }

        if plan.responseMode == "ask_clarifying_question" and plan.missingInfoQuestion:
            answer = _english_missing_info_answer(plan)
            return {
                "context": f"Clarifying question requested by query plan: {answer}",
                "answer": answer,
                "direct": True,
                "pending_action": None,
            }

        today = datetime.now().astimezone().date().isoformat()
        focus = set(plan.toolFocus or ["tasks", "notes", "summaries", "space_memory"])
        wants_notes = "notes" in focus
        wants_tasks = "tasks" in focus or "planning" in focus
        wants_summary = bool({"summaries", "space_memory", "planning"} & focus)
        wants_decisions = "decisions" in focus
        wants_issues = "issues" in focus

        include_all = not plan.toolFocus
        sections: list[str] = [f"Today: {today}"]

        memory = None
        user_profile: dict[str, Any] | None = None
        stats: dict[str, int] | None = None
        summaries: list[dict[str, Any]] = []
        tasks: list[dict[str, Any]] = []
        notes: list[dict[str, Any]] = []
        staged_tasks: list[dict[str, Any]] = []
        staged_notes: list[dict[str, Any]] = []
        staged_decisions: list[dict[str, Any]] = []
        staged_issues: list[dict[str, Any]] = []

        calls: dict[str, Any] = {
            "user_profile": _maybe_call(self.repository, "get_user_profile", user_id),
            "memory": self.repository.get_space_memory(user_id, space_id),
            "stats": _maybe_call(self.repository, "get_space_stats", user_id, space_id),
        }
        if wants_summary or include_all:
            calls["summaries"] = self.repository.list_recent_summaries(user_id, space_id, limit=8)
        if wants_tasks or wants_summary or include_all:
            calls["tasks"] = self.repository.list_tasks(user_id, space_id, limit=100)
            calls["staged_tasks"] = _maybe_call(self.repository, "list_staged_tasks", user_id, space_id, limit=50)
        if wants_notes or wants_summary or include_all:
            calls["notes"] = self.repository.list_recent_notes(user_id, space_id, limit=50)
            calls["staged_notes"] = _maybe_call(self.repository, "list_staged_notes", user_id, space_id, limit=50)
        if wants_summary or wants_decisions or wants_issues or include_all:
            calls["staged_decisions"] = _maybe_call(self.repository, "list_staged_decisions", user_id, space_id, limit=50)
            calls["staged_issues"] = _maybe_call(self.repository, "list_staged_issues", user_id, space_id, limit=50)

        results = await asyncio.gather(*calls.values())
        data = dict(zip(calls.keys(), results))

        user_profile = data.get("user_profile")
        memory = data["memory"]
        stats = data.get("stats")
        summaries = data.get("summaries") or []
        tasks = data.get("tasks") or []
        notes = data.get("notes") or []
        staged_tasks = data.get("staged_tasks") or []
        staged_notes = data.get("staged_notes") or []
        staged_decisions = data.get("staged_decisions") or []
        staged_issues = data.get("staged_issues") or []

        sections.append(_format_user_profile(user_profile))
        sections.append(_format_space_stats(stats))
        if wants_summary or include_all:
            sections.append(_format_space_memory(memory.model_dump(by_alias=True)))
            sections.append(_format_summaries(summaries))
        if wants_summary or wants_decisions or include_all:
            sections.append(_format_staged_decisions(staged_decisions))
        if wants_summary or wants_issues or include_all:
            sections.append(_format_staged_issues(staged_issues))
        if wants_tasks or wants_summary or include_all:
            sections.append(_format_tasks(tasks, today))
            sections.append(_format_staged_tasks(staged_tasks))
        if wants_notes or wants_summary or include_all:
            sections.append(_format_notes(notes))
            sections.append(_format_staged_notes(staged_notes))

        context = "\n\n".join(section for section in sections if section.strip())
        answer = _direct_answer(
            temporal_scope=plan.temporalScope,
            today=today,
            memory=memory.model_dump(by_alias=True),
            summaries=summaries,
            tasks=tasks,
            notes=notes,
            staged_tasks=staged_tasks,
            staged_notes=staged_notes,
            wants_summary=wants_summary,
            wants_tasks=wants_tasks,
            wants_notes=wants_notes,
            include_all=include_all,
        )
        return {
            "context": context,
            "answer": answer,
            "direct": bool(answer and (plan.directToolAnswerAllowed or not plan.useVectorSearch)),
            "pending_action": None,
        }


def _direct_answer(
    temporal_scope: str,
    today: str,
    memory: dict[str, Any],
    summaries: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    notes: list[dict[str, Any]],
    staged_tasks: list[dict[str, Any]],
    staged_notes: list[dict[str, Any]],
    wants_summary: bool,
    wants_tasks: bool,
    wants_notes: bool,
    include_all: bool,
) -> str | None:
    if include_all:
        return None

    parts: list[str] = []
    unfinished = _unfinished_tasks(tasks)
    due_today = _due_today_tasks(unfinished, today)

    if wants_summary:
        parts.append(_summary_answer(memory, summaries, tasks, notes, today))
    if wants_tasks:
        if temporal_scope == "today":
            parts.append(_today_tasks_answer(due_today, unfinished, staged_tasks))
        elif temporal_scope == "tomorrow":
            parts.append(_tomorrow_tasks_answer(unfinished, staged_tasks))
        else:
            parts.append(_tasks_answer(tasks, unfinished, staged_tasks))
    if wants_notes:
        parts.append(_notes_answer(notes, staged_notes))

    return "\n\n".join(part for part in parts if part.strip()) or None


def _summary_answer(
    memory: dict[str, Any],
    summaries: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    notes: list[dict[str, Any]],
    today: str,
) -> str:
    lines = ["Here is your workspace summary:"]
    summary = str(memory.get("currentSummary") or "").strip()
    if summary:
        lines.append(summary)

    today_summaries = [item for item in summaries if _is_today(item.get("createdAt"), today)]
    if today_summaries:
        lines.append("Today:")
        lines.extend(f"- {item.get('summary') or ''}" for item in today_summaries[:5])
    elif summaries:
        lines.append("Recent:")
        lines.extend(f"- {item.get('summary') or ''}" for item in summaries[:3])
    else:
        lines.append("No conversation summaries are saved yet for this space.")

    due_today = _due_today_tasks(_unfinished_tasks(tasks), today)
    if due_today:
        lines.append(f"Tasks due today: {len(due_today)}")
        lines.extend(_public_task_line(task) for task in due_today[:10])
    else:
        lines.append("Tasks due today: none found.")

    today_notes = [note for note in notes if _is_today(note.get("updatedAt") or note.get("createdAt"), today)]
    if today_notes:
        lines.append(f"Notes updated today: {len(today_notes)}")
        lines.extend(_public_note_line(note) for note in today_notes[:5])
    return "\n".join(lines)


def _today_tasks_answer(
    due_today: list[dict[str, Any]],
    unfinished: list[dict[str, Any]],
    staged_tasks: list[dict[str, Any]],
) -> str:
    if due_today:
        lines = [f"You have {len(due_today)} task(s) due today:"]
        lines.extend(_public_task_line(task) for task in due_today)
        return "\n".join(lines)
    staged_due_today = _due_today_tasks(staged_tasks, datetime.now().astimezone().date().isoformat())
    if staged_due_today:
        lines = [f"I found {len(staged_due_today)} staged task(s) due today:"]
        lines.extend(_public_staged_task_line(task) for task in staged_due_today)
        return "\n".join(lines)
    if unfinished:
        lines = ["I do not see any tasks due today, but you have unfinished tasks:"]
        lines.extend(_public_task_line(task) for task in unfinished[:10])
        return "\n".join(lines)
    if staged_tasks:
        lines = ["I do not see published tasks due today, but I found staged tasks:"]
        lines.extend(_public_staged_task_line(task) for task in staged_tasks[:10])
        return "\n".join(lines)
    return "I do not see any saved tasks due today or unfinished tasks in this space."


def _tomorrow_tasks_answer(unfinished: list[dict[str, Any]], staged_tasks: list[dict[str, Any]]) -> str:
    tomorrow = (datetime.now().astimezone().date() + timedelta(days=1)).isoformat()
    due_tomorrow = [
        task
        for task in unfinished
        if task.get("dueDateResolved") == tomorrow or str(task.get("dueDateText") or "").strip().lower() == "tomorrow"
    ]
    staged_due_tomorrow = [
        task
        for task in staged_tasks
        if task.get("dueDateResolved") == tomorrow or str(task.get("dueDateText") or "").strip().lower() == "tomorrow"
    ]
    if due_tomorrow:
        lines = [f"You have {len(due_tomorrow)} task(s) due tomorrow:"]
        lines.extend(_public_task_line(task) for task in due_tomorrow)
        return "\n".join(lines)
    if staged_due_tomorrow:
        lines = [f"I found {len(staged_due_tomorrow)} staged task(s) due tomorrow:"]
        lines.extend(_public_staged_task_line(task) for task in staged_due_tomorrow)
        return "\n".join(lines)
    if unfinished:
        lines = ["I do not see tasks explicitly due tomorrow. Here are unfinished tasks you can plan from:"]
        lines.extend(_public_task_line(task) for task in unfinished[:10])
        return "\n".join(lines)
    if staged_tasks:
        lines = ["I do not see published unfinished tasks, but I found staged tasks you can plan from:"]
        lines.extend(_public_staged_task_line(task) for task in staged_tasks[:10])
        return "\n".join(lines)
    return "I do not see any saved unfinished tasks in this space to plan for tomorrow."


def _tasks_answer(tasks: list[dict[str, Any]], unfinished: list[dict[str, Any]], staged_tasks: list[dict[str, Any]]) -> str:
    if not tasks and not staged_tasks:
        return "I do not see any saved tasks in this space yet."
    lines = [f"I found {len(tasks)} task(s) in this space. Unfinished: {len(unfinished)}."]
    if unfinished:
        lines.append("Unfinished tasks:")
        lines.extend(_public_task_line(task) for task in unfinished[:20])
    completed = [task for task in tasks if task.get("status") == "completed"]
    if completed:
        lines.append("Completed tasks:")
        lines.extend(_public_task_line(task) for task in completed[:10])
    if staged_tasks:
        lines.append(f"Staged tasks waiting in the app: {len(staged_tasks)}")
        lines.extend(_public_staged_task_line(task) for task in staged_tasks[:20])
    return "\n".join(lines)


def _notes_answer(notes: list[dict[str, Any]], staged_notes: list[dict[str, Any]]) -> str:
    if not notes and not staged_notes:
        return "I do not see any saved notes in this space yet."
    lines = [f"I found {len(notes)} published note(s) and {len(staged_notes)} staged note(s):"]
    if notes:
        lines.append("Published notes:")
        lines.extend(_public_note_line(note) for note in notes[:20])
    if staged_notes:
        lines.append("Staged notes:")
        lines.extend(_public_note_line(note) for note in staged_notes[:20])
    return "\n".join(lines)


def _format_space_memory(memory: dict[str, Any]) -> str:
    lines = ["Tool: space_memory"]
    summary = str(memory.get("currentSummary") or "").strip()
    lines.append(f"Current summary: {summary or 'No current space summary found.'}")
    lines.extend(_format_list("Important facts", memory.get("importantFacts") or []))
    lines.extend(_format_list("Important decisions", memory.get("importantDecisions") or []))
    return "\n".join(lines)


def _format_user_profile(profile: dict[str, Any] | None) -> str:
    lines = ["Tool: user_profile"]
    if not profile:
        lines.append("No user profile found.")
        return "\n".join(lines)
    name = _display_text(profile.get("name")) or "Unnamed user"
    provider = _display_text(profile.get("provider")) or "unknown"
    onboarding = profile.get("onboarding") if isinstance(profile.get("onboarding"), dict) else {}
    profession = _display_text(onboarding.get("profession")) if onboarding else ""
    usage_goal = _display_text(onboarding.get("usageGoal")) if onboarding else ""
    lines.append(f"Name: {name}")
    lines.append(f"Provider: {provider}")
    if profession:
        lines.append(f"Profession: {profession}")
    if usage_goal:
        lines.append(f"Usage goal: {usage_goal}")
    return "\n".join(lines)


def _format_space_stats(stats: dict[str, int] | None) -> str:
    lines = ["Tool: space_stats"]
    if not stats:
        lines.append("No workspace stats found.")
        return "\n".join(lines)
    lines.append(
        "Counts: "
        f"tasks={stats.get('tasksCount', 0)}, "
        f"completed_tasks={stats.get('doneTasksCount', 0)}, "
        f"notes={stats.get('notesCount', 0)}, "
        f"staged_tasks={stats.get('stagedTasksCount', 0)}, "
        f"staged_completed_tasks={stats.get('stagedDoneTasksCount', 0)}, "
        f"staged_notes={stats.get('stagedNotesCount', 0)}, "
        f"completion={stats.get('completionPercentage', 0)}%"
    )
    return "\n".join(lines)


def _format_spaces(spaces: list[dict[str, Any]]) -> str:
    lines = ["Tool: spaces"]
    if not spaces:
        lines.append("No spaces found for this user.")
        return "\n".join(lines)
    for index, space in enumerate(spaces, start=1):
        lines.append(f"{index}. {space.get('label') or 'Untitled space'}")
    return "\n".join(lines)


def _space_options_answer(spaces: list[dict[str, Any]]) -> str:
    if not spaces:
        return (
            "I do not see any spaces for your account yet. "
            "Create or select a space first, then I can help with its tasks, notes, and summaries."
        )
    lines = ["Which space should I use?"]
    lines.append("Available spaces:")
    for index, space in enumerate(spaces, start=1):
        label = space.get("label") or space.get("spaceId")
        lines.append(f"{index}. {label}")
    lines.append("Reply with the space name or number.")
    return "\n".join(lines)


def _english_missing_info_answer(plan: ChatQueryPlan) -> str:
    if plan.optionKind == "spaces" or plan.requiresSpace:
        return "Please choose a space first so I can continue."
    return "I need one more detail before I can answer. Please clarify what you mean."


def _plan_needs_space(plan: ChatQueryPlan) -> bool:
    return plan.requiresSpace or plan.optionKind == "spaces" or bool(plan.toolFocus)


def _space_pending_action(question: str, spaces: list[dict[str, Any]], plan: ChatQueryPlan) -> dict[str, Any] | None:
    if not spaces:
        return None
    return {
        "type": "select_option",
        "optionKind": "spaces",
        "originalQuestion": question,
        "plan": plan.model_dump(),
        "options": [
            {
                "index": index,
                "label": str(space.get("label") or space.get("spaceId") or ""),
                "value": str(space.get("spaceId") or ""),
            }
            for index, space in enumerate(spaces, start=1)
        ],
    }


def _format_summaries(summaries: list[dict[str, Any]]) -> str:
    lines = ["Tool: recent_conversation_summaries"]
    if not summaries:
        lines.append("No recent summaries found.")
        return "\n".join(lines)
    for item in summaries:
        created = _date_text(item.get("createdAt"))
        topics = ", ".join(str(topic) for topic in item.get("topics") or [])
        lines.append(f"- {created}: {item.get('summary') or ''} Topics: {topics or 'none'}")
    return "\n".join(lines)


def _format_tasks(tasks: list[dict[str, Any]], today: str) -> str:
    lines = ["Tool: tasks"]
    if not tasks:
        lines.append("No tasks found.")
        return "\n".join(lines)

    unfinished = _unfinished_tasks(tasks)
    due_today = _due_today_tasks(unfinished, today)
    lines.append(f"Counts: total={len(tasks)}, unfinished={len(unfinished)}, due_today={len(due_today)}")
    if due_today:
        lines.append("Due today:")
        lines.extend(_format_task_line(task) for task in due_today[:20])
    if unfinished:
        lines.append("Unfinished:")
        lines.extend(_format_task_line(task) for task in unfinished[:50])
    completed = [task for task in tasks if task.get("status") == "completed"]
    if completed:
        lines.append("Recently completed:")
        lines.extend(_format_task_line(task) for task in completed[:10])
    return "\n".join(lines)


def _format_notes(notes: list[dict[str, Any]]) -> str:
    lines = ["Tool: notes"]
    if not notes:
        lines.append("No notes found.")
        return "\n".join(lines)
    for note in notes:
        updated = _date_text(note.get("updatedAt") or note.get("createdAt"))
        lines.append(f"- {updated}: {note.get('title') or 'Untitled'} - {note.get('body') or ''}")
    return "\n".join(lines)


def _format_staged_tasks(tasks: list[dict[str, Any]]) -> str:
    lines = ["Tool: staged_tasks"]
    if not tasks:
        lines.append("No staged tasks found.")
        return "\n".join(lines)
    for task in tasks:
        title = task.get("title") or "Untitled"
        status = task.get("operation") or task.get("status") or "pending review"
        due = task.get("dueDateResolved") or task.get("dueDate") or task.get("dueDateText") or "no due date"
        body = str(task.get("description") or task.get("body") or "").strip()
        suffix = f" Body: {body}" if body else ""
        lines.append(f"- [{status}] {title}; due={due}.{suffix}")
    return "\n".join(lines)


def _format_staged_notes(notes: list[dict[str, Any]]) -> str:
    lines = ["Tool: staged_notes"]
    if not notes:
        lines.append("No staged notes found.")
        return "\n".join(lines)
    for note in notes:
        title = note.get("title") or "Untitled"
        body = str(note.get("body") or "").strip()
        lines.append(f"- {title}: {body}" if body else f"- {title}")
    return "\n".join(lines)


def _format_staged_decisions(decisions: list[dict[str, Any]]) -> str:
    lines = ["Tool: staged_decisions"]
    if not decisions:
        lines.append("No staged decisions found.")
        return "\n".join(lines)
    for decision in decisions:
        title = decision.get("title") or "Untitled decision"
        status = decision.get("status") or "unknown"
        rationale = str(decision.get("rationale") or decision.get("body") or "").strip()
        suffix = f" Rationale: {rationale}" if rationale else ""
        lines.append(f"- [{status}] {title}.{suffix}")
    return "\n".join(lines)


def _format_staged_issues(issues: list[dict[str, Any]]) -> str:
    lines = ["Tool: staged_issues"]
    if not issues:
        lines.append("No staged issues found.")
        return "\n".join(lines)
    for issue in issues:
        title = issue.get("title") or "Untitled issue"
        kind = issue.get("kind") or issue.get("status") or "issue"
        body = str(issue.get("body") or issue.get("description") or "").strip()
        suffix = f" Details: {body}" if body else ""
        lines.append(f"- [{kind}] {title}.{suffix}")
    return "\n".join(lines)


def _format_task_line(task: dict[str, Any]) -> str:
    due = task.get("dueDateResolved") or task.get("dueDateText") or "no due date"
    owner = task.get("ownerText") or "no owner"
    status = task.get("status") or "unknown"
    body = str(task.get("body") or "").strip()
    suffix = f" Body: {body}" if body else ""
    return f"- [{status}] {task.get('title') or 'Untitled'}; owner={owner}; due={due}.{suffix}"


def _public_task_line(task: dict[str, Any]) -> str:
    due = task.get("dueDateResolved") or task.get("dueDateText") or "no due date"
    owner = task.get("ownerText") or "no owner"
    status = task.get("status") or "unknown"
    body = str(task.get("body") or "").strip()
    detail = f" - {body}" if body else ""
    return f"- {task.get('title') or 'Untitled'} ({status}, owner: {owner}, due: {due}){detail}"


def _public_staged_task_line(task: dict[str, Any]) -> str:
    due = task.get("dueDateResolved") or task.get("dueDate") or task.get("dueDateText") or "no due date"
    owner = task.get("ownerText") or "no owner"
    status = task.get("operation") or task.get("status") or "pending review"
    body = str(task.get("description") or task.get("body") or "").strip()
    detail = f" - {body}" if body else ""
    return f"- {task.get('title') or 'Untitled'} ({status}, owner: {owner}, due: {due}){detail}"


def _public_note_line(note: dict[str, Any]) -> str:
    title = note.get("title") or "Untitled"
    body = str(note.get("body") or "").strip()
    return f"- {title}: {body}" if body else f"- {title}"


def _unfinished_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [task for task in tasks if task.get("status") in {"pending", "in_progress", "blocked", "needs_confirmation"}]


def _due_today_tasks(tasks: list[dict[str, Any]], today: str) -> list[dict[str, Any]]:
    return [task for task in tasks if task.get("dueDateResolved") == today or _same_date_text(task.get("dueDateText"), today)]


def _format_list(label: str, values: list[Any]) -> list[str]:
    if not values:
        return [f"{label}: none"]
    return [f"{label}:"] + [f"- {value}" for value in values]


def _date_text(value: Any) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    return str(value or "unknown date")


def _same_date_text(value: Any, today: str) -> bool:
    return str(value or "").strip().lower() in {"today", today}


def _is_today(value: Any, today: str) -> bool:
    if isinstance(value, datetime):
        return value.astimezone().date().isoformat() == today
    return str(value or "").startswith(today)


async def _maybe_call(repository: Any, method_name: str, *args: Any, **kwargs: Any) -> Any:
    method = getattr(repository, method_name, None)
    if method is None:
        return None
    return await method(*args, **kwargs)


def _display_text(value: Any) -> str:
    return str(value or "").strip()
