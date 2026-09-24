from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from services.chat.planner import ChatQueryPlan
from services.chat.write_tools import WRITE_TOOL_NAMES, ChatWriteToolRegistry, tool_catalog_for_prompt
from services.chat.writes import ChatWriteStore
from services.daily_briefing.timezones import DEFAULT_TIMEZONE, date_key_for
from services.llm.models import LLMMessage, StructuredLLMRequest
from services.llm.router import LLMCapability, get_llm_router
from services.reminders.dates import format_date_label


WriteAction = Literal[
    "none",
    "create_task",
    "create_note",
    "create_space",
    "create_reminder",
    "create_event",
    "update_task",
    "update_note",
    "update_space",
    "delete_task",
    "delete_note",
    "delete_space",
]


class ChatWriteSpec(BaseModel):
    action: WriteAction = Field(
        default="none",
        description=(
            "Correct write tool for this request. "
            "create_task = trackable commitment / to-do to complete. "
            "create_note = information to store for reference (not a to-do). "
            "Infer from user meaning and context; plannedAction is only a hint you may correct. "
            "Use none if the user is not asking to mutate workspace data."
        ),
    )
    title: str | None = None
    newTitle: str | None = None
    entityId: str | None = None
    description: str | None = ""
    spaceNameHint: str | None = None
    dueDate: str | None = None
    priority: str | None = "Medium"
    dateKey: str | None = None
    dateLabel: str | None = None
    timeLabel: str | None = None
    startTimeLabel: str | None = None
    endTimeLabel: str | None = None
    location: str | None = ""
    repeat: str = "once"
    missingFields: list[str] = Field(default_factory=list)
    ready: bool = False


async def extract_write_spec(
    question: str,
    plan: ChatQueryPlan,
    space_ids: list[str] | None = None,
) -> ChatWriteSpec:
    today = date_key_for(datetime.now().astimezone(), DEFAULT_TIMEZONE)
    planned_action: WriteAction = getattr(plan, "writeAction", "none") or "none"
    if planned_action == "none":
        return ChatWriteSpec(action="none")

    # Emergency field parse only if the extraction LLM fails — never decides task vs note.
    emergency = _fallback_spec(question, planned_action, today, plan)

    provider, model = get_llm_router().route(LLMCapability.NORMALIZATION)
    request = StructuredLLMRequest(
        model=model,
        temperature=0,
        max_tokens=700,
        schema_name="ChatWriteSpec",
        messages=[
            LLMMessage(
                role="system",
                content=(
                    "You extract arguments for one Buddy write tool and may correct the tool choice. "
                    f"{tool_catalog_for_prompt()} "
                    "Return JSON only. Do not invent values the user did not imply. "
                    f"Today's dateKey is {today}. Convert relative dates to YYYY-MM-DD. "
                    "Times must be like '3:30 PM'. "
                    "Set `action` to the best matching tool from allowedTools based on the user's intent. "
                    "plannedAction is a hint from an earlier planner — correct it when the meaning clearly "
                    "points to a different tool (especially create_task vs create_note). "
                    "create_task = a commitment or to-do the user intends to complete or track. "
                    "create_note = information, thoughts, or reference material to store — not a to-do. "
                    "Infer across languages and phrasings; do not use fixed keyword rules. "
                    "For create_task/create_note, title is required; description optional. "
                    "For create_space, title is the space name (min 3 chars). "
                    "For create_reminder, title + dateKey + timeLabel required. "
                    "For create_event, title + dateKey + startTimeLabel required; endTimeLabel optional. "
                    "For delete_task/delete_note/delete_space: set title (lookup name) or entityId. "
                    "For update_*: set title (lookup name) or entityId to identify the item; "
                    "put the replacement name in newTitle when renaming; "
                    "set description/dueDate/priority/dateKey only for fields the user wants changed. "
                    "Set missingFields to any required fields still missing. "
                    "Set ready=true only when all required fields are present. "
                    "spaceNameHint only if the user named a specific space/workspace."
                ),
            ),
            LLMMessage(
                role="user",
                content=json.dumps(
                    {
                        "question": question,
                        "plannedAction": planned_action,
                        "understoodRequest": plan.understoodRequest,
                        "spaceIdProvided": bool(space_ids),
                        "todayDateKey": today,
                        "planDateKey": plan.dateKey,
                        "allowedTools": list(WRITE_TOOL_NAMES),
                    },
                    ensure_ascii=True,
                ),
            ),
        ],
    )
    try:
        spec = await provider.generate_structured(request, ChatWriteSpec)
        return _normalize_spec(spec, planned_action, today, plan)
    except Exception:
        return emergency


async def execute_write_action(
    *,
    user_id: str,
    question: str,
    plan: ChatQueryPlan,
    space_ids: list[str] | None = None,
    pending_write: dict[str, Any] | None = None,
    auth_token: str | None = None,
    store: ChatWriteStore | None = None,
) -> dict[str, Any] | None:
    """Dispatch a create/update/delete write tool (Node API) and return a tools-style result."""
    pending = pending_write if pending_write and pending_write.get("type") == "complete_write" else None
    forced_action = str(pending.get("writeAction") or "") if pending else ""
    if forced_action in WRITE_TOOL_NAMES:
        plan.writeAction = forced_action  # type: ignore[assignment]

    # Writes are LLM-planned. Do not invent actions with static heuristics.
    if getattr(plan, "writeAction", "none") == "none" and not pending:
        return None

    store = store or ChatWriteStore(auth_token=auth_token)
    tools = ChatWriteToolRegistry(store)
    spec = await extract_write_spec(question, plan, space_ids)
    if pending:
        spec = _merge_partial_spec(spec, pending.get("partialSpec") or {}, question, str(pending.get("writeAction") or "none"))
    if spec.action == "none" and pending:
        spec.action = forced_action  # type: ignore[assignment]
    if spec.action == "none" or not tools.has(spec.action):
        return None

    # Keep plan in sync if the extractor corrected the tool (e.g. note → task).
    plan.writeAction = spec.action  # type: ignore[assignment]
    space_id: str | None = None
    space_label: str | None = None
    if spec.action in {"create_task", "create_note"}:
        space_id, space_label, spaces = await store.resolve_space_id(
            user_id,
            preferred_space_ids=space_ids,
            space_name_hint=spec.spaceNameHint,
        )
        if not space_id:
            if not spaces:
                return {
                    "context": "Write tool blocked: user has no spaces yet.",
                    "answer": (
                        "You don't have a space yet. Create one first - for example: "
                        '"create a space called Work" - then I can add tasks and notes there.'
                    ),
                    "direct": True,
                    "pending_action": None,
                    "tool_name": spec.action,
                }
            return {
                "context": "Write tool needs a space selection.\n" + _format_space_choices(spaces),
                "answer": (
                    "Which space should I use?\n"
                    + "\n".join(f"{i}. {s.get('label') or s.get('spaceId')}" for i, s in enumerate(spaces, start=1))
                    + "\nReply with the space name or number."
                ),
                "direct": True,
                "pending_action": {
                    "type": "select_option",
                    "optionKind": "spaces",
                    "originalQuestion": str(pending.get("originalQuestion") if pending else question),
                    "plan": {**plan.model_dump(), "writeAction": spec.action},
                    "options": [
                        {
                            "index": index,
                            "label": str(space.get("label") or space.get("spaceId") or ""),
                            "value": str(space.get("spaceId") or ""),
                        }
                        for index, space in enumerate(spaces, start=1)
                    ],
                },
                "tool_name": spec.action,
            }

    missing = _required_missing(spec)
    if missing:
        clean_missing = _dedupe_missing_labels(missing)
        return {
            "context": f"Write tool {spec.action} incomplete. Missing: {', '.join(clean_missing)}",
            "answer": _missing_prompt(spec.action, clean_missing),
            "direct": True,
            "pending_action": {
                "type": "complete_write",
                "writeAction": spec.action,
                "originalQuestion": str(pending.get("originalQuestion") if pending else question),
                "partialSpec": spec.model_dump(),
                "missingFields": clean_missing,
            },
            "tool_name": spec.action,
        }

    had_entity_id = bool((spec.entityId or "").strip())
    entity_id = (spec.entityId or "").strip() or None
    lookup_title = (spec.title or "").strip() or None
    entity_kind = spec.action.split("_", 1)[1] if "_" in spec.action else ""

    if spec.action.startswith(("update_", "delete_")) and not entity_id and lookup_title:
        if entity_kind in {"task", "note", "space"}:
            entity_id = await store.resolve_entity_id(
                user_id=user_id,
                entity_kind=entity_kind,  # type: ignore[arg-type]
                title=lookup_title,
                space_ids=space_ids,
            )
        if not entity_id:
            return {
                "context": f"Write tool {spec.action} could not resolve entity from title.",
                "answer": (
                    f"I couldn't find a {entity_kind or 'item'} named **{lookup_title}**. "
                    "Check the name and try again."
                ),
                "direct": True,
                "pending_action": {
                    "type": "complete_write",
                    "writeAction": spec.action,
                    "originalQuestion": str(pending.get("originalQuestion") if pending else question),
                    "partialSpec": spec.model_dump(),
                    "missingFields": ["title"],
                },
                "tool_name": spec.action,
            }

    update_title: str | None = None
    update_description: str | None = None
    update_due: str | None = None
    update_priority: str | None = None
    update_date_key: str | None = None
    if spec.action.startswith("update_"):
        if (spec.newTitle or "").strip():
            update_title = (spec.newTitle or "").strip()
        elif had_entity_id and (spec.title or "").strip():
            update_title = (spec.title or "").strip()
        if (spec.description or "").strip():
            update_description = (spec.description or "").strip()
        if spec.dueDate:
            update_due = spec.dueDate
        if (spec.priority or "").strip() and (spec.priority or "").strip() != "Medium":
            update_priority = (spec.priority or "").strip()
        elif (spec.priority or "").strip() in {"High", "Low", "Medium"} and "priority" in question.lower():
            update_priority = (spec.priority or "").strip()
        if spec.dateKey and spec.action == "update_note":
            update_date_key = spec.dateKey

    tool_args = {
        "user_id": user_id,
        "space_id": space_id,
        "space_label": space_label,
        "entity_id": entity_id,
        "title": update_title if spec.action.startswith("update_") else (spec.title or lookup_title),
        "description": update_description if spec.action.startswith("update_") else (spec.description or ""),
        "due_date": update_due if spec.action.startswith("update_") else spec.dueDate,
        "priority": update_priority if spec.action.startswith("update_") else (spec.priority or "Medium"),
        "date_key": update_date_key if spec.action.startswith("update_") else spec.dateKey,
        "date_label": spec.dateLabel,
        "time_label": spec.timeLabel,
        "start_time_label": spec.startTimeLabel,
        "end_time_label": spec.endTimeLabel,
        "location": spec.location or "",
        "repeat": spec.repeat or "once",
    }
    # For update, omit unchanged optional fields so Node only patches provided keys.
    if spec.action.startswith("update_"):
        for key in ("title", "description", "due_date", "priority", "date_key"):
            if tool_args.get(key) is None:
                tool_args.pop(key, None)
        if "description" not in tool_args and not (spec.description or "").strip():
            pass  # already omitted
    result = await tools.run(spec.action, tool_args)
    if not result.ok:
        return {
            "context": f"Write tool {spec.action} failed: {result.error}",
            "answer": result.error or "I couldn't save that just now. Please try again in a moment.",
            "direct": True,
            "pending_action": {
                "type": "complete_write",
                "writeAction": spec.action,
                "originalQuestion": str(pending.get("originalQuestion") if pending else question),
                "partialSpec": spec.model_dump(),
                "missingFields": [],
            },
            "tool_name": spec.action,
        }

    return {
        "context": (
            f"Write tool completed: {result.tool_name}\n"
            f"{json.dumps(result.data or {}, ensure_ascii=True, default=str)}"
        ),
        "answer": result.summary,
        "direct": True,
        "pending_action": None,
        "tool_name": result.tool_name,
    }


def _merge_partial_spec(
    spec: ChatWriteSpec,
    partial: dict[str, Any],
    reply: str,
    action: str,
) -> ChatWriteSpec:
    merged = ChatWriteSpec.model_validate({**partial, **{k: v for k, v in spec.model_dump().items() if v not in (None, "", [], "none")}})
    if action and merged.action == "none":
        merged.action = action  # type: ignore[assignment]
    reply_text = (reply or "").strip()
    # Follow-up replies like "preet demo" are usually the missing title/name.
    if reply_text and not (merged.title or "").strip():
        if action in {
            "create_space",
            "create_task",
            "create_note",
            "create_reminder",
            "create_event",
            "update_task",
            "update_note",
            "update_space",
            "delete_task",
            "delete_note",
            "delete_space",
        }:
            if len(reply_text) <= 80 and not _looks_like_new_command(reply_text):
                merged.title = reply_text
    if reply_text and action.startswith("update_") and not (merged.newTitle or "").strip():
        if (merged.title or "").strip() and (merged.entityId or "").strip():
            if len(reply_text) <= 80 and not _looks_like_new_command(reply_text):
                # If identity already known, a short reply is likely the new title/description.
                if "description" in " ".join(merged.missingFields).lower():
                    merged.description = reply_text
                else:
                    merged.newTitle = reply_text
        elif (merged.title or "").strip() and not _has_update_change(merged):
            if len(reply_text) <= 80 and not _looks_like_new_command(reply_text):
                merged.newTitle = reply_text
    if reply_text and action == "create_reminder" and not merged.timeLabel:
        extracted = _extract_time(reply_text)
        if extracted:
            merged.timeLabel = extracted
    if reply_text and action == "create_event" and not merged.startTimeLabel:
        extracted = _extract_time(reply_text)
        if extracted:
            merged.startTimeLabel = extracted
    merged.missingFields = _required_missing(merged)
    merged.ready = not merged.missingFields
    return merged


def _identity_ready(spec: ChatWriteSpec) -> bool:
    if spec.action.startswith(("update_", "delete_")):
        return bool((spec.entityId or "").strip() or (spec.title or "").strip())
    return bool((spec.title or "").strip())


def _has_update_change(spec: ChatWriteSpec) -> bool:
    if (spec.newTitle or "").strip():
        return True
    if (spec.entityId or "").strip() and (spec.title or "").strip():
        return True
    if (spec.description or "").strip():
        return True
    if spec.dueDate:
        return True
    if spec.dateKey and spec.action == "update_note":
        return True
    priority = (spec.priority or "").strip()
    if priority in {"High", "Low"}:
        return True
    return False


def _required_missing(spec: ChatWriteSpec) -> list[str]:
    missing: list[str] = []
    title = (spec.title or "").strip()
    entity_id = (spec.entityId or "").strip()

    if spec.action.startswith("delete_"):
        if not entity_id and not title:
            missing.append("space name" if spec.action == "delete_space" else "title")
        return _dedupe_missing_labels(missing)

    if spec.action.startswith("update_"):
        if not entity_id and not title:
            missing.append("space name" if spec.action == "update_space" else "title")
        elif not _has_update_change(spec):
            missing.append("what to change")
        return _dedupe_missing_labels(missing)

    if not title:
        if spec.action == "create_space":
            missing.append("space name")
        else:
            missing.append("title")
    elif spec.action == "create_space" and len(title) < 3:
        missing.append("space name (at least 3 characters)")
    if spec.action == "create_reminder":
        if not spec.dateKey:
            missing.append("date")
        if not spec.timeLabel:
            missing.append("time")
    if spec.action == "create_event":
        if not spec.dateKey:
            missing.append("date")
        if not spec.startTimeLabel:
            missing.append("start time")
    return _dedupe_missing_labels(missing)


def _dedupe_missing_labels(missing: list[str]) -> list[str]:
    aliases = {
        "title": "name",
        "space name": "name",
        "space name (at least 3 characters)": "name (at least 3 characters)",
    }
    # Prefer a single clear label for space naming.
    normalized: list[str] = []
    seen: set[str] = set()
    for item in missing:
        key = aliases.get(item, item)
        if key in seen:
            continue
        seen.add(key)
        if key == "name":
            normalized.append("space name" if "space" in " ".join(missing).lower() or item == "space name" else "title")
        elif key == "name (at least 3 characters)":
            normalized.append("space name (at least 3 characters)")
        else:
            normalized.append(item)
    # If both title and space name somehow remain, keep space name only for create_space flows.
    if "space name" in normalized and "title" in normalized:
        normalized = [item for item in normalized if item != "title"]
    return normalized


def _looks_like_new_command(text: str) -> bool:
    lowered = text.lower()
    return bool(
        re.search(
            r"\b(create|add|make|set|schedule|remind|list|show|summarize|summarise|delete|remove|trash|edit|update|rename)\b",
            lowered,
        )
    )


def _normalize_spec(spec: ChatWriteSpec, planned_action: WriteAction, today: str, plan: ChatQueryPlan) -> ChatWriteSpec:
    # Prefer the extractor LLM's action when it is a valid write tool.
    if spec.action not in WRITE_TOOL_NAMES:
        spec.action = planned_action if planned_action in WRITE_TOOL_NAMES else "none"
    action = spec.action
    if not spec.dateKey and plan.dateKey:
        spec.dateKey = plan.dateKey
    if not spec.dueDate and plan.temporalScope in {"today", "tomorrow"} and action == "create_task":
        spec.dueDate = plan.dateKey or today
    if spec.dateKey and not spec.dateLabel:
        try:
            spec.dateLabel = format_date_label(datetime.fromisoformat(spec.dateKey))
        except Exception:
            pass
    spec.missingFields = _required_missing(spec)
    spec.ready = not spec.missingFields
    return spec


def _fallback_spec(question: str, action: WriteAction, today: str, plan: ChatQueryPlan) -> ChatWriteSpec:
    rename_from, rename_to = _extract_rename_pair(question)
    title = rename_from or _extract_title(question, action)
    date_key = plan.dateKey
    time_label = _extract_time(question)
    priority = _extract_priority(question) if action in {"create_task", "update_task"} else "Medium"
    return _normalize_spec(
        ChatWriteSpec(
            action=action,
            title=title,
            newTitle=rename_to,
            description="",
            dueDate=date_key if action == "create_task" else None,
            priority=priority,
            dateKey=date_key or (today if action in {"create_reminder", "create_event"} and "today" in question.lower() else None),
            timeLabel=time_label if action == "create_reminder" else None,
            startTimeLabel=time_label if action == "create_event" else None,
            ready=False,
        ),
        action,
        today,
        plan,
    )


def _infer_action(_question: str) -> WriteAction:
    """Deprecated keyword path — writes must come from the planner LLM."""
    return "none"


def _extract_rename_pair(question: str) -> tuple[str | None, str | None]:
    match = re.search(
        r"rename\s+(?:the\s+)?(?:task|note|space|workspace|project)\s+[\"']?(.+?)[\"']?\s+to\s+[\"']?(.+?)[\"']?\s*$",
        question.strip(),
        re.I,
    )
    if not match:
        return None, None
    old = match.group(1).strip(" .")[:80]
    new = match.group(2).strip(" .")[:80]
    return (old or None), (new or None)


def _extract_title(question: str, action: WriteAction) -> str | None:
    patterns = [
        r"(?:delete|remove|trash|discard|edit|update|rename|change|modify)\s+(?:the\s+)?(?:task|note|space|workspace|project)\s+(?:called|named|titled|for|:)?\s*[\"']?(.+?)[\"']?(?:\s+to\s+.+)?$",
        r"(?:create|add|make|set|save|schedule|new)\s+(?:a\s+|an\s+)?(?:task|note|space|workspace|project|reminder|event|meeting)\s+(?:called|named|for|to|:)?\s*[\"']?(.+?)[\"']?$",
        r"remind me to\s+(.+)$",
        r"note(?:\s+about|\s+on|:)\s+(.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, question.strip(), re.I)
        if match:
            title = match.group(1).strip(" .")
            # Strip trailing time/date fluff lightly
            title = re.split(r"\b(today|tomorrow|at\s+\d|on\s+\d|to\s+)", title, maxsplit=1, flags=re.I)[0].strip(" ,.-")
            if action.startswith("update_") or action.startswith("delete_"):
                # Drop trailing "description/priority" clauses used for the change payload.
                title = re.split(
                    r"\b(description|priority|due|date|body)\b",
                    title,
                    maxsplit=1,
                    flags=re.I,
                )[0].strip(" ,.-")
            if title:
                return title[:80]
    return None


def _extract_priority(question: str) -> str:
    lowered = question.lower()
    if re.search(r"\b(high|urgent)\b", lowered):
        return "High"
    if re.search(r"\blow\b", lowered):
        return "Low"
    return "Medium"


def _extract_time(question: str) -> str | None:
    match = re.search(r"\b((1[0-2]|0?[1-9])(?::([0-5]\d))?\s?(am|pm))\b", question, re.I)
    if not match:
        match = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", question)
        if not match:
            return None
        hour = int(match.group(1))
        minute = int(match.group(2))
        period = "AM" if hour < 12 else "PM"
        hour12 = hour % 12 or 12
        return f"{hour12}:{minute:02d} {period}"
    hour = int(match.group(2))
    minute = int(match.group(3) or 0)
    period = match.group(4).upper()
    return f"{hour}:{minute:02d} {period}"


def _missing_prompt(action: WriteAction, missing: list[str]) -> str:
    if action == "create_space":
        return "Sure - what should I name this space? (at least 3 characters)"
    if action == "create_task" and missing == ["title"]:
        return "Sure - what should I call this task?"
    if action == "create_note" and missing == ["title"]:
        return "Sure - what should I title this note?"
    if action == "create_reminder":
        return f"I can set that reminder. Please share: {', '.join(missing)}."
    if action == "create_event":
        return f"I can schedule that event. Please share: {', '.join(missing)}."
    if action.startswith("delete_"):
        kind = action.replace("delete_", "")
        return f"Which {kind} should I delete? Share the name."
    if action.startswith("update_"):
        if "what to change" in missing:
            return "What should I change? Share the new title, description, due date, or priority."
        kind = action.replace("update_", "")
        return f"Which {kind} should I update? Share the name."
    labels = {
        "create_task": "task",
        "create_note": "note",
        "create_space": "space",
        "create_reminder": "reminder",
        "create_event": "event",
    }
    kind = labels.get(action, "item")
    return f"I can create that {kind}. Please share the missing detail(s): {', '.join(missing)}."


def _format_space_choices(spaces: list[dict[str, Any]]) -> str:
    lines = ["Available spaces:"]
    for index, space in enumerate(spaces, start=1):
        lines.append(f"{index}. {space.get('label') or space.get('spaceId')}")
    return "\n".join(lines)
