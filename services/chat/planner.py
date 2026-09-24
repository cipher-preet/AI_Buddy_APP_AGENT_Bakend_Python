from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel, Field

from services.daily_briefing.timezones import DEFAULT_TIMEZONE, date_key_for
from services.llm.models import LLMMessage, StructuredLLMRequest
from services.llm.router import LLMCapability, get_llm_router


ResponseMode = Literal["answer", "ask_clarifying_question", "list_options"]
OptionKind = Literal["none", "spaces"]
ToolFocus = Literal[
    "tasks",
    "notes",
    "summaries",
    "space_memory",
    "planning",
    "decisions",
    "issues",
    "profile",
    "stats",
    "day_summary",
    "calendar",
    "meetings",
]
TemporalScope = Literal["today", "tomorrow", "yesterday", "all", "unspecified"]
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


class ChatQueryPlan(BaseModel):
    understoodRequest: str = Field(description="Short restatement of what the user wants.")
    responseMode: ResponseMode = "answer"
    requiresSpace: bool = False
    useStructuredTools: bool = True
    useVectorSearch: bool = True
    directToolAnswerAllowed: bool = False
    missingInfoQuestion: str | None = None
    optionKind: OptionKind = "none"
    toolFocus: list[ToolFocus] = Field(default_factory=list)
    temporalScope: TemporalScope = "unspecified"
    dateKey: str | None = None
    writeAction: WriteAction = Field(
        default="none",
        description=(
            "Workspace mutation to perform, or none. "
            "create_task: user wants a trackable commitment / to-do they intend to complete. "
            "create_note: user wants to store information for later reference (not a to-do). "
            "create_space: new workspace/project container. "
            "create_reminder: notify the user at a time. "
            "create_event: calendar meeting/appointment with time. "
            "update_*/delete_*: change or remove an existing task, note, or space. "
            "Choose from meaning and context — not from fixed keywords. "
            "If the user is only asking a question or chatting, use none."
        ),
    )
    searchQueries: list[str] = Field(default_factory=list)


async def plan_chat_query(
    question: str,
    space_id: str | None,
    space_ids: list[str] | None = None,
) -> ChatQueryPlan:
    provider, model = get_llm_router().route(LLMCapability.NORMALIZATION)
    has_space = bool(space_id) or bool(space_ids)
    today = date_key_for(datetime.now().astimezone(), DEFAULT_TIMEZONE)
    request = StructuredLLMRequest(
        model=model,
        temperature=0,
        max_tokens=900,
        schema_name="ChatQueryPlan",
        messages=[
            LLMMessage(
                role="system",
                content=(
                    "You are Buddy's query planner for a personal companion agent. "
                    "Buddy has access to the user's spaces, tasks, notes, meeting transcripts, "
                    "embeddings, calendar events, reminders, and conversation memory. "
                    "Understand the user's request before any answer is generated. "
                    "Decide which data sources are needed and whether missing information should be requested. "
                    "Do not answer the user's domain question. Return only planning JSON. "
                    "All user-visible text fields, including missingInfoQuestion, must be in English only.\n\n"
                    f"Today's dateKey is {today} (timezone {DEFAULT_TIMEZONE}). "
                    "When the user says today/todays, set temporalScope=today and dateKey to today's dateKey. "
                    "When they say yesterday, set temporalScope=yesterday and dateKey to yesterday. "
                    "When they say tomorrow, set temporalScope=tomorrow and dateKey to tomorrow.\n\n"
                    "Rules:\n"
                    "- First classify intent: casual chat, general knowledge/drafting, workspace query, "
                    "day/meeting report, space details, WRITE action (create/update/delete), or list workspace options.\n"
                    "- WRITE actions: decide writeAction yourself from the user's meaning. Allowed values: "
                    "create_task | create_note | create_space | create_reminder | create_event | "
                    "update_task | update_note | update_space | delete_task | delete_note | delete_space | none.\n"
                    "- Distinguishing create_task vs create_note (critical): "
                    "create_task = the user is committing to something they need to do or track as work; "
                    "create_note = the user wants to capture information, thoughts, or reference material "
                    "without treating it as a to-do. Infer from intent and context across languages and phrasings; "
                    "do not rely on any fixed word list. When both could fit, prefer the interpretation that "
                    "best matches whether the user expects a completable item or stored information.\n"
                    "- For any WRITE action set useStructuredTools=true, useVectorSearch=false, "
                    "directToolAnswerAllowed=true. For create_task/create_note: if spaceIdProvided=false, "
                    "set requiresSpace=true and optionKind=spaces; if spaceIdProvided=true, requiresSpace=false. "
                    "create_space, create_reminder, create_event, and all update_*/delete_* actions do NOT require "
                    "a space (space context helps for task/note lookup but is optional).\n"
                    "- For casual chat such as greetings or thanks, "
                    "set useStructuredTools=false and useVectorSearch=false and writeAction=none.\n"
                    "- For general knowledge, education, writing, or drafting that is NOT about the user's "
                    "saved data, set useStructuredTools=false and useVectorSearch=false.\n"
                    "- If the user asks to list, show, choose, or see available spaces/workspaces/projects, "
                    "set responseMode=list_options, optionKind=spaces, requiresSpace=false.\n"
                    "- Day-level requests like 'summarize my day', 'today's meeting report', "
                    "'what did I do today', 'daily briefing', or 'how was my day' must set "
                    "toolFocus including day_summary (plus tasks/notes/calendar/meetings as needed), "
                    "requiresSpace=false, useStructuredTools=true, useVectorSearch=true, "
                    "and directToolAnswerAllowed=false so the answer model can synthesize a polished reply.\n"
                    "- Meeting/report requests for a date should include meetings and day_summary in toolFocus "
                    "and keep useVectorSearch=true to pull transcript embeddings.\n"
                    "- Space overview requests ('tell me about this space', 'what's in my space', "
                    "'space details') need space_memory, summaries, tasks, notes, stats; "
                    "if no space is selected, ask clarifying with optionKind=spaces.\n"
                    "- If space context is already provided (spaceIdProvided=true), do NOT ask for a space "
                    "again; set requiresSpace=false and answer using that context.\n"
                    "- If the user asks about saved tasks/notes/summaries for a specific space and no space "
                    "is selected, set responseMode=ask_clarifying_question and optionKind=spaces.\n"
                    "- Day summaries do NOT require a space; they can use all of the user's activity for that date.\n"
                    "- For simple list reads like 'all notes' or 'today's unfinished tasks' with a space selected, "
                    "set useStructuredTools=true, useVectorSearch=false, directToolAnswerAllowed=true.\n"
                    "- If the request mentions a topic, person, decision, issue, why/how, or needs transcript "
                    "evidence, keep useVectorSearch=true.\n"
                    "- Set toolFocus from: tasks, notes, summaries, space_memory, planning, decisions, issues, "
                    "profile, stats, day_summary, calendar, meetings.\n"
                    "- Generate 1-5 concise semantic search queries preserving the user's meaning and date intent."
                ),
            ),
            LLMMessage(
                role="user",
                content=json.dumps(
                    {
                        "question": question,
                        "spaceIdProvided": has_space,
                        "spaceCount": len(space_ids or ([space_id] if space_id else [])),
                        "todayDateKey": today,
                    },
                    ensure_ascii=True,
                ),
            ),
        ],
    )
    try:
        plan = await provider.generate_structured(request, ChatQueryPlan)
        return _normalize_plan(plan, question, has_space, today)
    except Exception:
        return _fallback_plan(question, has_space, today)


def _normalize_plan(plan: ChatQueryPlan, question: str, has_space: bool, today: str) -> ChatQueryPlan:
    """Apply structural flags only. Never override the LLM's writeAction choice."""
    plan.dateKey = plan.dateKey or _date_key_for_scope(plan.temporalScope, today)
    if plan.writeAction != "none":
        plan.useStructuredTools = True
        plan.useVectorSearch = False
        plan.directToolAnswerAllowed = True
        plan.responseMode = "answer"
        if plan.writeAction in {"create_task", "create_note"}:
            plan.requiresSpace = not has_space
            if not has_space:
                plan.optionKind = "spaces"
                plan.responseMode = "ask_clarifying_question"
            else:
                plan.optionKind = "none"
                plan.requiresSpace = False
        else:
            # create_space/reminder/event and all update_*/delete_* — no space required
            plan.requiresSpace = False
            plan.optionKind = "none"
        if plan.writeAction in {"create_task", "update_task", "delete_task"} and not plan.toolFocus:
            plan.toolFocus = ["tasks"]
        elif plan.writeAction in {"create_note", "update_note", "delete_note"} and not plan.toolFocus:
            plan.toolFocus = ["notes"]
        if not plan.searchQueries:
            plan.searchQueries = [question]
        return plan
    if "day_summary" in plan.toolFocus or plan.temporalScope in {"today", "yesterday"} and _looks_like_day_query(question):
        plan.requiresSpace = False
        if plan.optionKind == "spaces" and plan.responseMode == "ask_clarifying_question":
            plan.responseMode = "answer"
            plan.optionKind = "none"
        plan.directToolAnswerAllowed = False
        if "day_summary" not in plan.toolFocus:
            plan.toolFocus = [*plan.toolFocus, "day_summary"]
    if has_space and plan.optionKind == "spaces" and plan.responseMode == "ask_clarifying_question":
        plan.responseMode = "answer"
        plan.optionKind = "none"
        plan.requiresSpace = False
    if not plan.searchQueries:
        plan.searchQueries = [question]
    return plan


def _fallback_plan(question: str, has_space: bool, today: str) -> ChatQueryPlan:
    lowered = question.lower()
    write_action = _fallback_write_action(lowered)
    if write_action != "none":
        needs_space = write_action in {"create_task", "create_note"} and not has_space
        tool_focus: list[ToolFocus] = []
        if write_action in {"create_task", "update_task", "delete_task"}:
            tool_focus = ["tasks"]
        elif write_action in {"create_note", "update_note", "delete_note"}:
            tool_focus = ["notes"]
        return ChatQueryPlan(
            understoodRequest=question,
            responseMode="ask_clarifying_question" if needs_space else "answer",
            requiresSpace=needs_space,
            useStructuredTools=True,
            useVectorSearch=False,
            directToolAnswerAllowed=True,
            optionKind="spaces" if needs_space else "none",
            toolFocus=tool_focus,
            temporalScope=_fallback_temporal_scope(lowered),
            dateKey=_date_key_for_scope(_fallback_temporal_scope(lowered), today),
            writeAction=write_action,
            searchQueries=[question],
        )

    focus: list[ToolFocus] = []
    temporal = _fallback_temporal_scope(lowered)

    if _looks_like_day_query(lowered):
        focus.extend(["day_summary", "tasks", "notes", "calendar", "meetings", "summaries"])
        return ChatQueryPlan(
            understoodRequest=question,
            responseMode="answer",
            requiresSpace=False,
            useStructuredTools=True,
            useVectorSearch=True,
            directToolAnswerAllowed=False,
            toolFocus=list(dict.fromkeys(focus)),
            temporalScope=temporal if temporal != "unspecified" else "today",
            dateKey=_date_key_for_scope(temporal if temporal != "unspecified" else "today", today),
            searchQueries=[question, f"activity on {_date_key_for_scope(temporal if temporal != 'unspecified' else 'today', today)}"],
        )

    if "task" in lowered or "plan" in lowered:
        focus.append("tasks")
    if "note" in lowered:
        focus.append("notes")
    if "summary" in lowered or "summarize" in lowered or "summarise" in lowered:
        focus.extend(["summaries", "space_memory"])
    if "decision" in lowered:
        focus.append("decisions")
    if "issue" in lowered or "risk" in lowered or "blocker" in lowered:
        focus.append("issues")
    if "meeting" in lowered or "transcript" in lowered:
        focus.extend(["meetings", "summaries"])
    if "calendar" in lowered or "event" in lowered or "reminder" in lowered:
        focus.append("calendar")
    if re.search(r"\b(space|workspace|project)\b", lowered) and any(
        token in lowered for token in ("about", "detail", "overview", "what", "tell", "inside", "in this")
    ):
        focus.extend(["space_memory", "summaries", "tasks", "notes", "stats"])

    if "space" in lowered or "workspace" in lowered or "project" in lowered:
        if "list" in lowered or "show" in lowered or "available" in lowered:
            return ChatQueryPlan(
                understoodRequest=question,
                responseMode="list_options",
                requiresSpace=False,
                useStructuredTools=True,
                useVectorSearch=False,
                directToolAnswerAllowed=True,
                optionKind="spaces",
                searchQueries=[question],
            )

    if focus:
        needs_space = not has_space and "day_summary" not in focus
        return ChatQueryPlan(
            understoodRequest=question,
            responseMode="answer" if has_space or not needs_space else "ask_clarifying_question",
            requiresSpace=needs_space,
            useStructuredTools=True,
            useVectorSearch=_needs_vector(lowered, focus),
            directToolAnswerAllowed=not _needs_vector(lowered, focus),
            optionKind="none" if has_space or not needs_space else "spaces",
            toolFocus=list(dict.fromkeys(focus)),
            temporalScope=temporal,
            dateKey=_date_key_for_scope(temporal, today),
            searchQueries=[question],
        )

    return ChatQueryPlan(
        understoodRequest=question,
        responseMode="answer",
        requiresSpace=False,
        useStructuredTools=False,
        useVectorSearch=False,
        directToolAnswerAllowed=False,
        searchQueries=[question],
    )


def _fallback_write_action(_lowered: str) -> WriteAction:
    """
    Used only when the planner LLM is unavailable.
    Do not invent writes from keywords — leave writeAction as none until the LLM can decide.
    """
    return "none"


def _looks_like_day_query(text: str) -> bool:
    lowered = text.lower()
    patterns = (
        r"summar(y|ise|ize).{0,24}\bday\b",
        r"\bday\b.{0,24}summar",
        r"today.?s?\s+(meeting|meetings|report|brief|briefing|recap|overview)",
        r"(meeting|meetings)\s+report",
        r"what did i (do|work|cover|discuss)",
        r"how was my day",
        r"daily (brief|briefing|recap|summary)",
        r"end of (the )?day",
        r"my day",
    )
    return any(re.search(pattern, lowered) for pattern in patterns)


def _needs_vector(lowered: str, focus: list[ToolFocus]) -> bool:
    if any(item in focus for item in ("day_summary", "meetings", "space_memory", "decisions", "issues")):
        return True
    return any(token in lowered for token in ("why", "how", "about", "discuss", "said", "transcript", "detail"))


def _fallback_temporal_scope(lowered_question: str) -> TemporalScope:
    if "yesterday" in lowered_question:
        return "yesterday"
    if "today" in lowered_question or "todays" in lowered_question or "today's" in lowered_question:
        return "today"
    if "tomorrow" in lowered_question:
        return "tomorrow"
    if "all" in lowered_question:
        return "all"
    return "unspecified"


def _date_key_for_scope(scope: TemporalScope | str, today: str) -> str | None:
    try:
        base = datetime.fromisoformat(today).date()
    except ValueError:
        base = datetime.now().astimezone().date()
    if scope == "today":
        return base.isoformat()
    if scope == "yesterday":
        return (base - timedelta(days=1)).isoformat()
    if scope == "tomorrow":
        return (base + timedelta(days=1)).isoformat()
    return None
