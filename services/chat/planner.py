from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

from services.llm.models import LLMMessage, StructuredLLMRequest
from services.llm.router import LLMCapability, get_llm_router


ResponseMode = Literal["answer", "ask_clarifying_question", "list_options"]
OptionKind = Literal["none", "spaces"]
ToolFocus = Literal["tasks", "notes", "summaries", "space_memory", "planning", "decisions", "issues", "profile", "stats"]
TemporalScope = Literal["today", "tomorrow", "all", "unspecified"]


class ChatQueryPlan(BaseModel):
    understoodRequest: str
    responseMode: ResponseMode = "answer"
    requiresSpace: bool = False
    useStructuredTools: bool = True
    useVectorSearch: bool = True
    directToolAnswerAllowed: bool = False
    missingInfoQuestion: str | None = None
    optionKind: OptionKind = "none"
    toolFocus: list[ToolFocus] = Field(default_factory=list)
    temporalScope: TemporalScope = "unspecified"
    searchQueries: list[str] = Field(default_factory=list)


async def plan_chat_query(question: str, space_id: str | None) -> ChatQueryPlan:
    provider, model = get_llm_router().route(LLMCapability.NORMALIZATION)
    request = StructuredLLMRequest(
        model=model,
        temperature=0,
        max_tokens=700,
        schema_name="ChatQueryPlan",
        messages=[
            LLMMessage(
                role="system",
                content=(
                    "You are Buddy's query planner. Understand the user's request before any answer is generated. "
                    "Decide which data sources are needed and whether missing information should be requested. "
                    "Do not answer the user's domain question. Return only planning JSON. "
                    "All user-visible text fields, including missingInfoQuestion, must be in English only.\n\n"
                    "Rules:\n"
                    "- First classify intent: casual chat, general knowledge/drafting, workspace query, or list workspace options.\n"
                    "- For casual chat such as 'hi', 'hello', 'thanks', or 'how are you', set useStructuredTools=false and useVectorSearch=false.\n"
                    "- For general knowledge, education, writing, or drafting requests, set useStructuredTools=false and useVectorSearch=false unless the user explicitly asks about their saved workspace data.\n"
                    "- If the user asks to list, show, choose, or see available spaces/workspaces/projects, set responseMode=list_options, optionKind=spaces, requiresSpace=false.\n"
                    "- Treat the word 'space' as a Buddy workspace only when the user is asking about available spaces/workspaces/projects or their saved tasks, notes, summaries, planning, day review, or workspace memory.\n"
                    "- Do not require a Buddy workspace for general education, drafting, or note-generation requests like 'give me notes about the space company'; set useStructuredTools=false for those.\n"
                    "- If the user asks about their saved tasks, saved notes, summaries, planning, day review, or workspace memory and no space is selected, set responseMode=ask_clarifying_question and optionKind=spaces.\n"
                    "- For simple saved-data reads like all notes, recent notes, today's tasks, unfinished tasks, workspace summary, or available spaces, set useStructuredTools=true, useVectorSearch=false, and directToolAnswerAllowed=true.\n"
                    "- If a complete answer can be built from structured workspace data without semantic transcript reasoning, set directToolAnswerAllowed=true.\n"
                    "- If the request mentions a topic, project, person, decision, issue, or asks why/how/details, keep useVectorSearch=true so transcript memory can be used.\n"
                    "- Set toolFocus to the structured areas needed: tasks, notes, summaries, space_memory, planning, decisions, issues, profile, stats.\n"
                    "- Set temporalScope when the user asks about today, tomorrow, all time, or no clear time.\n"
                    "- Generate 1-5 concise semantic search queries preserving the user's meaning."
                ),
            ),
            LLMMessage(
                role="user",
                content=json.dumps(
                    {
                        "question": question,
                        "spaceIdProvided": bool(space_id),
                    },
                    ensure_ascii=True,
                ),
            ),
        ],
    )
    try:
        return await provider.generate_structured(request, ChatQueryPlan)
    except Exception:
        return _fallback_plan(question, space_id)


def _fallback_plan(question: str, space_id: str | None) -> ChatQueryPlan:
    lowered = question.lower()
    focus: list[ToolFocus] = []
    if "task" in lowered or "plan" in lowered:
        focus.append("tasks")
    if "note" in lowered:
        focus.append("notes")
    if "summary" in lowered or "summarize" in lowered:
        focus.extend(["summaries", "space_memory"])
    if "decision" in lowered:
        focus.append("decisions")
    if "issue" in lowered or "risk" in lowered or "blocker" in lowered:
        focus.append("issues")
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
        return ChatQueryPlan(
            understoodRequest=question,
            responseMode="answer" if space_id else "ask_clarifying_question",
            requiresSpace=not bool(space_id),
            useStructuredTools=True,
            useVectorSearch=False,
            directToolAnswerAllowed=True,
            optionKind="none" if space_id else "spaces",
            toolFocus=list(dict.fromkeys(focus)),
            temporalScope=_fallback_temporal_scope(lowered),
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


def _fallback_temporal_scope(lowered_question: str) -> TemporalScope:
    if "today" in lowered_question:
        return "today"
    if "tomorrow" in lowered_question:
        return "tomorrow"
    if "all" in lowered_question:
        return "all"
    return "unspecified"
