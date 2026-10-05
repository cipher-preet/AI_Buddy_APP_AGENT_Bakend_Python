from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

try:
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
except ImportError:  # Dependencies are declared; this keeps tests importable before install.
    AIMessage = BaseMessage = HumanMessage = None

from apps.api_gateway.config.setting import settings
from services.chat.actions import execute_write_action
from services.chat.meeting.context import MeetingContext, MeetingContextLoader, format_timestamp
from services.chat.meeting.index import MeetingTranscriptIndex
from services.chat.meeting.prompts import MEETING_ITEMS_PROMPT, MEETING_PLAN_PROMPT, MEETING_SYSTEM_PROMPT
from services.chat.meeting.retriever import MeetingRetrieval, MeetingRetriever
from services.chat.models import MAX_CHAT_MESSAGES
from services.chat.node_client import NodeApiError
from services.chat.planner import ChatQueryPlan, plan_chat_query
from services.chat.repository import ChatRepository, to_mongo_id
from services.chat.text_utils import match_option, strip_tool_call_markup
from services.chat.writes import ChatWriteStore
from services.daily_briefing.timezones import DEFAULT_TIMEZONE, date_key_for
from services.llm.models import LLMMessage, LLMRequest, StructuredLLMRequest
from services.llm.router import LLMCapability, get_llm_router
from services.observability.diagnostics import diag_log

HISTORY_TURNS_FOR_PROMPT = 12
MAX_DRAFT_ITEMS = 10
PENDING_SPACE_ACTION = "meeting_select_space"


class MeetingTurnPlan(BaseModel):
    intent: Literal["answer", "create_items", "workspace_write", "chitchat"] = "answer"
    standaloneQuestion: str = ""
    searchQueries: list[str] = Field(default_factory=list)
    spaceNameHint: str | None = None


class MeetingDraftItem(BaseModel):
    kind: Literal["task", "note"] = "task"
    title: str
    description: str = ""
    dueDate: str | None = None
    priority: Literal["High", "Medium", "Low"] = "Medium"


class MeetingDraftItems(BaseModel):
    items: list[MeetingDraftItem] = Field(default_factory=list)
    clarification: str | None = None


class MeetingChatService:
    """Meeting-scoped RAG chat: answers from one meeting and creates linked tasks/notes."""

    def __init__(
        self,
        repository: ChatRepository | None = None,
        loader: MeetingContextLoader | None = None,
        index: MeetingTranscriptIndex | None = None,
        retriever: MeetingRetriever | None = None,
    ):
        self.repository = repository or ChatRepository()
        self.loader = loader or MeetingContextLoader(self.repository.db)
        self.index = index or MeetingTranscriptIndex(self.repository.db)
        self.retriever = retriever or MeetingRetriever(self.index)

    async def ask(
        self,
        user_id: str,
        meeting_id: str,
        question: str,
        space_ids: list[str] | None = None,
        chat_id: str | None = None,
        auth_token: str | None = None,
    ) -> dict[str, Any]:
        user_id = user_id.strip()
        meeting_id = meeting_id.strip()
        question = question.strip()
        if not user_id:
            raise ValueError("userId is required")
        if not meeting_id:
            raise ValueError("meetingId is required")
        if not question:
            raise ValueError("question is required")
        if HumanMessage is None or AIMessage is None:
            raise RuntimeError("langchain-core is required for chat message history")

        context = await self.loader.load(user_id, meeting_id)
        requested_space_ids = [sid for sid in (space_ids or []) if sid]
        session = await self.repository.get_or_create_meeting_session(
            user_id,
            meeting_id,
            requested_space_ids[0] if requested_space_ids else context.space_id,
            chat_id,
        )
        created_new_chat = session.messageCount == 0 and session.title is None
        if not session.title:
            await self.repository.touch_title(session.id, question)

        await self.repository.ensure_chat_history_indexes()
        history_store = self.repository.get_message_history(session.id, MAX_CHAT_MESSAGES)
        history = await history_store.aget_messages()

        turn = await self._run_turn(
            context=context,
            question=question,
            history=history,
            requested_space_ids=requested_space_ids,
            pending_action=session.pendingAction,
            auth_token=auth_token,
        )

        await history_store.aadd_messages([HumanMessage(content=question), AIMessage(content=turn["answer"])])
        if turn.get("pending_action"):
            await self.repository.set_pending_action(session.id, turn["pending_action"])
        elif session.pendingAction:
            await self.repository.clear_pending_action(session.id)
        if turn.get("space_id") and str(session.spaceId or "") != turn["space_id"]:
            await self.repository.set_session_space(session.id, turn["space_id"])
        await self.repository.sync_message_count(session.id)

        return {
            "chatId": str(session.id),
            "createdNewChat": created_new_chat or (chat_id is not None and str(session.id) != chat_id),
            "answer": turn["answer"],
            "meetingId": meeting_id,
            "createdItems": turn.get("created_items") or [],
        }

    async def _run_turn(
        self,
        *,
        context: MeetingContext,
        question: str,
        history: list[BaseMessage],
        requested_space_ids: list[str],
        pending_action: dict[str, Any] | None,
        auth_token: str | None,
    ) -> dict[str, Any]:
        pending_type = (pending_action or {}).get("type")

        if pending_type == PENDING_SPACE_ACTION:
            selected = _resolve_option(question, pending_action.get("options") or [])
            if selected:
                drafts = [MeetingDraftItem.model_validate(item) for item in pending_action.get("items") or []]
                return await self._create_items(
                    context,
                    drafts,
                    space_id=str(selected["value"]),
                    space_label=str(selected.get("label") or selected["value"]),
                    auth_token=auth_token,
                )

        if pending_type in {"complete_write", "select_option"}:
            delegated = await self._continue_workspace_write(context, question, pending_action, requested_space_ids, auth_token)
            if delegated:
                return delegated

        corpus_task = asyncio.create_task(self.index.ensure_corpus(context))
        plan = await self._plan(context, question, history)
        effective_question = plan.standaloneQuestion.strip() or question

        if plan.intent == "workspace_write":
            corpus_task.add_done_callback(_swallow_task_error)
            delegated = await self._workspace_write(context, effective_question, requested_space_ids, auth_token)
            if delegated:
                return delegated

        corpus = await corpus_task
        queries = _dedupe([effective_question, *plan.searchQueries])[:5]
        retrieval = await self.retriever.retrieve(context.meeting_id, context.user_id, corpus, queries)

        if plan.intent == "create_items":
            drafts = await self._draft_items(context, effective_question, retrieval)
            if not drafts.items:
                return {
                    "answer": drafts.clarification
                    or "I couldn't find anything in this meeting that matches what you want me to create. "
                    "Tell me the exact task or note and I'll add it.",
                }
            store = ChatWriteStore(database=self.repository.db, auth_token=auth_token)
            preferred = [*requested_space_ids, *([context.space_id] if context.space_id else [])]
            space_id, space_label, spaces = await store.resolve_space_id(
                context.user_id,
                preferred_space_ids=_dedupe(preferred),
                space_name_hint=plan.spaceNameHint,
            )
            if not space_id:
                return _space_selection_turn(drafts.items, spaces)
            return await self._create_items(context, drafts.items, space_id, space_label or space_id, auth_token)

        answer = await self._answer(context, question, effective_question, history, retrieval)
        return {"answer": answer}

    async def _plan(self, context: MeetingContext, question: str, history: list[BaseMessage]) -> MeetingTurnPlan:
        fallback = MeetingTurnPlan(intent="answer", standaloneQuestion=question, searchQueries=[question])
        try:
            provider, model = get_llm_router().route(LLMCapability.NORMALIZATION)
            plan = await provider.generate_structured(
                StructuredLLMRequest(
                    model=model,
                    temperature=0,
                    max_tokens=500,
                    schema_name="MeetingTurnPlan",
                    messages=[
                        LLMMessage(role="system", content=MEETING_PLAN_PROMPT),
                        LLMMessage(
                            role="user",
                            content=json.dumps(
                                {
                                    "meetingTitle": context.title,
                                    "recentHistory": _history_payload(history, 6),
                                    "userMessage": question,
                                },
                                ensure_ascii=False,
                            ),
                        ),
                    ],
                ),
                MeetingTurnPlan,
            )
            plan.searchQueries = [query for query in plan.searchQueries if query.strip()][:4]
            if not plan.searchQueries:
                plan.searchQueries = [plan.standaloneQuestion or question]
            return plan
        except Exception as error:
            diag_log("meeting_chat_plan_failed", meetingId=context.meeting_id, error=str(error)[:300])
            return fallback

    async def _answer(
        self,
        context: MeetingContext,
        question: str,
        effective_question: str,
        history: list[BaseMessage],
        retrieval: MeetingRetrieval,
    ) -> str:
        messages = [
            LLMMessage(role="system", content=MEETING_SYSTEM_PROMPT),
            LLMMessage(role="system", content=_meeting_dossier(context)),
            LLMMessage(role="system", content=_transcript_block(context, retrieval)),
            *_history_messages(history[-HISTORY_TURNS_FOR_PROMPT * 2 :]),
        ]
        if effective_question and effective_question != question:
            messages.append(LLMMessage(role="system", content=f"Resolved meaning of the next user message: {effective_question}"))
        messages.append(LLMMessage(role="user", content=question))

        provider, model = get_llm_router().route(LLMCapability.CHAT_ANSWER)
        for attempt in range(2):
            try:
                response = await provider.generate(
                    LLMRequest(
                        model=model,
                        temperature=settings.LLM_TEMPERATURE,
                        max_tokens=1400,
                        messages=messages
                        if attempt == 0
                        else [
                            *messages,
                            LLMMessage(
                                role="user",
                                content="Your previous answer was empty. Answer now from the meeting evidence, in English.",
                            ),
                        ],
                    )
                )
                answer = _clean_answer(response.content)
                if answer:
                    return answer
            except Exception as error:
                diag_log("meeting_chat_answer_failed", meetingId=context.meeting_id, attempt=attempt, error=str(error)[:300])
        return (
            "I couldn't generate an answer from this meeting right now. "
            "Please try again in a moment, or ask a more specific question."
        )

    async def _draft_items(
        self,
        context: MeetingContext,
        effective_question: str,
        retrieval: MeetingRetrieval,
    ) -> MeetingDraftItems:
        existing = [item.title for item in [*context.tasks, *context.notes]]
        try:
            provider, model = get_llm_router().route(LLMCapability.CHAT_ANSWER)
            drafts = await provider.generate_structured(
                StructuredLLMRequest(
                    model=model,
                    temperature=0,
                    max_tokens=1600,
                    schema_name="MeetingDraftItems",
                    messages=[
                        LLMMessage(role="system", content=MEETING_ITEMS_PROMPT),
                        LLMMessage(role="system", content=_meeting_dossier(context)),
                        LLMMessage(role="system", content=_transcript_block(context, retrieval)),
                        LLMMessage(
                            role="user",
                            content=json.dumps(
                                {
                                    "request": effective_question,
                                    "todayDateKey": date_key_for(datetime.now().astimezone(), DEFAULT_TIMEZONE),
                                    "existingTitles": existing[:60],
                                },
                                ensure_ascii=False,
                            ),
                        ),
                    ],
                ),
                MeetingDraftItems,
            )
        except Exception as error:
            diag_log("meeting_chat_draft_failed", meetingId=context.meeting_id, error=str(error)[:300])
            return MeetingDraftItems(
                clarification="I couldn't prepare those items just now. Please try again, or tell me the exact title."
            )

        existing_keys = {_title_key(title) for title in existing}
        unique: list[MeetingDraftItem] = []
        for item in drafts.items:
            item.title = item.title.strip()[:80]
            item.description = (item.description or "").strip() or f"From meeting: {context.title}"
            if item.dueDate and not _is_date_key(item.dueDate):
                item.dueDate = None
            key = _title_key(item.title)
            if not item.title or key in existing_keys:
                continue
            existing_keys.add(key)
            unique.append(item)
        drafts.items = unique[:MAX_DRAFT_ITEMS]
        return drafts

    async def _create_items(
        self,
        context: MeetingContext,
        drafts: list[MeetingDraftItem],
        space_id: str,
        space_label: str,
        auth_token: str | None,
    ) -> dict[str, Any]:
        store = ChatWriteStore(database=self.repository.db, auth_token=auth_token)
        created: list[dict[str, Any]] = []
        failures: list[str] = []
        for draft in drafts:
            try:
                if draft.kind == "task":
                    result = await store.create_task(
                        user_id=context.user_id,
                        space_id=space_id,
                        title=draft.title,
                        description=draft.description,
                        due_date=draft.dueDate,
                        priority=draft.priority,
                    )
                else:
                    result = await store.create_note(
                        user_id=context.user_id,
                        space_id=space_id,
                        title=draft.title,
                        body=draft.description,
                    )
            except NodeApiError as error:
                failures.append(f"{draft.title} ({error})")
                if "sign in" in str(error).lower():
                    break
                continue
            except Exception as error:
                failures.append(f"{draft.title} ({error})")
                continue
            item_id = str(result.get("id") or "")
            if item_id:
                await self._link_to_meeting(draft.kind, item_id, context.meeting_id)
            created.append(
                {
                    "kind": draft.kind,
                    "id": item_id or None,
                    "title": draft.title,
                    "dueDate": draft.dueDate,
                    "priority": draft.priority if draft.kind == "task" else None,
                    "spaceId": space_id,
                    "spaceName": space_label,
                }
            )
        return {
            "answer": _creation_summary(created, failures, space_label),
            "created_items": created,
            "space_id": space_id if created else None,
        }

    async def _link_to_meeting(self, kind: str, item_id: str, meeting_id: str) -> None:
        collections = ("stagedTasks", "tasks") if kind == "task" else ("stagedNotes", "notes")
        update = {
            "$set": {
                "sourceConversationId": to_mongo_id(meeting_id),
                "meetingSessionId": meeting_id,
                "createdVia": "meeting_chat",
            }
        }
        for name in collections:
            try:
                result = await self.repository.db[name].update_one({"_id": to_mongo_id(item_id)}, update)
                if result.matched_count:
                    return
            except Exception as error:
                diag_log("meeting_chat_link_failed", meetingId=meeting_id, itemId=item_id, error=str(error)[:300])

    async def _workspace_write(
        self,
        context: MeetingContext,
        question: str,
        requested_space_ids: list[str],
        auth_token: str | None,
    ) -> dict[str, Any] | None:
        space_ids = _dedupe([*requested_space_ids, *([context.space_id] if context.space_id else [])])
        plan = await plan_chat_query(question, space_ids[0] if space_ids else None, space_ids=space_ids)
        if getattr(plan, "writeAction", "none") == "none":
            return None
        result = await execute_write_action(
            user_id=context.user_id,
            question=question,
            plan=plan,
            space_ids=space_ids,
            auth_token=auth_token,
        )
        if not result:
            return None
        return {"answer": str(result.get("answer") or "").strip() or "Done.", "pending_action": result.get("pending_action")}

    async def _continue_workspace_write(
        self,
        context: MeetingContext,
        question: str,
        pending_action: dict[str, Any],
        requested_space_ids: list[str],
        auth_token: str | None,
    ) -> dict[str, Any] | None:
        pending_type = pending_action.get("type")
        if pending_type == "complete_write":
            plan = ChatQueryPlan(understoodRequest=str(pending_action.get("originalQuestion") or question))
            original = str(pending_action.get("originalQuestion") or "").strip()
            combined = f"{original} {question}".strip() if original and question.lower() not in original.lower() else question
            space_ids = _dedupe([*requested_space_ids, *([context.space_id] if context.space_id else [])])
            result = await execute_write_action(
                user_id=context.user_id,
                question=combined,
                plan=plan,
                space_ids=space_ids,
                pending_write=pending_action,
                auth_token=auth_token,
            )
        else:
            selected = _resolve_option(question, pending_action.get("options") or [])
            if not selected or pending_action.get("optionKind") != "spaces" or not pending_action.get("plan"):
                return None
            try:
                plan = ChatQueryPlan.model_validate(pending_action["plan"])
            except Exception:
                return None
            result = await execute_write_action(
                user_id=context.user_id,
                question=str(pending_action.get("originalQuestion") or question),
                plan=plan,
                space_ids=[str(selected["value"])],
                auth_token=auth_token,
            )
        if not result:
            return None
        return {"answer": str(result.get("answer") or "").strip() or "Done.", "pending_action": result.get("pending_action")}


def _space_selection_turn(drafts: list[MeetingDraftItem], spaces: list[dict[str, Any]]) -> dict[str, Any]:
    if not spaces:
        return {
            "answer": (
                "You don't have a space yet, and tasks and notes need one. "
                'Create a space first (for example: "create a space called Work"), then ask me again.'
            )
        }
    options = [
        {"index": index, "label": str(space.get("label") or space.get("spaceId") or ""), "value": str(space.get("spaceId") or "")}
        for index, space in enumerate(spaces, start=1)
    ]
    preview = "\n".join(f"- {'Task' if item.kind == 'task' else 'Note'}: **{item.title}**" for item in drafts)
    choices = "\n".join(f"{option['index']}. {option['label']}" for option in options)
    return {
        "answer": (
            f"I've prepared {len(drafts)} item{'s' if len(drafts) != 1 else ''} from this meeting:\n{preview}\n\n"
            f"Which space should I save {'them' if len(drafts) != 1 else 'it'} in?\n{choices}\n"
            "Reply with the space name or number."
        ),
        "pending_action": {
            "type": PENDING_SPACE_ACTION,
            "items": [item.model_dump() for item in drafts],
            "options": options,
        },
    }


def _creation_summary(created: list[dict[str, Any]], failures: list[str], space_label: str) -> str:
    if not created and failures:
        return "I couldn't save that just now: " + "; ".join(failures[:3])
    tasks = [item for item in created if item["kind"] == "task"]
    notes = [item for item in created if item["kind"] == "note"]
    parts = []
    if tasks:
        parts.append(f"{len(tasks)} task{'s' if len(tasks) != 1 else ''}")
    if notes:
        parts.append(f"{len(notes)} note{'s' if len(notes) != 1 else ''}")
    lines = [f"Added {' and '.join(parts)} to **{space_label}**, linked to this meeting:"]
    for item in created:
        details = []
        if item.get("dueDate"):
            details.append(f"due {item['dueDate']}")
        if item["kind"] == "task" and item.get("priority") and item["priority"] != "Medium":
            details.append(f"{item['priority']} priority")
        suffix = f" ({', '.join(details)})" if details else ""
        lines.append(f"- {'Task' if item['kind'] == 'task' else 'Note'}: **{item['title']}**{suffix}")
    if failures:
        lines.append("")
        lines.append("Couldn't save: " + "; ".join(failures[:3]))
    return "\n".join(lines)


def _meeting_dossier(context: MeetingContext) -> str:
    lines = [f"MEETING: {context.title}"]
    if context.started_at:
        lines.append(f"Date: {context.started_at.strftime('%A, %d %B %Y, %H:%M UTC')}")
    if context.duration_ms:
        lines.append(f"Duration: {format_timestamp(context.duration_ms)}")
    if context.status:
        lines.append(f"Status: {context.status}")
    if context.speakers:
        lines.append(f"Speakers in transcript: {', '.join(context.speakers[:12])}")

    summary = context.summary or {}
    if summary.get("summary"):
        lines.append(f"\nSUMMARY:\n{summary['summary']}")
    for key, label in (
        ("topics", "Topics"),
        ("decisions", "Decisions"),
        ("importantFacts", "Important facts"),
        ("openQuestions", "Open questions"),
        ("blockers", "Blockers"),
    ):
        values = [_item_text(value) for value in summary.get(key) or []]
        values = [value for value in values if value]
        if values:
            lines.append(f"{label}: " + "; ".join(values[:15]))

    memory = context.memory or {}
    if memory.get("shortSummary") and memory.get("shortSummary") != summary.get("summary"):
        lines.append(f"\nMEETING MEMORY: {memory['shortSummary']}")
    for key, label in (
        ("decisions", "Decisions"),
        ("commitments", "Commitments"),
        ("requirements", "Requirements"),
        ("deadlines", "Deadlines"),
        ("openQuestions", "Open questions"),
        ("blockers", "Blockers"),
        ("importantFacts", "Facts"),
    ):
        values = [_item_text(value) for value in memory.get(key) or []]
        values = [value for value in values if value]
        if values:
            lines.append(f"Memory {label}: " + "; ".join(values[:15]))

    if context.tasks:
        lines.append("\nTASKS LINKED TO THIS MEETING:")
        for task in context.tasks[:40]:
            meta = [task.status or "open"]
            if task.owner:
                meta.append(f"owner {task.owner}")
            if task.due:
                meta.append(f"due {task.due}")
            body = f" - {task.body[:240]}" if task.body else ""
            lines.append(f"- {task.title} [{', '.join(meta)}]{body}")
    else:
        lines.append("\nTASKS LINKED TO THIS MEETING: none")

    if context.notes:
        lines.append("\nNOTES LINKED TO THIS MEETING:")
        for note in context.notes[:30]:
            body = f" - {note.body[:300]}" if note.body else ""
            lines.append(f"- {note.title}{body}")
    else:
        lines.append("\nNOTES LINKED TO THIS MEETING: none")
    return "\n".join(lines)


def _transcript_block(context: MeetingContext, retrieval: MeetingRetrieval) -> str:
    if retrieval.mode == "empty" or not retrieval.chunks:
        status = (context.status or "").lower()
        if status in {"recording", "active", "processing", "uploading"}:
            return "TRANSCRIPT: not available yet (the meeting is still being recorded or processed)."
        return "TRANSCRIPT: no transcript is available for this meeting."
    header = (
        "FULL TRANSCRIPT (chronological):"
        if retrieval.mode == "full"
        else "MOST RELEVANT TRANSCRIPT EXCERPTS (chronological; other parts of the meeting are omitted):"
    )
    blocks = [header]
    previous_index: int | None = None
    for chunk in retrieval.chunks:
        if retrieval.mode != "full" and previous_index is not None and chunk.index != previous_index + 1:
            blocks.append("...")
        blocks.append(chunk.text)
        previous_index = chunk.index
    return "\n".join(blocks)


def _history_messages(history: list[BaseMessage]) -> list[LLMMessage]:
    messages: list[LLMMessage] = []
    for message in history:
        if message.type == "human":
            messages.append(LLMMessage(role="user", content=str(message.content)))
        elif message.type == "ai":
            messages.append(LLMMessage(role="assistant", content=str(message.content)))
    return messages


def _history_payload(history: list[BaseMessage], turns: int) -> list[dict[str, str]]:
    payload = []
    for message in history[-turns * 2 :]:
        role = "user" if getattr(message, "type", None) == "human" else "assistant"
        content = str(getattr(message, "content", "") or "").strip()
        if content:
            payload.append({"role": role, "content": content[:800]})
    return payload


def _item_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("text", "title", "content", "summary", "description", "name", "value"):
            if value.get(key):
                text = str(value[key]).strip()
                owner = value.get("owner") or value.get("ownerText")
                due = value.get("dueDate") or value.get("dueDateText") or value.get("deadline")
                extras = [str(item) for item in (owner, due) if item]
                return f"{text} ({', '.join(extras)})" if extras else text
    return str(value or "").strip()


def _resolve_option(reply: str, options: list[dict[str, Any]]) -> dict[str, Any] | None:
    normalized = reply.strip().lower().rstrip(".")
    if not normalized:
        return None
    digits = re.fullmatch(r"(?:space|option|number|no\.?)?\s*(\d+)", normalized)
    if digits:
        wanted = int(digits.group(1))
        for option in options:
            if int(option.get("index") or 0) == wanted:
                return option
    for option in options:
        label = str(option.get("label") or "").strip().lower()
        if label and (normalized == label or normalized == str(option.get("value") or "").lower()):
            return option
    matches = [option for option in options if str(option.get("label") or "").strip().lower() in normalized]
    if len(matches) == 1:
        return matches[0]
    return match_option(reply, options)


_SOURCE_MARKERS = {"source", "sources", "references", "context", "retrieved context", "evidence"}


def _clean_answer(answer: str) -> str:
    cleaned = []
    for line in strip_tool_call_markup(answer or "").splitlines():
        if line.strip().strip("*#:- ").lower() in _SOURCE_MARKERS:
            break
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def _title_key(title: str) -> str:
    return " ".join(re.findall(r"\w+", (title or "").lower()))


def _is_date_key(value: str) -> bool:
    try:
        datetime.fromisoformat(value)
        return len(value) == 10
    except ValueError:
        return False


def _dedupe(values: list[str | None]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = " ".join(str(value or "").split())
        if normalized and normalized.lower() not in seen:
            seen.add(normalized.lower())
            result.append(normalized)
    return result


def _swallow_task_error(task: asyncio.Task) -> None:
    if not task.cancelled():
        task.exception()
