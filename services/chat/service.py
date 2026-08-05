from __future__ import annotations

import json
import re
from typing import Any, TypedDict

from pydantic import BaseModel, Field

try:
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
    from langgraph.graph import END, StateGraph
except ImportError:  # Dependencies are declared; this keeps tests importable before install.
    AIMessage = BaseMessage = HumanMessage = None
    END = None
    StateGraph = None

from apps.api_gateway.config.setting import settings
from services.chat.models import MAX_CHAT_MESSAGES
from services.chat.planner import ChatQueryPlan, plan_chat_query
from services.chat.repository import ChatRepository, encode_sessions_cursor
from services.chat.retrieval import ChatRetriever, format_context
from services.chat.tools import ChatToolRunner
from services.llm.models import LLMMessage, LLMRequest, StructuredLLMRequest
from services.llm.router import LLMCapability, get_llm_router


SYSTEM_PROMPT = """You are Buddy's chat assistant.
Answer in English only, even when the user asks in Hindi, Hinglish, or any other language.
Use the retrieved context as evidence. The context may be in Hindi, English, or mixed language; translate and reason over it internally.
Use structured tool context as the authoritative source for tasks, notes, today/due-date questions, unfinished work, recent summaries, and space memory.
For topic-specific questions, combine structured tool context with retrieved transcript context. If they differ, explain only what is supported and prefer saved structured task/note fields for task status and due dates.
For general knowledge, drafting, education, or note-generation requests that do not ask about the user's saved workspace data, answer from your general knowledge.
If the user asks about their saved workspace data and the answer is not supported by the retrieved context or chat history, say what is missing and ask a concise follow-up.
Be direct, useful, and avoid inventing facts.
Never return an empty response.
Never include raw source IDs, chunk numbers, citations, source lists, or verbatim retrieved context in the final answer.
Do not add a Source, Sources, Evidence, Context, or References section."""


class ChatState(TypedDict):
    user_id: str
    space_id: str | None
    question: str
    plan: ChatQueryPlan | None
    search_queries: list[str]
    history: list[BaseMessage]
    context_text: str
    tool_context: str
    tool_answer: str | None
    tool_direct: bool
    pending_action: dict[str, Any] | None
    answer: str
    provider: str | None
    model: str | None
    usage: dict[str, int]


class SpaceSelectionHistoryResolution(BaseModel):
    shouldResume: bool = False
    originalQuestion: str | None = None
    confidence: float = Field(default=0, ge=0, le=1)


class ChatService:
    def __init__(
        self,
        repository: ChatRepository | None = None,
        retriever: ChatRetriever | None = None,
        tool_runner: ChatToolRunner | None = None,
    ):
        self.repository = repository or ChatRepository()
        self.retriever = retriever or ChatRetriever()
        self.tool_runner = tool_runner or ChatToolRunner()

    async def create_chat_session(self, user_id: str, space_id: str | None = None) -> dict[str, Any]:
        user_id = user_id.strip()
        space_id = space_id.strip() if space_id else None
        if not user_id:
            raise ValueError("userId is required")
        session = await self.repository.create_session(user_id, space_id)
        return _session_response(session)

    async def ask(self, user_id: str, question: str, space_id: str | None = None, chat_id: str | None = None) -> dict[str, Any]:
        user_id = user_id.strip()
        space_id = space_id.strip() if space_id else None
        question = question.strip()
        if not user_id:
            raise ValueError("userId is required")
        if not question:
            raise ValueError("question is required")

        session = await self.repository.get_active_or_create_session(user_id, space_id, chat_id)
        created_new_chat = session.messageCount == 0 and session.title is None
        if not session.title:
            await self.repository.touch_title(session.id, question)

        await self.repository.ensure_chat_history_indexes()
        history_store = self.repository.get_message_history(session.id, MAX_CHAT_MESSAGES)
        history = await history_store.aget_messages()
        session_space_id = str(session.spaceId) if session.spaceId is not None else None
        effective_space_id = space_id
        effective_question = question
        if not effective_space_id:
            effective_space_id, effective_question = self._resolve_pending_action(
                question,
                None,
                session.pendingAction,
            )
        if not effective_space_id:
            effective_space_id, effective_question = await self._resolve_space_from_message_or_history(
                question,
                user_id,
                history,
            )
        if not effective_space_id:
            effective_space_id = session_space_id
        if effective_space_id:
            await self.repository.clear_pending_action(session.id)
            if not session_space_id or session_space_id != effective_space_id:
                await self.repository.set_session_space(session.id, effective_space_id)
        result = await self._invoke_graph(
            {
                "user_id": user_id,
                "space_id": effective_space_id,
                "question": effective_question,
                "history": history,
            }
        )
        if HumanMessage is None or AIMessage is None:
            raise RuntimeError("langchain-core is required for chat message history")
        await history_store.aadd_messages(
            [
                HumanMessage(content=question),
                AIMessage(content=result["answer"]),
            ]
        )
        pending_action = _merge_pending_space_action(session.pendingAction, result.get("pendingAction"))
        if pending_action:
            await self.repository.set_pending_action(session.id, pending_action)
        elif effective_space_id or session.pendingAction:
            await self.repository.clear_pending_action(session.id)
        await self.repository.sync_message_count(session.id)
        return {
            "chatId": str(session.id),
            "createdNewChat": created_new_chat or (chat_id is not None and str(session.id) != chat_id),
            "answer": result["answer"],
        }

    def _resolve_pending_action(
        self,
        question: str,
        space_id: str | None,
        pending_action: dict[str, Any] | None,
    ) -> tuple[str | None, str]:
        if space_id:
            return space_id, question
        if not pending_action:
            return None, question
        if pending_action.get("type") != "select_option":
            return None, question
        selected = _resolve_pending_option(question, pending_action.get("options") or [])
        if not selected:
            return None, question
        if pending_action.get("optionKind") == "spaces":
            selected_value = str(selected.get("value") or "")
            if selected_value:
                return selected_value, str(pending_action.get("originalQuestion") or question)
        return None, question

    async def _resolve_space_from_message_or_history(
        self,
        question: str,
        user_id: str,
        history: list[BaseMessage],
    ) -> tuple[str | None, str]:
        spaces = await self.tool_runner.repository.list_user_spaces(user_id)
        if not spaces:
            return None, question

        inline_selection = _extract_inline_space_selection(question)
        if inline_selection:
            selected = _resolve_pending_option(inline_selection, _space_options(spaces))
            if selected:
                cleaned_question = _remove_inline_space_selection(question).strip()
                return str(selected.get("value") or ""), cleaned_question or question

        if _looks_like_option_reply(question):
            selected = _resolve_pending_option(question, _space_options(spaces))
            previous_request = await _infer_original_question_from_history(question, history)
            if selected and previous_request:
                return str(selected.get("value") or ""), previous_request

        return None, question

    async def load_chat(self, user_id: str, chat_id: str) -> dict[str, Any]:
        session = await self.repository.get_session(chat_id)
        if not session:
            raise ValueError("Chat session not found")
        if str(session.userId) != user_id:
            raise PermissionError("Chat session does not belong to this user")
        messages = await self.repository.list_recent_messages(chat_id, MAX_CHAT_MESSAGES)
        return {
            "chat": _session_response(session),
            "messages": [_message_response(message) for message in messages],
        }

    async def list_chats(
        self,
        user_id: str,
        space_id: str | None = None,
        limit: int = 20,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        page_size = max(1, min(limit, 100))
        sessions = await self.repository.list_sessions(user_id, space_id, page_size + 1, cursor)
        visible_sessions = sessions[:page_size]
        has_more = len(sessions) > page_size
        return {
            "chats": [_session_response(session) for session in visible_sessions],
            "nextCursor": encode_sessions_cursor(visible_sessions[-1]) if has_more and visible_sessions else None,
            "hasMore": has_more,
            "limit": page_size,
        }

    async def _invoke_graph(self, inputs: dict[str, Any]) -> dict[str, Any]:
        graph = self._build_graph()
        history = inputs.get("history") or []
        state: ChatState = {
            "user_id": inputs["user_id"],
            "space_id": inputs.get("space_id"),
            "question": inputs["question"],
            "plan": None,
            "search_queries": [inputs["question"]],
            "history": history,
            "context_text": "",
            "tool_context": "",
            "tool_answer": None,
            "tool_direct": False,
            "pending_action": None,
            "answer": "",
            "provider": None,
            "model": None,
            "usage": {},
        }
        result = await graph.ainvoke(state) if hasattr(graph, "ainvoke") else await graph(state)
        return {
            "answer": result["answer"],
            "provider": result.get("provider"),
            "model": result.get("model"),
            "usage": result.get("usage") or {},
            "pendingAction": result.get("pending_action"),
        }

    def _build_graph(self):
        if StateGraph is None:
            return self._fallback_graph

        workflow = StateGraph(ChatState)
        workflow.add_node("plan_query", self._plan_query_node)
        workflow.add_node("expand_query", self._expand_query_node)
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("run_tools", self._run_tools_node)
        workflow.add_node("generate", self._generate_node)
        workflow.set_entry_point("plan_query")
        workflow.add_edge("plan_query", "expand_query")
        workflow.add_edge("expand_query", "retrieve")
        workflow.add_edge("retrieve", "run_tools")
        workflow.add_edge("run_tools", "generate")
        workflow.add_edge("generate", END)
        return workflow.compile()

    async def _fallback_graph(self, state: ChatState) -> ChatState:
        state = await self._plan_query_node(state)
        state = await self._expand_query_node(state)
        state = await self._retrieve_node(state)
        state = await self._run_tools_node(state)
        return await self._generate_node(state)

    async def _plan_query_node(self, state: ChatState) -> ChatState:
        state["plan"] = await plan_chat_query(state["question"], state["space_id"])
        return state

    async def _expand_query_node(self, state: ChatState) -> ChatState:
        plan = state.get("plan")
        planned_queries = list(plan.searchQueries) if plan else []
        state["search_queries"] = planned_queries or await _build_multilingual_search_queries(state["question"])
        return state

    async def _retrieve_node(self, state: ChatState) -> ChatState:
        plan = state.get("plan")
        if plan and not plan.useVectorSearch:
            state["context_text"] = "Vector retrieval skipped by query plan."
            return state
        contexts = await self.retriever.retrieve_many(
            state["search_queries"],
            state["user_id"],
            state["space_id"],
        )
        state["context_text"] = format_context(contexts)
        return state

    async def _run_tools_node(self, state: ChatState) -> ChatState:
        result = await self.tool_runner.run(
            state["question"],
            state["user_id"],
            state["space_id"],
            state.get("plan"),
        )
        state["tool_context"] = str(result.get("context") or "")
        answer = result.get("answer")
        state["tool_answer"] = str(answer) if answer else None
        state["tool_direct"] = bool(result.get("direct"))
        state["pending_action"] = result.get("pending_action")
        return state

    async def _generate_node(self, state: ChatState) -> ChatState:
        if state.get("tool_direct") and state.get("tool_answer"):
            state["answer"] = str(state["tool_answer"]).strip()
            state["provider"] = "structured-tools"
            state["model"] = "workspace-direct-answer"
            state["usage"] = {}
            return state

        provider, model = get_llm_router().route(LLMCapability.HIGH_ACCURACY_REASONING)
        response = await provider.generate(
            LLMRequest(
                model=model,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=1200,
                messages=_build_llm_messages(state),
            )
        )
        state["answer"] = response.content.strip()
        state["answer"] = _sanitize_public_answer(state["answer"])
        if not state["answer"]:
            retry_response = await provider.generate(
                LLMRequest(
                    model=model,
                    temperature=settings.LLM_TEMPERATURE,
                    max_tokens=1200,
                    messages=[
                        *_build_llm_messages(state),
                        LLMMessage(
                            role="user",
                            content=(
                                "Your previous answer was empty. Return a helpful English answer now. "
                                "If the retrieved context is insufficient, say that clearly and ask one follow-up question. "
                                "Do not include sources, source IDs, chunk numbers, citations, or raw context text."
                            ),
                        ),
                    ],
                )
            )
            state["answer"] = retry_response.content.strip()
            state["answer"] = _sanitize_public_answer(state["answer"])
            response = retry_response
        if not state["answer"]:
            state["answer"] = (
                "I could not find enough relevant context to answer that confidently. "
                "Please ask with a little more detail or tell me which conversation/topic to search."
            )
        state["provider"] = response.provider
        state["model"] = response.model
        state["usage"] = response.usage.model_dump()
        return state


def _build_llm_messages(state: ChatState) -> list[LLMMessage]:
    messages: list[LLMMessage] = [
        LLMMessage(role="system", content=SYSTEM_PROMPT),
        LLMMessage(role="system", content=f"Search queries used: {state['search_queries']}"),
        LLMMessage(role="system", content=f"Structured tool context:\n{state['tool_context']}"),
        LLMMessage(role="system", content=f"Structured draft answer, if any:\n{state.get('tool_answer') or 'none'}"),
        LLMMessage(role="system", content=f"Retrieved user-scoped context:\n{state['context_text']}"),
    ]
    messages.extend(_langchain_memory_to_llm_messages(state["history"][-20:]))
    messages.append(LLMMessage(role="user", content=state["question"]))
    return messages


async def _build_multilingual_search_queries(question: str) -> list[str]:
    queries = [question]
    try:
        provider, model = get_llm_router().route(LLMCapability.NORMALIZATION)
        response = await provider.generate(
            LLMRequest(
                model=model,
                temperature=0,
                max_tokens=300,
                messages=[
                    LLMMessage(
                        role="system",
                        content=(
                            "Create semantic search queries for multilingual RAG. "
                            "Return only plain text lines, no numbering, no markdown."
                        ),
                    ),
                    LLMMessage(
                        role="user",
                        content=(
                            "Rewrite this question into up to 4 search queries: original meaning in English, "
                            "Hindi Devanagari, Hinglish/Roman Hindi, and concise keywords. "
                            f"Question: {question}"
                        ),
                    ),
                ],
            )
        )
        queries.extend(line.strip(" -\t") for line in response.content.splitlines())
    except Exception:
        pass
    return _dedupe_texts(queries)[:5]


def _dedupe_texts(values: list[str]) -> list[str]:
    seen = set()
    unique = []
    for value in values:
        normalized = " ".join((value or "").strip().split())
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        unique.append(normalized)
    return unique


def _langchain_memory_to_llm_messages(history: list[BaseMessage]) -> list[LLMMessage]:
    messages: list[LLMMessage] = []
    for message in history:
        if message.type == "human":
            messages.append(LLMMessage(role="user", content=str(message.content)))
        elif message.type == "ai":
            messages.append(LLMMessage(role="assistant", content=str(message.content)))
    return messages


def _message_response(message: BaseMessage) -> dict[str, Any]:
    role = "user" if message.type == "human" else "assistant" if message.type == "ai" else message.type
    return {
        "role": role,
        "content": str(message.content),
    }


def _resolve_pending_option(selection: str, options: list[dict[str, Any]]) -> dict[str, Any] | None:
    normalized = selection.strip().lower()
    if not normalized:
        return None
    if normalized.isdigit():
        requested_index = int(normalized)
        for option in options:
            if int(option.get("index") or 0) == requested_index:
                return option

    for option in options:
        label = str(option.get("label") or "").strip().lower()
        value = str(option.get("value") or "").strip().lower()
        if normalized in {label, value}:
            return option
    return None


def _space_options(spaces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "label": str(space.get("label") or space.get("spaceId") or ""),
            "value": str(space.get("spaceId") or ""),
        }
        for index, space in enumerate(spaces, start=1)
    ]


def _extract_inline_space_selection(question: str) -> str | None:
    patterns = (
        r"\bspace\s+(\d+)\b",
        r"\bworkspace\s+(\d+)\b",
        r"\bproject\s+(\d+)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _remove_inline_space_selection(question: str) -> str:
    cleaned = re.sub(r"\b(?:for|from|in|inside|of)?\s*(?:space|workspace|project)\s+\d+\b", "", question, flags=re.IGNORECASE)
    return " ".join(cleaned.split())


def _looks_like_option_reply(question: str) -> bool:
    normalized = question.strip().lower()
    if not normalized:
        return False
    if normalized.isdigit():
        return True
    return bool(re.fullmatch(r"(?:space|workspace|project)\s+\d+", normalized))


async def _infer_original_question_from_history(selection: str, history: list[BaseMessage]) -> str | None:
    recent_messages = _recent_history_payload(history)
    if not recent_messages:
        return None
    try:
        provider, model = get_llm_router().route(LLMCapability.NORMALIZATION)
        response = await provider.generate_structured(
            StructuredLLMRequest(
                model=model,
                temperature=0,
                max_tokens=250,
                schema_name="SpaceSelectionHistoryResolution",
                messages=[
                    LLMMessage(
                        role="system",
                        content=(
                            "The latest user message is a workspace option selection. "
                            "Read the recent chat history and identify the earlier user request that should resume "
                            "after this workspace is selected. Use general language understanding, not keyword matching. "
                            "Return shouldResume=false if the history does not contain a clear workspace-dependent request. "
                            "The originalQuestion must be a clean English or original-language user request, with any "
                            "inline workspace option phrase removed."
                        ),
                    ),
                    LLMMessage(
                        role="user",
                        content=json.dumps(
                            {
                                "latestSelection": selection,
                                "recentMessages": recent_messages,
                            },
                            ensure_ascii=True,
                        ),
                    ),
                ],
            ),
            SpaceSelectionHistoryResolution,
        )
        if response.shouldResume and response.originalQuestion and response.confidence >= 0.65:
            return _remove_inline_space_selection(response.originalQuestion).strip() or response.originalQuestion
    except Exception:
        pass
    return _last_non_selection_user_request(history)


def _recent_history_payload(history: list[BaseMessage]) -> list[dict[str, str]]:
    payload: list[dict[str, str]] = []
    for message in history[-12:]:
        role = "user" if getattr(message, "type", None) == "human" else "assistant"
        content = str(getattr(message, "content", "") or "").strip()
        if content:
            payload.append({"role": role, "content": content[:1200]})
    return payload


def _last_non_selection_user_request(history: list[BaseMessage]) -> str | None:
    for message in reversed(history):
        if getattr(message, "type", None) != "human":
            continue
        content = str(getattr(message, "content", "") or "").strip()
        if not content or _looks_like_option_reply(content):
            continue
        return _remove_inline_space_selection(content).strip() or content
    return None


def _merge_pending_space_action(
    existing: dict[str, Any] | None,
    current: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not current:
        return None
    if not existing:
        return current
    if existing.get("type") != "select_option" or current.get("type") != "select_option":
        return current
    if existing.get("optionKind") != "spaces" or current.get("optionKind") != "spaces":
        return current

    current_plan = current.get("plan") or {}
    if current_plan.get("responseMode") != "list_options":
        return current

    merged = dict(current)
    previous_question = existing.get("originalQuestion")
    previous_plan = existing.get("plan")
    if previous_question:
        merged["originalQuestion"] = previous_question
    if previous_plan:
        merged["plan"] = previous_plan
    return merged


def _sanitize_public_answer(answer: str) -> str:
    lines = answer.splitlines()
    cleaned = []
    stop_markers = {
        "source:",
        "sources:",
        "source",
        "sources",
        "evidence:",
        "evidence",
        "context:",
        "context",
        "references:",
        "references",
        "retrieved context:",
        "retrieved context",
    }
    for line in lines:
        stripped = line.strip()
        marker = stripped.strip("*#:- ").lower()
        if marker in stop_markers:
            break
        if _looks_like_raw_context_line(stripped):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def _looks_like_raw_context_line(line: str) -> bool:
    lowered = line.lower()
    if " chunk " in lowered and "source " in lowered:
        return True
    if lowered.startswith("* [") or lowered.startswith("- [") or lowered.startswith("["):
        return "source" in lowered or "chunk" in lowered
    if "source " in lowered and "chunk" in lowered:
        return True
    return False


def _session_response(session) -> dict[str, Any]:
    return {
        "id": str(session.id),
        "userId": str(session.userId),
        "spaceId": None if session.spaceId is None else str(session.spaceId),
        "title": session.title,
        "status": session.status,
        "messageCount": session.messageCount,
        "createdAt": session.createdAt,
        "updatedAt": session.updatedAt,
    }
