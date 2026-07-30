from __future__ import annotations

from typing import Any, TypedDict

try:
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
    from langgraph.graph import END, StateGraph
except ImportError:  # Dependencies are declared; this keeps tests importable before install.
    AIMessage = BaseMessage = HumanMessage = None
    END = None
    StateGraph = None

from apps.api_gateway.config.setting import settings
from services.chat.models import MAX_CHAT_MESSAGES
from services.chat.repository import ChatRepository, encode_sessions_cursor
from services.chat.retrieval import ChatRetriever, format_context
from services.llm.models import LLMMessage, LLMRequest
from services.llm.router import LLMCapability, get_llm_router


SYSTEM_PROMPT = """You are Buddy's chat assistant.
Answer in English only, even when the user asks in Hindi, Hinglish, or any other language.
Use the retrieved context as evidence. The context may be in Hindi, English, or mixed language; translate and reason over it internally.
If the answer is not supported by the retrieved context or chat history, say what is missing and ask a concise follow-up.
Be direct, useful, and avoid inventing facts.
Never return an empty response.
Never include raw source IDs, chunk numbers, citations, source lists, or verbatim retrieved context in the final answer.
Do not add a Source, Sources, Evidence, Context, or References section."""


class ChatState(TypedDict):
    user_id: str
    space_id: str | None
    question: str
    search_queries: list[str]
    history: list[BaseMessage]
    context_text: str
    answer: str
    provider: str | None
    model: str | None
    usage: dict[str, int]


class ChatService:
    def __init__(
        self,
        repository: ChatRepository | None = None,
        retriever: ChatRetriever | None = None,
    ):
        self.repository = repository or ChatRepository()
        self.retriever = retriever or ChatRetriever()

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
        result = await self._invoke_graph(
            {
                "user_id": user_id,
                "space_id": space_id,
                "question": question,
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
        await self.repository.sync_message_count(session.id)
        return {
            "chatId": str(session.id),
            "createdNewChat": created_new_chat or (chat_id is not None and str(session.id) != chat_id),
            "answer": result["answer"],
        }

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
            "search_queries": [inputs["question"]],
            "history": history,
            "context_text": "",
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
        }

    def _build_graph(self):
        if StateGraph is None:
            return self._fallback_graph

        workflow = StateGraph(ChatState)
        workflow.add_node("expand_query", self._expand_query_node)
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("generate", self._generate_node)
        workflow.set_entry_point("expand_query")
        workflow.add_edge("expand_query", "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        return workflow.compile()

    async def _fallback_graph(self, state: ChatState) -> ChatState:
        state = await self._expand_query_node(state)
        state = await self._retrieve_node(state)
        return await self._generate_node(state)

    async def _expand_query_node(self, state: ChatState) -> ChatState:
        state["search_queries"] = await _build_multilingual_search_queries(state["question"])
        return state

    async def _retrieve_node(self, state: ChatState) -> ChatState:
        contexts = await self.retriever.retrieve_many(
            state["search_queries"],
            state["user_id"],
            state["space_id"],
        )
        state["context_text"] = format_context(contexts)
        return state

    async def _generate_node(self, state: ChatState) -> ChatState:
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
