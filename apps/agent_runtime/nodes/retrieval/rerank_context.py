"""LLM-backed reranking for task and note context."""

import logging
from typing import Any

from openai import AsyncOpenAI

from apps.agent_runtime.llms.prompts.task_note_orchestration_prompt import (
    RERANK_CONTEXT_CHAT_PROMPT,
)
from apps.agent_runtime.nodes.chunk_prompting import render_chunk_context
from apps.agent_runtime.state.task_note_state import TaskNoteState, append_error
from apps.api_gateway.config.setting import settings
from packages.schemas.memory_analysis_schema import ContextRerankOutput

logger = logging.getLogger(__name__)
client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)


async def rerank_context(state: TaskNoteState) -> dict[str, Any]:
    """Use structured LLM output to rank filtered chunks by usefulness."""
    chunks = state.get("filtered_chunks", [])
    if not chunks:
        return {"reranked_chunks": []}

    messages = RERANK_CONTEXT_CHAT_PROMPT.format_messages(
        user_id=state["user_id"],
        space_id=state["space_id"],
        chunks=render_chunk_context(chunks, limit=40),
    )

    try:
        response = await client.chat.completions.parse(
            model=settings.OPENAI_CHAT_MODEL,
            temperature=0,
            response_format=ContextRerankOutput,
            messages=messages,
        )
        parsed = response.choices[0].message.parsed
        if not parsed:
            raise ValueError("OpenAI returned no parsed context rerank output.")

        scores_by_id = {item.chunk_id: item.relevance_score for item in parsed.chunks}
        reasons_by_id = {item.chunk_id: item.reason for item in parsed.chunks}
        reranked_chunks = [
            {
                **chunk,
                "relevance_score": scores_by_id.get(str(chunk.get("chunkId")), 0.0),
                "rerank_reason": reasons_by_id.get(str(chunk.get("chunkId")), ""),
            }
            for chunk in chunks
        ]
        reranked_chunks.sort(
            key=lambda chunk: float(chunk.get("relevance_score") or 0.0),
            reverse=True,
        )

        return {"reranked_chunks": reranked_chunks}
    except Exception as error:
        logger.exception(
            "LLM context reranking failed.",
            extra={"user_id": state["user_id"], "space_id": state["space_id"]},
        )
        return {
            **append_error(state, f"rerank_context failed: {error}"),
            "reranked_chunks": [],
            "should_generate": False,
        }
