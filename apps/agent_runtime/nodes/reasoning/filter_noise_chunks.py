"""LLM-backed noise filtering for transcript chunks."""

import logging
from typing import Any

from openai import AsyncOpenAI

from apps.agent_runtime.llms.prompts.task_note_orchestration_prompt import (
    FILTER_CHUNKS_CHAT_PROMPT,
)
from apps.agent_runtime.nodes.chunk_prompting import render_chunk_context
from apps.agent_runtime.state.task_note_state import TaskNoteState, append_error
from apps.api_gateway.config.setting import settings
from packages.schemas.memory_analysis_schema import ChunkFilterOutput

logger = logging.getLogger(__name__)
client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)


def _metadata_eligible(chunk: dict[str, Any]) -> bool:
    text = str(chunk.get("text") or "").strip()
    return (
        bool(text)
        and chunk.get("isDamaged") is False
        and chunk.get("isUseful") is True
        and chunk.get("chunkStatus") == "active"
        and bool(chunk.get("chunkId"))
    )


async def filter_noise_chunks(state: TaskNoteState) -> dict[str, Any]:
    """Use structured LLM decisions to remove weak, damaged, duplicate, or noisy chunks."""
    candidate_chunks = [
        chunk for chunk in state.get("chunks", []) if _metadata_eligible(chunk)
    ]
    if not candidate_chunks:

        return {"filtered_chunks": []}

    messages = FILTER_CHUNKS_CHAT_PROMPT.format_messages(
        chunks=render_chunk_context(candidate_chunks, limit=40),
    )

    try:
        response = await client.chat.completions.parse(
            model=settings.OPENAI_CHAT_MODEL,
            temperature=0,
            response_format=ChunkFilterOutput,
            messages=messages,
        )
        parsed = response.choices[0].message.parsed
        if not parsed:
            raise ValueError("OpenAI returned no parsed chunk filter output.")

        print("this is parsed chunks ------->> ", parsed)

        useful_ids = {
            decision.chunk_id
            for decision in parsed.decisions
            if decision.is_useful and decision.confidence >= 0.65
        }
        seen_texts: set[str] = set()
        filtered_chunks: list[dict[str, Any]] = []
        for chunk in candidate_chunks:
            text_key = str(chunk.get("text") or "").strip().casefold()
            if chunk["chunkId"] not in useful_ids or text_key in seen_texts:
                continue
            seen_texts.add(text_key)
            filtered_chunks.append(chunk)

        logger.info(
            "LLM filtered chunks.",
            extra={
                "user_id": state["user_id"],
                "space_id": state["space_id"],
                "input_count": len(state.get("chunks", [])),
                "candidate_count": len(candidate_chunks),
                "output_count": len(filtered_chunks),
            },
        )
        return {"filtered_chunks": filtered_chunks}
    except Exception as error:
        logger.exception(
            "LLM chunk filtering failed.",
            extra={"user_id": state["user_id"], "space_id": state["space_id"]},
        )
        return {
            **append_error(state, f"filter_noise_chunks failed: {error}"),
            "filtered_chunks": [],
            "should_generate": False,
        }
