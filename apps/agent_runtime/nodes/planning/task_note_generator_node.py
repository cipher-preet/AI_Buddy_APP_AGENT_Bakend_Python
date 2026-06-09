"""LLM generation node for high-quality tasks and notes."""

import logging
from typing import Any

from openai import AsyncOpenAI

from apps.agent_runtime.llms.prompts.task_note_orchestration_prompt import (
    TASK_NOTE_GENERATOR_CHAT_PROMPT,
)
from apps.agent_runtime.nodes.chunk_prompting import render_chunk_context
from apps.agent_runtime.state.task_note_state import TaskNoteState, append_error
from apps.api_gateway.config.setting import settings
from packages.schemas.memory_analysis_schema import MemoryAnalysisOutput

logger = logging.getLogger(__name__)
client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)


async def generate_tasks_notes(state: TaskNoteState) -> dict[str, Any]:
    """Generate candidate tasks and notes from strong context."""
    if not state.get("should_generate"):
        return {"tasks": [], "notes": [], "source_chunk_ids": []}

    chunks = state.get("reranked_chunks", [])
    messages = TASK_NOTE_GENERATOR_CHAT_PROMPT.format_messages(
        user_id=state["user_id"],
        space_id=state["space_id"],
        chunks=render_chunk_context(chunks, limit=12),
    )

    try:
        response = await client.chat.completions.parse(
            model=settings.OPENAI_CHAT_MODEL,
            temperature=0,
            response_format=MemoryAnalysisOutput,
            messages=messages,
        )
        parsed = response.choices[0].message.parsed
        if not parsed:
            raise ValueError("OpenAI returned no parsed task/note output.")

        tasks = [task.model_dump(by_alias=True, mode="json") for task in parsed.tasks]
        notes = [note.model_dump(by_alias=True, mode="json") for note in parsed.notes]
        source_chunk_ids = sorted(
            {
                str(chunk_id)
                for item in [*tasks, *notes]
                for chunk_id in item.get("sourceChunkIds", [])
            }
        )
        if not parsed.should_publish_chunks:
            source_chunk_ids = []

        logger.info(
            "Generated task and note candidates.",
            extra={
                "user_id": state["user_id"],
                "space_id": state["space_id"],
                "tasks": len(tasks),
                "notes": len(notes),
            },
        )
        return {
            "tasks": tasks,
            "notes": notes,
            "source_chunk_ids": source_chunk_ids,
        }
    except Exception as error:
        logger.exception(
            "Failed to generate task and note candidates.",
            extra={"user_id": state["user_id"], "space_id": state["space_id"]},
        )
        return {
            **append_error(state, f"generate_tasks_notes failed: {error}"),
            "should_generate": False,
            "tasks": [],
            "notes": [],
            "source_chunk_ids": [],
        }
