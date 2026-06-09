"""LLM-backed validation for generated tasks and notes."""

import json
import logging
from datetime import date, datetime
from typing import Any

from openai import AsyncOpenAI

from apps.agent_runtime.llms.prompts.task_note_orchestration_prompt import (
    VALIDATE_TASK_NOTES_CHAT_PROMPT,
)
from apps.agent_runtime.nodes.chunk_prompting import render_chunk_context
from apps.agent_runtime.state.task_note_state import TaskNoteState, append_error
from apps.api_gateway.config.setting import settings
from packages.schemas.memory_analysis_schema import TaskNoteValidationOutput

logger = logging.getLogger(__name__)
client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)


def _source_ids(item: dict[str, Any]) -> list[str]:
    return [str(source_id) for source_id in item.get("sourceChunkIds", [])]


def _has_required_fields(item: dict[str, Any], *, body_key: str) -> bool:
    return (
        bool(str(item.get("title") or "").strip())
        and bool(str(item.get(body_key) or "").strip())
        and bool(_source_ids(item))
        and float(item.get("confidence") or 0.0) >= 0.7
    )


def _source_ids_exist(item: dict[str, Any], chunk_ids: set[str]) -> bool:
    return all(source_id in chunk_ids for source_id in _source_ids(item))


def _json_default(value: object) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


async def validate_tasks_notes(state: TaskNoteState) -> dict[str, Any]:
    """Validate generated output semantically against retrieved chunks."""
    tasks = state.get("tasks", [])
    notes = state.get("notes", [])
    if not tasks and not notes:
        return {"tasks": [], "notes": [], "source_chunk_ids": []}

    chunks = state.get("reranked_chunks", [])
    chunk_ids = {
        str(chunk.get("chunkId"))
        for chunk in chunks
        if chunk.get("chunkId")
    }

    generated_items = {
        "tasks": tasks,
        "notes": notes,
    }
    messages = VALIDATE_TASK_NOTES_CHAT_PROMPT.format_messages(
        user_id=state["user_id"],
        space_id=state["space_id"],
        chunks=render_chunk_context(chunks, limit=20),
        generated_items=json.dumps(
            generated_items,
            ensure_ascii=True,
            default=_json_default,
        ),
    )

    try:
        response = await client.chat.completions.parse(
            model=settings.OPENAI_CHAT_MODEL,
            temperature=0,
            response_format=TaskNoteValidationOutput,
            messages=messages,
        )
        parsed = response.choices[0].message.parsed
        if not parsed:
            raise ValueError("OpenAI returned no parsed task/note validation output.")

        valid_task_indexes = {
            decision.item_index
            for decision in parsed.decisions
            if decision.item_type == "task"
            and decision.is_valid
            and decision.confidence >= 0.7
        }
        valid_note_indexes = {
            decision.item_index
            for decision in parsed.decisions
            if decision.item_type == "note"
            and decision.is_valid
            and decision.confidence >= 0.7
        }

        valid_tasks = [
            task
            for index, task in enumerate(tasks)
            if index in valid_task_indexes
            and _has_required_fields(task, body_key="description")
            and _source_ids_exist(task, chunk_ids)
        ]
        valid_notes = [
            note
            for index, note in enumerate(notes)
            if index in valid_note_indexes
            and _has_required_fields(note, body_key="content")
            and _source_ids_exist(note, chunk_ids)
        ]
        source_chunk_ids = sorted(
            {
                source_id
                for item in [*valid_tasks, *valid_notes]
                for source_id in _source_ids(item)
            }
        )

        rejected_count = len(tasks) + len(notes) - len(valid_tasks) - len(valid_notes)
        errors = state.get("errors", [])
        if rejected_count:
            errors = [*errors, f"Rejected {rejected_count} generated task/note item(s)."]

        logger.info(
            "LLM validated generated tasks and notes.",
            extra={
                "user_id": state["user_id"],
                "space_id": state["space_id"],
                "valid_tasks": len(valid_tasks),
                "valid_notes": len(valid_notes),
                "rejected": rejected_count,
            },
        )
        return {
            "tasks": valid_tasks,
            "notes": valid_notes,
            "source_chunk_ids": source_chunk_ids,
            "errors": errors,
        }
    except Exception as error:
        logger.exception(
            "LLM task/note validation failed.",
            extra={"user_id": state["user_id"], "space_id": state["space_id"]},
        )
        return {
            **append_error(state, f"validate_tasks_notes failed: {error}"),
            "tasks": [],
            "notes": [],
            "source_chunk_ids": [],
        }
