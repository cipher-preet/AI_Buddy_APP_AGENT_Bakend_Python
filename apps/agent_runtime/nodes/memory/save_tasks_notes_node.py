import logging
from typing import Any

from apps.agent_runtime.services.task_note_service import save_generated_tasks_and_notes
from apps.agent_runtime.state.task_note_state import TaskNoteState, append_error
from packages.schemas.memory_analysis_schema import MemoryAnalysisOutput

logger = logging.getLogger(__name__)


def _request_id_from_chunks(chunks: list[dict[str, Any]]) -> str | None:
    for chunk in chunks:
        request_id = chunk.get("request_id")
        if request_id:
            return str(request_id)
    return None


async def save_tasks_notes(state: TaskNoteState) -> dict[str, Any]:
    """Save validated generated tasks and notes to MongoDB."""
    tasks = state.get("tasks", [])
    notes = state.get("notes", [])
    if not tasks and not notes:
        logger.info(
            "No validated tasks or notes to save.",
            extra={"user_id": state["user_id"], "space_id": state["space_id"]},
        )
        return {"source_chunk_ids": []}

    try:
        output = MemoryAnalysisOutput.model_validate(
            {
                "tasks": tasks,
                "notes": notes,
                "shouldPublishChunks": bool(state.get("source_chunk_ids")),
            }
        )
        counts = await save_generated_tasks_and_notes(
            user_id=state["user_id"],
            space_id=state["space_id"],
            request_id=_request_id_from_chunks(state.get("reranked_chunks", [])),
            output=output,
        )
        logger.info(
            "Saved validated tasks and notes.",
            extra={
                "user_id": state["user_id"],
                "space_id": state["space_id"],
                "tasks": counts["tasks"],
                "notes": counts["notes"],
            },
        )
        return {}
    except Exception as error:
        logger.exception(
            "Failed to save validated tasks and notes.",
            extra={"user_id": state["user_id"], "space_id": state["space_id"]},
        )
        return {
            **append_error(state, f"save_tasks_notes failed: {error}"),
            "source_chunk_ids": [],
        }
