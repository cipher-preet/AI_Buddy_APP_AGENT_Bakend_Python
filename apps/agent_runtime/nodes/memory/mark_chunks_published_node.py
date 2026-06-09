"""Mark source chunks as published after successful persistence."""

import logging
from typing import Any

from apps.agent_runtime.rag.vectorstores.qdrant_store import mark_vectors_as_published
from apps.agent_runtime.state.task_note_state import TaskNoteState, append_error

logger = logging.getLogger(__name__)


async def mark_chunks_published(state: TaskNoteState) -> dict[str, Any]:
    """Publish only source chunks used by validated, saved records."""
    source_chunk_ids = set(state.get("source_chunk_ids", []))
    if not source_chunk_ids:
        logger.info(
            "No source chunks to mark published.",
            extra={"user_id": state["user_id"], "space_id": state["space_id"]},
        )
        return {}

    point_ids = [
        str(chunk.get("point_id"))
        for chunk in state.get("reranked_chunks", [])
        if chunk.get("chunkId") in source_chunk_ids and chunk.get("point_id")
    ]

    if not point_ids:
        return append_error(state, "mark_chunks_published skipped: no matching point ids")

    try:
        await mark_vectors_as_published(point_ids)
        logger.info(
            "Marked chunks published.",
            extra={
                "user_id": state["user_id"],
                "space_id": state["space_id"],
                "count": len(point_ids),
            },
        )
        return {}
    except Exception as error:
        logger.exception(
            "Failed to mark chunks published.",
            extra={"user_id": state["user_id"], "space_id": state["space_id"]},
        )
        return append_error(state, f"mark_chunks_published failed: {error}")
