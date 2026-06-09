"""Retrieve unpublished vector chunks for task and note orchestration."""

import logging
from typing import Any

from apps.agent_runtime.rag.vectorstores.qdrant_store import (
    MemoryVector,
    fetch_strict_unpublished_chunks,
)
from apps.agent_runtime.state.task_note_state import TaskNoteState, append_error

logger = logging.getLogger(__name__)


def _chunk_from_vector(vector: MemoryVector) -> dict[str, Any]:
    payload = vector.payload
    return {
        "point_id": vector.point_id,
        "chunkId": vector.chunk_id,
        "text": vector.text,
        "userId": payload.get("userId") or payload.get("user_id"),
        "spaceId": payload.get("spaceId") or payload.get("space_id"),
        "isPublish": payload.get("isPublish", False),
        "isDamaged": payload.get("isDamaged", False),
        "isUseful": payload.get("isUseful", True),
        "chunkStatus": payload.get("chunkStatus", "active"),
        "createdAt": payload.get("createdAt"),
        "sourceType": payload.get("sourceType") or payload.get("source"),
        "request_id": payload.get("request_id") or payload.get("requestId"),
    }


async def retrieve_unpublished_chunks(state: TaskNoteState) -> dict[str, Any]:
    """Fetch strictly eligible unpublished chunks from Qdrant."""
    user_id = state["user_id"]
    space_id = state["space_id"]

    try:
        vectors = await fetch_strict_unpublished_chunks(user_id=user_id, space_id=space_id)
        chunks = [_chunk_from_vector(vector) for vector in vectors]
        return {"chunks": chunks}
    except Exception as error:
        logger.exception(
            "Failed to retrieve unpublished chunks.",
            extra={"user_id": user_id, "space_id": space_id},
        )
        return append_error(state, f"retrieve_unpublished_chunks failed: {error}")
