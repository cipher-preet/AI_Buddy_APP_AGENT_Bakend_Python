"""Public entry point for task and note graph invocation."""

import logging

from apps.agent_runtime.graphs.task_note_graph import build_task_note_graph
from apps.agent_runtime.state.task_note_state import TaskNoteState

logger = logging.getLogger(__name__)


def _initial_state(user_id: str, space_id: str) -> TaskNoteState:
    return {
        "user_id": user_id,
        "space_id": space_id,
        "chunks": [],
        "filtered_chunks": [],
        "reranked_chunks": [],
        "context_quality_score": 0.0,
        "should_generate": False,
        "tasks": [],
        "notes": [],
        "source_chunk_ids": [],
        "errors": [],
    }


async def invoke_task_note_graph(user_id: str, space_id: str) -> TaskNoteState:
    """Invoke the task/note orchestration graph for one user and space."""
    logger.info(
        "Invoking task note graph.",
        extra={"user_id": user_id, "space_id": space_id},
    )
    graph = build_task_note_graph()
    result = await graph.ainvoke(_initial_state(user_id=user_id, space_id=space_id))
    logger.info(
        "Task note graph invoke finished.",
        extra={
            "user_id": user_id,
            "space_id": space_id,
            "tasks": len(result.get("tasks", [])),
            "notes": len(result.get("notes", [])),
            "published_chunks": len(result.get("source_chunk_ids", [])),
            "errors": len(result.get("errors", [])),
        },
    )
    return result


async def run_task_note_orchestration(user_id: str, space_id: str) -> TaskNoteState:
    """Backward-compatible orchestration entry point."""
    return await invoke_task_note_graph(user_id=user_id, space_id=space_id)
