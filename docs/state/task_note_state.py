"""State shared by the task and note LangGraph orchestration."""

from typing import Any, TypedDict


class TaskNoteState(TypedDict):
    user_id: str
    space_id: str
    chunks: list[dict[str, Any]]
    filtered_chunks: list[dict[str, Any]]
    reranked_chunks: list[dict[str, Any]]
    context_quality_score: float
    should_generate: bool
    tasks: list[dict[str, Any]]
    notes: list[dict[str, Any]]
    source_chunk_ids: list[str]
    errors: list[str]


def append_error(state: TaskNoteState, message: str) -> dict[str, list[str]]:
    """Return a LangGraph state patch with one appended error."""
    return {"errors": [*state.get("errors", []), message]}
