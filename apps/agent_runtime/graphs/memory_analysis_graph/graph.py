"""Memory analysis orchestration graph."""

import logging

from apps.agent_runtime.llms.openai.task_note_generator import generate_tasks_and_notes
from apps.agent_runtime.rag.vectorstores.qdrant_store import (
    MemoryVector,
    fetch_full_context_vectors,
    fetch_unpublished_vectors,
    mark_vectors_as_published,
)
from apps.agent_runtime.services.task_note_service import save_generated_tasks_and_notes
from packages.schemas.memory_analysis_schema import AnalysisJob

logger = logging.getLogger(__name__)


def _join_context(vectors: list[MemoryVector]) -> str:
    return "\n\n".join(vector.text.strip() for vector in vectors if vector.text.strip())


def _request_id_from_vectors(
    vectors: list[MemoryVector], fallback: str | None
) -> str | None:
    for vector in vectors:
        if vector.request_id:
            return vector.request_id
    return fallback


async def run_memory_analysis(job: AnalysisJob) -> dict[str, int | str]:
    """Run the full memory analysis pipeline for one user space."""
    unpublished_vectors = await fetch_unpublished_vectors(
        user_id=job.user_id,
        space_id=job.space_id,
    )

    if not unpublished_vectors:
        logger.info(
            "No unpublished vectors found for memory analysis.",
            extra={"user_id": job.user_id, "space_id": job.space_id},
        )
        return {"status": "skipped", "tasks": 0, "notes": 0}

    full_context_vectors = await fetch_full_context_vectors(
        user_id=job.user_id,
        space_id=job.space_id,
    )

    output = await generate_tasks_and_notes(
        full_context=_join_context(full_context_vectors),
        new_context=_join_context(unpublished_vectors),
    )

    counts = await save_generated_tasks_and_notes(
        user_id=job.user_id,
        space_id=job.space_id,
        request_id=_request_id_from_vectors(unpublished_vectors, job.request_id),
        output=output,
    )

    await mark_vectors_as_published([vector.point_id for vector in unpublished_vectors])

    return {"status": "completed", **counts}
