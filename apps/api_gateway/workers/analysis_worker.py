import asyncio
import logging

from apps.agent_runtime.task_note_orchestration import invoke_task_note_graph
from apps.agent_runtime.services.task_note_service import ensure_memory_collections
from packages.schemas.memory_analysis_schema import AnalysisJob
from services.queue.redis_queue import pop_analysis_job

logger = logging.getLogger(__name__)


async def start_analysis_consumer():
    """Continuously process queued memory analysis jobs."""
    logger.info("Analysis worker started.")
    storage_ready = False

    while True:
        try:
            if not storage_ready:
                await ensure_memory_collections()
                storage_ready = True

            raw_job = await pop_analysis_job()

            if not raw_job:
                await asyncio.sleep(1)
                continue

            job = AnalysisJob.model_validate(raw_job)
            logger.info(
                "Processing analysis job.",
                extra={"user_id": job.user_id, "space_id": job.space_id},
            )

            result = await invoke_task_note_graph(
                user_id=job.user_id,
                space_id=job.space_id,
            )
            logger.info(
                "Analysis job finished.",
                extra={
                    "user_id": job.user_id,
                    "space_id": job.space_id,
                    "tasks": len(result.get("tasks", [])),
                    "notes": len(result.get("notes", [])),
                    "published_chunks": len(result.get("source_chunk_ids", [])),
                    "errors": result.get("errors", []),
                },
            )

        except Exception:
            logger.exception("Analysis worker error.")
            await asyncio.sleep(2)
