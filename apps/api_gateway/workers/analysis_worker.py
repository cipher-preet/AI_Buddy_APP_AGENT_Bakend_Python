import asyncio
import logging

from apps.agent_runtime.services.task_note_service import ensure_memory_collections
from apps.agent_runtime.services.transcript_analysis_service import (
    acquire_space_lock,
    process_transcript_analysis_job,
    release_space_lock,
)
from apps.api_gateway.config.setting import settings
from packages.schemas.memory_analysis_schema import AnalysisJob
from services.queue.redis_queue import (
    clear_transcript_analysis_queued,
    pop_analysis_job,
    push_analysis_job,
    push_failed_analysis_job,
)

logger = logging.getLogger(__name__)


async def start_analysis_consumer():
    """Continuously process queued memory analysis jobs."""
    logger.info("Analysis worker started.")
    storage_ready = False

    while True:
        raw_job = None
        try:
            if not storage_ready:
                await ensure_memory_collections()
                storage_ready = True

            raw_job = await pop_analysis_job()

            if not raw_job:
                await asyncio.sleep(1)
                continue

            job = AnalysisJob.model_validate(raw_job)
            if settings.TRANSCRIPT_ANALYSIS_DEBOUNCE_SECONDS > 0:
                await asyncio.sleep(settings.TRANSCRIPT_ANALYSIS_DEBOUNCE_SECONDS)

            logger.info(
                "Processing analysis job.",
                extra={
                    "user_id": job.user_id,
                    "space_id": job.space_id,
                    "window_id": job.window_id,
                    "attempt": job.attempt,
                },
            )

            lock = await acquire_space_lock(job.user_id, job.space_id)
            if not lock:
                retry_job = {**job.model_dump(mode="json"), "attempt": job.attempt + 1}
                if job.attempt < settings.TRANSCRIPT_ANALYSIS_MAX_ATTEMPTS:
                    await push_analysis_job(retry_job)
                else:
                    await push_failed_analysis_job({**retry_job, "error": "lock_timeout"})
                    await clear_transcript_analysis_queued(job.user_id, job.space_id)
                continue

            try:
                result = await process_transcript_analysis_job(job)
            finally:
                await release_space_lock(*lock)
                await clear_transcript_analysis_queued(job.user_id, job.space_id)

            if result.get("status") == "partial_completed":
                await push_analysis_job(
                    {
                        "job_type": "analyze_transcript_window",
                        "user_id": job.user_id,
                        "space_id": job.space_id,
                        "request_id": job.request_id,
                        "trigger_reason": "continue_unpublished_window",
                        "attempt": 1,
                    }
                )

            logger.info(
                "Analysis job finished.",
                extra={
                    "user_id": job.user_id,
                    "space_id": job.space_id,
                    "window_id": result.get("window_id"),
                    "status": result.get("status"),
                    "tasks_created": result.get("tasks_created", 0),
                    "tasks_updated": result.get("tasks_updated", 0),
                    "notes_created": result.get("notes_created", 0),
                    "notes_updated": result.get("notes_updated", 0),
                },
            )

        except Exception as error:
            logger.exception("Analysis worker error.")
            if raw_job:
                attempt = int(raw_job.get("attempt") or 1)
                failed_job = {**raw_job, "attempt": attempt + 1, "error": str(error)}
                if attempt < settings.TRANSCRIPT_ANALYSIS_MAX_ATTEMPTS:
                    await push_analysis_job(failed_job)
                else:
                    await push_failed_analysis_job(failed_job)
                    user_id = str(raw_job.get("user_id") or "").strip()
                    space_id = str(raw_job.get("space_id") or "").strip()
                    if user_id and space_id:
                        await clear_transcript_analysis_queued(user_id, space_id)
            await asyncio.sleep(2)
