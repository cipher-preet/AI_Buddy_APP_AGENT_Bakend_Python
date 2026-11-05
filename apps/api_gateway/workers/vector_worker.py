# apps/api_gateway/workers/vector_worker.py

import asyncio
import os
import socket
from datetime import datetime, timezone
from pathlib import Path

from apps.api_gateway.config.setting import settings
from services.conversation.repository import ConversationRepository
from services.db.mongo import get_database
from services.observability.diagnostics import diag_log
from services.queue.job_lease import current_lease, logical_job_key, mark_current_job_completed
from services.queue.redis_queue import (
    delete_speech_job,
    get_job_result,
    mark_vector_stored,
    pop_completed_speech_job,
    push_completed_speech_job,
    record_vector_attempt,
    speech_vector_transport,
    vector_stage_settled,
)
from services.queue.streams import EventEnvelope, NonRetryableQueueError, RedisStreamConsumer, RedisStreamProducer

from services.vector.vector_service import (
    store_transcript_in_vector_db,
)

UPLOAD_DIR = Path("resources/audio_jobs").resolve()


def _remove_processed_audio_file(file_path: str | None) -> bool:
    if not file_path:
        return False

    path = Path(file_path).resolve()
    if UPLOAD_DIR not in path.parents or not path.is_file():
        return False

    try:
        path.unlink()
        return True
    except OSError as error:
        print(
            "Processed audio cleanup failed:",
            {
                "file_path": str(path),
                "error": str(error),
            },
        )
        return False


async def process_completed_speech_job(job_id: str) -> None:
    job = await get_job_result(job_id)
    if await vector_stage_settled(job_id) or (job and str(job.get("vector_status") or "") == "stored"):
        if job:
            await delete_speech_job(job_id)
        return

    if not job or job.get("status") != "completed":
        # Stream delivery must not finalize or embed while the transcript hash is missing.
        if current_lease() is not None:
            raise RuntimeError("completed speech is not durable yet")
        print("Completed speech job is not ready for vector processing:", job_id)
        return

    result = job.get("result") or {}

    transcript = str(result.get("transcript") or "").strip()
    language_code = result.get("language_code")
    request_id = result.get("request_id")
    provider = str(result.get("provider") or "unknown").strip() or "unknown"

    user_id = str(job.get("user_id") or "").strip()
    space_id = str(job.get("space_id") or "").strip()

    if not user_id or not space_id:
        if current_lease() is not None:
            raise NonRetryableQueueError("completed speech job is missing user or space")
        print("Invalid completed job data:", {"job_id": job_id})
        return

    conversation_id = str(job.get("conversation_id") or "").strip()
    sequence_number = job.get("sequence_number")
    # A vector retry re-enters here; the transcript chunk was already completed on the first pass.
    is_vector_retry = bool(job.get("vector_attempts"))
    if conversation_id and sequence_number is not None and not is_vector_retry:
        repository = ConversationRepository(get_database())
        await repository.complete_transcript_chunk(
            conversation_id=conversation_id,
            sequence_number=int(sequence_number),
            raw_text=str(transcript or ""),
            language_code=language_code,
            request_id=request_id,
            provider=provider,
        )
        conversation = await repository.get_conversation(conversation_id)
        if conversation and conversation.expectedLastSequence is not None:
            await RedisStreamProducer().publish(
                settings.REDIS_FINALIZATION_STREAM,
                EventEnvelope(
                    eventType="conversation.finalization.requested",
                    correlationId=conversation_id,
                    userId=user_id,
                    spaceId=space_id,
                    conversationId=conversation_id,
                    payload={"expectedLastSequence": conversation.expectedLastSequence},
                ),
            )

    if not transcript:
        audio_removed = _remove_processed_audio_file(job.get("file_path"))
        print(
            "Transcript vector job skipped empty transcript:",
            {
                "job_id": job_id,
                "user_id": user_id,
                "space_id": space_id,
                "conversation_id": conversation_id or None,
                "sequence_number": sequence_number,
                "provider": provider,
                "audio_removed": audio_removed,
            },
        )
        await mark_vector_stored(job_id)
        await delete_speech_job(job_id)
        return

    try:
        await store_transcript_in_vector_db(
            user_id=user_id,
            space_id=space_id,
            job_id=job_id,
            transcript=transcript,
            language_code=language_code,
            request_id=request_id,
        )
        await mark_vector_stored(job_id)
    except Exception as error:
        lease = current_lease()
        if lease is not None:
            attempt_number = int(lease.retry_count) + 1
            await record_vector_attempt(job_id, attempt_number)
            if attempt_number >= VECTOR_MAX_ATTEMPTS:
                await _record_vector_failure(
                    job_id, user_id, space_id, conversation_id, transcript, error
                )
                raise NonRetryableQueueError(str(error)[:300]) from error
            raise
        if await _schedule_vector_retry(job_id, job, error):
            return
        # Out of retries: keep the transcript durable for re-indexing; chat falls back to
        # lexical search over transcript_chunks meanwhile.
        await _record_vector_failure(job_id, user_id, space_id, conversation_id, transcript, error)
    audio_removed = _remove_processed_audio_file(job.get("file_path"))

    print(
        "Transcript vector job completed:",
        {
            "job_id": job_id,
            "user_id": user_id,
            "space_id": space_id,
            "conversation_id": conversation_id or None,
            "sequence_number": sequence_number,
            "provider": provider,
            "audio_removed": audio_removed,
        },
    )

    await delete_speech_job(job_id)


VECTOR_MAX_ATTEMPTS = 5
VECTOR_RETRY_DELAYS_SECONDS = (5, 30, 120, 300)
_retry_tasks: set[asyncio.Task] = set()


async def _schedule_vector_retry(job_id: str, job: dict, error: Exception) -> bool:
    attempts = int(job.get("vector_attempts") or 0) + 1
    if attempts >= VECTOR_MAX_ATTEMPTS:
        return False
    try:
        await record_vector_attempt(job_id, attempts)
    except Exception:
        return False
    delay = VECTOR_RETRY_DELAYS_SECONDS[min(attempts - 1, len(VECTOR_RETRY_DELAYS_SECONDS) - 1)]
    print(
        "Transcript vector job failed, retrying:",
        {"job_id": job_id, "attempt": attempts, "retry_in_seconds": delay, "error": str(error)[:300]},
    )
    task = asyncio.create_task(_requeue_after(job_id, delay))
    _retry_tasks.add(task)
    task.add_done_callback(_retry_tasks.discard)
    return True


async def _requeue_after(job_id: str, delay_seconds: float) -> None:
    await asyncio.sleep(delay_seconds)
    try:
        await push_completed_speech_job(job_id)
    except Exception as error:
        print("Transcript vector retry requeue failed:", {"job_id": job_id, "error": str(error)[:300]})


async def _record_vector_failure(
    job_id: str,
    user_id: str,
    space_id: str,
    conversation_id: str,
    transcript: str,
    error: Exception,
) -> None:
    print("Transcript vector job failed permanently:", {"job_id": job_id, "error": str(error)[:300]})
    try:
        await get_database().vector_index_failures.update_one(
            {"_id": job_id},
            {
                "$set": {
                    "userId": user_id,
                    "spaceId": space_id,
                    "conversationId": conversation_id or None,
                    "transcript": transcript,
                    "error": str(error)[:500],
                    "failedAt": datetime.now(timezone.utc),
                    "status": "PENDING_REINDEX",
                }
            },
            upsert=True,
        )
    except Exception as record_error:
        print("Transcript vector failure record failed:", {"job_id": job_id, "error": str(record_error)[:300]})


def _vector_consumer_name() -> str:
    host = (socket.gethostname() or "worker").split(".")[0][:24]
    return f"vector-{host}-{os.getpid()}"


def _wait_ms(event: EventEnvelope) -> int:
    created = event.createdAt
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return max(0, int((datetime.now(timezone.utc) - created).total_seconds() * 1000))


async def handle_vector_stream_event(event: EventEnvelope) -> None:
    logical_id = logical_job_key(event)
    job_id = str((event.payload or {}).get("jobId") or (event.payload or {}).get("job_id") or "").strip()
    diag_log(
        "vector_job_claimed",
        job_type=event.eventType,
        logical_job_id=logical_id,
        queue_name=settings.REDIS_VECTOR_STREAM,
        attempt=event.attempt,
        wait_ms=_wait_ms(event),
    )
    if event.attempt:
        diag_log(
            "vector_job_recovered",
            job_type=event.eventType,
            logical_job_id=logical_id,
            queue_name=settings.REDIS_VECTOR_STREAM,
            attempt=event.attempt,
        )
    if not job_id:
        raise NonRetryableQueueError("vector job is missing jobId")
    await process_completed_speech_job(job_id)
    mark_current_job_completed()
    diag_log(
        "vector_job_completed",
        job_type=event.eventType,
        logical_job_id=logical_id,
        queue_name=settings.REDIS_VECTOR_STREAM,
        attempt=event.attempt,
    )


def build_vector_stream_consumer(redis=None) -> RedisStreamConsumer:
    # One embedding at a time, matching the previous BRPOP loop.
    return RedisStreamConsumer(
        stream=settings.REDIS_VECTOR_STREAM,
        group=settings.REDIS_VECTOR_GROUP,
        consumer_name=_vector_consumer_name(),
        handler=handle_vector_stream_event,
        concurrency=1,
        max_retries=VECTOR_MAX_ATTEMPTS - 1,
        redis=redis,
    )


async def start_vector_consumer():
    if speech_vector_transport() == "stream":
        await build_vector_stream_consumer().run_forever()
        return
    print("Vector worker started...")

    while True:
        try:
            job_id = await pop_completed_speech_job()

            if not job_id:
                continue

            await process_completed_speech_job(job_id)

        except Exception as error:
            print("Vector worker error:", str(error))
            await asyncio.sleep(2)
