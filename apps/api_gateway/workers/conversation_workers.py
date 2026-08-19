from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

from apps.api_gateway.config.setting import settings
from services.conversation.finalization import ConversationFinalizationCoordinator
from services.conversation.inactivity import ConversationInactivityScanner
from services.conversation.incremental import IncrementalMeetingProcessor
from services.conversation.repository import ConversationRepository
from services.conversation.workflow import ConversationProcessingWorkflow
from services.db.mongo import get_database
from services.queue.streams import EventEnvelope, NonRetryableQueueError, RedisStreamConsumer
from services.queue.streams import RedisStreamProducer
from services.queue.redis_queue import redis_client
from services.storage.s3_audio_storage import (
    get_s3_audio_storage,
    safe_temp_audio_path,
    temp_audio_root,
    validate_conversation_audio_object_key,
)
from services.speech.transcription_router import transcribe_from_path_with_fallback
from services.conversation.models import ConversationStatus, STTStatus


async def handle_stt_event(event: EventEnvelope) -> None:
    repository = ConversationRepository(get_database())
    payload = event.payload
    conversation_id = payload["conversationId"]
    sequence_number = int(payload["sequenceNumber"])
    existing = await repository.get_transcript_chunk(conversation_id, sequence_number)
    if existing and existing.sttStatus == STTStatus.COMPLETED:
        print(
            "Duplicate STT event skipped:",
            {
                "eventId": event.eventId,
                "correlationId": event.correlationId,
                "conversationId": conversation_id,
                "sequenceNumber": sequence_number,
                "stage": "stt_duplicate_completed",
            },
        )
        return

    await repository.mark_transcript_chunk_processing(conversation_id, sequence_number)
    local_path: Path | None = None
    job_dir: Path | None = None

    try:
        file_path = payload.get("filePath")
        filename = payload.get("filename") or f"{payload.get('chunkId')}.audio"
        content_type = payload.get("contentType") or "audio/wav"
        if isinstance(file_path, str) and file_path.startswith("s3://"):
            parsed = urlparse(file_path)
            payload = {
                **payload,
                "storageProvider": "s3",
                "bucket": parsed.netloc,
                "objectKey": parsed.path.lstrip("/"),
            }
        if str(payload.get("storageProvider") or "").lower() == "s3" or payload.get("objectKey") or payload.get("s3ObjectKey"):
            bucket = str(payload.get("bucket") or payload.get("s3Bucket") or "")
            object_key = str(payload.get("objectKey") or payload.get("s3ObjectKey") or "")
            validate_conversation_audio_object_key(
                object_key=object_key,
                user_id=event.userId,
                space_id=event.spaceId,
                conversation_id=conversation_id,
            )
            local_path = safe_temp_audio_path(
                {
                    "job_id": payload.get("jobId") or event.eventId,
                    "filename": filename,
                }
            )
            job_dir = local_path.parent
            _cleanup_job_dir(job_dir)
            downloaded = await get_s3_audio_storage().download_file(
                bucket=bucket,
                object_key=object_key,
                destination=local_path,
            )
            file_path = str(downloaded)

        if not file_path:
            raise ValueError("STT event is missing audio file reference")
        print(
            "Conversation chunk STT routing:",
            {
                "eventId": event.eventId,
                "conversationId": conversation_id,
                "sequenceNumber": sequence_number,
                "filename": filename,
                "contentType": content_type,
            },
        )
        result = await transcribe_from_path_with_fallback(
            file_path=file_path,
            filename=filename,
            content_type=content_type,
        )
        print(
            "Conversation chunk STT provider selected:",
            {
                "eventId": event.eventId,
                "conversationId": conversation_id,
                "sequenceNumber": sequence_number,
                "provider": result.get("provider"),
                "model": result.get("model"),
                "languageCode": result.get("language_code"),
                "isEmptyTranscript": result.get("is_empty_transcript"),
            },
        )
        await repository.complete_transcript_chunk(
            conversation_id=conversation_id,
            sequence_number=sequence_number,
            raw_text=result["transcript"],
            language_code=result.get("language_code"),
            request_id=result.get("request_id"),
            provider=result.get("provider") or "unknown",
        )
        conversation = await repository.get_conversation(conversation_id)
        if conversation and conversation.status in {ConversationStatus.COMPLETED, ConversationStatus.FAILED}:
            print(
                "Late STT completion after terminal conversation status:",
                {
                    "eventId": event.eventId,
                    "conversationId": conversation_id,
                    "sequenceNumber": sequence_number,
                    "status": conversation.status.value,
                    "isEmptyTranscript": result.get("is_empty_transcript"),
                    "transcriptChars": len(str(result.get("transcript") or "")),
                },
            )
            return
        await RedisStreamProducer().publish(
            settings.REDIS_TRANSCRIPT_READY_STREAM,
            EventEnvelope(
                eventType="conversation.transcript.ready",
                correlationId=conversation_id,
                userId=event.userId,
                spaceId=event.spaceId,
                conversationId=conversation_id,
                causationId=event.eventId,
                payload={
                    "conversationId": conversation_id,
                    "sequenceNumber": sequence_number,
                },
            ),
        )
        if conversation and conversation.expectedLastSequence is not None:
            await RedisStreamProducer().publish(
                settings.REDIS_FINALIZATION_STREAM,
                EventEnvelope(
                    eventType="conversation.finalization.requested",
                    correlationId=conversation_id,
                    userId=conversation.userId,
                    spaceId=conversation.spaceId,
                    conversationId=conversation_id,
                    causationId=event.eventId,
                    payload={"expectedLastSequence": conversation.expectedLastSequence},
                ),
            )
    except ValueError as error:
        await repository.fail_transcript_chunk(conversation_id, sequence_number, str(error))
        await _publish_finalization_if_stopped(repository, conversation_id, event.eventId)
        raise NonRetryableQueueError(str(error)) from error
    except Exception as error:
        await repository.fail_transcript_chunk(conversation_id, sequence_number, str(error))
        raise
    finally:
        if job_dir is not None:
            try:
                _cleanup_job_dir(job_dir)
            except Exception as cleanup_error:
                print("Conversation worker temporary audio cleanup failed:", str(cleanup_error))


async def handle_audio_event(event: EventEnvelope) -> None:
    await RedisStreamProducer().publish(
        settings.REDIS_STT_STREAM,
        EventEnvelope(
            eventType="stt.requested",
            correlationId=event.correlationId,
            causationId=event.eventId,
            userId=event.userId,
            spaceId=event.spaceId,
            conversationId=event.conversationId,
            payload=event.payload,
        ),
    )


async def _publish_finalization_if_stopped(
    repository: ConversationRepository,
    conversation_id: str,
    causation_id: str,
) -> None:
    conversation = await repository.get_conversation(conversation_id)
    if conversation and conversation.expectedLastSequence is not None:
        await RedisStreamProducer().publish(
            settings.REDIS_FINALIZATION_STREAM,
            EventEnvelope(
                eventType="conversation.finalization.requested",
                correlationId=conversation_id,
                userId=str(conversation.userId),
                spaceId=str(conversation.spaceId),
                conversationId=conversation_id,
                causationId=causation_id,
                payload={"expectedLastSequence": conversation.expectedLastSequence},
            ),
        )


async def handle_finalization_event(event: EventEnvelope) -> None:
    repository = ConversationRepository(get_database())
    await ConversationFinalizationCoordinator(repository).finalize(event.conversationId)


async def handle_transcript_ready_event(event: EventEnvelope) -> None:
    repository = ConversationRepository(get_database())
    await IncrementalMeetingProcessor(repository).close_ready_windows(event.conversationId)


async def handle_window_extraction_event(event: EventEnvelope) -> None:
    repository = ConversationRepository(get_database())
    window_id = event.payload.get("windowId")
    if not window_id:
        raise ValueError("Window extraction event missing windowId")
    conversation = await repository.get_conversation(event.conversationId)
    if conversation and conversation.status in {ConversationStatus.COMPLETED, ConversationStatus.FAILED}:
        print(
            "Late window extraction skipped after terminal conversation status:",
            {
                "conversationId": event.conversationId,
                "windowId": window_id,
                "status": conversation.status.value,
            },
        )
        return
    await IncrementalMeetingProcessor(repository).extract_window(str(window_id))
    await _publish_finalization_if_stopped(repository, event.conversationId, event.eventId)


async def handle_processing_event(event: EventEnvelope) -> None:
    repository = ConversationRepository(get_database())
    try:
        await asyncio.wait_for(
            ConversationProcessingWorkflow(repository).run(event.conversationId),
            timeout=settings.CONVERSATION_PROCESSING_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as error:
        message = f"Conversation processing timed out after {settings.CONVERSATION_PROCESSING_TIMEOUT_SECONDS} seconds."
        await repository.mark_active_extraction_run_failed(event.conversationId, message)
        await repository.mark_conversation_failed(
            event.conversationId,
            message,
        )


def build_stt_consumer() -> RedisStreamConsumer:
    return RedisStreamConsumer(
        stream=settings.REDIS_STT_STREAM,
        group=settings.REDIS_STT_GROUP,
        consumer_name=f"stt-{uuid4().hex[:8]}",
        handler=handle_stt_event,
        concurrency=settings.STT_WORKER_CONCURRENCY or min(settings.WORKER_CONCURRENCY, settings.SARVAM_MAX_CONCURRENCY),
    )


def build_audio_consumer() -> RedisStreamConsumer:
    return RedisStreamConsumer(
        stream=settings.REDIS_AUDIO_STREAM,
        group=settings.REDIS_AUDIO_GROUP,
        consumer_name=f"audio-{uuid4().hex[:8]}",
        handler=handle_audio_event,
        concurrency=settings.AUDIO_WORKER_CONCURRENCY or settings.WORKER_CONCURRENCY,
    )


def build_finalization_consumer() -> RedisStreamConsumer:
    return RedisStreamConsumer(
        stream=settings.REDIS_FINALIZATION_STREAM,
        group=settings.REDIS_FINALIZATION_GROUP,
        consumer_name=f"finalization-{uuid4().hex[:8]}",
        handler=handle_finalization_event,
        concurrency=settings.FINALIZATION_WORKER_CONCURRENCY or settings.WORKER_CONCURRENCY,
    )


def build_transcript_ready_consumer() -> RedisStreamConsumer:
    return RedisStreamConsumer(
        stream=settings.REDIS_TRANSCRIPT_READY_STREAM,
        group=settings.REDIS_TRANSCRIPT_GROUP,
        consumer_name=f"transcript-{uuid4().hex[:8]}",
        handler=handle_transcript_ready_event,
        concurrency=settings.TRANSCRIPT_WINDOW_WORKER_CONCURRENCY or settings.WORKER_CONCURRENCY,
    )


def build_window_extraction_consumer() -> RedisStreamConsumer:
    return RedisStreamConsumer(
        stream=settings.REDIS_WINDOW_EXTRACTION_STREAM,
        group=settings.REDIS_WINDOW_EXTRACTION_GROUP,
        consumer_name=f"window-extraction-{uuid4().hex[:8]}",
        handler=handle_window_extraction_event,
        concurrency=settings.WINDOW_EXTRACTION_WORKER_CONCURRENCY or settings.MAX_ACTIVE_LLM_CALLS_PER_CONVERSATION,
    )


def build_processing_consumer() -> RedisStreamConsumer:
    return RedisStreamConsumer(
        stream=settings.REDIS_PROCESSING_STREAM,
        group=settings.REDIS_PROCESSING_GROUP,
        consumer_name=f"processing-{uuid4().hex[:8]}",
        handler=handle_processing_event,
        concurrency=settings.PROCESSING_WORKER_CONCURRENCY,
    )


async def run_inactivity_scanner() -> None:
    import asyncio

    repository = ConversationRepository(get_database())
    scanner = ConversationInactivityScanner(repository)
    interval = max(30, settings.CONVERSATION_INACTIVITY_TIMEOUT_SECONDS // 3)
    while True:
        await scanner.scan_once()
        await asyncio.sleep(interval)


async def run_retry_relay() -> None:
    import asyncio
    import time

    producer = RedisStreamProducer()
    while True:
        entries = await redis_client.xrange(settings.REDIS_RETRY_STREAM, min="-", max="+", count=50)
        now = time.time()
        for message_id, fields in entries:
            not_before = float(fields.get("notBefore") or 0)
            if not_before > now:
                continue
            target_stream = fields.get("targetStream")
            raw_event = fields.get("event")
            if target_stream and raw_event:
                event = EventEnvelope.model_validate_json(raw_event)
                retry_event = event.model_copy(
                    update={
                        "eventId": str(uuid4()),
                        "causationId": event.eventId,
                    }
                )
                await producer.publish(target_stream, retry_event)
            await redis_client.xdel(settings.REDIS_RETRY_STREAM, message_id)
        await asyncio.sleep(1)


def _cleanup_job_dir(job_dir: Path) -> None:
    root = temp_audio_root()
    resolved = job_dir.resolve()
    if resolved == root:
        raise ValueError(f"Refusing to clean temporary audio root: {root}")
    if root not in resolved.parents:
        raise ValueError(f"Refusing to clean path outside temporary audio root: {root}")
    if resolved.exists():
        shutil.rmtree(resolved)
