from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

from redis.exceptions import ConnectionError, RedisError, TimeoutError as RedisTimeoutError

from apps.api_gateway.config.setting import settings
from services.conversation.finalization import ConversationFinalizationCoordinator
from services.conversation.inactivity import ConversationInactivityScanner
from services.conversation.incremental import IncrementalMeetingProcessor
from services.conversation.repository import ConversationRepository
from services.conversation.stt_failure import (
    TERMINAL_FAILED_PERMANENTLY,
    classify_stt_failure,
    is_terminal_failed_chunk,
    log_stt_terminal_state,
)
from services.conversation.workflow import ConversationProcessingWorkflow
from services.db.mongo import get_database
from services.queue.streams import EventEnvelope, NonRetryableQueueError, RedisStreamConsumer
from services.queue.streams import RedisStreamProducer
from services.queue.redis_queue import redis_client
from services.storage.s3_audio_storage import (
    PermanentS3StorageError,
    get_s3_audio_storage,
    safe_temp_audio_path,
    temp_audio_root,
    validate_conversation_audio_object_key,
)
from services.speech.transcription_router import transcribe_from_path_with_fallback
from services.conversation.models import ConversationStatus, STTStatus
from services.llm.router import LLMCapability, get_llm_router


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
    if existing and is_terminal_failed_chunk(existing):
        print(
            "Duplicate STT event skipped:",
            {
                "eventId": event.eventId,
                "correlationId": event.correlationId,
                "conversationId": conversation_id,
                "sequenceNumber": sequence_number,
                "stage": "stt_duplicate_terminal_failed",
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
        stt_kwargs = {
            "file_path": file_path,
            "filename": filename,
            "content_type": content_type,
        }
        keyterm_context = {}
        for field in ("keyterms", "keyterm", "space_keyterms", "terminology", "terms"):
            value = payload.get(field)
            if value and not isinstance(value, (bool, dict)):
                keyterm_context[field] = value
        if keyterm_context:
            stt_kwargs["context"] = keyterm_context
        result = await transcribe_from_path_with_fallback(**stt_kwargs)
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
                "isUncertainTranscript": result.get("is_uncertain_transcript"),
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
    except Exception as error:
        await _record_stt_sequence_failure(
            repository,
            event=event,
            conversation_id=conversation_id,
            sequence_number=sequence_number,
            error=error,
            existing=existing,
            retry_exhausted=False,
        )
        classification = classify_stt_failure(error)
        if classification.permanent:
            raise NonRetryableQueueError(classification.sanitized_message) from error
        raise
    finally:
        if job_dir is not None:
            try:
                _cleanup_job_dir(job_dir)
            except Exception as cleanup_error:
                print("Conversation worker temporary audio cleanup failed:", str(cleanup_error))


async def handle_stt_dead_letter(event: EventEnvelope, error: Exception) -> None:
    payload = event.payload or {}
    conversation_id = str(payload.get("conversationId") or event.conversationId or "").strip()
    sequence_value = payload.get("sequenceNumber")
    if not conversation_id or sequence_value is None:
        return
    repository = ConversationRepository(get_database())
    existing = await repository.get_transcript_chunk(conversation_id, int(sequence_value))
    await _record_stt_sequence_failure(
        repository,
        event=event,
        conversation_id=conversation_id,
        sequence_number=int(sequence_value),
        error=error,
        existing=existing,
        retry_exhausted=True,
    )


async def _record_stt_sequence_failure(
    repository: ConversationRepository,
    *,
    event: EventEnvelope,
    conversation_id: str,
    sequence_number: int,
    error: Exception,
    existing,
    retry_exhausted: bool,
) -> bool:
    payload = event.payload or {}
    job_id = str(payload.get("jobId") or payload.get("chunkId") or event.eventId)
    stage = "queue_dlq" if retry_exhausted else None
    if stage is None and (isinstance(error, PermanentS3StorageError) or _is_s3_event(payload)):
        stage = "s3_download"
    classification = classify_stt_failure(error, retry_exhausted=retry_exhausted, stage=stage)
    retry_count = int(getattr(existing, "sttAttempts", 0) or event.attempt or 0) + 1
    terminal = bool(classification.permanent or retry_exhausted)
    transitioned = await repository.fail_transcript_chunk(
        conversation_id,
        sequence_number,
        classification.sanitized_message,
        job_id=job_id,
        failure_stage=classification.failure_stage,
        failure_type=classification.failure_type,
        provider=classification.provider,
        retry_count=retry_count,
        terminal=terminal,
    )
    if terminal:
        if transitioned:
            log_stt_terminal_state(
                conversation_id=conversation_id,
                sequence_number=sequence_number,
                job_id=job_id,
                failure_type=classification.failure_type,
                retry_count=retry_count,
                terminal_state=TERMINAL_FAILED_PERMANENTLY,
            )
        await _publish_finalization_if_stopped(repository, conversation_id, event.eventId)
    return transitioned


def _is_s3_event(payload: dict) -> bool:
    return str(payload.get("storageProvider") or payload.get("storage_provider") or "").lower() == "s3" or bool(
        payload.get("objectKey") or payload.get("s3ObjectKey") or payload.get("bucket")
    )


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
    provider, model = get_llm_router().route(LLMCapability.HIGH_ACCURACY_REASONING)
    print(
        "Window extraction LLM route:",
        {
            "conversationId": event.conversationId,
            "windowId": window_id,
            "provider": getattr(provider, "name", None),
            "model": model,
            "krutrimConfigured": _provider_configured("krutrim"),
            "mistralConfigured": _provider_configured("mistral"),
        },
    )
    await IncrementalMeetingProcessor(repository).extract_window(str(window_id))
    await _publish_finalization_if_stopped(repository, event.conversationId, event.eventId)


async def handle_processing_event(event: EventEnvelope) -> None:
    repository = ConversationRepository(get_database())
    provider, model = get_llm_router().route(LLMCapability.HIGH_ACCURACY_REASONING)
    print(
        "Meeting processing LLM route:",
        {
            "conversationId": event.conversationId,
            "provider": getattr(provider, "name", None),
            "model": model,
            "krutrimConfigured": _provider_configured("krutrim"),
            "mistralConfigured": _provider_configured("mistral"),
        },
    )
    try:
        await asyncio.wait_for(
            ConversationProcessingWorkflow(repository).run(event.conversationId),
            timeout=settings.CONVERSATION_PROCESSING_TIMEOUT_SECONDS,
        )
        print("Meeting processing completed:", {"conversationId": event.conversationId, "provider": getattr(provider, "name", None), "model": model})
    except asyncio.TimeoutError as error:
        message = f"Conversation processing timed out after {settings.CONVERSATION_PROCESSING_TIMEOUT_SECONDS} seconds."
        print("Meeting processing timed out:", {"conversationId": event.conversationId, "error": message})
        await repository.mark_active_extraction_run_failed(event.conversationId, message)
        await repository.mark_conversation_failed(
            event.conversationId,
            message,
        )


def _provider_configured(name: str) -> bool:
    provider = get_llm_router().providers.get(name)
    return bool(provider) and getattr(provider, "configured", True) is not False


def build_stt_consumer() -> RedisStreamConsumer:
    return RedisStreamConsumer(
        stream=settings.REDIS_STT_STREAM,
        group=settings.REDIS_STT_GROUP,
        consumer_name=f"stt-{uuid4().hex[:8]}",
        handler=handle_stt_event,
        concurrency=settings.STT_WORKER_CONCURRENCY or min(settings.WORKER_CONCURRENCY, settings.SARVAM_MAX_CONCURRENCY),
        on_dead_letter=handle_stt_dead_letter,
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
    repository = ConversationRepository(get_database())
    scanner = ConversationInactivityScanner(repository)
    interval = max(30, settings.CONVERSATION_INACTIVITY_TIMEOUT_SECONDS // 3)
    while True:
        try:
            await scanner.scan_once()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            print(f"Inactivity scanner failed: {error}", flush=True)
        await asyncio.sleep(interval)


async def run_retry_relay() -> None:
    import time

    producer = RedisStreamProducer()
    while True:
        try:
            entries = await redis_client.xrange(
                settings.REDIS_RETRY_STREAM, min="-", max="+", count=50
            )
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
        except asyncio.CancelledError:
            raise
        except (RedisTimeoutError, ConnectionError, RedisError) as error:
            print(f"Retry relay Redis error: {error}", flush=True)
            await asyncio.sleep(2)
            continue
        except Exception as error:
            print(f"Retry relay failed: {error}", flush=True)
            await asyncio.sleep(2)
            continue
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
