from __future__ import annotations

from uuid import uuid4

from apps.api_gateway.config.setting import settings
from services.conversation.finalization import ConversationFinalizationCoordinator
from services.conversation.inactivity import ConversationInactivityScanner
from services.conversation.repository import ConversationRepository
from services.conversation.workflow import ConversationProcessingWorkflow
from services.db.mongo import get_database
from services.queue.streams import EventEnvelope, RedisStreamConsumer
from services.queue.streams import RedisStreamProducer
from services.queue.redis_queue import redis_client
from services.speech.providers.sarvam_provider import sarvam_transcribe_from_path


async def handle_stt_event(event: EventEnvelope) -> None:
    repository = ConversationRepository(get_database())
    payload = event.payload
    conversation_id = payload["conversationId"]
    sequence_number = int(payload["sequenceNumber"])

    try:
        result = await sarvam_transcribe_from_path(
            file_path=payload["filePath"],
            filename=payload.get("filename") or f"{payload.get('chunkId')}.audio",
            content_type=payload.get("contentType") or "audio/wav",
        )
        await repository.complete_transcript_chunk(
            conversation_id=conversation_id,
            sequence_number=sequence_number,
            raw_text=result["transcript"],
            language_code=result.get("language_code"),
            request_id=result.get("request_id"),
            provider="sarvam",
        )
        conversation = await repository.get_conversation(conversation_id)
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
        await repository.fail_transcript_chunk(conversation_id, sequence_number, str(error))
        raise


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


async def handle_finalization_event(event: EventEnvelope) -> None:
    repository = ConversationRepository(get_database())
    await ConversationFinalizationCoordinator(repository).finalize(event.conversationId)


async def handle_processing_event(event: EventEnvelope) -> None:
    repository = ConversationRepository(get_database())
    await ConversationProcessingWorkflow(repository).run(event.conversationId)


def build_stt_consumer() -> RedisStreamConsumer:
    return RedisStreamConsumer(
        stream=settings.REDIS_STT_STREAM,
        group=settings.REDIS_STT_GROUP,
        consumer_name=f"stt-{uuid4().hex[:8]}",
        handler=handle_stt_event,
    )


def build_audio_consumer() -> RedisStreamConsumer:
    return RedisStreamConsumer(
        stream=settings.REDIS_AUDIO_STREAM,
        group=settings.REDIS_AUDIO_GROUP,
        consumer_name=f"audio-{uuid4().hex[:8]}",
        handler=handle_audio_event,
    )


def build_finalization_consumer() -> RedisStreamConsumer:
    return RedisStreamConsumer(
        stream=settings.REDIS_FINALIZATION_STREAM,
        group=settings.REDIS_FINALIZATION_GROUP,
        consumer_name=f"finalization-{uuid4().hex[:8]}",
        handler=handle_finalization_event,
    )


def build_processing_consumer() -> RedisStreamConsumer:
    return RedisStreamConsumer(
        stream=settings.REDIS_PROCESSING_STREAM,
        group=settings.REDIS_PROCESSING_GROUP,
        consumer_name=f"processing-{uuid4().hex[:8]}",
        handler=handle_processing_event,
        concurrency=1,
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
                await producer.publish(target_stream, EventEnvelope.model_validate_json(raw_event))
            await redis_client.xdel(settings.REDIS_RETRY_STREAM, message_id)
        await asyncio.sleep(1)
