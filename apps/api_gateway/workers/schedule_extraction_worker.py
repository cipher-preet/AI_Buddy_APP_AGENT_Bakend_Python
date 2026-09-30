from __future__ import annotations

from uuid import uuid4

from apps.api_gateway.config.setting import settings
from services.conversation.repository import ConversationRepository
from services.db.mongo import get_database
from services.observability.diagnostics import active_jobs
from services.queue.streams import EventEnvelope, NonRetryableQueueError, RedisStreamConsumer, RedisStreamProducer
from services.schedule_extraction.job import ScheduleExtractionJob
from services.schedule_extraction.log import extraction_log

EVENT_TYPE = "conversation.schedule_extraction.requested"


async def handle_schedule_extraction_event(event: EventEnvelope) -> None:
    if not event.conversationId:
        raise NonRetryableQueueError("schedule extraction event is missing conversationId")
    database = get_database()
    job = ScheduleExtractionJob(ConversationRepository(database), database)
    with active_jobs.track("schedule_extraction"):
        await job.run(
            event.conversationId,
            event.eventId,
            final_attempt=event.attempt >= settings.SCHEDULE_EXTRACTION_MAX_RETRIES,
        )


async def request_schedule_extraction(conversation_id: str, user_id: str, space_id: str, source_event_id: str) -> None:
    """Best effort: a publish failure must never fail the meeting processing job."""
    if not settings.SCHEDULE_EXTRACTION_ENABLED:
        return
    try:
        await RedisStreamProducer().publish(
            settings.REDIS_SCHEDULE_EXTRACTION_STREAM,
            EventEnvelope(
                eventType=EVENT_TYPE,
                correlationId=conversation_id,
                userId=user_id,
                spaceId=space_id,
                conversationId=conversation_id,
                payload={"sourceEventId": source_event_id},
            ),
        )
    except Exception as error:
        extraction_log(
            "schedule_extraction_publish_failed",
            conversationId=conversation_id,
            error=f"{type(error).__name__}: {str(error)[:200]}",
        )


def build_schedule_extraction_consumer() -> RedisStreamConsumer | None:
    if not settings.SCHEDULE_EXTRACTION_ENABLED:
        return None
    return RedisStreamConsumer(
        stream=settings.REDIS_SCHEDULE_EXTRACTION_STREAM,
        group=settings.REDIS_SCHEDULE_EXTRACTION_GROUP,
        consumer_name=f"schedule-extraction-{uuid4().hex[:8]}",
        handler=handle_schedule_extraction_event,
        concurrency=settings.SCHEDULE_EXTRACTION_CONCURRENCY,
        max_retries=settings.SCHEDULE_EXTRACTION_MAX_RETRIES,
    )
