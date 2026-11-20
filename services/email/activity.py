"""Queue a dashboard or meeting email after tasks and notes are saved.

The conversation pipeline only publishes a small Redis job. Sending stays on
the Node mail worker so a slow SMTP call cannot stall transcription.
"""

from __future__ import annotations

from apps.api_gateway.config.setting import settings
from services.queue.shared_redis import get_shared_redis_client, normalize_redis_url
from services.queue.streams import EventEnvelope, RedisStreamProducer

EMAIL_EVENT_TYPE = "email.activity.ready"
QUEUED_TTL_SECONDS = 14 * 24 * 60 * 60


def activity_kind(source_type: str | None) -> str:
    if str(source_type or "").strip() == "meeting_extension":
        return "meeting"
    return "listening"


def activity_email_event_id(kind: str, conversation_id: str, processing_version: str) -> str:
    return f"email:{kind}:{conversation_id}:{processing_version}"


def activity_email_redis_url(reminder_url: str | None = None, conversation_url: str | None = None) -> str:
    """The Node mail worker reads REMINDER_REDIS_URL first, then REDIS_URL."""
    reminder = settings.REMINDER_REDIS_URL if reminder_url is None else reminder_url
    conversation = settings.REDIS_URL if conversation_url is None else conversation_url
    return str(reminder or "").strip() or str(conversation or "").strip()


def get_activity_email_redis():
    url = activity_email_redis_url()
    if not url:
        raise RuntimeError("Activity email Redis is not configured")
    return get_shared_redis_client(
        normalize_redis_url(url),
        max_connections=settings.MINDMAP_REDIS_MAX_CONNECTIONS,
        label="email",
    )


def should_enqueue_activity_email(kind: str, task_count: int, note_count: int) -> bool:
    """Listening mail needs dashboard cards. A finished meeting always queues."""
    if kind == "meeting":
        return True
    return task_count > 0 or note_count > 0


async def enqueue_activity_email(
    *,
    user_id: str,
    space_id: str,
    conversation_id: str,
    source_type: str | None,
    processing_version: int | str,
    task_count: int,
    note_count: int,
) -> None:
    kind = activity_kind(source_type)
    if not should_enqueue_activity_email(kind, task_count, note_count):
        return

    version = str(processing_version or 1)
    event_id = activity_email_event_id(kind, conversation_id, version)
    claim_key = f"email:queued:{event_id}"
    redis = None
    try:
        redis = get_activity_email_redis()
        claimed = await redis.set(claim_key, "1", nx=True, ex=QUEUED_TTL_SECONDS)
        if not claimed:
            return
        await RedisStreamProducer(redis, force_direct=True).publish(
            settings.REDIS_EMAIL_STREAM,
            EventEnvelope(
                eventId=event_id,
                eventType=EMAIL_EVENT_TYPE,
                correlationId=conversation_id,
                userId=str(user_id),
                spaceId=str(space_id or ""),
                conversationId=conversation_id,
                payload={
                    "kind": kind,
                    "conversationId": conversation_id,
                    "processingVersion": version,
                    "userId": str(user_id),
                    "spaceId": str(space_id or ""),
                    "taskCount": int(task_count),
                    "noteCount": int(note_count),
                },
            ),
        )
    except Exception as error:
        if redis is not None:
            try:
                await redis.delete(claim_key)
            except Exception:
                pass
        print(
            "activity email was not queued",
            {
                "conversationId": conversation_id,
                "kind": kind,
                "error": type(error).__name__,
            },
            flush=True,
        )
