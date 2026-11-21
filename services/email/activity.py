"""Queue a dashboard or meeting email after tasks and notes are saved.

The conversation pipeline only publishes a small Redis job. Sending stays on
the Node mail worker so a slow SMTP call cannot stall transcription.
"""

from __future__ import annotations

from apps.api_gateway.config.setting import settings
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
    """Mail uses REMINDER_REDIS_URL only. The worker's REDIS_URL is private and Node cannot read it."""
    del conversation_url
    reminder = settings.REMINDER_REDIS_URL if reminder_url is None else reminder_url
    return str(reminder or "").strip()


def get_activity_email_redis():
    from services.reminders.redis_client import get_reminder_redis_client

    return get_reminder_redis_client()


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
        from services.reminders.redis_client import describe_redis_url, get_reminder_redis_url

        target = describe_redis_url(get_reminder_redis_url())
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
        print(
            "Activity email queued on reminder Redis",
            {
                "conversationId": conversation_id,
                "kind": kind,
                "host": target["host"],
                "port": target["port"],
            },
            flush=True,
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
