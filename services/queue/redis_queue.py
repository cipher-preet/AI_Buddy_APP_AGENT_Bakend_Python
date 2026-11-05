import asyncio
import json
import re

import redis.asyncio as redis
from redis.exceptions import ConnectionError, RedisError, ResponseError, TimeoutError

from apps.api_gateway.config.setting import settings
from services.queue.shared_redis import (
    capacity_backoff_seconds,
    get_shared_redis_client,
    is_redis_capacity_error,
    normalize_redis_url,
    recover_redis_capacity,
)

SPEECH_QUEUE = "speech_transcribe_queue"
COMPLETED_SPEECH_QUEUE = "completed_speech_queue"
SPEECH_MIGRATION_PROCESSING = "speech_transcribe_queue:migrating"
SPEECH_MIGRATION_FAILED = "speech_transcribe_queue:migration_failed"
COMPLETED_MIGRATION_PROCESSING = "completed_speech_queue:migrating"
COMPLETED_MIGRATION_FAILED = "completed_speech_queue:migration_failed"
SPEECH_JOB_TTL_SECONDS = 86400
SPEECH_EVENT_TYPE = "speech.transcribe.requested"
VECTOR_EVENT_TYPE = "vector.index.requested"
EXPORT_LOCK_KEY = "buddy:speech-vector:export-lock"
VECTOR_DONE_PREFIX = "speech_vector_done:"

_PUBLISH_GUARD_LUA = """
if redis.call('EXISTS', KEYS[1]) == 1 then
  return redis.error_reply('speech-vector-export-locked')
end
if ARGV[1] == 'list' then
  redis.call('LPUSH', KEYS[2], ARGV[2])
  return 'list'
end
return redis.call('XADD', KEYS[2], '*', 'event', ARGV[2])
"""


class SpeechVectorCutoverLocked(RuntimeError):
    """Publishing is paused so rollback cannot race a new queue write."""


def speech_vector_transport() -> str:
    """Authoritative transport. Default list leaves production behavior unchanged."""
    value = str(settings.SPEECH_VECTOR_QUEUE_TRANSPORT or "list").strip().lower()
    if value not in {"list", "stream"}:
        return "list"
    return value


def speech_queue_target_name() -> str:
    if speech_vector_transport() == "stream":
        return settings.REDIS_SPEECH_STREAM
    return SPEECH_QUEUE


def vector_queue_target_name() -> str:
    if speech_vector_transport() == "stream":
        return settings.REDIS_VECTOR_STREAM
    return COMPLETED_SPEECH_QUEUE


def speech_job_event(job: dict):
    """Stable speech envelope. jobId is the logical job, not a new id per retry."""
    from services.queue.streams import EventEnvelope

    job_id = str(job.get("job_id") or job.get("jobId") or "").strip()
    if not job_id:
        raise ValueError("Speech job payload is missing job_id")
    payload = dict(job)
    payload["jobId"] = job_id
    payload["job_id"] = job_id
    for key, value in list(payload.items()):
        if value is None:
            payload.pop(key)
    conversation = str(payload.get("conversation_id") or payload.get("conversationId") or job_id)
    return EventEnvelope(
        eventType=SPEECH_EVENT_TYPE,
        correlationId=str(payload.get("request_id") or payload.get("requestId") or conversation),
        userId=str(payload.get("user_id") or payload.get("userId") or "unknown"),
        spaceId=str(payload.get("space_id") or payload.get("spaceId") or ""),
        conversationId=conversation,
        payload=payload,
    )


def vector_job_event(job_id: str, job: dict | None = None):
    from services.queue.streams import EventEnvelope

    job_id = str(job_id or "").strip()
    if not job_id:
        raise ValueError("Completed speech job is missing job_id")
    source = job or {}
    return EventEnvelope(
        eventType=VECTOR_EVENT_TYPE,
        correlationId=job_id,
        userId=str(source.get("user_id") or source.get("userId") or "unknown"),
        spaceId=str(source.get("space_id") or source.get("spaceId") or ""),
        conversationId=str(source.get("conversation_id") or source.get("conversationId") or job_id),
        payload={"jobId": job_id, "job_id": job_id},
    )


def _redact_redis_secrets(message: str) -> str:
    return re.sub(r"redis(?:s)?://[^\s\"']+", "redis://<redacted>", message, flags=re.I)


def get_conversation_redis_client() -> redis.Redis:
    return get_shared_redis_client(
        normalize_redis_url(settings.REDIS_URL),
        max_connections=settings.MINDMAP_REDIS_MAX_CONNECTIONS,
        label="conversation",
    )


class _LazyRedis:
    """Avoid opening a Redis TCP client at import time."""

    def __getattr__(self, name: str):
        return getattr(get_conversation_redis_client(), name)

    def __repr__(self) -> str:
        return "LazyRedis(conversation)"


# Back-compat for existing imports. Resolves to the shared pool on first use.
redis_client = _LazyRedis()


async def test_redis_connection(*, attempts: int = 5, delay_seconds: float = 2.0) -> bool:
    from services.queue.shared_redis import prepare_shared_redis

    last_error: Exception | None = None
    client = get_conversation_redis_client()
    for attempt in range(1, max(1, attempts) + 1):
        try:
            await prepare_shared_redis(client)
            print("Conversation Redis connected: True")
            return True
        except Exception as error:
            last_error = error
            print(
                "Conversation Redis connection failed "
                f"(attempt {attempt}/{attempts}): {_redact_redis_secrets(str(error))}"
            )
            if attempt < attempts:
                await asyncio.sleep(delay_seconds)
    raise ConnectionError(
        "Conversation workers require Redis at REDIS_URL. "
        "Start local Redis, then retry: docker compose -f docker-compose.local.yml up -d"
    ) from last_error


def _redis_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


async def guarded_queue_write(client, key: str, mode: str, payload: str) -> str:
    """LPUSH or XADD only when a rollback export is not holding the cutover lock."""
    try:
        result = await client.eval(_PUBLISH_GUARD_LUA, 2, EXPORT_LOCK_KEY, key, mode, payload)
    except ResponseError as error:
        if "speech-vector-export-locked" in str(error):
            raise SpeechVectorCutoverLocked("speech and vector publishing is paused for rollback") from error
        raise
    return _redis_text(result)


async def publish_speech_job(job: dict, redis=None) -> str:
    """Publish one speech job to the authoritative transport. Returns the queue name."""
    client = redis or redis_client
    if speech_vector_transport() == "stream":
        from services.observability.diagnostics import diag_log
        from services.queue.job_lease import logical_job_key

        event = speech_job_event(job)
        message_id = await guarded_queue_write(
            client,
            settings.REDIS_SPEECH_STREAM,
            "stream",
            event.model_dump_json(),
        )
        diag_log(
            "speech_stream_published",
            job_type=event.eventType,
            logical_job_id=logical_job_key(event),
            queue_name=settings.REDIS_SPEECH_STREAM,
            message_id=message_id,
            job_id=str(job.get("job_id") or job.get("jobId") or ""),
        )
        return settings.REDIS_SPEECH_STREAM
    await guarded_queue_write(client, SPEECH_QUEUE, "list", json.dumps(job))
    return SPEECH_QUEUE


async def push_speech_job(job: dict) -> str:
    return await publish_speech_job(job)


async def pop_speech_job():
    try:
        data = await redis_client.brpop(SPEECH_QUEUE, timeout=1)
        if not data:
            return None

        _, job = data
        return json.loads(job)
    except TimeoutError:
        return None
    except ConnectionError as error:
        print("Redis connection error:", str(error))
        if is_redis_capacity_error(error):
            await recover_redis_capacity()
            await asyncio.sleep(capacity_backoff_seconds(3))
        else:
            await asyncio.sleep(2)
        return None
    except RedisError as error:
        print("Redis error:", str(error))
        if is_redis_capacity_error(error):
            await recover_redis_capacity()
            await asyncio.sleep(capacity_backoff_seconds(3))
        else:
            await asyncio.sleep(2)
        return None


async def expire_speech_job(job_id: str) -> None:
    await redis_client.expire(f"speech_job:{job_id}", SPEECH_JOB_TTL_SECONDS)


async def record_vector_attempt(job_id: str, attempts: int) -> None:
    await redis_client.hset(f"speech_job:{job_id}", mapping={"vector_attempts": attempts})
    await expire_speech_job(job_id)


async def mark_vector_stored(job_id: str) -> None:
    """Durable idempotency flag. A crash before XACK must not embed again."""
    await redis_client.hset(f"speech_job:{job_id}", mapping={"vector_status": "stored"})
    await expire_speech_job(job_id)
    # Survives deletion of the speech hash, which happens before XACK.
    await redis_client.set(
        f"{VECTOR_DONE_PREFIX}{job_id}",
        "1",
        ex=SPEECH_JOB_TTL_SECONDS,
    )


async def vector_stage_settled(job_id: str) -> bool:
    return bool(await redis_client.get(f"{VECTOR_DONE_PREFIX}{job_id}"))


async def mark_job_processing(job_id: str):
    await redis_client.hset(
        f"speech_job:{job_id}",
        mapping={"status": "processing"},
    )
    await expire_speech_job(job_id)


async def mark_job_failed(job_id: str, error: str):
    await redis_client.hset(
        f"speech_job:{job_id}",
        mapping={
            "status": "failed",
            "error": error,
        },
    )
    await expire_speech_job(job_id)


async def save_job_result(job_id: str, result: dict):
    await redis_client.hset(
        f"speech_job:{job_id}",
        mapping={
            "status": "completed",
            "result": json.dumps(result),
        },
    )
    await expire_speech_job(job_id)


async def get_job_result(job_id: str):
    data = await redis_client.hgetall(f"speech_job:{job_id}")
    if not data:
        return None

    if data.get("result"):
        data["result"] = json.loads(data["result"])

    return data


async def publish_completed_speech_job(job_id: str, redis=None, job: dict | None = None) -> str:
    """Publish the vector stage for one completed speech job."""
    client = redis or redis_client
    if speech_vector_transport() == "stream":
        from services.observability.diagnostics import diag_log
        from services.queue.job_lease import logical_job_key

        source = job
        if source is None:
            try:
                if client is redis_client:
                    source = await get_job_result(job_id)
                else:
                    source = await client.hgetall(f"speech_job:{job_id}")
            except Exception:
                source = None
        event = vector_job_event(job_id, source if isinstance(source, dict) else None)
        message_id = await guarded_queue_write(
            client,
            settings.REDIS_VECTOR_STREAM,
            "stream",
            event.model_dump_json(),
        )
        diag_log(
            "vector_stream_published",
            job_type=event.eventType,
            logical_job_id=logical_job_key(event),
            queue_name=settings.REDIS_VECTOR_STREAM,
            message_id=message_id,
            job_id=str(job_id),
        )
        return settings.REDIS_VECTOR_STREAM
    await guarded_queue_write(client, COMPLETED_SPEECH_QUEUE, "list", job_id)
    return COMPLETED_SPEECH_QUEUE


async def push_completed_speech_job(job_id: str) -> str:
    return await publish_completed_speech_job(job_id)


async def pop_completed_speech_job():
    try:
        data = await redis_client.brpop(COMPLETED_SPEECH_QUEUE, timeout=1)
        if not data:
            return None

        _, job_id = data
        return job_id
    except TimeoutError:
        return None
    except RedisError as error:
        print("Redis completed queue error:", str(error))
        if is_redis_capacity_error(error):
            await recover_redis_capacity()
            await asyncio.sleep(capacity_backoff_seconds(3))
        else:
            await asyncio.sleep(2)
        return None


async def delete_speech_job(job_id: str):
    await redis_client.delete(f"speech_job:{job_id}")
