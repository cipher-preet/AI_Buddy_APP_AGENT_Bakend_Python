import asyncio
import json
import re

import redis.asyncio as redis
from redis.exceptions import ConnectionError, RedisError, TimeoutError

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
SPEECH_JOB_TTL_SECONDS = 86400


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


async def push_speech_job(job: dict):
    await redis_client.lpush(SPEECH_QUEUE, json.dumps(job))


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


async def push_completed_speech_job(job_id: str):
    await redis_client.lpush(COMPLETED_SPEECH_QUEUE, job_id)


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
