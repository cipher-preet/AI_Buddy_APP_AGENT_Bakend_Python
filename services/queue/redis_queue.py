import asyncio
import json
import re
from urllib.parse import urlparse, urlunparse

import redis.asyncio as redis
from redis.exceptions import ConnectionError, RedisError, TimeoutError

from apps.api_gateway.config.setting import settings

SPEECH_QUEUE = "speech_transcribe_queue"
COMPLETED_SPEECH_QUEUE = "completed_speech_queue"


def prefer_ipv4_loopback_url(url: str) -> str:
    """Rewrite localhost to 127.0.0.1 so Windows does not try IPv6 first."""
    parsed = urlparse(url)
    if (parsed.hostname or "").lower() != "localhost":
        return url
    userinfo = ""
    if parsed.password is not None:
        userinfo = f"{parsed.username or ''}:{parsed.password}@"
    elif parsed.username:
        userinfo = f"{parsed.username}@"
    hostport = "127.0.0.1"
    if parsed.port is not None:
        hostport = f"{hostport}:{parsed.port}"
    return urlunparse(parsed._replace(netloc=f"{userinfo}{hostport}"))


def _redact_redis_secrets(message: str) -> str:
    return re.sub(r"redis(?:s)?://[^\s\"']+", "redis://<redacted>", message, flags=re.I)


redis_client = redis.from_url(
    prefer_ipv4_loopback_url(settings.REDIS_URL),
    decode_responses=True,
    socket_connect_timeout=10,
    socket_timeout=30,
    health_check_interval=30,
)


async def test_redis_connection(*, attempts: int = 5, delay_seconds: float = 2.0) -> bool:
    last_error: Exception | None = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            pong = await redis_client.ping()
            print("Conversation Redis connected:", pong)
            return bool(pong)
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
        data = await redis_client.brpop(SPEECH_QUEUE, timeout=5)
        if not data:
            return None

        _, job = data
        return json.loads(job)
    except TimeoutError:
        return None
    except ConnectionError as error:
        print("Redis connection error:", str(error))
        await asyncio.sleep(2)
        return None
    except RedisError as error:
        print("Redis error:", str(error))
        await asyncio.sleep(2)
        return None


async def mark_job_processing(job_id: str):
    await redis_client.hset(
        f"speech_job:{job_id}",
        mapping={"status": "processing"},
    )


async def mark_job_failed(job_id: str, error: str):
    await redis_client.hset(
        f"speech_job:{job_id}",
        mapping={
            "status": "failed",
            "error": error,
        },
    )


async def save_job_result(job_id: str, result: dict):
    await redis_client.hset(
        f"speech_job:{job_id}",
        mapping={
            "status": "completed",
            "result": json.dumps(result),
        },
    )


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
        data = await redis_client.brpop(COMPLETED_SPEECH_QUEUE, timeout=5)
        if not data:
            return None

        _, job_id = data
        return job_id
    except TimeoutError:
        return None
    except RedisError as error:
        print("Redis completed queue error:", str(error))
        await asyncio.sleep(2)
        return None


async def delete_speech_job(job_id: str):
    await redis_client.delete(f"speech_job:{job_id}")
