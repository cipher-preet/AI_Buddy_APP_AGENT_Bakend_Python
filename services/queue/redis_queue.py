import json
import asyncio

import redis.asyncio as redis
from redis.exceptions import TimeoutError, ConnectionError, RedisError

from apps.api_gateway.config.setting import settings

SPEECH_QUEUE = "speech_transcribe_queue"
COMPLETED_SPEECH_QUEUE = "completed_speech_queue"
ANALYSIS_QUEUE = "analysis_queue"


redis_client = redis.from_url(
    settings.REDIS_URL,
    decode_responses=True,
    socket_connect_timeout=10,
    socket_timeout=30,
    health_check_interval=30,
)


async def test_redis_connection():
    try:
        pong = await redis_client.ping()
        print("Redis connected:", pong)
    except Exception as error:
        print("Redis connection failed:", str(error))


async def push_speech_job(job: dict):
    await redis_client.lpush(SPEECH_QUEUE, json.dumps(job))


async def push_analysis_job(job: dict):
    await redis_client.lpush(ANALYSIS_QUEUE, json.dumps(job))


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


async def save_job_result(job_id: str, result: dict):
    await redis_client.hset(
        f"speech_job:{job_id}",
        mapping={
            "status": "completed",
            "result": json.dumps(result),
        },
    )


async def mark_job_processing(job_id: str):
    await redis_client.hset(
        f"speech_job:{job_id}",
        mapping={
            "status": "processing",
        },
    )


async def mark_job_failed(job_id: str, error: str):
    await redis_client.hset(
        f"speech_job:{job_id}",
        mapping={
            "status": "failed",
            "error": error,
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


async def pop_analysis_job():
    try:
        data = await redis_client.brpop(ANALYSIS_QUEUE, timeout=5)

        if not data:
            return None

        _, job = data
        return json.loads(job)

    except TimeoutError:
        return None

    except ConnectionError as error:
        print("Redis analysis queue connection error:", str(error))
        await asyncio.sleep(2)
        return None

    except RedisError as error:
        print("Redis analysis queue error:", str(error))
        await asyncio.sleep(2)
        return None


async def delete_speech_job(job_id: str):
    await redis_client.delete(f"speech_job:{job_id}")
