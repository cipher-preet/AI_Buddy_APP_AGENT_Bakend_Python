import json
import asyncio
import time
import uuid

import redis.asyncio as redis
from redis.exceptions import TimeoutError, ConnectionError, RedisError

from apps.api_gateway.config.setting import settings

SPEECH_QUEUE = "speech_transcribe_queue"
COMPLETED_SPEECH_QUEUE = "completed_speech_queue"
ANALYSIS_QUEUE = "analysis_queue"
FAILED_ANALYSIS_QUEUE = "failed_analysis_queue"
TRANSCRIPT_SESSION_PREFIX = "transcript_session"
TRANSCRIPT_ANALYSIS_QUEUED_PREFIX = "transcript_analysis_queued"


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


def _session_key(user_id: str, space_id: str) -> str:
    return f"{TRANSCRIPT_SESSION_PREFIX}:{user_id}:{space_id}"


def _analysis_queued_key(user_id: str, space_id: str) -> str:
    return f"{TRANSCRIPT_ANALYSIS_QUEUED_PREFIX}:{user_id}:{space_id}"


async def mark_transcript_session_started(user_id: str, space_id: str) -> dict:
    now = int(time.time())
    session_id = str(uuid.uuid4())
    key = _session_key(user_id, space_id)
    await redis_client.hset(
        key,
        mapping={
            "session_id": session_id,
            "user_id": user_id,
            "space_id": space_id,
            "status": "active",
            "pending_jobs": 0,
            "started_at": now,
            "ended_at": "",
            "last_activity_at": now,
        },
    )
    await clear_transcript_analysis_queued(user_id, space_id)
    return await redis_client.hgetall(key)


async def register_transcript_job_queued(user_id: str, space_id: str) -> dict:
    now = int(time.time())
    key = _session_key(user_id, space_id)
    existing = await redis_client.hgetall(key)
    session_id = existing.get("session_id") or str(uuid.uuid4())
    await redis_client.hset(
        key,
        mapping={
            "session_id": session_id,
            "user_id": user_id,
            "space_id": space_id,
            "last_activity_at": now,
        },
    )
    await redis_client.hsetnx(key, "status", "active")
    await redis_client.hincrby(key, "pending_jobs", 1)
    return await redis_client.hgetall(key)


async def mark_transcript_job_vectorized(
    user_id: str,
    space_id: str,
    session_id: str | None = None,
) -> dict:
    now = int(time.time())
    key = _session_key(user_id, space_id)
    existing = await redis_client.hgetall(key)
    if session_id and existing.get("session_id") and session_id != existing.get("session_id"):
        return existing
    await redis_client.hset(
        key,
        mapping={
            "user_id": user_id,
            "space_id": space_id,
            "last_activity_at": now,
        },
    )
    pending = await redis_client.hincrby(key, "pending_jobs", -1)
    if pending < 0:
        await redis_client.hset(key, "pending_jobs", 0)
    return await redis_client.hgetall(key)


async def mark_transcript_session_ended(user_id: str, space_id: str) -> dict:
    now = int(time.time())
    key = _session_key(user_id, space_id)
    existing = await redis_client.hgetall(key)
    session_id = existing.get("session_id") or str(uuid.uuid4())
    await redis_client.hset(
        key,
        mapping={
            "session_id": session_id,
            "user_id": user_id,
            "space_id": space_id,
            "status": "ended",
            "ended_at": now,
            "last_activity_at": now,
        },
    )
    await redis_client.hsetnx(key, "pending_jobs", 0)
    return await redis_client.hgetall(key)


async def maybe_queue_transcript_analysis(
    *,
    user_id: str,
    space_id: str,
    request_id: str | None = None,
    reason: str,
    force: bool = False,
) -> bool:
    queued_key = _analysis_queued_key(user_id, space_id)
    if force:
        await redis_client.set(queued_key, "1", ex=60 * 60)
    else:
        queued = await redis_client.set(
            queued_key,
            "1",
            nx=True,
            ex=60 * 60,
        )
        if not queued:
            return False

    await push_analysis_job(
        {
            "job_type": "analyze_transcript_window",
            "user_id": user_id,
            "space_id": space_id,
            "request_id": request_id,
            "trigger_reason": reason,
            "attempt": 1,
        }
    )
    return True


async def clear_transcript_analysis_queued(user_id: str, space_id: str) -> None:
    await redis_client.delete(_analysis_queued_key(user_id, space_id))


async def maybe_queue_transcript_analysis_for_session(
    *,
    user_id: str,
    space_id: str,
    request_id: str | None = None,
    reason: str,
    session: dict | None = None,
    force: bool = False,
) -> bool:
    data = session or await redis_client.hgetall(_session_key(user_id, space_id))
    pending_jobs = int(data.get("pending_jobs") or 0)
    if data.get("status") != "ended" or pending_jobs > 0:
        return False
    return await maybe_queue_transcript_analysis(
        user_id=user_id,
        space_id=space_id,
        request_id=request_id,
        reason=reason,
        force=force,
    )


async def iter_idle_transcript_sessions(*, idle_seconds: int) -> list[dict]:
    cutoff = int(time.time()) - idle_seconds
    sessions: list[dict] = []
    async for key in redis_client.scan_iter(f"{TRANSCRIPT_SESSION_PREFIX}:*"):
        data = await redis_client.hgetall(key)
        if data.get("status") != "active":
            continue
        if int(data.get("pending_jobs") or 0) > 0:
            continue
        last_activity_at = int(data.get("last_activity_at") or 0)
        if last_activity_at and last_activity_at <= cutoff:
            sessions.append(data)
    return sessions


async def push_failed_analysis_job(job: dict):
    await redis_client.lpush(FAILED_ANALYSIS_QUEUE, json.dumps(job))


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
