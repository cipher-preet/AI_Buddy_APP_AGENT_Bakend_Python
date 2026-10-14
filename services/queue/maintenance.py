from __future__ import annotations

from typing import Any

from apps.api_gateway.config.setting import settings


STUCK_IDLE_MS = 15 * 60 * 1000
STUCK_DELIVERIES = 8
DEFAULT_STREAM_MAXLEN = 200


def should_reap_pending(*, idle_ms: int, deliveries: int, idle_limit_ms: int = STUCK_IDLE_MS, max_deliveries: int = STUCK_DELIVERIES) -> bool:
    """Drop jobs that loop forever instead of retrying them forever."""
    return int(idle_ms) >= idle_limit_ms or int(deliveries) >= max_deliveries


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


async def reap_stuck_stream(
    redis,
    stream: str,
    group: str,
    *,
    maxlen: int | None = DEFAULT_STREAM_MAXLEN,
    idle_limit_ms: int = STUCK_IDLE_MS,
) -> dict[str, int]:
    reaped = 0
    trimmed = 0
    try:
        pending = await redis.xpending_range(stream, group, min="-", max="+", count=50)
    except Exception:
        pending = []
    for item in pending or []:
        if isinstance(item, dict):
            message_id = str(item.get("message_id") or item.get("name") or "")
            idle_ms = _as_int(item.get("time_since_delivered") or item.get("idle"))
            deliveries = _as_int(item.get("times_delivered") or item.get("times"))
        else:
            message_id = str(item[0] if item else "")
            idle_ms = _as_int(item[2] if len(item) > 2 else 0)
            deliveries = _as_int(item[3] if len(item) > 3 else 0)
        if not message_id or not should_reap_pending(idle_ms=idle_ms, deliveries=deliveries, idle_limit_ms=idle_limit_ms):
            continue
        try:
            await redis.xack(stream, group, message_id)
            await redis.xdel(stream, message_id)
            reaped += 1
        except Exception:
            continue
    if maxlen and maxlen > 0:
        try:
            trimmed = int(await redis.xtrim(stream, maxlen=maxlen, approximate=True) or 0)
        except Exception:
            trimmed = 0
    if reaped or trimmed:
        print(
            "Redis reaped stuck jobs:",
            {"stream": stream, "group": group, "reaped": reaped, "trimmed": trimmed},
            flush=True,
        )
    return {"reaped": reaped, "trimmed": trimmed}


def _conversation_targets() -> list[tuple[str, str, int]]:
    maxlen = DEFAULT_STREAM_MAXLEN
    return [
        (settings.REDIS_AUDIO_STREAM, settings.REDIS_AUDIO_GROUP, maxlen),
        (settings.REDIS_STT_STREAM, settings.REDIS_STT_GROUP, maxlen),
        (settings.REDIS_TRANSCRIPT_READY_STREAM, settings.REDIS_TRANSCRIPT_GROUP, maxlen),
        (settings.REDIS_WINDOW_EXTRACTION_STREAM, settings.REDIS_WINDOW_EXTRACTION_GROUP, maxlen),
        (settings.REDIS_FINALIZATION_STREAM, settings.REDIS_FINALIZATION_GROUP, maxlen),
        (settings.REDIS_PROCESSING_STREAM, settings.REDIS_PROCESSING_GROUP, maxlen),
        (settings.REDIS_DAILY_BRIEFING_STREAM, settings.REDIS_DAILY_BRIEFING_GROUP, maxlen),
        (settings.REDIS_MEETING_MERGE_STREAM, settings.REDIS_MEETING_MERGE_GROUP, maxlen),
        (settings.REDIS_SCHEDULE_EXTRACTION_STREAM, settings.REDIS_SCHEDULE_EXTRACTION_GROUP, maxlen),
        (settings.REDIS_MINDMAP_STREAM, settings.REDIS_MINDMAP_GROUP, settings.REDIS_MINDMAP_STREAM_MAXLEN),
        (settings.REDIS_DOCUMENT_STREAM, settings.REDIS_DOCUMENT_GROUP, settings.REDIS_DOCUMENT_STREAM_MAXLEN),
        (settings.REDIS_RETRY_STREAM, "", maxlen),
        (settings.REDIS_DEAD_LETTER_STREAM, "", settings.REDIS_CLOUD_DEAD_LETTER_MAXLEN),
    ]


async def maintain_conversation_redis(redis) -> dict[str, int]:
    totals = {"reaped": 0, "trimmed": 0}
    for stream, group, maxlen in _conversation_targets():
        if group:
            result = await reap_stuck_stream(redis, stream, group, maxlen=maxlen)
        else:
            result = {"reaped": 0, "trimmed": 0}
            try:
                result["trimmed"] = int(await redis.xtrim(stream, maxlen=maxlen or DEFAULT_STREAM_MAXLEN, approximate=True) or 0)
            except Exception:
                pass
        totals["reaped"] += result["reaped"]
        totals["trimmed"] += result["trimmed"]
    return totals


async def maintain_goal_redis(redis) -> dict[str, int]:
    totals = {"reaped": 0, "trimmed": 0}
    for stream, group in (
        (settings.REDIS_GOAL_INTERACTIVE_STREAM, settings.REDIS_GOAL_INTERACTIVE_GROUP),
        (settings.REDIS_GOAL_JOBS_STREAM, settings.REDIS_GOAL_JOBS_GROUP),
    ):
        result = await reap_stuck_stream(
            redis,
            stream,
            group,
            maxlen=settings.REDIS_GOAL_STREAM_MAXLEN,
        )
        totals["reaped"] += result["reaped"]
        totals["trimmed"] += result["trimmed"]
    return totals


async def run_redis_maintenance_once() -> None:
    from services.goals.redis_client import get_goal_redis_client
    from services.queue.redis_queue import get_conversation_redis_client

    conversation = await maintain_conversation_redis(get_conversation_redis_client())
    goals = await maintain_goal_redis(get_goal_redis_client())
    if conversation["reaped"] or conversation["trimmed"] or goals["reaped"] or goals["trimmed"]:
        print(
            "Redis maintenance:",
            {"conversation": conversation, "goals": goals},
            flush=True,
        )


async def run_redis_maintenance(interval_seconds: float = 600.0) -> None:
    import asyncio

    await run_redis_maintenance_once()
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            await run_redis_maintenance_once()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            print("Redis maintenance failed:", type(error).__name__, flush=True)
