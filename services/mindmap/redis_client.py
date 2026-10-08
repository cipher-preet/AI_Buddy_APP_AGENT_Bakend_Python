from __future__ import annotations

import redis.asyncio as redis

from apps.api_gateway.config.setting import settings
from services.reminders.redis_client import (
    describe_redis_url,
    format_redis_target,
    redact_redis_secrets,
)


class MindmapRedisConfigError(RuntimeError):
    pass


def get_mindmap_redis_url() -> str:
    """Prefer MINDMAP_REDIS_URL; otherwise reuse the cloud reminder Redis."""
    explicit = (settings.MINDMAP_REDIS_URL or "").strip()
    if explicit:
        return explicit
    fallback = (settings.REMINDER_REDIS_URL or "").strip()
    if fallback:
        return fallback
    raise MindmapRedisConfigError(
        "MINDMAP_REDIS_URL is not configured (and REMINDER_REDIS_URL is empty). "
        "Set MINDMAP_REDIS_URL or REMINDER_REDIS_URL to your cloud Redis URL."
    )


_mindmap_redis_client: redis.Redis | None = None


def get_mindmap_redis_client() -> redis.Redis:
    global _mindmap_redis_client
    if _mindmap_redis_client is None:
        _mindmap_redis_client = redis.from_url(
            get_mindmap_redis_url(),
            decode_responses=True,
            socket_connect_timeout=10,
            socket_timeout=30,
            health_check_interval=30,
        )
    return _mindmap_redis_client


async def test_mindmap_redis_connection() -> bool:
    url = get_mindmap_redis_url()
    target = describe_redis_url(url)
    client = get_mindmap_redis_client()
    try:
        pong = await client.ping()
        print(
            "Mindmap Redis connected:",
            format_redis_target(target),
            flush=True,
        )
        return bool(pong)
    except Exception as error:
        print(
            f"Mindmap Redis connection failed: {redact_redis_secrets(str(error))}",
            flush=True,
        )
        raise


async def close_mindmap_redis_client() -> None:
    global _mindmap_redis_client
    if _mindmap_redis_client is not None:
        await _mindmap_redis_client.aclose()
        _mindmap_redis_client = None
