from __future__ import annotations

import redis.asyncio as redis

from apps.api_gateway.config.setting import settings
from services.reminders.redis_client import (
    describe_redis_url,
    format_redis_target,
    get_reminder_redis_client,
    get_reminder_redis_url,
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


def _urls_match(left: str, right: str) -> bool:
    return left.strip().rstrip("/") == right.strip().rstrip("/")


_mindmap_redis_client: redis.Redis | None = None
_mindmap_redis_owns_client = False


def get_mindmap_redis_client() -> redis.Redis:
    """Small pooled client for Redis Cloud Essentials (shared with reminders when URLs match)."""
    global _mindmap_redis_client, _mindmap_redis_owns_client
    if _mindmap_redis_client is not None:
        return _mindmap_redis_client

    url = get_mindmap_redis_url()
    # Same cloud DB as reminders → reuse one pool so we do not burn Essentials connection quota.
    try:
        reminder_url = get_reminder_redis_url()
    except Exception:
        reminder_url = ""
    if reminder_url and _urls_match(url, reminder_url):
        _mindmap_redis_client = get_reminder_redis_client()
        _mindmap_redis_owns_client = False
        return _mindmap_redis_client

    max_connections = max(2, int(settings.MINDMAP_REDIS_MAX_CONNECTIONS))
    _mindmap_redis_client = redis.from_url(
        url,
        decode_responses=True,
        max_connections=max_connections,
        socket_connect_timeout=10,
        socket_timeout=30,
        socket_keepalive=True,
        health_check_interval=30,
        retry_on_timeout=True,
    )
    _mindmap_redis_owns_client = True
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
            f"max_connections={settings.MINDMAP_REDIS_MAX_CONNECTIONS}",
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
    global _mindmap_redis_client, _mindmap_redis_owns_client
    if _mindmap_redis_client is not None and _mindmap_redis_owns_client:
        await _mindmap_redis_client.aclose()
    _mindmap_redis_client = None
    _mindmap_redis_owns_client = False
