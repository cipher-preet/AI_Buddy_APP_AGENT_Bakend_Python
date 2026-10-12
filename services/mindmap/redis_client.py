from __future__ import annotations

import redis.asyncio as redis

from apps.api_gateway.config.setting import settings
from services.queue.shared_redis import get_shared_redis_client, normalize_redis_url, redis_pool_limit
from services.reminders.redis_client import (
    describe_redis_url,
    format_redis_target,
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


def get_mindmap_redis_client() -> redis.Redis:
    """Shared pool with conversation/reminder when the URL matches."""
    return get_shared_redis_client(
        normalize_redis_url(get_mindmap_redis_url()),
        max_connections=settings.MINDMAP_REDIS_MAX_CONNECTIONS,
        label="mindmap",
    )


async def test_mindmap_redis_connection() -> bool:
    from services.queue.shared_redis import normalize_redis_url, prepare_shared_redis

    url = get_mindmap_redis_url()
    target = describe_redis_url(url)
    client = get_mindmap_redis_client()
    # Same cloud DB as conversation → do not open a second pool or extra ping storm.
    same_as_conversation = normalize_redis_url(url) == normalize_redis_url(settings.REDIS_URL)
    try:
        if same_as_conversation:
            print(
                "Mindmap Redis connected:",
                format_redis_target(target),
                f"max_connections={redis_pool_limit(url, settings.MINDMAP_REDIS_MAX_CONNECTIONS)}",
                "shared_with=conversation",
                flush=True,
            )
            return True
        await prepare_shared_redis(client)
        print(
            "Mindmap Redis connected:",
            format_redis_target(target),
            f"max_connections={redis_pool_limit(url, settings.MINDMAP_REDIS_MAX_CONNECTIONS)}",
            flush=True,
        )
        return True
    except Exception as error:
        print(
            f"Mindmap Redis connection failed: {redact_redis_secrets(str(error))}",
            flush=True,
        )
        raise


async def close_mindmap_redis_client() -> None:
    # Shared registry owns the client. Closing happens with process shutdown.
    return
