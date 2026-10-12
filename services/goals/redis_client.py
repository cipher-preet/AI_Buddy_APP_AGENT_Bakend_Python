from __future__ import annotations

import redis.asyncio as redis

from apps.api_gateway.config.setting import settings
from services.queue.shared_redis import (
    get_shared_redis_client,
    normalize_redis_url,
    prepare_shared_redis,
    redis_pool_limit,
)
from services.reminders.redis_client import describe_redis_url, format_redis_target, redact_redis_secrets


class GoalRedisConfigError(RuntimeError):
    pass


def get_goal_redis_url() -> str:
    """Dedicated Redis for Achieve Goal only. Does not fall back to conversation Redis."""
    explicit = (settings.GOAL_REDIS_URL or "").strip()
    if explicit:
        return explicit
    raise GoalRedisConfigError(
        "GOAL_REDIS_URL is not configured. Set GOAL_REDIS_URL to the Achieve Goal Redis Cloud URL."
    )


def get_goal_redis_client() -> redis.Redis:
    return get_shared_redis_client(
        normalize_redis_url(get_goal_redis_url()),
        max_connections=settings.GOAL_REDIS_MAX_CONNECTIONS,
        label="goals",
    )


async def test_goal_redis_connection() -> bool:
    url = get_goal_redis_url()
    target = describe_redis_url(url)
    client = get_goal_redis_client()
    try:
        await prepare_shared_redis(client)
        print(
            "Goal Redis connected:",
            format_redis_target(target),
            f"max_connections={redis_pool_limit(url, settings.GOAL_REDIS_MAX_CONNECTIONS)}",
            flush=True,
        )
        return True
    except Exception as error:
        print(
            f"Goal Redis connection failed: {redact_redis_secrets(str(error))}",
            flush=True,
        )
        raise


async def ensure_goal_redis() -> None:
    await test_goal_redis_connection()
