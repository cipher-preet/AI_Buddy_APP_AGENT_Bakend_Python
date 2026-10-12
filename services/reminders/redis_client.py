from __future__ import annotations

from urllib.parse import urlparse

import redis.asyncio as redis

from apps.api_gateway.config.setting import settings
from services.queue.shared_redis import get_shared_redis_client, normalize_redis_url


class ReminderRedisConfigError(RuntimeError):
    pass


def describe_redis_url(url: str) -> dict[str, str | int | bool]:
    parsed = urlparse(url)
    tls = parsed.scheme == "rediss"
    port = parsed.port or (6380 if tls else 6379)
    return {
        "host": parsed.hostname or "",
        "port": port,
        "tls": tls,
    }


def format_redis_target(target: dict[str, str | int | bool]) -> str:
    tls = "true" if target.get("tls") else "false"
    return f"host={target.get('host')} port={target.get('port')} tls={tls}"


def redact_redis_secrets(message: str) -> str:
    import re

    return re.sub(r"redis(?:s)?://[^\s\"']+", "redis://<redacted>", message, flags=re.I)


def get_reminder_redis_url() -> str:
    url = (settings.REMINDER_REDIS_URL or "").strip()
    if not url:
        raise ReminderRedisConfigError("REMINDER_REDIS_URL is not configured")
    return url


def get_reminder_redis_client() -> redis.Redis:
    return get_shared_redis_client(
        normalize_redis_url(get_reminder_redis_url()),
        max_connections=settings.MINDMAP_REDIS_MAX_CONNECTIONS,
        label="reminder",
    )


async def test_reminder_redis_connection() -> bool:
    from services.queue.shared_redis import normalize_redis_url, prepare_shared_redis

    url = get_reminder_redis_url()
    target = describe_redis_url(url)
    client = get_reminder_redis_client()
    try:
        # Same shared pool as conversation — do not open another client or spam ping.
        if normalize_redis_url(url) == normalize_redis_url(settings.REDIS_URL):
            print(
                f"Reminder Redis connected: host={target['host']} port={target['port']} shared_with=conversation",
                flush=True,
            )
            return True
        await prepare_shared_redis(client)
        print(
            f"Reminder Redis connected: host={target['host']} port={target['port']}",
            flush=True,
        )
        return True
    except Exception as error:
        print(
            f"Reminder Redis connection failed: {redact_redis_secrets(str(error))}",
            flush=True,
        )
        raise
