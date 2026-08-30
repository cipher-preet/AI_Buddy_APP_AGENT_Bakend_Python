from __future__ import annotations

from urllib.parse import urlparse

import redis.asyncio as redis

from apps.api_gateway.config.setting import settings


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


_reminder_redis_client: redis.Redis | None = None


def get_reminder_redis_client() -> redis.Redis:
    global _reminder_redis_client
    if _reminder_redis_client is None:
        _reminder_redis_client = redis.from_url(
            get_reminder_redis_url(),
            decode_responses=True,
            socket_connect_timeout=10,
            socket_timeout=30,
            health_check_interval=30,
        )
    return _reminder_redis_client


async def test_reminder_redis_connection() -> bool:
    url = get_reminder_redis_url()
    target = describe_redis_url(url)
    client = get_reminder_redis_client()
    try:
        pong = await client.ping()
        print(
            f"Reminder Redis connected: host={target['host']} port={target['port']}",
            flush=True,
        )
        return bool(pong)
    except Exception as error:
        print(
            f"Reminder Redis connection failed: {redact_redis_secrets(str(error))}",
            flush=True,
        )
        raise
