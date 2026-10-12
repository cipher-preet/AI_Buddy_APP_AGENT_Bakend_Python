from __future__ import annotations

import asyncio
import os
import time
from urllib.parse import urlparse, urlunparse

import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool
from redis.exceptions import ConnectionError, MaxConnectionsError, RedisError, ResponseError

from apps.api_gateway.config.setting import settings


_clients: dict[str, redis.Redis] = {}
_client_names: dict[str, str] = {}
_cleanup_lock = asyncio.Lock()
_capacity_until = 0.0
_CAPACITY_MARKERS = (
    "max number of clients reached",
    "maxclients",
    "err max number of clients",
)


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


def normalize_redis_url(url: str) -> str:
    return prefer_ipv4_loopback_url((url or "").strip()).rstrip("/")


def is_cloud_redis_url(url: str) -> bool:
    host = (urlparse(normalize_redis_url(url)).hostname or "").lower()
    if not host or host in {"127.0.0.1", "localhost", "::1"}:
        return False
    if host.endswith(".local"):
        return False
    return True


def is_redis_capacity_error(error: BaseException | None) -> bool:
    if error is None:
        return False
    if isinstance(error, MaxConnectionsError):
        return True
    text = str(error).lower()
    return any(marker in text for marker in _CAPACITY_MARKERS)


def mark_redis_capacity_pressure(seconds: float = 20.0) -> None:
    global _capacity_until
    _capacity_until = max(_capacity_until, time.monotonic() + max(1.0, seconds))


def redis_under_capacity_pressure() -> bool:
    return time.monotonic() < _capacity_until


def capacity_backoff_seconds(failures: int) -> float:
    return min(30.0, 1.0 * (2 ** max(0, min(int(failures), 5))))


def supervisor_restart_delay(error: BaseException, *, storm: bool = False) -> float:
    if is_redis_capacity_error(error):
        return 45.0 if storm else 20.0
    return 8.0 if storm else 2.0


def redis_pool_limit(url: str, requested: int | None = None) -> int:
    """
    Size the shared pool after multiplexed stream reads.

    Too small → local MaxConnectionsError.
    Too large → Redis Cloud "max number of clients reached".
    Essentials free plans usually allow ~30 clients across every process.
    """
    configured = int(requested if requested is not None else (settings.MINDMAP_REDIS_MAX_CONNECTIONS or 8))
    if is_cloud_redis_url(url):
        return max(2, min(configured, 8))
    return max(2, min(configured, 64))


def _process_client_name(label: str) -> str:
    return f"buddy-orch-{label}-{os.getpid()}"


class WaitingConnectionPool(ConnectionPool):
    """Wait for a free pooled socket instead of crashing every stream worker."""

    def __init__(self, *args, wait_timeout_seconds: float = 45.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._wait_timeout_seconds = wait_timeout_seconds

    async def get_connection(self, command_name=None, *keys, **options):
        deadline = time.monotonic() + self._wait_timeout_seconds
        delay = 0.05
        while True:
            try:
                return await super().get_connection(command_name, *keys, **options)
            except asyncio.CancelledError:
                raise
            except (MaxConnectionsError, ConnectionError, ResponseError, RedisError) as error:
                if not isinstance(error, MaxConnectionsError) and not is_redis_capacity_error(error):
                    raise
                mark_redis_capacity_pressure()
                if time.monotonic() >= deadline:
                    raise
                await asyncio.sleep(delay)
                delay = min(delay * 1.7, 2.0)


def get_shared_redis_client(
    url: str,
    *,
    max_connections: int | None = None,
    label: str = "redis",
) -> redis.Redis:
    """
    One asyncio Redis client per normalized URL for this process.

    REDIS_URL, REMINDER_REDIS_URL, and MINDMAP_REDIS_URL often point at the same
    Redis Cloud DB. Creating separate pools burns the plan's client quota.
    """
    normalized = normalize_redis_url(url)
    if not normalized:
        raise ValueError(f"{label} Redis URL is empty")
    existing = _clients.get(normalized)
    if existing is not None:
        return existing

    limit = redis_pool_limit(normalized, max_connections)
    client_name = _process_client_name(label)
    pool = WaitingConnectionPool.from_url(
        normalized,
        decode_responses=True,
        max_connections=limit,
        socket_connect_timeout=8,
        socket_timeout=35,
        socket_keepalive=True,
        health_check_interval=120,
        retry_on_timeout=True,
        client_name=client_name,
    )
    client = redis.Redis(connection_pool=pool)
    _clients[normalized] = client
    _client_names[normalized] = client_name
    host = urlparse(normalized).hostname or ""
    print(
        f"Redis pool ready: label={label} host={host} max_connections={limit} "
        f"shared=true wait=true client_name={client_name}",
        flush=True,
    )
    return client


async def reclaim_stale_redis_clients(client: redis.Redis, *, idle_seconds: int = 30) -> int:
    """Close leftover buddy-orch / idle buddy-node clients from dead processes."""
    killed = 0
    my_pid = str(os.getpid())
    async with _cleanup_lock:
        try:
            rows = await client.execute_command("CLIENT", "LIST")
        except Exception as error:
            print(f"Redis CLIENT LIST skipped: {type(error).__name__}", flush=True)
            return 0
        text = rows.decode() if isinstance(rows, (bytes, bytearray)) else str(rows or "")
        for line in text.splitlines():
            fields = dict(part.split("=", 1) for part in line.split() if "=" in part)
            name = fields.get("name") or ""
            addr = fields.get("addr") or ""
            # Also reclaim unnamed idle clients from this app host if clearly abandoned.
            is_buddy = name.startswith("buddy-orch-") or name.startswith("buddy-node-")
            try:
                idle = int(fields.get("idle") or 0)
            except ValueError:
                idle = 0
            if my_pid and my_pid in name:
                continue
            if is_buddy and name.startswith("buddy-orch-"):
                pass
            elif is_buddy and idle >= idle_seconds:
                pass
            elif not name and idle >= max(idle_seconds * 4, 120):
                # Very idle anonymous clients — often leftovers from older code.
                pass
            else:
                continue
            client_id = fields.get("id")
            if not client_id:
                continue
            try:
                await client.execute_command("CLIENT", "KILL", "ID", client_id)
                killed += 1
            except Exception:
                continue
            _ = addr
    if killed:
        print(f"Redis closed unused clients: {killed}", flush=True)
    return killed


async def disconnect_idle_pool_connections() -> None:
    """Drop unused pooled TCP sockets so idle workers do not keep clients open."""
    for client in list(_clients.values()):
        pool = getattr(client, "connection_pool", None)
        if pool is None:
            continue
        try:
            await pool.disconnect(inuse_connections=False)
        except TypeError:
            try:
                await pool.disconnect()
            except Exception:
                pass
        except Exception:
            pass


async def recover_redis_capacity(client: redis.Redis | None = None) -> int:
    mark_redis_capacity_pressure()
    target = client
    if target is None:
        values = list(_clients.values())
        target = values[0] if values else None
    if target is None:
        return 0
    try:
        return await reclaim_stale_redis_clients(target, idle_seconds=15)
    except Exception:
        return 0


async def prepare_shared_redis(client: redis.Redis) -> None:
    await client.ping()
    for url, name in list(_client_names.items()):
        if _clients.get(url) is client:
            try:
                await client.client_setname(name)
            except Exception:
                try:
                    await client.execute_command("CLIENT", "SETNAME", name)
                except Exception:
                    pass
            break
    await reclaim_stale_redis_clients(client)


async def close_shared_redis_clients() -> None:
    clients = list(_clients.values())
    _clients.clear()
    _client_names.clear()
    for client in clients:
        try:
            pool = getattr(client, "connection_pool", None)
            if pool is not None:
                await pool.disconnect(inuse_connections=True)
        except Exception:
            pass
        try:
            await client.aclose()
        except Exception:
            pass
