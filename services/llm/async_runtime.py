"""Event-loop ownership for async HTTP/SDK clients.

The application/worker runtime owns the event loop. Provider objects may be
process-wide singletons, but loop-bound transports must be created for the
currently running loop and discarded when that loop is gone.

Do not use asyncio.run() inside an already running worker/request loop.
"""

from __future__ import annotations

import asyncio
import os
import socket
import threading
from collections.abc import Callable
from typing import Any
from uuid import uuid4

ASYNC_LIFECYCLE_ERROR = "ASYNC_LIFECYCLE_ERROR"

_LIFECYCLE_MARKERS = (
    "event loop is closed",
    "attached to a different loop",
    "bound to a different event loop",
    "got future",
    "no running event loop",
    "loop is closed",
)


_LOOP_TOKEN_ATTR = "_buddy_async_runtime_token"


def current_loop_id() -> str | None:
    """Stable identity for the running loop. Do not use id(loop): CPython can reuse it."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None
    if loop.is_closed():
        return None
    token = getattr(loop, _LOOP_TOKEN_ATTR, None)
    if not token:
        token = uuid4().hex
        setattr(loop, _LOOP_TOKEN_ATTR, token)
    return token


def worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


def is_async_lifecycle_error(error: BaseException | None) -> bool:
    if error is None:
        return False
    failure_reason = getattr(error, "failure_reason", None)
    if str(failure_reason or "") == ASYNC_LIFECYCLE_ERROR:
        return True
    name = type(error).__name__
    if name in {"AsyncLifecycleError"}:
        return True
    text = f"{name} {error}".casefold()
    if "event loop is closed" in text:
        return True
    if "bound to a different event loop" in text:
        return True
    if "attached to a different loop" in text:
        return True
    if "got future" in text and "different" in text:
        return True
    return any(marker in text for marker in ("no running event loop", "loop is closed"))


def reraise_if_hard_runtime(error: BaseException) -> None:
    if is_async_lifecycle_error(error):
        raise error
    name = type(error).__name__
    if name in {"PipelineBudgetExceeded", "EventPipelineHardFailure"}:
        raise error


async def close_async_resource_safely(resource: Any, created_loop_id: str | None) -> None:
    if resource is None:
        return
    closer = getattr(resource, "aclose", None) or getattr(resource, "close", None)
    if closer is None:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    same_loop = (
        loop is not None
        and not loop.is_closed()
        and created_loop_id is not None
        and created_loop_id == current_loop_id()
    )
    if not same_loop:
        return
    try:
        result = closer()
        if asyncio.iscoroutine(result):
            await result
    except Exception:
        return


class LoopBoundAsyncClient:
    """httpx.AsyncClient (or compatible) bound to the current running loop."""

    def __init__(self, factory: Callable[[], Any]):
        self._factory = factory
        self._client: Any | None = None
        self._created_loop_id: str | None = None
        self._client_id: str | None = None
        self._closed = True
        self._lock = threading.Lock()

    def debug(self) -> dict[str, Any]:
        return {
            "client_id": self._client_id,
            "created_loop_id": self._created_loop_id,
            "current_loop_id": current_loop_id(),
            "closed": bool(self._closed or self._client is None or getattr(self._client, "is_closed", False)),
        }

    def stale(self) -> bool:
        current = current_loop_id()
        if self._client is None or self._closed:
            return True
        if current is None or self._created_loop_id != current:
            return True
        if getattr(self._client, "is_closed", False):
            return True
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return True
        return bool(loop.is_closed())

    async def get(self) -> Any:
        if not self.stale():
            return self._client
        return await self.replace()

    async def replace(self) -> Any:
        old = self._client
        old_loop = self._created_loop_id
        client = self._factory()
        with self._lock:
            self._client = client
            self._created_loop_id = current_loop_id()
            self._client_id = f"{id(client):x}-{uuid4().hex[:8]}"
            self._closed = False
        await close_async_resource_safely(old, old_loop)
        return client

    async def aclose(self) -> None:
        old = self._client
        old_loop = self._created_loop_id
        with self._lock:
            self._client = None
            self._created_loop_id = None
            self._client_id = None
            self._closed = True
        await close_async_resource_safely(old, old_loop)


class LoopLocalSemaphore:
    """asyncio.Semaphore is loop-bound; keep one per running loop."""

    def __init__(self, value: int):
        self._value = max(1, int(value))
        self._by_loop: dict[str, asyncio.Semaphore] = {}
        self._lock = threading.Lock()

    def get(self) -> asyncio.Semaphore:
        loop_id = current_loop_id()
        if loop_id is None:
            raise RuntimeError(f"{ASYNC_LIFECYCLE_ERROR}: no running event loop")
        with self._lock:
            semaphore = self._by_loop.get(loop_id)
            if semaphore is None:
                semaphore = asyncio.Semaphore(self._value)
                self._by_loop[loop_id] = semaphore
            stale = [key for key in self._by_loop if key != loop_id]
            for key in stale:
                self._by_loop.pop(key, None)
            return semaphore


class LoopLocalResource:
    """Generic per-loop resource cache (SDK clients, etc.)."""

    def __init__(self, factory: Callable[[], Any]):
        self._factory = factory
        self._by_loop: dict[str, tuple[Any, str]] = {}
        self._lock = threading.Lock()

    def debug(self, loop_id: str | None = None) -> dict[str, Any]:
        loop_id = loop_id if loop_id is not None else current_loop_id()
        item = self._by_loop.get(loop_id) if loop_id is not None else None
        resource, client_id = item if item else (None, None)
        return {
            "client_id": client_id,
            "created_loop_id": loop_id if item else None,
            "current_loop_id": current_loop_id(),
            "closed": resource is None,
        }

    def get(self) -> Any:
        loop_id = current_loop_id()
        if loop_id is None:
            raise RuntimeError(f"{ASYNC_LIFECYCLE_ERROR}: no running event loop")
        with self._lock:
            item = self._by_loop.get(loop_id)
            if item is not None:
                return item[0]
            resource = self._factory()
            self._by_loop[loop_id] = (resource, f"{id(resource):x}-{uuid4().hex[:8]}")
            return resource

    async def aclose_current(self) -> None:
        loop_id = current_loop_id()
        if loop_id is None:
            return
        with self._lock:
            item = self._by_loop.pop(loop_id, None)
        if item:
            await close_async_resource_safely(item[0], loop_id)

    async def aclose_all(self) -> None:
        with self._lock:
            items = list(self._by_loop.items())
            self._by_loop.clear()
        current = current_loop_id()
        for loop_id, (resource, _) in items:
            await close_async_resource_safely(resource, loop_id if loop_id == current else None)
