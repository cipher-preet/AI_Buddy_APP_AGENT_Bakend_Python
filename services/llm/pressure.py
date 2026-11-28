"""One in-flight call per model provider, then a pause after a refusal.

Parallel meeting windows share this gate. After a rate limit or an overload
response, later calls are not sent. The notes job goes back to the Redis queue.
"""

from __future__ import annotations

import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from services.llm.async_runtime import LoopLocalSemaphore
from services.llm.errors import LLMProviderError

_COOLDOWN_SECONDS = 20.0
# Mistral code 3505: the model is out of capacity. Use the other model for a while.
_CAPACITY_COOLDOWN_SECONDS = 180.0
_until: dict[str, float] = {}
_until_lock = threading.Lock()
_turns: dict[str, LoopLocalSemaphore] = {}
_turns_lock = threading.Lock()


def note_provider_pressure(
    provider: str,
    retry_after: str | None = None,
    *,
    capacity: bool = False,
) -> None:
    floor = _CAPACITY_COOLDOWN_SECONDS if capacity else _COOLDOWN_SECONDS
    cap = 600.0 if capacity else 60.0
    seconds = floor
    if retry_after:
        try:
            seconds = min(cap, max(floor, float(retry_after)))
        except ValueError:
            seconds = floor
    with _until_lock:
        _until[provider] = max(_until.get(provider, 0.0), time.monotonic() + seconds)


def provider_cooling(provider: str) -> bool:
    return pause_remaining_seconds(provider) > 0


def pause_remaining_seconds(provider: str) -> float:
    with _until_lock:
        return max(0.0, _until.get(provider, 0.0) - time.monotonic())


def reset_provider_pressure() -> None:
    with _until_lock:
        _until.clear()


def _turn(provider: str) -> LoopLocalSemaphore:
    with _turns_lock:
        turn = _turns.get(provider)
        if turn is None:
            turn = LoopLocalSemaphore(1)
            _turns[provider] = turn
        return turn


@asynccontextmanager
async def provider_slot(provider: str) -> AsyncIterator[None]:
    """Hold the provider until this call finishes. A cooling provider is not called."""
    async with _turn(provider).get():
        if provider_cooling(provider):
            print(
                "LLM provider paused after a refusal; using the next model:",
                {"provider": provider},
            )
            raise LLMProviderError(
                f"{provider} is paused after a rate limit; using the next model",
                retryable=True,
                status_code=429,
                failure_reason="RATE_LIMITED",
            )
        yield
