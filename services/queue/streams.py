from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationInfo, field_validator
from redis.exceptions import (
    ConnectionError,
    MaxConnectionsError,
    ResponseError,
    TimeoutError as RedisTimeoutError,
)

from apps.api_gateway.config.setting import settings
from services.queue.factory import use_queue_api
from services.queue.http_queue import QueueApiPublisher
from services.queue.redis_queue import redis_client
from services.queue.shared_redis import (
    capacity_backoff_seconds,
    is_redis_capacity_error,
    mark_redis_capacity_pressure,
    recover_redis_capacity,
    redis_under_capacity_pressure,
)


class NonRetryableQueueError(RuntimeError):
    pass


class EventEnvelope(BaseModel):
    eventId: str = Field(default_factory=lambda: str(uuid4()))
    eventType: str
    eventVersion: int = 1
    correlationId: str
    causationId: str | None = None
    userId: str
    spaceId: str
    conversationId: str
    payload: dict[str, Any] = Field(default_factory=dict)
    attempt: int = 0
    createdAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator(
        "eventId",
        "eventType",
        "correlationId",
        "causationId",
        "userId",
        "spaceId",
        "conversationId",
        mode="before",
    )
    @classmethod
    def stringify_identifiers(cls, value, info: ValidationInfo):
        # Extension meetings are unscoped: never let a raw None crash Redis publish.
        if info.field_name == "spaceId":
            if value is None:
                return ""
            text = str(value).strip()
            if not text or text.lower() in {"none", "null", "undefined"}:
                return ""
            return text
        if value is None:
            return value
        return str(value)


class RedisStreamProducer:
    def __init__(self, redis=None, *, force_direct: bool = False):
        self._redis = redis or redis_client
        self._force_direct = force_direct

    async def publish(
        self,
        stream: str,
        event: EventEnvelope,
        *,
        maxlen: int | None = None,
    ) -> str:
        if use_queue_api() and not self._force_direct:
            payload = event.model_dump(mode="json")
            payload["targetStream"] = stream
            await QueueApiPublisher().publish(payload)
            return event.eventId
        fields = {"event": event.model_dump_json()}
        if maxlen and maxlen > 0:
            # Approximate trim keeps memory bounded without blocking every XADD.
            return await self._redis.xadd(stream, fields, maxlen=maxlen, approximate=True)
        return await self._redis.xadd(stream, fields)


class RedisStreamConsumer:
    def __init__(
        self,
        stream: str,
        group: str,
        consumer_name: str,
        handler: Callable[[EventEnvelope], Awaitable[None]],
        concurrency: int | None = None,
        max_retries: int | None = None,
        on_dead_letter: Callable[[EventEnvelope, Exception], Awaitable[None]] | None = None,
        redis=None,
        *,
        delete_after_ack: bool = False,
        stream_maxlen: int | None = None,
    ):
        self.stream = stream
        self.group = group
        self.consumer_name = consumer_name
        self.handler = handler
        self.concurrency = concurrency or settings.WORKER_CONCURRENCY
        self.max_retries = max_retries or settings.WORKER_MAX_RETRIES
        self.on_dead_letter = on_dead_letter
        self.delete_after_ack = delete_after_ack
        self.stream_maxlen = int(stream_maxlen) if stream_maxlen and stream_maxlen > 0 else None
        self._redis = redis or redis_client
        self._uses_default_redis = redis is None
        self._shutdown = asyncio.Event()
        self._semaphore = asyncio.Semaphore(self.concurrency)
        self._last_claim_at = 0.0
        self._capacity_failures = 0

    async def _ack_and_cleanup(self, message_id: str) -> None:
        await self._redis.xack(self.stream, self.group, message_id)
        if not self.delete_after_ack:
            return
        try:
            # Drop finished entries immediately so Redis Cloud memory stays free.
            await self._redis.xdel(self.stream, message_id)
        except Exception as error:
            print(
                "Redis stream xdel after ack failed:",
                {
                    "stream": self.stream,
                    "message_id": message_id,
                    "error": str(error),
                },
                flush=True,
            )
        if self.stream_maxlen:
            try:
                # Exact trim after each finished job — critical on Essentials 30MB.
                await self._redis.xtrim(self.stream, maxlen=self.stream_maxlen, approximate=False)
            except Exception as error:
                print(
                    "Redis stream xtrim after ack failed:",
                    {
                        "stream": self.stream,
                        "maxlen": self.stream_maxlen,
                        "error": str(error),
                    },
                    flush=True,
                )

    async def _xadd_bounded(self, stream: str, fields: dict[str, Any], *, maxlen: int | None = None) -> str:
        limit = maxlen if maxlen and maxlen > 0 else self.stream_maxlen
        if limit and limit > 0:
            return await self._redis.xadd(stream, fields, maxlen=limit, approximate=True)
        return await self._redis.xadd(stream, fields)

    async def ensure_group(self) -> None:
        try:
            await self._redis.xgroup_create(self.stream, self.group, id="0", mkstream=True)
            print(
                "Redis stream consumer group created:",
                {"stream": self.stream, "group": self.group},
            )
        except ResponseError as error:
            if "BUSYGROUP" not in str(error):
                raise
        if self.stream_maxlen and self.delete_after_ack:
            try:
                trimmed = await self._redis.xtrim(
                    self.stream,
                    maxlen=self.stream_maxlen,
                    approximate=False,
                )
                if trimmed:
                    print(
                        "Redis stream trimmed on startup:",
                        {
                            "stream": self.stream,
                            "maxlen": self.stream_maxlen,
                            "removed": int(trimmed),
                        },
                        flush=True,
                    )
            except Exception as error:
                print(
                    "Redis stream startup trim failed:",
                    {"stream": self.stream, "error": str(error)},
                    flush=True,
                )

    async def stop(self) -> None:
        self._shutdown.set()

    def pool_identity(self) -> int:
        pool = getattr(self._redis, "connection_pool", None)
        return id(pool) if pool is not None else id(self._redis)

    async def read_batch(self, *, block_ms: int) -> list:
        return (
            await self._redis.xreadgroup(
                groupname=self.group,
                consumername=self.consumer_name,
                streams={self.stream: ">"},
                count=settings.REDIS_BATCH_SIZE,
                block=max(0, int(block_ms)),
            )
            or []
        )

    async def dispatch_messages(self, messages) -> None:
        tasks: list[asyncio.Task[None]] = []
        for _, entries in messages or []:
            for message_id, fields in entries:
                if self._shutdown.is_set():
                    break
                await self._semaphore.acquire()
                tasks.append(asyncio.create_task(self._handle_message(message_id, fields)))
        if tasks:
            await asyncio.gather(*tasks)

    async def maybe_claim_stale(self) -> None:
        interval = float(getattr(settings, "REDIS_CLAIM_INTERVAL_SECONDS", 30) or 30)
        now = time.monotonic()
        if now - self._last_claim_at < interval:
            return
        self._last_claim_at = now
        if redis_under_capacity_pressure():
            return
        await self.claim_stale()

    async def _backoff_capacity(self, error: BaseException) -> None:
        self._capacity_failures += 1
        mark_redis_capacity_pressure()
        print(
            "Redis stream consumer backing off:",
            {
                "stream": self.stream,
                "group": self.group,
                "error": str(error),
                "failures": self._capacity_failures,
            },
            flush=True,
        )
        await recover_redis_capacity(self._redis)
        await asyncio.sleep(capacity_backoff_seconds(self._capacity_failures))

    async def _read_or_recover(self, *, block_ms: int):
        try:
            messages = await self.read_batch(block_ms=block_ms)
            self._capacity_failures = 0
            return messages
        except ResponseError as error:
            if "NOGROUP" in str(error):
                await self.ensure_group()
                return []
            if is_redis_capacity_error(error):
                await self._backoff_capacity(error)
                return []
            raise
        except MaxConnectionsError as error:
            await self._backoff_capacity(error)
            return []
        except (RedisTimeoutError, ConnectionError) as error:
            if is_redis_capacity_error(error):
                await self._backoff_capacity(error)
                return []
            print(
                "Redis stream consumer read failed:",
                {
                    "stream": self.stream,
                    "group": self.group,
                    "error": str(error),
                },
                flush=True,
            )
            await asyncio.sleep(1)
            return []

    async def run_forever(self) -> None:
        await self.ensure_group()
        print(
            "Redis stream consumer started:",
            {
                "stream": self.stream,
                "group": self.group,
                "consumer": self.consumer_name,
                "concurrency": self.concurrency,
            },
        )
        while not self._shutdown.is_set():
            await self.maybe_claim_stale()
            messages = await self._read_or_recover(block_ms=settings.REDIS_BLOCK_MS)
            await self.dispatch_messages(messages)

    async def claim_stale(self) -> None:
        try:
            claimed = await self._redis.xautoclaim(
                self.stream,
                self.group,
                self.consumer_name,
                min_idle_time=settings.REDIS_CLAIM_IDLE_MS,
                start_id="0-0",
                count=settings.REDIS_BATCH_SIZE,
            )
        except (ResponseError, RedisTimeoutError, ConnectionError, MaxConnectionsError):
            return
        entries = claimed[1] if len(claimed) > 1 else []
        for message_id, fields in entries:
            if self._shutdown.is_set():
                break
            await self._semaphore.acquire()
            await self._handle_message(message_id, fields)

    async def _handle_message(self, message_id: str, fields: dict[str, Any]) -> None:
        try:
            event = EventEnvelope.model_validate_json(fields["event"])
            await self.handler(event)
            await self._ack_and_cleanup(message_id)
        except Exception as error:
            await self._handle_failure(message_id, fields, error)
        finally:
            self._semaphore.release()

    async def _handle_failure(self, message_id: str, fields: dict[str, Any], error: Exception) -> None:
        try:
            event = EventEnvelope.model_validate_json(fields["event"])
        except Exception:
            print(
                "Redis stream message sent to dead letter:",
                {
                    "stream": self.stream,
                    "group": self.group,
                    "message_id": message_id,
                    "error": str(error),
                    "reason": "invalid_event_payload",
                },
            )
            await self._xadd_bounded(
                settings.REDIS_DEAD_LETTER_STREAM,
                {"event": json.dumps({"raw": fields, "error": str(error)})},
                maxlen=settings.REDIS_CLOUD_DEAD_LETTER_MAXLEN if not self._uses_default_redis else None,
            )
            await self._ack_and_cleanup(message_id)
            return

        if isinstance(error, NonRetryableQueueError) or event.attempt >= self.max_retries:
            print(
                "Redis stream event sent to dead letter:",
                {
                    "stream": self.stream,
                    "group": self.group,
                    "message_id": message_id,
                    "eventId": event.eventId,
                    "eventType": event.eventType,
                    "correlationId": event.correlationId,
                    "attempt": event.attempt,
                    "error": str(error),
                    "retryable": not isinstance(error, NonRetryableQueueError),
                },
            )
            await self._xadd_bounded(
                settings.REDIS_DEAD_LETTER_STREAM,
                {
                    "event": event.model_dump_json(),
                    "error": str(error),
                    "sourceStream": self.stream,
                },
                maxlen=settings.REDIS_CLOUD_DEAD_LETTER_MAXLEN if not self._uses_default_redis else None,
            )
            await self._ack_and_cleanup(message_id)
            if self.on_dead_letter is not None:
                try:
                    await self.on_dead_letter(event, error)
                except Exception as callback_error:
                    print(
                        "Redis stream dead-letter callback failed:",
                        {
                            "stream": self.stream,
                            "eventId": event.eventId,
                            "eventType": event.eventType,
                            "error": str(callback_error),
                        },
                    )
            return

        retry = event.model_copy(update={"attempt": event.attempt + 1})
        print(
            "Redis stream event scheduled for retry:",
            {
                "stream": self.stream,
                "group": self.group,
                "message_id": message_id,
                "eventId": event.eventId,
                "eventType": event.eventType,
                "correlationId": event.correlationId,
                "nextAttempt": retry.attempt,
                "error": str(error),
            },
        )
        if self._uses_default_redis:
            # Conversation retry relay re-publishes from REDIS_RETRY_STREAM.
            await self._redis.xadd(
                settings.REDIS_RETRY_STREAM,
                {
                    "event": retry.model_dump_json(),
                    "targetStream": self.stream,
                    "notBefore": str(datetime.now(timezone.utc).timestamp() + retry_delay(event.attempt)),
                },
            )
        else:
            # Dedicated Redis (e.g. cloud mindmap) has no shared retry relay — re-queue bounded.
            await self._xadd_bounded(self.stream, {"event": retry.model_dump_json()})
        await self._ack_and_cleanup(message_id)


class MultiplexedStreamHub:
    """One non-blocking poller for many streams that share a Redis pool."""

    def __init__(self, consumers: Sequence[RedisStreamConsumer]):
        if not consumers:
            raise ValueError("MultiplexedStreamHub needs at least one consumer")
        self.consumers = list(consumers)
        self.stream = f"hub:{self.consumers[0].stream}"
        self._shutdown = asyncio.Event()

    async def ensure_group(self) -> None:
        for consumer in self.consumers:
            await consumer.ensure_group()
            await asyncio.sleep(0.02)

    async def stop(self) -> None:
        self._shutdown.set()
        for consumer in self.consumers:
            await consumer.stop()

    async def run_forever(self) -> None:
        await self.ensure_group()
        idle_ms = int(getattr(settings, "REDIS_POLL_IDLE_MS", 120) or 120)
        print(
            "Redis multiplexed stream hub started:",
            {
                "streams": [consumer.stream for consumer in self.consumers],
                "idle_ms": idle_ms,
            },
            flush=True,
        )
        while not self._shutdown.is_set():
            had_work = False
            for consumer in self.consumers:
                if self._shutdown.is_set():
                    break
                await consumer.maybe_claim_stale()
                messages = await consumer._read_or_recover(block_ms=0)
                if messages:
                    had_work = True
                    await consumer.dispatch_messages(messages)
            if not had_work:
                await asyncio.sleep(max(0.04, idle_ms / 1000))


def group_stream_runners(items: Sequence[Any]) -> list[Any]:
    """Collapse RedisStreamConsumers that share a pool into one hub each."""
    buckets: dict[int, list[RedisStreamConsumer]] = {}
    others: list[Any] = []
    for item in items:
        if isinstance(item, RedisStreamConsumer):
            buckets.setdefault(item.pool_identity(), []).append(item)
        else:
            others.append(item)
    runners: list[Any] = []
    for group in buckets.values():
        if len(group) == 1:
            runners.append(group[0])
        else:
            runners.append(MultiplexedStreamHub(group))
    runners.extend(others)
    return runners


def retry_delay(attempt: int) -> float:
    base = settings.WORKER_RETRY_BASE_SECONDS
    return min(settings.WORKER_RETRY_MAX_SECONDS, base * (2**attempt) + random.uniform(0, base))


async def stream_metrics(stream: str, group: str) -> dict[str, int]:
    info = await redis_client.xinfo_stream(stream)
    groups = await redis_client.xinfo_groups(stream)
    group_info = next((item for item in groups if item.get("name") == group), {})
    return {
        "length": int(info.get("length", 0)),
        "pending": int(group_info.get("pending", 0) or 0),
        "lag": int(group_info.get("lag", 0) or 0),
    }
