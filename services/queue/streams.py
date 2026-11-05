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
    RedisError,
    ResponseError,
    TimeoutError as RedisTimeoutError,
)

from apps.api_gateway.config.setting import settings
from services.observability.diagnostics import diag_log
from services.queue.factory import use_queue_api
from services.queue.http_queue import QueueApiPublisher
from services.queue.job_lease import (
    LeaseUnavailable,
    LeavePending,
    StaleOwnershipError,
    acquire_lease,
    logical_job_key,
    reclaim_min_idle_ms,
    release_lease,
    remember_completion,
    renew_lease,
    reset_lease,
    spawn_heartbeat,
    preserve_pending_delivery,
    push_lease,
)
from services.queue.redis_queue import redis_client
from services.queue.shared_redis import (
    capacity_backoff_seconds,
    is_redis_capacity_error,
    mark_redis_capacity_pressure,
    recover_redis_capacity,
    recycle_broken_redis_pool,
    redis_under_capacity_pressure,
    should_log_redis_timeout,
)


def xreadgroup_block(block_ms: int | None) -> int | None:
    """
    Redis BLOCK 0 waits forever. That plus socket_timeout looks like a TimeoutError
    on every empty stream. Omit BLOCK (None) for a true non-blocking read.
    """
    if block_ms is None:
        return None
    value = int(block_ms)
    return value if value > 0 else None


class IdlePoll:
    """Slow down Redis reads when queues are empty so Cloud is not hammered."""

    def __init__(self, min_seconds: float = 0.4, max_seconds: float = 2.5):
        self.min_seconds = min_seconds
        self.max_seconds = max_seconds
        self.seconds = min_seconds

    def hit(self) -> None:
        self.seconds = self.min_seconds

    def miss(self) -> float:
        self.seconds = min(self.max_seconds, max(self.min_seconds, self.seconds * 1.5))
        return self.seconds


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
        # maxlen is accepted and ignored. Redis 7 MAXLEN removes entries that
        # another consumer group has not delivered or acknowledged.
        del maxlen
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
        self._timeout_failures = 0
        self._inflight: set[str] = set()
        self._handler_tasks: set[asyncio.Task[None]] = set()

    async def _ack_and_cleanup(self, message_id: str) -> None:
        await self._redis.xack(self.stream, self.group, message_id)
        if not self.delete_after_ack:
            return
        try:
            # Delete only this acknowledged id. Do not trim the rest of the stream:
            # a length cap also deletes entries that are still pending.
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

    async def _xadd_bounded(self, stream: str, fields: dict[str, Any], *, maxlen: int | None = None) -> str:
        del maxlen
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

    async def stop(self) -> None:
        self._shutdown.set()

    async def drain(self) -> None:
        """Finish in-flight handlers. Cancel only after the existing drain timeout."""
        pending = {task for task in self._handler_tasks if not task.done()}
        if not pending:
            return
        timeout = float(settings.WORKER_GRACEFUL_SHUTDOWN_SECONDS)
        try:
            await asyncio.wait(pending, timeout=timeout)
        except asyncio.CancelledError:
            await asyncio.shield(asyncio.wait(pending, timeout=timeout))
            raise
        leftover = {task for task in pending if not task.done()}
        for task in leftover:
            task.cancel()
        if leftover:
            await asyncio.wait(leftover)

    def pool_identity(self) -> int:
        pool = getattr(self._redis, "connection_pool", None)
        return id(pool) if pool is not None else id(self._redis)

    async def read_batch(self, *, block_ms: int) -> list:
        options: dict[str, Any] = {
            "groupname": self.group,
            "consumername": self.consumer_name,
            "streams": {self.stream: ">"},
            "count": settings.REDIS_BATCH_SIZE,
        }
        block = xreadgroup_block(block_ms)
        if block is not None:
            options["block"] = block
        return (await self._redis.xreadgroup(**options)) or []

    async def dispatch_messages(self, messages) -> None:
        tasks: list[asyncio.Task[None]] = []
        for _, entries in messages or []:
            for message_id, fields in entries:
                if self._shutdown.is_set():
                    break
                await self._semaphore.acquire()
                if self._shutdown.is_set():
                    self._semaphore.release()
                    break
                task = asyncio.create_task(self._handle_message(message_id, fields))
                self._handler_tasks.add(task)
                task.add_done_callback(self._handler_tasks.discard)
                tasks.append(task)
        pending = set(tasks)
        while pending and not self._shutdown.is_set():
            _, pending = await asyncio.wait(pending, timeout=0.2)

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
            self._timeout_failures = 0
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
            self._timeout_failures += 1
            if self._timeout_failures >= 3:
                await recycle_broken_redis_pool(self._redis)
                self._timeout_failures = 0
            if should_log_redis_timeout():
                print(
                    "Redis stream consumer read failed:",
                    {
                        "stream": self.stream,
                        "group": self.group,
                        "error": str(error),
                    },
                    flush=True,
                )
            await asyncio.sleep(min(4.0, 0.5 * max(1, self._timeout_failures or 1)))
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
        try:
            while not self._shutdown.is_set():
                await self.maybe_claim_stale()
                if self._shutdown.is_set():
                    break
                messages = await self._read_or_recover(block_ms=settings.REDIS_BLOCK_MS)
                if self._shutdown.is_set():
                    break
                await self.dispatch_messages(messages)
        finally:
            self._shutdown.set()
            await self.drain()

    async def claim_stale(self) -> None:
        if self._shutdown.is_set():
            return
        try:
            claimed = await self._redis.xautoclaim(
                self.stream,
                self.group,
                self.consumer_name,
                min_idle_time=reclaim_min_idle_ms(),
                start_id="0-0",
                count=settings.REDIS_BATCH_SIZE,
            )
        except (ResponseError, RedisTimeoutError, ConnectionError, MaxConnectionsError):
            return
        entries = claimed[1] if len(claimed) > 1 else []
        for message_id, fields in entries:
            if self._shutdown.is_set():
                break
            message_key = _message_key(message_id)
            if message_key in self._inflight:
                diag_log(
                    "job_duplicate_suppressed",
                    stream=self.stream,
                    message_id=message_key,
                    worker_id=self.consumer_name,
                    reason="reclaim_inflight",
                )
                diag_log(
                    "queue_duplicate_suppressed",
                    stream=self.stream,
                    message_id=message_key,
                    worker_id=self.consumer_name,
                    reason="reclaim_inflight",
                )
                continue
            diag_log(
                "job_reclaimed",
                stream=self.stream,
                group=self.group,
                message_id=message_key,
                worker_id=self.consumer_name,
                min_idle_ms=reclaim_min_idle_ms(),
            )
            await self._semaphore.acquire()
            if self._shutdown.is_set():
                self._semaphore.release()
                break
            task = asyncio.create_task(self._handle_message(message_id, fields, reclaimed=True))
            self._handler_tasks.add(task)
            task.add_done_callback(self._handler_tasks.discard)
            while not task.done() and not self._shutdown.is_set():
                await asyncio.wait({task}, timeout=0.2)

    async def _handle_message(self, message_id: str, fields: dict[str, Any], *, reclaimed: bool = False) -> None:
        message_key = _message_key(message_id)
        try:
            if message_key in self._inflight:
                diag_log(
                    "job_duplicate_suppressed",
                    stream=self.stream,
                    message_id=message_key,
                    worker_id=self.consumer_name,
                    reason="in_flight",
                )
                diag_log(
                    "queue_duplicate_suppressed",
                    stream=self.stream,
                    message_id=message_key,
                    worker_id=self.consumer_name,
                    reason="in_flight",
                )
                return
            self._inflight.add(message_key)
            try:
                await self._process_message(message_key, fields, reclaimed=reclaimed)
            finally:
                self._inflight.discard(message_key)
        finally:
            self._semaphore.release()

    async def _process_message(self, message_id: str, fields: dict[str, Any], *, reclaimed: bool) -> None:
        started = time.perf_counter()
        try:
            event = EventEnvelope.model_validate_json(fields["event"])
        except Exception as error:
            await self._handle_failure(message_id, fields, error)
            return

        try:
            decision = await acquire_lease(
                self._redis,
                event,
                owner=self.consumer_name,
                message_id=message_id,
            )
        except LeaseUnavailable:
            return

        if decision.action in {"busy", "duplicate"}:
            await preserve_pending_delivery(
                self._redis,
                stream=self.stream,
                group=self.group,
                consumer_name=self.consumer_name,
                message_id=message_id,
            )
            return
        if decision.action == "completed":
            await self._ack_and_cleanup(message_id)
            return
        lease = decision.lease
        if lease is None:
            return

        scope = push_lease(lease)
        heartbeat = spawn_heartbeat(
            self._redis,
            lease,
            stream=self.stream,
            group=self.group,
            consumer_name=self.consumer_name,
            message_id=message_id,
        )
        try:
            diag_log(
                "job_processing_started",
                job_type=lease.event_type,
                logical_job_id=lease.logical_key,
                attempt=lease.token,
                worker_id=lease.owner,
                retry_count=lease.retry_count,
                reclaimed=reclaimed,
                stream=self.stream,
            )
            try:
                await self.handler(event)
            except asyncio.CancelledError:
                lease.lost = True
                lease.reason = "cancelled"
                raise
            if lease.lost or not await _still_owned(lease):
                diag_log(
                    "job_lease_expired",
                    job_type=lease.event_type,
                    logical_job_id=lease.logical_key,
                    attempt=lease.token,
                    worker_id=lease.owner,
                    retry_count=lease.retry_count,
                    reason=lease.reason or "lost_before_ack",
                )
                return
            if lease.settled:
                remembered = await remember_completion(self._redis, lease)
                if not remembered:
                    return
            diag_log(
                "job_processing_completed",
                job_type=lease.event_type,
                logical_job_id=lease.logical_key,
                attempt=lease.token,
                worker_id=lease.owner,
                retry_count=lease.retry_count,
                duration_ms=int((time.perf_counter() - started) * 1000),
                stream=self.stream,
            )
            await self._ack_and_cleanup(message_id)
        except LeavePending:
            await preserve_pending_delivery(
                self._redis,
                stream=self.stream,
                group=self.group,
                consumer_name=self.consumer_name,
                message_id=message_id,
            )
            diag_log(
                "job_claim_rejected_active_owner",
                job_type=lease.event_type,
                logical_job_id=lease.logical_key,
                attempt=lease.token,
                worker_id=lease.owner,
                retry_count=lease.retry_count,
                reason="durable_owner",
                stream=self.stream,
            )
            return
        except StaleOwnershipError:
            lease.lost = True
            diag_log(
                "job_lease_expired",
                job_type=lease.event_type,
                logical_job_id=lease.logical_key,
                attempt=lease.token,
                worker_id=lease.owner,
                retry_count=lease.retry_count,
                reason="stale_commit",
            )
            return
        except Exception as error:
            if lease.lost:
                diag_log(
                    "job_lease_expired",
                    job_type=lease.event_type,
                    logical_job_id=lease.logical_key,
                    attempt=lease.token,
                    worker_id=lease.owner,
                    retry_count=lease.retry_count,
                    reason=lease.reason or type(error).__name__,
                )
                return
            diag_log(
                "job_processing_failed",
                job_type=lease.event_type,
                logical_job_id=lease.logical_key,
                attempt=lease.token,
                worker_id=lease.owner,
                retry_count=lease.retry_count,
                duration_ms=int((time.perf_counter() - started) * 1000),
                exception_type=type(error).__name__,
                stream=self.stream,
            )
            await self._handle_failure(message_id, fields, error)
        finally:
            heartbeat.cancel()
            try:
                await heartbeat
            except asyncio.CancelledError:
                pass
            reset_lease(scope)
            if lease is not None and not lease.released and not lease.lost:
                await release_lease(self._redis, lease)

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
            )
            diag_log(
                "job_dead_lettered",
                stream=self.stream,
                message_id=message_id,
                worker_id=self.consumer_name,
                reason="invalid_event_payload",
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
            )
            dead_letter_reason = (
                "retry_exhausted" if not isinstance(error, NonRetryableQueueError) else "non_retryable"
            )
            diag_log(
                "job_dead_lettered",
                job_type=event.eventType,
                logical_job_id=logical_job_key(event),
                worker_id=self.consumer_name,
                retry_count=event.attempt,
                stream=self.stream,
                message_id=message_id,
                reason=dead_letter_reason,
            )
            diag_log(
                "queue_dead_lettered",
                job_type=event.eventType,
                logical_job_id=logical_job_key(event),
                worker_id=self.consumer_name,
                retry_count=event.attempt,
                stream=self.stream,
                message_id=message_id,
                reason=dead_letter_reason,
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
        diag_log(
            "job_retry_scheduled",
            job_type=event.eventType,
            logical_job_id=logical_job_key(event),
            worker_id=self.consumer_name,
            retry_count=retry.attempt,
            stream=self.stream,
            message_id=message_id,
        )
        diag_log(
            "queue_retry_scheduled",
            job_type=event.eventType,
            logical_job_id=logical_job_key(event),
            worker_id=self.consumer_name,
            retry_count=retry.attempt,
            stream=self.stream,
            message_id=message_id,
        )
        if self._uses_default_redis:
            # Conversation retry relay re-publishes from REDIS_RETRY_STREAM.
            # The original message is acknowledged only after this write succeeds.
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
        self._claim_index = 0
        self._inflight: set[asyncio.Task[None]] = set()
        self._idle = IdlePoll()

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
        idle_ms = int(getattr(settings, "REDIS_POLL_IDLE_MS", 400) or 400)
        self._idle = IdlePoll(min_seconds=max(0.25, idle_ms / 1000), max_seconds=2.5)
        print(
            "Redis multiplexed stream hub started:",
            {
                "streams": [consumer.stream for consumer in self.consumers],
                "idle_ms": idle_ms,
            },
            flush=True,
        )
        try:
            while not self._shutdown.is_set():
                had_work = False
                claim_consumer = self.consumers[self._claim_index % len(self.consumers)]
                self._claim_index += 1
                await claim_consumer.maybe_claim_stale()
                for consumer in self.consumers:
                    if self._shutdown.is_set():
                        break
                    if consumer._semaphore.locked():
                        continue
                    messages = await consumer._read_or_recover(block_ms=0)
                    if not messages:
                        continue
                    had_work = True
                    task = asyncio.create_task(consumer.dispatch_messages(messages))
                    self._inflight.add(task)
                    task.add_done_callback(self._inflight.discard)
                if had_work:
                    self._idle.hit()
                else:
                    await asyncio.sleep(self._idle.miss())
        finally:
            self._shutdown.set()
            for consumer in self.consumers:
                await consumer.stop()
            if self._inflight:
                await asyncio.wait(self._inflight, timeout=1)
            for consumer in self.consumers:
                await consumer.drain()


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


def _message_key(message_id: Any) -> str:
    if isinstance(message_id, bytes):
        return message_id.decode("utf-8", errors="replace")
    return str(message_id)


async def _still_owned(lease) -> bool:
    try:
        return await renew_lease(lease.redis, lease)
    except (RedisError, OSError, LeaseUnavailable):
        lease.lost = True
        lease.reason = "ownership_uncertain"
        return False


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
