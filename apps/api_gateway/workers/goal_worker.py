from __future__ import annotations

import asyncio
import json
import os
import socket

from redis.exceptions import (
    ConnectionError,
    MaxConnectionsError,
    ResponseError,
    TimeoutError as RedisTimeoutError,
)

from apps.api_gateway.config.setting import settings
from services.db.mongo import get_database
from services.goals.jobs import GoalJobHandler
from services.goals.redis_client import get_goal_redis_client
from services.queue.shared_redis import (
    capacity_backoff_seconds,
    is_redis_capacity_error,
    recover_redis_capacity,
    recycle_broken_redis_pool,
    should_log_redis_timeout,
)
from services.queue.streams import EventEnvelope, IdlePoll, NonRetryableQueueError, xreadgroup_block


async def handle_goal_event(event: EventEnvelope) -> None:
    await GoalJobHandler(get_database()).handle(event)


async def run_goal_scheduler() -> None:
    from services.goals.reviews import enqueue_due_reviews

    while True:
        try:
            database = get_database()
            await enqueue_due_reviews(database)
        except Exception as error:
            print("goal scheduler failed", {"error": type(error).__name__}, flush=True)
        await asyncio.sleep(60)


def _consumer_name() -> str:
    host = (socket.gethostname() or "worker").split(".")[0][:24]
    return f"goal-{host}-{os.getpid()}"


class GoalStreamsRunner:
    """One Redis client loop for both goal streams — avoids a second blocked connection."""

    def __init__(self) -> None:
        self.redis = get_goal_redis_client()
        self.consumer_name = _consumer_name()
        self.stream = "buddy:goals"
        self.streams = {
            settings.REDIS_GOAL_INTERACTIVE_STREAM: settings.REDIS_GOAL_INTERACTIVE_GROUP,
            settings.REDIS_GOAL_JOBS_STREAM: settings.REDIS_GOAL_JOBS_GROUP,
        }
        self.maxlen = settings.REDIS_GOAL_STREAM_MAXLEN
        self._max_active = max(1, min(settings.GOAL_CLARIFY_MAX_CONCURRENCY, 3))
        self._semaphore = asyncio.Semaphore(self._max_active)
        self._capacity_failures = 0
        self._active = 0
        self._inflight: set[asyncio.Task] = set()
        self._last_claim_at = 0.0
        self._timeout_failures = 0
        self._idle = IdlePoll(min_seconds=0.4, max_seconds=2.5)

    async def ensure_group(self) -> None:
        await self.ensure_groups()

    async def ensure_groups(self) -> None:
        for stream, group in self.streams.items():
            try:
                await self.redis.xgroup_create(stream, group, id="0", mkstream=True)
                print("Redis stream consumer group created:", {"stream": stream, "group": group})
            except ResponseError as error:
                if "BUSYGROUP" not in str(error):
                    raise
            if self.maxlen:
                try:
                    await self.redis.xtrim(stream, maxlen=self.maxlen, approximate=False)
                except Exception:
                    pass

    async def _ack(self, stream: str, group: str, message_id: str) -> None:
        await self.redis.xack(stream, group, message_id)
        try:
            await self.redis.xdel(stream, message_id)
        except Exception:
            pass
        if self.maxlen:
            try:
                await self.redis.xtrim(stream, maxlen=self.maxlen, approximate=True)
            except Exception:
                pass

    async def _handle_message(self, stream: str, group: str, message_id: str, fields: dict) -> None:
        async with self._semaphore:
            raw = fields.get("event") or "{}"
            try:
                event = EventEnvelope.model_validate_json(raw if isinstance(raw, str) else json.dumps(raw))
            except Exception as error:
                print("goal event invalid", {"error": type(error).__name__}, flush=True)
                await self._ack(stream, group, message_id)
                return
            try:
                await handle_goal_event(event)
                await self._ack(stream, group, message_id)
            except NonRetryableQueueError as error:
                print("goal event non-retryable", {"error": str(error)}, flush=True)
                await self._ack(stream, group, message_id)
            except Exception as error:
                print("goal event failed", {"error": type(error).__name__}, flush=True)

    def _spawn(self, stream: str, group: str, message_id: str, fields: dict) -> None:
        self._active += 1

        async def run() -> None:
            try:
                await self._handle_message(stream, group, message_id, fields)
            finally:
                self._active = max(0, self._active - 1)

        task = asyncio.create_task(run())
        self._inflight.add(task)
        task.add_done_callback(self._inflight.discard)

    async def _read_stream(self, stream: str, group: str, count: int):
        try:
            options: dict = {
                "groupname": group,
                "consumername": self.consumer_name,
                "streams": {stream: ">"},
                "count": count,
            }
            block = xreadgroup_block(0)
            if block is not None:
                options["block"] = block
            messages = await self.redis.xreadgroup(**options)
            self._capacity_failures = 0
            self._timeout_failures = 0
            return messages or []
        except ResponseError as error:
            if "NOGROUP" in str(error):
                await self.ensure_groups()
                return []
            if is_redis_capacity_error(error):
                self._capacity_failures += 1
                await recover_redis_capacity(self.redis)
                await asyncio.sleep(capacity_backoff_seconds(self._capacity_failures))
                return []
            raise
        except (MaxConnectionsError, RedisTimeoutError, ConnectionError) as error:
            if is_redis_capacity_error(error) or isinstance(error, MaxConnectionsError):
                self._capacity_failures += 1
                await recover_redis_capacity(self.redis)
                await asyncio.sleep(capacity_backoff_seconds(self._capacity_failures))
                return []
            self._timeout_failures += 1
            if self._timeout_failures >= 3:
                await recycle_broken_redis_pool(self.redis)
                self._timeout_failures = 0
            if should_log_redis_timeout():
                print("goal redis read failed", {"error": type(error).__name__, "stream": stream}, flush=True)
            await asyncio.sleep(min(4.0, 0.8 * max(1, self._timeout_failures or 1)))
            return []

    async def _claim_stale(self) -> None:
        now = asyncio.get_running_loop().time()
        if now - self._last_claim_at < 45:
            return
        self._last_claim_at = now
        idle_ms = min(int(settings.REDIS_CLAIM_IDLE_MS), 10000)
        for stream, group in self.streams.items():
            try:
                claimed = await self.redis.xautoclaim(
                    stream,
                    group,
                    self.consumer_name,
                    min_idle_time=idle_ms,
                    start_id="0-0",
                    count=2,
                )
            except Exception:
                continue
            entries = claimed[1] if len(claimed) > 1 else []
            for message_id, fields in entries:
                self._spawn(stream, group, message_id, fields)

    async def run_forever(self) -> None:
        await self.ensure_groups()
        print(
            "Redis goal streams consumer started:",
            {
                "streams": list(self.streams),
                "consumer": self.consumer_name,
            },
            flush=True,
        )
        while True:
            slots = self._max_active - self._active
            if slots <= 0:
                await asyncio.sleep(0.08)
                continue
            await self._claim_stale()
            interactive = await self._read_stream(
                settings.REDIS_GOAL_INTERACTIVE_STREAM,
                settings.REDIS_GOAL_INTERACTIVE_GROUP,
                min(2, max(1, slots)),
            )
            claimed = sum(len(entries) for _, entries in interactive)
            jobs = []
            if claimed < slots:
                jobs = await self._read_stream(
                    settings.REDIS_GOAL_JOBS_STREAM,
                    settings.REDIS_GOAL_JOBS_GROUP,
                    1,
                )
            spawned = False
            for stream, entries in list(interactive) + list(jobs):
                group = self.streams[stream]
                for message_id, fields in entries:
                    self._spawn(stream, group, message_id, fields)
                    spawned = True
            if spawned:
                self._idle.hit()
            else:
                await asyncio.sleep(self._idle.miss())


def build_goal_consumers():
    """Return a single runner object with ensure_group + run_forever for main.py."""
    return [GoalStreamsRunner()]
