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
)
from services.queue.streams import EventEnvelope, NonRetryableQueueError


async def handle_goal_event(event: EventEnvelope) -> None:
    await GoalJobHandler(get_database()).handle(event)


async def run_goal_scheduler() -> None:
    from services.goals.reviews import enqueue_due_reviews
    from services.queue.shared_redis import disconnect_idle_pool_connections

    while True:
        try:
            database = get_database()
            await enqueue_due_reviews(database)
        except Exception as error:
            print("goal scheduler failed", {"error": type(error).__name__}, flush=True)
        try:
            await disconnect_idle_pool_connections()
        except Exception:
            pass
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
        self._semaphore = asyncio.Semaphore(max(1, min(settings.GOAL_CLARIFY_MAX_CONCURRENCY, 3)))
        self._capacity_failures = 0

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
            try:
                # Non-blocking reads so this loop never parks a cloud socket.
                messages = await self.redis.xreadgroup(
                    groupname=settings.REDIS_GOAL_INTERACTIVE_GROUP,
                    consumername=self.consumer_name,
                    streams={settings.REDIS_GOAL_INTERACTIVE_STREAM: ">"},
                    count=2,
                    block=0,
                )
                job_messages = None
                if not messages:
                    job_messages = await self.redis.xreadgroup(
                        groupname=settings.REDIS_GOAL_JOBS_GROUP,
                        consumername=self.consumer_name,
                        streams={settings.REDIS_GOAL_JOBS_STREAM: ">"},
                        count=1,
                        block=0,
                    )
                self._capacity_failures = 0
            except ResponseError as error:
                if "NOGROUP" in str(error):
                    await self.ensure_groups()
                    continue
                if is_redis_capacity_error(error):
                    self._capacity_failures += 1
                    await recover_redis_capacity(self.redis)
                    await asyncio.sleep(capacity_backoff_seconds(self._capacity_failures))
                    continue
                raise
            except (MaxConnectionsError, RedisTimeoutError, ConnectionError) as error:
                if is_redis_capacity_error(error) or isinstance(error, MaxConnectionsError):
                    self._capacity_failures += 1
                    print("goal redis capacity", {"error": type(error).__name__}, flush=True)
                    await recover_redis_capacity(self.redis)
                    await asyncio.sleep(capacity_backoff_seconds(self._capacity_failures))
                    continue
                print("goal redis read failed", {"error": type(error).__name__}, flush=True)
                await asyncio.sleep(1)
                continue

            batches = []
            if messages:
                batches.extend(messages)
            if job_messages:
                batches.extend(job_messages)
            tasks = []
            for stream, entries in batches:
                group = self.streams[stream]
                for message_id, fields in entries:
                    tasks.append(asyncio.create_task(self._handle_message(stream, group, message_id, fields)))
            if tasks:
                await asyncio.gather(*tasks)
            else:
                await asyncio.sleep(0.12)


def build_goal_consumers():
    """Return a single runner object with ensure_group + run_forever for main.py."""
    return [GoalStreamsRunner()]
