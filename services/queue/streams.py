from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field
from redis.exceptions import ResponseError

from apps.api_gateway.config.setting import settings
from services.queue.factory import get_message_publisher, use_pubsub
from services.queue.pubsub import topic_for_orchestration, topic_for_speech
from services.queue.redis_queue import redis_client


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


class RedisStreamProducer:
    async def publish(self, stream: str, event: EventEnvelope) -> str:
        if use_pubsub():
            await get_message_publisher().publish(
                _topic_for_stream(stream),
                event.model_dump(mode="json"),
                attributes={
                    "event_type": event.eventType,
                    "user_id": event.userId,
                    "space_id": event.spaceId,
                    "request_id": event.correlationId,
                    "source": stream,
                },
            )
            return event.eventId
        return await redis_client.xadd(stream, {"event": event.model_dump_json()})


def _topic_for_stream(stream: str) -> str:
    if stream in {settings.REDIS_AUDIO_STREAM, settings.REDIS_STT_STREAM}:
        return topic_for_speech()
    if stream in {settings.REDIS_FINALIZATION_STREAM, settings.REDIS_PROCESSING_STREAM}:
        return topic_for_orchestration()
    raise ValueError(f"No Pub/Sub topic configured for Redis stream replacement: {stream}")


class RedisStreamConsumer:
    def __init__(
        self,
        stream: str,
        group: str,
        consumer_name: str,
        handler: Callable[[EventEnvelope], Awaitable[None]],
        concurrency: int | None = None,
        max_retries: int | None = None,
    ):
        self.stream = stream
        self.group = group
        self.consumer_name = consumer_name
        self.handler = handler
        self.concurrency = concurrency or settings.WORKER_CONCURRENCY
        self.max_retries = max_retries or settings.WORKER_MAX_RETRIES
        self._shutdown = asyncio.Event()
        self._semaphore = asyncio.Semaphore(self.concurrency)

    async def ensure_group(self) -> None:
        try:
            await redis_client.xgroup_create(self.stream, self.group, id="0", mkstream=True)
        except ResponseError as error:
            if "BUSYGROUP" not in str(error):
                raise

    async def stop(self) -> None:
        self._shutdown.set()

    async def run_forever(self) -> None:
        await self.ensure_group()
        while not self._shutdown.is_set():
            await self.claim_stale()
            try:
                messages = await redis_client.xreadgroup(
                    groupname=self.group,
                    consumername=self.consumer_name,
                    streams={self.stream: ">"},
                    count=settings.REDIS_BATCH_SIZE,
                    block=settings.REDIS_BLOCK_MS,
                )
            except ResponseError as error:
                if "NOGROUP" not in str(error):
                    raise
                await self.ensure_group()
                continue
            tasks: list[asyncio.Task[None]] = []
            for _, entries in messages or []:
                for message_id, fields in entries:
                    if self._shutdown.is_set():
                        break
                    await self._semaphore.acquire()
                    tasks.append(asyncio.create_task(self._handle_message(message_id, fields)))
            if tasks:
                await asyncio.gather(*tasks)

    async def claim_stale(self) -> None:
        try:
            claimed = await redis_client.xautoclaim(
                self.stream,
                self.group,
                self.consumer_name,
                min_idle_time=settings.REDIS_CLAIM_IDLE_MS,
                start_id="0-0",
                count=settings.REDIS_BATCH_SIZE,
            )
        except ResponseError:
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
            await redis_client.xack(self.stream, self.group, message_id)
        except Exception as error:
            await self._handle_failure(message_id, fields, error)
        finally:
            self._semaphore.release()

    async def _handle_failure(self, message_id: str, fields: dict[str, Any], error: Exception) -> None:
        try:
            event = EventEnvelope.model_validate_json(fields["event"])
        except Exception:
            await redis_client.xadd(
                settings.REDIS_DEAD_LETTER_STREAM,
                {"event": json.dumps({"raw": fields, "error": str(error)})},
            )
            await redis_client.xack(self.stream, self.group, message_id)
            return

        if event.attempt >= self.max_retries:
            await redis_client.xadd(
                settings.REDIS_DEAD_LETTER_STREAM,
                {
                    "event": event.model_dump_json(),
                    "error": str(error),
                    "sourceStream": self.stream,
                },
            )
            await redis_client.xack(self.stream, self.group, message_id)
            return

        retry = event.model_copy(update={"attempt": event.attempt + 1})
        await redis_client.xadd(
            settings.REDIS_RETRY_STREAM,
            {
                "event": retry.model_dump_json(),
                "targetStream": self.stream,
                "notBefore": str(datetime.now(timezone.utc).timestamp() + retry_delay(event.attempt)),
            },
        )
        await redis_client.xack(self.stream, self.group, message_id)


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
