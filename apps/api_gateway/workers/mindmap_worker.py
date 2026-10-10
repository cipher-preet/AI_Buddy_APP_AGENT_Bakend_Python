from __future__ import annotations

import os
import socket

from apps.api_gateway.config.setting import settings
from services.db.mongo import get_database
from services.mindmap.jobs import MindmapJobHandler
from services.mindmap.redis_client import get_mindmap_redis_client, test_mindmap_redis_connection
from services.observability.diagnostics import active_jobs
from services.queue.streams import EventEnvelope, RedisStreamConsumer


async def handle_mindmap_event(event: EventEnvelope) -> None:
    handler = MindmapJobHandler(get_database())
    with active_jobs.track("mindmap"):
        await handler.handle(event)


def _stable_consumer_name(prefix: str) -> str:
    # Stable names avoid orphan consumers piling up on Redis Cloud after restarts.
    host = (socket.gethostname() or "worker").split(".")[0][:24]
    pid = os.getpid()
    return f"{prefix}-{host}-{pid}"


def build_mindmap_consumer() -> RedisStreamConsumer:
    return RedisStreamConsumer(
        stream=settings.REDIS_MINDMAP_STREAM,
        group=settings.REDIS_MINDMAP_GROUP,
        consumer_name=_stable_consumer_name("mindmap"),
        handler=handle_mindmap_event,
        concurrency=settings.MINDMAP_MAX_CONCURRENCY,
        max_retries=settings.MINDMAP_MAX_RETRIES,
        redis=get_mindmap_redis_client(),
        delete_after_ack=True,
        stream_maxlen=settings.REDIS_MINDMAP_STREAM_MAXLEN,
    )


async def ensure_mindmap_redis() -> None:
    await test_mindmap_redis_connection()
