from __future__ import annotations

from uuid import uuid4

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


def build_mindmap_consumer() -> RedisStreamConsumer:
    return RedisStreamConsumer(
        stream=settings.REDIS_MINDMAP_STREAM,
        group=settings.REDIS_MINDMAP_GROUP,
        consumer_name=f"mindmap-{uuid4().hex[:8]}",
        handler=handle_mindmap_event,
        concurrency=settings.MINDMAP_MAX_CONCURRENCY,
        max_retries=settings.MINDMAP_MAX_RETRIES,
        redis=get_mindmap_redis_client(),
    )


async def ensure_mindmap_redis() -> None:
    await test_mindmap_redis_connection()
