from __future__ import annotations

from uuid import uuid4

from apps.api_gateway.config.setting import settings
from services.db.mongo import get_database
from services.document_it.jobs import DocumentJobHandler
from services.mindmap.redis_client import get_mindmap_redis_client
from services.observability.diagnostics import active_jobs
from services.queue.streams import EventEnvelope, RedisStreamConsumer


async def handle_document_event(event: EventEnvelope) -> None:
    handler = DocumentJobHandler(get_database())
    with active_jobs.track("document"):
        await handler.handle(event)


def build_document_consumer() -> RedisStreamConsumer:
    return RedisStreamConsumer(
        stream=settings.REDIS_DOCUMENT_STREAM,
        group=settings.REDIS_DOCUMENT_GROUP,
        consumer_name=f"document-{uuid4().hex[:8]}",
        handler=handle_document_event,
        concurrency=settings.DOCUMENT_MAX_CONCURRENCY,
        max_retries=settings.DOCUMENT_MAX_RETRIES,
        redis=get_mindmap_redis_client(),
        delete_after_ack=True,
    )
