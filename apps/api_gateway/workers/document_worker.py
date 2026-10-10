from __future__ import annotations

import os
import socket

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


def _stable_consumer_name(prefix: str) -> str:
    host = (socket.gethostname() or "worker").split(".")[0][:24]
    pid = os.getpid()
    return f"{prefix}-{host}-{pid}"


def build_document_consumer() -> RedisStreamConsumer:
    return RedisStreamConsumer(
        stream=settings.REDIS_DOCUMENT_STREAM,
        group=settings.REDIS_DOCUMENT_GROUP,
        consumer_name=_stable_consumer_name("document"),
        handler=handle_document_event,
        concurrency=settings.DOCUMENT_MAX_CONCURRENCY,
        max_retries=settings.DOCUMENT_MAX_RETRIES,
        redis=get_mindmap_redis_client(),
        delete_after_ack=True,
        stream_maxlen=settings.REDIS_DOCUMENT_STREAM_MAXLEN,
    )
