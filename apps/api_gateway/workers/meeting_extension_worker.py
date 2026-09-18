from __future__ import annotations

from uuid import uuid4

from apps.api_gateway.config.setting import settings
from services.meeting_extension.processor import process_meeting_video_chunk
from services.meeting_extension.video_merge import process_meeting_video_merge
from services.observability.diagnostics import active_jobs
from services.queue.streams import EventEnvelope, RedisStreamConsumer


def _tracked(name: str, handler):
    async def wrapped(event: EventEnvelope) -> None:
        with active_jobs.track(name):
            await handler(event)

    wrapped.__name__ = getattr(handler, "__name__", name)
    return wrapped


def build_meeting_video_consumer() -> RedisStreamConsumer | None:
    # Audio/muxed meeting jobs share buddy:stt:jobs with the mobile app.
    # A second consumer group on that stream would duplicate every STT job.
    if settings.REDIS_MEETING_VIDEO_STREAM == settings.REDIS_STT_STREAM:
        return None
    return RedisStreamConsumer(
        stream=settings.REDIS_MEETING_VIDEO_STREAM,
        group=settings.REDIS_MEETING_VIDEO_GROUP,
        consumer_name=f"meeting-video-{uuid4().hex[:8]}",
        handler=_tracked("meeting_video", process_meeting_video_chunk),
        concurrency=settings.MEETING_WORKER_CONCURRENCY or settings.STT_WORKER_CONCURRENCY or settings.WORKER_CONCURRENCY,
    )


def build_meeting_merge_consumer() -> RedisStreamConsumer:
    return RedisStreamConsumer(
        stream=settings.REDIS_MEETING_MERGE_STREAM,
        group=settings.REDIS_MEETING_MERGE_GROUP,
        consumer_name=f"meeting-merge-{uuid4().hex[:8]}",
        handler=_tracked("meeting_merge", process_meeting_video_merge),
        concurrency=1,
    )
