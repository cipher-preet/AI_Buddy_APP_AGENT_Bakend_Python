from __future__ import annotations

import asyncio
from uuid import uuid4

from apps.api_gateway.config.setting import settings
from services.daily_briefing.jobs import DailyBriefingJobHandler
from services.daily_briefing.scheduler import DailyBriefingScheduler
from services.db.mongo import get_database
from services.observability.diagnostics import active_jobs, diag_log
from services.queue.streams import EventEnvelope, RedisStreamConsumer


async def handle_daily_briefing_event(event: EventEnvelope) -> None:
    handler = DailyBriefingJobHandler(get_database())
    with active_jobs.track("briefing"):
        await handler.handle(event)


def build_daily_briefing_consumer() -> RedisStreamConsumer:
    return RedisStreamConsumer(
        stream=settings.REDIS_DAILY_BRIEFING_STREAM,
        group=settings.REDIS_DAILY_BRIEFING_GROUP,
        consumer_name=f"daily-briefing-{uuid4().hex[:8]}",
        handler=handle_daily_briefing_event,
        concurrency=settings.DAILY_BRIEFING_MAX_CONCURRENCY,
        max_retries=settings.DAILY_BRIEFING_MAX_RETRIES,
    )


async def run_daily_briefing_scheduler() -> None:
    scheduler = DailyBriefingScheduler(get_database())
    interval = settings.DAILY_BRIEFING_SCAN_INTERVAL_SECONDS
    while True:
        try:
            await scheduler.scan_once()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            diag_log(
                "daily_briefing_scan",
                errors=1,
                exception_type=type(error).__name__,
            )
            print(f"Daily briefing scheduler failed: {error}", flush=True)
        await asyncio.sleep(interval)
