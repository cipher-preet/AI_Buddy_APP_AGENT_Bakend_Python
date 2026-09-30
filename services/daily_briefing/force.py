from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from apps.api_gateway.config.setting import settings
from services.daily_briefing.jobs import DailyBriefingJobHandler
from services.daily_briefing.log import briefing_log
from services.daily_briefing.scheduler import resolve_user_timezone
from services.daily_briefing.schemas import BriefingStatus
from services.daily_briefing.store import DailyBriefingStore, is_stale_for_requeue
from services.daily_briefing.timezones import (
    date_key_for,
    local_day_bounds_utc,
    previous_date_key,
)
from services.queue.streams import EventEnvelope, RedisStreamProducer


async def _run_force_job(database, event: EventEnvelope) -> None:
    try:
        handler = DailyBriefingJobHandler(database)
        await handler.handle(event)
    except Exception as error:
        briefing_log(
            "daily_briefing_force_failed",
            userId=event.userId,
            dateKey=str((event.payload or {}).get("dateKey") or ""),
            jobId=event.eventId,
            error=type(error).__name__,
        )


async def _dispatch(database, event: EventEnvelope) -> None:
    """Queue to the briefing worker (it holds the LLM keys); run in-process only if publishing fails."""
    try:
        await RedisStreamProducer().publish(settings.REDIS_DAILY_BRIEFING_STREAM, event)
    except Exception as error:
        briefing_log(
            "daily_briefing_force_publish_failed",
            userId=event.userId,
            jobId=event.eventId,
            error=f"{type(error).__name__}: {str(error)[:200]}",
        )
        asyncio.create_task(_run_force_job(database, event))


async def force_generate_daily_briefing(
    database,
    *,
    user_id: str,
    date_key: str | None = None,
    period: str = "yesterday",
) -> dict:
    """
    Temporary test helper: delete any existing briefing for the target day,
    reserve PENDING, then run the normal job handler in the background.

    period:
      - today: use current local date (best for manual testing)
      - yesterday: match production scheduler behavior
    """
    if not settings.DAILY_BRIEFING_ALLOW_FORCE_GENERATE:
        raise PermissionError("Force daily briefing is disabled.")

    user = await database.users.find_one({"_id": user_id}, {"_id": 1, "timezone": 1})
    if user is None:
        try:
            from bson import ObjectId

            if ObjectId.is_valid(user_id):
                user = await database.users.find_one(
                    {"_id": ObjectId(user_id)},
                    {"_id": 1, "timezone": 1},
                )
        except Exception:
            user = None
    if user is None:
        # Briefings are keyed by string userId even if the users collection differs.
        user = {"_id": user_id}

    timezone_name = await resolve_user_timezone(database, user)
    now = datetime.now(timezone.utc)
    if date_key:
        target_date = date_key
    elif period == "yesterday":
        target_date = previous_date_key(now, timezone_name)
    else:
        target_date = date_key_for(now, timezone_name)

    period_start, period_end = local_day_bounds_utc(target_date, timezone_name)
    store = DailyBriefingStore(database)
    existing = await store.get(user_id, target_date)
    status = (existing or {}).get("status")
    if status == BriefingStatus.READY.value or (
        status in {BriefingStatus.PENDING.value, BriefingStatus.PROCESSING.value}
        and not is_stale_for_requeue(existing)
    ):
        return {
            "userId": user_id,
            "dateKey": target_date,
            "timezone": timezone_name,
            "status": status,
            "forced": False,
            "message": "Briefing already exists or is being generated.",
        }
    force_count = int((existing or {}).get("forceCount") or 0)
    if force_count >= settings.DAILY_BRIEFING_FORCE_DAILY_LIMIT:
        raise PermissionError("Daily briefing regenerate limit reached for this day.")
    if existing is not None and status != BriefingStatus.FAILED.value:
        await store.delete(user_id, target_date)
    reserved = await store.reserve(
        user_id,
        target_date,
        timezone_name,
        period_start,
        period_end,
    )
    if not reserved:
        raise RuntimeError("Could not reserve daily briefing for force generate.")
    await store.set_force_count(user_id, target_date, force_count + 1)

    event = EventEnvelope(
        eventType="daily.briefing.requested",
        correlationId=f"daily-briefing-force:{user_id}:{target_date}",
        userId=user_id,
        spaceId="daily-briefing",
        conversationId=f"daily-briefing:{user_id}:{target_date}",
        payload={
            "dateKey": target_date,
            "timezone": timezone_name,
            "periodStartUtc": period_start.isoformat(),
            "periodEndUtc": period_end.isoformat(),
            "forced": True,
        },
    )
    briefing_log(
        "daily_briefing_force_started",
        userId=user_id,
        dateKey=target_date,
        timezone=timezone_name,
        jobId=event.eventId,
        period=period,
    )
    await _dispatch(database, event)
    return {
        "userId": user_id,
        "dateKey": target_date,
        "timezone": timezone_name,
        "status": "PENDING",
        "forced": True,
        "message": "Briefing generation started. Poll getDailyBriefing until READY.",
    }
