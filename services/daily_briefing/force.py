from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from apps.api_gateway.config.setting import settings
from services.daily_briefing.jobs import DailyBriefingJobHandler
from services.daily_briefing.log import briefing_log
from services.daily_briefing.scheduler import resolve_user_timezone
from services.daily_briefing.store import DailyBriefingStore
from services.daily_briefing.timezones import (
    date_key_for,
    local_day_bounds_utc,
    previous_date_key,
)
from services.queue.streams import EventEnvelope


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


async def force_generate_daily_briefing(
    database,
    *,
    user_id: str,
    date_key: str | None = None,
    period: str = "today",
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
    asyncio.create_task(_run_force_job(database, event))
    return {
        "userId": user_id,
        "dateKey": target_date,
        "timezone": timezone_name,
        "status": "PENDING",
        "forced": True,
        "message": "Briefing generation started. Poll getDailyBriefing until READY.",
    }
