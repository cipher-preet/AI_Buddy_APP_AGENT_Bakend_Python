from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Awaitable, Callable
import time

from apps.api_gateway.config.setting import settings
from services.daily_briefing.log import briefing_log
from services.daily_briefing.store import DailyBriefingStore, _as_utc, is_stale_for_requeue
from services.daily_briefing.timezones import (
    DEFAULT_TIMEZONE,
    is_after_trigger,
    local_day_bounds_utc,
    previous_date_key,
    resolve_zone,
)
from services.observability.diagnostics import diag_log, note_briefing_requeue
from services.queue.streams import EventEnvelope, RedisStreamProducer

EnqueueFn = Callable[[EventEnvelope], Awaitable[str]]


async def resolve_user_timezone(database, user: dict[str, Any]) -> str:
    timezone_name = str(user.get("timezone") or "").strip()
    if timezone_name:
        try:
            resolve_zone(timezone_name)
            return timezone_name
        except Exception:
            pass
    reminder = await database.reminders.find_one(
        {"userId": user.get("_id")},
        {"timezone": 1},
        sort=[("updatedAt", -1), ("_id", -1)],
    )
    reminder_zone = str((reminder or {}).get("timezone") or "").strip()
    if reminder_zone:
        try:
            resolve_zone(reminder_zone)
            return reminder_zone
        except Exception:
            pass
    return DEFAULT_TIMEZONE


def _processing_age_seconds(existing: dict[str, Any] | None, now: datetime) -> int | None:
    if not existing:
        return None
    stamp = _as_utc(existing.get("claimedAt") or existing.get("updatedAt"))
    if stamp is None:
        return None
    current = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    return max(0, int((current - stamp).total_seconds()))


class DailyBriefingScheduler:
    def __init__(
        self,
        database,
        producer: RedisStreamProducer | None = None,
        enqueue: EnqueueFn | None = None,
        now_factory: Callable[[], datetime] | None = None,
    ):
        self.database = database
        self.store = DailyBriefingStore(database)
        self.producer = producer or RedisStreamProducer()
        self.enqueue = enqueue or self._publish
        self.now_factory = now_factory or (lambda: datetime.now(timezone.utc))

    async def _publish(self, event: EventEnvelope) -> str:
        return await self.producer.publish(settings.REDIS_DAILY_BRIEFING_STREAM, event)

    async def scan_once(self) -> dict[str, int]:
        counts = {
            "scanned": 0,
            "enqueued": 0,
            "skipped": 0,
            "disabled": 0,
            "users_due": 0,
            "failed_requeued": 0,
            "stale_processing_requeued": 0,
            "already_ready_skipped": 0,
            "pending_skipped": 0,
            "processing_skipped": 0,
            "errors": 0,
        }
        started = time.perf_counter()
        if not settings.DAILY_BRIEFING_ENABLED:
            counts["disabled"] = 1
            self._emit_scan(counts, started)
            return counts

        now = self.now_factory()
        last_id = None
        while True:
            query: dict[str, Any] = {}
            if last_id is not None:
                query["_id"] = {"$gt": last_id}
            users = await self.database.users.find(
                query,
                {"_id": 1, "timezone": 1},
            ).sort("_id", 1).limit(settings.DAILY_BRIEFING_BATCH_SIZE).to_list(
                length=settings.DAILY_BRIEFING_BATCH_SIZE
            )
            if not users:
                break
            for user in users:
                counts["scanned"] += 1
                last_id = user["_id"]
                enqueued = await self._consider_user(user, now, counts)
                if enqueued:
                    counts["enqueued"] += 1
                else:
                    counts["skipped"] += 1
            if len(users) < settings.DAILY_BRIEFING_BATCH_SIZE:
                break
        self._emit_scan(counts, started)
        return counts

    def _emit_scan(self, counts: dict[str, int], started: float) -> None:
        duration_ms = int((time.perf_counter() - started) * 1000)
        briefing_log("daily_briefing_scheduler_scan", **{key: counts[key] for key in ("scanned", "enqueued", "skipped", "disabled")})
        diag_log(
            "daily_briefing_scan",
            scan_duration_ms=duration_ms,
            users_scanned=counts["scanned"],
            users_due=counts["users_due"],
            jobs_enqueued=counts["enqueued"],
            failed_requeued=counts["failed_requeued"],
            stale_processing_requeued=counts["stale_processing_requeued"],
            already_ready_skipped=counts["already_ready_skipped"],
            pending_skipped=counts["pending_skipped"],
            processing_skipped=counts["processing_skipped"],
            errors=counts["errors"],
        )
        diag_log(
            "mongo_query_timing",
            operation="daily_briefing_user_scan",
            duration_ms=duration_ms,
            result_count=counts["scanned"],
        )

    async def _consider_user(self, user: dict[str, Any], now: datetime, counts: dict[str, int]) -> bool:
        user_id = str(user["_id"])
        timezone_name = await resolve_user_timezone(self.database, user)
        if not is_after_trigger(
            now,
            timezone_name,
            settings.DAILY_BRIEFING_TRIGGER_HOUR,
            settings.DAILY_BRIEFING_TRIGGER_MINUTE,
            settings.DAILY_BRIEFING_GRACE_MINUTES,
        ):
            return False
        counts["users_due"] += 1
        date_key = previous_date_key(now, timezone_name)
        existing = await self.store.get(user_id, date_key)
        status = (existing or {}).get("status")
        if status in {"READY", "SKIPPED"}:
            counts["already_ready_skipped"] += 1
            return False
        if status in {"PENDING", "PROCESSING"} and not is_stale_for_requeue(existing, now):
            if status == "PENDING":
                counts["pending_skipped"] += 1
            else:
                counts["processing_skipped"] += 1
            return False
        period_start, period_end = local_day_bounds_utc(date_key, timezone_name)
        if status not in {"PENDING", "PROCESSING"}:
            reserved = await self.store.reserve(
                user_id,
                date_key,
                timezone_name,
                period_start,
                period_end,
            )
            if not reserved:
                return False
        event = EventEnvelope(
            eventType="daily.briefing.requested",
            correlationId=f"daily-briefing:{user_id}:{date_key}",
            userId=user_id,
            spaceId="daily-briefing",
            conversationId=f"daily-briefing:{user_id}:{date_key}",
            payload={
                "dateKey": date_key,
                "timezone": timezone_name,
                "periodStartUtc": period_start.isoformat(),
                "periodEndUtc": period_end.isoformat(),
            },
        )
        await self.enqueue(event)
        briefing_log(
            "daily_briefing_enqueued",
            userId=user_id,
            dateKey=date_key,
            timezone=timezone_name,
            jobId=event.eventId,
            requeued=status in {"PENDING", "PROCESSING"},
        )
        if status == "FAILED":
            counts["failed_requeued"] += 1
            diag_log(
                "daily_briefing_failed_requeue",
                user_id=user_id,
                date_key=date_key,
                previous_status=status,
                event_id=event.eventId,
                requeue_count=note_briefing_requeue(user_id, date_key),
            )
        elif status == "PROCESSING":
            counts["stale_processing_requeued"] += 1
            diag_log(
                "daily_briefing_stale_requeue",
                user_id=user_id,
                date_key=date_key,
                processing_age_seconds=_processing_age_seconds(existing, now),
                event_id=event.eventId,
            )
        return True
