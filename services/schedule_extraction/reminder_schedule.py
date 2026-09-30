"""Python port of Node's buildScheduleFields/upsertReminderSchedule so AI-created reminders fire identically."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Any

from apps.api_gateway.config.setting import settings
from services.reminders.occurrence import (
    DEFAULT_TIMEZONE,
    compute_next_trigger,
    delivery_type_from_flags,
    occurrence_id,
    to_occurrence_key,
)
from services.reminders.schedule_store import ReminderRedisKeys

REDIS_ATTEMPTS = 3


def build_schedule_fields(
    *,
    date_key: str,
    time_label: str,
    repeat: str,
    timezone_name: str | None,
    ai_calling: bool,
    beeping: bool,
    notification: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = now or datetime.now(timezone.utc)
    zone = (timezone_name or "").strip() or DEFAULT_TIMEZONE
    fields: dict[str, Any] = {
        "timezone": zone,
        "deliveryType": delivery_type_from_flags(ai_calling, beeping, notification),
        "deliveryStatus": "CANCELLED",
        "nextTriggerAtUtc": None,
        "scheduledOccurrenceId": None,
        "retryCount": 0,
        "retryAtUtc": None,
    }
    if not fields["deliveryType"]:
        return fields
    grace = timedelta(seconds=settings.REMINDER_LATE_GRACE_SECONDS)
    nxt = compute_next_trigger(date_key, time_label, zone, repeat, current - grace)
    fields["nextTriggerAtUtc"] = nxt
    if nxt is None or (repeat == "once" and nxt < current - grace):
        fields["deliveryStatus"] = "FAILED"
        return fields
    fields["deliveryStatus"] = "SCHEDULED"
    return fields


class ReminderScheduler:
    def __init__(self, redis=None, keys: ReminderRedisKeys | None = None):
        self._redis = redis
        self.keys = keys or ReminderRedisKeys(
            schedule=settings.REMINDER_SCHEDULE_KEY,
            processing=settings.REMINDER_PROCESSING_KEY,
            retry=settings.REMINDER_RETRY_KEY,
            dead_letter=settings.REMINDER_DEAD_LETTER_KEY,
        )

    @property
    def redis(self):
        if self._redis is None:
            from services.reminders.redis_client import get_reminder_redis_client

            self._redis = get_reminder_redis_client()
        return self._redis

    @staticmethod
    def occurrence_for(reminder_id: str, fields: dict[str, Any]) -> str | None:
        nxt = fields.get("nextTriggerAtUtc")
        if fields.get("deliveryStatus") != "SCHEDULED" or not isinstance(nxt, datetime):
            return None
        return occurrence_id(reminder_id, nxt)

    async def ensure_scheduled(self, reminder: dict[str, Any]) -> bool:
        """Idempotently put the reminder's current occurrence on the Redis schedule."""
        member = reminder.get("scheduledOccurrenceId")
        nxt = reminder.get("nextTriggerAtUtc")
        if reminder.get("deliveryStatus") != "SCHEDULED" or not member or not isinstance(nxt, datetime):
            return False
        if nxt.tzinfo is None:
            nxt = nxt.replace(tzinfo=timezone.utc)
        payload = json.dumps(
            {
                "version": 1,
                "eventId": member,
                "reminderId": str(reminder["_id"]),
                "userId": str(reminder.get("userId")),
                "occurrenceAtUtc": to_occurrence_key(nxt),
                "timezone": reminder.get("timezone") or DEFAULT_TIMEZONE,
                "type": reminder.get("deliveryType"),
                "title": reminder.get("title") or "",
                "message": reminder.get("description") or "",
                "createdAt": datetime.now(timezone.utc).isoformat(),
            }
        )
        last_error: Exception | None = None
        for attempt in range(REDIS_ATTEMPTS):
            try:
                pipe = self.redis.pipeline()
                pipe.zadd(self.keys.schedule, {member: int(nxt.timestamp())})
                pipe.set(self.keys.payload(member), payload, ex=settings.REMINDER_PAYLOAD_TTL_SECONDS)
                await pipe.execute()
                return True
            except Exception as error:
                last_error = error
                await asyncio.sleep(0.2 * (attempt + 1))
        raise RuntimeError(f"reminder_schedule_failed occurrence={member}: {last_error}")
