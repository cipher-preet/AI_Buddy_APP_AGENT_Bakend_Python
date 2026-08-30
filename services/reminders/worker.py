from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from services.reminders.dispatcher import DeliveryError, ReminderDeliveryDispatcher
from services.reminders.occurrence import (
    TriggerEvent,
    compute_next_trigger,
    occurrence_id,
    parse_occurrence_id,
    parse_occurrence_stamp,
    to_occurrence_key,
)
from services.reminders.schedule_store import ReminderRedisKeys

UTC = timezone.utc
logger = logging.getLogger("buddy.reminders")

RETRY_DELAYS_SECONDS = (5, 30, 120)


def _log(event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    logger.info(json.dumps(payload, default=str))


def _as_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    return None


def _user_id(value: Any) -> str:
    return str(value)


@dataclass
class ReminderWorkerConfig:
    lookahead_seconds: int = 3600
    poll_ms: int = 1000
    late_grace_seconds: int = 300
    max_retries: int = 4
    claim_ttl_seconds: int = 60
    batch_size: int = 50


class ReminderWorker:
    def __init__(
        self,
        schedule_store,
        reminder_store,
        dispatcher: ReminderDeliveryDispatcher,
        config: ReminderWorkerConfig | None = None,
        now_fn: Callable[[], datetime] | None = None,
        keys: ReminderRedisKeys | None = None,
    ):
        self.schedule = schedule_store
        self.reminders = reminder_store
        self.dispatcher = dispatcher
        self.config = config or ReminderWorkerConfig()
        self.now_fn = now_fn or (lambda: datetime.now(UTC))
        self.keys = keys or ReminderRedisKeys()

    async def recover(self) -> None:
        now = self.now_fn()
        until = now + timedelta(seconds=self.config.lookahead_seconds)
        upcoming = await self.reminders.upcoming(until)
        for reminder in upcoming:
            nxt = _as_datetime(reminder.get("nextTriggerAtUtc"))
            reminder_id = str(reminder.get("_id"))
            if nxt is None:
                continue
            member = reminder.get("scheduledOccurrenceId") or occurrence_id(reminder_id, nxt)
            score = int(nxt.timestamp())
            already = False
            if hasattr(self.schedule, "has_scheduled"):
                already = await self.schedule.has_scheduled(member)
            if not already:
                await self.schedule.schedule(member, score)
                _log(
                    "reminder_recovered",
                    reminderId=reminder_id,
                    userId=_user_id(reminder.get("userId")),
                    occurrenceId=member,
                    reminderType=reminder.get("deliveryType"),
                    scheduledTime=to_occurrence_key(nxt),
                )

        expired = await self.schedule.expired_processing(int(now.timestamp()))
        for member in expired:
            parsed = parse_occurrence_id(member)
            if not parsed:
                await self.schedule.complete(member)
                continue
            stamp = parse_occurrence_stamp(parsed[1])
            score = int(stamp.timestamp()) if stamp else int(now.timestamp())
            await self.schedule.requeue(member, score)

    async def tick(self) -> int:
        now = self.now_fn()
        now_ts = int(now.timestamp())
        processed = 0
        due = await self.schedule.due_members(
            self.keys.schedule,
            now_ts,
            self.config.batch_size,
        )
        retry_due = await self.schedule.due_members(
            self.keys.retry,
            now_ts,
            self.config.batch_size,
        )
        claim_until = now_ts + self.config.claim_ttl_seconds
        for member in [*due, *retry_due]:
            zset = self.keys.retry if member in retry_due else self.keys.schedule
            claimed = await self.schedule.claim(zset, member, now_ts, claim_until)
            if not claimed:
                _log("reminder_duplicate_skipped", occurrenceId=member, reason="redis_claim")
                continue
            await self._execute(member, now)
            processed += 1
        return processed

    async def _execute(self, member: str, now: datetime) -> None:
        parsed = parse_occurrence_id(member)
        if not parsed:
            _log("reminder_delivery_failed", occurrenceId=member, reason="malformed_occurrence")
            await self.schedule.dead_letter_push(member)
            await self.schedule.complete(member)
            return

        reminder_id, stamp = parsed
        occurrence_at = parse_occurrence_stamp(stamp)
        if occurrence_at is None:
            _log("reminder_delivery_failed", reminderId=reminder_id, occurrenceId=member, reason="malformed_time")
            await self.schedule.dead_letter_push(member)
            await self.schedule.complete(member)
            return

        reminder = await self.reminders.get_by_id(reminder_id)
        if reminder is None:
            _log("reminder_duplicate_skipped", reminderId=reminder_id, occurrenceId=member, reason="deleted")
            await self.schedule.complete(member)
            return

        status = reminder.get("deliveryStatus")
        if status in {"CANCELLED", "FAILED"}:
            _log("reminder_duplicate_skipped", reminderId=reminder_id, occurrenceId=member, reason=status)
            await self.schedule.complete(member)
            return

        scheduled_id = reminder.get("scheduledOccurrenceId")
        if scheduled_id and scheduled_id != member:
            _log(
                "reminder_duplicate_skipped",
                reminderId=reminder_id,
                occurrenceId=member,
                reason="stale_schedule",
            )
            await self.schedule.complete(member)
            return

        if reminder.get("lastDeliveredOccurrenceKey") == member:
            _log("reminder_duplicate_skipped", reminderId=reminder_id, occurrenceId=member, reason="already_delivered")
            await self.schedule.complete(member)
            return

        lateness = max(0, int((now - occurrence_at).total_seconds()))
        if lateness > self.config.late_grace_seconds:
            await self.reminders.mark_failed(reminder_id, now)
            await self.schedule.complete(member)
            _log(
                "reminder_delivery_failed",
                reminderId=reminder_id,
                userId=_user_id(reminder.get("userId")),
                occurrenceId=member,
                reminderType=reminder.get("deliveryType"),
                scheduledTime=stamp,
                actualTriggerTime=to_occurrence_key(now),
                lateness=lateness,
                reason="late_grace_exceeded",
            )
            return

        claimed = await self.reminders.claim(reminder_id, member, now)
        if claimed is None:
            _log("reminder_duplicate_skipped", reminderId=reminder_id, occurrenceId=member, reason="mongo_claim")
            await self.schedule.complete(member)
            return

        _log(
            "reminder_claimed",
            reminderId=reminder_id,
            userId=_user_id(claimed.get("userId")),
            occurrenceId=member,
            reminderType=claimed.get("deliveryType"),
            scheduledTime=stamp,
            actualTriggerTime=to_occurrence_key(now),
            lateness=lateness,
            attemptNumber=int(claimed.get("retryCount") or 0) + 1,
        )

        delivery_type = claimed.get("deliveryType")
        if delivery_type not in {"NORMAL_NOTIFICATION", "ALARM_NOTIFICATION", "AI_CALL"}:
            await self.reminders.mark_failed(reminder_id, now)
            await self.schedule.dead_letter_push(member)
            await self.schedule.complete(member)
            _log("reminder_delivery_failed", reminderId=reminder_id, occurrenceId=member, reason="invalid_type")
            return

        event = TriggerEvent(
            version=1,
            event_id=member,
            reminder_id=reminder_id,
            user_id=_user_id(claimed.get("userId")),
            occurrence_at_utc=occurrence_at,
            timezone=claimed.get("timezone") or "Asia/Kolkata",
            type=delivery_type,
            title=claimed.get("title") or "",
            message=claimed.get("description") or "",
            created_at=to_occurrence_key(now),
            attempt=int(claimed.get("retryCount") or 0) + 1,
        )
        _log(
            "reminder_triggered",
            reminderId=reminder_id,
            userId=event.user_id,
            occurrenceId=member,
            reminderType=event.type,
            scheduledTime=stamp,
            actualTriggerTime=to_occurrence_key(now),
            lateness=lateness,
            attemptNumber=event.attempt,
        )
        try:
            await self.dispatcher.dispatch(event)
        except DeliveryError as error:
            await self._handle_failure(claimed, member, now, error, lateness)
            return
        except Exception as error:  # noqa: BLE001
            _log(
                "reminder_delivery_failed",
                reminderId=reminder_id,
                occurrenceId=member,
                reason="worker_exception",
                message=str(error),
            )
            await self._handle_failure(
                claimed,
                member,
                now,
                DeliveryError(str(error), retryable=True),
                lateness,
            )
            return

        await self._handle_success(claimed, member, occurrence_at, now, lateness)

    async def _handle_success(
        self,
        reminder: dict[str, Any],
        member: str,
        occurrence_at: datetime,
        now: datetime,
        lateness: int,
    ) -> None:
        reminder_id = str(reminder.get("_id"))
        repeat = reminder.get("repeat") or "once"
        next_trigger = None
        next_id = None
        status = "DELIVERED"
        if repeat != "once":
            next_trigger = compute_next_trigger(
                reminder.get("dateKey") or "",
                reminder.get("timeLabel") or "",
                reminder.get("timezone"),
                repeat,
                occurrence_at,
            )
            if next_trigger:
                next_id = occurrence_id(reminder_id, next_trigger)
                await self.schedule.schedule(next_id, int(next_trigger.timestamp()))
                status = "SCHEDULED"
        await self.reminders.mark_delivered(
            reminder_id,
            member,
            now,
            next_trigger,
            next_id,
            status,
        )
        await self.schedule.complete(member)
        _log(
            "reminder_delivery_success",
            reminderId=reminder_id,
            userId=_user_id(reminder.get("userId")),
            occurrenceId=member,
            reminderType=reminder.get("deliveryType"),
            scheduledTime=to_occurrence_key(occurrence_at),
            actualTriggerTime=to_occurrence_key(now),
            lateness=lateness,
            attemptNumber=int(reminder.get("retryCount") or 0) + 1,
        )

    async def _handle_failure(
        self,
        reminder: dict[str, Any],
        member: str,
        now: datetime,
        error: DeliveryError,
        lateness: int,
    ) -> None:
        reminder_id = str(reminder.get("_id"))
        retry_count = int(reminder.get("retryCount") or 0) + 1
        if error.retryable and retry_count < self.config.max_retries:
            delay = RETRY_DELAYS_SECONDS[min(retry_count - 1, len(RETRY_DELAYS_SECONDS) - 1)]
            retry_at = now + timedelta(seconds=delay)
            await self.reminders.mark_retry(reminder_id, retry_count, retry_at, now)
            await self.schedule.retry_at(member, int(retry_at.timestamp()))
            _log(
                "reminder_delivery_retry",
                reminderId=reminder_id,
                userId=_user_id(reminder.get("userId")),
                occurrenceId=member,
                reminderType=reminder.get("deliveryType"),
                attemptNumber=retry_count + 1,
                lateness=lateness,
                message=str(error),
            )
            return

        await self.reminders.mark_failed(reminder_id, now)
        await self.schedule.dead_letter_push(member)
        await self.schedule.complete(member)
        _log(
            "reminder_delivery_failed",
            reminderId=reminder_id,
            userId=_user_id(reminder.get("userId")),
            occurrenceId=member,
            reminderType=reminder.get("deliveryType"),
            attemptNumber=retry_count,
            lateness=lateness,
            message=str(error),
        )


async def run_reminder_worker_forever(worker: ReminderWorker) -> None:
    import asyncio

    await worker.recover()
    poll_seconds = max(worker.config.poll_ms, 200) / 1000
    while True:
        try:
            await worker.tick()
        except Exception as error:  # noqa: BLE001
            _log("reminder_worker_tick_failed", message=str(error))
        await asyncio.sleep(poll_seconds)
