from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from services.reminders.dispatcher import ReminderDeliveryDispatcher, android_payload
from services.reminders.fcm import RecordingFcmSender
from services.reminders.mongo_store import InMemoryReminderStore
from services.reminders.occurrence import (
    TriggerEvent,
    occurrence_id,
    to_occurrence_key,
    zoned_local_to_utc,
)
from services.reminders.schedule_store import InMemoryScheduleStore
from services.reminders.worker import ReminderWorker, ReminderWorkerConfig

UTC = timezone.utc
NOW = datetime(2026, 8, 30, 10, 0, tzinfo=UTC)


def _reminder(
    reminder_id="rem_123",
    user_id="user_1",
    delivery_type="NORMAL_NOTIFICATION",
    status="SCHEDULED",
    occurrence=None,
    repeat="once",
    **extra,
):
    occurrence = occurrence or NOW
    return {
        "_id": reminder_id,
        "userId": user_id,
        "title": extra.get("title", "Drink water"),
        "description": extra.get("description", "Stay hydrated"),
        "dateKey": "2026-08-30",
        "timeLabel": "3:30 PM",
        "timezone": extra.get("timezone", "Asia/Kolkata"),
        "repeat": repeat,
        "aiCalling": delivery_type == "AI_CALL",
        "beeping": delivery_type == "ALARM_NOTIFICATION",
        "notification": True,
        "deliveryType": delivery_type,
        "deliveryStatus": status,
        "nextTriggerAtUtc": occurrence,
        "scheduledOccurrenceId": extra.get(
            "scheduledOccurrenceId",
            occurrence_id(reminder_id, occurrence),
        ),
        "lastDeliveredOccurrenceKey": extra.get("lastDeliveredOccurrenceKey"),
        "retryCount": extra.get("retryCount", 0),
    }


def _worker(reminder, tokens=None, fcm=None, now=NOW, extra_reminders=None):
    reminders = [reminder, *(extra_reminders or [])]
    store = InMemoryReminderStore(reminders, tokens or {str(reminder["userId"]): ["tok_1"]})
    schedule = InMemoryScheduleStore()
    sender = fcm or RecordingFcmSender()
    dispatcher = ReminderDeliveryDispatcher(sender, store.tokens_for_user)
    worker = ReminderWorker(
        schedule,
        store,
        dispatcher,
        config=ReminderWorkerConfig(late_grace_seconds=300, max_retries=4, claim_ttl_seconds=60),
        now_fn=lambda: now,
    )
    member = reminder["scheduledOccurrenceId"]
    score = int(reminder["nextTriggerAtUtc"].timestamp())
    asyncio.run(schedule.schedule(member, score))
    return worker, store, schedule, sender, member


def test_utc_conversion_asia_kolkata():
    value = zoned_local_to_utc("2026-08-30", "3:30 PM", "Asia/Kolkata")
    assert value is not None
    assert to_occurrence_key(value) == "2026-08-30T10:00:00Z"


def test_one_time_reminder_fires_once():
    reminder = _reminder()
    worker, store, schedule, sender, member = _worker(reminder)
    asyncio.run(worker.tick())
    asyncio.run(worker.tick())
    assert len(sender.sent) == 1
    assert store.reminders["rem_123"]["deliveryStatus"] == "DELIVERED"
    assert member not in schedule.zsets[schedule.keys.schedule]


def test_normal_notification_routing():
    payload = android_payload(
        TriggerEvent(1, "e", "r", "u", NOW, "Asia/Kolkata", "NORMAL_NOTIFICATION", "T", "M", "t")
    )
    assert payload["type"] == "reminder_notification"


def test_alarm_notification_routing():
    reminder = _reminder(delivery_type="ALARM_NOTIFICATION")
    worker, store, schedule, sender, member = _worker(reminder)
    asyncio.run(worker.tick())
    assert sender.sent[0]["payload"]["type"] == "reminder_alarm"
    assert sender.sent[0]["payload"]["sound"] == "true"
    assert sender.sent[0]["high_priority"] is True


def test_ai_call_routing():
    reminder = _reminder(delivery_type="AI_CALL", title="Buddy AI Reminder")
    worker, store, schedule, sender, member = _worker(reminder)
    asyncio.run(worker.tick())
    payload = sender.sent[0]["payload"]
    assert payload["type"] == "ai_reminder_call"
    assert "callId" in payload
    assert "expiresAt" in payload
    assert sender.sent[0]["high_priority"] is True


def test_update_removes_old_schedule():
    schedule = InMemoryScheduleStore()
    old_id = "rem_123:2026-08-30T09:00:00Z"
    new_id = "rem_123:2026-08-30T10:00:00Z"

    async def run():
        await schedule.schedule(old_id, int(NOW.timestamp()) - 60)
        await schedule.cancel(old_id)
        await schedule.schedule(new_id, int(NOW.timestamp()))
        assert await schedule.has_scheduled(old_id) is False
        assert await schedule.has_scheduled(new_id) is True

    asyncio.run(run())


def test_cancelled_reminder_never_fires():
    reminder = _reminder(status="CANCELLED")
    worker, store, schedule, sender, member = _worker(reminder)
    asyncio.run(worker.tick())
    assert sender.sent == []


def test_deleted_reminder_never_fires():
    reminder = _reminder()
    worker, store, schedule, sender, member = _worker(reminder)
    store.reminders.clear()
    asyncio.run(worker.tick())
    assert sender.sent == []


def test_two_workers_race_only_one_succeeds():
    reminder = _reminder()
    store = InMemoryReminderStore([reminder], {"user_1": ["tok_1"]})
    schedule = InMemoryScheduleStore()
    sender = RecordingFcmSender()
    dispatcher = ReminderDeliveryDispatcher(sender, store.tokens_for_user)
    now = NOW
    worker_a = ReminderWorker(schedule, store, dispatcher, now_fn=lambda: now)
    worker_b = ReminderWorker(schedule, store, dispatcher, now_fn=lambda: now)
    member = reminder["scheduledOccurrenceId"]
    asyncio.run(schedule.schedule(member, int(now.timestamp())))

    async def race():
        await asyncio.gather(worker_a.tick(), worker_b.tick())

    asyncio.run(race())
    assert len(sender.sent) == 1


def test_duplicate_redis_event_one_delivery():
    reminder = _reminder()
    worker, store, schedule, sender, member = _worker(reminder)
    asyncio.run(schedule.schedule(member, int(NOW.timestamp())))
    asyncio.run(worker.tick())
    asyncio.run(schedule.schedule(member, int(NOW.timestamp())))
    asyncio.run(worker.tick())
    assert len(sender.sent) == 1


def test_retry_does_not_duplicate_successful_delivery():
    reminder = _reminder()
    fcm = RecordingFcmSender()
    fcm.fail_next = True
    worker, store, schedule, sender, member = _worker(reminder, fcm=fcm)
    asyncio.run(worker.tick())
    assert sender.sent == []
    assert store.reminders["rem_123"]["deliveryStatus"] == "RETRY_PENDING"
    later = NOW + timedelta(seconds=6)
    worker.now_fn = lambda: later
    asyncio.run(worker.tick())
    assert len(sender.sent) == 1
    asyncio.run(worker.tick())
    assert len(sender.sent) == 1


def test_worker_restart_recovery():
    reminder = _reminder()
    store = InMemoryReminderStore([reminder], {"user_1": ["tok_1"]})
    schedule = InMemoryScheduleStore()
    sender = RecordingFcmSender()
    dispatcher = ReminderDeliveryDispatcher(sender, store.tokens_for_user)
    worker = ReminderWorker(schedule, store, dispatcher, now_fn=lambda: NOW)
    asyncio.run(worker.recover())
    assert asyncio.run(schedule.has_scheduled(reminder["scheduledOccurrenceId"]))
    asyncio.run(worker.tick())
    assert len(sender.sent) == 1


def test_redis_reconnect_requeue_expired_claim():
    reminder = _reminder()
    worker, store, schedule, sender, member = _worker(reminder)
    schedule.zsets[schedule.keys.processing][member] = float(int(NOW.timestamp()) - 10)
    schedule.zsets[schedule.keys.schedule].pop(member, None)
    asyncio.run(worker.recover())
    assert asyncio.run(schedule.has_scheduled(member))


def test_missing_device_token_marks_failed():
    reminder = _reminder()
    worker, store, schedule, sender, member = _worker(reminder, tokens={"user_1": []})
    asyncio.run(worker.tick())
    assert sender.sent == []
    assert store.reminders["rem_123"]["deliveryStatus"] == "FAILED"


def test_dry_run_without_tokens_is_permanent_failure():
    from services.reminders.fcm import FcmPermanentError
    from services.reminders.fcm_factory import DryRunFcmSender

    sender = DryRunFcmSender()
    try:
        asyncio.run(sender.send([], {"type": "reminder_notification", "title": "T"}, True))
    except FcmPermanentError as error:
        assert "no device tokens" in str(error)
    else:
        raise AssertionError("expected FcmPermanentError")


def test_already_delivered_is_not_replayed():
    reminder = _reminder(
        status="DELIVERED",
        lastDeliveredOccurrenceKey=occurrence_id("rem_123", NOW),
    )
    worker, store, schedule, sender, member = _worker(reminder)
    asyncio.run(worker.tick())
    assert sender.sent == []


def test_recurring_schedules_next_occurrence():
    reminder = _reminder(repeat="daily")
    worker, store, schedule, sender, member = _worker(reminder)
    asyncio.run(worker.tick())
    nxt = store.reminders["rem_123"]["nextTriggerAtUtc"]
    assert store.reminders["rem_123"]["deliveryStatus"] == "SCHEDULED"
    assert to_occurrence_key(nxt) == "2026-08-31T10:00:00Z"
    assert asyncio.run(schedule.has_scheduled(occurrence_id("rem_123", nxt)))


def test_stale_redis_payload_rejected_after_db_update():
    future = NOW + timedelta(hours=1)
    reminder = _reminder(
        occurrence=future,
        scheduledOccurrenceId="rem_123:2026-08-30T11:00:00Z",
    )
    worker, store, schedule, sender, member = _worker(reminder)
    asyncio.run(schedule.schedule("rem_123:2026-08-30T10:00:00Z", int(NOW.timestamp())))
    asyncio.run(worker.tick())
    assert sender.sent == []


def test_malformed_reminder_does_not_crash_worker():
    reminder = _reminder()
    worker, store, schedule, sender, member = _worker(reminder)
    asyncio.run(schedule.cancel(member))
    asyncio.run(schedule.schedule("not-an-id", int(NOW.timestamp())))
    asyncio.run(worker.tick())
    assert sender.sent == []
    assert "not-an-id" in schedule.dead_letter
