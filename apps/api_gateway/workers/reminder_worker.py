from __future__ import annotations

import traceback

from apps.api_gateway.config.setting import settings
from services.db.mongo import get_database
from services.reminders.dispatcher import ReminderDeliveryDispatcher
from services.reminders.fcm_factory import (
    ReminderFcmConfigError,
    build_fcm_sender,
    fcm_sender_mode,
)
from services.reminders.mongo_store import MongoReminderStore
from services.reminders.redis_client import (
    ReminderRedisConfigError,
    get_reminder_redis_client,
    redact_redis_secrets,
    test_reminder_redis_connection,
)
from services.reminders.schedule_store import RedisScheduleStore, ReminderRedisKeys
from services.reminders.worker import ReminderWorker, ReminderWorkerConfig, run_reminder_worker_forever


def build_reminder_worker() -> ReminderWorker:
    keys = ReminderRedisKeys(
        schedule=settings.REMINDER_SCHEDULE_KEY,
        processing=settings.REMINDER_PROCESSING_KEY,
        retry=settings.REMINDER_RETRY_KEY,
        dead_letter=settings.REMINDER_DEAD_LETTER_KEY,
    )
    reminder_store = MongoReminderStore(get_database())
    fcm = build_fcm_sender(settings, on_invalid_token=reminder_store.delete_token)
    print(
        f"Reminder FCM: enabled={'true' if settings.FCM_ENABLED else 'false'} mode={fcm_sender_mode(fcm)}",
        flush=True,
    )
    dispatcher = ReminderDeliveryDispatcher(
        fcm=fcm,
        token_lookup=reminder_store.tokens_for_user,
    )
    return ReminderWorker(
        schedule_store=RedisScheduleStore(get_reminder_redis_client(), keys),
        reminder_store=reminder_store,
        dispatcher=dispatcher,
        config=ReminderWorkerConfig(
            lookahead_seconds=settings.REMINDER_LOOKAHEAD_SECONDS,
            poll_ms=settings.REMINDER_TRIGGER_POLL_MS,
            late_grace_seconds=settings.REMINDER_LATE_GRACE_SECONDS,
            max_retries=settings.REMINDER_MAX_RETRIES,
        ),
        keys=keys,
    )


async def start_reminder_worker() -> None:
    enabled = bool(settings.ENABLE_REMINDER_WORKER)
    print(f"Reminder worker config: enabled={'true' if enabled else 'false'}", flush=True)
    if not enabled:
        print("Reminder worker disabled", flush=True)
        return
    print("Reminder worker starting...", flush=True)
    try:
        await test_reminder_redis_connection()
        worker = build_reminder_worker()
        print(
            f"Reminder worker started: schedule_key={settings.REMINDER_SCHEDULE_KEY}",
            flush=True,
        )
        await run_reminder_worker_forever(worker)
    except ReminderRedisConfigError as error:
        print(f"Reminder worker startup failed: {error}", flush=True)
        raise
    except ReminderFcmConfigError as error:
        print(f"Reminder worker startup failed: {error}", flush=True)
        raise
    except Exception as error:
        print(
            "Reminder worker startup failed: "
            f"{redact_redis_secrets(str(error))}\n"
            f"{redact_redis_secrets(traceback.format_exc())}",
            flush=True,
        )
        raise
