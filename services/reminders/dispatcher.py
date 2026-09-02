from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from services.reminders.fcm import FcmPermanentError, FcmSender, FcmTransientError
from services.reminders.occurrence import TriggerEvent

UTC = timezone.utc
AI_CALL_TTL_SECONDS = 60


class DeliveryError(Exception):
    def __init__(self, message: str, retryable: bool):
        super().__init__(message)
        self.retryable = retryable


def android_payload(event: TriggerEvent, now: datetime | None = None) -> dict[str, str]:
    now = now or datetime.now(UTC)
    if event.type == "ALARM_NOTIFICATION":
        return {
            "type": "reminder_alarm",
            "reminderId": event.reminder_id,
            "title": event.title,
            "message": event.message,
            "sound": "true",
            "channelId": "buddy_reminder_alarms_v2",
        }
    if event.type == "AI_CALL":
        expires = now + timedelta(seconds=AI_CALL_TTL_SECONDS)
        return {
            "type": "ai_reminder_call",
            "reminderId": event.reminder_id,
            "callId": str(uuid.uuid4()),
            "title": event.title or "Buddy AI Reminder",
            "message": event.message,
            "expiresAt": expires.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "channelId": "buddy_reminder_calls_v2",
        }
    return {
        "type": "reminder_notification",
        "reminderId": event.reminder_id,
        "title": event.title,
        "message": event.message,
        "channelId": "buddy_reminders",
    }


class ReminderDeliveryDispatcher:
    def __init__(self, fcm: FcmSender, token_lookup):
        self.fcm = fcm
        self.token_lookup = token_lookup

    async def dispatch(self, event: TriggerEvent) -> dict[str, Any]:
        if event.type not in {"NORMAL_NOTIFICATION", "ALARM_NOTIFICATION", "AI_CALL"}:
            raise DeliveryError(f"unknown reminder type {event.type}", retryable=False)

        tokens = await self.token_lookup(event.user_id)
        payload = android_payload(event)
        high_priority = True
        print(
            "Reminder FCM dispatch: "
            f"reminderId={event.reminder_id} tokenCount={len(tokens)} type={event.type}",
            flush=True,
        )
        try:
            await self.fcm.send(tokens, payload, high_priority)
        except FcmTransientError as error:
            raise DeliveryError(str(error), retryable=True) from error
        except FcmPermanentError as error:
            raise DeliveryError(str(error), retryable=False) from error
        return payload
