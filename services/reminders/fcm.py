from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class FcmTransientError(Exception):
    pass


class FcmPermanentError(Exception):
    pass


class FcmSender(Protocol):
    async def send(self, tokens: list[str], payload: dict[str, str], high_priority: bool) -> None: ...


@dataclass
class RecordingFcmSender:
    sent: list[dict[str, Any]]

    def __init__(self):
        self.sent = []
        self.fail_next = False
        self.fail_retryable = True

    async def send(self, tokens: list[str], payload: dict[str, str], high_priority: bool) -> None:
        if not tokens:
            raise FcmPermanentError("no device tokens")
        if self.fail_next:
            self.fail_next = False
            if self.fail_retryable:
                raise FcmTransientError("fcm unavailable")
            raise FcmPermanentError("invalid payload")
        self.sent.append(
            {
                "tokens": list(tokens),
                "payload": dict(payload),
                "high_priority": high_priority,
            }
        )


def raise_for_fcm_result(sent: int, transient: int, errors: int) -> None:
    if sent > 0:
        return
    if transient:
        raise FcmTransientError("fcm transient failure")
    if errors:
        raise FcmPermanentError("all device tokens rejected")
    raise FcmPermanentError("no FCM messages were accepted")


class HttpFcmSender:
    def __init__(self, project_id: str, access_token_provider, http_post):
        self.project_id = project_id
        self.access_token_provider = access_token_provider
        self.http_post = http_post

    async def send(self, tokens: list[str], payload: dict[str, str], high_priority: bool) -> None:
        if not tokens:
            raise FcmPermanentError("no device tokens")
        token = await self.access_token_provider()
        errors = 0
        transient = 0
        sent = 0
        for device_token in tokens:
            data_only = payload.get("type") in {"reminder_alarm", "ai_reminder_call"}
            message: dict[str, Any] = {
                "token": device_token,
                "data": payload,
                "android": {
                    "priority": "HIGH" if high_priority else "NORMAL",
                },
            }
            if not data_only:
                message["notification"] = {
                    "title": payload.get("title") or "Buddy",
                    "body": payload.get("message") or "",
                }
                message["android"]["notification"] = {
                    "channel_id": payload.get("channelId") or "buddy_reminders",
                    "sound": "default",
                    "icon": "ic_stat_buddy_mic",
                    "default_sound": True,
                    "notification_priority": "PRIORITY_HIGH",
                }
            body = {"message": message}
            status, _response = await self.http_post(
                f"https://fcm.googleapis.com/v1/projects/{self.project_id}/messages:send",
                body,
                token,
            )
            if status >= 500 or status == 429:
                transient += 1
            elif status >= 400:
                errors += 1
            else:
                sent += 1
        raise_for_fcm_result(sent, transient, errors)
