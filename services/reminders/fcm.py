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
        for device_token in tokens:
            body = {
                "message": {
                    "token": device_token,
                    "notification": {
                        "title": payload.get("title") or "Buddy",
                        "body": payload.get("message") or "",
                    },
                    "data": payload,
                    "android": {
                        "priority": "high" if high_priority else "normal",
                    },
                }
            }
            status, _response = await self.http_post(
                f"https://fcm.googleapis.com/v1/projects/{self.project_id}/messages:send",
                body,
                token,
            )
            if status >= 500 or status == 429:
                transient += 1
            elif status >= 400:
                errors += 1
        if transient:
            raise FcmTransientError("fcm transient failure")
        if errors == len(tokens):
            raise FcmPermanentError("all device tokens rejected")
