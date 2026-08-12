from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any

import httpx

from apps.api_gateway.config.setting import settings


class QueueApiError(RuntimeError):
    pass


@dataclass(frozen=True)
class QueueApiPublishResult:
    accepted: bool
    duplicate: bool
    event_id: str
    correlation_id: str


class QueueApiPublisher:
    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or settings.QUEUE_API_BASE_URL).rstrip("/")

    async def publish(self, payload: dict[str, Any]) -> QueueApiPublishResult:
        if not self.base_url:
            raise QueueApiError("QUEUE_API_BASE_URL is required")

        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        if len(body) > settings.QUEUE_API_MAX_BODY_BYTES:
            raise QueueApiError("Queue API payload exceeds configured body limit")

        headers = _auth_headers(body)
        timeout = httpx.Timeout(settings.QUEUE_API_REQUEST_TIMEOUT_SECONDS)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(f"{self.base_url}/internal/events", content=body, headers=headers)
        if response.status_code != 202:
            raise QueueApiError(f"Queue API publish failed with status {response.status_code}")
        data = response.json()
        return QueueApiPublishResult(
            accepted=bool(data.get("accepted", True)),
            duplicate=bool(data.get("duplicate", False)),
            event_id=str(data.get("eventId") or payload.get("eventId") or ""),
            correlation_id=str(data.get("correlationId") or payload.get("correlationId") or ""),
        )


def _auth_headers(body: bytes) -> dict[str, str]:
    headers = {"content-type": "application/json"}
    token = settings.secret_value(settings.QUEUE_API_SERVICE_TOKEN)
    if token:
        headers["authorization"] = f"Bearer {token}"

    secret = settings.secret_value(settings.QUEUE_API_HMAC_SECRET)
    if secret:
        timestamp = str(int(time.time()))
        signature = hmac.new(
            secret.encode("utf-8"),
            timestamp.encode("utf-8") + b"." + body,
            hashlib.sha256,
        ).hexdigest()
        headers["x-buddy-timestamp"] = timestamp
        headers["x-buddy-signature"] = f"sha256={signature}"
    return headers
