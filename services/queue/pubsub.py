from __future__ import annotations

import asyncio
import base64
import json
import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from fastapi import Request

from apps.api_gateway.config.setting import settings

logger = logging.getLogger(__name__)


class InvalidPubSubEnvelope(ValueError):
    pass


class InvalidPubSubPayload(ValueError):
    pass


class TemporaryProcessingError(RuntimeError):
    pass


class PermanentProcessingError(RuntimeError):
    pass


@dataclass(frozen=True)
class PubSubPushMessage:
    payload: dict[str, Any]
    message_id: str | None
    attributes: dict[str, str]
    subscription: str | None
    delivery_attempt: int | None


class MessagePublisher:
    async def publish(
        self,
        topic: str,
        payload: dict[str, Any],
        attributes: dict[str, Any] | None = None,
    ) -> str:
        raise NotImplementedError


class PubSubMessagePublisher(MessagePublisher):
    def __init__(self, project_id: str | None = None, timeout_seconds: float | None = None):
        self.project_id = project_id or settings.GOOGLE_CLOUD_PROJECT
        self.timeout_seconds = timeout_seconds or settings.PUBSUB_PUBLISH_TIMEOUT_SECONDS
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from google.cloud import pubsub_v1

            self._client = pubsub_v1.PublisherClient()
        return self._client

    async def publish(
        self,
        topic: str,
        payload: dict[str, Any],
        attributes: dict[str, Any] | None = None,
    ) -> str:
        if not self.project_id:
            raise RuntimeError("GOOGLE_CLOUD_PROJECT is required when QUEUE_PROVIDER=pubsub")
        if not topic:
            raise RuntimeError("Pub/Sub topic is required")
        data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        topic_path = topic if topic.startswith("projects/") else self.client.topic_path(self.project_id, topic)
        safe_attributes = _safe_attributes(payload, attributes)

        def _publish() -> str:
            future = self.client.publish(topic_path, data, **safe_attributes)
            return future.result(timeout=self.timeout_seconds)

        try:
            message_id = await asyncio.wait_for(asyncio.to_thread(_publish), timeout=self.timeout_seconds + 1)
        except Exception:
            logger.exception(
                "Pub/Sub publish failed",
                extra={"topic": topic, "job_id": payload.get("job_id"), "event_type": _event_type(payload, attributes)},
            )
            raise

        logger.info(
            "Pub/Sub message published",
            extra={
                "pubsub_message_id": message_id,
                "topic": topic,
                "job_id": payload.get("job_id"),
                "request_id": payload.get("request_id"),
                "user_id": payload.get("user_id") or payload.get("userId"),
                "space_id": payload.get("space_id") or payload.get("spaceId"),
                "event_type": _event_type(payload, attributes),
            },
        )
        return message_id


def _event_type(payload: dict[str, Any], attributes: dict[str, Any] | None = None) -> str:
    if attributes and attributes.get("event_type"):
        return str(attributes["event_type"])
    return str(payload.get("eventType") or payload.get("event_type") or "job.requested")


def _safe_attributes(payload: dict[str, Any], attributes: dict[str, Any] | None = None) -> dict[str, str]:
    merged: dict[str, Any] = {
        "event_type": _event_type(payload, attributes),
        "job_id": payload.get("job_id"),
        "user_id": payload.get("user_id") or payload.get("userId"),
        "space_id": payload.get("space_id") or payload.get("spaceId"),
        "request_id": payload.get("request_id") or payload.get("correlationId"),
        "source": "ai-orchestration",
    }
    if attributes:
        merged.update(attributes)
    return {key: str(value) for key, value in merged.items() if value is not None and value != ""}


def decode_pubsub_push_envelope(envelope: dict[str, Any]) -> PubSubPushMessage:
    message = envelope.get("message")
    if not isinstance(message, dict):
        raise InvalidPubSubEnvelope("Pub/Sub envelope is missing message")
    encoded = message.get("data")
    if not encoded:
        raise InvalidPubSubEnvelope("Pub/Sub message is missing data")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except Exception as error:
        raise InvalidPubSubPayload("Pub/Sub data is not valid base64") from error
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise InvalidPubSubPayload("Pub/Sub data is not valid UTF-8") from error
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise InvalidPubSubPayload("Pub/Sub data is not valid JSON") from error
    if not isinstance(payload, dict):
        raise InvalidPubSubPayload("Pub/Sub payload must be a JSON object")

    attributes = message.get("attributes") or {}
    if not isinstance(attributes, dict):
        raise InvalidPubSubEnvelope("Pub/Sub message attributes must be an object")
    delivery_attempt = envelope.get("deliveryAttempt")
    if delivery_attempt is not None:
        try:
            delivery_attempt = int(delivery_attempt)
        except (TypeError, ValueError) as error:
            raise InvalidPubSubEnvelope("Pub/Sub deliveryAttempt must be an integer") from error

    return PubSubPushMessage(
        payload=payload,
        message_id=message.get("messageId") or message.get("message_id"),
        attributes={str(key): str(value) for key, value in attributes.items()},
        subscription=envelope.get("subscription"),
        delivery_attempt=delivery_attempt,
    )


async def verify_pubsub_push_auth(request: Request) -> None:
    if not settings.PUBSUB_VERIFY_PUSH_AUTH:
        return
    authorization = request.headers.get("authorization", "")
    if not authorization.lower().startswith("bearer "):
        raise PermissionError("Missing Pub/Sub push bearer token")
    if not settings.PUBSUB_WORKER_AUDIENCE:
        raise PermissionError("PUBSUB_WORKER_AUDIENCE is required when push auth verification is enabled")

    token = authorization.split(" ", 1)[1].strip()

    def _verify() -> None:
        from google.auth.transport import requests
        from google.oauth2 import id_token

        id_token.verify_oauth2_token(token, requests.Request(), audience=settings.PUBSUB_WORKER_AUDIENCE)

    await asyncio.to_thread(_verify)


def topic_for_speech() -> str:
    return settings.PUBSUB_SPEECH_TOPIC


def topic_for_vector() -> str:
    return settings.PUBSUB_VECTOR_TOPIC


def topic_for_orchestration() -> str:
    return settings.PUBSUB_ORCHESTRATION_TOPIC


def log_processing_result(
    message: PubSubPushMessage,
    processing_stage: str,
    started_at: float,
    success: bool,
    error: str | None = None,
) -> None:
    payload = message.payload
    logger.info(
        "Pub/Sub push processing finished",
        extra={
            "pubsub_message_id": message.message_id,
            "job_id": payload.get("job_id"),
            "request_id": payload.get("request_id") or payload.get("correlationId"),
            "user_id": payload.get("user_id") or payload.get("userId"),
            "space_id": payload.get("space_id") or payload.get("spaceId"),
            "event_type": message.attributes.get("event_type") or payload.get("eventType"),
            "subscription": message.subscription,
            "delivery_attempt": message.delivery_attempt,
            "processing_stage": processing_stage,
            "processing_duration_ms": int((perf_counter() - started_at) * 1000),
            "success": success,
            "error": error,
        },
    )
