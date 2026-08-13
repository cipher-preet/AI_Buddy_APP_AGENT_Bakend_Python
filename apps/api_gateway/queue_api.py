from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from apps.api_gateway.config.setting import settings
from services.queue.redis_queue import SPEECH_QUEUE, redis_client
from services.queue.streams import EventEnvelope


app = FastAPI(title=f"{settings.APP_NAME} Queue API", version=settings.APP_VERSION)

ALLOWED_EVENT_TYPES = {
    "audio.ingested",
    "speech.transcribe.requested",
    "stt.requested",
    "conversation.transcript.ready",
    "conversation.window.extraction.requested",
    "conversation.finalization.requested",
    "conversation.processing.requested",
    "job.retry.requested",
}


@app.get("/health/live")
async def live():
    return {"status": "ok"}


@app.get("/health/ready")
async def ready():
    try:
        await redis_client.ping()
    except Exception:
        return JSONResponse({"status": "unready"}, status_code=503)
    return {"status": "ready"}


@app.post("/internal/events")
async def accept_event(request: Request):
    body = await request.body()
    if len(body) > settings.QUEUE_API_MAX_BODY_BYTES:
        return JSONResponse({"detail": "request body too large"}, status_code=413)
    try:
        _verify_request(request, body)
        payload = json.loads(body.decode("utf-8"))
        event = EventEnvelope.model_validate(payload)
        if event.eventType not in ALLOWED_EVENT_TYPES:
            return JSONResponse({"detail": "unsupported event type"}, status_code=400)
        if event.eventType == "speech.transcribe.requested":
            return await _accept_speech_job(event)
        stream = _stream_for_event(payload, event)
        accepted_key = f"queue_api:event:{event.eventId}"
        previous = await redis_client.hgetall(accepted_key)
        if previous:
            return JSONResponse(
                {
                    "accepted": True,
                    "duplicate": True,
                    "eventId": event.eventId,
                    "correlationId": event.correlationId,
                    "stream": previous.get("stream") or stream,
                },
                status_code=202,
            )

        redis_message_id = await redis_client.xadd(stream, {"event": event.model_dump_json()})
        await redis_client.hset(
            accepted_key,
            mapping={
                "stream": stream,
                "redisMessageId": redis_message_id,
                "correlationId": event.correlationId,
                "acceptedAt": str(int(time.time())),
            },
        )
        await redis_client.expire(accepted_key, max(86400, settings.S3_PRESIGNED_URL_TTL_SECONDS * 24))
        return JSONResponse(
            {
                "accepted": True,
                "duplicate": False,
                "eventId": event.eventId,
                "correlationId": event.correlationId,
                "stream": stream,
            },
            status_code=202,
        )
    except PermissionError as error:
        return JSONResponse({"detail": str(error)}, status_code=401)
    except (ValueError, json.JSONDecodeError) as error:
        return JSONResponse({"detail": str(error)}, status_code=400)


def _verify_request(request: Request, body: bytes) -> None:
    token = settings.secret_value(settings.QUEUE_API_SERVICE_TOKEN)
    secret = settings.secret_value(settings.QUEUE_API_HMAC_SECRET)
    if not token and not secret:
        raise PermissionError("Queue API authentication is not configured")

    if token:
        authorization = request.headers.get("authorization", "")
        expected = f"Bearer {token}"
        if not hmac.compare_digest(authorization, expected):
            raise PermissionError("Invalid service token")

    if secret:
        timestamp = request.headers.get("x-buddy-timestamp", "")
        signature = request.headers.get("x-buddy-signature", "")
        try:
            timestamp_int = int(timestamp)
        except ValueError as error:
            raise PermissionError("Invalid request timestamp") from error
        if abs(int(time.time()) - timestamp_int) > settings.QUEUE_API_SIGNATURE_TOLERANCE_SECONDS:
            raise PermissionError("Stale request timestamp")
        expected_sig = hmac.new(
            secret.encode("utf-8"),
            timestamp.encode("utf-8") + b"." + body,
            hashlib.sha256,
        ).hexdigest()
        expected_header = f"sha256={expected_sig}"
        if not hmac.compare_digest(signature, expected_header):
            raise PermissionError("Invalid request signature")


def _stream_for_event(payload: dict[str, Any], event: EventEnvelope) -> str:
    explicit = str(payload.get("targetStream") or "").strip()
    if explicit:
        allowed_streams = {
            settings.REDIS_AUDIO_STREAM,
            settings.REDIS_STT_STREAM,
            settings.REDIS_TRANSCRIPT_READY_STREAM,
            settings.REDIS_WINDOW_EXTRACTION_STREAM,
            settings.REDIS_FINALIZATION_STREAM,
            settings.REDIS_PROCESSING_STREAM,
            settings.REDIS_RETRY_STREAM,
        }
        if explicit not in allowed_streams:
            raise ValueError("targetStream is not allowed")
        return explicit
    if event.eventType == "audio.ingested":
        return settings.REDIS_AUDIO_STREAM
    if event.eventType == "stt.requested":
        return settings.REDIS_STT_STREAM
    if event.eventType == "conversation.transcript.ready":
        return settings.REDIS_TRANSCRIPT_READY_STREAM
    if event.eventType == "conversation.window.extraction.requested":
        return settings.REDIS_WINDOW_EXTRACTION_STREAM
    if event.eventType == "conversation.finalization.requested":
        return settings.REDIS_FINALIZATION_STREAM
    if event.eventType == "conversation.processing.requested":
        return settings.REDIS_PROCESSING_STREAM
    if event.eventType == "job.retry.requested":
        return settings.REDIS_RETRY_STREAM
    raise ValueError("unsupported event type")


async def _accept_speech_job(event: EventEnvelope) -> JSONResponse:
    job = dict(event.payload or {})
    job_id = str(job.get("job_id") or job.get("jobId") or event.eventId).strip()
    if not job_id:
        raise ValueError("speech job payload is missing job_id")

    job.setdefault("job_id", job_id)
    job.setdefault("user_id", event.userId)
    job.setdefault("space_id", event.spaceId)
    if event.conversationId:
        job.setdefault("conversation_id", event.conversationId)
    job.setdefault("status", "queued")

    accepted_key = f"queue_api:event:{event.eventId}"
    previous = await redis_client.hgetall(accepted_key)
    if previous:
        return JSONResponse(
            {
                "accepted": True,
                "duplicate": True,
                "eventId": event.eventId,
                "correlationId": event.correlationId,
                "stream": previous.get("stream") or SPEECH_QUEUE,
            },
            status_code=202,
        )

    await redis_client.hset(f"speech_job:{job_id}", mapping={key: str(value) for key, value in job.items() if value is not None})
    await redis_client.lpush(SPEECH_QUEUE, json.dumps(job))
    await redis_client.hset(
        accepted_key,
        mapping={
            "stream": SPEECH_QUEUE,
            "redisMessageId": job_id,
            "correlationId": event.correlationId,
            "acceptedAt": str(int(time.time())),
        },
    )
    await redis_client.expire(accepted_key, max(86400, settings.S3_PRESIGNED_URL_TTL_SECONDS * 24))
    return JSONResponse(
        {
            "accepted": True,
            "duplicate": False,
            "eventId": event.eventId,
            "correlationId": event.correlationId,
            "stream": SPEECH_QUEUE,
        },
        status_code=202,
    )
