from __future__ import annotations

from time import perf_counter

from fastapi import FastAPI, Request
from fastapi.responses import Response

from apps.api_gateway.config.setting import settings
from apps.api_gateway.workers.conversation_workers import (
    handle_audio_event,
    handle_finalization_event,
    handle_processing_event,
    handle_stt_event,
)
from apps.api_gateway.workers.speech_worker import process_speech_job
from apps.api_gateway.workers.vector_worker import process_completed_speech_job
from services.db.mongo import close_mongo_client, ensure_mongo_indexes
from services.queue.pubsub import (
    InvalidPubSubEnvelope,
    InvalidPubSubPayload,
    PubSubPushMessage,
    decode_pubsub_push_envelope,
    log_processing_result,
    verify_pubsub_push_auth,
)
from services.queue.streams import EventEnvelope

app = FastAPI(title=f"{settings.APP_NAME} Worker", version=settings.APP_VERSION)


@app.on_event("startup")
async def startup():
    await ensure_mongo_indexes()


@app.on_event("shutdown")
async def shutdown():
    await close_mongo_client()


@app.get("/health")
async def health():
    return {"status": "ok", "queue_provider": settings.QUEUE_PROVIDER}


@app.post("/pubsub/speech")
async def pubsub_speech(request: Request):
    return await _handle_push(request, "speech", _route_speech)


@app.post("/pubsub/vector")
async def pubsub_vector(request: Request):
    return await _handle_push(request, "vector", _route_vector)


@app.post("/pubsub/orchestration")
async def pubsub_orchestration(request: Request):
    return await _handle_push(request, "orchestration", _route_orchestration)


async def _handle_push(request: Request, stage: str, handler):
    started_at = perf_counter()
    try:
        await verify_pubsub_push_auth(request)
    except PermissionError:
        return Response(status_code=401)
    except Exception:
        return Response(status_code=403)

    try:
        envelope = await request.json()
        message = decode_pubsub_push_envelope(envelope)
    except (InvalidPubSubEnvelope, InvalidPubSubPayload) as error:
        print(f"Invalid Pub/Sub {stage} payload:", str(error))
        return Response(status_code=204)
    except Exception as error:
        print(f"Unreadable Pub/Sub {stage} payload:", str(error))
        return Response(status_code=204)

    try:
        await handler(message)
        log_processing_result(message, stage, started_at, success=True)
        return Response(status_code=204)
    except ValueError as error:
        print(f"Permanent Pub/Sub {stage} processing error:", str(error))
        log_processing_result(message, stage, started_at, success=False, error=str(error))
        return Response(status_code=204)
    except Exception as error:
        print(f"Temporary Pub/Sub {stage} processing error:", str(error))
        log_processing_result(message, stage, started_at, success=False, error=str(error))
        return Response(status_code=500)


async def _route_speech(message: PubSubPushMessage) -> None:
    payload = message.payload
    event_type = message.attributes.get("event_type") or payload.get("eventType")
    source = message.attributes.get("source")
    if source == settings.REDIS_AUDIO_STREAM:
        await handle_audio_event(EventEnvelope.model_validate(payload))
        return
    if source == settings.REDIS_STT_STREAM or event_type == "stt.requested":
        await handle_stt_event(EventEnvelope.model_validate(payload))
        return
    if event_type == "audio.ingested":
        await handle_audio_event(EventEnvelope.model_validate(payload))
        return
    await process_speech_job(payload)


async def _route_vector(message: PubSubPushMessage) -> None:
    job_id = message.payload.get("job_id")
    if not job_id:
        raise ValueError("Vector Pub/Sub payload is missing job_id")
    await process_completed_speech_job(str(job_id))


async def _route_orchestration(message: PubSubPushMessage) -> None:
    payload = EventEnvelope.model_validate(message.payload)
    event_type = message.attributes.get("event_type") or payload.eventType
    if event_type == "conversation.finalization.requested":
        await handle_finalization_event(payload)
        return
    if event_type == "conversation.processing.requested":
        await handle_processing_event(payload)
        return
    raise ValueError(f"Unknown orchestration event_type: {event_type}")
