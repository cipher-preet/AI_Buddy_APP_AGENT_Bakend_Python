from __future__ import annotations

from uuid import uuid4

from apps.api_gateway.config.setting import settings
from services.llm.router import get_llm_router
from services.mindmap.log import mindmap_log
from services.mindmap.redis_client import MindmapRedisConfigError, get_mindmap_redis_client
from services.mindmap.schemas import MindmapStatus
from services.mindmap.store import MindmapStore, public_mindmap_doc
from services.queue.streams import EventEnvelope, RedisStreamProducer
from services.schedule_extraction.llm import parse_model_route


def openrouter_configured() -> bool:
    router = get_llm_router()
    for name, _model in parse_model_route(settings.MINDMAP_MODELS):
        provider = router.providers.get(name)
        if provider and getattr(provider, "configured", True) is not False:
            return True
    return False


async def enqueue_mindmap_generation(database, *, user_id: str, space_id: str) -> dict:
    store = MindmapStore(database)
    active = await store.find_active(user_id, space_id)
    if active is not None:
        mindmap_log(
            "enqueue_reuse_active",
            userId=user_id,
            spaceId=space_id,
            mindmapId=str(active.get("_id")),
            status=active.get("status"),
        )
        doc = public_mindmap_doc(active) or {}
        return {
            "success": True,
            "reused": True,
            "jobId": doc.get("jobId"),
            "mindmapId": doc.get("mindmapId"),
            "status": doc.get("status"),
            "stage": doc.get("stage"),
            "progress": doc.get("progress"),
            "message": doc.get("message") or "Generation already in progress",
        }

    if not openrouter_configured():
        raise PermissionError(
            "Mind map generation is unavailable: configure OPENROUTER_API_KEY (or MINDMAP_MODELS)."
        )

    job_id = str(uuid4())
    reserved = await store.reserve(user_id=user_id, space_id=space_id, job_id=job_id)
    mindmap_id = str(reserved["_id"])

    event = EventEnvelope(
        eventId=job_id,
        eventType="mindmap.generate.requested",
        correlationId=mindmap_id,
        userId=user_id,
        spaceId=space_id,
        conversationId=mindmap_id,
        payload={"mindmapId": mindmap_id},
    )

    try:
        client = get_mindmap_redis_client()
        await RedisStreamProducer(client, force_direct=True).publish(settings.REDIS_MINDMAP_STREAM, event)
    except MindmapRedisConfigError as error:
        await store.mark_failed(mindmap_id, str(error))
        raise PermissionError(str(error)) from error
    except Exception as error:
        await store.mark_failed(mindmap_id, f"Failed to enqueue job: {type(error).__name__}")
        raise RuntimeError(f"Failed to enqueue mindmap job: {error}") from error

    mindmap_log(
        "enqueued",
        userId=user_id,
        spaceId=space_id,
        mindmapId=mindmap_id,
        jobId=job_id,
    )
    return {
        "success": True,
        "reused": False,
        "jobId": job_id,
        "mindmapId": mindmap_id,
        "status": MindmapStatus.QUEUED.value,
        "stage": "queued",
        "progress": 5,
        "message": "Queued for generation",
    }
