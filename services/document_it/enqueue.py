from __future__ import annotations

from uuid import uuid4

from apps.api_gateway.config.setting import settings
from services.document_it.log import document_log
from services.document_it.schemas import DocumentStatus
from services.document_it.store import DocumentStore, public_document_doc
from services.document_it.templates import load_template_spec
from services.llm.router import get_llm_router
from services.mindmap.redis_client import MindmapRedisConfigError, get_mindmap_redis_client
from services.queue.streams import EventEnvelope, RedisStreamProducer
from services.schedule_extraction.llm import parse_model_route


def openrouter_configured() -> bool:
    router = get_llm_router()
    spec = settings.DOCUMENT_MODELS or settings.MINDMAP_MODELS
    for name, _model in parse_model_route(spec):
        provider = router.providers.get(name)
        if provider and getattr(provider, "configured", True) is not False:
            return True
    return False


async def enqueue_document_generation(
    database,
    *,
    user_id: str,
    space_id: str,
    template_code: str,
) -> dict:
    code = str(template_code or "").strip().lower()
    if not code:
        raise ValueError("templateCode is required.")

    store = DocumentStore(database)
    template = await load_template_spec(database, code)
    active = await store.find_active(user_id, space_id, template_code=code)
    if active is not None:
        document_log(
            "enqueue_reuse_active",
            userId=user_id,
            spaceId=space_id,
            documentId=str(active.get("_id")),
            templateCode=code,
            status=active.get("status"),
        )
        doc = public_document_doc(active) or {}
        return {
            "success": True,
            "reused": True,
            "jobId": doc.get("jobId"),
            "documentId": doc.get("documentId"),
            "status": doc.get("status"),
            "stage": doc.get("stage"),
            "progress": doc.get("progress"),
            "message": doc.get("message") or "Generation already in progress",
            "templateCode": code,
            "templateTitle": template.title,
        }

    if not openrouter_configured():
        raise PermissionError(
            "Document generation is unavailable: configure NVIDIA_API_KEY or KRUTRIM_API_KEY (or DOCUMENT_MODELS)."
        )

    job_id = str(uuid4())
    reserved = await store.reserve(
        user_id=user_id,
        space_id=space_id,
        job_id=job_id,
        template_code=code,
        template_title=template.title,
    )
    document_id = str(reserved["_id"])

    event = EventEnvelope(
        eventId=job_id,
        eventType="document.generate.requested",
        correlationId=document_id,
        userId=user_id,
        spaceId=space_id,
        conversationId=document_id,
        payload={
            "documentId": document_id,
            "templateCode": code,
            "templateTitle": template.title,
        },
    )

    try:
        client = get_mindmap_redis_client()
        await RedisStreamProducer(client, force_direct=True).publish(
            settings.REDIS_DOCUMENT_STREAM,
            event,
            maxlen=settings.REDIS_DOCUMENT_STREAM_MAXLEN,
        )
    except MindmapRedisConfigError as error:
        await store.mark_failed(document_id, str(error))
        raise PermissionError(str(error)) from error
    except Exception as error:
        await store.mark_failed(document_id, f"Failed to enqueue job: {type(error).__name__}")
        raise RuntimeError(f"Failed to enqueue document job: {error}") from error

    document_log(
        "enqueued",
        userId=user_id,
        spaceId=space_id,
        documentId=document_id,
        jobId=job_id,
        templateCode=code,
    )
    return {
        "success": True,
        "reused": False,
        "jobId": job_id,
        "documentId": document_id,
        "status": DocumentStatus.QUEUED.value,
        "stage": "queued",
        "progress": 5,
        "message": "Queued for generation",
        "templateCode": code,
        "templateTitle": template.title,
    }
