from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from pymongo import ReturnDocument

from apps.api_gateway.config.setting import settings
from services.conversation.models import utc_now
from services.conversation.repository import mongo_id_candidates
from services.document_it.schemas import (
    ACTIVE_STATUSES,
    PIPELINE_VERSION,
    DocumentStatus,
    GeneratedDocumentContent,
    SourceStats,
    public_source_stats,
)

CLAIM_STALE = timedelta(minutes=20)
PUBLIC_OMIT = {"claimedBy": 0, "docxBase64": 0}
LIST_OMIT = {"claimedBy": 0, "claimedAt": 0, "docxBase64": 0, "content": 0}


def _as_utc(value: datetime | None) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def is_stale_active(doc: dict[str, Any] | None, now: datetime | None = None) -> bool:
    if not doc or doc.get("status") not in ACTIVE_STATUSES:
        return False
    stamp = doc.get("claimedAt") or doc.get("updatedAt") or doc.get("createdAt")
    parsed = _as_utc(stamp)
    if parsed is None:
        return True
    current = now or utc_now()
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return parsed <= current - CLAIM_STALE


def build_document_preview(content: dict[str, Any] | None) -> dict[str, Any]:
    payload = content if isinstance(content, dict) else {}
    sections = payload.get("sections") if isinstance(payload.get("sections"), list) else []
    excerpt = ""
    for section in sections:
        if not isinstance(section, dict):
            continue
        body = str(section.get("body") or "").strip()
        bullets = section.get("bullets") if isinstance(section.get("bullets"), list) else []
        if body:
            excerpt = body
            break
        if bullets:
            excerpt = str(bullets[0] or "").strip()
            break
    if len(excerpt) > 160:
        excerpt = excerpt[:159].rstrip() + "…"
    return {
        "title": str(payload.get("title") or "Document"),
        "subtitle": payload.get("subtitle"),
        "excerpt": excerpt or None,
        "sectionCount": len(sections),
    }


class DocumentStore:
    def __init__(self, database):
        self.db = database
        self.collection = database.spaceDocuments

    async def find_active(
        self,
        user_id: str,
        space_id: str,
        *,
        template_code: str,
    ) -> dict[str, Any] | None:
        docs = await (
            self.collection.find(
                {
                    "userId": {"$in": mongo_id_candidates(user_id)},
                    "spaceId": {"$in": mongo_id_candidates(space_id)},
                    "templateCode": str(template_code),
                    "status": {"$in": list(ACTIVE_STATUSES)},
                }
            )
            .sort([("updatedAt", -1)])
            .limit(5)
            .to_list(length=5)
        )
        for doc in docs:
            if not is_stale_active(doc):
                return doc
        return None

    async def get(self, document_id: str, *, include_docx: bool = False) -> dict[str, Any] | None:
        projection = None if include_docx else PUBLIC_OMIT
        return await self.collection.find_one(
            {"_id": {"$in": mongo_id_candidates(document_id)}},
            projection,
        )

    async def get_by_job(self, job_id: str) -> dict[str, Any] | None:
        return await self.collection.find_one({"jobId": str(job_id)}, PUBLIC_OMIT)

    async def next_version(self, user_id: str, space_id: str, template_code: str) -> int:
        latest = await self.collection.find_one(
            {
                "userId": {"$in": mongo_id_candidates(user_id)},
                "spaceId": {"$in": mongo_id_candidates(space_id)},
                "templateCode": str(template_code),
            },
            {"version": 1},
            sort=[("version", -1)],
        )
        return int((latest or {}).get("version") or 0) + 1

    async def reserve(
        self,
        *,
        user_id: str,
        space_id: str,
        job_id: str,
        template_code: str,
        template_title: str,
    ) -> dict[str, Any]:
        now = utc_now()
        document_id = str(uuid4())
        version = await self.next_version(user_id, space_id, template_code)
        doc = {
            "_id": document_id,
            "userId": str(user_id),
            "spaceId": str(space_id),
            "jobId": str(job_id),
            "templateCode": str(template_code),
            "templateTitle": str(template_title),
            "status": DocumentStatus.QUEUED.value,
            "stage": "queued",
            "progress": 5,
            "message": "Queued for generation",
            "error": None,
            "content": None,
            "preview": {"title": template_title, "sectionCount": 0},
            "docxBase64": None,
            "docxBytes": 0,
            "fileName": None,
            "mimeType": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "sourceStats": SourceStats().model_dump(),
            "model": None,
            "pipelineVersion": PIPELINE_VERSION,
            "version": version,
            "claimedBy": None,
            "claimedAt": None,
            "createdAt": now,
            "updatedAt": now,
        }
        await self.collection.insert_one(doc)
        await self._prune_old_versions(user_id, space_id)
        return doc

    async def update_progress(
        self,
        document_id: str,
        *,
        status: str | None = None,
        stage: str | None = None,
        progress: int | None = None,
        message: str | None = None,
        error: str | None = None,
        source_stats: SourceStats | dict[str, Any] | None = None,
        model: str | None = None,
        claimed_by: str | None = None,
    ) -> dict[str, Any] | None:
        updates: dict[str, Any] = {"updatedAt": utc_now()}
        if status is not None:
            updates["status"] = status
        if stage is not None:
            updates["stage"] = stage
        if progress is not None:
            updates["progress"] = max(0, min(100, int(progress)))
        if message is not None:
            updates["message"] = message
        if error is not None:
            updates["error"] = error
        if source_stats is not None:
            updates["sourceStats"] = public_source_stats(source_stats)
        if model is not None:
            updates["model"] = model
        if claimed_by is not None:
            updates["claimedBy"] = claimed_by
            updates["claimedAt"] = utc_now()
        return await self.collection.find_one_and_update(
            {"_id": {"$in": mongo_id_candidates(document_id)}},
            {"$set": updates},
            return_document=ReturnDocument.AFTER,
        )

    async def claim(self, document_id: str, job_id: str) -> tuple[str, dict[str, Any] | None]:
        existing = await self.get(document_id)
        if existing is None:
            return "missing", None
        if existing.get("status") == DocumentStatus.READY.value:
            return "exists", existing
        if (
            existing.get("status") in ACTIVE_STATUSES
            and existing.get("claimedBy")
            and existing.get("claimedBy") != job_id
            and not is_stale_active(existing)
            and existing.get("status") != DocumentStatus.QUEUED.value
        ):
            return "busy", existing

        updated = await self.collection.find_one_and_update(
            {
                "_id": {"$in": mongo_id_candidates(document_id)},
                "status": {
                    "$in": [
                        DocumentStatus.QUEUED.value,
                        DocumentStatus.FAILED.value,
                        *ACTIVE_STATUSES,
                    ]
                },
            },
            {
                "$set": {
                    "status": DocumentStatus.GATHERING.value,
                    "stage": "gathering_context",
                    "progress": 15,
                    "message": "Gathering notes, tasks, and transcripts",
                    "claimedBy": job_id,
                    "claimedAt": utc_now(),
                    "updatedAt": utc_now(),
                    "error": None,
                }
            },
            return_document=ReturnDocument.AFTER,
        )
        if updated is None:
            return "busy", existing
        return "claimed", updated

    async def save_ready(
        self,
        document_id: str,
        *,
        content: GeneratedDocumentContent,
        docx_bytes: bytes,
        file_name: str,
        source_stats: SourceStats,
        model: str | None,
        space_name: str,
    ) -> dict[str, Any] | None:
        content_payload = content.model_dump()
        preview = build_document_preview(content_payload)
        preview["spaceName"] = space_name
        return await self.collection.find_one_and_update(
            {"_id": {"$in": mongo_id_candidates(document_id)}},
            {
                "$set": {
                    "status": DocumentStatus.READY.value,
                    "stage": "completed",
                    "progress": 100,
                    "message": "Document ready",
                    "error": None,
                    "content": content_payload,
                    "preview": preview,
                    "docxBase64": base64.b64encode(docx_bytes).decode("ascii"),
                    "docxBytes": len(docx_bytes),
                    "fileName": file_name,
                    "sourceStats": source_stats.model_dump(),
                    "model": model,
                    "updatedAt": utc_now(),
                }
            },
            return_document=ReturnDocument.AFTER,
        )

    async def list_for_space(
        self,
        user_id: str,
        space_id: str,
        *,
        limit: int = 24,
    ) -> list[dict[str, Any]]:
        capped = max(1, min(int(limit), 50))
        return await (
            self.collection.find(
                {
                    "userId": {"$in": mongo_id_candidates(user_id)},
                    "spaceId": {"$in": mongo_id_candidates(space_id)},
                    "status": DocumentStatus.READY.value,
                },
                LIST_OMIT,
            )
            .sort([("updatedAt", -1), ("version", -1)])
            .limit(capped)
            .to_list(length=capped)
        )

    async def mark_failed(self, document_id: str, error: str) -> dict[str, Any] | None:
        return await self.collection.find_one_and_update(
            {"_id": {"$in": mongo_id_candidates(document_id)}},
            {
                "$set": {
                    "status": DocumentStatus.FAILED.value,
                    "stage": "failed",
                    "progress": 100,
                    "message": "Generation failed",
                    "error": str(error)[:500],
                    "updatedAt": utc_now(),
                }
            },
            return_document=ReturnDocument.AFTER,
        )

    async def _prune_old_versions(self, user_id: str, space_id: str) -> None:
        max_versions = settings.DOCUMENT_MAX_VERSIONS_PER_SPACE
        docs = await (
            self.collection.find(
                {
                    "userId": {"$in": mongo_id_candidates(user_id)},
                    "spaceId": {"$in": mongo_id_candidates(space_id)},
                    "status": {
                        "$in": [DocumentStatus.READY.value, DocumentStatus.FAILED.value]
                    },
                },
                {"_id": 1, "updatedAt": 1},
            )
            .sort([("updatedAt", -1)])
            .to_list(length=200)
        )
        if len(docs) <= max_versions:
            return
        drop_ids = [doc["_id"] for doc in docs[max_versions:]]
        if drop_ids:
            await self.collection.delete_many({"_id": {"$in": drop_ids}})


def public_document_doc(doc: dict[str, Any] | None, *, include_docx: bool = False) -> dict[str, Any] | None:
    if not doc:
        return None
    payload = dict(doc)
    payload["documentId"] = str(payload.get("_id") or "")
    payload["_id"] = str(payload.get("_id") or "")
    payload.pop("claimedBy", None)
    if not include_docx:
        payload.pop("docxBase64", None)
    payload["hasDocx"] = bool(doc.get("docxBase64") or doc.get("docxBytes"))
    return payload


def public_document_summary(doc: dict[str, Any] | None) -> dict[str, Any] | None:
    if not doc:
        return None
    payload = public_document_doc(doc) or {}
    payload.pop("content", None)
    payload.pop("docxBase64", None)
    if not payload.get("preview"):
        payload["preview"] = build_document_preview(None)
    return payload
