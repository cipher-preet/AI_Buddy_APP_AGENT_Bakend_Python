from __future__ import annotations

import re

from services.document_it.context_pack import build_document_context_pack, has_usable_context
from services.document_it.docx_builder import build_docx_bytes
from services.document_it.generator import DocumentGenerateError, generate_document_content
from services.document_it.log import document_log
from services.document_it.schemas import DocumentStatus, SourceStats
from services.document_it.store import DocumentStore
from services.document_it.templates import load_template_spec
from services.mindmap.sources import MindmapSource
from services.queue.streams import EventEnvelope, NonRetryableQueueError


def _safe_file_stem(*parts: str) -> str:
    joined = "-".join(str(part or "").strip() for part in parts if str(part or "").strip())
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", joined).strip("-._")
    return (cleaned or "document")[:80]


class DocumentJobHandler:
    def __init__(self, database):
        self.database = database
        self.store = DocumentStore(database)
        self.sources = MindmapSource(database)

    async def handle(self, event: EventEnvelope) -> None:
        payload = event.payload or {}
        document_id = str(payload.get("documentId") or "").strip()
        template_code = str(payload.get("templateCode") or "").strip().lower()
        user_id = str(event.userId or "").strip()
        space_id = str(event.spaceId or "").strip()
        if not document_id or not user_id or not space_id or not template_code:
            raise NonRetryableQueueError(
                "document job missing documentId/userId/spaceId/templateCode"
            )

        document_log(
            "job_start",
            documentId=document_id,
            userId=user_id,
            spaceId=space_id,
            jobId=event.eventId,
            templateCode=template_code,
            attempt=event.attempt,
        )

        claim_state, doc = await self.store.claim(document_id, event.eventId)
        if claim_state == "missing":
            raise NonRetryableQueueError(f"document not found: {document_id}")
        if claim_state == "exists":
            return
        if claim_state == "busy":
            document_log("job_busy", documentId=document_id, jobId=event.eventId)
            return

        try:
            template = await load_template_spec(self.database, template_code)
            await self.store.update_progress(
                document_id,
                status=DocumentStatus.GATHERING.value,
                stage="gathering_context",
                progress=20,
                message="Gathering notes, tasks, and transcripts",
            )
            bundle = await self.sources.load(user_id, space_id)
            pack = build_document_context_pack(bundle, template=template)
            await self.store.update_progress(
                document_id,
                stage="packing_context",
                progress=35,
                message="Preparing context for generation",
                source_stats=bundle.sourceStats,
            )

            if not has_usable_context(pack):
                raise NonRetryableQueueError(
                    "No notes, tasks, or meeting transcripts found for this space yet."
                )

            await self.store.update_progress(
                document_id,
                status=DocumentStatus.GENERATING.value,
                stage="generating",
                progress=55,
                message=f"Generating {template.title} with AI",
            )
            content, model = await generate_document_content(
                pack,
                template=template,
                space_name=bundle.spaceName,
            )

            await self.store.update_progress(
                document_id,
                status=DocumentStatus.RENDERING.value,
                stage="rendering_docx",
                progress=80,
                message="Building Word document",
                model=model,
            )
            docx_bytes = build_docx_bytes(
                content,
                template_title=template.title,
                space_name=bundle.spaceName,
            )
            file_name = f"{_safe_file_stem(bundle.spaceName, template.title)}.docx"

            await self.store.update_progress(
                document_id,
                status=DocumentStatus.SAVING.value,
                stage="saving",
                progress=92,
                message="Saving document",
            )
            await self.store.save_ready(
                document_id,
                content=content,
                docx_bytes=docx_bytes,
                file_name=file_name,
                source_stats=bundle.sourceStats or SourceStats(),
                model=model,
                space_name=bundle.spaceName,
            )
            document_log(
                "job_ready",
                documentId=document_id,
                userId=user_id,
                spaceId=space_id,
                templateCode=template_code,
                sectionCount=len(content.sections),
                docxBytes=len(docx_bytes),
                model=model,
            )
        except NonRetryableQueueError as error:
            await self.store.mark_failed(document_id, str(error))
            document_log("job_failed_nonretryable", documentId=document_id, error=str(error)[:200])
            raise
        except DocumentGenerateError as error:
            await self.store.mark_failed(document_id, str(error))
            raise NonRetryableQueueError(str(error)) from error
        except Exception as error:
            await self.store.mark_failed(document_id, f"{type(error).__name__}: {error}")
            document_log(
                "job_failed",
                documentId=document_id,
                error=type(error).__name__,
                detail=str(error)[:200],
            )
            raise
