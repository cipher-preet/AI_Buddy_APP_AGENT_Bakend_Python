from __future__ import annotations

from services.mindmap.context_pack import build_context_pack, has_usable_context
from services.mindmap.generator import MindmapGenerateError, generate_mindmap_graph
from services.mindmap.log import mindmap_log
from services.mindmap.schemas import MindmapStatus, SourceStats
from services.mindmap.sources import MindmapSource
from services.mindmap.store import MindmapStore
from services.queue.streams import EventEnvelope, NonRetryableQueueError


class MindmapJobHandler:
    def __init__(self, database):
        self.database = database
        self.store = MindmapStore(database)
        self.sources = MindmapSource(database)

    async def handle(self, event: EventEnvelope) -> None:
        payload = event.payload or {}
        mindmap_id = str(payload.get("mindmapId") or "").strip()
        user_id = str(event.userId or "").strip()
        space_id = str(event.spaceId or "").strip()
        if not mindmap_id or not user_id or not space_id:
            raise NonRetryableQueueError("mindmap job missing mindmapId/userId/spaceId")

        mindmap_log(
            "job_start",
            mindmapId=mindmap_id,
            userId=user_id,
            spaceId=space_id,
            jobId=event.eventId,
            attempt=event.attempt,
        )

        claim_state, doc = await self.store.claim(mindmap_id, event.eventId)
        if claim_state == "missing":
            raise NonRetryableQueueError(f"mindmap document not found: {mindmap_id}")
        if claim_state == "exists":
            return
        if claim_state == "busy":
            mindmap_log("job_busy", mindmapId=mindmap_id, jobId=event.eventId)
            return

        try:
            await self.store.update_progress(
                mindmap_id,
                status=MindmapStatus.GATHERING.value,
                stage="gathering_context",
                progress=20,
                message="Gathering notes, tasks, and transcripts",
            )
            bundle = await self.sources.load(user_id, space_id)
            pack = build_context_pack(bundle)
            await self.store.update_progress(
                mindmap_id,
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
                mindmap_id,
                status=MindmapStatus.GENERATING.value,
                stage="generating",
                progress=55,
                message="Generating mind map with AI",
            )
            graph, model = await generate_mindmap_graph(pack, space_name=bundle.spaceName)

            await self.store.update_progress(
                mindmap_id,
                status=MindmapStatus.SAVING.value,
                stage="saving",
                progress=90,
                message="Saving mind map",
                model=model,
            )
            await self.store.save_ready(
                mindmap_id,
                graph=graph,
                source_stats=bundle.sourceStats or SourceStats(),
                model=model,
            )
            mindmap_log(
                "job_ready",
                mindmapId=mindmap_id,
                userId=user_id,
                spaceId=space_id,
                nodeCount=len(graph.nodes),
                model=model,
            )
        except NonRetryableQueueError as error:
            await self.store.mark_failed(mindmap_id, str(error))
            mindmap_log("job_failed_nonretryable", mindmapId=mindmap_id, error=str(error)[:200])
            raise
        except MindmapGenerateError as error:
            await self.store.mark_failed(mindmap_id, str(error))
            raise NonRetryableQueueError(str(error)) from error
        except Exception as error:
            await self.store.mark_failed(mindmap_id, f"{type(error).__name__}: {error}")
            mindmap_log(
                "job_failed",
                mindmapId=mindmap_id,
                error=type(error).__name__,
                detail=str(error)[:200],
            )
            raise
