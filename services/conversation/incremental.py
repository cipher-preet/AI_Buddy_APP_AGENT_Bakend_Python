from __future__ import annotations

import time
from typing import Any

from apps.api_gateway.config.setting import settings
from services.conversation import agents
from services.conversation.artifact_resolver import resolve_incoming_artifacts
from services.conversation.artifacts import artifacts_from_window, compact_artifact
from services.conversation.context import load_space_context
from services.conversation.meeting_memory import build_meeting_memory, select_context_for_window
from services.conversation.models import WindowProcessingStatus
from services.conversation.repository import ConversationRepository
from services.conversation.windowing import build_ready_windows
from services.llm.router import LLMRouter, get_llm_router
from services.queue.streams import EventEnvelope, RedisStreamProducer


class IncrementalMeetingProcessor:
    def __init__(
        self,
        repository: ConversationRepository,
        producer: RedisStreamProducer | None = None,
        router: LLMRouter | None = None,
    ):
        self.repository = repository
        self.producer = producer or RedisStreamProducer()
        self.router = router or get_llm_router()

    async def close_ready_windows(
        self,
        conversation_id: str,
        force_final: bool = False,
        through_sequence: int | None = None,
    ) -> list[str]:
        if not settings.ENABLE_INCREMENTAL_MEETING_PROCESSING:
            return []
        conversation = await self.repository.get_conversation(conversation_id)
        if not conversation:
            return []
        chunks = await self.repository.list_completed_unwindowed_transcript_chunks(
            conversation_id,
            through_sequence=through_sequence,
        )
        existing_windows = await self.repository.list_conversation_windows(conversation_id)
        start_index = len(existing_windows)
        expected_start = existing_windows[-1].sequenceEnd + 1 if existing_windows else 0
        chunks = [chunk for chunk in chunks if chunk.sequenceNumber >= expected_start]
        if chunks and chunks[0].sequenceNumber != expected_start:
            return []
        overlap_prefix = []
        if existing_windows:
            last = existing_windows[-1]
            if last.overlapSequenceStart is not None:
                overlap_prefix = await self.repository.list_transcript_chunks_in_range(
                    conversation_id,
                    last.overlapSequenceStart,
                    last.sequenceEnd,
                )
        built_windows = build_ready_windows(
            conversation,
            chunks,
            start_index,
            force_final=force_final,
            overlap_prefix=overlap_prefix,
        )
        window_ids: list[str] = []
        for built in built_windows:
            saved = await self.repository.create_conversation_window(built.window, built.owned_sequence_numbers)
            window_ids.append(str(saved.id))
            if saved.status in {WindowProcessingStatus.PENDING, WindowProcessingStatus.FAILED}:
                await self.repository.mark_window_queued(saved.id)
                await self.producer.publish(
                    settings.REDIS_WINDOW_EXTRACTION_STREAM,
                    EventEnvelope(
                        eventType="conversation.window.extraction.requested",
                        correlationId=conversation_id,
                        userId=str(conversation.userId),
                        spaceId=str(conversation.spaceId),
                        conversationId=conversation_id,
                        payload={
                            "windowId": str(saved.id),
                            "windowIndex": saved.windowIndex,
                            "sequenceStart": saved.sequenceStart,
                            "sequenceEnd": saved.sequenceEnd,
                            "ownedSequenceCount": len(built.owned_sequence_numbers),
                            "overlapSequenceCount": max(0, len(built.sequence_numbers) - len(built.owned_sequence_numbers)),
                        },
                    ),
                )
        if built_windows:
            print(
                "Incremental windows closed:",
                {
                    "conversationId": conversation_id,
                    "windowCount": len(built_windows),
                    "forceFinal": force_final,
                    "overlapPrefixCount": len(overlap_prefix),
                },
            )
        return window_ids

    async def extract_window(self, window_id: str, recovery: bool = False) -> None:
        started = time.perf_counter()
        window = await self.repository.mark_window_processing(window_id)
        if not window:
            existing = await self.repository.get_conversation_window(window_id)
            if existing and existing.status == WindowProcessingStatus.COMPLETED:
                if recovery:
                    await self._recover_window(existing)
                    return
                print(
                    "Duplicate window extraction skipped:",
                    {"conversationId": str(existing.conversationId), "windowId": window_id},
                )
                return
            return
        try:
            context = await load_space_context(self.repository, window.userId, window.spaceId)
            meeting_context, existing_artifacts = await self._meeting_context_for_window(window)
            result, provider, model = await agents.extract_window(
                self.router,
                window,
                context,
                meeting_context=meeting_context,
                recovery=recovery,
            )
            incoming = artifacts_from_window(window, result)
            resolved = resolve_incoming_artifacts(existing_artifacts, incoming)
            await self.repository.upsert_meeting_artifacts(resolved)
            await self._refresh_meeting_memory(str(window.conversationId), window.userId, window.spaceId)
            await self.repository.complete_window(window.id, result, provider, model)
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            metrics = {
                "conversationId": str(window.conversationId),
                "windowId": str(window.id),
                "windowIndex": window.windowIndex,
                "tokenCount": window.tokenCount,
                "chunkCount": window.sequenceEnd - window.sequenceStart + 1,
                "elapsedMs": elapsed_ms,
                "provider": provider,
                "model": model,
                "retryCount": window.attemptCount,
                "artifactsExtracted": len(incoming),
                "artifactCountAfterResolve": len(resolved),
                "taskCount": len(result.tasks),
                "noteCount": len(result.notes),
                "decisionCount": len(result.decisions),
                "issueCount": len(result.issues),
                "recovery": recovery,
            }
            print("Window extraction completed:", metrics)
            await self.repository.append_meeting_debug_trace(
                str(window.conversationId),
                window.userId,
                window.spaceId,
                "window_extraction",
                {
                    **metrics,
                    "artifactIds": [str(item.id) for item in incoming],
                },
            )
        except Exception as error:
            await self.repository.fail_window(window.id, error)
            print(
                "Window extraction failed:",
                {
                    "conversationId": str(window.conversationId),
                    "windowId": str(window.id),
                    "windowIndex": window.windowIndex,
                    "error": str(error)[:500],
                },
            )
            raise

    async def recover_windows(self, conversation_id: str, window_indexes: list[int]) -> int:
        if not window_indexes:
            return 0
        windows = await self.repository.list_conversation_windows(conversation_id)
        recovered = 0
        wanted = set(window_indexes[: settings.SELECTIVE_RECOVERY_MAX_WINDOWS])
        for window in windows:
            if window.windowIndex not in wanted:
                continue
            try:
                await self._recover_window(window)
                recovered += 1
            except Exception as error:
                print(
                    "Selective window recovery failed:",
                    {
                        "conversationId": conversation_id,
                        "windowId": str(window.id),
                        "windowIndex": window.windowIndex,
                        "error": str(error)[:500],
                    },
                )
        return recovered

    async def _recover_window(self, window) -> None:
        context = await load_space_context(self.repository, window.userId, window.spaceId)
        meeting_context, existing_artifacts = await self._meeting_context_for_window(window)
        result, provider, model = await agents.extract_window(
            self.router,
            window,
            context,
            meeting_context=meeting_context,
            recovery=True,
        )
        incoming = artifacts_from_window(window, result)
        resolved = resolve_incoming_artifacts(existing_artifacts, incoming)
        await self.repository.upsert_meeting_artifacts(resolved)
        await self._refresh_meeting_memory(str(window.conversationId), window.userId, window.spaceId)
        if window.result is None:
            await self.repository.complete_window(window.id, result, provider, model)
        print(
            "Selective window recovery completed:",
            {
                "conversationId": str(window.conversationId),
                "windowId": str(window.id),
                "windowIndex": window.windowIndex,
                "artifactsExtracted": len(incoming),
                "provider": provider,
                "model": model,
            },
        )
        await self.repository.append_meeting_debug_trace(
            str(window.conversationId),
            window.userId,
            window.spaceId,
            "selective_recovery",
            {
                "windowId": str(window.id),
                "windowIndex": window.windowIndex,
                "artifactsExtracted": len(incoming),
            },
        )

    async def _meeting_context_for_window(self, window) -> tuple[dict[str, Any], list]:
        conversation_id = str(window.conversationId)
        memory = await self.repository.get_meeting_memory(conversation_id)
        artifacts = await self.repository.list_meeting_artifacts(conversation_id)
        context = select_context_for_window(memory, artifacts, window.text, window.result.topics if window.result else [])
        return context, artifacts

    async def _refresh_meeting_memory(self, conversation_id: str, user_id: Any, space_id: Any) -> None:
        artifacts = await self.repository.list_meeting_artifacts(conversation_id)
        windows = await self.repository.list_conversation_windows(conversation_id)
        previous = await self.repository.get_meeting_memory(conversation_id)
        memory = build_meeting_memory(conversation_id, user_id, space_id, artifacts, windows, previous)
        await self.repository.save_meeting_memory(memory)


def compact_window_payload(window: Any) -> dict[str, Any]:
    result = window.result
    return {
        "windowIndex": window.windowIndex,
        "sequenceStart": window.sequenceStart,
        "sequenceEnd": window.sequenceEnd,
        "summary": result.summary if result else "",
        "topics": result.topics if result else [],
        "importantFacts": result.importantFacts if result else [],
        "tasks": [item.model_dump() for item in result.tasks] if result else [],
        "notes": [item.model_dump() for item in result.notes] if result else [],
        "decisions": [item.model_dump() for item in result.decisions] if result else [],
        "issues": [item.model_dump() for item in result.issues] if result else [],
        "openQuestions": result.openQuestions if result else [],
        "artifactCount": len(result.tasks) + len(result.notes) + len(result.decisions) + len(result.issues) if result else 0,
    }


def compact_window_summary(window: Any) -> dict[str, Any]:
    result = window.result
    return {
        "windowIndex": window.windowIndex,
        "sequenceStart": window.sequenceStart,
        "sequenceEnd": window.sequenceEnd,
        "tokenCount": window.tokenCount,
        "summary": (result.summary if result else "")[:400],
        "topics": result.topics if result else [],
    }


def compact_artifact_payload(artifacts: list) -> list[dict[str, Any]]:
    return [compact_artifact(artifact) for artifact in artifacts]
