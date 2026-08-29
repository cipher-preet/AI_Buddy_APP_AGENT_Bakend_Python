from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

from apps.api_gateway.config.setting import settings
from services.conversation import agents
from services.conversation.artifact_resolver import reconcile_incoming_artifacts
from services.conversation.artifacts import artifacts_from_window, compact_artifact
from services.conversation.context import load_space_context
from services.conversation.meeting_memory import build_meeting_memory, select_context_for_window
from services.conversation.models import (
    ExtractionOutcome,
    STTStatus,
    TranscriptExclusionReason,
    TranscriptProcessingStatus,
    WindowExtractionResult,
    WindowProcessingStatus,
)
from services.conversation.repository import ConversationRepository
from services.conversation.stt_failure import is_terminal_failed_chunk
from services.conversation.windowing import (
    CLOSE_REASON_FORCED_FINAL,
    build_ready_windows,
    is_useful_chunk,
    leading_skippable_sequences,
)
from services.conversation.semantic_input import (
    as_sequence_number,
    assemble_semantic_window_input,
)
from services.conversation.event_pipeline import (
    event_pipeline_selected_for,
    event_pipeline_shadow,
    extract_window_events,
)
from services.conversation.event_pipeline.store import ConversationEventStore
from services.conversation.meeting_pipeline import meeting_pipeline_enabled
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
        skippable_sequences: set[int] | None = None,
    ) -> list[str]:
        if not settings.ENABLE_INCREMENTAL_MEETING_PROCESSING:
            return []
        conversation = await self.repository.get_conversation(conversation_id)
        if not conversation:
            return []
        all_chunks = await self.repository.list_transcript_chunks(conversation_id)
        if through_sequence is not None:
            through = as_sequence_number(through_sequence)
            all_chunks = [chunk for chunk in all_chunks if as_sequence_number(chunk.sequenceNumber) <= through]
        skippable = {as_sequence_number(item) for item in (skippable_sequences or ())}
        skippable.update(_derived_skippable_sequences(all_chunks))
        unwindowed = [
            chunk
            for chunk in all_chunks
            if chunk.sttStatus == STTStatus.COMPLETED
            and chunk.processingStatus == TranscriptProcessingStatus.UNPROCESSED
            and not chunk.exclusionReason
        ]
        existing_windows = await self.repository.list_conversation_windows(conversation_id)
        start_index = len(existing_windows)
        expected_start = as_sequence_number(existing_windows[-1].sequenceEnd) + 1 if existing_windows else 0
        unwindowed = [chunk for chunk in unwindowed if as_sequence_number(chunk.sequenceNumber) >= expected_start]
        useful_sequences = {
            as_sequence_number(chunk.sequenceNumber)
            for chunk in all_chunks
            if chunk.sttStatus == STTStatus.COMPLETED and is_useful_chunk(chunk)
        }
        covered = _sequences_covered_by_windows(existing_windows)
        if force_final and useful_sequences and useful_sequences <= covered:
            print(
                "Duplicate final window skipped; durable membership already covers useful transcripts:",
                {
                    "conversationId": conversation_id,
                    "usefulSequenceCount": len(useful_sequences),
                    "coveredSequenceCount": len(covered),
                    "existingWindowCount": len(existing_windows),
                },
            )
            return [str(window.id) for window in existing_windows]
        if not unwindowed and not skippable:
            return []

        failed_terminal = [
            as_sequence_number(chunk.sequenceNumber)
            for chunk in all_chunks
            if as_sequence_number(chunk.sequenceNumber) >= expected_start
            and chunk.sttStatus == STTStatus.FAILED
            and as_sequence_number(chunk.sequenceNumber) in skippable
        ]
        if failed_terminal:
            await self.repository.mark_transcripts_excluded(
                conversation_id,
                failed_terminal,
                TranscriptExclusionReason.STT_FAILED,
            )

        leading_empty = leading_skippable_sequences(unwindowed, skippable)
        if leading_empty:
            await self.repository.mark_transcripts_excluded(
                conversation_id,
                leading_empty,
                TranscriptExclusionReason.EMPTY_TRANSCRIPT,
            )
            unwindowed = [chunk for chunk in unwindowed if chunk.sequenceNumber not in set(leading_empty)]

        if unwindowed and as_sequence_number(unwindowed[0].sequenceNumber) != expected_start:
            hole = list(range(expected_start, as_sequence_number(unwindowed[0].sequenceNumber)))
            if any(sequence not in skippable for sequence in hole):
                return []
            present = {chunk.sequenceNumber for chunk in all_chunks}
            missing_hole = [sequence for sequence in hole if sequence not in present]
            failed_hole = [sequence for sequence in hole if sequence in present]
            if missing_hole:
                await self.repository.mark_transcripts_excluded(
                    conversation_id,
                    missing_hole,
                    TranscriptExclusionReason.SEQUENCE_MISSING,
                )
            if failed_hole:
                await self.repository.mark_transcripts_excluded(
                    conversation_id,
                    failed_hole,
                    TranscriptExclusionReason.STT_FAILED,
                )

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
            unwindowed,
            start_index,
            force_final=force_final,
            overlap_prefix=overlap_prefix,
            skippable_sequences=skippable,
        )
        window_ids: list[str] = []
        for built in built_windows:
            saved = await self.repository.create_conversation_window(
                built.window,
                built.owned_sequence_numbers,
                skipped_sequence_numbers=built.skipped_sequence_numbers,
            )
            window_ids.append(str(saved.id))
            retain_raw = saved.isFinalPartial or built.close_reason == CLOSE_REASON_FORCED_FINAL
            if retain_raw:
                await self.repository.complete_window(
                    saved.id,
                    WindowExtractionResult(isCheckpoint=False),
                    provider="none",
                    model="raw-passthrough",
                    artifact_count=0,
                    artifact_persistence_ok=True,
                    extraction_skipped=True,
                    checkpoint_kind="raw_final",
                )
                print(
                    "Final partial window retained as raw transcript:",
                    {
                        "conversationId": conversation_id,
                        "windowId": str(saved.id),
                        "windowIndex": saved.windowIndex,
                        "closeReason": saved.closeReason or built.close_reason,
                        "usefulTokenCount": saved.usefulTokenCount,
                        "meaningfulSpeechMs": saved.meaningfulSpeechMs,
                    },
                )
            elif saved.status in {WindowProcessingStatus.PENDING, WindowProcessingStatus.FAILED, WindowProcessingStatus.RETRYING} and saved.queuedAt is None:
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
                            "closeReason": built.close_reason,
                            "emptyChunkCount": saved.emptyChunkCount,
                            "nonEmptyChunkCount": saved.nonEmptyChunkCount,
                            "usefulTokenCount": saved.usefulTokenCount,
                        },
                    ),
                )
            print(
                "Window closed:",
                {
                    "conversationId": conversation_id,
                    "windowIndex": saved.windowIndex,
                    "sequenceStart": saved.sequenceStart,
                    "sequenceEnd": saved.sequenceEnd,
                    "sequenceCount": saved.sequenceCount,
                    "emptyChunkCount": saved.emptyChunkCount,
                    "nonEmptyChunkCount": saved.nonEmptyChunkCount,
                    "usefulTokenCount": saved.usefulTokenCount,
                    "usefulWordCount": saved.usefulWordCount,
                    "wallClockSpanMs": saved.wallClockSpanMs,
                    "meaningfulSpeechMs": saved.meaningfulSpeechMs,
                    "closeReason": saved.closeReason or built.close_reason,
                },
            )
            await self.repository.append_meeting_debug_trace(
                conversation_id,
                conversation.userId,
                conversation.spaceId,
                "window_closed",
                {
                    "windowId": str(saved.id),
                    "windowIndex": saved.windowIndex,
                    "sequenceStart": saved.sequenceStart,
                    "sequenceEnd": saved.sequenceEnd,
                    "ownedSequences": built.owned_sequence_numbers,
                    "skippedSequences": built.skipped_sequence_numbers,
                    "closeReason": built.close_reason,
                    "emptyChunkCount": saved.emptyChunkCount,
                    "nonEmptyChunkCount": saved.nonEmptyChunkCount,
                    "usefulTokenCount": saved.usefulTokenCount,
                },
            )
        if force_final:
            leftover_empty = [
                chunk.sequenceNumber
                for chunk in unwindowed
                if not is_useful_chunk(chunk) and not chunk.exclusionReason
            ]
            if leftover_empty:
                await self.repository.mark_transcripts_excluded(
                    conversation_id,
                    leftover_empty,
                    TranscriptExclusionReason.EMPTY_TRANSCRIPT,
                )
        if built_windows:
            print(
                "Incremental windows closed:",
                {
                    "conversationId": conversation_id,
                    "windowCount": len(built_windows),
                    "forceFinal": force_final,
                    "overlapPrefixCount": len(overlap_prefix),
                    "skippableCount": len(skippable),
                },
            )
        return window_ids

    async def extract_window(self, window_id: str, recovery: bool = False) -> None:
        started = time.perf_counter()
        existing = await self.repository.get_conversation_window(window_id)
        if existing:
            conversation = await self.repository.get_conversation(str(existing.conversationId))
            if conversation and conversation.status.value in {"COMPLETED", "FAILED"}:
                print(
                    "Late window extraction skipped after terminal conversation status:",
                    {
                        "conversationId": str(existing.conversationId),
                        "windowId": window_id,
                        "status": conversation.status.value,
                    },
                )
                return
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
            if meeting_pipeline_enabled():
                result = WindowExtractionResult(
                    extractionOutcome=ExtractionOutcome.SUCCESS,
                    extractionDiagnostics={
                        "meetingPipelineDeferredToFinalization": True,
                        "pipelineMode": "meeting_pipeline",
                    },
                )
                await self.repository.complete_window(
                    window.id,
                    result,
                    "meeting-pipeline",
                    "deferred-to-finalization",
                    artifact_count=0,
                    artifact_persistence_ok=True,
                    extraction_skipped=True,
                    checkpoint_kind="deferred_meeting_pipeline",
                )
                print(
                    "Window extraction deferred to meeting pipeline finalization:",
                    {
                        "conversationId": str(window.conversationId),
                        "windowId": str(window.id),
                        "windowIndex": window.windowIndex,
                    },
                )
                return
            context = await load_space_context(self.repository, window.userId, window.spaceId)
            meeting_context, existing_artifacts = await self._meeting_context_for_window(window)
            window = await self._attach_semantic_window_input(window)
            shadow_events = None
            selected = event_pipeline_selected_for(str(window.userId), str(window.conversationId))
            if selected:
                result, provider, model = await self._extract_window_events(window)
            else:
                if event_pipeline_shadow():
                    try:
                        shadow_events, _, _ = await self._extract_window_events(window)
                    except Exception as error:
                        print(
                            "Event pipeline shadow window extraction failed; legacy continues:",
                            {"windowId": str(window.id), "error": str(error)[:300]},
                        )
                result, provider, model = await agents.extract_window(
                    self.router,
                    window,
                    context,
                    meeting_context=meeting_context,
                    recovery=recovery,
                )
            if result.extractionOutcome in {
                ExtractionOutcome.EXTRACTION_FAILED,
                ExtractionOutcome.SEMANTIC_INPUT_ASSEMBLY_FAILED,
            }:
                message = result.extractionError or "window extraction failed after structured-output recovery"
                await self.repository.fail_window(window.id, message)
                raise RuntimeError(message)
            incoming = [] if selected else artifacts_from_window(window, result)
            resolved = existing_artifacts
            if incoming:
                resolved = await reconcile_incoming_artifacts(self.router, existing_artifacts, incoming, window.text)
                await self.repository.upsert_meeting_artifacts(resolved)
            persisted = await self.repository.count_meeting_artifacts_for_window(str(window.conversationId), window.id) if incoming else 0
            if incoming and persisted <= 0:
                await self.repository.fail_window(window.id, "artifact persistence failed")
                print(
                    "Window extraction artifact persistence failed:",
                    {"conversationId": str(window.conversationId), "windowId": str(window.id), "incoming": len(incoming)},
                )
                return
            if incoming:
                await self._refresh_meeting_memory(str(window.conversationId), window.userId, window.spaceId)
            if shadow_events is not None:
                diagnostics = dict(getattr(result, "extractionDiagnostics", None) or {})
                diagnostics["eventPipelineShadow"] = {
                    "atomicEventCount": len(getattr(shadow_events, "atomicEvents", None) or []),
                    "checkpointKind": "atomic_events",
                    "publishedFrom": "legacy",
                }
                result.extractionDiagnostics = diagnostics
            await self.repository.complete_window(
                window.id,
                result,
                provider,
                model,
                artifact_count=len(incoming),
                artifact_persistence_ok=True,
                checkpoint_kind="atomic_events" if selected else "semantic_checkpoint",
            )
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            metrics = {
                "conversationId": str(window.conversationId),
                "windowId": str(window.id),
                "windowIndex": window.windowIndex,
                "pipelineStages": [
                    "context_reconstruction",
                    "conversation_understanding",
                    "candidate_extraction",
                    "evidence_retrieval",
                    "enrichment",
                    "semantic_deduplication_merge",
                    "critic_validator",
                    "deterministic_confidence",
                    "final_publish",
                    "memory_update",
                ],
                "tokenCount": window.tokenCount,
                "usefulTokenCount": window.usefulTokenCount,
                "emptyChunkCount": window.emptyChunkCount,
                "nonEmptyChunkCount": window.nonEmptyChunkCount,
                "closeReason": window.closeReason,
                "chunkCount": window.sequenceEnd - window.sequenceStart + 1,
                "elapsedMs": elapsed_ms,
                "provider": provider,
                "model": model,
                "retryCount": window.attemptCount,
                "artifactsExtracted": len(incoming),
                "artifactsPersisted": persisted,
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
                    "ownedSequences": list(range(window.sequenceStart, window.sequenceEnd + 1)),
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

    async def _extract_window_events(self, window):
        chunks = await self.repository.list_transcript_chunks_in_range(
            str(window.conversationId),
            window.sequenceStart,
            window.sequenceEnd,
        )
        store = ConversationEventStore(self.repository)
        result, provider, model = await extract_window_events(
            chunks,
            str(window.conversationId),
            str(window.userId),
            str(window.spaceId),
            router=self.router,
            event_store=store,
            repository=self.repository,
        )
        result.extractionDiagnostics = {
            **(result.extractionDiagnostics or {}),
            "checkpointKind": "atomic_events",
            "windowId": str(window.id),
            "windowIndex": window.windowIndex,
        }
        return result, provider, model

    async def _attach_semantic_window_input(self, window):
        chunks = await self.repository.list_transcript_chunks(str(window.conversationId))
        assembly = assemble_semantic_window_input(
            conversation_id=str(window.conversationId),
            chunks=chunks,
            windows=[window],
            sequence_start=window.sequenceStart,
            sequence_end=window.sequenceEnd,
            mode="window_range",
        )
        text = assembly.text or window.text
        attached = SimpleNamespace(**window.model_dump(by_alias=False), semanticInputDiagnostics=assembly.diagnostics)
        attached.id = window.id
        attached.text = text
        attached.result = window.result
        attached.status = window.status
        return attached

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
        window = await self._attach_semantic_window_input(window)
        result, provider, model = await agents.extract_window(
            self.router,
            window,
            context,
            meeting_context=meeting_context,
            recovery=True,
        )
        if result.extractionOutcome in {
            ExtractionOutcome.EXTRACTION_FAILED,
            ExtractionOutcome.SEMANTIC_INPUT_ASSEMBLY_FAILED,
        }:
            message = result.extractionError or "window recovery extraction failed after structured-output recovery"
            await self.repository.fail_window(window.id, message)
            raise RuntimeError(message)
        incoming = artifacts_from_window(window, result)
        resolved = await reconcile_incoming_artifacts(self.router, existing_artifacts, incoming, window.text)
        await self.repository.upsert_meeting_artifacts(resolved)
        await self._refresh_meeting_memory(str(window.conversationId), window.userId, window.spaceId)
        if window.result is None:
            persisted = await self.repository.count_meeting_artifacts_for_window(str(window.conversationId), window.id)
            if incoming and persisted <= 0:
                await self.repository.fail_window(window.id, "artifact persistence failed during recovery")
                return
            await self.repository.complete_window(
                window.id,
                result,
                provider,
                model,
                artifact_count=len(incoming),
                artifact_persistence_ok=True,
                checkpoint_kind="semantic_checkpoint",
            )
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
        "usefulTokenCount": getattr(window, "usefulTokenCount", 0),
        "closeReason": getattr(window, "closeReason", None),
        "isFinalPartial": bool(getattr(window, "isFinalPartial", False)),
        "extractionSkipped": bool(getattr(window, "extractionSkipped", False)),
        "checkpointKind": getattr(window, "checkpointKind", None),
        "narrative": ((result.narrative if result else "") or (result.summary if result else ""))[:800],
        "summary": (result.summary if result else "")[:400],
        "topics": result.topics if result else [],
        "semanticUnits": [unit.model_dump() for unit in getattr(result, "semanticUnits", [])] if result else [],
        "importantFacts": result.importantFacts if result else [],
        "openQuestions": result.openQuestions if result else [],
    }


def compact_artifact_payload(artifacts: list) -> list[dict[str, Any]]:
    return [compact_artifact(artifact) for artifact in artifacts]


def _sequences_covered_by_windows(windows) -> set[int]:
    covered: set[int] = set()
    for window in windows or []:
        start = as_sequence_number(getattr(window, "sequenceStart", 0), 0)
        end = as_sequence_number(getattr(window, "sequenceEnd", 0), 0)
        if end >= start:
            covered.update(range(start, end + 1))
    return covered


def _derived_skippable_sequences(chunks) -> set[int]:
    skippable: set[int] = set()
    for chunk in chunks:
        if chunk.exclusionReason:
            skippable.add(chunk.sequenceNumber)
            continue
        if chunk.sttStatus == STTStatus.COMPLETED and not is_useful_chunk(chunk):
            skippable.add(chunk.sequenceNumber)
            continue
        if chunk.sttStatus == STTStatus.FAILED and is_terminal_failed_chunk(chunk):
            skippable.add(chunk.sequenceNumber)
    return skippable
