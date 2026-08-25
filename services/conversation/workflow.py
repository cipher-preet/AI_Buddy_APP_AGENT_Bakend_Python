from __future__ import annotations

import asyncio
import re

from apps.api_gateway.config.setting import settings
from services.conversation import agents
from services.conversation.context import load_space_context
from services.conversation.fingerprints import note_fingerprint, task_fingerprint
from services.conversation.intelligence import score_and_filter_result
from services.conversation.incremental import IncrementalMeetingProcessor
from services.conversation.models import ConversationStatus, ConversationSummaryDocument, ExtractionOutcome, ExtractionRunStatus, WindowProcessingStatus
from services.conversation.repository import ConversationRepository
from services.conversation.transcript import assemble_transcript, estimate_tokens, segment_transcript
from services.conversation.semantic_input import (
    SEMANTIC_INPUT_ASSEMBLY_FAILED,
    assemble_semantic_window_input,
)
from services.conversation.workflow_state import ConversationGraphState
from services.llm.router import LLMCapability, LLMRouter, get_llm_router
from services.queue.streams import EventEnvelope, RedisStreamProducer


class ConversationProcessingWorkflow:
    def __init__(
        self,
        repository: ConversationRepository,
        router: LLMRouter | None = None,
    ):
        self.repository = repository
        self.router = router or get_llm_router()

    async def run(self, conversation_id: str) -> None:
        conversation = await self.repository.get_conversation(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation not found: {conversation_id}")
        if conversation.status == ConversationStatus.COMPLETED:
            return
        if conversation.status in {
            ConversationStatus.STOP_REQUESTED,
            ConversationStatus.WAITING_FOR_TRANSCRIPTS,
            ConversationStatus.FINALIZING,
        }:
            await RedisStreamProducer().publish(
                settings.REDIS_FINALIZATION_STREAM,
                EventEnvelope(
                    eventType="conversation.finalization.requested",
                    correlationId=conversation_id,
                    userId=str(conversation.userId),
                    spaceId=str(conversation.spaceId),
                    conversationId=conversation_id,
                    payload={"expectedLastSequence": conversation.expectedLastSequence, "source": "processing-not-ready"},
                ),
            )
            return
        if conversation.status in {ConversationStatus.FAILED, ConversationStatus.VALIDATING}:
            conversation = await self.repository.transition(
                conversation_id,
                ConversationStatus.RETRY_PENDING,
                {"missingSequences": []},
            )
        if conversation.status not in {
            ConversationStatus.READY_FOR_PROCESSING,
            ConversationStatus.PARTIAL,
            ConversationStatus.PROCESSING,
            ConversationStatus.RETRY_PENDING,
        }:
            raise ValueError(f"Conversation is not ready for processing: {conversation.status.value}")
        run = None
        try:
            provider, model = self.router.route(LLMCapability.HIGH_ACCURACY_REASONING)
            run = await self.repository.create_extraction_run(conversation, provider.name, model)
            run.status = ExtractionRunStatus.PROCESSING
            await self.repository.transition(
                conversation_id,
                ConversationStatus.PROCESSING,
                {"activeExtractionRunId": run.id},
            )

            if settings.ENABLE_INCREMENTAL_MEETING_PROCESSING:
                windows = await self.repository.list_conversation_windows(conversation_id)
                unwindowed = await self.repository.count_unwindowed_non_empty_transcripts(conversation_id)
                incomplete = [
                    window
                    for window in windows
                    if window.status != WindowProcessingStatus.COMPLETED and not window.extractionSkipped
                ]
                if unwindowed or incomplete:
                    print(
                        "Processing deferred until drain completes:",
                        {
                            "conversationId": conversation_id,
                            "unwindowedNonEmpty": unwindowed,
                            "incompleteWindows": len(incomplete),
                            "windowCount": len(windows),
                        },
                    )
                    if conversation.status != ConversationStatus.FINALIZING:
                        await self.repository.transition(conversation_id, ConversationStatus.FINALIZING)
                    await RedisStreamProducer().publish(
                        settings.REDIS_FINALIZATION_STREAM,
                        EventEnvelope(
                            eventType="conversation.finalization.requested",
                            correlationId=conversation_id,
                            userId=str(conversation.userId),
                            spaceId=str(conversation.spaceId),
                            conversationId=conversation_id,
                            payload={"expectedLastSequence": conversation.expectedLastSequence, "source": "processing-drain-guard"},
                        ),
                    )
                    return
                if windows:
                    checkpoints = [window for window in windows if _is_completed_checkpoint(window)]
                    if not checkpoints:
                        await self._run_short_session_finalization(conversation, run, windows)
                        return
                    await self._run_incremental_finalization(conversation, run, windows, context=None)
                    return

            chunks = await self.repository.list_transcript_chunks(conversation_id)
            assembled = assemble_transcript(chunks)
            context = await load_space_context(self.repository, conversation.userId, conversation.spaceId)
            segments = segment_transcript(
                conversation_id,
                chunks,
                settings.TRANSCRIPT_SEGMENT_TARGET_TOKENS,
                settings.TRANSCRIPT_SEGMENT_OVERLAP_RATIO,
                settings.MAX_TRANSCRIPT_SEGMENTS,
            )
            run.segmentCount = len(segments)
            run.checkpoints["assembled_transcript"] = {"chunkCount": len(chunks)}
            run.checkpoints["loaded_space_context"] = {
                "activeTaskCount": len(context["activeTasks"]),
                "recentSummaryCount": len(context["recentSummaries"]),
            }
            run.checkpoints["segmented_transcript"] = {"segmentCount": len(segments)}
            await self.repository.save_extraction_run(run)
            state = ConversationGraphState(
                conversation_id=conversation_id,
                user_id=str(conversation.userId),
                space_id=str(conversation.spaceId),
                processing_version=conversation.processingVersion,
                extraction_run_id=str(run.id),
                conversation_status=ConversationStatus.PROCESSING,
                raw_transcript=assembled.raw_transcript,
                normalized_transcript=assembled.normalized_transcript,
                segments=segments,
                space_memory=context["spaceMemory"],
                relevant_previous_summaries=context["recentSummaries"],
                active_tasks=context["activeTasks"],
            )

            section_results = await self._parallel_section_extraction(state, context)
            state.section_results = section_results
            self._merge(state)
            await self._review_extracted_outputs(state, context)
            self._deterministic_validate(state)
            run.processedSegmentCount = len(section_results)
            run.checkpoints["parallel_section_extraction"] = {"processedSegmentCount": len(section_results)}
            run.checkpoints["merge_results"] = {
                "taskCount": len(state.merged_tasks),
                "noteCount": len(state.merged_notes),
                "decisionCount": len(state.merged_decisions),
                "issueCount": len(state.merged_questions) + len(state.merged_blockers),
            }
            await self.repository.save_extraction_run(run)
            outputs = self._outputs(state)
            state.coverage_report = await agents.validate_coverage(
                self.router,
                state.normalized_transcript,
                outputs,
                context,
            )
            if state.coverage_report.criticalMissingCount:
                await self._repair_coverage_gaps(state, context, outputs)
                outputs = self._outputs(state)
                state.coverage_report = await agents.validate_coverage(
                    self.router,
                    state.normalized_transcript,
                    outputs,
                    context,
                )
                if state.coverage_report.criticalMissingCount:
                    state.validation_errors.append(
                        {
                            "code": "CRITICAL_COVERAGE_GAP",
                            "coverage": state.coverage_report.model_dump(),
                        }
                    )
            run.coverageScore = state.coverage_report.score if state.coverage_report else None
            run.validationErrors = state.validation_errors
            run.warningCount = len(state.warnings)
            run.stagedTasks = state.merged_tasks
            run.stagedNotes = state.merged_notes
            run.stagedDecisions = state.merged_decisions
            run.stagedIssues = state.merged_questions + state.merged_blockers
            run.checkpoints["coverage_validation"] = {
                "score": run.coverageScore,
                "validationErrorCount": len(run.validationErrors),
            }

            await self.repository.transition(conversation_id, ConversationStatus.VALIDATING)
            summary = await agents.summarize_conversation(
                self.router,
                conversation_id,
                conversation.userId,
                conversation.spaceId,
                state.normalized_transcript,
                outputs,
                conversation.processingVersion,
            )
            previous_memory = await self.repository.get_space_memory(conversation.userId, conversation.spaceId)
            memory = await agents.update_space_memory(self.router, previous_memory, summary)
            await self.repository.publish_outputs(run, summary, memory)
            await self.repository.schedule_transcript_expiry(conversation_id)
            run.status = ExtractionRunStatus.PUBLISHED
            await self.repository.save_extraction_run(run)
            terminal_status = ConversationStatus.PARTIAL if state.validation_errors else ConversationStatus.COMPLETED
            await self.repository.transition(conversation_id, terminal_status)
        except Exception as error:
            if run is not None:
                await self.repository.mark_extraction_run_failed(run.id, error)
            latest = await self.repository.get_conversation(conversation_id)
            if latest and latest.status != ConversationStatus.COMPLETED:
                await self.repository.transition(conversation_id, ConversationStatus.FAILED)
            raise

    async def _parallel_section_extraction(self, state: ConversationGraphState, context: dict) -> list:
        semaphore = asyncio.Semaphore(settings.MAX_ACTIVE_LLM_CALLS_PER_CONVERSATION)

        async def run_segment(segment):
            async with semaphore:
                return await agents.extract_segment(self.router, segment, context, state.user_id, state.space_id)

        return await asyncio.gather(*(run_segment(segment) for segment in state.segments))

    async def _run_short_session_finalization(self, conversation, run, windows) -> None:
        started = utc_ms()
        conversation_id = str(conversation.id)
        context = await load_space_context(self.repository, conversation.userId, conversation.spaceId)
        chunks = await self.repository.list_transcript_chunks(conversation_id)
        assembly = assemble_semantic_window_input(
            conversation_id=conversation_id,
            chunks=chunks,
            windows=windows,
            mode="final_raw",
        )
        if assembly.failed:
            raise RuntimeError(SEMANTIC_INPUT_ASSEMBLY_FAILED)
        transcript = assembly.text
        result, provider, model = await agents.extract_from_raw_transcript(
            self.router,
            conversation_id,
            str(conversation.userId),
            str(conversation.spaceId),
            transcript,
            context,
            sequence_start=assembly.sequence_start,
            sequence_end=assembly.sequence_end,
            window_index=assembly.window_index,
            window_id=assembly.window_id,
            semantic_input_diagnostics=assembly.diagnostics,
        )
        if result.extractionOutcome in {
            ExtractionOutcome.EXTRACTION_FAILED,
            ExtractionOutcome.SEMANTIC_INPUT_ASSEMBLY_FAILED,
        }:
            raise RuntimeError(result.extractionError or "raw transcript extraction failed after structured-output recovery")
        result = await self._quality_review_and_repair(result, transcript, context, conversation_id, str(conversation.spaceId))
        diagnostics = dict(result.extractionDiagnostics or {})
        diagnostics["qualityAcceptedTaskCount"] = len(result.tasks)
        diagnostics["qualityAcceptedNoteCount"] = len(result.notes)
        result.extractionDiagnostics = diagnostics
        await self._publish_final_result(
            conversation,
            run,
            result,
            provider,
            model,
            windows,
            started,
            path="short_raw_transcript",
        )

    async def _run_incremental_finalization(self, conversation, run, windows, context: dict | None = None) -> None:
        started = utc_ms()
        conversation_id = str(conversation.id)
        context = context or await load_space_context(self.repository, conversation.userId, conversation.spaceId)
        ordered_windows = sorted(windows, key=lambda window: (window.sequenceStart, window.windowIndex))
        artifacts = await self.repository.list_meeting_artifacts(conversation_id)
        artifacts = await self._ensure_window_artifacts(conversation_id, ordered_windows, artifacts)

        from services.conversation.coverage import evaluate_coverage, preserve_unrepresented
        from services.conversation.incremental import compact_artifact_payload, compact_window_summary
        from services.conversation.meeting_memory import build_meeting_memory

        memory = await self.repository.get_meeting_memory(conversation_id)
        if memory is None:
            memory = build_meeting_memory(
                conversation_id,
                conversation.userId,
                conversation.spaceId,
                artifacts,
                ordered_windows,
                None,
            )
            await self.repository.save_meeting_memory(memory)

        checkpoints = [compact_window_summary(window) for window in ordered_windows if _is_completed_checkpoint(window)]
        leftover_windows = [
            window for window in ordered_windows if window.isFinalPartial or window.extractionSkipped
        ]
        leftover_chunks = await self.repository.list_transcript_chunks(conversation_id)
        leftover_assembly = assemble_semantic_window_input(
            conversation_id=conversation_id,
            chunks=leftover_chunks,
            windows=ordered_windows,
            mode="leftover",
        )
        if leftover_assembly.failed:
            raise RuntimeError(SEMANTIC_INPUT_ASSEMBLY_FAILED)
        leftover = leftover_assembly.text or "\n\n".join(
            window.text for window in leftover_windows
        ).strip()
        artifact_payload = compact_artifact_payload(artifacts)
        window_summaries = [compact_window_summary(window) for window in ordered_windows]
        accounting = conversation.lastAccounting or {}
        result, provider, model = await agents.reconcile_meeting(
            self.router,
            conversation_id,
            str(conversation.userId),
            str(conversation.spaceId),
            artifact_payload,
            window_summaries,
            memory.model_dump(),
            context,
            conversation.processingVersion,
            leftover_raw=leftover,
            checkpoints=checkpoints,
        )
        llm_coverage = evaluate_coverage(ordered_windows, artifacts, result)
        recovery_triggered = False
        if llm_coverage.suspicious and llm_coverage.weakWindowIndexes:
            processor = IncrementalMeetingProcessor(self.repository, router=self.router)
            recovered = await processor.recover_windows(conversation_id, llm_coverage.weakWindowIndexes)
            recovery_triggered = recovered > 0
            if recovered:
                artifacts = await self.repository.list_meeting_artifacts(conversation_id)
                result, provider, model = await agents.reconcile_meeting(
                    self.router,
                    conversation_id,
                    str(conversation.userId),
                    str(conversation.spaceId),
                    compact_artifact_payload(artifacts),
                    window_summaries,
                    memory.model_dump(),
                    context,
                    conversation.processingVersion,
                    leftover_raw=leftover,
                    checkpoints=checkpoints,
                )
        result = preserve_unrepresented(result, artifacts, conversation_id, str(conversation.spaceId))
        coverage = evaluate_coverage(ordered_windows, artifacts, result)
        evidence_corpus = leftover or "\n".join(window.text for window in ordered_windows)
        result = await self._quality_review_and_repair(
            result,
            evidence_corpus,
            context,
            conversation_id,
            str(conversation.spaceId),
        )
        result.extractionDiagnostics = {
            **(result.extractionDiagnostics or {}),
            "qualityAcceptedTaskCount": len(result.tasks),
            "qualityAcceptedNoteCount": len(result.notes),
        }
        run.checkpoints["incremental_window_finalization"] = {
            "path": "long_checkpoint_synthesis",
            "windowCount": len(ordered_windows),
            "checkpointCount": len(checkpoints),
            "leftoverRawTokens": estimate_tokens(leftover) if leftover else 0,
            "inputTokenEstimate": estimate_tokens(str(artifact_payload) + leftover),
            "durationMs": utc_ms() - started,
            "provisionalArtifactCount": coverage.meaningfulArtifactCount,
            "finalArtifactCount": coverage.finalArtifactCount,
            "llmCompressionRatio": llm_coverage.compressionRatio,
            "compressionRatio": coverage.compressionRatio,
            "coverageScore": coverage.coverageScore,
            "llmCoverageScore": llm_coverage.coverageScore,
            "recoveryTriggered": recovery_triggered,
            "weakWindowIndexes": llm_coverage.weakWindowIndexes or coverage.weakWindowIndexes,
            "taskCount": len(result.tasks),
            "noteCount": len(result.notes),
            "decisionCount": len(result.decisions),
            "issueCount": len(result.issues),
            "provider": provider,
            "model": model,
            "receivedAudioChunkCount": conversation.receivedAudioChunkCount,
            "emptyTranscriptCount": accounting.get("emptyTranscripts"),
            "nonEmptyTranscriptCount": accounting.get("validTranscripts"),
            "failedSTTCount": accounting.get("failedTranscripts"),
            "nonEmptyWindowedCount": accounting.get("validWindowed"),
            "nonEmptyUnwindowedCount": accounting.get("validUnwindowed"),
            "windowCompletedCount": accounting.get("windowsCompleted"),
            "windowFailedCount": accounting.get("windowsFailed"),
            "finalTaskCount": len(result.tasks),
            "finalNoteCount": len(result.notes),
        }
        print("Meeting reconciliation completed:", run.checkpoints["incremental_window_finalization"])
        await self.repository.append_meeting_debug_trace(
            conversation_id,
            conversation.userId,
            conversation.spaceId,
            "reconciliation",
            run.checkpoints["incremental_window_finalization"],
        )
        await self._publish_final_result(
            conversation,
            run,
            result,
            provider,
            model,
            ordered_windows,
            started,
            path="long_checkpoint_synthesis",
            coverage=coverage,
            accounting=accounting,
        )

    async def _publish_final_result(
        self,
        conversation,
        run,
        result,
        provider: str,
        model: str,
        windows,
        started: int,
        path: str,
        coverage=None,
        accounting: dict | None = None,
    ) -> None:
        conversation_id = str(conversation.id)
        accounting = accounting or conversation.lastAccounting or {}
        run.provider = provider
        run.model = model
        run.processedSegmentCount = len(windows)
        run.segmentCount = len(windows)
        run.stagedTasks = result.tasks
        run.stagedNotes = result.notes
        run.stagedDecisions = result.decisions
        run.stagedIssues = result.issues
        run.coverageScore = coverage.coverageScore if coverage else None
        run.validationErrors = [{"code": reason} for reason in coverage.reasons] if coverage and coverage.suspicious else []
        run.warningCount = len(coverage.unrepresentedTitles) if coverage else 0
        run.checkpoints[path] = {
            **(run.checkpoints.get(path) or {}),
            "durationMs": utc_ms() - started,
            "finalTaskCount": len(result.tasks),
            "finalNoteCount": len(result.notes),
            "provider": provider,
            "model": model,
            "evidenceCoverage": bool(result.tasks or result.notes),
        }
        await self.repository.save_extraction_run(run)
        await self.repository.transition(conversation_id, ConversationStatus.VALIDATING)
        summary = ConversationSummaryDocument(
            conversationId=conversation.id,
            userId=conversation.userId,
            spaceId=conversation.spaceId,
            summary=result.summary or result.narrative,
            topics=result.topics,
            importantFacts=result.importantFacts,
            decisions=[decision.title for decision in result.decisions],
            openQuestions=result.openQuestions,
            blockers=[issue.title for issue in result.issues if issue.kind in {"blocker", "risk"}],
            processingVersion=conversation.processingVersion,
            modelProvider=provider,
            modelName=model,
            promptVersion="final-synthesis-v1",
        )
        previous_memory = await self.repository.get_space_memory(conversation.userId, conversation.spaceId)
        memory_update = await agents.update_space_memory(self.router, previous_memory, summary)
        expected_tasks = [task for task in result.tasks if task.operation != "NO_ACTION"]
        expected_notes = list(result.notes)
        persistence = {
            "persistenceAttempted": True,
            "tasksPersistedCount": 0,
            "notesPersistedCount": 0,
            "persistedTaskIds": [],
            "persistedNoteIds": [],
        }
        result.extractionDiagnostics = {**(result.extractionDiagnostics or {}), **persistence}
        try:
            published = await self.repository.publish_outputs(run, summary, memory_update)
        except Exception as error:
            result.extractionDiagnostics["persistenceOutcome"] = "PERSISTENCE_FAILED"
            agents._log_final_synthesis(result.extractionDiagnostics)
            run.validationErrors = [*(run.validationErrors or []), {"code": "PERSISTENCE_FAILED", "message": str(error)[:500]}]
            await self.repository.save_extraction_run(run)
            raise agents.PersistenceFailedError("PERSISTENCE_FAILED") from error
        task_ids = [str(item) for item in (published or {}).get("taskIds") or []]
        note_ids = [str(item) for item in (published or {}).get("noteIds") or []]
        result.extractionDiagnostics.update(
            {
                "persistenceAttempted": True,
                "tasksPersistedCount": len(task_ids),
                "notesPersistedCount": len(note_ids),
                "persistedTaskIds": task_ids,
                "persistedNoteIds": note_ids,
            }
        )
        summary.taskIds = task_ids
        if expected_tasks or expected_notes:
            if (expected_tasks and not task_ids) or (expected_notes and not note_ids):
                result.extractionDiagnostics["persistenceOutcome"] = "PERSISTENCE_FAILED"
                agents._log_final_synthesis(result.extractionDiagnostics)
                run.validationErrors = [*(run.validationErrors or []), {"code": "PERSISTENCE_FAILED"}]
                await self.repository.save_extraction_run(run)
                raise agents.PersistenceFailedError("PERSISTENCE_FAILED")
            result.extractionDiagnostics["persistenceOutcome"] = "PERSISTED"
        else:
            result.extractionDiagnostics["persistenceOutcome"] = "NO_PUBLISHABLE_ARTIFACTS"
        agents._log_final_synthesis(result.extractionDiagnostics)
        run.checkpoints[path] = {
            **(run.checkpoints.get(path) or {}),
            **{
                key: result.extractionDiagnostics.get(key)
                for key in (
                    "validatedSemanticUnitCount",
                    "finalSynthesisInvoked",
                    "finalSynthesisInputUnitCount",
                    "finalSynthesisProvider",
                    "finalSynthesisModel",
                    "finalSynthesisRawTaskCount",
                    "finalSynthesisRawNoteCount",
                    "finalSynthesisParsedTaskCount",
                    "finalSynthesisParsedNoteCount",
                    "finalSynthesisVerdict",
                    "taskCountAfterConfidence",
                    "noteCountAfterConfidence",
                    "qualityAcceptedTaskCount",
                    "qualityAcceptedNoteCount",
                    "qualityRejectedTaskCount",
                    "qualityRejectedNoteCount",
                    "qualityArtifactDiagnostics",
                    "qualityRepairAttempted",
                    "qualityRepairRound",
                    "requiredConfidence",
                    "persistenceAttempted",
                    "persistenceOutcome",
                    "tasksPersistedCount",
                    "notesPersistedCount",
                    "persistedTaskIds",
                    "persistedNoteIds",
                    "persistedTranscriptCount",
                    "persistedNonEmptyTranscriptCount",
                    "persistedSequenceNumbers",
                    "queriedTranscriptCount",
                    "queriedSequenceNumbers",
                    "windowId",
                    "windowIndex",
                    "sequenceStart",
                    "sequenceEnd",
                    "expectedSequenceCount",
                    "windowTranscriptCountBeforeFiltering",
                    "emptyFilteredCount",
                    "unusableFilteredCount",
                    "usefulTranscriptCountAfterFiltering",
                    "usefulSequenceNumbers",
                    "semanticInputTranscriptCount",
                    "semanticInputCharacterCount",
                    "semanticInputEstimatedTokens",
                    "semanticInputAssemblyFailed",
                    "rejectionCounts",
                )
            },
        }
        await self.repository.mark_transcripts_published(conversation_id)
        await self.repository.schedule_transcript_expiry(conversation_id)
        run.status = ExtractionRunStatus.PUBLISHED
        await self.repository.save_extraction_run(run)
        terminal = ConversationStatus.COMPLETED
        if accounting.get("permanentFailures") or (path != "short_raw_transcript" and not result.tasks and not result.notes and coverage and coverage.meaningfulArtifactCount):
            terminal = ConversationStatus.PARTIAL
        await self.repository.transition(conversation_id, terminal)

    async def _quality_review_and_repair(self, result, transcript: str, context: dict, conversation_id: str, space_id: str):
        outputs = {
            "tasks": [item.model_dump() for item in result.tasks],
            "notes": [item.model_dump() for item in result.notes],
        }
        if not outputs["tasks"] and not outputs["notes"]:
            return result
        try:
            review = await agents.review_extraction_quality(self.router, transcript, outputs, context)
        except Exception as error:
            print("Quality review skipped:", {"conversationId": conversation_id, "error": str(error)[:300]})
            return result
        result.tasks, result.notes = _apply_quality_decisions(result.tasks, result.notes, review)
        needs_repair = bool(review.failed or review.missingActionable or review.missingNotes)
        already_repaired = bool((result.extractionDiagnostics or {}).get("qualityRepairAttempted"))
        if needs_repair and settings.MAX_QUALITY_REPAIR_ROUNDS >= 1 and not already_repaired:
            try:
                repair = await agents.repair_missing_items(
                    self.router,
                    transcript,
                    [{"label": "missing", "reason": item} for item in [*review.missingActionable, *review.missingNotes]],
                    {"tasks": [item.model_dump() for item in result.tasks], "notes": [item.model_dump() for item in result.notes]},
                    context,
                    conversation_id,
                    space_id,
                )
                result.tasks = _dedupe_items_by_key([*result.tasks, *repair.tasks])
                result.notes = _dedupe_items_by_key([*result.notes, *repair.notes])
            except Exception as error:
                print("Quality repair skipped:", {"conversationId": conversation_id, "error": str(error)[:300]})
        return result

    async def _ensure_window_artifacts(self, conversation_id: str, windows: list, artifacts: list) -> list:
        from services.conversation.artifact_resolver import reconcile_incoming_artifacts
        from services.conversation.artifacts import artifacts_from_window

        represented = {
            str(artifact.sourceWindowId)
            for artifact in artifacts
            if artifact.sourceWindowId is not None
        }
        represented.update(window_id for artifact in artifacts for window_id in artifact.sourceWindowIds)
        working = list(artifacts)
        repaired = 0
        for window in windows:
            window_id = str(window.id)
            expected = 0
            if window.result is not None:
                expected = (
                    len(window.result.tasks)
                    + len(window.result.notes)
                    + len(window.result.decisions)
                    + len(window.result.issues)
                    + len(window.result.importantFacts)
                    + len(window.result.openQuestions)
                )
            if expected <= 0:
                continue
            if window_id in represented and window.artifactPersistenceOk:
                continue
            incoming = artifacts_from_window(window)
            if not incoming:
                continue
            working = await reconcile_incoming_artifacts(self.router, working, incoming, window.text)
            repaired += 1
        if repaired:
            await self.repository.upsert_meeting_artifacts(working)
            working = await self.repository.list_meeting_artifacts(conversation_id)
            print(
                "Repaired window artifact persistence gaps:",
                {"conversationId": conversation_id, "repairedWindows": repaired, "artifactCount": len(working)},
            )
        return working

    def _merge(self, state: ConversationGraphState) -> None:
        seen_tasks: set[str] = set()
        seen_notes: set[str] = set()
        for result in state.section_results:
            state.warnings.extend(result.warnings)
            for task in result.tasks:
                if task.operation == "NO_ACTION":
                    continue
                task.fingerprint = task.fingerprint or task_fingerprint(state.space_id, task)
                if task.fingerprint not in seen_tasks:
                    seen_tasks.add(task.fingerprint)
                    state.merged_tasks.append(task)
            for note in result.notes:
                note.fingerprint = note.fingerprint or note_fingerprint(state.space_id, note)
                if note.fingerprint not in seen_notes:
                    seen_notes.add(note.fingerprint)
                    state.merged_notes.append(note)
            state.merged_decisions.extend(result.decisions)
            state.merged_blockers.extend(item for item in result.issues if item.kind in {"blocker", "risk"})
            state.merged_questions.extend(item for item in result.issues if item.kind in {"open_question", "missing_information"})

    async def _repair_coverage_gaps(
        self,
        state: ConversationGraphState,
        context: dict,
        outputs: dict,
    ) -> None:
        missing_items = [
            item.model_dump()
            for item in (state.coverage_report.items if state.coverage_report else [])
            if item.label in {"missing", "uncertain"}
        ]
        if not missing_items:
            return
        state.repair_round += 1
        try:
            repair = await agents.repair_missing_items(
                self.router,
                state.normalized_transcript,
                missing_items,
                outputs,
                context,
                state.conversation_id,
                state.space_id,
            )
        except Exception as error:
            state.warnings.append(f"missing-item-repair-v1 failed: {str(error)[:500]}")
            return

        seen_tasks = {task.fingerprint or task_fingerprint(state.space_id, task) for task in state.merged_tasks}
        for task in repair.tasks:
            if task.operation == "NO_ACTION":
                continue
            task.fingerprint = task.fingerprint or task_fingerprint(state.space_id, task)
            if task.fingerprint in seen_tasks:
                continue
            seen_tasks.add(task.fingerprint)
            state.merged_tasks.append(task)

        seen_notes = {note.fingerprint or note_fingerprint(state.space_id, note) for note in state.merged_notes}
        for note in repair.notes:
            note.fingerprint = note.fingerprint or note_fingerprint(state.space_id, note)
            if note.fingerprint in seen_notes:
                continue
            seen_notes.add(note.fingerprint)
            state.merged_notes.append(note)

    def _outputs(self, state: ConversationGraphState) -> dict:
        return {
            "tasks": [item.model_dump() for item in state.merged_tasks],
            "notes": [item.model_dump() for item in state.merged_notes],
            "decisions": [item.model_dump() for item in state.merged_decisions],
            "issues": [item.model_dump() for item in state.merged_questions + state.merged_blockers],
        }

    async def _review_extracted_outputs(self, state: ConversationGraphState, context: dict) -> None:
        outputs = {
            "tasks": [item.model_dump() for item in state.merged_tasks],
            "notes": [item.model_dump() for item in state.merged_notes],
        }
        if not outputs["tasks"] and not outputs["notes"]:
            return
        try:
            review = await agents.review_extraction_quality(
                self.router,
                state.normalized_transcript,
                outputs,
                context,
            )
        except Exception as error:
            state.warnings.append(f"extraction-quality-review-v1 failed: {str(error)[:500]}")
            return

        task_decisions = {item.index: item for item in review.decisions if item.kind == "task"}
        note_decisions = {item.index: item for item in review.decisions if item.kind == "note"}
        kept_tasks = []
        for index, task in enumerate(state.merged_tasks):
            decision = task_decisions.get(index)
            if decision and decision.revisedBody:
                task.body = _clean_text(decision.revisedBody)
                task.fingerprint = task_fingerprint(state.space_id, task)
            if decision and decision.quality:
                task.changes = {**task.changes, "quality": decision.quality, "synthesisSource": "llm"}
            if decision and not decision.keep:
                state.warnings.append(f"REVIEW_REJECTED_TASK: {task.title} ({decision.reason})")
                continue
            kept_tasks.append(task)
        kept_notes = []
        for index, note in enumerate(state.merged_notes):
            decision = note_decisions.get(index)
            if decision and decision.revisedBody:
                note.body = _clean_text(decision.revisedBody)
                note.fingerprint = note_fingerprint(state.space_id, note)
            if decision and decision.quality:
                note.debug = {**note.debug, "quality": decision.quality, "synthesisSource": "llm"}
            if decision and not decision.keep:
                state.warnings.append(f"REVIEW_REJECTED_NOTE: {note.title} ({decision.reason})")
                continue
            kept_notes.append(note)
        state.merged_tasks = kept_tasks
        state.merged_notes = kept_notes

    def _deterministic_validate(self, state: ConversationGraphState) -> None:
        scored = score_and_filter_result(
            type("_Result", (), {
                "tasks": state.merged_tasks,
                "notes": state.merged_notes,
                "decisions": state.merged_decisions,
                "issues": state.merged_questions + state.merged_blockers,
            })(),
            state.normalized_transcript,
        )
        state.merged_tasks = scored.tasks
        state.merged_notes = scored.notes
        state.merged_decisions = scored.decisions
        state.merged_questions = [item for item in scored.issues if item.kind in {"open_question", "missing_information"}]
        state.merged_blockers = [item for item in scored.issues if item.kind in {"blocker", "risk"}]
        for collection_name, items in {
            "tasks": state.merged_tasks,
            "notes": state.merged_notes,
            "decisions": state.merged_decisions,
            "issues": state.merged_questions + state.merged_blockers,
        }.items():
            for item in items:
                if not item.evidence:
                    state.validation_errors.append(
                        {"code": "MISSING_EVIDENCE", "collection": collection_name, "title": getattr(item, "title", "")}
                    )
                for evidence in item.evidence:
                    if not _evidence_matches_transcript(evidence.text, state.raw_transcript, state.normalized_transcript):
                        state.warnings.append(
                            f"EVIDENCE_TEXT_NOT_EXACT_MATCH: {getattr(item, 'title', '')}"
                        )


def _evidence_matches_transcript(evidence_text: str, raw_transcript: str, normalized_transcript: str) -> bool:
    evidence = _normalize_for_evidence_match(evidence_text)
    if not evidence:
        return False

    raw = _normalize_for_evidence_match(raw_transcript)
    normalized = _normalize_for_evidence_match(normalized_transcript)
    if evidence in raw or evidence in normalized:
        return True

    words = evidence.split()
    if len(words) < 4:
        return False

    for size in range(min(10, len(words)), 3, -1):
        for index in range(0, len(words) - size + 1):
            phrase = " ".join(words[index : index + size])
            if phrase in raw or phrase in normalized:
                return True
    return False


def _normalize_for_evidence_match(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _clean_text(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _dedupe_window_payload(payload: dict) -> dict:
    for key in ("tasks", "notes", "decisions", "issues"):
        seen: set[str] = set()
        unique = []
        for item in payload.get(key, []):
            identity = "|".join(str(item.get(field, "")) for field in ("title", "body", "operation", "status", "kind"))
            if identity in seen:
                continue
            seen.add(identity)
            unique.append(item)
        payload[key] = unique
    for key in ("topics", "importantFacts", "openQuestions"):
        seen_values: set[str] = set()
        values = []
        for value in payload.get(key, []):
            normalized = _clean_text(str(value)).casefold()
            if not normalized or normalized in seen_values:
                continue
            seen_values.add(normalized)
            values.append(value)
        payload[key] = values
    return payload


def _fit_payload_to_token_limit(windows: list[dict], token_limit: int) -> list[dict]:
    if estimate_tokens(str(windows)) <= token_limit:
        return windows
    compacted: list[dict] = []
    for window in windows:
        compacted.append(
            {
                "windowIndex": window.get("windowIndex"),
                "sequenceStart": window.get("sequenceStart"),
                "sequenceEnd": window.get("sequenceEnd"),
                "summary": window.get("summary", ""),
                "topics": window.get("topics", []),
                "importantFacts": window.get("importantFacts", []),
                "tasks": _compact_items(window.get("tasks", [])),
                "notes": _compact_items(window.get("notes", [])),
                "decisions": _compact_items(window.get("decisions", [])),
                "issues": _compact_items(window.get("issues", [])),
                "openQuestions": window.get("openQuestions", []),
            }
        )
    return compacted


def _compact_items(items: list[dict]) -> list[dict]:
    return [
        {
            "title": item.get("title"),
            "body": item.get("body"),
            "confidence": item.get("confidence"),
            "sourceConversationId": item.get("sourceConversationId"),
            "fingerprint": item.get("fingerprint"),
            "operation": item.get("operation"),
            "status": item.get("status"),
            "kind": item.get("kind"),
            "existingTaskId": item.get("existingTaskId"),
            "dueDateText": item.get("dueDateText"),
            "dueDateResolved": item.get("dueDateResolved"),
            "dueDateStatus": item.get("dueDateStatus"),
            "ownerText": item.get("ownerText"),
            "ownerUserId": item.get("ownerUserId"),
            "needsConfirmation": item.get("needsConfirmation"),
            "evidence": item.get("evidence", []),
        }
        for item in items
    ]


def utc_ms() -> int:
    import time

    return int(time.time() * 1000)


def _is_completed_checkpoint(window) -> bool:
    if window.isFinalPartial or getattr(window, "extractionSkipped", False):
        return False
    if window.status.value != "COMPLETED":
        return False
    result = window.result
    if result is None:
        return False
    return bool(
        getattr(result, "isCheckpoint", False)
        or getattr(result, "semanticUnits", None)
        or result.tasks
        or result.notes
        or result.decisions
        or result.issues
        or result.importantFacts
        or result.summary
    )


def _apply_quality_decisions(tasks, notes, review):
    task_decisions = {item.index: item for item in review.decisions if item.kind == "task"}
    note_decisions = {item.index: item for item in review.decisions if item.kind == "note"}
    kept_tasks = []
    for index, task in enumerate(tasks):
        decision = task_decisions.get(index)
        if decision and decision.revisedBody:
            task.body = _clean_text(decision.revisedBody)
        if decision and decision.quality:
            task.changes = {**task.changes, "quality": decision.quality, "synthesisSource": "llm"}
        if decision and not decision.keep:
            continue
        kept_tasks.append(task)
    kept_notes = []
    for index, note in enumerate(notes):
        decision = note_decisions.get(index)
        if decision and decision.revisedBody:
            note.body = _clean_text(decision.revisedBody)
        if decision and decision.quality:
            note.debug = {**(note.debug or {}), "quality": decision.quality, "synthesisSource": "llm"}
        if decision and not decision.keep:
            continue
        kept_notes.append(note)
    return kept_tasks, kept_notes


def _dedupe_items_by_key(items: list) -> list:
    seen: set[str] = set()
    unique = []
    for item in items:
        key = str((getattr(item, "changes", {}) or getattr(item, "debug", {}) or {}).get("semanticArtifactKey") or "")
        identity = key or f"{getattr(item, 'title', '')}|{getattr(item, 'body', '')[:120]}"
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(item)
    return unique
