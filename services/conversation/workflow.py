from __future__ import annotations

import asyncio
import re

from apps.api_gateway.config.setting import settings
from services.conversation import agents
from services.conversation.context import load_space_context
from services.conversation.fingerprints import note_fingerprint, task_fingerprint
from services.conversation.incremental import compact_window_payload
from services.conversation.models import ConversationStatus, ConversationSummaryDocument, ExtractionRunStatus, WindowProcessingStatus
from services.conversation.repository import ConversationRepository
from services.conversation.safety import detect_rule_signals
from services.conversation.transcript import assemble_transcript, estimate_tokens, segment_transcript
from services.conversation.workflow_state import ConversationGraphState
from services.llm.router import LLMCapability, LLMRouter, get_llm_router


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
                if windows:
                    if all(window.status == WindowProcessingStatus.COMPLETED for window in windows):
                        await self._run_incremental_finalization(conversation, run, windows, context=None)
                        return
                    await self.repository.transition(conversation_id, ConversationStatus.FINALIZING)
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

    async def _run_incremental_finalization(self, conversation, run, windows, context: dict | None = None) -> None:
        started = utc_ms()
        context = context or await load_space_context(self.repository, conversation.userId, conversation.spaceId)
        ordered_windows = sorted(windows, key=lambda window: (window.sequenceStart, window.windowIndex))
        compacted = [_dedupe_window_payload(compact_window_payload(window)) for window in ordered_windows]
        final_payload = _fit_payload_to_token_limit(compacted, settings.FINAL_MODEL_INPUT_TOKEN_LIMIT)
        result, provider, model = await agents.finalize_from_window_results(
            self.router,
            str(conversation.id),
            str(conversation.userId),
            str(conversation.spaceId),
            final_payload,
            context,
            conversation.processingVersion,
        )
        run.provider = provider
        run.model = model
        run.processedSegmentCount = len(ordered_windows)
        run.segmentCount = len(ordered_windows)
        run.stagedTasks = result.tasks
        run.stagedNotes = result.notes
        run.stagedDecisions = result.decisions
        run.stagedIssues = result.issues
        run.coverageScore = None
        run.validationErrors = []
        run.warningCount = 0
        run.checkpoints["incremental_window_finalization"] = {
            "windowCount": len(ordered_windows),
            "inputTokenEstimate": estimate_tokens(str(final_payload)),
            "durationMs": utc_ms() - started,
        }
        await self.repository.save_extraction_run(run)
        await self.repository.transition(str(conversation.id), ConversationStatus.VALIDATING)
        summary = ConversationSummaryDocument(
            conversationId=conversation.id,
            userId=conversation.userId,
            spaceId=conversation.spaceId,
            summary=result.summary,
            topics=result.topics,
            importantFacts=result.importantFacts,
            decisions=[decision.title for decision in result.decisions],
            openQuestions=result.openQuestions,
            blockers=[issue.title for issue in result.issues if issue.kind in {"blocker", "risk"}],
            processingVersion=conversation.processingVersion,
            modelProvider=provider,
            modelName=model,
            promptVersion="meeting-finalizer-v1",
        )
        previous_memory = await self.repository.get_space_memory(conversation.userId, conversation.spaceId)
        memory = await agents.update_space_memory(self.router, previous_memory, summary)
        await self.repository.publish_outputs(run, summary, memory)
        await self.repository.mark_transcripts_published(str(conversation.id))
        await self.repository.schedule_transcript_expiry(str(conversation.id))
        run.status = ExtractionRunStatus.PUBLISHED
        await self.repository.save_extraction_run(run)
        await self.repository.transition(str(conversation.id), ConversationStatus.COMPLETED)

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
            if decision and not decision.keep:
                state.warnings.append(f"DROPPED_TASK_BY_LLM_REVIEW: {task.title} ({decision.reason})")
                continue
            kept_tasks.append(task)
        kept_notes = []
        for index, note in enumerate(state.merged_notes):
            decision = note_decisions.get(index)
            if decision and decision.revisedBody:
                note.body = _clean_text(decision.revisedBody)
                note.fingerprint = note_fingerprint(state.space_id, note)
            if decision and not decision.keep:
                state.warnings.append(f"DROPPED_NOTE_BY_LLM_REVIEW: {note.title} ({decision.reason})")
                continue
            kept_notes.append(note)
        state.merged_tasks = kept_tasks
        state.merged_notes = kept_notes

    def _deterministic_validate(self, state: ConversationGraphState) -> None:
        transcript_signals = detect_rule_signals(state.normalized_transcript)
        if transcript_signals["actionSignals"] and not state.merged_tasks:
            state.warnings.append("Action-like language detected but no tasks were extracted.")
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
