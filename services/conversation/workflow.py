from __future__ import annotations

import asyncio
import re

from apps.api_gateway.config.setting import settings
from services.conversation import agents
from services.conversation.context import load_space_context
from services.conversation.fingerprints import note_fingerprint, task_fingerprint
from services.conversation.models import ConversationStatus, ExtractionRunStatus
from services.conversation.repository import ConversationRepository
from services.conversation.safety import detect_rule_signals
from services.conversation.transcript import assemble_transcript, segment_transcript
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
        if conversation.status in {
            ConversationStatus.VALIDATING,
            ConversationStatus.COMPLETED,
        }:
            return
        if conversation.status == ConversationStatus.FAILED:
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
        provider, model = self.router.route(LLMCapability.HIGH_ACCURACY_REASONING)
        run = await self.repository.create_extraction_run(conversation, provider.name, model)
        await self.repository.transition(
            conversation_id,
            ConversationStatus.PROCESSING,
            {"activeExtractionRunId": run.id},
        )

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
            user_id=conversation.userId,
            space_id=conversation.spaceId,
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
        outputs = {
            "tasks": [item.model_dump() for item in state.merged_tasks],
            "notes": [item.model_dump() for item in state.merged_notes],
            "decisions": [item.model_dump() for item in state.merged_decisions],
            "issues": [item.model_dump() for item in state.merged_questions + state.merged_blockers],
        }
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
        if state.validation_errors:
            run.status = ExtractionRunStatus.FAILED
            await self.repository.save_extraction_run(run)
            await self.repository.transition(conversation_id, ConversationStatus.FAILED)
            return

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
        await self.repository.transition(conversation_id, ConversationStatus.COMPLETED)

    async def _parallel_section_extraction(self, state: ConversationGraphState, context: dict) -> list:
        semaphore = asyncio.Semaphore(settings.MAX_ACTIVE_LLM_CALLS_PER_CONVERSATION)

        async def run_segment(segment):
            async with semaphore:
                return await agents.extract_segment(self.router, segment, context, state.user_id, state.space_id)

        return await asyncio.gather(*(run_segment(segment) for segment in state.segments))

    def _merge(self, state: ConversationGraphState) -> None:
        seen_tasks: set[str] = set()
        seen_notes: set[str] = set()
        for result in state.section_results:
            for task in result.tasks:
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
