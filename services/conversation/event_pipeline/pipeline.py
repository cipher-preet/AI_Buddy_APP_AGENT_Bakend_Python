"""Hierarchical conversation-intelligence orchestrator."""

from __future__ import annotations

from typing import Any, Iterable

from apps.api_gateway.config.setting import settings
from services.conversation.event_pipeline.channels import split_action_and_memory
from services.conversation.event_pipeline.cleaning import clean_transcripts
from services.conversation.event_pipeline.cost import cost_report
from services.conversation.event_pipeline.budget import PipelineBudget, bind_budget, reset_budget
from services.conversation.event_pipeline.completeness import (
    LLMCompletenessReviewer,
    bootstrap_semantic_records,
    review_and_repair_semantic_completeness,
)
from services.conversation.event_pipeline.coverage import (
    apply_action_coverage,
    apply_memory_coverage,
    apply_semantic_coverage,
    build_coverage_ledger,
    mark_block_no_event,
    trace_missing_memory,
    unpublished_action_events,
    unpublished_memory_events,
)
from services.conversation.event_pipeline.embeddings import CachedEmbedder, default_embedder
from services.conversation.event_pipeline.events import (
    EventExtractor,
    LLMEventExtractor,
    events_to_semantic_units,
    recover_uncovered_content_islands,
)
from services.conversation.event_pipeline.flags import (
    coverage_ledger_enabled,
    semantic_microblocks_enabled,
)
from services.conversation.event_pipeline.microblocks import build_micro_blocks
from services.conversation.event_pipeline.observability import bind_observability, log_pipeline, reset_observability, timed_stage
from services.conversation.event_pipeline.versions import stamp_artifact_provenance, version_metadata
from services.conversation.event_pipeline.schemas import (
    AtomicEvent,
    EventDisposition,
    EventKind,
    EventPipelineResult,
    GlobalThread,
    MicroBlock,
    PipelineObservability,
)
from services.conversation.event_pipeline.store import ConversationEventStore
from services.conversation.event_pipeline.synthesis import (
    DeterministicNoteSynthesizer,
    DeterministicTaskSynthesizer,
    LLMNoteSynthesizer,
    LLMTaskSynthesizer,
    NoteSynthesizer,
    TaskSynthesizer,
)
from services.conversation.event_pipeline.textutil import casefold_text, evidence_sequence_ids, sequence_map_from_records, token_count
from services.conversation.event_pipeline.threads import link_global_threads
from services.conversation.event_pipeline.topics import _is_filler_block, segment_local_topics
from services.conversation.event_pipeline.snapshots import build_snapshots, persist_snapshots
from services.conversation.event_pipeline.validation import LLMArtifactValidator, mixed_thread_rate, validate_artifact
from services.conversation.models import (
    ExtractionOutcome,
    ExtractedNote,
    ExtractedTask,
    TranscriptChunkDocument,
    WindowExtractionResult,
)
from services.llm.router import LLMRouter


async def run_event_pipeline(
    chunks: list[TranscriptChunkDocument],
    conversation_id: str,
    user_id: str,
    space_id: str,
    *,
    checkpoint_events: list[AtomicEvent] | None = None,
    router: LLMRouter | None = None,
    embedder: CachedEmbedder | None = None,
    event_extractor: EventExtractor | None = None,
    task_synthesizer: TaskSynthesizer | None = None,
    note_synthesizer: NoteSynthesizer | None = None,
    event_store: ConversationEventStore | None = None,
    repository: Any | None = None,
    polish_with_llm: bool = False,
    completeness_reviewer: Any | None = None,
) -> EventPipelineResult:
    obs = PipelineObservability()
    obs.sessionId = conversation_id
    obs_token = bind_observability(obs)
    budget = PipelineBudget()
    budget_token = bind_budget(budget)
    embedder = embedder or default_embedder()
    store = event_store or ConversationEventStore(repository)
    extractor = event_extractor or (LLMEventExtractor(router) if router is not None else None)
    task_synth = task_synthesizer or DeterministicTaskSynthesizer()
    note_synth = note_synthesizer or DeterministicNoteSynthesizer()
    if polish_with_llm and router is not None:
        if task_synthesizer is None:
            task_synth = LLMTaskSynthesizer(router, fallback=task_synth)
        if note_synthesizer is None:
            note_synth = LLMNoteSynthesizer(router, fallback=note_synth)
    llm_validator = LLMArtifactValidator(router) if polish_with_llm and router is not None else None

    try:
        return await _run_event_pipeline_body(
            chunks,
            conversation_id,
            user_id,
            space_id,
            checkpoint_events=checkpoint_events,
            router=router,
            embedder=embedder,
            extractor=extractor,
            task_synth=task_synth,
            note_synth=note_synth,
            store=store,
            polish_with_llm=polish_with_llm,
            llm_validator=llm_validator,
            completeness_reviewer=completeness_reviewer,
            obs=obs,
        )
    finally:
        reset_budget(budget_token)
        reset_observability(obs_token)


async def _run_event_pipeline_body(
    chunks,
    conversation_id: str,
    user_id: str,
    space_id: str,
    *,
    checkpoint_events,
    router,
    embedder,
    extractor,
    task_synth,
    note_synth,
    store,
    polish_with_llm: bool,
    llm_validator,
    completeness_reviewer,
    obs,
) -> EventPipelineResult:
    with timed_stage(obs, "cleaning") as stage:
        cleaning = clean_transcripts(chunks, conversation_id=conversation_id, user_id=user_id, space_id=space_id)
        stage.extra = {"useful": cleaning.usefulSequences, "excluded": cleaning.excludedStructuralSequences}
    if not cleaning.complete:
        raise RuntimeError("sequence accounting incomplete: a transcript disappeared during cleaning")

    sequence_text = sequence_map_from_records(cleaning.useful)
    checkpoint_events = [
        _with_conversation(event, conversation_id, user_id, space_id) for event in (checkpoint_events or [])
    ]
    covered = {sequence for event in checkpoint_events for sequence in event.sequenceIds}
    remaining = [record for record in cleaning.useful if record.sequenceId not in covered]

    with timed_stage(obs, "micro_blocks") as stage:
        if semantic_microblocks_enabled():
            blocks = await build_micro_blocks(remaining, embedder)
        else:
            blocks = await build_micro_blocks(remaining, embedder)
        stage.embeddingCalls = getattr(embedder, "calls", 0)
        stage.extra = {"count": len(blocks)}

    with timed_stage(obs, "topics") as stage:
        topics = await segment_local_topics(
            blocks,
            embedder,
            router=router if isinstance(extractor, LLMEventExtractor) else None,
        )
        stage.llmCalls = 0
        stage.extra = {"count": len(topics)}

    new_events: list[AtomicEvent] = []
    semantic_records: list = []
    semantic_review_ran = False
    reviewer = completeness_reviewer
    if reviewer is None and isinstance(extractor, LLMEventExtractor) and router is not None:
        reviewer = LLMCompletenessReviewer(router)
    with timed_stage(obs, "event_extraction") as stage:
        if extractor is None:
            new_events = _abstain_events(topics, blocks, conversation_id, user_id, space_id)
        else:
            for topic in topics:
                topic_blocks = [block for block in blocks if block.microBlockId in set(topic.microBlockIds)]
                try:
                    if topic_blocks and all(_is_filler_block(block) for block in topic_blocks):
                        extracted = []
                    else:
                        extracted = await extractor.extract(topic, topic_blocks, sequence_text)
                except Exception as error:
                    from services.llm.async_runtime import reraise_if_hard_runtime

                    reraise_if_hard_runtime(error)
                    extracted = []
                extracted = [
                    _with_conversation(event, conversation_id, user_id, space_id) for event in extracted
                ]
                extracted.extend(
                    recover_uncovered_content_islands(
                        topic,
                        topic_blocks,
                        sequence_text,
                        extracted,
                        conversation_id=conversation_id,
                        user_id=user_id,
                        space_id=space_id,
                    )
                )
                for event in extracted:
                    _attach_block_ids(event, topic_blocks)
                if reviewer is not None:
                    repaired, records, ran = await review_and_repair_semantic_completeness(
                        topic,
                        topic_blocks,
                        sequence_text,
                        extracted,
                        reviewer,
                        extractor,
                        conversation_id=conversation_id,
                        user_id=user_id,
                        space_id=space_id,
                    )
                    semantic_review_ran = semantic_review_ran or ran
                    semantic_records.extend(records)
                    for event in repaired:
                        _attach_block_ids(event, topic_blocks)
                    extracted.extend(repaired)
                if not extracted:
                    extracted = _no_event_placeholder(topic, topic_blocks, conversation_id, user_id, space_id)
                for event in extracted:
                    _attach_block_ids(event, topic_blocks)
                new_events.extend(extracted)
        stage.llmCalls = getattr(extractor, "calls", 0) if extractor is not None else 0
        stage.extra = {"events": len(new_events), "semanticReviewRan": semantic_review_ran}

    incoming = [*checkpoint_events, *new_events]
    merged = await store.upsert(conversation_id, incoming)
    incoming_ids = {event.eventId for event in incoming}
    incoming_seqs = {sequence for event in incoming for sequence in (event.sequenceIds or [])}
    merged = [
        event
        for event in merged
        if event.eventId in incoming_ids or set(event.sequenceIds or []) & incoming_seqs
    ] or incoming
    for event in merged:
        _attach_block_ids(event, blocks)

    with timed_stage(obs, "thread_linking") as stage:
        threads, links, comparisons = await link_global_threads(
            merged,
            embedder,
            router=router if polish_with_llm else None,
        )
        obs.comparisonCount = comparisons
        stage.extra = {"threads": len(threads), "comparisons": comparisons}

    actions, memory, other = split_action_and_memory(merged)
    thread_by_id = {thread.threadId: thread for thread in threads}
    synthesis_input = list(actions)

    with timed_stage(obs, "task_synthesis") as stage:
        tasks = await _synthesize_tasks(actions, thread_by_id, task_synth)
        stage.extra = {"accepted": len(tasks), "input": len(synthesis_input)}
    with timed_stage(obs, "note_synthesis") as stage:
        notes = await _synthesize_notes(memory, thread_by_id, note_synth, existing_tasks=tasks)
        stage.extra = {"accepted": len(notes)}

    with timed_stage(obs, "evidence_validation"):
        pre_validation_tasks = len(tasks)
        tasks, notes = await _validate_artifacts(tasks, notes, sequence_text, merged, llm_validator)
        obs.taskCandidates = pre_validation_tasks
        obs.taskValidationAccepted = len(tasks)
        obs.taskValidationRejected = max(0, pre_validation_tasks - len(tasks))

    _account_blocks_from_events(blocks, merged)
    ledger = None
    memory_traces: list = []
    if coverage_ledger_enabled():
        with timed_stage(obs, "coverage"):
            ledger = build_coverage_ledger(
                total_raw_sequences=cleaning.totalSequences,
                useful_sequences=cleaning.usefulSequences,
                excluded_structural_sequences=cleaning.excludedStructuralSequences,
                micro_blocks=_all_blocks(blocks, checkpoint_events),
                topics_count=len(topics),
                events=merged,
                tasks=tasks,
                notes=notes,
            )
            _account_explicit_no_events(ledger, blocks, merged)
            ledger = await _repair(
                ledger,
                merged,
                thread_by_id,
                task_synth,
                note_synth,
                sequence_text,
                tasks,
                notes,
                llm_validator=llm_validator,
            )
            tasks, notes = ledger._repaired_tasks, ledger._repaired_notes  # type: ignore[attr-defined]
            ledger.tasks_generated = len(tasks)
            ledger.notes_generated = len(notes)
            from services.conversation.event_pipeline.memory_identity import set_memory_disposition
            from services.conversation.event_pipeline.schemas import MemoryDisposition

            for event in unpublished_memory_events(merged):
                set_memory_disposition(event, MemoryDisposition.UNSUPPORTED, "no_publishable_note")
            apply_memory_coverage(ledger, merged)
            from services.conversation.event_pipeline.channels import set_action_disposition
            from services.conversation.event_pipeline.schemas import ActionDisposition

            for event in unpublished_action_events(merged):
                set_action_disposition(event, ActionDisposition.UNSUPPORTED, "no_publishable_task")
            apply_action_coverage(ledger, merged)
            if not semantic_records:
                semantic_records = bootstrap_semantic_records(blocks, merged)
            else:
                known_ids = {record.eventId for record in semantic_records if record.eventId}
                semantic_records.extend(
                    bootstrap_semantic_records(
                        blocks,
                        [event for event in merged if event.eventId not in known_ids],
                    )
                )
            apply_semantic_coverage(ledger, semantic_records, merged)
            ledger.semanticReviewRan = semantic_review_ran
            leftover = unpublished_memory_events(merged)
            leftover_actions = unpublished_action_events(merged)
            ledger.hardFailure = (
                ledger.unaccounted_blocks > 0
                or bool(leftover)
                or bool(leftover_actions)
                or bool(ledger.memoryCoverageFailure)
                or bool(ledger.actionCoverageFailure)
                or bool(ledger.semanticCoverageFailure)
            )
            if leftover and "unpublished_memory_events" not in ledger.suspicious:
                ledger.suspicious.append("unpublished_memory_events")
            if not leftover and "unpublished_memory_events" in ledger.suspicious:
                ledger.suspicious.remove("unpublished_memory_events")
            if leftover_actions and "unpublished_action_events" not in ledger.suspicious:
                ledger.suspicious.append("unpublished_action_events")
            if not leftover_actions and "unpublished_action_events" in ledger.suspicious:
                ledger.suspicious.remove("unpublished_action_events")
            memory_traces = trace_missing_memory(merged, blocks, topics, notes)

    stamp_artifact_provenance(tasks, notes, pipeline_mode=obs.mode or "event_pipeline")

    snapshots = build_snapshots(
        cleaning=cleaning,
        blocks=blocks,
        topics=topics,
        events=merged,
        threads=threads,
        actions=actions,
        memory=memory,
        tasks=tasks,
        notes=notes,
        coverage=ledger,
    )
    snapshot_path = persist_snapshots(conversation_id, snapshots)
    log_pipeline(obs, cleaning, blocks, topics, merged, actions, memory, other, threads, links, tasks, notes, ledger)
    cost = cost_report(obs)
    versions = version_metadata()
    from services.conversation.event_pipeline.budget import current_budget
    from services.conversation.event_pipeline.publish_gate import publication_ready

    budget = current_budget()
    ready, blocked = publication_ready(
        EventPipelineResult(
            tasks=tasks,
            notes=notes,
            events=merged,
            threads=threads,
            topics=topics,
            microBlocks=blocks,
            cleaning=cleaning,
            coverage=ledger,
            observability=obs,
        )
    )
    diagnostics = {
        "pipeline": "event-hierarchical-v1",
        "pipelineVersion": versions["pipelineVersion"],
        "eventSchemaVersion": versions["eventSchemaVersion"],
        "promptVersion": versions["promptVersion"],
        "artifactPipelineVersion": versions["artifactPipelineVersion"],
        "pipelineMode": obs.mode,
        "publicationReady": ready,
        "publicationBlockedReason": blocked,
        "budget": budget.snapshot() if budget else {},
        "finalSynthesisInvoked": True,
        "finalSynthesisVerdict": "PUBLISH" if tasks or notes else "NO_PUBLISHABLE_ARTIFACTS",
        "validatedSemanticUnitCount": len(merged),
        "qualityAcceptedTaskCount": len(tasks),
        "qualityAcceptedNoteCount": len(notes),
        "mixedThreadArtifactRate": mixed_thread_rate([*tasks, *notes], merged),
        "coverage": ledger.as_metrics() if ledger else {},
        "taskPipelineTrace": {
            "atomicEvents": obs.atomicEvents,
            "actionableEvents": obs.actionableEvents,
            "explicitActionEvents": obs.explicitActionEvents,
            "groundedActionObjects": obs.groundedActionObjects,
            "actionChannelEvents": obs.actionChannelEvents,
            "taskSynthesisInputEvents": obs.taskSynthesisInputEvents,
            "taskCandidates": obs.taskCandidates,
            "taskValidationAccepted": obs.taskValidationAccepted,
            "taskValidationRejected": obs.taskValidationRejected,
            "tasksPersisted": obs.tasksPersisted,
            "tasksReturnedByApi": obs.tasksReturnedByApi,
        },
        "semanticCoverageTrace": {
            "semanticUnitsDetected": getattr(ledger, "semanticUnitsDetected", 0) if ledger else 0,
            "semanticUnitsCreated": getattr(ledger, "semanticUnitsCreated", 0) if ledger else 0,
            "unaccountedSemanticUnits": getattr(ledger, "unaccountedSemanticUnits", 0) if ledger else 0,
            "semanticCoverage": getattr(ledger, "semanticCoverage", 1.0) if ledger else 1.0,
            "semanticReviewRan": getattr(ledger, "semanticReviewRan", False) if ledger else False,
        },
        "stageSnapshots": {key: snapshots[key] for key in ("MICRO_BLOCKS", "TOPICS", "ATOMIC_EVENTS", "GLOBAL_THREADS", "COVERAGE_LEDGER") if key in snapshots},
        "traces": snapshots.get("TRACES") or [],
        "memoryTraces": memory_traces,
        "snapshotPath": snapshot_path,
        "cost": cost,
        "modelRoutes": [item.model_dump() for item in obs.modelRoutes],
        **{log.split("]")[0].strip("[]").lower(): log for log in obs.logs},
    }
    if ledger:
        diagnostics.update(ledger.as_metrics())
    provider = getattr(extractor, "last_provider", None) or "event-pipeline"
    model = getattr(extractor, "last_model", None) or "hierarchical-v1"
    return EventPipelineResult(
        tasks=tasks,
        notes=notes,
        events=merged,
        threads=threads,
        topics=topics,
        microBlocks=blocks,
        cleaning=cleaning,
        coverage=ledger,
        observability=obs,
        snapshots=snapshots,
        cost=cost,
        provider=str(provider),
        model=str(model),
        diagnostics=diagnostics,
    )


async def extract_window_events(
    chunks: list[TranscriptChunkDocument],
    conversation_id: str,
    user_id: str,
    space_id: str,
    *,
    router: LLMRouter | None = None,
    embedder: CachedEmbedder | None = None,
    event_extractor: EventExtractor | None = None,
    event_store: ConversationEventStore | None = None,
    repository: Any | None = None,
) -> tuple[WindowExtractionResult, str, str]:
    """Checkpoint path: persist atomic events, not lossy prose summaries."""
    result = await run_event_pipeline(
        chunks,
        conversation_id,
        user_id,
        space_id,
        router=router,
        embedder=embedder,
        event_extractor=event_extractor,
        event_store=event_store,
        repository=repository,
        polish_with_llm=False,
    )
    window_result = to_window_result(result, checkpoint=True)
    return window_result, result.provider, result.model


def to_window_result(result: EventPipelineResult, checkpoint: bool = False) -> WindowExtractionResult:
    summary = result.threads[0].latestState if result.threads else ""
    return WindowExtractionResult(
        summary=summary,
        topics=[topic.label for topic in result.topics],
        importantFacts=[event.meaning for event in result.events if event.kind != EventKind.NOISE][:20],
        semanticUnits=events_to_semantic_units(result.events),
        atomicEvents=[event.model_dump() for event in result.events],
        tasks=[] if checkpoint else result.tasks,
        notes=[] if checkpoint else result.notes,
        isCheckpoint=checkpoint,
        extractionOutcome=ExtractionOutcome.SUCCESS,
        extractionDiagnostics=result.diagnostics,
    )


def events_from_windows(windows: Iterable[Any]) -> list[AtomicEvent]:
    events: list[AtomicEvent] = []
    seen: set[str] = set()
    for window in windows:
        result = getattr(window, "result", None)
        payloads = []
        if result is not None:
            payloads.extend(getattr(result, "atomicEvents", None) or [])
            diagnostics = getattr(result, "extractionDiagnostics", None) or {}
            payloads.extend(diagnostics.get("atomicEvents") or [])
        for payload in payloads:
            event = payload if isinstance(payload, AtomicEvent) else AtomicEvent.model_validate(payload)
            if event.eventId in seen:
                continue
            seen.add(event.eventId)
            events.append(event)
    return events


async def _synthesize_tasks(actions: list[AtomicEvent], threads: dict[str, GlobalThread], synthesizer: TaskSynthesizer) -> list[ExtractedTask]:
    from services.conversation.event_pipeline.channels import (
        action_synthesis_abstain_disposition,
        set_action_disposition,
    )
    from services.conversation.event_pipeline.schemas import ActionDisposition

    tasks: list[ExtractedTask] = []
    seen: set[str] = set()
    seen_objects: set[str] = set()
    for event in actions:
        if event.kind == EventKind.NOISE:
            set_action_disposition(event, ActionDisposition.INTENTIONALLY_NONPUBLISHABLE, "noise")
            continue
        artifact = await synthesizer.synthesize(event, threads.get(event.threadId or ""))
        if artifact is None:
            disposition, reason = action_synthesis_abstain_disposition(event)
            set_action_disposition(event, disposition, reason)
            continue
        object_key = _task_identity_key(event, artifact)
        if object_key and object_key in seen_objects:
            set_action_disposition(event, ActionDisposition.DUPLICATE, "duplicate_action_identity")
            continue
        key = artifact.fingerprint or artifact.title
        if key in seen:
            set_action_disposition(event, ActionDisposition.DUPLICATE, "duplicate_task_fingerprint")
            continue
        seen.add(key)
        if object_key:
            seen_objects.add(object_key)
        set_action_disposition(event, ActionDisposition.PUBLISHED_TASK, "published")
        tasks.append(artifact)
    return tasks


async def _synthesize_notes(
    memory: list[AtomicEvent],
    threads: dict[str, GlobalThread],
    synthesizer: NoteSynthesizer,
    existing_tasks: list[ExtractedTask] | None = None,
) -> list[ExtractedNote]:
    from services.conversation.event_pipeline.memory_identity import (
        event_is_publishable_memory,
        find_memory_relation,
        memory_importance,
        set_memory_disposition,
    )
    from services.conversation.event_pipeline.schemas import MemoryDisposition

    notes: list[ExtractedNote] = []
    seen: set[str] = set()
    published_events: list[AtomicEvent] = []
    existing_tasks = existing_tasks or []
    for event in memory:
        prior = event.disposition
        if not event_is_publishable_memory(event):
            if memory_importance(event) == "LOW" or (event.memorySignal and not event.memorySignal.isMemoryWorthy):
                set_memory_disposition(event, MemoryDisposition.LOW_VALUE, "low_value_memory")
                if prior not in {
                    EventDisposition.TASK,
                    EventDisposition.NOTE,
                    EventDisposition.DUPLICATE,
                    EventDisposition.SUPERSEDED,
                    EventDisposition.REJECTED,
                }:
                    event.disposition = EventDisposition.INTENTIONALLY_NON_PUBLISHABLE
                    event.dispositionReason = event.dispositionReason or "low_value_memory"
                continue
            if prior not in {
                EventDisposition.TASK,
                EventDisposition.INTENTIONALLY_NON_PUBLISHABLE,
                EventDisposition.DUPLICATE,
                EventDisposition.SUPERSEDED,
                EventDisposition.REJECTED,
                EventDisposition.NOTE,
            }:
                event.disposition = EventDisposition.INTENTIONALLY_NON_PUBLISHABLE
                event.dispositionReason = event.dispositionReason or "low_value_memory"
                set_memory_disposition(event, MemoryDisposition.LOW_VALUE, "low_value_memory")
            continue
        relation, prior_event = find_memory_relation(event, published_events)
        if relation == "DUPLICATE":
            event.disposition = EventDisposition.DUPLICATE if prior != EventDisposition.TASK else prior
            event.dispositionReason = "duplicate_memory_identity"
            set_memory_disposition(event, MemoryDisposition.DUPLICATE, "duplicate_memory_identity")
            continue
        artifact = await synthesizer.synthesize(event, threads.get(event.threadId or ""))
        if artifact is None:
            set_memory_disposition(event, MemoryDisposition.UNSUPPORTED, "note_synthesis_abstained")
            if prior in {
                EventDisposition.TASK,
                EventDisposition.INTENTIONALLY_NON_PUBLISHABLE,
                EventDisposition.DUPLICATE,
                EventDisposition.SUPERSEDED,
                EventDisposition.REJECTED,
            }:
                continue
            event.disposition = EventDisposition.INTENTIONALLY_NON_PUBLISHABLE
            event.dispositionReason = event.dispositionReason or "note_synthesis_abstained"
            continue
        key = artifact.fingerprint or artifact.title
        if key in seen and relation in {None, "DUPLICATE"}:
            set_memory_disposition(event, MemoryDisposition.DUPLICATE, "duplicate_note_fingerprint")
            if prior not in {EventDisposition.TASK, EventDisposition.INTENTIONALLY_NON_PUBLISHABLE}:
                event.disposition = EventDisposition.DUPLICATE
            continue
        seen.add(key if relation not in {"UPDATE", "SUPERSEDE", "DISTINCT"} else event.eventId)
        metadata = artifact.debug or {}
        if relation in {"UPDATE", "SUPERSEDE", "RELATED", "DISTINCT"}:
            metadata["memoryRelation"] = relation
            if prior_event is not None:
                metadata["relatedEventId"] = prior_event.eventId
            artifact.debug = metadata
            if relation == "SUPERSEDE" and prior_event is not None:
                set_memory_disposition(prior_event, MemoryDisposition.SUPERSEDED, "superseded_by_later_state")
                if prior_event.disposition != EventDisposition.TASK:
                    prior_event.disposition = EventDisposition.SUPERSEDED
        if prior == EventDisposition.TASK:
            event.dispositionReason = event.dispositionReason or "task_and_note"
        else:
            event.disposition = EventDisposition.NOTE
            if relation == "SUPERSEDE":
                event.dispositionReason = "supersede_prior_memory"
            elif relation == "UPDATE":
                event.dispositionReason = "status_update"
        set_memory_disposition(event, MemoryDisposition.PUBLISHED_NOTE, relation or "published")
        notes.append(artifact)
        published_events.append(event)
    return notes


async def _validate_artifacts(
    tasks: list[ExtractedTask],
    notes: list[ExtractedNote],
    sequence_text: dict[int, str],
    events: list[AtomicEvent],
    llm_validator: LLMArtifactValidator | None = None,
) -> tuple[list[ExtractedTask], list[ExtractedNote]]:
    kept_tasks: list[ExtractedTask] = []
    for task in tasks:
        result = validate_artifact(task, sequence_text, events, artifact_kind="task")
        if result.action.value == "REJECT":
            _mark_events(events, task, EventDisposition.REJECTED, result.reasons[0] if result.reasons else "validation_rejected")
            continue
        if llm_validator is not None:
            llm_result = await llm_validator.review(result.item, events, "task")
            if llm_result is not None and llm_result.action.value == "REJECT":
                _mark_events(events, task, EventDisposition.REJECTED, llm_result.reasons[0] if llm_result.reasons else "llm_validation_rejected")
                continue
            if llm_result is not None:
                result = llm_result
        kept_tasks.append(result.item)
    kept_notes: list[ExtractedNote] = []
    for note in notes:
        result = validate_artifact(note, sequence_text, events, artifact_kind="note")
        if result.action.value == "REJECT":
            _mark_events(events, note, EventDisposition.REJECTED, result.reasons[0] if result.reasons else "validation_rejected")
            continue
        if llm_validator is not None:
            llm_result = await llm_validator.review(result.item, events, "note")
            if llm_result is not None and llm_result.action.value == "REJECT":
                _mark_events(events, note, EventDisposition.REJECTED, llm_result.reasons[0] if llm_result.reasons else "llm_validation_rejected")
                continue
            if llm_result is not None:
                result = llm_result
        kept_notes.append(result.item)
    return kept_tasks, kept_notes


async def _repair(
    ledger,
    events: list[AtomicEvent],
    threads: dict[str, GlobalThread],
    task_synth: TaskSynthesizer,
    note_synth: NoteSynthesizer,
    sequence_text: dict[int, str],
    tasks: list[ExtractedTask],
    notes: list[ExtractedNote],
    llm_validator: LLMArtifactValidator | None = None,
):
    rounds = int(getattr(settings, "EVENT_PIPELINE_MAX_REPAIR_ROUNDS", 1))
    repaired_tasks = list(tasks)
    repaired_notes = list(notes)
    for _ in range(max(0, rounds)):
        missing_memory = unpublished_memory_events(events)
        if missing_memory:
            extra = await _synthesize_notes(missing_memory, threads, note_synth, existing_tasks=repaired_tasks)
            _, extra = await _validate_artifacts([], extra, sequence_text, events, llm_validator)
            repaired_notes = _dedupe_notes([*repaired_notes, *extra])
        missing_actions = unpublished_action_events(events)
        if missing_actions:
            extra_tasks = await _synthesize_tasks(missing_actions, threads, task_synth)
            extra_tasks, _ = await _validate_artifacts(extra_tasks, [], sequence_text, events, llm_validator)
            repaired_tasks = _dedupe_tasks([*repaired_tasks, *extra_tasks])
        if not unpublished_memory_events(events):
            break
    ledger._repaired_tasks = repaired_tasks
    ledger._repaired_notes = repaired_notes
    return ledger


def _account_blocks_from_events(blocks: list[MicroBlock], events: list[AtomicEvent]) -> None:
    covered = {block_id for event in events for block_id in event.microBlockIds}
    for block in blocks:
        if block.microBlockId in covered:
            continue
        # Placeholder events already cover blocks; leftover blocks stay for ledger.


def _account_explicit_no_events(ledger, blocks: list[MicroBlock], events: list[AtomicEvent]) -> None:
    covered = {block_id for event in events for block_id in event.microBlockIds}
    for record in ledger.blocks:
        if record.disposition.value == "NO_EVENT" and record.reason == "unaccounted" and record.microBlockId in covered:
            record.reason = "produced_via_event"
        if record.disposition.value == "NO_EVENT" and record.reason == "unaccounted":
            related = [event for event in events if record.microBlockId in event.microBlockIds]
            if related and all(event.kind == EventKind.NOISE for event in related):
                mark_block_no_event(ledger, record.microBlockId, "no_supported_event")
    # Recalculate unaccounted after explicit reasons.
    still = 0
    for record in ledger.blocks:
        if record.disposition.value == "NO_EVENT" and record.reason in {"", "unaccounted"}:
            if record.microBlockId not in covered:
                record.reason = "no_supported_event"
            else:
                record.reason = "produced_via_event"
        if record.reason == "unaccounted":
            still += 1
    ledger.unaccounted_blocks = still
    ledger.hardFailure = still > 0 or "memory_events_without_notes" in ledger.suspicious


def _all_blocks(blocks: list[MicroBlock], checkpoint_events: list[AtomicEvent]) -> list[MicroBlock]:
    extra: list[MicroBlock] = []
    seen = {block.microBlockId for block in blocks}
    for event in checkpoint_events:
        for block_id in event.microBlockIds:
            if block_id in seen:
                continue
            seen.add(block_id)
            extra.append(
                MicroBlock(
                    microBlockId=block_id,
                    sequenceStart=min(event.sequenceIds or [0]),
                    sequenceEnd=max(event.sequenceIds or [0]),
                    sequenceIds=list(event.sequenceIds),
                    sourceIds=list(event.sourceIds),
                    text=event.meaning,
                    tokenCount=token_count(event.meaning),
                )
            )
    return [*extra, *blocks]


def _abstain_events(topics, blocks, conversation_id, user_id, space_id) -> list[AtomicEvent]:
    events: list[AtomicEvent] = []
    for topic in topics:
        topic_blocks = [block for block in blocks if block.microBlockId in set(topic.microBlockIds)]
        events.extend(_no_event_placeholder(topic, topic_blocks, conversation_id, user_id, space_id))
    return events


def _no_event_placeholder(topic, blocks: list[MicroBlock], conversation_id: str, user_id: str, space_id: str) -> list[AtomicEvent]:
    from services.conversation.event_pipeline.textutil import stable_id
    from services.conversation.models import EvidenceSpan

    events: list[AtomicEvent] = []
    for block in blocks or [
        MicroBlock(
            microBlockId=f"{topic.topicId}-empty",
            sequenceStart=topic.sequenceStart,
            sequenceEnd=topic.sequenceEnd,
            sequenceIds=list(topic.sequenceIds),
            text=topic.text,
            tokenCount=token_count(topic.text),
        )
    ]:
        span_text = block.text.split("]", 1)[-1].strip() or topic.label or "no supported event"
        events.append(
            AtomicEvent(
                eventId=stable_id("E", conversation_id, "NOISE", block.microBlockId),
                topicId=topic.topicId,
                kind=EventKind.NOISE,
                meaning="No supported atomic event in this micro-block.",
                uncertainty=["extractor_abstained"],
                evidence=[
                    EvidenceSpan(
                        sequenceStart=block.sequenceStart,
                        sequenceEnd=block.sequenceEnd,
                        text=span_text[:500] or "no supported event",
                    )
                ],
                microBlockIds=[block.microBlockId],
                sourceIds=list(block.sourceIds),
                sequenceIds=list(block.sequenceIds),
                channel="other",
                disposition=EventDisposition.INTENTIONALLY_NON_PUBLISHABLE,
                dispositionReason="no_supported_event",
                conversationId=conversation_id,
                userId=user_id,
                spaceId=space_id,
            )
        )
    return events


def _attach_block_ids(event: AtomicEvent, blocks: list[MicroBlock]) -> None:
    sequences = set(event.sequenceIds or evidence_sequence_ids(event.evidence))
    if not sequences:
        return
    for block in blocks:
        if not sequences & set(block.sequenceIds):
            continue
        if block.microBlockId not in event.microBlockIds:
            event.microBlockIds.append(block.microBlockId)
        for source, sequence in zip(block.sourceIds, block.sequenceIds):
            if sequence in sequences and source not in event.sourceIds:
                event.sourceIds.append(source)


def _with_conversation(event: AtomicEvent, conversation_id: str, user_id: str, space_id: str) -> AtomicEvent:
    copy = event.model_copy(deep=True)
    copy.conversationId = copy.conversationId or conversation_id
    copy.userId = copy.userId or user_id
    copy.spaceId = copy.spaceId or space_id
    if not copy.sequenceIds:
        copy.sequenceIds = evidence_sequence_ids(copy.evidence)
    return copy


def _mark_events(events: list[AtomicEvent], item, disposition: EventDisposition, reason: str) -> None:
    from services.conversation.event_pipeline.channels import set_action_disposition
    from services.conversation.event_pipeline.schemas import ActionDisposition, MemoryDisposition

    metadata = getattr(item, "changes", None) or getattr(item, "debug", None) or {}
    ids = set(metadata.get("sourceSemanticUnitIds") or [])
    for event in events:
        if event.eventId not in ids:
            continue
        if metadata.get("coverageStatus") == "note" and event.disposition == EventDisposition.TASK:
            event.memoryDisposition = MemoryDisposition.REJECTED_WITH_REASON
            event.memoryDispositionReason = reason
            continue
        event.disposition = disposition
        event.dispositionReason = reason
        if disposition == EventDisposition.REJECTED:
            if event.memoryDisposition is None:
                event.memoryDisposition = MemoryDisposition.REJECTED_WITH_REASON
                event.memoryDispositionReason = reason
            if event_is_actionable_safe(event):
                set_action_disposition(event, ActionDisposition.VALIDATION_REJECTED, reason)


def event_is_actionable_safe(event: AtomicEvent) -> bool:
    from services.conversation.event_pipeline.channels import event_is_actionable

    return event_is_actionable(event)


def _task_identity_key(event: AtomicEvent, artifact: ExtractedTask | None = None) -> str:
    """Canonical Task identity: verb + object + thread. Title is not identity."""
    thread = casefold_text(event.threadId or "")
    obj = ""
    verb = ""
    if event.actionSignal:
        obj = casefold_text(event.actionSignal.canonicalActionObject or event.actionSignal.object or event.object or "")
        verb = casefold_text(event.actionSignal.verb or "")
    else:
        obj = casefold_text(event.object or (artifact.title if artifact else ""))
    obj = obj.removeprefix("the ").strip()
    if not obj:
        return ""
    return f"{thread}|{verb}|{obj}"


def _task_object_key(event: AtomicEvent, artifact: ExtractedTask) -> str:
    return _task_identity_key(event, artifact)


def _dedupe_tasks(items: list[ExtractedTask]) -> list[ExtractedTask]:
    seen: set[str] = set()
    unique: list[ExtractedTask] = []
    for item in items:
        metadata = item.changes or {}
        verb = casefold_text(str(metadata.get("actionVerb") or ""))
        obj = casefold_text(str(metadata.get("canonicalActionObject") or metadata.get("actionObject") or "")).removeprefix("the ").strip()
        thread = casefold_text(str(metadata.get("threadId") or ""))
        key = f"{thread}|{verb}|{obj}" if verb and obj else (item.fingerprint or item.title)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _dedupe_notes(items: list[ExtractedNote]) -> list[ExtractedNote]:
    seen: set[str] = set()
    unique: list[ExtractedNote] = []
    for item in items:
        metadata = item.debug or {}
        relation = str(metadata.get("memoryRelation") or "")
        if relation in {"UPDATE", "SUPERSEDE", "DISTINCT"}:
            key = str(metadata.get("eventId") or item.fingerprint or item.title)
        else:
            key = item.fingerprint or item.title
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique
