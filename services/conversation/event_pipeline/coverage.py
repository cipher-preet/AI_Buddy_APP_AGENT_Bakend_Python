"""Coverage ledger and targeted repair. Silent data loss is a hard failure."""

from __future__ import annotations

from services.conversation.event_pipeline.channels import (
    action_object_grounded,
    action_strength,
    event_is_actionable,
    event_is_memory_worthy,
    object_grounding_type,
)
from services.conversation.event_pipeline.memory_identity import event_is_memory_candidate
from services.conversation.event_pipeline.schemas import (
    NON_PUBLISHABLE_KINDS,
    ActionDisposition,
    AtomicEvent,
    BlockDisposition,
    CoverageBlockRecord,
    CoverageEventRecord,
    CoverageLedger,
    CoverageSemanticUnitRecord,
    EventDisposition,
    EventKind,
    MemoryDisposition,
    MicroBlock,
    SemanticUnitDisposition,
)
from services.conversation.models import ExtractedNote, ExtractedTask


def build_coverage_ledger(
    *,
    total_raw_sequences: int,
    useful_sequences: int,
    excluded_structural_sequences: int,
    micro_blocks: list[MicroBlock],
    topics_count: int,
    events: list[AtomicEvent],
    tasks: list[ExtractedTask],
    notes: list[ExtractedNote],
) -> CoverageLedger:
    events_by_block: dict[str, list[AtomicEvent]] = {}
    for event in events:
        for block_id in event.microBlockIds or ["unassigned"]:
            events_by_block.setdefault(block_id, []).append(event)
    block_records: list[CoverageBlockRecord] = []
    unaccounted = 0
    for block in micro_blocks:
        related = events_by_block.get(block.microBlockId, [])
        if related:
            block_records.append(
                CoverageBlockRecord(
                    microBlockId=block.microBlockId,
                    sequenceIds=list(block.sequenceIds),
                    disposition=BlockDisposition.PRODUCED_EVENTS,
                    eventIds=[event.eventId for event in related],
                )
            )
        else:
            unaccounted += 1
            block_records.append(
                CoverageBlockRecord(
                    microBlockId=block.microBlockId,
                    sequenceIds=list(block.sequenceIds),
                    disposition=BlockDisposition.NO_EVENT,
                    reason="unaccounted",
                )
            )
    event_records = []
    rejected = 0
    for event in events:
        disposition = event.disposition or EventDisposition.REJECTED
        if disposition == EventDisposition.REJECTED:
            rejected += 1
        event_records.append(
            CoverageEventRecord(
                eventId=event.eventId,
                disposition=disposition,
                reason=event.dispositionReason or event.memoryDispositionReason or "",
                memoryDisposition=event.memoryDisposition.value if event.memoryDisposition else None,
                actionDisposition=event.actionDisposition.value if event.actionDisposition else None,
            )
        )
    action_events = [event for event in events if event_is_actionable(event)]
    memory_events = [event for event in events if event_is_memory_candidate(event)]
    other_events = [event for event in events if event.kind in NON_PUBLISHABLE_KINDS or event.channel == "other"]
    ledger = CoverageLedger(
        total_raw_sequences=total_raw_sequences,
        useful_sequences=useful_sequences,
        excluded_structural_sequences=excluded_structural_sequences,
        micro_blocks=len(micro_blocks),
        topics=topics_count,
        events=len(events),
        action_events=len(action_events),
        memory_events=len(memory_events),
        other_events=len(other_events),
        tasks_generated=len(tasks),
        notes_generated=len(notes),
        rejected_events=rejected,
        unaccounted_blocks=unaccounted,
        blocks=block_records,
        eventRecords=event_records,
    )
    if unaccounted > 0:
        ledger.hardFailure = True
        ledger.suspicious.append("unaccounted_blocks")
    apply_memory_coverage(ledger, events)
    apply_action_coverage(ledger, events)
    unpublished_memory = unpublished_memory_events(events)
    if unpublished_memory:
        ledger.hardFailure = True
        ledger.suspicious.append("unpublished_memory_events")
        if memory_events and not notes:
            ledger.suspicious.append("memory_events_without_notes")
    elif memory_events and not notes:
        leftover = [
            event
            for event in memory_events
            if event.memoryDisposition
            not in {
                MemoryDisposition.PUBLISHED_NOTE,
                MemoryDisposition.DUPLICATE,
                MemoryDisposition.SUPERSEDED,
                MemoryDisposition.LOW_VALUE,
                MemoryDisposition.UNSUPPORTED,
                MemoryDisposition.RELATED_CONTEXT_ONLY,
                MemoryDisposition.REJECTED_WITH_REASON,
            }
        ]
        if leftover:
            ledger.hardFailure = True
            ledger.suspicious.append("memory_events_without_notes")
    unpublished_actions = unpublished_action_events(events)
    if unpublished_actions:
        ledger.suspicious.append("unpublished_action_events")
    if action_events and not tasks:
        if unpublished_actions:
            ledger.suspicious.append("action_events_without_tasks")
    if _suspicious_zero_task_output(events, tasks):
        ledger.suspicious.append("SUSPICIOUS_ZERO_TASK_OUTPUT")
    return ledger


def mark_block_no_event(ledger: CoverageLedger, micro_block_id: str, reason: str) -> None:
    for record in ledger.blocks:
        if record.microBlockId == micro_block_id:
            record.disposition = BlockDisposition.NO_EVENT
            record.reason = reason
    ledger.unaccounted_blocks = sum(
        1 for record in ledger.blocks if record.disposition == BlockDisposition.NO_EVENT and record.reason == "unaccounted"
    )
    ledger.hardFailure = ledger.unaccounted_blocks > 0 or "memory_events_without_notes" in ledger.suspicious


_TERMINAL_ACTION = {
    ActionDisposition.PUBLISHED_TASK: "actionPublished",
    ActionDisposition.DUPLICATE: "actionDuplicates",
    ActionDisposition.SUPERSEDED: "actionSuperseded",
    ActionDisposition.UNSUPPORTED: "actionUnsupported",
    ActionDisposition.UNRESOLVED_OBJECT: "actionUnresolved",
    ActionDisposition.AMBIGUOUS: "actionAmbiguous",
    ActionDisposition.INTENTIONALLY_NONPUBLISHABLE: "actionNonpublishable",
    ActionDisposition.VALIDATION_REJECTED: "actionRejected",
}


def unpublished_action_events(events: list[AtomicEvent]) -> list[AtomicEvent]:
    return [
        event
        for event in events
        if event_is_actionable(event)
        and event.actionDisposition not in _TERMINAL_ACTION
        and event.disposition
        not in {
            EventDisposition.TASK,
            EventDisposition.INTENTIONALLY_NON_PUBLISHABLE,
            EventDisposition.DUPLICATE,
            EventDisposition.SUPERSEDED,
            EventDisposition.REJECTED,
        }
    ]


def apply_action_coverage(ledger: CoverageLedger, events: list[AtomicEvent]) -> None:
    counts = {field: 0 for field in _TERMINAL_ACTION.values()}
    unaccounted = 0
    action_events = [event for event in events if event_is_actionable(event)]
    for event in action_events:
        field = _TERMINAL_ACTION.get(event.actionDisposition)
        if field:
            counts[field] += 1
        elif event.disposition == EventDisposition.TASK:
            counts["actionPublished"] += 1
            if event.actionDisposition is None:
                event.actionDisposition = ActionDisposition.PUBLISHED_TASK
        elif event.disposition == EventDisposition.DUPLICATE:
            counts["actionDuplicates"] += 1
            event.actionDisposition = event.actionDisposition or ActionDisposition.DUPLICATE
        elif event.disposition == EventDisposition.SUPERSEDED:
            counts["actionSuperseded"] += 1
            event.actionDisposition = event.actionDisposition or ActionDisposition.SUPERSEDED
        elif event.disposition == EventDisposition.REJECTED:
            counts["actionRejected"] += 1
            event.actionDisposition = event.actionDisposition or ActionDisposition.VALIDATION_REJECTED
        elif event.disposition == EventDisposition.INTENTIONALLY_NON_PUBLISHABLE:
            if unresolved_reason(event):
                counts["actionUnresolved"] += 1
                event.actionDisposition = event.actionDisposition or ActionDisposition.UNRESOLVED_OBJECT
            else:
                counts["actionNonpublishable"] += 1
                event.actionDisposition = event.actionDisposition or ActionDisposition.INTENTIONALLY_NONPUBLISHABLE
        else:
            unaccounted += 1
    accounted = sum(counts.values())
    ledger.actionPublished = counts["actionPublished"]
    ledger.actionDuplicates = counts["actionDuplicates"]
    ledger.actionSuperseded = counts["actionSuperseded"]
    ledger.actionUnsupported = counts["actionUnsupported"]
    ledger.actionUnresolved = counts["actionUnresolved"]
    ledger.actionAmbiguous = counts["actionAmbiguous"]
    ledger.actionNonpublishable = counts["actionNonpublishable"]
    ledger.actionRejected = counts["actionRejected"]
    ledger.actionUnaccounted = unaccounted
    ledger.action_events = len(action_events)
    ledger.actionCoverageFailure = (accounted + unaccounted) != len(action_events) or unaccounted > 0
    if ledger.actionCoverageFailure:
        ledger.hardFailure = True
        if "ACTION_COVERAGE_FAILURE" not in ledger.suspicious:
            ledger.suspicious.append("ACTION_COVERAGE_FAILURE")
    elif "ACTION_COVERAGE_FAILURE" in ledger.suspicious:
        ledger.suspicious.remove("ACTION_COVERAGE_FAILURE")
    line = (
        f"[ACTION_COVERAGE] actionableEvents={len(action_events)} "
        f"published={ledger.actionPublished} duplicates={ledger.actionDuplicates} "
        f"superseded={ledger.actionSuperseded} unsupported={ledger.actionUnsupported} "
        f"unresolved={ledger.actionUnresolved} ambiguous={ledger.actionAmbiguous} "
        f"nonpublishable={ledger.actionNonpublishable} rejected={ledger.actionRejected} "
        f"unaccounted={ledger.actionUnaccounted}"
    )
    print(line)
    from services.conversation.event_pipeline.observability import current_observability

    obs = current_observability()
    if obs is not None:
        obs.logs.append(line)
        obs.actionCoverageFailures = 1 if ledger.actionCoverageFailure else 0
    by_id = {event.eventId: event for event in events}
    for record in ledger.eventRecords:
        event = by_id.get(record.eventId)
        if event is not None and event.actionDisposition is not None:
            record.actionDisposition = event.actionDisposition.value
            if not record.reason:
                record.reason = event.actionDispositionReason or event.dispositionReason or ""


def unresolved_reason(event: AtomicEvent) -> bool:
    reason = f"{event.actionDispositionReason or ''} {event.dispositionReason or ''}"
    return "unresolved" in reason or object_grounding_type(event) in {"INFERRED", "UNRESOLVED"}


def _suspicious_zero_task_output(events: list[AtomicEvent], tasks: list) -> bool:
    grounded_explicit = [
        event
        for event in events
        if event_is_actionable(event)
        and action_strength(event.actionSignal) == "EXPLICIT"
        and action_object_grounded(event)
        and object_grounding_type(event) not in {"INFERRED", "UNRESOLVED"}
    ]
    if not grounded_explicit or tasks:
        return False
    accounted = 0
    for event in grounded_explicit:
        if event.actionDisposition in {
            ActionDisposition.DUPLICATE,
            ActionDisposition.SUPERSEDED,
            ActionDisposition.UNSUPPORTED,
            ActionDisposition.UNRESOLVED_OBJECT,
            ActionDisposition.AMBIGUOUS,
            ActionDisposition.INTENTIONALLY_NONPUBLISHABLE,
            ActionDisposition.VALIDATION_REJECTED,
        }:
            accounted += 1
    return accounted < len(grounded_explicit)


def unpublished_memory_events(events: list[AtomicEvent]) -> list[AtomicEvent]:
    return [
        event
        for event in events
        if event_is_memory_candidate(event)
        and event.memoryDisposition
        not in {
            MemoryDisposition.PUBLISHED_NOTE,
            MemoryDisposition.DUPLICATE,
            MemoryDisposition.SUPERSEDED,
            MemoryDisposition.LOW_VALUE,
            MemoryDisposition.UNSUPPORTED,
            MemoryDisposition.RELATED_CONTEXT_ONLY,
            MemoryDisposition.REJECTED_WITH_REASON,
        }
        and event.disposition
        not in {
            EventDisposition.NOTE,
            EventDisposition.INTENTIONALLY_NON_PUBLISHABLE,
            EventDisposition.DUPLICATE,
            EventDisposition.SUPERSEDED,
            EventDisposition.REJECTED,
        }
    ]


_TERMINAL_MEMORY = {
    MemoryDisposition.PUBLISHED_NOTE: "memoryPublished",
    MemoryDisposition.DUPLICATE: "memoryDuplicates",
    MemoryDisposition.SUPERSEDED: "memorySuperseded",
    MemoryDisposition.LOW_VALUE: "memoryLowValue",
    MemoryDisposition.UNSUPPORTED: "memoryUnsupported",
    MemoryDisposition.RELATED_CONTEXT_ONLY: "memoryRelatedContext",
    MemoryDisposition.REJECTED_WITH_REASON: "memoryRejected",
}


def apply_memory_coverage(ledger: CoverageLedger, events: list[AtomicEvent]) -> None:
    counts = {field: 0 for field in _TERMINAL_MEMORY.values()}
    unaccounted = 0
    memory_events = [event for event in events if event_is_memory_candidate(event)]
    for event in memory_events:
        field = _TERMINAL_MEMORY.get(event.memoryDisposition)
        if field:
            counts[field] += 1
        elif event.disposition == EventDisposition.NOTE:
            counts["memoryPublished"] += 1
            if event.memoryDisposition is None:
                event.memoryDisposition = MemoryDisposition.PUBLISHED_NOTE
        elif event.disposition == EventDisposition.DUPLICATE:
            counts["memoryDuplicates"] += 1
            event.memoryDisposition = event.memoryDisposition or MemoryDisposition.DUPLICATE
        elif event.disposition == EventDisposition.SUPERSEDED:
            counts["memorySuperseded"] += 1
            event.memoryDisposition = event.memoryDisposition or MemoryDisposition.SUPERSEDED
        elif event.disposition == EventDisposition.REJECTED:
            counts["memoryRejected"] += 1
            event.memoryDisposition = event.memoryDisposition or MemoryDisposition.REJECTED_WITH_REASON
        elif event.disposition == EventDisposition.INTENTIONALLY_NON_PUBLISHABLE:
            counts["memoryLowValue"] += 1
            event.memoryDisposition = event.memoryDisposition or MemoryDisposition.LOW_VALUE
        else:
            unaccounted += 1
    accounted = sum(counts.values())
    ledger.memoryPublished = counts["memoryPublished"]
    ledger.memoryDuplicates = counts["memoryDuplicates"]
    ledger.memorySuperseded = counts["memorySuperseded"]
    ledger.memoryLowValue = counts["memoryLowValue"]
    ledger.memoryUnsupported = counts["memoryUnsupported"]
    ledger.memoryRelatedContext = counts["memoryRelatedContext"]
    ledger.memoryRejected = counts["memoryRejected"]
    ledger.memoryUnaccounted = unaccounted
    ledger.memory_events = len(memory_events)
    ledger.memoryUpdates = sum(
        1
        for event in memory_events
        if event.memoryDisposition == MemoryDisposition.PUBLISHED_NOTE
        and (event.dispositionReason in {"status_update", "supersede_prior_memory"} or "UPDATE" in (event.memoryDispositionReason or ""))
    )
    ledger.memoryCoverageFailure = (accounted + unaccounted) != len(memory_events) or unaccounted > 0
    if ledger.memoryCoverageFailure:
        ledger.hardFailure = True
        if "MEMORY_COVERAGE_FAILURE" not in ledger.suspicious:
            ledger.suspicious.append("MEMORY_COVERAGE_FAILURE")
    elif "MEMORY_COVERAGE_FAILURE" in ledger.suspicious:
        ledger.suspicious.remove("MEMORY_COVERAGE_FAILURE")
    line = (
        f"[MEMORY_COVERAGE] memoryEvents={len(memory_events)} "
        f"published={ledger.memoryPublished} duplicates={ledger.memoryDuplicates} "
        f"superseded={ledger.memorySuperseded} lowValue={ledger.memoryLowValue} "
        f"unsupported={ledger.memoryUnsupported} relatedContext={ledger.memoryRelatedContext} "
        f"rejected={ledger.memoryRejected} unaccounted={ledger.memoryUnaccounted}"
    )
    print(line)
    from services.conversation.event_pipeline.observability import current_observability

    obs = current_observability()
    if obs is not None:
        obs.logs.append(line)
    by_id = {event.eventId: event for event in events}
    for record in ledger.eventRecords:
        event = by_id.get(record.eventId)
        if event is not None and event.memoryDisposition is not None:
            record.memoryDisposition = event.memoryDisposition.value
            if not record.reason:
                record.reason = event.memoryDispositionReason or event.dispositionReason or ""


_TERMINAL_SEMANTIC = {
    SemanticUnitDisposition.EVENT_CREATED: "semanticUnitsCreated",
    SemanticUnitDisposition.MERGED_WITH_EVENT: "semanticUnitsMerged",
    SemanticUnitDisposition.LOW_VALUE: "semanticUnitsLowValue",
    SemanticUnitDisposition.NOISE: "semanticUnitsNoise",
    SemanticUnitDisposition.UNSUPPORTED: "semanticUnitsUnsupported",
    SemanticUnitDisposition.DUPLICATE: "semanticUnitsDuplicate",
    SemanticUnitDisposition.AMBIGUOUS: "semanticUnitsAmbiguous",
}


def apply_semantic_coverage(
    ledger: CoverageLedger,
    records: list[CoverageSemanticUnitRecord] | None,
    events: list[AtomicEvent] | None = None,
) -> None:
    from services.conversation.event_pipeline.completeness import bootstrap_semantic_records

    records = list(records or ledger.semanticUnitRecords or [])
    if not records and events:
        records = bootstrap_semantic_records([], events)
    counts = {field: 0 for field in _TERMINAL_SEMANTIC.values()}
    unaccounted = 0
    seen: set[tuple] = set()
    unique: list[CoverageSemanticUnitRecord] = []
    for record in records:
        if record.eventId:
            key = ("event", record.eventId, record.disposition.value if record.disposition else "none")
        else:
            key = ("unit", record.microBlockId, (record.meaning or "").casefold(), record.reason)
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
        field = _TERMINAL_SEMANTIC.get(record.disposition) if record.disposition is not None else None
        if field:
            counts[field] += 1
        else:
            unaccounted += 1
    detected = len(unique)
    accounted = sum(counts.values())
    ledger.semanticUnitRecords = unique
    ledger.semanticUnitsDetected = detected
    ledger.semanticUnitsCreated = counts["semanticUnitsCreated"]
    ledger.semanticUnitsMerged = counts["semanticUnitsMerged"]
    ledger.semanticUnitsLowValue = counts["semanticUnitsLowValue"]
    ledger.semanticUnitsNoise = counts["semanticUnitsNoise"]
    ledger.semanticUnitsAmbiguous = counts["semanticUnitsAmbiguous"]
    ledger.semanticUnitsUnsupported = counts["semanticUnitsUnsupported"]
    ledger.semanticUnitsDuplicate = counts["semanticUnitsDuplicate"]
    ledger.unaccountedSemanticUnits = unaccounted
    useful = detected - counts["semanticUnitsNoise"] - counts["semanticUnitsLowValue"]
    accounted_useful = useful - unaccounted
    ledger.semanticCoverage = (accounted_useful / useful) if useful else 1.0
    ledger.semanticCoverageFailure = unaccounted > 0 or (accounted + unaccounted) != detected
    if ledger.semanticCoverageFailure:
        ledger.hardFailure = True
        if "SEMANTIC_COVERAGE_FAILURE" not in ledger.suspicious:
            ledger.suspicious.append("SEMANTIC_COVERAGE_FAILURE")
    elif "SEMANTIC_COVERAGE_FAILURE" in ledger.suspicious:
        ledger.suspicious.remove("SEMANTIC_COVERAGE_FAILURE")
    line = (
        f"[SEMANTIC_COVERAGE] semanticUnitsDetected={detected} "
        f"eventsCreated={ledger.semanticUnitsCreated} merged={ledger.semanticUnitsMerged} "
        f"lowValue={ledger.semanticUnitsLowValue} noise={ledger.semanticUnitsNoise} "
        f"ambiguous={ledger.semanticUnitsAmbiguous} unsupported={ledger.semanticUnitsUnsupported} "
        f"duplicate={ledger.semanticUnitsDuplicate} unaccountedSemanticUnits={unaccounted} "
        f"semanticCoverage={ledger.semanticCoverage:.3f}"
    )
    print(line)
    from services.conversation.event_pipeline.observability import current_observability

    obs = current_observability()
    if obs is not None:
        obs.logs.append(line)
        obs.semanticCoverageFailures = 1 if ledger.semanticCoverageFailure else 0
        obs.unaccountedSemanticUnits = unaccounted
        obs.semanticUnitsDetected = detected
        obs.semanticUnitsCreated = ledger.semanticUnitsCreated


def trace_missing_memory(
    events: list[AtomicEvent],
    blocks: list[MicroBlock],
    topics: list,
    notes: list,
    gold_notes: list[dict] | None = None,
) -> list[dict]:
    """Trace memory-worthy events that did not publish a Note."""
    traces: list[dict] = []
    published_ids = set()
    for note in notes or []:
        metadata = getattr(note, "debug", None) or {}
        published_ids.update(metadata.get("sourceSemanticUnitIds") or [])
    by_block = {block.microBlockId: block for block in blocks}
    by_topic = {getattr(topic, "topicId", ""): topic for topic in topics or []}
    for event in events:
        if not event_is_memory_candidate(event):
            continue
        if event.memoryDisposition in {
            MemoryDisposition.PUBLISHED_NOTE,
            MemoryDisposition.DUPLICATE,
            MemoryDisposition.SUPERSEDED,
            MemoryDisposition.LOW_VALUE,
            MemoryDisposition.RELATED_CONTEXT_ONLY,
        } or event.eventId in published_ids:
            continue
        if event.disposition == EventDisposition.NOTE:
            continue
        block = by_block.get(event.microBlockIds[0]) if event.microBlockIds else None
        topic = by_topic.get(event.topicId)
        first_stage = _first_failure_stage(event)
        trace = {
            "sequenceIds": list(event.sequenceIds or []),
            "microBlock": getattr(block, "microBlockId", None),
            "topic": event.topicId,
            "topicLabel": getattr(topic, "label", None),
            "event": event.eventId,
            "meaning": event.meaning,
            "memoryWorthiness": bool(event.memorySignal and event.memorySignal.isMemoryWorthy),
            "importance": getattr(event.memorySignal, "importance", None) if event.memorySignal else None,
            "thread": event.threadId,
            "noteCandidate": event.disposition == EventDisposition.NOTE or event.eventId in published_ids,
            "finalDisposition": (event.memoryDisposition.value if event.memoryDisposition else None)
            or (event.disposition.value if event.disposition else None),
            "reason": event.memoryDispositionReason or event.dispositionReason or "",
            "firstFailureStage": first_stage,
        }
        line = (
            f"[MISSING_MEMORY_TRACE] sequenceIds={trace['sequenceIds']} "
            f"microBlock={trace['microBlock']} topic={trace['topic']} event={trace['event']} "
            f"memoryWorthiness={trace['memoryWorthiness']} thread={trace['thread']} "
            f"noteCandidate={trace['noteCandidate']} finalDisposition={trace['finalDisposition']} "
            f"firstFailureStage={first_stage}"
        )
        print(line)
        traces.append(trace)
        from services.conversation.event_pipeline.observability import current_observability

        obs = current_observability()
        if obs is not None:
            obs.logs.append(line)
    return traces


def _first_failure_stage(event: AtomicEvent) -> str:
    if event.memorySignal is None and event.kind not in {
        EventKind.DECISION,
        EventKind.REQUIREMENT,
        EventKind.ISSUE,
        EventKind.STATE,
        EventKind.RESULT,
        EventKind.FACT,
        EventKind.PROPOSAL,
        EventKind.CONSTRAINT,
        EventKind.IMPORTANT_CONTEXT,
        EventKind.CONTRADICTION,
        EventKind.OPEN_QUESTION,
    }:
        return "atomic_event"
    if event.memorySignal is not None and not event.memorySignal.isMemoryWorthy:
        return "memory_classification"
    if event.memoryDisposition == MemoryDisposition.LOW_VALUE:
        return "memory_classification"
    if event.memoryDisposition == MemoryDisposition.DUPLICATE:
        return "dedupe"
    if event.memoryDisposition == MemoryDisposition.REJECTED_WITH_REASON:
        return "validation"
    if event.memoryDisposition == MemoryDisposition.UNSUPPORTED:
        return "note_synthesis"
    if event.disposition == EventDisposition.TASK and event.memoryDisposition is None:
        return "task_note_overlap"
    return event.memoryDispositionReason or event.dispositionReason or "unknown"
