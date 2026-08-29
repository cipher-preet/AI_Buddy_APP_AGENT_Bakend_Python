"""Honest gold scoring. Valid additions are not automatic false positives."""

from __future__ import annotations

from enum import Enum
from typing import Any, Iterable

from services.conversation.eval_metrics import PredictedItem, score_case
from services.conversation.event_pipeline.channels import (
    action_object_grounded,
    action_strength,
    event_is_actionable,
    event_is_task_eligible,
    is_generic_task_text,
    is_structurally_generic,
    object_grounding_type,
)
from services.conversation.event_pipeline.schemas import ACTION_EVENT_KINDS, EventDisposition, EventKind, EventPipelineResult
from services.conversation.event_pipeline.object_canon import (
    canonicalize_action_object,
    objects_semantically_equivalent,
    surface_normalize_object,
)
from services.conversation.event_pipeline.textutil import (
    casefold_text,
    content_tokens,
    evidence_sequence_ids,
    information_density,
    token_jaccard,
    tokenize,
)
from services.conversation.event_pipeline.validation import mixed_thread_rate

NOT_MEASURED = "not measured"


class ArtifactLabel(str, Enum):
    MATCHED_GOLD = "MATCHED_GOLD"
    VALID_ADDITIONAL = "VALID_ADDITIONAL"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    DUPLICATE = "DUPLICATE"
    TOO_VAGUE = "TOO_VAGUE"
    UNSUPPORTED = "UNSUPPORTED"
    AMBIGUOUS = "AMBIGUOUS"


class NoteQuality(str, Enum):
    USEFUL_MEMORY = "USEFUL_MEMORY"
    LOW_VALUE_CONTEXT = "LOW_VALUE_CONTEXT"
    FILLER_DERIVED = "FILLER_DERIVED"
    DUPLICATE = "DUPLICATE"
    STATUS_UPDATE = "STATUS_UPDATE"
    UNSUPPORTED = "UNSUPPORTED"


class GoldReviewStatus(str, Enum):
    REQUIRED = "REQUIRED"
    OPTIONAL_VALID = "OPTIONAL_VALID"
    LOW_VALUE = "LOW_VALUE"
    INVALID_GOLD = "INVALID_GOLD"


class GoldFailureClass(str, Enum):
    EXTRACTION_MISS = "EXTRACTION_MISS"
    ACTIONABILITY_MISS = "ACTIONABILITY_MISS"
    OBJECT_GROUNDING_MISS = "OBJECT_GROUNDING_MISS"
    THREADING_MISS = "THREADING_MISS"
    SYNTHESIS_MISS = "SYNTHESIS_MISS"
    VALIDATION_REJECT = "VALIDATION_REJECT"
    DEDUPE_ERROR = "DEDUPE_ERROR"
    SCORER_SEMANTIC_MISMATCH = "SCORER_SEMANTIC_MISMATCH"
    ACTUALLY_NOT_ACTIONABLE = "ACTUALLY_NOT_ACTIONABLE"


class NoteReviewClass(str, Enum):
    USEFUL_MEMORY = "USEFUL_MEMORY"
    LOW_VALUE_CONTEXT = "LOW_VALUE_CONTEXT"
    DUPLICATE_CONTEXT = "DUPLICATE_CONTEXT"
    TASK_SUPPORTING_CONTEXT = "TASK_SUPPORTING_CONTEXT"
    EXTRACTION_MISS = "EXTRACTION_MISS"


_VERB_FAMILIES = (
    frozenset({"keep", "preserve", "maintain", "retain", "stay", "leave", "rakhna", "rakho", "rakh", "rahegi", "rahega"}),
    frozenset({"hold", "pause", "block", "stop", "prevent", "defer", "withhold", "avoid"}),
    frozenset({"update", "revise", "edit", "change", "document"}),
    frozenset({"fix", "repair", "resolve", "correct"}),
    frozenset({"create", "make", "add", "open", "start"}),
    frozenset({"share", "send", "provide"}),
    frozenset({"check", "review", "inspect", "verify"}),
    frozenset({"follow", "track"}),
    frozenset({"sign", "execute"}),
    frozenset({"test", "try"}),
    frozenset({"request", "ask"}),
    frozenset({"use", "apply"}),
    frozenset({"finalize", "lock", "confirm"}),
    frozenset({"put", "ship", "release", "deploy", "dalna", "dalo"}),
)
_NEGATION = frozenset({"not", "no", "mat", "nahi", "dont", "never", "n't"})
_BLOCKING_VERBS = frozenset({"hold", "pause", "block", "stop", "prevent", "withhold", "avoid"})


def predicted_from_pipeline(result: EventPipelineResult) -> list[PredictedItem]:
    items: list[PredictedItem] = []
    for task in result.tasks:
        items.append(
            PredictedItem(
                kind="task",
                meaning=f"{task.title} {task.body}",
                evidenceSequences=evidence_sequence_ids(task.evidence),
                ownerText=task.ownerText,
                dueDateText=task.dueDateText,
            )
        )
    for note in result.notes:
        items.append(
            PredictedItem(
                kind="note",
                meaning=f"{note.title} {note.body}",
                evidenceSequences=evidence_sequence_ids(note.evidence),
            )
        )
    return items


def classify_generated_items(
    generated: list,
    gold_items: list[dict],
    valid_additional: list[dict] | None = None,
    *,
    kind: str,
    sequence_text: dict[int, str] | None = None,
    events: list | None = None,
) -> list[dict[str, Any]]:
    remaining_gold = list(gold_items or [])
    remaining_extra = list(valid_additional or [])
    used_meanings: list[str] = []
    rows: list[dict[str, Any]] = []
    by_id = {getattr(event, "eventId", ""): event for event in (events or [])}
    for item in generated:
        meaning = f"{getattr(item, 'title', '')} {getattr(item, 'body', '')}".strip()
        sequences = evidence_sequence_ids(getattr(item, "evidence", []))
        source_event = _source_event(item, by_id)
        gold_hit = _best_match(meaning, sequences, remaining_gold)
        extra_hit = _best_match(meaning, sequences, remaining_extra)
        if gold_hit is not None:
            remaining_gold.remove(gold_hit)
            label = ArtifactLabel.MATCHED_GOLD
            matched = gold_hit
        elif extra_hit is not None:
            remaining_extra.remove(extra_hit)
            label = ArtifactLabel.VALID_ADDITIONAL
            matched = extra_hit
        elif _is_duplicate(meaning, used_meanings):
            label = ArtifactLabel.DUPLICATE
            matched = None
        elif kind == "task" and is_generic_task_text(getattr(item, "title", ""), getattr(item, "body", "")):
            label = ArtifactLabel.TOO_VAGUE
            matched = None
        elif is_structurally_generic(meaning):
            label = ArtifactLabel.TOO_VAGUE
            matched = None
        elif not _evidence_supported(item, sequence_text or {}):
            label = ArtifactLabel.UNSUPPORTED
            matched = None
        elif kind == "task" and _task_is_non_action(source_event, item):
            label = ArtifactLabel.FALSE_POSITIVE
            matched = None
        elif _looks_grounded_and_specific(meaning, sequences, sequence_text or {}):
            label = ArtifactLabel.VALID_ADDITIONAL
            matched = None
        else:
            label = ArtifactLabel.FALSE_POSITIVE
            matched = None
        used_meanings.append(meaning)
        rows.append(
            {
                "title": getattr(item, "title", ""),
                "meaning": meaning,
                "label": label.value,
                "evidenceSequences": sequences,
                "matchedGoldId": (matched or {}).get("id") if matched else None,
                "actionRole": _item_action_role(item, source_event),
                "verb": getattr(getattr(source_event, "actionSignal", None), "verb", None) if source_event else None,
                "object": (getattr(source_event, "object", None) if source_event else None),
                "objectGroundingType": object_grounding_type(source_event) if source_event else None,
                "threadId": getattr(source_event, "threadId", None) if source_event else (getattr(item, "changes", None) or getattr(item, "debug", None) or {}).get("threadId"),
            }
        )
    return rows


def pipeline_benchmark(
    result: EventPipelineResult,
    gold_tasks: list[dict],
    gold_notes: list[dict],
    case_id: str = "gold",
    transcript: str = "",
    valid_additional_notes: list[dict] | None = None,
    valid_additional_tasks: list[dict] | None = None,
    gold_events: list | None = None,
    gold_threads: list[list[str]] | None = None,
    gold_complete: bool = True,
    original_actionable_ids: list[str] | None = None,
    reviewed_actionable_ids: list[str] | None = None,
) -> dict[str, Any]:
    case = {
        "id": case_id,
        "category": "long_meeting",
        "transcript": transcript,
        "goldTasks": gold_tasks,
        "goldNotes": gold_notes,
    }
    score = score_case(case, predicted_from_pipeline(result))
    sequence_text = _sequence_text_from_transcript(transcript)
    note_rows = classify_generated_items(
        result.notes,
        gold_notes,
        valid_additional_notes,
        kind="note",
        sequence_text=sequence_text,
        events=result.events,
    )
    task_rows = classify_generated_items(
        result.tasks,
        gold_tasks,
        valid_additional_tasks,
        kind="task",
        sequence_text=sequence_text,
        events=result.events,
    )
    note_counts = _label_counts(note_rows)
    task_counts = _label_counts(task_rows)
    generic = sum(1 for task in result.tasks if is_generic_task_text(task.title, task.body))
    mixed = mixed_thread_rate([*result.tasks, *result.notes], result.events)
    unaccounted = result.coverage.unaccounted_blocks if result.coverage else 0
    grounded_note_den = max(len(result.notes), 1)
    grounded_note_num = note_counts[ArtifactLabel.MATCHED_GOLD] + note_counts[ArtifactLabel.VALID_ADDITIONAL]
    grounded_task_num = task_counts[ArtifactLabel.MATCHED_GOLD] + task_counts[ArtifactLabel.VALID_ADDITIONAL]
    strict_note_precision = _ratio(note_counts[ArtifactLabel.MATCHED_GOLD], len(result.notes))
    grounded_note_precision = _ratio(grounded_note_num, grounded_note_den if result.notes else 0)
    strict_task_precision = _ratio(task_counts[ArtifactLabel.MATCHED_GOLD], len(result.tasks))
    grounded_task_precision = _ratio(grounded_task_num, len(result.tasks) if result.tasks else 0)
    event_metrics = score_events(result.events, gold_events) if gold_events is not None else _unmeasured_event_metrics()
    thread_metrics = score_threads(result.events, gold_threads) if gold_threads is not None else _unmeasured_thread_metrics()
    action_metrics = (
        score_action_signals(
            result.events,
            gold_events,
            original_actionable_ids=original_actionable_ids,
            reviewed_actionable_ids=reviewed_actionable_ids,
        )
        if gold_events is not None
        else _unmeasured_action_metrics()
    )
    note_quality = classify_note_quality(note_rows, result.notes, result.events, sequence_text, valid_additional_notes)
    grounded_objects = sum(1 for event in result.events if event_is_actionable(event) and action_object_grounded(event))
    review = _review_metrics(gold_tasks, gold_notes, task_rows, note_rows)
    gold_traces = build_gold_traces(result, gold_tasks, gold_notes, task_rows, note_rows)
    gold_failures = [
        row
        for row in gold_traces
        if row.get("kind") == "task" and gold_review_status(_gold_by_id(gold_tasks, row.get("goldId"))) == GoldReviewStatus.REQUIRED and row.get("failureClass")
    ]
    report = {
        "extractorMode": "scripted" if _scripted(result) else "real_or_mixed",
        "goldComplete": gold_complete,
        "expectedTasks": score.goldTaskCount,
        "generatedTasks": score.predictedTaskCount,
        "expectedNotes": score.goldNoteCount,
        "generatedNotes": score.predictedNoteCount,
        "matchedTasks": task_counts[ArtifactLabel.MATCHED_GOLD],
        "matchedNotes": note_counts[ArtifactLabel.MATCHED_GOLD],
        "validAdditionalTasks": task_counts[ArtifactLabel.VALID_ADDITIONAL],
        "validAdditionalNotes": note_counts[ArtifactLabel.VALID_ADDITIONAL],
        "falsePositiveTasks": task_counts[ArtifactLabel.FALSE_POSITIVE],
        "falsePositiveNotes": note_counts[ArtifactLabel.FALSE_POSITIVE],
        "duplicateTasks": task_counts[ArtifactLabel.DUPLICATE],
        "duplicateNotes": note_counts[ArtifactLabel.DUPLICATE],
        "tooVagueTasks": task_counts[ArtifactLabel.TOO_VAGUE],
        "tooVagueNotes": note_counts[ArtifactLabel.TOO_VAGUE],
        "unsupportedTasks": task_counts[ArtifactLabel.UNSUPPORTED],
        "unsupportedNotes": note_counts[ArtifactLabel.UNSUPPORTED],
        "validUsefulAdditionalNotes": note_quality["validUsefulAdditional"],
        "lowValueGroundedNotes": note_quality["lowValueGrounded"],
        "fillerDerivedNotes": note_quality["fillerDerived"],
        "statusUpdateNotes": note_quality["statusUpdate"],
        "noteFactualPrecision": grounded_note_precision,
        "noteUsefulnessPrecision": note_quality["usefulnessPrecision"],
        "noteDuplicateRate": _ratio(note_counts[ArtifactLabel.DUPLICATE], len(result.notes) if result.notes else 0),
        "duplicateTaskRate": _ratio(task_counts[ArtifactLabel.DUPLICATE], len(result.tasks) if result.tasks else 0),
        "duplicateArtifactRate": _ratio(
            note_counts[ArtifactLabel.DUPLICATE] + task_counts[ArtifactLabel.DUPLICATE],
            max(len(result.notes) + len(result.tasks), 1),
        ),
        "memoryCoverageFailure": bool(result.coverage.memoryCoverageFailure) if result.coverage else False,
        "memoryUnaccounted": result.coverage.memoryUnaccounted if result.coverage else 0,
        "memoryPublished": result.coverage.memoryPublished if result.coverage else 0,
        "memoryDuplicates": result.coverage.memoryDuplicates if result.coverage else 0,
        "memoryUpdates": result.coverage.memoryUpdates if result.coverage else 0,
        "memorySuperseded": result.coverage.memorySuperseded if result.coverage else 0,
        "memoryLowValue": result.coverage.memoryLowValue if result.coverage else 0,
        "memoryUnsupported": result.coverage.memoryUnsupported if result.coverage else 0,
        "memoryRelatedContext": result.coverage.memoryRelatedContext if result.coverage else 0,
        "memoryRejected": result.coverage.memoryRejected if result.coverage else 0,
        "memoryEvents": result.coverage.memory_events if result.coverage else 0,
        "noteQuality": note_quality["rows"],
        "missingNotes": review["missingRequiredNotes"],
        "missingTasks": review["missingRequiredTasks"],
        "strictGoldPrecisionNotes": strict_note_precision if gold_complete else NOT_MEASURED,
        "groundedPrecisionNotes": grounded_note_precision,
        "strictGoldPrecisionTasks": strict_task_precision if gold_complete else NOT_MEASURED,
        "groundedPrecisionTasks": grounded_task_precision,
        "taskPrecision": score.taskPrecision if gold_complete else NOT_MEASURED,
        "taskRecall": review["requiredTaskRecall"] if gold_complete else NOT_MEASURED,
        "notePrecision": strict_note_precision if gold_complete else NOT_MEASURED,
        "noteRecall": review["requiredNoteRecall"] if gold_complete else NOT_MEASURED,
        "requiredTaskRecall": review["requiredTaskRecall"],
        "requiredNoteRecall": review["requiredNoteRecall"],
        "optionalValidFound": review["optionalValidFound"],
        "lowValueSuppressed": review["lowValueSuppressed"],
        "invalidGoldCount": review["invalidGoldCount"],
        "requiredTaskCount": review["requiredTaskCount"],
        "requiredNoteCount": review["requiredNoteCount"],
        "goldTraces": gold_traces,
        "goldFailures": gold_failures,
        "evidencePrecision": score.evidenceAccuracy if gold_complete else NOT_MEASURED,
        "mixedThreadRate": mixed,
        "genericTaskRate": generic / max(len(result.tasks), 1) if result.tasks else 0.0,
        "unaccountedBlocks": unaccounted,
        "unaccountedSemanticUnits": result.coverage.unaccountedSemanticUnits if result.coverage else 0,
        "semanticCoverage": result.coverage.semanticCoverage if result.coverage else 1.0,
        "semanticCoverageFailure": bool(result.coverage.semanticCoverageFailure) if result.coverage else False,
        "semanticUnitsDetected": result.coverage.semanticUnitsDetected if result.coverage else 0,
        "semanticUnitsCreated": result.coverage.semanticUnitsCreated if result.coverage else 0,
        "duplicateRate": score.duplicateRate,
        "noteClassifications": note_rows,
        "taskClassifications": task_rows,
        **event_metrics,
        **thread_metrics,
        **action_metrics,
        "counts": {
            "rawChunks": result.cleaning.totalSequences if result.cleaning else None,
            "usefulChunks": result.cleaning.usefulSequences if result.cleaning else None,
            "microBlocks": len(result.microBlocks),
            "topics": len(result.topics),
            "events": len(result.events),
            "threads": len(result.threads),
            "actionEvents": result.coverage.action_events if result.coverage else sum(1 for event in result.events if event_is_actionable(event)),
            "memoryEvents": result.coverage.memory_events if result.coverage else sum(1 for event in result.events if event.channel == "memory" or (event.memorySignal and event.memorySignal.isMemoryWorthy)),
            "groundedActionObjects": grounded_objects,
            "tasks": len(result.tasks),
            "notes": len(result.notes),
            "rejected": result.coverage.rejected_events if result.coverage else 0,
            "unaccounted": unaccounted,
        },
        "observability": {
            "llmCalls": result.observability.llm_calls(),
            "embeddingCalls": result.observability.embedding_calls(),
            "embeddingItems": result.observability.embeddingItems,
            "gemmaCalls": result.observability.gemmaCalls,
            "gptOss120bCalls": result.observability.gptOss120bCalls,
            "gptOss20bCalls": result.observability.gptOss20bCalls,
            "tokens": result.observability.tokens(),
            "inputTokens": result.observability.inputTokens,
            "outputTokens": result.observability.outputTokens,
            "fallbackCount": result.observability.fallbackCount,
            "retryCount": result.observability.retryCount,
            "comparisonCount": result.observability.comparisonCount,
            "stageMs": {stage.name: stage.durationMs for stage in result.observability.stages},
            "estimatedCostUsd": result.observability.estimatedCostUsd
            if result.observability.estimatedCostUsd is not None
            else NOT_MEASURED,
            "modelRoutes": [item.model_dump() for item in result.observability.modelRoutes],
        },
    }
    return report


def score_events(events, gold_events: list | None) -> dict[str, Any]:
    if gold_events is None:
        return _unmeasured_event_metrics()
    predicted = [event for event in events if getattr(event, "kind", None) != EventKind.NOISE]
    remaining = list(gold_events)
    matched = 0
    type_ok = 0
    evidence_ok = 0
    actor_ok = 0
    deadline_ok = 0
    actor_n = 0
    deadline_n = 0
    merged = 0
    unsupported = 0
    used: set[int] = set()
    for event in predicted:
        index = _best_event_index(event, remaining)
        if index is None:
            if not event.evidence:
                unsupported += 1
            continue
        gold = remaining.pop(index)
        matched += 1
        gold_kind = gold.kind if hasattr(gold, "kind") else gold.get("kind")
        if str(event.kind.value if hasattr(event.kind, "value") else event.kind) == str(
            gold_kind.value if hasattr(gold_kind, "value") else gold_kind
        ):
            type_ok += 1
        gold_seqs = set(getattr(gold, "sequenceIds", None) or gold.get("evidenceSequences") or gold.get("sequenceIds") or [])
        pred_seqs = set(event.sequenceIds or evidence_sequence_ids(event.evidence))
        if pred_seqs and (not gold_seqs or pred_seqs & gold_seqs):
            evidence_ok += 1
        gold_actor = getattr(gold, "actor", None) if not isinstance(gold, dict) else gold.get("actor")
        if gold_actor:
            actor_n += 1
            if casefold_text(event.actor or "") == casefold_text(gold_actor):
                actor_ok += 1
        gold_time = getattr(gold, "timeExpression", None) if not isinstance(gold, dict) else gold.get("timeExpression")
        if gold_time:
            deadline_n += 1
            if casefold_text(event.timeExpression or "") == casefold_text(gold_time) or casefold_text(gold_time) in casefold_text(
                event.timeExpression or ""
            ):
                deadline_ok += 1
        key = _event_match_key(event)
        if key in used:
            merged += 1
        used.add(key)
    generated = len(predicted)
    expected = len(gold_events)
    return {
        "eventExpected": expected,
        "eventGenerated": generated,
        "eventRecall": _ratio(matched, expected),
        "eventPrecision": _ratio(matched, generated),
        "eventTypeAccuracy": _ratio(type_ok, matched),
        "eventEvidenceAccuracy": _ratio(evidence_ok, matched),
        "eventActorAccuracy": _ratio(actor_ok, actor_n) if actor_n else NOT_MEASURED,
        "eventDeadlineAccuracy": _ratio(deadline_ok, deadline_n) if deadline_n else NOT_MEASURED,
        "unsupportedInferenceRate": _ratio(unsupported, generated),
        "mergedEventRate": _ratio(merged, generated),
    }


def score_action_signals(
    events,
    gold_events: list | None,
    *,
    original_actionable_ids: list[str] | None = None,
    reviewed_actionable_ids: list[str] | None = None,
) -> dict[str, Any]:
    if gold_events is None:
        return _unmeasured_action_metrics()
    predicted = [event for event in events if getattr(event, "kind", None) != EventKind.NOISE]
    original_ids = {str(item) for item in (original_actionable_ids or [])}
    reviewed_ids = {str(item) for item in (reviewed_actionable_ids or [])}
    if not original_ids:
        original_ids = {str(getattr(gold, "eventId", "") or gold.get("eventId")) for gold in gold_events if _gold_is_actionable(gold)}
    if not reviewed_ids:
        reviewed_ids = set(original_ids)
        for gold in gold_events:
            if _gold_is_actionable(gold):
                reviewed_ids.add(str(getattr(gold, "eventId", "") or gold.get("eventId")))
    original_gold = [gold for gold in gold_events if _gold_id(gold) in original_ids or (not original_actionable_ids and _gold_is_actionable(gold))]
    reviewed_gold = [gold for gold in gold_events if _gold_id(gold) in reviewed_ids or _gold_is_actionable(gold)]
    remaining_reviewed = list(reviewed_gold)
    remaining_original = list(original_gold)
    remaining_all = list(gold_events)
    action_rows: list[dict[str, Any]] = []
    used_meanings: list[str] = []
    action_tp = 0
    action_fp = 0
    verb_tp = 0
    verb_fp = 0
    verb_fn = 0
    object_tp = 0
    object_fp = 0
    object_fn = 0
    object_ok = 0
    object_n = 0
    verb_ok = 0
    verb_n = 0
    grounded_tp = 0
    grounded_fp = 0
    grounded_fn = 0
    actor_ok = 0
    actor_n = 0
    deadline_ok = 0
    deadline_n = 0
    generic_actions = 0
    unsupported_actions = 0
    explicit_ok = 0
    explicit_n = 0
    coref_ok = 0
    coref_n = 0
    inferred_rejected = 0
    inferred_n = 0
    semantic_ok = 0
    surface_ok = 0
    grounding_ok = 0
    grounding_n = 0
    original_matched = 0
    reviewed_matched = 0
    valid_additional_actions = 0
    actual_fp = 0
    duplicate_actions = 0
    ambiguous_actions = 0
    failures: list[dict[str, Any]] = []

    for event in predicted:
        is_actionable = event_is_actionable(event)
        gold_index = _best_event_index(event, remaining_all)
        gold = remaining_all[gold_index] if gold_index is not None else None
        if gold is not None:
            remaining_all.pop(gold_index)
        pred_obj = _predicted_action_object(event)
        gold_obj = _gold_field(gold, "object") if gold is not None else None
        grounded = action_object_grounded(event) if is_actionable else False
        grounding = object_grounding_type(event)
        evidence_text = " ".join(span.text for span in (event.evidence or []))
        if is_actionable and is_structurally_generic(event.meaning) and not grounded:
            generic_actions += 1
        if is_actionable and not event.evidence:
            unsupported_actions += 1
        if is_actionable:
            row = _classify_action_event(event, gold, remaining_original, remaining_reviewed, used_meanings)
            action_rows.append(row)
            used_meanings.append(event.meaning)
            if row["label"] == ArtifactLabel.MATCHED_GOLD.value:
                original_matched += 1
                reviewed_matched += 1
                action_tp += 1
            elif row["label"] == ArtifactLabel.VALID_ADDITIONAL.value:
                valid_additional_actions += 1
                reviewed_matched += 1
                action_tp += 1
            elif row["label"] == ArtifactLabel.DUPLICATE.value:
                duplicate_actions += 1
            elif row["label"] == ArtifactLabel.AMBIGUOUS.value:
                ambiguous_actions += 1
            elif row["label"] == ArtifactLabel.UNSUPPORTED.value:
                unsupported_actions += 1
                action_fp += 1
            else:
                actual_fp += 1
                action_fp += 1
        gold_actionable = _gold_is_actionable(gold) if gold is not None else False
        pred_verb = getattr(event.actionSignal, "verb", None) if event.actionSignal else None
        gold_verb = _gold_field(gold, "verb") or _gold_nested(gold, "actionSignal", "verb") if gold is not None else None
        expected_grounding = _gold_nested(gold, "actionSignal", "objectGroundingType") if gold is not None else None
        if expected_grounding is None and gold is not None:
            expected_grounding = _gold_field(gold, "objectGroundingType")
        # Object/verb metrics only when the gold event is actually an action.
        if gold is not None and gold_actionable:
            if gold_verb:
                verb_n += 1
                if _field_match(pred_verb, gold_verb):
                    verb_ok += 1
                    verb_tp += 1
                else:
                    verb_fn += 1
                    if pred_verb:
                        verb_fp += 1
            elif pred_verb and is_actionable:
                verb_fp += 1
            if gold_obj:
                object_n += 1
                semantic = objects_semantically_equivalent(pred_obj, gold_obj, evidence_text)
                surface = surface_normalize_object(pred_obj) == surface_normalize_object(gold_obj) or (
                    pred_obj
                    and gold_obj
                    and surface_normalize_object(pred_obj) in surface_normalize_object(gold_obj)
                )
                if semantic:
                    object_ok += 1
                    object_tp += 1
                    semantic_ok += 1
                else:
                    object_fn += 1
                    if pred_obj:
                        object_fp += 1
                    failures.append(_object_failure(event, gold_obj, pred_obj, grounding, accepted=bool(pred_obj and grounded), semantic=semantic))
                if surface:
                    surface_ok += 1
            elif pred_obj and is_actionable:
                object_fp += 1
            gold_actor = _gold_field(gold, "actor")
            if gold_actor:
                actor_n += 1
                if _field_match(event.actor, gold_actor):
                    actor_ok += 1
            gold_deadline = _gold_field(gold, "timeExpression") or _gold_field(gold, "deadline")
            if gold_deadline:
                deadline_n += 1
                if _field_match(event.timeExpression, gold_deadline):
                    deadline_ok += 1
            if str(expected_grounding or "").upper() == "EXPLICIT" or (
                gold_actionable and gold_obj and str(expected_grounding or "").upper() != "LOCAL_COREFERENCE"
            ):
                explicit_n += 1
                if grounding == "EXPLICIT" and objects_semantically_equivalent(pred_obj, gold_obj, evidence_text):
                    explicit_ok += 1
            if str(expected_grounding or "").upper() == "LOCAL_COREFERENCE":
                coref_n += 1
                if grounding == "LOCAL_COREFERENCE" and objects_semantically_equivalent(pred_obj, gold_obj, evidence_text):
                    coref_ok += 1
            if str(expected_grounding or "").upper() == "INFERRED" or _gold_field(gold, "inferredObject"):
                inferred_n += 1
                if grounding in {"INFERRED", "UNRESOLVED"} or not pred_obj:
                    inferred_rejected += 1
            grounding_n += 1
            if grounded:
                grounding_ok += 1
                if objects_semantically_equivalent(pred_obj, gold_obj, evidence_text):
                    grounded_tp += 1
                else:
                    grounded_fp += 1
            else:
                grounded_fn += 1
        elif is_actionable and gold is None:
            if pred_verb:
                verb_fp += 1
            if pred_obj:
                object_fp += 1
            if grounded:
                grounded_fp += 1
            if grounding == "INFERRED":
                inferred_n += 1
                inferred_rejected += 1
            failures.append(_object_failure(event, None, pred_obj, grounding, accepted=bool(pred_obj and grounded)))

    action_fn = sum(1 for gold in remaining_reviewed if _gold_is_actionable(gold) or _gold_id(gold) in reviewed_ids)
    predicted_action_n = sum(1 for event in predicted if event_is_actionable(event))
    gold_action_n = len(reviewed_gold) if reviewed_gold else sum(1 for gold in gold_events if _gold_is_actionable(gold))
    original_n = len(original_gold)
    for gold in remaining_reviewed:
        gold_verb = _gold_field(gold, "verb") or _gold_nested(gold, "actionSignal", "verb")
        gold_obj = _gold_field(gold, "object")
        if gold_verb:
            verb_fn += 1
        if gold_obj and (_gold_is_actionable(gold) or _gold_id(gold) in reviewed_ids):
            object_fn += 1
            failures.append(
                {
                    "evidence": _gold_evidence_text(gold),
                    "expectedObject": gold_obj,
                    "generatedObject": None,
                    "groundingType": None,
                    "reason": "missed_gold_object",
                    "accepted": False,
                }
            )
    return {
        "actionabilityPrecision": _ratio(action_tp, action_tp + action_fp),
        "actionabilityRecall": _ratio(action_tp, gold_action_n if gold_action_n else action_tp + action_fn),
        "strictOriginalGoldPrecision": _ratio(original_matched, predicted_action_n),
        "reviewedActionPrecision": _ratio(original_matched + valid_additional_actions, predicted_action_n),
        "reviewedActionRecall": _ratio(reviewed_matched, gold_action_n),
        "validAdditionalActionCount": valid_additional_actions,
        "actualFalsePositiveCount": actual_fp,
        "actionClassifications": action_rows,
        "actionVerbAccuracy": _ratio(verb_ok, verb_n) if verb_n else NOT_MEASURED,
        "actionVerbPrecision": _ratio(verb_tp, verb_tp + verb_fp),
        "actionVerbRecall": _ratio(verb_tp, verb_tp + verb_fn),
        "actionObjectAccuracy": _ratio(object_ok, object_n) if object_n else NOT_MEASURED,
        "actionObjectPrecision": _ratio(object_tp, object_tp + object_fp),
        "actionObjectRecall": _ratio(object_tp, object_tp + object_fn),
        "objectSemanticAccuracy": _ratio(semantic_ok, object_n) if object_n else NOT_MEASURED,
        "objectSurfaceNormalizationAccuracy": _ratio(surface_ok, object_n) if object_n else NOT_MEASURED,
        "objectGroundingAccuracy": _ratio(grounding_ok, grounding_n) if grounding_n else NOT_MEASURED,
        "explicitObjectAccuracy": _ratio(explicit_ok, explicit_n) if explicit_n else NOT_MEASURED,
        "coreferenceObjectAccuracy": _ratio(coref_ok, coref_n) if coref_n else NOT_MEASURED,
        "inferredObjectRejectionRate": _ratio(inferred_rejected, inferred_n) if inferred_n else NOT_MEASURED,
        "actionObjectGroundingPrecision": _ratio(grounded_tp, grounded_tp + grounded_fp),
        "actionObjectGroundingRecall": _ratio(grounded_tp, grounded_tp + grounded_fn),
        "actionActorAccuracy": _ratio(actor_ok, actor_n) if actor_n else NOT_MEASURED,
        "actionDeadlineAccuracy": _ratio(deadline_ok, deadline_n) if deadline_n else NOT_MEASURED,
        "genericActionRate": _ratio(generic_actions, predicted_action_n) if predicted_action_n else 0.0,
        "unsupportedActionRate": _ratio(unsupported_actions, predicted_action_n) if predicted_action_n else 0.0,
        "groundedActionObjects": grounded_tp + grounded_fp,
        "actionObjectFailures": failures,
        "duplicateActionCount": duplicate_actions,
        "ambiguousActionCount": ambiguous_actions,
        "originalActionCount": original_n,
        "reviewedActionCount": gold_action_n,
    }


def _gold_id(gold) -> str:
    if gold is None:
        return ""
    if isinstance(gold, dict):
        return str(gold.get("eventId") or gold.get("id") or "")
    return str(getattr(gold, "eventId", "") or "")


def _classify_action_event(event, gold, remaining_original, remaining_reviewed, used_meanings: list[str]) -> dict[str, Any]:
    meaning = event.meaning
    sequences = list(event.sequenceIds or evidence_sequence_ids(event.evidence))
    original_hit = _best_event_index(event, remaining_original)
    reviewed_hit = _best_event_index(event, remaining_reviewed)
    if original_hit is not None:
        label = ArtifactLabel.MATCHED_GOLD
        matched = remaining_original.pop(original_hit)
        remaining_reviewed[:] = [item for item in remaining_reviewed if _gold_id(item) != _gold_id(matched)]
    elif reviewed_hit is not None:
        label = ArtifactLabel.VALID_ADDITIONAL
        matched = remaining_reviewed.pop(reviewed_hit)
    elif _is_duplicate(meaning, used_meanings):
        label = ArtifactLabel.DUPLICATE
        matched = None
    elif not event.evidence:
        label = ArtifactLabel.UNSUPPORTED
        matched = None
    elif action_strength(event.actionSignal) in {"NONE", "POSSIBLE"}:
        label = ArtifactLabel.AMBIGUOUS
        matched = None
    else:
        label = ArtifactLabel.FALSE_POSITIVE
        matched = None
    return {
        "meaning": meaning,
        "label": label.value,
        "evidenceSequences": sequences,
        "matchedGoldId": _gold_id(matched) if matched else None,
        "verb": getattr(event.actionSignal, "verb", None) if event.actionSignal else None,
        "object": _predicted_action_object(event),
        "rawActionObject": getattr(event.actionSignal, "rawActionObject", None) if event.actionSignal else None,
        "canonicalActionObject": getattr(event.actionSignal, "canonicalActionObject", None) if event.actionSignal else event.object,
        "objectGroundingType": object_grounding_type(event),
    }


def classify_note_quality(
    note_rows: list[dict[str, Any]],
    notes: list,
    events: list | None,
    sequence_text: dict[int, str],
    valid_additional: list[dict] | None,
) -> dict[str, Any]:
    by_id = {getattr(event, "eventId", ""): event for event in (events or [])}
    useful = 0
    low_value = 0
    filler = 0
    duplicate = 0
    status_update = 0
    unsupported = 0
    rows: list[dict[str, Any]] = []
    useful_extra_ids = {str(item.get("id")) for item in (valid_additional or [])}
    for row, note in zip(note_rows, notes):
        label = row.get("label")
        source = _source_event(note, by_id)
        evidence_blob = " ".join(sequence_text.get(seq, "") for seq in row.get("evidenceSequences") or [])
        density = information_density(evidence_blob or f"{row.get('title')} {row.get('meaning')}")
        quality = NoteQuality.LOW_VALUE_CONTEXT
        if label == ArtifactLabel.DUPLICATE.value:
            quality = NoteQuality.DUPLICATE
            duplicate += 1
        elif label == ArtifactLabel.UNSUPPORTED.value:
            quality = NoteQuality.UNSUPPORTED
            unsupported += 1
        elif label == ArtifactLabel.MATCHED_GOLD.value:
            quality = NoteQuality.USEFUL_MEMORY
            useful += 1
        elif source is not None and source.kind.value in {"STATE", "RESULT", "FACT"} and label == ArtifactLabel.VALID_ADDITIONAL.value:
            if density < 0.28:
                quality = NoteQuality.FILLER_DERIVED
                filler += 1
            else:
                quality = NoteQuality.STATUS_UPDATE
                status_update += 1
                useful += 1
        elif density < 0.28:
            quality = NoteQuality.FILLER_DERIVED
            filler += 1
        elif label == ArtifactLabel.VALID_ADDITIONAL.value:
            if row.get("matchedGoldId") in useful_extra_ids or _looks_grounded_and_specific(row.get("meaning") or "", row.get("evidenceSequences") or [], sequence_text):
                importance = getattr(getattr(source, "memorySignal", None), "importance", None) if source else None
                if importance == "LOW":
                    quality = NoteQuality.LOW_VALUE_CONTEXT
                    low_value += 1
                else:
                    quality = NoteQuality.USEFUL_MEMORY
                    useful += 1
            else:
                quality = NoteQuality.LOW_VALUE_CONTEXT
                low_value += 1
        else:
            quality = NoteQuality.LOW_VALUE_CONTEXT
            low_value += 1
        rows.append({**row, "quality": quality.value, "informationDensity": density})
    generated = len(note_rows)
    return {
        "rows": rows,
        "useful": useful,
        "lowValueGrounded": low_value,
        "fillerDerived": filler,
        "duplicates": duplicate,
        "statusUpdate": status_update,
        "unsupported": unsupported,
        "validUsefulAdditional": max(0, useful - sum(1 for row in note_rows if row.get("label") == ArtifactLabel.MATCHED_GOLD.value)),
        "usefulnessPrecision": _ratio(useful, generated),
    }


def _gold_is_actionable(gold) -> bool:
    if gold is None:
        return False
    signal = getattr(gold, "actionSignal", None) if not isinstance(gold, dict) else gold.get("actionSignal")
    if signal is not None:
        if isinstance(signal, dict):
            return bool(signal.get("isActionable"))
        return bool(getattr(signal, "isActionable", False))
    if isinstance(gold, dict) and "actionable" in gold:
        return bool(gold.get("actionable"))
    kind = gold.kind if hasattr(gold, "kind") else gold.get("kind") if isinstance(gold, dict) else None
    kind_value = kind.value if hasattr(kind, "value") else kind
    return str(kind_value) in {item.value for item in ACTION_EVENT_KINDS} if kind_value else False


def _gold_field(gold, name: str):
    if gold is None:
        return None
    if isinstance(gold, dict):
        return gold.get(name)
    return getattr(gold, name, None)


def _gold_nested(gold, outer: str, inner: str):
    value = _gold_field(gold, outer)
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(inner)
    return getattr(value, inner, None)


def _predicted_action_object(event) -> str | None:
    if event.actionSignal and event.actionSignal.object:
        return event.actionSignal.object
    return event.object


def _field_match(predicted: str | None, gold: str | None) -> bool:
    if not gold:
        return not predicted
    if not predicted:
        return False
    if casefold_text(gold) in casefold_text(predicted) or casefold_text(predicted) in casefold_text(gold):
        return True
    return token_jaccard(predicted, gold) >= 0.5


def score_threads(events, gold_clusters: list[list[str]] | None) -> dict[str, Any]:
    if gold_clusters is None:
        return _unmeasured_thread_metrics()
    gold_ids = {event_id for cluster in gold_clusters for event_id in cluster}
    scoped = [event for event in events if event.eventId in gold_ids]
    by_id = {event.eventId: event.threadId for event in scoped}
    gold_pairs = _pairs_from_clusters(gold_clusters)
    predicted_clusters: dict[str, list[str]] = {}
    for event_id, thread_id in by_id.items():
        if not thread_id:
            continue
        predicted_clusters.setdefault(thread_id, []).append(event_id)
    predicted_pairs = _pairs_from_clusters(list(predicted_clusters.values()))
    true_pos = len(gold_pairs & predicted_pairs)
    false_merge = len(predicted_pairs - gold_pairs)
    false_split = len(gold_pairs - predicted_pairs)
    return {
        "threadPrecision": _ratio(true_pos, len(predicted_pairs)),
        "threadRecall": _ratio(true_pos, len(gold_pairs)),
        "falseMergeRate": _ratio(false_merge, max(len(predicted_pairs), 1)),
        "falseSplitRate": _ratio(false_split, max(len(gold_pairs), 1)),
        "threadMatches": true_pos,
    }


def _source_event(item, by_id: dict):
    metadata = getattr(item, "changes", None) or getattr(item, "debug", None) or {}
    ids = list(metadata.get("sourceSemanticUnitIds") or [])
    for event_id in ids:
        if event_id in by_id:
            return by_id[event_id]
    return None


def _task_is_non_action(event, item) -> bool:
    if event is None:
        return False
    kind = event.kind.value if hasattr(event.kind, "value") else event.kind
    if str(kind) == "PROPOSAL":
        return True
    strength = action_strength(event.actionSignal)
    if strength in {"NONE", "POSSIBLE"}:
        return True
    return False


def _item_action_role(item, event) -> str | None:
    if event is not None and event.actionSignal:
        return event.actionSignal.role
    metadata = getattr(item, "changes", None) or getattr(item, "debug", None) or {}
    return metadata.get("actionRole")


def _object_failure(event, expected, generated, grounding, *, accepted: bool, semantic: bool | None = None) -> dict[str, Any]:
    evidence = " ".join(span.text for span in (event.evidence or []))
    reason = "accepted" if accepted else "rejected"
    if grounding == "INFERRED":
        reason = "rejected_inferred"
    elif grounding == "UNRESOLVED":
        reason = "rejected_unresolved"
    elif expected and generated and semantic is False:
        reason = "accepted_wrong_object" if accepted else "rejected_mismatch"
    elif expected and generated and not objects_semantically_equivalent(generated, expected, evidence):
        reason = "accepted_wrong_object" if accepted else "rejected_mismatch"
    return {
        "evidence": evidence,
        "expectedObject": expected,
        "generatedObject": generated,
        "canonicalObject": getattr(getattr(event, "actionSignal", None), "canonicalActionObject", None),
        "rawObject": getattr(getattr(event, "actionSignal", None), "rawActionObject", None),
        "groundingType": grounding,
        "reason": reason,
        "accepted": accepted,
    }


def _gold_evidence_text(gold) -> str:
    spans = getattr(gold, "evidence", None) if not isinstance(gold, dict) else gold.get("evidence")
    if spans:
        return " ".join(getattr(span, "text", None) or span.get("text", "") for span in spans)
    return str(_gold_field(gold, "meaning") or "")


def _best_match(meaning: str, sequences: list[int], candidates: list[dict]) -> dict | None:
    best = None
    best_score = 0.0
    pred_seqs = set(sequences)
    for item in candidates:
        gold_meaning = str(item.get("meaning") or item.get("title") or "")
        gold_seqs = set(item.get("evidenceSequences") or [])
        overlap = len(pred_seqs & gold_seqs)
        lexical = token_jaccard(meaning, gold_meaning)
        semantic = artifacts_semantically_equivalent(
            meaning,
            gold_meaning,
            pred_seqs=list(pred_seqs),
            gold_seqs=list(gold_seqs),
        )
        score = lexical + (0.35 if overlap else 0.0)
        if overlap and lexical >= 0.18:
            score += 0.25
        if casefold_text(gold_meaning) in casefold_text(meaning) or casefold_text(meaning) in casefold_text(gold_meaning):
            score += 0.2
        if semantic:
            score += 0.45
        if score > best_score and (lexical >= 0.22 or overlap >= 1 or semantic):
            best_score = score
            best = item
    return best if best_score >= 0.28 else None


def _is_duplicate(meaning: str, previous: Iterable[str]) -> bool:
    title = meaning.split(" ", 1)[0] if meaning else ""
    for item in previous:
        if token_jaccard(meaning, item) >= 0.62:
            return True
        prev_title = item.split(" ", 1)[0] if item else ""
        if title and prev_title and token_jaccard(meaning[:40], item[:40]) >= 0.8:
            return True
    return False


def _evidence_supported(item, sequence_text: dict[int, str]) -> bool:
    spans = getattr(item, "evidence", []) or []
    if not spans:
        return False
    if not sequence_text:
        return True
    for span in spans:
        start = int(span.sequenceStart)
        end = int(span.sequenceEnd)
        combined = " ".join(sequence_text.get(seq, "") for seq in range(start, end + 1)).strip()
        if combined and (casefold_text(span.text) in casefold_text(combined) or token_jaccard(span.text, combined) >= 0.3):
            return True
    return False


def _looks_grounded_and_specific(meaning: str, sequences: list[int], sequence_text: dict[int, str]) -> bool:
    if not sequences:
        return False
    tokens = [token.casefold() for token in content_tokens(meaning)]
    if len(tokens) < 2:
        return False
    if is_structurally_generic(meaning):
        return False
    if sequence_text:
        blob = casefold_text(" ".join(sequence_text.get(seq, "") for seq in sequences))
        if not blob:
            return False
        overlap = [token for token in tokens if token in blob]
        return len(overlap) >= 1
    return True


def _label_counts(rows: list[dict[str, Any]]) -> dict[ArtifactLabel, int]:
    counts = {label: 0 for label in ArtifactLabel}
    for row in rows:
        counts[ArtifactLabel(row["label"])] += 1
    return counts


def _scripted(result: EventPipelineResult) -> bool:
    return result.observability.gemmaCalls == 0 and result.observability.gptOss120bCalls == 0


def _sequence_text_from_transcript(transcript: str) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for line in (transcript or "").splitlines():
        if line.startswith("[") and "]" in line:
            raw, _, rest = line.partition("]")
            try:
                mapping[int(raw.strip("[]"))] = rest.strip()
            except ValueError:
                continue
    return mapping


def _best_event_index(event, gold_events: list) -> int | None:
    best_i = None
    best = 0.0
    for index, gold in enumerate(gold_events):
        gold_meaning = gold.meaning if hasattr(gold, "meaning") else gold.get("meaning")
        score = token_jaccard(event.meaning, gold_meaning)
        gold_seqs = set(getattr(gold, "sequenceIds", None) or gold.get("evidenceSequences") or gold.get("sequenceIds") or [])
        if gold_seqs & set(event.sequenceIds or []):
            score += 0.35
        if score > best:
            best = score
            best_i = index
    return best_i if best >= 0.28 else None


def _event_match_key(event) -> int:
    return hash((str(event.kind), casefold_text(event.meaning)[:80], tuple(sorted(event.sequenceIds or []))))


def _pairs_from_clusters(clusters: list[list[str]]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for cluster in clusters:
        ids = [item for item in cluster if item]
        for i, left in enumerate(ids):
            for right in ids[i + 1 :]:
                pairs.add(tuple(sorted((left, right))))
    return pairs


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 1.0 if numerator <= 0 else 0.0
    return numerator / denominator


def _unmeasured_event_metrics() -> dict[str, Any]:
    return {
        "eventExpected": NOT_MEASURED,
        "eventGenerated": NOT_MEASURED,
        "eventRecall": NOT_MEASURED,
        "eventPrecision": NOT_MEASURED,
        "eventTypeAccuracy": NOT_MEASURED,
        "eventEvidenceAccuracy": NOT_MEASURED,
        "eventActorAccuracy": NOT_MEASURED,
        "eventDeadlineAccuracy": NOT_MEASURED,
        "unsupportedInferenceRate": NOT_MEASURED,
        "mergedEventRate": NOT_MEASURED,
    }


def _unmeasured_action_metrics() -> dict[str, Any]:
    return {
        "actionabilityPrecision": NOT_MEASURED,
        "actionabilityRecall": NOT_MEASURED,
        "actionVerbAccuracy": NOT_MEASURED,
        "actionVerbPrecision": NOT_MEASURED,
        "actionVerbRecall": NOT_MEASURED,
        "actionObjectAccuracy": NOT_MEASURED,
        "actionObjectPrecision": NOT_MEASURED,
        "actionObjectRecall": NOT_MEASURED,
        "explicitObjectAccuracy": NOT_MEASURED,
        "coreferenceObjectAccuracy": NOT_MEASURED,
        "inferredObjectRejectionRate": NOT_MEASURED,
        "actionObjectGroundingPrecision": NOT_MEASURED,
        "actionObjectGroundingRecall": NOT_MEASURED,
        "actionActorAccuracy": NOT_MEASURED,
        "actionDeadlineAccuracy": NOT_MEASURED,
        "genericActionRate": NOT_MEASURED,
        "unsupportedActionRate": NOT_MEASURED,
        "groundedActionObjects": NOT_MEASURED,
        "actionObjectFailures": [],
        "strictOriginalGoldPrecision": NOT_MEASURED,
        "reviewedActionPrecision": NOT_MEASURED,
        "reviewedActionRecall": NOT_MEASURED,
        "validAdditionalActionCount": NOT_MEASURED,
        "actualFalsePositiveCount": NOT_MEASURED,
        "objectSemanticAccuracy": NOT_MEASURED,
        "objectSurfaceNormalizationAccuracy": NOT_MEASURED,
        "objectGroundingAccuracy": NOT_MEASURED,
        "actionClassifications": [],
    }


def _unmeasured_thread_metrics() -> dict[str, Any]:
    return {
        "threadPrecision": NOT_MEASURED,
        "threadRecall": NOT_MEASURED,
        "falseMergeRate": NOT_MEASURED,
        "falseSplitRate": NOT_MEASURED,
        "threadMatches": NOT_MEASURED,
    }


def gold_review_status(item: Any) -> GoldReviewStatus:
    if item is None:
        return GoldReviewStatus.REQUIRED
    raw = ""
    if isinstance(item, dict):
        raw = str(item.get("reviewStatus") or item.get("review_status") or "")
    else:
        raw = str(getattr(item, "reviewStatus", None) or "")
    value = raw.strip().upper()
    try:
        return GoldReviewStatus(value) if value else GoldReviewStatus.REQUIRED
    except ValueError:
        return GoldReviewStatus.REQUIRED


def artifacts_semantically_equivalent(
    predicted_meaning: str,
    gold_meaning: str,
    *,
    pred_seqs: list[int] | None = None,
    gold_seqs: list[int] | None = None,
    pred_object: str | None = None,
    gold_object: str | None = None,
    pred_verb: str | None = None,
    gold_verb: str | None = None,
) -> bool:
    """Benchmark-only: same grounded action/memory, not title equality."""
    if not gold_meaning or not predicted_meaning:
        return False
    if objects_semantically_equivalent(predicted_meaning, gold_meaning):
        if _verbs_compatible(pred_verb, gold_verb, predicted_meaning, gold_meaning) or _constraint_compatible(
            predicted_meaning, gold_meaning
        ):
            return True
        if token_jaccard(predicted_meaning, gold_meaning) >= 0.28:
            return True
        if _core_object_overlap(predicted_meaning, gold_meaning, pred_object, gold_object):
            return True
    if pred_object and gold_object and objects_semantically_equivalent(pred_object, gold_object):
        if _verbs_compatible(pred_verb, gold_verb, predicted_meaning, gold_meaning):
            return True
    lexical = token_jaccard(predicted_meaning, gold_meaning)
    overlap = bool(set(pred_seqs or []) & set(gold_seqs or []))
    if lexical >= 0.45 and _verbs_compatible(pred_verb, gold_verb, predicted_meaning, gold_meaning):
        return True
    if overlap and lexical >= 0.22 and _core_object_overlap(predicted_meaning, gold_meaning, pred_object, gold_object):
        return True
    if _core_object_overlap(predicted_meaning, gold_meaning, pred_object, gold_object) and _verbs_compatible(
        pred_verb, gold_verb, predicted_meaning, gold_meaning
    ):
        return True
    return False


def _review_metrics(
    gold_tasks: list[dict],
    gold_notes: list[dict],
    task_rows: list[dict[str, Any]],
    note_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    matched_task_ids = {row.get("matchedGoldId") for row in task_rows if row.get("label") == ArtifactLabel.MATCHED_GOLD.value}
    matched_note_ids = {row.get("matchedGoldId") for row in note_rows if row.get("label") == ArtifactLabel.MATCHED_GOLD.value}
    matched_ids = {item for item in matched_task_ids | matched_note_ids if item}
    required_tasks = [item for item in gold_tasks if gold_review_status(item) == GoldReviewStatus.REQUIRED]
    required_notes = [item for item in gold_notes if gold_review_status(item) == GoldReviewStatus.REQUIRED]
    optional = [item for item in [*gold_tasks, *gold_notes] if gold_review_status(item) == GoldReviewStatus.OPTIONAL_VALID]
    low_value = [item for item in [*gold_tasks, *gold_notes] if gold_review_status(item) == GoldReviewStatus.LOW_VALUE]
    invalid = [item for item in [*gold_tasks, *gold_notes] if gold_review_status(item) == GoldReviewStatus.INVALID_GOLD]
    required_task_matched = sum(1 for item in required_tasks if item.get("id") in matched_ids)
    required_note_matched = sum(1 for item in required_notes if item.get("id") in matched_ids)
    return {
        "requiredTaskCount": len(required_tasks),
        "requiredNoteCount": len(required_notes),
        "requiredTaskRecall": _ratio(required_task_matched, len(required_tasks)),
        "requiredNoteRecall": _ratio(required_note_matched, len(required_notes)),
        "missingRequiredTasks": max(0, len(required_tasks) - required_task_matched),
        "missingRequiredNotes": max(0, len(required_notes) - required_note_matched),
        "optionalValidFound": sum(1 for item in optional if item.get("id") in matched_ids),
        "lowValueSuppressed": sum(1 for item in low_value if item.get("id") not in matched_ids),
        "invalidGoldCount": len(invalid),
    }


def build_gold_traces(
    result: EventPipelineResult,
    gold_tasks: list[dict],
    gold_notes: list[dict],
    task_rows: list[dict[str, Any]] | None = None,
    note_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    task_rows = task_rows or []
    note_rows = note_rows or []
    matched_task_ids = {row.get("matchedGoldId") for row in task_rows if row.get("label") == ArtifactLabel.MATCHED_GOLD.value}
    matched_note_ids = {row.get("matchedGoldId") for row in note_rows if row.get("label") == ArtifactLabel.MATCHED_GOLD.value}
    traces = []
    for gold in gold_tasks:
        traces.append(
            _trace_gold_item(gold, "task", result, task_rows, matched_task_ids)
        )
    for gold in gold_notes:
        traces.append(
            _trace_gold_item(gold, "note", result, note_rows, matched_note_ids)
        )
    return traces


def _trace_gold_item(
    gold: dict,
    kind: str,
    result: EventPipelineResult,
    rows: list[dict[str, Any]],
    matched_ids: set,
) -> dict[str, Any]:
    sequences = list(gold.get("evidenceSequences") or [])
    sequence_text = {span.sequenceStart: span.text for event in result.events for span in (event.evidence or [])}
    for block in result.microBlocks:
        for seq, text in zip(block.sequenceIds, (block.text or "").splitlines() or [block.text]):
            sequence_text.setdefault(seq, (block.text or "")[:180])
    events = [event for event in result.events if set(sequences) & set(event.sequenceIds or [])]
    blocks = [block for block in result.microBlocks if set(sequences) & set(block.sequenceIds or [])]
    topics = [
        topic
        for topic in result.topics
        if set(sequences) & set(topic.sequenceIds or []) or any(block.microBlockId in set(topic.microBlockIds or []) for block in blocks)
    ]
    threads = {event.threadId for event in events if event.threadId}
    matched_row = next((row for row in rows if row.get("matchedGoldId") == gold.get("id")), None)
    source = events[0] if events else None
    signal = getattr(source, "actionSignal", None) if source else None
    memory = getattr(source, "memorySignal", None) if source else None
    generated = result.tasks if kind == "task" else result.notes
    nearby = [
        item
        for item in generated
        if set(evidence_sequence_ids(getattr(item, "evidence", []))) & set(sequences)
        or artifacts_semantically_equivalent(
            f"{getattr(item, 'title', '')} {getattr(item, 'body', '')}",
            str(gold.get("meaning") or ""),
        )
    ]
    failure = None
    note_review = None
    if gold.get("id") not in matched_ids:
        if kind == "task":
            failure = _classify_task_failure(gold, result, events, nearby, rows)
        else:
            note_review = _classify_note_review(gold, result, events, nearby, rows)
            if note_review == NoteReviewClass.EXTRACTION_MISS:
                failure = GoldFailureClass.EXTRACTION_MISS.value
    return {
        "goldId": gold.get("id"),
        "kind": kind,
        "reviewStatus": gold_review_status(gold).value,
        "goldMeaning": gold.get("meaning"),
        "sourceTranscript": [sequence_text.get(seq) or f"seq {seq}" for seq in sequences],
        "microBlocks": [block.microBlockId for block in blocks],
        "topics": [topic.topicId for topic in topics],
        "events": [
            {
                "eventId": event.eventId,
                "kind": event.kind.value if hasattr(event.kind, "value") else event.kind,
                "meaning": event.meaning,
                "disposition": event.disposition.value if event.disposition else None,
                "dispositionReason": event.dispositionReason,
                "channel": event.channel,
            }
            for event in events
        ],
        "actionSignal": {
            "isActionable": getattr(signal, "isActionable", None) if signal else None,
            "role": getattr(signal, "role", None) if signal else None,
            "actionStrength": getattr(signal, "actionStrength", None) if signal else None,
            "rawVerb": getattr(signal, "verb", None) if signal else None,
            "rawObject": getattr(signal, "rawActionObject", None) or getattr(signal, "object", None) if signal else None,
            "canonicalVerb": getattr(signal, "verb", None) if signal else None,
            "canonicalObject": getattr(signal, "canonicalActionObject", None) or getattr(signal, "object", None) if signal else (source.object if source else None),
            "objectGroundingType": object_grounding_type(source) if source else None,
        },
        "memorySignal": {
            "isMemoryWorthy": getattr(memory, "isMemoryWorthy", None) if memory else None,
            "importance": getattr(memory, "importance", None) if memory else None,
        },
        "threads": sorted(threads),
        "taskCandidate": [{"title": getattr(item, "title", ""), "sequences": evidence_sequence_ids(getattr(item, "evidence", []))} for item in nearby] if kind == "task" else [],
        "finalArtifact": {"title": matched_row.get("title"), "meaning": matched_row.get("meaning"), "label": matched_row.get("label")} if matched_row else None,
        "benchmarkMatch": bool(gold.get("id") in matched_ids),
        "failureClass": failure.value if isinstance(failure, GoldFailureClass) else failure,
        "noteReviewClass": note_review.value if isinstance(note_review, NoteReviewClass) else note_review,
    }


def _classify_task_failure(
    gold: dict,
    result: EventPipelineResult,
    events: list,
    nearby: list,
    rows: list[dict[str, Any]],
) -> GoldFailureClass:
    gold_meaning = str(gold.get("meaning") or "")
    unmatched_generated = [
        row
        for row in rows
        if row.get("label") != ArtifactLabel.MATCHED_GOLD.value
        and artifacts_semantically_equivalent(row.get("meaning") or "", gold_meaning)
    ]
    if unmatched_generated or any(
        artifacts_semantically_equivalent(f"{getattr(item, 'title', '')} {getattr(item, 'body', '')}", gold_meaning)
        for item in result.tasks
    ):
        if not any(row.get("matchedGoldId") == gold.get("id") for row in rows):
            if unmatched_generated or nearby:
                return GoldFailureClass.SCORER_SEMANTIC_MISMATCH
    if not events:
        return GoldFailureClass.EXTRACTION_MISS
    actionable = [event for event in events if event_is_actionable(event) or event_is_task_eligible(event)]
    if not actionable:
        return GoldFailureClass.ACTIONABILITY_MISS
    if actionable and all(not action_object_grounded(event) for event in actionable):
        return GoldFailureClass.OBJECT_GROUNDING_MISS
    if any(event.disposition == EventDisposition.REJECTED for event in actionable):
        return GoldFailureClass.VALIDATION_REJECT
    if any(event.disposition == EventDisposition.DUPLICATE for event in actionable):
        return GoldFailureClass.DEDUPE_ERROR
    mixed = mixed_thread_rate(nearby or result.tasks, result.events)
    if mixed and nearby:
        return GoldFailureClass.THREADING_MISS
    if any(
        event.disposition == EventDisposition.INTENTIONALLY_NON_PUBLISHABLE
        or str(event.dispositionReason or "").startswith("generic")
        for event in actionable
    ):
        return GoldFailureClass.SYNTHESIS_MISS
    if not nearby:
        return GoldFailureClass.SYNTHESIS_MISS
    return GoldFailureClass.SCORER_SEMANTIC_MISMATCH


def _classify_note_review(
    gold: dict,
    result: EventPipelineResult,
    events: list,
    nearby: list,
    rows: list[dict[str, Any]],
) -> NoteReviewClass:
    gold_meaning = str(gold.get("meaning") or "")
    if any(
        artifacts_semantically_equivalent(f"{getattr(item, 'title', '')} {getattr(item, 'body', '')}", gold_meaning)
        for item in result.notes
    ):
        return NoteReviewClass.DUPLICATE_CONTEXT
    supporting_tasks = [
        task
        for task in result.tasks
        if artifacts_semantically_equivalent(f"{task.title} {task.body}", gold_meaning)
        or set(evidence_sequence_ids(task.evidence)) & set(gold.get("evidenceSequences") or [])
    ]
    if supporting_tasks:
        return NoteReviewClass.TASK_SUPPORTING_CONTEXT
    if not events:
        return NoteReviewClass.EXTRACTION_MISS
    memory_events = [
        event
        for event in events
        if event.memorySignal is not None or str(event.channel) == "memory"
    ]
    if memory_events and all(
        (not getattr(event.memorySignal, "isMemoryWorthy", False) if event.memorySignal else True)
        or str(getattr(event.memorySignal, "importance", "")).upper() == "LOW"
        or event.disposition == EventDisposition.INTENTIONALLY_NON_PUBLISHABLE
        for event in memory_events
    ):
        return NoteReviewClass.LOW_VALUE_CONTEXT
    if nearby:
        return NoteReviewClass.USEFUL_MEMORY
    if memory_events:
        return NoteReviewClass.LOW_VALUE_CONTEXT
    return NoteReviewClass.EXTRACTION_MISS


def _gold_by_id(items: list[dict], gold_id: str | None) -> dict | None:
    if not gold_id:
        return None
    return next((item for item in items if item.get("id") == gold_id), None)


def _verbs_compatible(pred_verb: str | None, gold_verb: str | None, pred_meaning: str, gold_meaning: str) -> bool:
    pred = casefold_text(pred_verb or "") or _leading_verb(pred_meaning)
    gold = casefold_text(gold_verb or "") or _leading_verb(gold_meaning)
    if not pred or not gold:
        return True
    if pred == gold or pred in gold or gold in pred:
        return True
    pred_family = _verb_family(pred)
    gold_family = _verb_family(gold)
    if pred_family and gold_family and pred_family == gold_family:
        return True
    if _has_negation(gold_meaning) and pred in _BLOCKING_VERBS:
        return True
    if _has_negation(pred_meaning) and gold in _BLOCKING_VERBS:
        return True
    return False


def _verb_family(verb: str) -> frozenset[str] | None:
    key = casefold_text(verb)
    for family in _VERB_FAMILIES:
        if key in family or any(key.startswith(item) or item.startswith(key) for item in family if len(item) >= 4):
            return family
    return None


def _leading_verb(meaning: str) -> str:
    tokens = [token.casefold() for token in content_tokens(meaning)]
    return tokens[0] if tokens else ""


def _has_negation(text: str) -> bool:
    tokens = {token.casefold() for token in content_tokens(text)} | set(casefold_text(text).split())
    return bool(tokens & _NEGATION) or "n't" in casefold_text(text)


def _constraint_compatible(predicted_meaning: str, gold_meaning: str) -> bool:
    gold_nums = {token for token in tokenize(gold_meaning) if token.isdigit()}
    pred_nums = {token for token in tokenize(predicted_meaning) if token.isdigit()}
    if gold_nums and pred_nums and gold_nums & pred_nums:
        return True
    if not gold_nums:
        return True
    return bool(gold_nums & pred_nums)


def _core_object_overlap(
    predicted_meaning: str,
    gold_meaning: str,
    pred_object: str | None,
    gold_object: str | None,
) -> bool:
    left = pred_object or predicted_meaning
    right = gold_object or gold_meaning
    return objects_semantically_equivalent(left, right)


def e2e_scale_report(result: EventPipelineResult, *, gold: dict | None = None, case_id: str = "scale") -> dict[str, Any]:
    gold = gold or {}
    gold_complete = bool(gold.get("goldComplete", bool(gold.get("goldTasks") and gold.get("goldNotes"))))
    if gold_complete:
        return pipeline_benchmark(
            result,
            gold.get("goldTasks") or [],
            gold.get("goldNotes") or [],
            case_id=case_id,
            transcript=gold.get("transcript") or "",
            valid_additional_notes=gold.get("validAdditionalNotes"),
            valid_additional_tasks=gold.get("validAdditionalTasks"),
            gold_events=gold.get("events"),
            gold_threads=gold.get("goldThreads"),
            gold_complete=True,
            original_actionable_ids=gold.get("originalActionableEventIds"),
            reviewed_actionable_ids=gold.get("reviewedActionableEventIds"),
        )
    generic = sum(1 for task in result.tasks if is_generic_task_text(task.title, task.body))
    mixed = mixed_thread_rate([*result.tasks, *result.notes], result.events)
    return {
        "goldComplete": False,
        "taskPrecision": NOT_MEASURED,
        "taskRecall": NOT_MEASURED,
        "notePrecision": NOT_MEASURED,
        "noteRecall": NOT_MEASURED,
        "eventRecall": NOT_MEASURED,
        "evidencePrecision": NOT_MEASURED,
        "threadPrecision": NOT_MEASURED,
        "threadRecall": NOT_MEASURED,
        "genericTaskRate": generic / max(len(result.tasks), 1) if result.tasks else 0.0,
        "mixedThreadRate": mixed,
        "unaccountedBlocks": result.coverage.unaccounted_blocks if result.coverage else 0,
        "unaccountedSemanticUnits": result.coverage.unaccountedSemanticUnits if result.coverage else 0,
        "semanticCoverage": result.coverage.semanticCoverage if result.coverage else 1.0,
        "semanticCoverageFailure": bool(result.coverage.semanticCoverageFailure) if result.coverage else False,
        "counts": {
            "rawChunks": result.cleaning.totalSequences if result.cleaning else None,
            "usefulChunks": result.cleaning.usefulSequences if result.cleaning else None,
            "microBlocks": len(result.microBlocks),
            "topics": len(result.topics),
            "events": len(result.events),
            "threads": len(result.threads),
            "actionEvents": result.coverage.action_events if result.coverage else sum(1 for event in result.events if event.channel == "action"),
            "memoryEvents": result.coverage.memory_events if result.coverage else sum(1 for event in result.events if event.channel == "memory"),
            "tasks": len(result.tasks),
            "notes": len(result.notes),
            "rejected": result.coverage.rejected_events if result.coverage else 0,
            "unaccounted": result.coverage.unaccounted_blocks if result.coverage else 0,
        },
        "observability": {
            "llmCalls": result.observability.llm_calls(),
            "embeddingCalls": result.observability.embedding_calls(),
            "gemmaCalls": result.observability.gemmaCalls,
            "gptOss120bCalls": result.observability.gptOss120bCalls,
            "gptOss20bCalls": result.observability.gptOss20bCalls,
            "stageMs": {stage.name: stage.durationMs for stage in result.observability.stages},
            "estimatedCostUsd": result.observability.estimatedCostUsd
            if result.observability.estimatedCostUsd is not None
            else NOT_MEASURED,
        },
    }
