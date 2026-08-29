"""Atomic grounded event extraction. One meaning per event. Never invents evidence IDs."""

from __future__ import annotations

import json
from typing import Any, Protocol

from pydantic import BaseModel, Field

from services.conversation.event_pipeline.schemas import (
    ABSTAIN_UNRESOLVED_OBJECT,
    ACTION_EVENT_KINDS,
    ACTION_PRONOUNS,
    ACTION_ROLES,
    ACTION_STRENGTHS,
    DEICTIC_OR_TIME,
    GENERIC_ACTION_OBJECTS,
    MEMORY_EVENT_KINDS,
    OBJECT_GROUNDING_TYPES,
    ActionSignal,
    AtomicEvent,
    EventKind,
    FieldEvidence,
    LocalTopic,
    MemorySignal,
    MicroBlock,
)
from services.conversation.event_pipeline.textutil import (
    casefold_text,
    content_tokens,
    evidence_sequence_ids,
    extract_entities,
    normalize_text,
    stable_id,
)
from services.conversation.event_pipeline.routing import PipelineStage
from services.conversation.models import EvidenceSpan
from services.llm.router import LLMCapability, LLMRouter


class EventExtractor(Protocol):
    async def extract(self, topic: LocalTopic, blocks: list[MicroBlock], sequence_text: dict[int, str]) -> list[AtomicEvent]:
        ...

    async def extract_missing(
        self,
        topic: LocalTopic,
        blocks: list[MicroBlock],
        sequence_text: dict[int, str],
        missing_units: list[Any],
        existing: list[AtomicEvent] | None = None,
    ) -> list[AtomicEvent]:
        ...


class ActionSignalLLMItem(BaseModel):
    isActionable: bool = False
    role: str | None = None
    actionStrength: str | None = None
    verb: str | None = None
    object: str | None = None
    objectGroundingType: str | None = None
    actor: str | None = None
    deadline: str | None = None


class MemorySignalLLMItem(BaseModel):
    isMemoryWorthy: bool = False
    importance: str | None = None
    reason: str | None = None


class FieldEvidenceLLMItem(BaseModel):
    actionVerb: list[EvidenceSpan] = Field(default_factory=list)
    actionObject: list[EvidenceSpan] = Field(default_factory=list)
    actor: list[EvidenceSpan] = Field(default_factory=list)
    deadline: list[EvidenceSpan] = Field(default_factory=list)


class AtomicEventLLMItem(BaseModel):
    kind: EventKind = EventKind.FACT
    meaning: str
    actor: str | None = None
    object: str | None = None
    timeExpression: str | None = None
    entities: list[str] = Field(default_factory=list)
    uncertainty: list[str] = Field(default_factory=list)
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    abstain: bool = False
    abstainReason: str = ""
    actionSignal: ActionSignalLLMItem | None = None
    memorySignal: MemorySignalLLMItem | None = None
    fieldEvidence: FieldEvidenceLLMItem | None = None


class AtomicEventLLMResponse(BaseModel):
    events: list[AtomicEventLLMItem] = Field(default_factory=list)
    noEventReason: str | None = None


class LLMEventExtractor:
    def __init__(
        self,
        router: LLMRouter,
        capability: LLMCapability = LLMCapability.SEMANTIC_EXTRACTION,
        stage: PipelineStage = PipelineStage.ATOMIC_EVENTS,
    ):
        self.router = router
        self.capability = capability
        self.stage = stage
        self.calls = 0
        self.failures = 0
        self.last_provider = "none"
        self.last_model = "none"
        self.requested_capabilities: list[LLMCapability] = []

    async def extract(self, topic: LocalTopic, blocks: list[MicroBlock], sequence_text: dict[int, str]) -> list[AtomicEvent]:
        from services.conversation.event_pipeline.llm import generate_structured_for_stage
        from services.conversation.event_pipeline.routing import capability_for_stage
        from services.conversation.event_pipeline.topics import _is_filler_block

        self.calls += 1
        self.requested_capabilities.append(self.capability or capability_for_stage(self.stage))
        content_blocks = [block for block in blocks if not _is_filler_block(block)]
        filler_ratio = 1.0 - (len(content_blocks) / max(len(blocks), 1))
        extract_blocks = content_blocks if content_blocks and filler_ratio >= 0.5 else blocks
        transcript = "\n".join(block.text for block in extract_blocks) if extract_blocks is not blocks else topic.text
        payload = {
            "topicId": topic.topicId,
            "topicLabel": topic.label,
            "entities": topic.entities,
            "sequenceStart": topic.sequenceStart,
            "sequenceEnd": topic.sequenceEnd,
            "transcript": transcript,
            "fullTopicTranscript": topic.text,
            "microBlocks": [
                {
                    "microBlockId": block.microBlockId,
                    "sequenceIds": list(block.sequenceIds),
                    "text": block.text,
                    "filler": _is_filler_block(block),
                }
                for block in blocks
            ],
            "contentMicroBlocks": [
                {
                    "microBlockId": block.microBlockId,
                    "sequenceIds": list(block.sequenceIds),
                    "text": block.text,
                }
                for block in content_blocks
            ],
        }
        try:
            response, provider, model = await generate_structured_for_stage(
                self.router,
                self.stage,
                "atomic-event-extractor-v1",
                AtomicEventLLMResponse,
                payload,
                background=json.dumps({"instruction": "Extract atomic grounded events only."}, ensure_ascii=True),
            )
            self.last_provider = getattr(provider, "name", None) or getattr(provider, "last_successful_provider", None) or self.last_provider
            self.last_model = model
        except Exception as error:
            from services.llm.async_runtime import reraise_if_hard_runtime
            from services.conversation.event_pipeline.observability import record_failure

            record_failure(error)
            reraise_if_hard_runtime(error)
            self.failures += 1
            return []
        return materialize_events(response, topic, blocks, sequence_text)

    async def extract_missing(
        self,
        topic: LocalTopic,
        blocks: list[MicroBlock],
        sequence_text: dict[int, str],
        missing_units: list[Any],
        existing: list[AtomicEvent] | None = None,
    ) -> list[AtomicEvent]:
        from services.conversation.event_pipeline.llm import generate_structured_for_stage
        from services.conversation.event_pipeline.routing import capability_for_stage
        from services.conversation.event_pipeline.topics import _is_filler_block

        if not missing_units:
            return []
        self.calls += 1
        self.requested_capabilities.append(self.capability or capability_for_stage(self.stage))
        content_blocks = [block for block in blocks if not _is_filler_block(block)]
        payload = {
            "topicId": topic.topicId,
            "topicLabel": topic.label,
            "transcript": "\n".join(block.text for block in content_blocks or blocks),
            "microBlocks": [
                {
                    "microBlockId": block.microBlockId,
                    "sequenceIds": list(block.sequenceIds),
                    "text": block.text,
                }
                for block in (content_blocks or blocks)
            ],
            "existingEvents": [
                {
                    "eventId": event.eventId,
                    "kind": event.kind.value if hasattr(event.kind, "value") else str(event.kind),
                    "meaning": event.meaning,
                    "sequenceIds": list(event.sequenceIds or evidence_sequence_ids(event.evidence)),
                }
                for event in (existing or [])
                if event.kind != EventKind.NOISE
            ],
            "missingSemanticUnits": [
                unit.model_dump() if hasattr(unit, "model_dump") else dict(unit)
                for unit in missing_units
            ],
        }
        try:
            response, provider, model = await generate_structured_for_stage(
                self.router,
                PipelineStage.COVERAGE_REPAIR_EVENTS,
                "semantic-completeness-repair-v1",
                AtomicEventLLMResponse,
                payload,
                background=json.dumps(
                    {"instruction": "Extract only the listed missing semantic units. Do not invent."},
                    ensure_ascii=True,
                ),
            )
            self.last_provider = getattr(provider, "name", None) or getattr(provider, "last_successful_provider", None) or self.last_provider
            self.last_model = model
        except Exception as error:
            from services.llm.async_runtime import reraise_if_hard_runtime
            from services.conversation.event_pipeline.observability import record_failure

            record_failure(error)
            reraise_if_hard_runtime(error)
            self.failures += 1
            return []
        return materialize_events(response, topic, blocks, sequence_text)


class ScriptedEventExtractor:
    """Test double: maps topic text/sequences onto preloaded atomic events."""

    def __init__(
        self,
        events_by_topic: dict[str, list[AtomicEvent]] | None = None,
        events: list[AtomicEvent] | None = None,
        repair_events: list[AtomicEvent] | None = None,
    ):
        self.events_by_topic = events_by_topic or {}
        self.events = events or []
        self.repair_events = repair_events or []

    async def extract(self, topic: LocalTopic, blocks: list[MicroBlock], sequence_text: dict[int, str]) -> list[AtomicEvent]:
        if topic.topicId in self.events_by_topic:
            return [event.model_copy(deep=True) for event in self.events_by_topic[topic.topicId]]
        topic_sequences = set(topic.sequenceIds)
        matched = [
            event.model_copy(deep=True)
            for event in self.events
            if topic_sequences & set(event.sequenceIds or evidence_sequence_ids(event.evidence))
        ]
        for event in matched:
            event.topicId = topic.topicId
        return matched

    async def extract_missing(
        self,
        topic: LocalTopic,
        blocks: list[MicroBlock],
        sequence_text: dict[int, str],
        missing_units: list[Any],
        existing: list[AtomicEvent] | None = None,
    ) -> list[AtomicEvent]:
        if not missing_units:
            return []
        existing_ids = {event.eventId for event in (existing or [])}
        recovered: list[AtomicEvent] = []
        for event in self.repair_events:
            if event.eventId in existing_ids:
                continue
            copy = event.model_copy(deep=True)
            copy.topicId = topic.topicId
            recovered.append(copy)
        return recovered


def materialize_events(
    response: AtomicEventLLMResponse | dict[str, Any],
    topic: LocalTopic,
    blocks: list[MicroBlock],
    sequence_text: dict[int, str],
    conversation_id: str = "",
    user_id: str = "",
    space_id: str = "",
) -> list[AtomicEvent]:
    if isinstance(response, dict):
        response = AtomicEventLLMResponse.model_validate(response)
    events: list[AtomicEvent] = []
    block_by_sequence = _block_index(blocks)
    for item in response.events:
        if item.abstain or not normalize_text(item.meaning):
            continue
        evidence = _ground_evidence(item.evidence, sequence_text, topic.sequenceIds)
        if not evidence and item.fieldEvidence is not None:
            extra = [
                *(item.fieldEvidence.actionVerb or []),
                *(item.fieldEvidence.actionObject or []),
                *(item.fieldEvidence.actor or []),
                *(item.fieldEvidence.deadline or []),
            ]
            evidence = _ground_evidence(extra, sequence_text, topic.sequenceIds)
        if not evidence:
            evidence = _evidence_from_meaning(item.meaning, item.actionSignal, topic, sequence_text)
        if not evidence:
            continue
        sequence_ids = evidence_sequence_ids(evidence)
        source_ids = []
        micro_ids = []
        for sequence in sequence_ids:
            block = block_by_sequence.get(sequence)
            if block:
                if block.microBlockId not in micro_ids:
                    micro_ids.append(block.microBlockId)
                for source, seq in zip(block.sourceIds, block.sequenceIds):
                    if seq == sequence and source not in source_ids:
                        source_ids.append(source)
        field_evidence = _materialize_field_evidence(item.fieldEvidence, sequence_text, topic.sequenceIds)
        actor = _ground_field(
            (item.actionSignal.actor if item.actionSignal else None) or item.actor,
            field_evidence.actor,
            evidence,
            sequence_text,
            topic.sequenceIds,
            paraphrase_ok=False,
        )
        time_expression = _ground_field(
            (item.actionSignal.deadline if item.actionSignal else None) or item.timeExpression,
            field_evidence.deadline,
            evidence,
            sequence_text,
            topic.sequenceIds,
            paraphrase_ok=False,
        )
        verb = _ground_verb(
            item.actionSignal.verb if item.actionSignal else None,
            field_evidence.actionVerb,
            evidence,
            sequence_text,
            topic.sequenceIds,
        )
        requested_object = (item.actionSignal.object if item.actionSignal else None) or item.object
        object_text, object_spans, object_status, grounding_type = _ground_action_object(
            requested_object,
            field_evidence.actionObject,
            evidence,
            topic,
            blocks,
            sequence_text,
            sequence_ids,
            actionable=_requested_explicit_action(item),
            claimed_grounding=item.actionSignal.objectGroundingType if item.actionSignal else None,
        )
        if object_spans:
            field_evidence.actionObject = _merge_spans(field_evidence.actionObject, object_spans)
            evidence = _merge_spans(evidence, object_spans)
            for span in object_spans:
                for sequence in range(int(span.sequenceStart), int(span.sequenceEnd) + 1):
                    if sequence not in sequence_ids:
                        sequence_ids.append(sequence)
                    block = block_by_sequence.get(sequence)
                    if block and block.microBlockId not in micro_ids:
                        micro_ids.append(block.microBlockId)
        raw_object = object_text
        canonical_object = object_text
        if object_text:
            from services.conversation.event_pipeline.object_canon import canonicalize_action_object

            evidence_blob = " ".join(span.text for span in evidence) + " " + " ".join(
                sequence_text.get(sequence, "") for sequence in sequence_ids
            )
            canonical_object = canonicalize_action_object(object_text, evidence_blob) or object_text
            object_text = canonical_object
        kind = item.kind if isinstance(item.kind, EventKind) else EventKind(str(item.kind))
        uncertainty = list(item.uncertainty or [])
        action_signal = _materialize_action_signal(
            item,
            verb,
            object_text,
            actor,
            time_expression,
            object_status,
            grounding_type,
            kind,
            raw_object=raw_object,
            canonical_object=canonical_object,
        )
        if (action_signal and action_signal.isActionable and not object_text) or (
            action_signal is None and kind in ACTION_EVENT_KINDS and not object_text
        ):
            if "missing_object" not in uncertainty:
                uncertainty.append("missing_object")
            if action_signal is not None:
                action_signal.artifactStatus = action_signal.artifactStatus or ABSTAIN_UNRESOLVED_OBJECT
                action_signal.objectGroundingType = action_signal.objectGroundingType or "UNRESOLVED"
        memory_signal = _materialize_memory_signal(item, kind, action_signal, evidence)
        for flag in _source_quality_flags(evidence, item.meaning):
            if flag not in uncertainty:
                uncertainty.append(flag)
        if "LOW_CONFIDENCE_SOURCE" in uncertainty and memory_signal and memory_signal.importance == "HIGH":
            memory_signal.importance = "MEDIUM"
        event = AtomicEvent(
            eventId=stable_id("E", conversation_id or topic.topicId, kind.value, item.meaning, sequence_ids),
            topicId=topic.topicId,
            kind=kind,
            meaning=normalize_text(item.meaning),
            actor=actor,
            object=object_text,
            timeExpression=time_expression,
            entities=item.entities or extract_entities(item.meaning + " " + " ".join(span.text for span in evidence)),
            uncertainty=uncertainty,
            evidence=evidence,
            microBlockIds=micro_ids,
            sourceIds=source_ids,
            sequenceIds=sequence_ids,
            conversationId=conversation_id,
            userId=user_id,
            spaceId=space_id,
            actionSignal=action_signal,
            memorySignal=memory_signal,
            fieldEvidence=field_evidence,
        )
        events.append(event)
    return events


def events_to_semantic_units(events: list[AtomicEvent]):
    from services.conversation.models import SemanticUnit

    units = []
    for event in events:
        actionable = bool(event.actionSignal and event.actionSignal.isActionable) or event.kind in ACTION_EVENT_KINDS
        memory_worthy = bool(event.memorySignal and event.memorySignal.isMemoryWorthy) or event.kind in MEMORY_EVENT_KINDS
        kind = "action" if actionable else "fact" if memory_worthy else "noise"
        units.append(
            SemanticUnit(
                semanticKey=event.eventId,
                kind=kind,
                meaning=event.meaning,
                ownerText=event.actor,
                dueDateText=event.timeExpression,
                evidence=event.evidence,
                evidenceIds=event.sequenceIds,
                quality={"grounded": True, "independentlyUseful": event.kind != EventKind.NOISE},
            )
        )
    return units


def _materialize_action_signal(
    item: AtomicEventLLMItem,
    verb: str | None,
    object_text: str | None,
    actor: str | None,
    deadline: str | None,
    object_status: str | None,
    grounding_type: str | None,
    kind: EventKind,
    raw_object: str | None = None,
    canonical_object: str | None = None,
) -> ActionSignal | None:
    raw = item.actionSignal
    if raw is None:
        return None
    role = _coerce_role(raw.role)
    strength = _coerce_strength(raw.actionStrength, raw.isActionable, role, kind)
    is_actionable = bool(raw.isActionable)
    if kind == EventKind.PROPOSAL and strength != "EXPLICIT":
        strength = strength if strength in {"NONE", "POSSIBLE"} else "POSSIBLE"
        is_actionable = False
        role = None
    if strength in {"NONE", "POSSIBLE"}:
        is_actionable = False
    elif strength == "EXPLICIT":
        # Kind is not the Task classifier. An EXPLICIT action on FACT/DECISION/STATE
        # stays actionable; infer role from kind when the model omitted it.
        if role is None:
            role = _role_from_kind(kind) if is_actionable or kind in ACTION_EVENT_KINDS or kind == EventKind.REQUIREMENT else None
        is_actionable = True if is_actionable or role in ACTION_ROLES else False
        if not is_actionable:
            strength = "POSSIBLE"
    status = object_status
    grounding = grounding_type or _coerce_grounding(raw.objectGroundingType)
    if grounding == "INFERRED":
        object_text = None
        raw_object = None
        canonical_object = None
        status = ABSTAIN_UNRESOLVED_OBJECT
        grounding = "INFERRED"
    elif is_actionable and not object_text:
        status = ABSTAIN_UNRESOLVED_OBJECT
        grounding = grounding or "UNRESOLVED"
    rendered = canonical_object or object_text
    return ActionSignal(
        isActionable=is_actionable,
        role=role if is_actionable else (role if strength == "POSSIBLE" else None),
        actionStrength=strength,
        verb=verb if is_actionable or strength == "POSSIBLE" else None,
        object=rendered,
        rawActionObject=raw_object or object_text,
        canonicalActionObject=rendered,
        objectGroundingType=grounding,
        actor=actor,
        deadline=deadline,
        artifactStatus=status,
    )


def _requested_explicit_action(item: AtomicEventLLMItem) -> bool:
    if item.actionSignal is None:
        return False
    strength = _coerce_strength(
        item.actionSignal.actionStrength,
        item.actionSignal.isActionable,
        _coerce_role(item.actionSignal.role),
        item.kind if isinstance(item.kind, EventKind) else EventKind(str(item.kind)),
    )
    return bool(item.actionSignal.isActionable) and strength == "EXPLICIT"


def _materialize_memory_signal(
    item: AtomicEventLLMItem,
    kind: EventKind,
    action_signal: ActionSignal | None,
    evidence: list[EvidenceSpan] | None = None,
) -> MemorySignal | None:
    raw = item.memorySignal
    from services.conversation.event_pipeline.textutil import is_low_information_text

    importance = None
    worthy = False
    reason = None
    if raw is not None:
        worthy = bool(raw.isMemoryWorthy)
        importance = str(raw.importance or "").strip().upper() or None
        if importance not in {"LOW", "MEDIUM", "HIGH"}:
            importance = "MEDIUM" if worthy else "LOW"
        reason = str(raw.reason or "").strip() or None
    spans = evidence or []
    if spans and all(is_low_information_text(span.text) for span in spans):
        return MemorySignal(isMemoryWorthy=False, importance="LOW", reason="FILLER")
    if kind in MEMORY_EVENT_KINDS:
        if raw is None or not worthy:
            worthy = True
            if importance in {None, "LOW"}:
                importance = "HIGH" if kind in {
                    EventKind.DECISION,
                    EventKind.REQUIREMENT,
                    EventKind.ISSUE,
                    EventKind.RESULT,
                    EventKind.CONSTRAINT,
                    EventKind.CONTRADICTION,
                    EventKind.OPEN_QUESTION,
                } else "MEDIUM"
            reason = reason or kind.value
        if kind == EventKind.OPEN_QUESTION:
            reason = reason or "OPEN_QUESTION"
        return MemorySignal(isMemoryWorthy=True, importance=importance or "MEDIUM", reason=reason)
    if raw is None:
        return None
    return MemorySignal(isMemoryWorthy=worthy, importance=importance, reason=reason)


def _materialize_field_evidence(
    raw: FieldEvidenceLLMItem | None,
    sequence_text: dict[int, str],
    allowed: list[int],
) -> FieldEvidence:
    if raw is None:
        return FieldEvidence()
    return FieldEvidence(
        actionVerb=_ground_evidence(raw.actionVerb, sequence_text, allowed),
        actionObject=_ground_evidence(raw.actionObject, sequence_text, allowed),
        actor=_ground_evidence(raw.actor, sequence_text, allowed),
        deadline=_ground_evidence(raw.deadline, sequence_text, allowed),
    )


def _ground_verb(
    value: str | None,
    field_spans: list[EvidenceSpan],
    evidence: list[EvidenceSpan],
    sequence_text: dict[int, str],
    allowed: list[int],
) -> str | None:
    text = normalize_text(value)
    if not text:
        return None
    blob = _support_blob(field_spans, evidence, sequence_text, allowed)
    if casefold_text(text) in blob:
        return text
    # Keep the model's verb as a semantic label when the action itself is
    # grounded. Hinglish/multilingual transcripts rarely contain the English
    # paraphrase ("build" vs "banana hai").
    if field_spans or evidence:
        return text
    return None


def _ground_field(
    value: str | None,
    field_spans: list[EvidenceSpan],
    evidence: list[EvidenceSpan],
    sequence_text: dict[int, str],
    allowed: list[int],
    *,
    paraphrase_ok: bool,
) -> str | None:
    text = normalize_text(value)
    if not text:
        return None
    blob = _support_blob(field_spans, evidence, sequence_text, allowed)
    if casefold_text(text) in blob:
        return text
    tokens = [token for token in content_tokens(text) if len(token) > 2]
    if tokens and all(token.casefold() in blob for token in tokens):
        return text
    if paraphrase_ok and field_spans:
        return text
    return None


def _ground_action_object(
    value: str | None,
    field_spans: list[EvidenceSpan],
    evidence: list[EvidenceSpan],
    topic: LocalTopic,
    blocks: list[MicroBlock],
    sequence_text: dict[int, str],
    event_sequences: list[int],
    *,
    actionable: bool,
    claimed_grounding: str | None = None,
) -> tuple[str | None, list[EvidenceSpan], str | None, str | None]:
    text = normalize_text(value)
    if text and not _is_unresolved_object_text(text):
        grounding, spans = _object_grounding_in_scope(text, field_spans, evidence, topic, sequence_text)
        if grounding == "EXPLICIT":
            return text, spans, None, "EXPLICIT"
        if grounding == "INFERRED":
            return None, [], ABSTAIN_UNRESOLVED_OBJECT if actionable else None, "INFERRED"
        if _is_pronoun_or_deictic(text):
            resolved, resolved_spans, resolved_status = _resolve_pronoun_locally(topic, blocks, sequence_text, event_sequences)
            if resolved:
                return resolved, resolved_spans, None, "LOCAL_COREFERENCE"
            return None, [], ABSTAIN_UNRESOLVED_OBJECT if actionable else None, "UNRESOLVED"
        return None, [], ABSTAIN_UNRESOLVED_OBJECT if actionable else None, "UNRESOLVED"
    if actionable or _is_unresolved_object_text(text) or _is_pronoun_or_deictic(text):
        resolved, spans, status = _resolve_pronoun_locally(topic, blocks, sequence_text, event_sequences)
        if resolved:
            return resolved, spans, None, "LOCAL_COREFERENCE"
        if actionable:
            return None, [], ABSTAIN_UNRESOLVED_OBJECT, "UNRESOLVED"
    claimed = _coerce_grounding(claimed_grounding)
    if claimed == "INFERRED":
        return None, [], ABSTAIN_UNRESOLVED_OBJECT if actionable else None, "INFERRED"
    return None, [], None, "UNRESOLVED" if actionable else None


def _object_grounding_in_scope(
    text: str,
    field_spans: list[EvidenceSpan],
    evidence: list[EvidenceSpan],
    topic: LocalTopic,
    sequence_text: dict[int, str],
) -> tuple[str | None, list[EvidenceSpan]]:
    blob = casefold_text(
        " ".join(span.text for span in [*field_spans, *evidence])
        + " "
        + " ".join(sequence_text.get(sequence, "") for sequence in topic.sequenceIds)
    )
    blob_tokens = {token.casefold() for token in content_tokens(blob)}
    folded = casefold_text(text)
    if folded and folded in blob:
        spans = list(field_spans) or _spans_for_tokens(text, topic.sequenceIds, sequence_text)
        return "EXPLICIT", spans
    tokens = [token for token in content_tokens(text) if len(token) > 1]
    distinctive = [token for token in tokens if token.casefold() not in GENERIC_ACTION_OBJECTS | ACTION_PRONOUNS | DEICTIC_OR_TIME]
    if not distinctive:
        return None, []
    in_blob = [token for token in distinctive if _token_supported(token, blob, blob_tokens)]
    extra = [token for token in distinctive if token not in in_blob]
    if extra and len(in_blob) >= 1:
        return "INFERRED", []
    if in_blob and not extra:
        spans = list(field_spans) or _spans_for_tokens(" ".join(in_blob), topic.sequenceIds, sequence_text)
        return "EXPLICIT", spans
    return None, []


def _token_supported(token: str, blob: str, blob_tokens: set[str]) -> bool:
    folded = token.casefold()
    if folded in blob or folded in blob_tokens:
        return True
    if len(folded) < 3:
        return False
    return any(
        folded.startswith(item) or item.startswith(folded)
        for item in blob_tokens
        if len(item) >= 3
    )


def _resolve_pronoun_locally(
    topic: LocalTopic,
    blocks: list[MicroBlock],
    sequence_text: dict[int, str],
    event_sequences: list[int],
) -> tuple[str | None, list[EvidenceSpan], str | None]:
    event_set = set(event_sequences or [])
    search_sequences = _local_antecedent_sequences(blocks, event_set)
    candidates: list[tuple[str, int]] = []
    seen: set[str] = set()
    for sequence in search_sequences:
        line = sequence_text.get(sequence, "")
        if _mostly_referential(line) or not content_tokens(line):
            continue
        entities = [
            entity
            for entity in extract_entities(line)
            if entity.casefold() not in GENERIC_ACTION_OBJECTS | ACTION_PRONOUNS | DEICTIC_OR_TIME
        ]
        if entities:
            phrase = " ".join(entities)
            key = phrase.casefold()
            if key not in seen:
                seen.add(key)
                candidates.append((phrase, sequence))
            continue
        tokens = [
            token
            for token in content_tokens(line)
            if token.casefold() not in GENERIC_ACTION_OBJECTS | ACTION_PRONOUNS | DEICTIC_OR_TIME
        ]
        if 1 <= len(tokens) <= 4:
            phrase = " ".join(tokens)
            key = phrase.casefold()
            if key not in seen:
                seen.add(key)
                candidates.append((phrase, sequence))
    if len(candidates) != 1:
        return None, [], ABSTAIN_UNRESOLVED_OBJECT
    phrase, sequence = candidates[0]
    span = EvidenceSpan(sequenceStart=sequence, sequenceEnd=sequence, text=sequence_text.get(sequence, phrase))
    return phrase, [span], None


def _local_antecedent_sequences(blocks: list[MicroBlock], event_set: set[int]) -> list[int]:
    """Same micro-block, then the immediately preceding coherent micro-block only."""
    if not blocks:
        return []
    current_index = None
    for index, block in enumerate(blocks):
        if event_set & set(block.sequenceIds):
            current_index = index
            break
    if current_index is None:
        return []
    sequences: list[int] = []
    current = blocks[current_index]
    for sequence in current.sequenceIds:
        if sequence not in event_set and sequence not in sequences:
            sequences.append(sequence)
    for index in range(current_index - 1, -1, -1):
        prior = blocks[index]
        if _mostly_referential(prior.text) or not _block_has_object_tokens(prior.text):
            continue
        for sequence in prior.sequenceIds:
            if sequence not in event_set and sequence not in sequences:
                sequences.append(sequence)
        break
    return sequences


def _block_has_object_tokens(text: str) -> bool:
    tokens = [token.casefold() for token in content_tokens(text)]
    distinctive = [token for token in tokens if token not in DEICTIC_OR_TIME and token not in ACTION_PRONOUNS]
    return len(distinctive) >= 2


def _mostly_referential(text: str) -> bool:
    tokens = [token.casefold() for token in content_tokens(text)]
    if not tokens:
        return False
    referential = DEICTIC_OR_TIME | ACTION_PRONOUNS
    distinctive = [token for token in tokens if token not in referential]
    return len(distinctive) <= 1


def _is_unresolved_object_text(text: str | None) -> bool:
    value = normalize_text(text)
    if not value:
        return True
    if casefold_text(value) in GENERIC_ACTION_OBJECTS | ACTION_PRONOUNS:
        return True
    tokens = [token.casefold() for token in content_tokens(value)]
    distinctive = [token for token in tokens if token not in GENERIC_ACTION_OBJECTS | ACTION_PRONOUNS | DEICTIC_OR_TIME]
    return len(distinctive) == 0


def _local_scope_blob(
    topic: LocalTopic,
    blocks: list[MicroBlock],
    sequence_text: dict[int, str],
    event_sequences: list[int],
) -> str:
    parts = [topic.text]
    event_set = set(event_sequences or [])
    for block in blocks:
        if event_set & set(block.sequenceIds):
            parts.append(block.text)
    parts.extend(sequence_text.get(sequence, "") for sequence in topic.sequenceIds)
    return casefold_text(" ".join(parts))


def _support_blob(
    field_spans: list[EvidenceSpan],
    evidence: list[EvidenceSpan],
    sequence_text: dict[int, str],
    allowed: list[int],
) -> str:
    parts = [span.text for span in [*field_spans, *evidence]]
    parts.extend(sequence_text.get(sequence, "") for sequence in allowed)
    return casefold_text(" ".join(parts))


def _spans_for_tokens(text: str, allowed: list[int], sequence_text: dict[int, str]) -> list[EvidenceSpan]:
    needles = [token.casefold() for token in content_tokens(text)]
    if not needles:
        return []
    found: list[EvidenceSpan] = []
    for sequence in allowed:
        line = sequence_text.get(sequence, "")
        blob = casefold_text(line)
        if blob and (casefold_text(text) in blob or all(token in blob for token in needles if len(token) > 1)):
            found.append(EvidenceSpan(sequenceStart=sequence, sequenceEnd=sequence, text=line))
    return found


def _merge_spans(existing: list[EvidenceSpan], extra: list[EvidenceSpan]) -> list[EvidenceSpan]:
    merged = list(existing or [])
    seen = {(span.sequenceStart, span.sequenceEnd, span.text) for span in merged}
    for span in extra or []:
        key = (span.sequenceStart, span.sequenceEnd, span.text)
        if key in seen:
            continue
        seen.add(key)
        merged.append(span)
    return merged


def _is_pronoun_or_deictic(text: str | None) -> bool:
    value = casefold_text(text)
    return bool(value) and value in ACTION_PRONOUNS


def _coerce_strength(value: str | None, is_actionable: bool, role: str | None, kind: EventKind) -> str:
    key = str(value or "").strip().upper()
    if key in ACTION_STRENGTHS:
        return key
    if is_actionable or role in ACTION_ROLES or kind in ACTION_EVENT_KINDS:
        return "EXPLICIT"
    return "NONE"


def _coerce_grounding(value: str | None) -> str | None:
    if not value:
        return None
    key = str(value).strip().upper().replace(" ", "_")
    return key if key in OBJECT_GROUNDING_TYPES else None


def _coerce_role(value: str | None) -> str | None:
    if not value:
        return None
    key = str(value).strip().upper().replace(" ", "_")
    return key if key in ACTION_ROLES else None


def _role_from_kind(kind: EventKind) -> str | None:
    mapping = {
        EventKind.REQUEST: "REQUEST",
        EventKind.COMMITMENT: "COMMITMENT",
        EventKind.ASSIGNMENT: "ASSIGNMENT",
        EventKind.FOLLOW_UP: "FOLLOW_UP",
        EventKind.DEADLINE: "REQUEST",
        EventKind.COMPLETION: "COMMITMENT",
        EventKind.CANCELLATION: "REQUEST",
        EventKind.REQUIREMENT: "INSTRUCTION",
        EventKind.DECISION: "COMMITMENT",
        EventKind.FACT: "COMMITMENT",
        EventKind.STATE: "COMMITMENT",
        EventKind.RESULT: "COMMITMENT",
        EventKind.ISSUE: "REQUEST",
        EventKind.OPEN_QUESTION: "FOLLOW_UP",
        EventKind.PROPOSAL: "COMMITMENT",
    }
    return mapping.get(kind)


def _source_quality_flags(evidence: list[EvidenceSpan] | None, meaning: str | None) -> list[str]:
    """Structural source-quality flags. Not a language or domain blacklist."""
    flags: list[str] = []
    blob = " ".join(span.text for span in evidence or [])
    if not blob.strip():
        return flags
    if "�" in blob or "Ã" in blob:
        flags.append("LOW_CONFIDENCE_SOURCE")
    return flags


def _evidence_from_meaning(
    meaning: str,
    action_signal: ActionSignalLLMItem | None,
    topic: LocalTopic,
    sequence_text: dict[int, str],
) -> list[EvidenceSpan]:
    needles = {token.casefold() for token in content_tokens(meaning)}
    if action_signal:
        needles.update(token.casefold() for token in content_tokens(action_signal.object))
        needles.update(token.casefold() for token in content_tokens(action_signal.verb))
    needles -= GENERIC_ACTION_OBJECTS | ACTION_PRONOUNS | DEICTIC_OR_TIME
    if not needles:
        return []
    scored: list[tuple[int, int, str]] = []
    for sequence in topic.sequenceIds:
        line = sequence_text.get(sequence, "")
        if not line:
            continue
        line_tokens = {token.casefold() for token in content_tokens(line)}
        overlap = len(needles & line_tokens)
        if overlap <= 0:
            continue
        scored.append((overlap, sequence, line))
    scored.sort(key=lambda item: (-item[0], item[1]))
    spans: list[EvidenceSpan] = []
    for overlap, sequence, line in scored[:3]:
        if overlap < 1:
            continue
        spans.append(EvidenceSpan(sequenceStart=sequence, sequenceEnd=sequence, text=line))
    return spans


def _ground_evidence(
    spans: list[EvidenceSpan],
    sequence_text: dict[int, str],
    allowed: list[int],
) -> list[EvidenceSpan]:
    allowed_set = set(allowed)
    grounded: list[EvidenceSpan] = []
    seen: set[tuple[int, int, str]] = set()
    for span in spans or []:
        start = int(span.sequenceStart)
        end = int(span.sequenceEnd)
        if start not in allowed_set and end not in allowed_set:
            continue
        texts: list[str] = []
        for sequence in range(start, end + 1):
            if sequence not in allowed_set:
                continue
            line = sequence_text.get(sequence, "")
            if line:
                texts.append(line)
        combined = " ".join(texts).strip()
        if not combined:
            continue
        cited = normalize_text(span.text)
        if cited and casefold_text(cited) not in casefold_text(combined) and casefold_text(combined) not in casefold_text(cited):
            cited = combined
        elif not cited:
            cited = combined
        key = (start, end, cited)
        if key in seen:
            continue
        seen.add(key)
        grounded.append(EvidenceSpan(sequenceStart=start, sequenceEnd=end, text=cited))
    return grounded


def _block_index(blocks: list[MicroBlock]) -> dict[int, MicroBlock]:
    mapping: dict[int, MicroBlock] = {}
    for block in blocks:
        for sequence in block.sequenceIds:
            mapping[sequence] = block
    return mapping


def recover_uncovered_content_islands(
    topic: LocalTopic,
    blocks: list[MicroBlock],
    sequence_text: dict[int, str],
    existing: list[AtomicEvent],
    conversation_id: str = "",
    user_id: str = "",
    space_id: str = "",
) -> list[AtomicEvent]:
    """If extraction skipped a content island inside filler, keep the meaning.

    Sequence overlap alone is not coverage. The existing event must be about
    the same content. Filler stays accounted by placeholders or neighbors.
    """
    from services.conversation.event_pipeline.memory_identity import event_is_publishable_memory
    from services.conversation.event_pipeline.textutil import is_low_information_text
    from services.conversation.event_pipeline.topics import _has_object_tokens
    from services.conversation.event_pipeline.flags import topic_filler_density_threshold

    recovered: list[AtomicEvent] = []
    threshold = topic_filler_density_threshold()
    existing_events = list(existing or [])
    for block in blocks or []:
        content_sequences: list[int] = []
        content_lines: list[str] = []
        for sequence in block.sequenceIds:
            text = sequence_text.get(sequence) or ""
            if text and not is_low_information_text(text, threshold):
                content_sequences.append(sequence)
                content_lines.append(text)
        if not content_sequences:
            continue
        content = normalize_text(" ".join(content_lines))
        if not content or not _has_object_tokens(content):
            continue
        covering = [
            event
            for event in existing_events
            if event.kind != EventKind.NOISE and _event_covers_content(event, content, content_sequences)
        ]
        if covering:
            for event in covering:
                if not event_is_publishable_memory(event):
                    event.memorySignal = MemorySignal(isMemoryWorthy=True, importance="HIGH", reason=event.kind.value)
            continue
        evidence = [
            EvidenceSpan(sequenceStart=sequence, sequenceEnd=sequence, text=sequence_text.get(sequence) or content)
            for sequence in content_sequences
        ]
        recovered.append(
            AtomicEvent(
                eventId=stable_id("E", conversation_id or topic.topicId, "FACT", content, content_sequences),
                topicId=topic.topicId,
                kind=EventKind.FACT,
                meaning=content,
                entities=extract_entities(content),
                evidence=evidence,
                microBlockIds=[block.microBlockId],
                sequenceIds=list(content_sequences),
                conversationId=conversation_id,
                userId=user_id,
                spaceId=space_id,
                memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH", reason="CONTEXT"),
            )
        )
        existing_events.append(recovered[-1])
    return recovered


def _event_covers_content(event: AtomicEvent, content: str, content_sequences: list[int]) -> bool:
    from services.conversation.event_pipeline.textutil import token_jaccard

    if token_jaccard(content, event.meaning) >= 0.28:
        return True
    evidence_blob = " ".join(span.text for span in event.evidence or [])
    if token_jaccard(content, evidence_blob) >= 0.28:
        return True
    if set(content_sequences) & set(event.sequenceIds or []) and token_jaccard(content, event.meaning) >= 0.18:
        return True
    return False
