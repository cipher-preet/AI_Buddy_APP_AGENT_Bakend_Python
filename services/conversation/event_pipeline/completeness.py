"""Proposition-level semantic completeness review and targeted repair.

Block-level coverage is not semantic completeness. A micro-block that produced
one event may still have independent supported meanings with no atomic event.
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field

from services.conversation.event_pipeline.schemas import (
    AtomicEvent,
    CoverageSemanticUnitRecord,
    EventKind,
    LocalTopic,
    MicroBlock,
    SemanticUnitDisposition,
)
from services.conversation.event_pipeline.textutil import evidence_sequence_ids, normalize_text, token_jaccard
from services.llm.router import LLMRouter


_ACCOUNTING_STATUSES = {
    "COVERED": SemanticUnitDisposition.EVENT_CREATED,
    "EVENT_CREATED": SemanticUnitDisposition.EVENT_CREATED,
    "MERGED": SemanticUnitDisposition.MERGED_WITH_EVENT,
    "MERGED_WITH_EVENT": SemanticUnitDisposition.MERGED_WITH_EVENT,
    "LOW_VALUE": SemanticUnitDisposition.LOW_VALUE,
    "NOISE": SemanticUnitDisposition.NOISE,
    "UNSUPPORTED": SemanticUnitDisposition.UNSUPPORTED,
    "DUPLICATE": SemanticUnitDisposition.DUPLICATE,
    "AMBIGUOUS": SemanticUnitDisposition.AMBIGUOUS,
    "MISSING": None,
}


class MissingSemanticUnit(BaseModel):
    meaning: str
    kind: str | None = None
    microBlockId: str = ""
    sequenceStart: int | None = None
    sequenceEnd: int | None = None
    evidenceText: str = ""
    status: str = "MISSING"


class CompletenessBlockReview(BaseModel):
    microBlockId: str = ""
    complete: bool = True
    missingSemanticUnits: list[MissingSemanticUnit] = Field(default_factory=list)
    units: list[MissingSemanticUnit] = Field(default_factory=list)


class CompletenessReviewResponse(BaseModel):
    complete: bool | None = None
    missingSemanticUnits: list[MissingSemanticUnit] = Field(default_factory=list)
    blocks: list[CompletenessBlockReview] = Field(default_factory=list)


class CompletenessReviewer(Protocol):
    async def review(
        self,
        topic: LocalTopic,
        blocks: list[MicroBlock],
        events: list[AtomicEvent],
        sequence_text: dict[int, str],
    ) -> CompletenessReviewResponse:
        ...


class ScriptedCompletenessReviewer:
    """Test double. Empty expected_units means every reviewed block is complete."""

    def __init__(self, expected_units: list[MissingSemanticUnit] | None = None):
        self.expected_units = list(expected_units or [])
        self.calls = 0

    async def review(
        self,
        topic: LocalTopic,
        blocks: list[MicroBlock],
        events: list[AtomicEvent],
        sequence_text: dict[int, str],
    ) -> CompletenessReviewResponse:
        from services.conversation.event_pipeline.topics import _is_filler_block

        self.calls += 1
        block_reviews: list[CompletenessBlockReview] = []
        for block in blocks:
            if _is_filler_block(block):
                continue
            missing: list[MissingSemanticUnit] = []
            for unit in self.expected_units:
                if not _unit_belongs_to_block(unit, block):
                    continue
                if any(event.kind != EventKind.NOISE and event_covers_proposition(event, unit.meaning) for event in events):
                    continue
                item = unit.model_copy(deep=True)
                item.microBlockId = block.microBlockId
                missing.append(item)
            block_reviews.append(
                CompletenessBlockReview(
                    microBlockId=block.microBlockId,
                    complete=not missing,
                    missingSemanticUnits=missing,
                )
            )
        return CompletenessReviewResponse(blocks=block_reviews)


class LLMCompletenessReviewer:
    def __init__(self, router: LLMRouter):
        self.router = router
        self.calls = 0
        self.failures = 0

    async def review(
        self,
        topic: LocalTopic,
        blocks: list[MicroBlock],
        events: list[AtomicEvent],
        sequence_text: dict[int, str],
    ) -> CompletenessReviewResponse:
        from services.conversation.event_pipeline.llm import generate_structured_for_stage
        from services.conversation.event_pipeline.routing import PipelineStage
        from services.conversation.event_pipeline.topics import _is_filler_block

        content_blocks = [block for block in blocks if not _is_filler_block(block)]
        if not content_blocks:
            return CompletenessReviewResponse(complete=True, blocks=[])
        self.calls += 1
        payload = {
            "topicId": topic.topicId,
            "topicLabel": topic.label,
            "microBlocks": [
                {
                    "microBlockId": block.microBlockId,
                    "sequenceIds": list(block.sequenceIds),
                    "text": block.text,
                }
                for block in content_blocks
            ],
            "extractedEvents": [
                {
                    "eventId": event.eventId,
                    "kind": event.kind.value if hasattr(event.kind, "value") else str(event.kind),
                    "meaning": event.meaning,
                    "microBlockIds": list(event.microBlockIds or []),
                    "sequenceIds": list(event.sequenceIds or evidence_sequence_ids(event.evidence)),
                    "evidence": [span.model_dump() for span in (event.evidence or [])],
                }
                for event in events
                if event.kind != EventKind.NOISE
            ],
        }
        try:
            response, _, _ = await generate_structured_for_stage(
                self.router,
                PipelineStage.SEMANTIC_COMPLETENESS,
                "semantic-completeness-reviewer-v1",
                CompletenessReviewResponse,
                payload,
            )
        except Exception as error:
            from services.conversation.event_pipeline.observability import record_failure
            from services.llm.async_runtime import reraise_if_hard_runtime

            record_failure(error)
            reraise_if_hard_runtime(error)
            self.failures += 1
            return CompletenessReviewResponse(complete=True, blocks=[])
        if isinstance(response, dict):
            response = CompletenessReviewResponse.model_validate(response)
        return _normalize_review(response, content_blocks)


async def review_and_repair_semantic_completeness(
    topic: LocalTopic,
    blocks: list[MicroBlock],
    sequence_text: dict[int, str],
    events: list[AtomicEvent],
    reviewer: CompletenessReviewer,
    extractor: Any,
    *,
    conversation_id: str = "",
    user_id: str = "",
    space_id: str = "",
) -> tuple[list[AtomicEvent], list[CoverageSemanticUnitRecord], bool]:
    """One completeness review + at most one targeted repair pass for this topic."""
    from services.conversation.event_pipeline.topics import _is_filler_block

    content_blocks = [block for block in blocks if not _is_filler_block(block)]
    records: list[CoverageSemanticUnitRecord] = []
    if not content_blocks:
        return [], records, False

    try:
        review = await reviewer.review(topic, content_blocks, events, sequence_text)
    except Exception as error:
        from services.conversation.event_pipeline.observability import record_failure
        from services.llm.async_runtime import reraise_if_hard_runtime

        record_failure(error)
        reraise_if_hard_runtime(error)
        return [], records, True

    review = _normalize_review(review, content_blocks)
    missing: list[MissingSemanticUnit] = []
    covered_ids: set[str] = set()
    for block_review in review.blocks:
        block = _block_by_id(content_blocks, block_review.microBlockId)
        for unit in block_review.units:
            status = str(unit.status or "").strip().upper()
            disposition = _ACCOUNTING_STATUSES.get(status)
            sequences = _unit_sequences(unit, block)
            matched = _matching_event(events, unit.meaning, sequences)
            if disposition is None and status in {"", "MISSING"}:
                if matched is not None:
                    records.append(
                        _record_for_unit(
                            unit,
                            block_review.microBlockId,
                            SemanticUnitDisposition.EVENT_CREATED,
                            matched.eventId,
                            "already_extracted",
                            sequences,
                        )
                    )
                    covered_ids.add(matched.eventId)
                else:
                    missing.append(unit)
                continue
            if disposition is None:
                missing.append(unit)
                continue
            records.append(
                _record_for_unit(
                    unit,
                    block_review.microBlockId,
                    disposition,
                    matched.eventId if matched is not None else None,
                    status.lower(),
                    sequences,
                )
            )
            if matched is not None:
                covered_ids.add(matched.eventId)
        for unit in block_review.missingSemanticUnits:
            item = unit.model_copy(deep=True)
            item.microBlockId = item.microBlockId or block_review.microBlockId
            sequences = _unit_sequences(item, block)
            matched = _matching_event(events, item.meaning, sequences)
            if matched is not None:
                records.append(
                    _record_for_unit(
                        item,
                        item.microBlockId,
                        SemanticUnitDisposition.EVENT_CREATED,
                        matched.eventId,
                        "already_extracted",
                        sequences,
                    )
                )
                covered_ids.add(matched.eventId)
                continue
            missing.append(item)

    recovered: list[AtomicEvent] = []
    if missing:
        extract_missing = getattr(extractor, "extract_missing", None)
        if callable(extract_missing):
            try:
                recovered = await extract_missing(
                    topic,
                    content_blocks,
                    sequence_text,
                    missing,
                    existing=events,
                )
            except Exception as error:
                from services.conversation.event_pipeline.observability import record_failure
                from services.llm.async_runtime import reraise_if_hard_runtime

                record_failure(error)
                reraise_if_hard_runtime(error)
                recovered = []
        recovered = [
            event
            for event in recovered or []
            if event.kind != EventKind.NOISE
            and not any(
                existing.kind != EventKind.NOISE and event_covers_proposition(existing, event.meaning)
                for existing in events
            )
        ]
        for event in recovered:
            event.conversationId = event.conversationId or conversation_id
            event.userId = event.userId or user_id
            event.spaceId = event.spaceId or space_id
            if not event.sequenceIds:
                event.sequenceIds = evidence_sequence_ids(event.evidence)

    remaining_missing = list(missing)
    kept: list[AtomicEvent] = []
    for event in recovered:
        sequences = list(event.sequenceIds or evidence_sequence_ids(event.evidence))
        matched_unit = None
        for index, unit in enumerate(remaining_missing):
            if event_covers_proposition(event, unit.meaning):
                matched_unit = remaining_missing.pop(index)
                break
        if matched_unit is None:
            records.append(
                CoverageSemanticUnitRecord(
                    microBlockId=(event.microBlockIds[0] if event.microBlockIds else ""),
                    meaning=event.meaning,
                    kind=event.kind.value if hasattr(event.kind, "value") else str(event.kind),
                    disposition=SemanticUnitDisposition.EVENT_CREATED,
                    eventId=event.eventId,
                    reason="repair_additional_grounded",
                    sequenceIds=sequences,
                )
            )
            kept.append(event)
            continue
        uncertainty = {str(flag).strip().upper() for flag in (event.uncertainty or [])}
        if {"AMBIGUOUS", "NOISE", "LOW_CONFIDENCE_SOURCE"} & uncertainty and not event.evidence:
            records.append(
                _record_for_unit(
                    matched_unit,
                    matched_unit.microBlockId,
                    SemanticUnitDisposition.AMBIGUOUS,
                    event.eventId,
                    "repair_uncertain",
                    sequences,
                )
            )
            continue
        records.append(
            _record_for_unit(
                matched_unit,
                matched_unit.microBlockId,
                SemanticUnitDisposition.EVENT_CREATED,
                event.eventId,
                "repair_extracted",
                sequences,
            )
        )
        kept.append(event)

    for unit in remaining_missing:
        status = str(unit.status or "MISSING").strip().upper()
        disposition = _ACCOUNTING_STATUSES.get(status) or SemanticUnitDisposition.UNSUPPORTED
        if disposition is None:
            disposition = SemanticUnitDisposition.UNSUPPORTED
        records.append(
            _record_for_unit(
                unit,
                unit.microBlockId,
                disposition,
                None,
                "repair_abstained" if disposition == SemanticUnitDisposition.UNSUPPORTED else status.lower(),
                _unit_sequences(unit, _block_by_id(content_blocks, unit.microBlockId)),
            )
        )

    for event in events:
        if event.eventId in covered_ids:
            continue
        if event.kind == EventKind.NOISE:
            continue
        block_id = event.microBlockIds[0] if event.microBlockIds else ""
        if block_id and block_id not in {block.microBlockId for block in content_blocks}:
            continue
        records.append(
            CoverageSemanticUnitRecord(
                microBlockId=block_id,
                meaning=event.meaning,
                kind=event.kind.value if hasattr(event.kind, "value") else str(event.kind),
                disposition=SemanticUnitDisposition.EVENT_CREATED,
                eventId=event.eventId,
                reason="first_pass",
                sequenceIds=list(event.sequenceIds or evidence_sequence_ids(event.evidence)),
            )
        )

    return kept, records, True


def bootstrap_semantic_records(
    blocks: list[MicroBlock],
    events: list[AtomicEvent],
) -> list[CoverageSemanticUnitRecord]:
    """When completeness review did not run, account extracted events only."""
    records: list[CoverageSemanticUnitRecord] = []
    seen: set[str] = set()
    for event in events:
        if event.eventId in seen:
            continue
        seen.add(event.eventId)
        disposition = (
            SemanticUnitDisposition.NOISE if event.kind == EventKind.NOISE else SemanticUnitDisposition.EVENT_CREATED
        )
        block_id = event.microBlockIds[0] if event.microBlockIds else ""
        records.append(
            CoverageSemanticUnitRecord(
                microBlockId=block_id,
                meaning=event.meaning,
                kind=event.kind.value if hasattr(event.kind, "value") else str(event.kind),
                disposition=disposition,
                eventId=event.eventId,
                reason="event_as_unit",
                sequenceIds=list(event.sequenceIds or evidence_sequence_ids(event.evidence)),
            )
        )
    return records


def _normalize_review(
    review: CompletenessReviewResponse | dict[str, Any],
    blocks: list[MicroBlock],
) -> CompletenessReviewResponse:
    if isinstance(review, dict):
        review = CompletenessReviewResponse.model_validate(review)
    if review.blocks:
        by_id = {block.microBlockId: block for block in blocks}
        for item in review.blocks:
            if not item.microBlockId and len(blocks) == 1:
                item.microBlockId = blocks[0].microBlockId
            block = by_id.get(item.microBlockId)
            for unit in [*item.missingSemanticUnits, *item.units]:
                unit.microBlockId = unit.microBlockId or item.microBlockId
                if unit.sequenceStart is None and block is not None:
                    unit.sequenceStart = min(block.sequenceIds or [0])
                    unit.sequenceEnd = max(block.sequenceIds or [0])
        return review
    if len(blocks) == 1:
        missing = list(review.missingSemanticUnits)
        for unit in missing:
            unit.microBlockId = blocks[0].microBlockId
        return CompletenessReviewResponse(
            blocks=[
                CompletenessBlockReview(
                    microBlockId=blocks[0].microBlockId,
                    complete=bool(review.complete) if review.complete is not None else not missing,
                    missingSemanticUnits=missing,
                )
            ]
        )
    if review.complete is False and review.missingSemanticUnits:
        assigned: list[CompletenessBlockReview] = []
        remaining = list(review.missingSemanticUnits)
        for block in blocks:
            belonging = [unit for unit in remaining if _unit_belongs_to_block(unit, block)]
            remaining = [unit for unit in remaining if unit not in belonging]
            for unit in belonging:
                unit.microBlockId = block.microBlockId
            assigned.append(
                CompletenessBlockReview(
                    microBlockId=block.microBlockId,
                    complete=not belonging,
                    missingSemanticUnits=belonging,
                )
            )
        if remaining and assigned:
            assigned[-1].missingSemanticUnits.extend(remaining)
            assigned[-1].complete = False
        return CompletenessReviewResponse(blocks=assigned)
    return CompletenessReviewResponse(
        blocks=[
            CompletenessBlockReview(microBlockId=block.microBlockId, complete=True)
            for block in blocks
        ]
    )


def _unit_belongs_to_block(unit: MissingSemanticUnit, block: MicroBlock) -> bool:
    if unit.microBlockId and unit.microBlockId == block.microBlockId:
        return True
    sequences = set(_unit_sequences(unit, None))
    if sequences and sequences & set(block.sequenceIds or []):
        return True
    blob = f"{block.text or ''}"
    needle = normalize_text(unit.evidenceText or unit.meaning)
    return bool(needle) and needle.casefold() in blob.casefold()


def _unit_sequences(unit: MissingSemanticUnit | None, block: MicroBlock | None) -> list[int]:
    if unit is not None and unit.sequenceStart is not None:
        end = unit.sequenceEnd if unit.sequenceEnd is not None else unit.sequenceStart
        return list(range(int(unit.sequenceStart), int(end) + 1))
    if block is not None:
        return list(block.sequenceIds or [])
    return []


def _block_by_id(blocks: list[MicroBlock], block_id: str) -> MicroBlock | None:
    for block in blocks:
        if block.microBlockId == block_id:
            return block
    return None


def event_covers_proposition(event: AtomicEvent, meaning: str) -> bool:
    """Same proposition, not merely the same subject or the same evidence span."""
    if not meaning or event.kind == EventKind.NOISE:
        return False
    return token_jaccard(meaning, event.meaning) >= 0.45


def _matching_event(events: list[AtomicEvent], meaning: str, sequences: list[int]) -> AtomicEvent | None:
    for event in events:
        if event_covers_proposition(event, meaning):
            return event
    return None


def _record_for_unit(
    unit: MissingSemanticUnit,
    block_id: str,
    disposition: SemanticUnitDisposition,
    event_id: str | None,
    reason: str,
    sequences: list[int],
) -> CoverageSemanticUnitRecord:
    return CoverageSemanticUnitRecord(
        microBlockId=block_id or unit.microBlockId,
        meaning=unit.meaning,
        kind=unit.kind,
        disposition=disposition,
        eventId=event_id,
        reason=reason,
        sequenceIds=list(sequences or []),
    )
