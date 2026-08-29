"""Canonical intermediate event store. Mongo when available, memory otherwise.

Upsert is idempotent. Overlapping checkpoint/STOP extractions with the same
canonical identity merge instead of emitting duplicate events. Distinct
updates on the same thread remain separate.
"""

from __future__ import annotations

import threading
from typing import Any

from services.conversation.event_pipeline.schemas import AtomicEvent
from services.conversation.event_pipeline.textutil import casefold_text, evidence_sequence_ids, token_jaccard


class ConversationEventStore:
    def __init__(self, repository: Any | None = None):
        self.repository = repository
        self._events: dict[str, dict[str, AtomicEvent]] = {}
        self._lock = threading.Lock()

    async def upsert(self, conversation_id: str, events: list[AtomicEvent]) -> list[AtomicEvent]:
        with self._lock:
            bucket = self._events.setdefault(conversation_id, {})
            for event in events:
                existing = bucket.get(event.eventId)
                if existing is None:
                    duplicate = _find_canonical_duplicate(event, list(bucket.values()))
                    if duplicate is not None:
                        _merge_event(duplicate, event)
                        bucket[duplicate.eventId] = duplicate
                        continue
                    bucket[event.eventId] = event
                    continue
                _merge_event(existing, event)
                bucket[event.eventId] = existing
            snapshot = [event.model_dump() for event in bucket.values()]
            result = list(bucket.values())
        if self.repository is not None and hasattr(self.repository, "upsert_conversation_events"):
            await self.repository.upsert_conversation_events(conversation_id, snapshot)
        return result

    async def list(self, conversation_id: str) -> list[AtomicEvent]:
        if self.repository is not None and hasattr(self.repository, "list_conversation_events"):
            stored = await self.repository.list_conversation_events(conversation_id)
            if stored:
                events = [
                    item if isinstance(item, AtomicEvent) else AtomicEvent.model_validate(item)
                    for item in stored
                ]
                self._events[conversation_id] = {event.eventId: event for event in events}
                return events
        bucket = self._events.get(conversation_id, {})
        return list(bucket.values())

    async def replace(self, conversation_id: str, events: list[AtomicEvent]) -> list[AtomicEvent]:
        self._events[conversation_id] = {event.eventId: event for event in events}
        if self.repository is not None and hasattr(self.repository, "replace_conversation_events"):
            await self.repository.replace_conversation_events(conversation_id, [event.model_dump() for event in events])
        elif self.repository is not None and hasattr(self.repository, "upsert_conversation_events"):
            await self.repository.upsert_conversation_events(conversation_id, [event.model_dump() for event in events])
        return events


def _find_canonical_duplicate(event: AtomicEvent, existing: list[AtomicEvent]) -> AtomicEvent | None:
    incoming_seqs = set(event.sequenceIds or evidence_sequence_ids(event.evidence))
    for other in existing:
        if other.eventId == event.eventId:
            return other
        from services.conversation.event_pipeline.memory_identity import memory_relation

        relation = memory_relation(event, other)
        if other.kind != event.kind and relation in {"UPDATE", "SUPERSEDE", None}:
            continue
        similarity = token_jaccard(event.meaning, other.meaning)
        if similarity < 0.78:
            continue
        if relation in {"DISTINCT", "RELATED", "UPDATE", "SUPERSEDE"}:
            continue
        other_seqs = set(other.sequenceIds or evidence_sequence_ids(other.evidence))
        if incoming_seqs & other_seqs:
            return other
        if incoming_seqs and other_seqs:
            gap = abs(min(incoming_seqs) - min(other_seqs))
            object_match = casefold_text(event.object or "") == casefold_text(other.object or "") and bool(event.object)
            if gap <= 3 and (similarity >= 0.9 or object_match):
                return other
    return None


def _merge_event(target: AtomicEvent, incoming: AtomicEvent) -> None:
    seen_seq = set(target.sequenceIds or [])
    for sequence in incoming.sequenceIds or []:
        if sequence not in seen_seq:
            target.sequenceIds.append(sequence)
            seen_seq.add(sequence)
    seen_source = set(target.sourceIds or [])
    for source in incoming.sourceIds or []:
        if source not in seen_source:
            target.sourceIds.append(source)
            seen_source.add(source)
    seen_blocks = set(target.microBlockIds or [])
    for block_id in incoming.microBlockIds or []:
        if block_id not in seen_blocks:
            target.microBlockIds.append(block_id)
            seen_blocks.add(block_id)
    seen_evidence = {(span.sequenceStart, span.sequenceEnd, span.text) for span in target.evidence}
    for span in incoming.evidence or []:
        key = (span.sequenceStart, span.sequenceEnd, span.text)
        if key not in seen_evidence:
            target.evidence.append(span)
            seen_evidence.add(key)
    if incoming.entities:
        known = {item.casefold() for item in target.entities}
        for entity in incoming.entities:
            if entity.casefold() not in known:
                target.entities.append(entity)
                known.add(entity.casefold())
    if incoming.object and not target.object:
        target.object = incoming.object
    if incoming.actor and not target.actor:
        target.actor = incoming.actor
    if incoming.timeExpression and not target.timeExpression:
        target.timeExpression = incoming.timeExpression
    if incoming.actionSignal and (target.actionSignal is None or incoming.actionSignal.isActionable):
        target.actionSignal = incoming.actionSignal
    if incoming.memorySignal and (target.memorySignal is None or incoming.memorySignal.isMemoryWorthy):
        target.memorySignal = incoming.memorySignal
    if incoming.fieldEvidence and target.fieldEvidence is None:
        target.fieldEvidence = incoming.fieldEvidence
    if len(incoming.meaning or "") > len(target.meaning or ""):
        target.meaning = incoming.meaning
