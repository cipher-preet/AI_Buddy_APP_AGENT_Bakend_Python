"""Task/Note synthesis from validated events + thread context. Wording only."""

from __future__ import annotations

import json
from typing import Any, Protocol

from pydantic import BaseModel, Field

from services.conversation.event_pipeline.channels import (
    action_object_grounded,
    event_is_task_eligible,
    is_generic_task_text,
    object_grounding_type,
    unresolved_action_object,
)
from services.conversation.event_pipeline.schemas import AtomicEvent, EventKind, GlobalThread
from services.conversation.event_pipeline.textutil import normalize_text
from services.conversation.fingerprints import note_fingerprint, task_fingerprint
from services.conversation.models import ExtractedNote, ExtractedTask
from services.llm.router import LLMCapability, LLMRouter


class TaskSynthesizer(Protocol):
    async def synthesize(self, event: AtomicEvent, thread: GlobalThread | None) -> ExtractedTask | None:
        ...


class NoteSynthesizer(Protocol):
    async def synthesize(self, event: AtomicEvent, thread: GlobalThread | None) -> ExtractedNote | None:
        ...


class SynthesizedTaskItem(BaseModel):
    title: str
    body: str = ""
    dueDateText: str | None = None
    ownerText: str | None = None


class SynthesizedNoteItem(BaseModel):
    title: str
    body: str


class DeterministicTaskSynthesizer:
    async def synthesize(self, event: AtomicEvent, thread: GlobalThread | None) -> ExtractedTask | None:
        if event.kind in {EventKind.NOISE}:
            return None
        if not event_is_task_eligible(event):
            return None
        grounding = object_grounding_type(event)
        if grounding in {"INFERRED", "UNRESOLVED"}:
            return None
        if unresolved_action_object(event) or not action_object_grounded(event):
            return None
        title = _task_title(event)
        body = _task_body(event, thread)
        if is_generic_task_text(title, body, event.object):
            return None
        operation = "COMPLETE" if event.kind == EventKind.COMPLETION else "CANCEL" if event.kind == EventKind.CANCELLATION else "CREATE"
        evidence = _artifact_evidence_spans(event)
        task = ExtractedTask(
            title=title,
            body=body,
            operation=operation,
            ownerText=event.actor,
            dueDateText=event.timeExpression,
            dueDateStatus="ambiguous" if event.timeExpression else "none",
            confidence=0.72 if not event.uncertainty else 0.6,
            sourceConversationId=event.conversationId or "conversation",
            evidence=evidence,
            origin="explicit",
            changes=_artifact_metadata(event, thread, "task", evidence),
        )
        task.fingerprint = task_fingerprint(event.spaceId or event.conversationId, task)
        return task


class DeterministicNoteSynthesizer:
    async def synthesize(self, event: AtomicEvent, thread: GlobalThread | None) -> ExtractedNote | None:
        if event.kind == EventKind.NOISE:
            return None
        if {"NOISE", "LOW_CONFIDENCE_SOURCE"} & {str(flag).strip().upper() for flag in (event.uncertainty or [])}:
            if event.kind != EventKind.DECISION:
                return None
        title = _note_title(event)
        body = _note_body(event, thread)
        if not title or not body or normalize_text(title).casefold() == normalize_text(body).casefold():
            body = event.meaning if len(event.meaning) > len(title) else f"{title}. {event.meaning}".strip()
        if len(normalize_text(body)) <= len(normalize_text(title)):
            body = event.meaning
        note = ExtractedNote(
            title=title,
            body=body,
            confidence=0.7,
            sourceConversationId=event.conversationId or "conversation",
            evidence=list(event.evidence),
            debug=_artifact_metadata(event, thread, "note", list(event.evidence)),
        )
        note.fingerprint = note_fingerprint(event.spaceId or event.conversationId, note)
        return _preserve_note_epistemic_status(event, note)


class LLMTaskSynthesizer:
    def __init__(self, router: LLMRouter, fallback: TaskSynthesizer | None = None):
        self.router = router
        self.fallback = fallback or DeterministicTaskSynthesizer()
        self.calls = 0
        self.requested_capabilities: list[LLMCapability] = []

    async def synthesize(self, event: AtomicEvent, thread: GlobalThread | None) -> ExtractedTask | None:
        drafted = await self.fallback.synthesize(event, thread)
        if drafted is None:
            return None
        from services.conversation.event_pipeline.llm import compact_event, compact_thread, generate_structured_for_stage
        from services.conversation.event_pipeline.routing import PipelineStage, capability_for_stage

        self.calls += 1
        self.requested_capabilities.append(capability_for_stage(PipelineStage.TASK_SYNTHESIS))
        try:
            polished, _, _ = await generate_structured_for_stage(
                self.router,
                PipelineStage.TASK_SYNTHESIS,
                "task-synthesizer-v1",
                SynthesizedTaskItem,
                {"event": compact_event(event), "draft": {"title": drafted.title, "body": drafted.body}},
                background=json.dumps({"thread": compact_thread(thread)}, default=str),
            )
        except Exception as error:
            from services.llm.async_runtime import reraise_if_hard_runtime
            from services.conversation.event_pipeline.observability import record_failure

            record_failure(error)
            reraise_if_hard_runtime(error)
            return drafted
        if polished and getattr(polished, "title", None):
            drafted.title = normalize_text(polished.title) or drafted.title
            drafted.body = normalize_text(polished.body) or drafted.body
            if polished.dueDateText and event.timeExpression:
                drafted.dueDateText = event.timeExpression
            if polished.ownerText and event.actor:
                drafted.ownerText = event.actor
            if is_generic_task_text(drafted.title, drafted.body, event.object):
                return None
        return drafted


class LLMNoteSynthesizer:
    def __init__(self, router: LLMRouter, fallback: NoteSynthesizer | None = None):
        self.router = router
        self.fallback = fallback or DeterministicNoteSynthesizer()
        self.calls = 0
        self.requested_capabilities: list[LLMCapability] = []

    async def synthesize(self, event: AtomicEvent, thread: GlobalThread | None) -> ExtractedNote | None:
        drafted = await self.fallback.synthesize(event, thread)
        if drafted is None:
            return None
        from services.conversation.event_pipeline.llm import compact_event, compact_thread, generate_structured_for_stage
        from services.conversation.event_pipeline.routing import PipelineStage, capability_for_stage

        self.calls += 1
        self.requested_capabilities.append(capability_for_stage(PipelineStage.NOTE_SYNTHESIS))
        try:
            polished, _, _ = await generate_structured_for_stage(
                self.router,
                PipelineStage.NOTE_SYNTHESIS,
                "note-synthesizer-v1",
                SynthesizedNoteItem,
                {"event": compact_event(event), "draft": {"title": drafted.title, "body": drafted.body}},
                background=json.dumps({"thread": compact_thread(thread)}, default=str),
            )
        except Exception as error:
            from services.llm.async_runtime import reraise_if_hard_runtime
            from services.conversation.event_pipeline.observability import record_failure

            record_failure(error)
            reraise_if_hard_runtime(error)
            return drafted
        if polished and getattr(polished, "title", None):
            drafted.title = normalize_text(polished.title) or drafted.title
            drafted.body = normalize_text(polished.body) or drafted.body
        drafted = _preserve_note_epistemic_status(event, drafted)
        return drafted


def _task_title(event: AtomicEvent) -> str:
    obj = normalize_text(event.object) or _object_from_meaning(event.meaning)
    verb = ""
    role = None
    if event.actionSignal:
        verb = normalize_text(event.actionSignal.verb)
        role = event.actionSignal.role
        obj = (
            normalize_text(event.actionSignal.canonicalActionObject)
            or normalize_text(event.actionSignal.object)
            or obj
        )
    if verb and obj:
        title = f"{verb} {obj}".strip()
        return title[0].upper() + title[1:] if title else obj
    if role == "FOLLOW_UP" or event.kind == EventKind.FOLLOW_UP:
        return f"Follow up on {obj}".strip()
    if event.kind == EventKind.DEADLINE:
        return f"Meet deadline for {obj}".strip()
    if event.kind in {EventKind.COMPLETION}:
        return f"Complete {obj}".strip()
    if event.kind == EventKind.CANCELLATION:
        return f"Cancel {obj}".strip()
    meaning = event.meaning.rstrip(".")
    if len(meaning) <= 72:
        return meaning[0].upper() + meaning[1:] if meaning else obj
    return f"Handle {obj}" if obj else meaning[:72]


def _task_body(event: AtomicEvent, thread: GlobalThread | None) -> str:
    parts = [event.meaning]
    if event.actor:
        parts.append(f"Owner mentioned: {event.actor}.")
    if event.timeExpression:
        parts.append(f"Time mentioned: {event.timeExpression}.")
    body = " ".join(parts).strip()
    title = _task_title(event)
    if normalize_text(body).casefold() == normalize_text(title).casefold() or len(normalize_text(body)) <= len(normalize_text(title)):
        obj = normalize_text(event.object)
        if event.actionSignal:
            obj = (
                normalize_text(event.actionSignal.canonicalActionObject)
                or normalize_text(event.actionSignal.object)
                or obj
            )
        if obj and obj.casefold() not in body.casefold():
            body = f"{body} Object: {obj}.".strip()
    return body


def _note_title(event: AtomicEvent) -> str:
    if event.object:
        return event.object[:80]
    if event.entities:
        return " ".join(event.entities[:4])[:80]
    return event.meaning[:80]


def _note_body(event: AtomicEvent, thread: GlobalThread | None) -> str:
    return event.meaning


_UNRESOLVED_MEMORY_REASONS = frozenset({"OPEN_QUESTION", "QUESTION", "OPEN_DECISION", "ISSUE", "PROPOSAL"})
_UNRESOLVED_KINDS = frozenset({EventKind.OPEN_QUESTION, EventKind.ISSUE, EventKind.PROPOSAL})


def _event_is_unresolved_epistemic(event: AtomicEvent) -> bool:
    if event.kind in _UNRESOLVED_KINDS:
        return True
    reason = str(getattr(event.memorySignal, "reason", "") or "").strip().upper()
    if reason in _UNRESOLVED_MEMORY_REASONS:
        return True
    uncertainty = {str(item).strip().upper() for item in (event.uncertainty or [])}
    if uncertainty & {"UNRESOLVED_QUESTION", "AMBIGUOUS", "LOW_CONFIDENCE_SOURCE", "NOISE"}:
        return True
    evidence_blob = " ".join(span.text for span in event.evidence or [])
    return "?" in evidence_blob or "?" in (event.meaning or "")


def _preserve_note_epistemic_status(event: AtomicEvent, note: ExtractedNote) -> ExtractedNote:
    if not _event_is_unresolved_epistemic(event):
        return note
    note.body = event.meaning
    if event.kind == EventKind.OPEN_QUESTION or "?" in (event.meaning or ""):
        note.title = event.object or event.meaning[:80]
    metadata = dict(note.debug or {})
    metadata["epistemicStatus"] = "unresolved"
    note.debug = metadata
    return note


def _object_from_meaning(meaning: str) -> str:
    text = normalize_text(meaning)
    for prefix in ("please ", "we will ", "we need to ", "i will ", "let's ", "lets "):
        if text.casefold().startswith(prefix):
            text = text[len(prefix) :]
    return text[:80]


def _artifact_evidence_spans(event: AtomicEvent) -> list:
    """Sequences supporting verb + object + deadline/actor only. Not thread context."""
    field = event.fieldEvidence
    if field is not None:
        spans = [
            *(field.actionVerb or []),
            *(field.actionObject or []),
            *(field.actor or []),
            *(field.deadline or []),
        ]
        if spans:
            return _unique_spans(spans)
    return list(event.evidence or [])


def _unique_spans(spans: list) -> list:
    merged = []
    seen: set[tuple] = set()
    for span in spans:
        key = (getattr(span, "sequenceStart", None), getattr(span, "sequenceEnd", None), getattr(span, "text", None))
        if key in seen:
            continue
        seen.add(key)
        merged.append(span)
    return merged


def _artifact_metadata(event: AtomicEvent, thread: GlobalThread | None, kind: str, evidence=None) -> dict[str, Any]:
    used = evidence if evidence is not None else event.evidence
    evidence_sequences = [span.sequenceStart for span in used] + [span.sequenceEnd for span in used]
    return {
        "eventId": event.eventId,
        "threadId": event.threadId or (thread.threadId if thread else None),
        "topicId": event.topicId,
        "microBlockId": event.microBlockIds[0] if event.microBlockIds else None,
        "sourceSemanticUnitIds": [event.eventId],
        "semanticArtifactKey": event.eventId,
        "quality": {"grounded": True, "independentlyUseful": True},
        "synthesisSource": "event-pipeline",
        "coverageStatus": kind,
        "validationMetadata": {"uncertainty": event.uncertainty},
        "threadContextEvents": list(thread.eventIds) if thread else [],
        "artifactEvidence": sorted({int(value) for value in evidence_sequences if value is not None}),
        "actionStrength": getattr(event.actionSignal, "actionStrength", None) if event.actionSignal else None,
        "objectGroundingType": getattr(event.actionSignal, "objectGroundingType", None) if event.actionSignal else None,
        "rawActionObject": getattr(event.actionSignal, "rawActionObject", None) if event.actionSignal else None,
        "canonicalActionObject": getattr(event.actionSignal, "canonicalActionObject", None) if event.actionSignal else event.object,
        "actionVerb": getattr(event.actionSignal, "verb", None) if event.actionSignal else None,
        "actionObject": getattr(event.actionSignal, "object", None) if event.actionSignal else event.object,
        "memoryImportance": getattr(event.memorySignal, "importance", None) if event.memorySignal else None,
        "memoryReason": getattr(event.memorySignal, "reason", None) if event.memorySignal else None,
    }
