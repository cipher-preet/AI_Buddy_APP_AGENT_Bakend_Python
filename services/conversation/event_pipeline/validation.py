"""Factual evidence integrity. Never invents a new semantic interpretation."""

from __future__ import annotations

from pydantic import BaseModel, Field

from services.conversation.event_pipeline.channels import is_generic_task_text
from services.conversation.event_pipeline.schemas import AtomicEvent, ValidationAction
from services.conversation.event_pipeline.textutil import casefold_text, content_tokens, evidence_sequence_ids, normalize_text
from services.conversation.models import EvidenceSpan, ExtractedNote, ExtractedTask
from services.llm.router import LLMCapability, LLMRouter


class ValidationResult:
    def __init__(self, action: ValidationAction, item, reasons: list[str] | None = None):
        self.action = action
        self.item = item
        self.reasons = reasons or []


class ArtifactValidationItem(BaseModel):
    key: str
    action: str = "ACCEPT"
    reasons: list[str] = Field(default_factory=list)
    schemaOk: bool = True
    mixedThread: bool = False
    unsupportedDetails: bool = False
    actionSpecific: bool = True


class ArtifactValidationResponse(BaseModel):
    items: list[ArtifactValidationItem] = Field(default_factory=list)


class LLMArtifactValidator:
    def __init__(self, router: LLMRouter):
        self.router = router
        self.calls = 0
        self.requested_capabilities: list[LLMCapability] = []

    async def review(self, item: ExtractedTask | ExtractedNote, events: list[AtomicEvent], artifact_kind: str) -> ValidationResult | None:
        from services.conversation.event_pipeline.llm import compact_event, generate_structured_for_stage
        from services.conversation.event_pipeline.routing import PipelineStage, capability_for_stage

        self.calls += 1
        self.requested_capabilities.append(capability_for_stage(PipelineStage.VALIDATION))
        metadata = getattr(item, "changes", None) or getattr(item, "debug", None) or {}
        source = [event for event in events if event.eventId in set(metadata.get("sourceSemanticUnitIds") or [])]
        payload = {
            "kind": artifact_kind,
            "title": item.title,
            "body": item.body,
            "evidence": [span.model_dump() for span in (item.evidence or [])],
            "events": [compact_event(event) for event in source[:4]],
        }
        try:
            response, _, _ = await generate_structured_for_stage(
                self.router,
                PipelineStage.VALIDATION,
                "event-artifact-validator-v1",
                ArtifactValidationResponse,
                payload,
            )
        except Exception as error:
            from services.llm.async_runtime import reraise_if_hard_runtime
            from services.conversation.event_pipeline.observability import record_failure

            record_failure(error)
            reraise_if_hard_runtime(error)
            return None
        verdict = next((row for row in response.items if row.key in {item.title, artifact_kind, "item"}), response.items[0] if response.items else None)
        if verdict is None:
            return None
        if (verdict.mixedThread or not verdict.schemaOk) and str(verdict.action).upper() != "ACCEPT":
            return ValidationResult(ValidationAction.REJECT, item, verdict.reasons or ["llm_validation_rejected"])
        if str(verdict.action).upper() == "REJECT":
            return ValidationResult(ValidationAction.REJECT, item, verdict.reasons or ["llm_validation_rejected"])
        if str(verdict.action).upper() == "REWRITE":
            rewritten = _rewrite_from_events(item, source)
            if rewritten is None:
                return ValidationResult(ValidationAction.REJECT, item, verdict.reasons or ["llm_validation_rewrite_failed"])
            return ValidationResult(ValidationAction.REWRITE_FROM_EXISTING_EVENTS, rewritten, verdict.reasons)
        return None



WEAK_EVIDENCE_TOKENS = frozenset(
    {
        "server",
        "connection",
        "issue",
        "problem",
        "task",
        "work",
        "id",
        "network",
        "setup",
        "pending",
        "tracking",
        "port",
        "string",
    }
)


def validate_artifact(
    item: ExtractedTask | ExtractedNote,
    sequence_text: dict[int, str],
    events: list[AtomicEvent],
    *,
    artifact_kind: str,
) -> ValidationResult:
    metadata = getattr(item, "changes", None) or getattr(item, "debug", None) or {}
    event_ids = list(metadata.get("sourceSemanticUnitIds") or [])
    source_events = [event for event in events if event.eventId in event_ids] if event_ids else _events_for_evidence(item, events)
    reasons: list[str] = []
    evidence = list(item.evidence or [])
    if not evidence:
        return ValidationResult(ValidationAction.REJECT, item, ["missing_evidence"])

    kept, removed = _filter_evidence(evidence, sequence_text, item, source_events)
    if removed:
        reasons.append("unrelated_or_ungrounded_evidence")
        item.evidence = kept
    if not item.evidence:
        return ValidationResult(ValidationAction.REJECT, item, reasons + ["insufficient_evidence"])

    thread_ids = {event.threadId for event in source_events if event.threadId}
    if len(thread_ids) > 1:
        reasons.append("mixed_thread_evidence")
        return ValidationResult(ValidationAction.REJECT, item, reasons)

    if artifact_kind == "task":
        if is_generic_task_text(item.title, item.body, getattr(item, "ownerText", None)):
            return ValidationResult(ValidationAction.REJECT, item, reasons + ["generic_task"])
        _strip_ungrounded_optional_fields(item)
    if artifact_kind == "note" and _certainty_inversion(item, source_events):
        rewritten = _rewrite_from_events(item, source_events)
        if rewritten is None:
            return ValidationResult(ValidationAction.REJECT, item, reasons + ["certainty_inversion"])
        return ValidationResult(ValidationAction.REWRITE_FROM_EXISTING_EVENTS, rewritten, reasons + ["certainty_inversion"])
    if _introduces_unsupported_details(item, item.evidence):
        reasons.append("unsupported_details")
        rewritten = _rewrite_from_events(item, source_events)
        if rewritten is None:
            return ValidationResult(ValidationAction.REJECT, item, reasons)
        return ValidationResult(ValidationAction.REWRITE_FROM_EXISTING_EVENTS, rewritten, reasons)
    _sync_artifact_evidence_metadata(item)
    if removed:
        return ValidationResult(ValidationAction.REMOVE_BAD_EVIDENCE, item, reasons)
    return ValidationResult(ValidationAction.ACCEPT, item, reasons)


def mixed_thread_rate(items: list, events: list[AtomicEvent]) -> float:
    if not items:
        return 0.0
    mixed = 0
    for item in items:
        metadata = getattr(item, "changes", None) or getattr(item, "debug", None) or {}
        event_ids = set(metadata.get("sourceSemanticUnitIds") or [])
        source = [event for event in events if event.eventId in event_ids]
        source_threads = {event.threadId for event in source if event.threadId}
        if len(source_threads) > 1:
            mixed += 1
            continue
        artifact_sequences = set(metadata.get("artifactEvidence") or evidence_sequence_ids(getattr(item, "evidence", [])))
        source_sequences = {sequence for event in source for sequence in (event.sequenceIds or [])}
        foreign_sequences = artifact_sequences - source_sequences
        if not foreign_sequences:
            continue
        other_threads = {
            event.threadId
            for event in events
            if event.threadId and event.eventId not in event_ids and foreign_sequences & set(event.sequenceIds or [])
        }
        if other_threads - source_threads:
            mixed += 1
    return mixed / len(items)


def _filter_evidence(
    evidence: list[EvidenceSpan],
    sequence_text: dict[int, str],
    item,
    source_events: list[AtomicEvent] | None = None,
) -> tuple[list[EvidenceSpan], list[EvidenceSpan]]:
    kept: list[EvidenceSpan] = []
    removed: list[EvidenceSpan] = []
    claim = casefold_text(f"{getattr(item, 'title', '')} {getattr(item, 'body', '')}")
    source_sequences: set[int] = set()
    for event in source_events or []:
        source_sequences.update(event.sequenceIds or evidence_sequence_ids(event.evidence))
    claim_tokens = set(token.casefold() for token in content_tokens(claim))
    for span in evidence:
        span_sequences = set(range(int(span.sequenceStart), int(span.sequenceEnd) + 1))
        combined = " ".join(sequence_text.get(sequence, "") for sequence in sorted(span_sequences)).strip()
        if not combined:
            removed.append(span)
            continue
        if source_sequences and not (span_sequences & source_sequences):
            removed.append(span)
            continue
        if casefold_text(span.text) not in casefold_text(combined) and casefold_text(combined) not in casefold_text(span.text):
            if not _token_overlap(span.text, combined):
                removed.append(span)
                continue
        span_tokens = set(token.casefold() for token in content_tokens(span.text + " " + combined))
        if not _span_supports_artifact(claim_tokens, span_tokens, item):
            removed.append(span)
            continue
        kept.append(span)
    return kept, removed


def _span_supports_artifact(claim_tokens: set[str], span_tokens: set[str], item) -> bool:
    if not claim_tokens:
        return False
    overlap = claim_tokens & span_tokens
    if not overlap:
        return False
    distinctive = {token for token in overlap if token not in WEAK_EVIDENCE_TOKENS and len(token) > 2}
    if distinctive:
        return True
    object_tokens = set(token.casefold() for token in content_tokens(getattr(item, "title", "") or ""))
    object_tokens -= WEAK_EVIDENCE_TOKENS
    if object_tokens and object_tokens & span_tokens:
        return True
    return False


def _sync_artifact_evidence_metadata(item) -> None:
    metadata = getattr(item, "changes", None) or getattr(item, "debug", None)
    if not isinstance(metadata, dict):
        return
    metadata["artifactEvidence"] = evidence_sequence_ids(getattr(item, "evidence", []))
    metadata.setdefault("threadContextEvents", [])


def _introduces_unsupported_details(item, evidence: list[EvidenceSpan]) -> bool:
    blob = casefold_text(" ".join(span.text for span in evidence))
    extras = []
    for field in ("title", "body"):
        text = normalize_text(getattr(item, field, "") or "")
        for token in content_tokens(text):
            if len(token) <= 3:
                continue
            if token.casefold() not in blob and token.casefold() not in casefold_text(text[:20]):
                extras.append(token)
    return len(extras) >= 4


def _rewrite_from_events(item, events: list[AtomicEvent]):
    if not events:
        return None
    event = events[0]
    item.title = event.object or event.meaning[:80]
    item.body = event.meaning
    item.evidence = list(event.evidence)
    return item


_UNRESOLVED_KINDS = frozenset({"OPEN_QUESTION", "ISSUE", "PROPOSAL"})
_UNRESOLVED_REASONS = frozenset({"OPEN_QUESTION", "QUESTION", "OPEN_DECISION", "ISSUE", "PROPOSAL"})


def _certainty_inversion(item, source_events: list[AtomicEvent]) -> bool:
    """Reject notes that turn an unresolved question/issue into a confirmed fact."""
    if not source_events:
        return "?" in f"{getattr(item, 'title', '')} {getattr(item, 'body', '')}" and _note_asserts_settled_fact(item)
    event = source_events[0]
    kind = getattr(event.kind, "value", str(event.kind))
    reason = str(getattr(getattr(event, "memorySignal", None), "reason", "") or "").strip().upper()
    uncertainty = {str(flag).strip().upper() for flag in (getattr(event, "uncertainty", None) or [])}
    evidence_blob = " ".join(span.text for span in (getattr(event, "evidence", None) or []))
    source_unresolved = (
        kind in _UNRESOLVED_KINDS
        or reason in _UNRESOLVED_REASONS
        or bool(uncertainty & {"UNRESOLVED_QUESTION", "AMBIGUOUS", "LOW_CONFIDENCE_SOURCE", "NOISE"})
        or "?" in (event.meaning or "")
        or "?" in evidence_blob
    )
    if not source_unresolved:
        return False
    return _note_asserts_settled_fact(item, event)


def _note_asserts_settled_fact(item, event=None) -> bool:
    body = casefold_text(getattr(item, "body", "") or "")
    title = casefold_text(getattr(item, "title", "") or "")
    claim = f"{title} {body}"
    if "?" in claim:
        return False
    meaning = casefold_text(getattr(event, "meaning", "") or "") if event is not None else ""
    if meaning and claim.strip() == meaning.strip():
        return False
    return bool(claim.strip()) and (not meaning or meaning not in claim)


def _strip_ungrounded_optional_fields(task: ExtractedTask) -> None:
    blob = casefold_text(" ".join(span.text for span in task.evidence))
    if task.ownerText and casefold_text(task.ownerText) not in blob:
        task.ownerText = None
        task.ownerUserId = None
    if task.dueDateText and casefold_text(task.dueDateText) not in blob:
        task.dueDateText = None
        task.dueDateResolved = None
        task.dueDateStatus = "none"


def _events_for_evidence(item, events: list[AtomicEvent]) -> list[AtomicEvent]:
    sequences = set(evidence_sequence_ids(getattr(item, "evidence", [])))
    return [event for event in events if sequences & set(event.sequenceIds)]


def _token_overlap(left: str, right: str) -> bool:
    left_tokens = set(token.casefold() for token in content_tokens(left))
    right_tokens = set(token.casefold() for token in content_tokens(right))
    if not left_tokens or not right_tokens:
        return False
    return len(left_tokens & right_tokens) >= max(1, min(3, len(left_tokens) // 2))
