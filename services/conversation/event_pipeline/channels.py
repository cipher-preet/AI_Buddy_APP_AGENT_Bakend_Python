"""Independent action and memory channels. Issues are not tasks."""

from __future__ import annotations

from services.conversation.event_pipeline.schemas import (
    ABSTAIN_UNRESOLVED_OBJECT,
    ACTION_EVENT_KINDS,
    DEICTIC_OR_TIME,
    EventKind,
    GENERIC_ACTION_OBJECTS,
    GENERIC_TASK_TITLES,
    MEMORY_EVENT_KINDS,
    NON_PUBLISHABLE_KINDS,
    AtomicEvent,
    EventDisposition,
)
from services.conversation.event_pipeline.textutil import casefold_text, content_tokens, normalize_text


GENERIC_ACTION_VERBS = frozenset(
    {
        "complete",
        "pending",
        "fix",
        "handle",
        "check",
        "do",
        "resolve",
        "track",
        "update",
        "follow",
        "review",
        "look",
        "see",
        "karo",
        "karna",
        "kardo",
    }
)


def split_action_and_memory(events: list[AtomicEvent]) -> tuple[list[AtomicEvent], list[AtomicEvent], list[AtomicEvent]]:
    from services.conversation.event_pipeline.memory_identity import event_is_memory_candidate

    actions: list[AtomicEvent] = []
    memory: list[AtomicEvent] = []
    other: list[AtomicEvent] = []
    for event in events:
        if event.kind in NON_PUBLISHABLE_KINDS:
            event.channel = "other"
            event.disposition = EventDisposition.INTENTIONALLY_NON_PUBLISHABLE
            event.dispositionReason = event.dispositionReason or "noise"
            other.append(event)
            continue
        task_eligible = event_is_task_eligible(event)
        actionable = event_is_actionable(event)
        memory_candidate = event_is_memory_candidate(event)
        if actionable:
            event.channel = "action"
            if not action_object_grounded(event):
                event.uncertainty = list(event.uncertainty or [])
                if "missing_object" not in event.uncertainty:
                    event.uncertainty.append("missing_object")
            actions.append(event)
        if memory_candidate:
            if not actionable:
                event.channel = "memory"
            memory.append(event)
        if not actionable and not memory_candidate:
            event.channel = "other"
            other.append(event)
    return actions, memory, other


def event_is_actionable(event: AtomicEvent) -> bool:
    signal = event.actionSignal
    if signal is not None:
        return bool(signal.isActionable) and action_strength(signal) == "EXPLICIT"
    return event.kind in ACTION_EVENT_KINDS


def event_is_task_eligible(event: AtomicEvent) -> bool:
    """Only EXPLICIT action strength may publish a Task."""
    if event.kind == EventKind.PROPOSAL and action_strength(event.actionSignal) != "EXPLICIT":
        return False
    if not event_is_actionable(event):
        return False
    grounding = object_grounding_type(event)
    if grounding in {"INFERRED", "UNRESOLVED"}:
        return False
    return True


def action_strength(signal) -> str:
    if signal is None:
        return "NONE"
    raw = str(getattr(signal, "actionStrength", None) or "").strip().upper()
    if raw in {"NONE", "POSSIBLE", "EXPLICIT"}:
        return raw
    if getattr(signal, "isActionable", False):
        return "EXPLICIT"
    return "NONE"


def object_grounding_type(event: AtomicEvent) -> str | None:
    if event.actionSignal and event.actionSignal.objectGroundingType:
        return str(event.actionSignal.objectGroundingType).strip().upper()
    if event.actionSignal and event.actionSignal.artifactStatus == ABSTAIN_UNRESOLVED_OBJECT:
        return "UNRESOLVED"
    if event.object or (event.actionSignal and event.actionSignal.object):
        return "EXPLICIT"
    return None


def event_is_memory_worthy(event: AtomicEvent) -> bool:
    from services.conversation.event_pipeline.memory_identity import event_is_publishable_memory

    return event_is_publishable_memory(event)


def action_object_text(event: AtomicEvent) -> str | None:
    if event.object:
        return normalize_text(event.object)
    if event.actionSignal and event.actionSignal.object:
        return normalize_text(event.actionSignal.object)
    return None


def action_object_grounded(event: AtomicEvent) -> bool:
    """Structurally specific action object, not an English-title blacklist."""
    grounding = object_grounding_type(event)
    if grounding in {"INFERRED", "UNRESOLVED"}:
        return False
    if event.actionSignal and event.actionSignal.artifactStatus == ABSTAIN_UNRESOLVED_OBJECT:
        return False
    if "missing_object" in (event.uncertainty or []):
        obj = action_object_text(event)
        if not obj or casefold_text(obj) in GENERIC_ACTION_OBJECTS:
            return False
    obj = action_object_text(event)
    if obj and casefold_text(obj) not in GENERIC_ACTION_OBJECTS and not is_structurally_generic(obj):
        return True
    evidence_blob = " ".join(span.text for span in event.evidence)
    distinctive = _distinctive_tokens(f"{event.meaning} {evidence_blob} {obj or ''}")
    if not distinctive:
        return False
    if is_structurally_generic(event.meaning) and not obj:
        return False
    return bool(event.entities or event.evidence)


def is_generic_task_text(title: str, body: str = "", obj: str | None = None) -> bool:
    blob = casefold_text(f"{title} {body}")
    if casefold_text(title) in GENERIC_TASK_TITLES or blob.strip() in GENERIC_TASK_TITLES:
        return True
    if is_structurally_generic(title) and is_structurally_generic(body or title):
        return True
    if obj and casefold_text(obj) in GENERIC_ACTION_OBJECTS and not _distinctive_tokens(f"{title} {body}"):
        return True
    return False


def is_structurally_generic(text: str | None) -> bool:
    tokens = [token.casefold() for token in content_tokens(text)]
    if not tokens:
        return True
    generic = GENERIC_ACTION_OBJECTS | DEICTIC_OR_TIME | GENERIC_ACTION_VERBS | {"the", "a", "an", "please"}
    distinctive = [token for token in tokens if token not in generic]
    return len(distinctive) == 0


def unresolved_action_object(event: AtomicEvent) -> bool:
    if event.actionSignal and event.actionSignal.artifactStatus == ABSTAIN_UNRESOLVED_OBJECT:
        return True
    return not action_object_grounded(event) or "missing_object" in (event.uncertainty or [])


def set_action_disposition(event: AtomicEvent, disposition, reason: str = "") -> None:
    from services.conversation.event_pipeline.schemas import ActionDisposition, EventDisposition

    event.actionDisposition = disposition
    event.actionDispositionReason = reason or getattr(disposition, "value", str(disposition))
    if disposition == ActionDisposition.PUBLISHED_TASK:
        event.disposition = EventDisposition.TASK
    elif disposition == ActionDisposition.DUPLICATE:
        event.disposition = EventDisposition.DUPLICATE
    elif disposition == ActionDisposition.SUPERSEDED:
        event.disposition = EventDisposition.SUPERSEDED
    elif disposition == ActionDisposition.VALIDATION_REJECTED:
        event.disposition = EventDisposition.REJECTED
        event.dispositionReason = reason or "validation_rejected"
    elif disposition in {
        ActionDisposition.INTENTIONALLY_NONPUBLISHABLE,
        ActionDisposition.UNSUPPORTED,
        ActionDisposition.UNRESOLVED_OBJECT,
        ActionDisposition.AMBIGUOUS,
    }:
        if event.disposition not in {
            EventDisposition.TASK,
            EventDisposition.NOTE,
            EventDisposition.DUPLICATE,
            EventDisposition.SUPERSEDED,
            EventDisposition.REJECTED,
        }:
            event.disposition = EventDisposition.INTENTIONALLY_NON_PUBLISHABLE
            event.dispositionReason = reason or event.dispositionReason or disposition.value


def action_synthesis_abstain_disposition(event: AtomicEvent):
    from services.conversation.event_pipeline.schemas import ActionDisposition

    if event.kind == EventKind.NOISE:
        return ActionDisposition.INTENTIONALLY_NONPUBLISHABLE, "noise"
    if unresolved_action_object(event) or object_grounding_type(event) in {"INFERRED", "UNRESOLVED"}:
        return ActionDisposition.UNRESOLVED_OBJECT, "unresolved_object"
    if "AMBIGUOUS" in (event.uncertainty or []):
        return ActionDisposition.AMBIGUOUS, "ambiguous_action"
    if not event_is_task_eligible(event):
        return ActionDisposition.INTENTIONALLY_NONPUBLISHABLE, "not_task_eligible"
    return ActionDisposition.UNSUPPORTED, "generic_or_ungrounded_action"


def _distinctive_tokens(text: str | None) -> list[str]:
    generic = GENERIC_ACTION_OBJECTS | DEICTIC_OR_TIME | GENERIC_ACTION_VERBS | {"the", "a", "an", "please"}
    return [token.casefold() for token in content_tokens(text) if token.casefold() not in generic]
