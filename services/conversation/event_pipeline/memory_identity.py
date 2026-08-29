"""Canonical memory identity for Notes. Evidence overlap is not identity.

Same paraphrase → DUPLICATE.
New status on the same subject → UPDATE (keep both).
Invalidated prior state → SUPERSEDE.
Same subject, different useful fact → DISTINCT.
"""

from __future__ import annotations

from services.conversation.event_pipeline.schemas import AtomicEvent, EventKind, MemoryDisposition
from services.conversation.event_pipeline.textutil import casefold_text, content_tokens, token_jaccard

MEMORY_IMPORTANCE = frozenset({"LOW", "MEDIUM", "HIGH"})
NOTE_RELATIONS = frozenset({"UPDATE", "SUPERSEDE", "RELATED", "DUPLICATE", "DISTINCT"})
MEMORY_KINDS = frozenset(
    {
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
    }
)
_SUBJECT_STOP = frozenset({"issue", "problem", "status", "update", "the", "a", "an"})
_UPDATE_MARKERS = frozenset(
    {
        "still",
        "again",
        "after",
        "now",
        "yet",
        "already",
        "changed",
        "updated",
        "remaining",
        "anymore",
        "longer",
        "phir",
        "wapas",
        "abhi",
    }
)


def memory_importance(event: AtomicEvent) -> str:
    signal = event.memorySignal
    if signal is None:
        return "LOW"
    raw = str(getattr(signal, "importance", None) or "").strip().upper()
    if raw in MEMORY_IMPORTANCE:
        return raw
    if getattr(signal, "isMemoryWorthy", False):
        return "MEDIUM"
    return "LOW"


def event_is_memory_candidate(event: AtomicEvent) -> bool:
    if event.kind == EventKind.NOISE:
        return False
    if event.memorySignal is not None:
        return True
    return event.kind in MEMORY_KINDS


def event_is_publishable_memory(event: AtomicEvent) -> bool:
    if not event_is_memory_candidate(event):
        return False
    signal = event.memorySignal
    if signal is not None:
        if not bool(signal.isMemoryWorthy):
            return False
        return memory_importance(event) != "LOW"
    return event.kind in MEMORY_KINDS


def memory_subject_key(event: AtomicEvent) -> str:
    obj = casefold_text(event.object or "")
    if obj:
        tokens = [token for token in content_tokens(obj) if token.casefold() not in _SUBJECT_STOP]
        if tokens:
            return " ".join(sorted({token.casefold() for token in tokens}))
    entities = [item.casefold() for item in (event.entities or []) if item]
    if entities:
        return " ".join(sorted(set(entities))[:4])
    tokens = [token.casefold() for token in content_tokens(event.meaning) if token.casefold() not in _SUBJECT_STOP]
    return " ".join(sorted(tokens)[:6])


def memory_state_key(event: AtomicEvent) -> str:
    kind = event.kind.value if hasattr(event.kind, "value") else str(event.kind)
    tokens = " ".join(sorted({token.casefold() for token in content_tokens(event.meaning)}))
    return f"{kind}|{tokens}"


def memory_identity_key(event: AtomicEvent) -> str:
    thread = event.threadId or ""
    return f"{thread}|{memory_subject_key(event)}|{memory_state_key(event)}"


def memory_relation(incoming: AtomicEvent, existing: AtomicEvent) -> str | None:
    if incoming.eventId == existing.eventId:
        return "DUPLICATE"
    if not _same_subject(incoming, existing):
        return None
    similarity = token_jaccard(incoming.meaning, existing.meaning)
    incoming_tokens = _meaning_tokens(incoming)
    existing_tokens = _meaning_tokens(existing)
    extra_in = incoming_tokens - existing_tokens
    extra_ex = existing_tokens - incoming_tokens
    paraphrase = _is_paraphrase(incoming_tokens, existing_tokens, extra_in, extra_ex, similarity)
    if paraphrase and not ((extra_in | extra_ex) & _UPDATE_MARKERS):
        return "DUPLICATE"
    if (extra_in | extra_ex) & _UPDATE_MARKERS:
        if incoming.kind in {EventKind.RESULT, EventKind.COMPLETION} and existing.kind in {EventKind.ISSUE, EventKind.STATE}:
            return "SUPERSEDE"
        return "UPDATE"
    if incoming.kind in {EventKind.RESULT, EventKind.COMPLETION} and existing.kind in {EventKind.ISSUE, EventKind.STATE}:
        return "SUPERSEDE"
    if incoming.kind == EventKind.CONTRADICTION or existing.kind == EventKind.CONTRADICTION:
        return "SUPERSEDE"
    if incoming.kind != existing.kind:
        return "UPDATE"
    if extra_in and extra_ex:
        return "DISTINCT"
    if extra_in or extra_ex:
        distinctive = {token for token in (extra_in | extra_ex) if len(token) > 3 and token not in _UPDATE_MARKERS}
        if distinctive and not ((extra_in | extra_ex) & _UPDATE_MARKERS):
            return "DISTINCT"
        return "UPDATE"
    same_thread = bool(incoming.threadId and incoming.threadId == existing.threadId)
    if same_thread:
        return "RELATED"
    return "DISTINCT"


def find_memory_relation(incoming: AtomicEvent, published: list[AtomicEvent]) -> tuple[str | None, AtomicEvent | None]:
    best: tuple[float, str, AtomicEvent] | None = None
    for other in published:
        relation = memory_relation(incoming, other)
        if relation is None:
            continue
        score = token_jaccard(incoming.meaning, other.meaning) + (
            0.2 if incoming.threadId and incoming.threadId == other.threadId else 0.0
        )
        if best is None or score > best[0]:
            best = (score, relation, other)
    if best is None:
        return None, None
    return best[1], best[2]


def set_memory_disposition(event: AtomicEvent, disposition: MemoryDisposition, reason: str = "") -> None:
    event.memoryDisposition = disposition
    event.memoryDispositionReason = reason or disposition.value


def _meaning_tokens(event: AtomicEvent) -> set[str]:
    return {token.casefold() for token in content_tokens(event.meaning)}


def _is_paraphrase(
    left: set[str],
    right: set[str],
    extra_left: set[str],
    extra_right: set[str],
    similarity: float,
) -> bool:
    if not left or not right:
        return False
    if (extra_left | extra_right) & _UPDATE_MARKERS:
        return False
    if similarity >= 0.85:
        return True
    if not extra_left and not extra_right:
        return True
    # One meaning is a wording expansion of the other (no new content tokens of length > 3 beyond stop-like fillers).
    if not extra_left and extra_right and similarity >= 0.45:
        distinctive = {token for token in extra_right if len(token) > 3}
        return len(distinctive) <= 2
    if not extra_right and extra_left and similarity >= 0.45:
        distinctive = {token for token in extra_left if len(token) > 3}
        return len(distinctive) <= 2
    return False


def _same_subject(left: AtomicEvent, right: AtomicEvent) -> bool:
    left_key = memory_subject_key(left)
    right_key = memory_subject_key(right)
    if left_key and left_key == right_key:
        return True
    left_tokens = set(left_key.split())
    right_tokens = set(right_key.split())
    if left_tokens and right_tokens and (left_tokens & right_tokens):
        shared = left_tokens & right_tokens
        return len(shared) >= max(1, min(len(left_tokens), len(right_tokens)) - 1)
    return False
