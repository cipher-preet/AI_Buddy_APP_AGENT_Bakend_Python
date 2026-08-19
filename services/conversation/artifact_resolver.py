from __future__ import annotations

import re
from dataclasses import dataclass

from apps.api_gateway.config.setting import settings
from services.conversation.models import (
    ArtifactHistoryEntry,
    ArtifactLifecycleStatus,
    ArtifactResolutionKind,
    ArtifactType,
    EvidenceSpan,
    MeetingArtifactDocument,
    utc_now,
)


_SPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]+")
STOPWORDS = {
    "the",
    "a",
    "an",
    "to",
    "for",
    "of",
    "and",
    "or",
    "on",
    "in",
    "at",
    "we",
    "let",
    "lets",
    "need",
    "needs",
    "should",
    "must",
    "please",
    "this",
    "that",
    "with",
    "from",
    "into",
    "our",
    "it",
    "is",
    "be",
    "do",
    "then",
    "first",
    "also",
}
ACTION_VERBS = {
    "add",
    "download",
    "build",
    "test",
    "deploy",
    "host",
    "move",
    "check",
    "fix",
    "update",
    "create",
    "configure",
    "install",
    "review",
    "send",
    "call",
    "schedule",
    "prepare",
    "write",
    "document",
    "migrate",
    "replace",
    "remove",
    "delete",
    "verify",
    "validate",
    "release",
    "publish",
    "monitor",
    "investigate",
    "assign",
    "confirm",
    "complete",
    "finish",
    "cancel",
    "start",
}
COMPLETION_VERBS = {"complete", "completed", "done", "finished", "shipped", "closed"}
RELATED_TYPE_GROUPS = (
    {ArtifactType.TASK, ArtifactType.FOLLOW_UP, ArtifactType.COMMITMENT, ArtifactType.DEADLINE},
    {ArtifactType.NOTE, ArtifactType.FACT, ArtifactType.PREFERENCE, ArtifactType.IDEA, ArtifactType.REQUIREMENT, ArtifactType.REFERENCE, ArtifactType.ANSWER},
    {ArtifactType.QUESTION},
    {ArtifactType.RISK, ArtifactType.BLOCKER},
    {ArtifactType.DECISION},
)


@dataclass(frozen=True)
class ResolutionResult:
    incoming: MeetingArtifactDocument
    resolution: ArtifactResolutionKind
    matched: MeetingArtifactDocument | None = None


def resolve_incoming_artifacts(
    existing: list[MeetingArtifactDocument],
    incoming: list[MeetingArtifactDocument],
) -> list[MeetingArtifactDocument]:
    working = [artifact.model_copy(deep=True) for artifact in existing]
    for item in incoming:
        result = resolve_one(working, item)
        applied = apply_resolution(working, result)
        if applied is not None:
            _replace_or_append(working, applied)
    return working


def resolve_one(existing: list[MeetingArtifactDocument], incoming: MeetingArtifactDocument) -> ResolutionResult:
    active = [
        artifact
        for artifact in existing
        if artifact.status not in {ArtifactLifecycleStatus.MERGED, ArtifactLifecycleStatus.REJECTED}
    ]
    exact = next((artifact for artifact in active if artifact.identityKey == incoming.identityKey), None)
    if exact:
        return _resolution_for_existing(exact, incoming)

    best: tuple[float, MeetingArtifactDocument] | None = None
    for artifact in active:
        if not _compatible_types(artifact.artifactType, incoming.artifactType):
            continue
        score = _title_jaccard(artifact.title, incoming.title)
        if best is None or score > best[0]:
            best = (score, artifact)

    if best is None:
        return ResolutionResult(incoming=incoming, resolution=ArtifactResolutionKind.NEW)

    score, matched = best
    incoming_verb = _primary_verb(incoming.title)
    matched_verb = _primary_verb(matched.title)
    object_overlap = _object_jaccard(matched.title, incoming.title)

    if incoming_verb and matched_verb and incoming_verb != matched_verb and score < settings.ARTIFACT_TITLE_JACCARD_DUPLICATE:
        if _is_contradicting_decision(matched, incoming, incoming_verb, matched_verb):
            return ResolutionResult(incoming=incoming, resolution=ArtifactResolutionKind.CONTRADICTION, matched=matched)
        if object_overlap >= 0.35 or score >= 0.35:
            return ResolutionResult(incoming=incoming, resolution=ArtifactResolutionKind.RELATED, matched=matched)
        return ResolutionResult(incoming=incoming, resolution=ArtifactResolutionKind.NEW)

    if score >= settings.ARTIFACT_TITLE_JACCARD_DUPLICATE and _same_scope(matched, incoming):
        return _resolution_for_existing(matched, incoming)

    if score >= settings.ARTIFACT_TITLE_JACCARD_UPDATE:
        return _resolution_for_existing(matched, incoming)

    if object_overlap >= 0.6 and incoming_verb and matched_verb and incoming_verb != matched_verb:
        if _is_contradicting_decision(matched, incoming, incoming_verb, matched_verb):
            return ResolutionResult(incoming=incoming, resolution=ArtifactResolutionKind.CONTRADICTION, matched=matched)
        return ResolutionResult(incoming=incoming, resolution=ArtifactResolutionKind.RELATED, matched=matched)

    return ResolutionResult(incoming=incoming, resolution=ArtifactResolutionKind.NEW)


def apply_resolution(existing: list[MeetingArtifactDocument], result: ResolutionResult) -> MeetingArtifactDocument | None:
    incoming = result.incoming.model_copy(deep=True)
    incoming.resolution = result.resolution
    incoming.updatedAt = utc_now()
    if result.resolution == ArtifactResolutionKind.NEW or result.matched is None:
        incoming.status = incoming.status or ArtifactLifecycleStatus.PROVISIONAL
        return incoming

    matched = result.matched.model_copy(deep=True)
    if result.resolution == ArtifactResolutionKind.RELATED:
        incoming.status = ArtifactLifecycleStatus.PROVISIONAL
        incoming.relatedArtifactIds = _unique([*incoming.relatedArtifactIds, str(matched.id)])
        if _looks_like_parent(matched, incoming):
            incoming.parentArtifactId = incoming.parentArtifactId or str(matched.id)
        matched.relatedArtifactIds = _unique([*matched.relatedArtifactIds, str(incoming.id)])
        _replace_or_append(existing, matched)
        return incoming

    if result.resolution == ArtifactResolutionKind.DUPLICATE:
        matched.sourceWindowIds = _unique([*matched.sourceWindowIds, *incoming.sourceWindowIds])
        matched.sourceChunkIds = sorted(set(matched.sourceChunkIds + incoming.sourceChunkIds))
        matched.evidence = _merge_evidence(matched.evidence, incoming.evidence)
        matched.sourceStartIndex = _min_optional(matched.sourceStartIndex, incoming.sourceStartIndex)
        matched.sourceEndIndex = _max_optional(matched.sourceEndIndex, incoming.sourceEndIndex)
        matched.confidence = max(matched.confidence, incoming.confidence)
        matched.updatedAt = utc_now()
        return matched

    if result.resolution == ArtifactResolutionKind.COMPLETION:
        _append_history(matched, incoming, ArtifactResolutionKind.COMPLETION, matched.content)
        matched.status = ArtifactLifecycleStatus.COMPLETED
        matched.operation = "COMPLETE"
        matched.sourceWindowIds = _unique([*matched.sourceWindowIds, *incoming.sourceWindowIds])
        matched.sourceChunkIds = sorted(set(matched.sourceChunkIds + incoming.sourceChunkIds))
        matched.evidence = _merge_evidence(matched.evidence, incoming.evidence)
        matched.updatedAt = utc_now()
        return matched

    if result.resolution == ArtifactResolutionKind.CONTRADICTION:
        _append_history(matched, incoming, ArtifactResolutionKind.CONTRADICTION, matched.content)
        matched.status = ArtifactLifecycleStatus.SUPERSEDED
        matched.supersededBy = str(incoming.id)
        matched.updatedAt = utc_now()
        incoming.status = ArtifactLifecycleStatus.CONFIRMED
        incoming.supersedes = str(matched.id)
        incoming.reason = incoming.reason or incoming.content or incoming.title
        incoming.relatedArtifactIds = _unique([*incoming.relatedArtifactIds, str(matched.id)])
        _replace_or_append(existing, matched)
        return incoming

    _append_history(matched, incoming, ArtifactResolutionKind.UPDATE, matched.content)
    matched.title = incoming.title or matched.title
    matched.content = _prefer_richer(matched.content, incoming.content)
    matched.ownerText = incoming.ownerText or matched.ownerText
    matched.ownerUserId = incoming.ownerUserId or matched.ownerUserId
    matched.dueDateText = incoming.dueDateText or matched.dueDateText
    matched.dueDateResolved = incoming.dueDateResolved or matched.dueDateResolved
    if incoming.dueDateStatus != "none":
        matched.dueDateStatus = incoming.dueDateStatus
    matched.topic = incoming.topic or matched.topic
    matched.status = ArtifactLifecycleStatus.ACTIVE
    matched.resolution = ArtifactResolutionKind.UPDATE
    matched.sourceWindowIds = _unique([*matched.sourceWindowIds, *incoming.sourceWindowIds])
    matched.sourceChunkIds = sorted(set(matched.sourceChunkIds + incoming.sourceChunkIds))
    matched.evidence = _merge_evidence(matched.evidence, incoming.evidence)
    matched.sourceStartIndex = _min_optional(matched.sourceStartIndex, incoming.sourceStartIndex)
    matched.sourceEndIndex = _max_optional(matched.sourceEndIndex, incoming.sourceEndIndex)
    matched.confidence = max(matched.confidence, incoming.confidence)
    matched.needsConfirmation = matched.needsConfirmation and incoming.needsConfirmation
    matched.updatedAt = utc_now()
    return matched


def item_is_represented(title: str, represented_titles: list[str], strict: bool = False) -> bool:
    normalized = _normalize(title)
    if not normalized:
        return False
    for other in represented_titles:
        if _normalize(other) == normalized:
            return True
        if strict:
            continue
        if _title_jaccard(title, other) >= settings.ARTIFACT_TITLE_JACCARD_DUPLICATE and _primary_verb(title) == _primary_verb(other):
            return True
    return False


def _resolution_for_existing(matched: MeetingArtifactDocument, incoming: MeetingArtifactDocument) -> ResolutionResult:
    if _is_completion(incoming, matched):
        return ResolutionResult(incoming=incoming, resolution=ArtifactResolutionKind.COMPLETION, matched=matched)
    if _is_contradicting_decision(matched, incoming, _primary_verb(incoming.title), _primary_verb(matched.title)):
        return ResolutionResult(incoming=incoming, resolution=ArtifactResolutionKind.CONTRADICTION, matched=matched)
    if _has_material_change(matched, incoming):
        return ResolutionResult(incoming=incoming, resolution=ArtifactResolutionKind.UPDATE, matched=matched)
    return ResolutionResult(incoming=incoming, resolution=ArtifactResolutionKind.DUPLICATE, matched=matched)


def _is_completion(incoming: MeetingArtifactDocument, matched: MeetingArtifactDocument) -> bool:
    if incoming.status == ArtifactLifecycleStatus.COMPLETED or incoming.operation == "COMPLETE":
        return True
    blob = _normalize(f"{incoming.title} {incoming.content}")
    return any(verb in blob.split() for verb in COMPLETION_VERBS) and _title_jaccard(incoming.title, matched.title) >= settings.ARTIFACT_TITLE_JACCARD_UPDATE


def _is_contradicting_decision(
    matched: MeetingArtifactDocument,
    incoming: MeetingArtifactDocument,
    incoming_verb: str | None,
    matched_verb: str | None,
) -> bool:
    if matched.artifactType != ArtifactType.DECISION and incoming.artifactType != ArtifactType.DECISION:
        return False
    if incoming_verb and matched_verb and incoming_verb in {"move", "replace", "migrate", "host"} and incoming_verb != matched_verb:
        return True
    matched_objects = _object_tokens(matched.title + " " + matched.content)
    incoming_objects = _object_tokens(incoming.title + " " + incoming.content)
    shared = matched_objects & incoming_objects
    distinct = (incoming_objects - matched_objects) | (matched_objects - incoming_objects)
    return bool(shared) and bool(distinct) and incoming.content.strip().casefold() != matched.content.strip().casefold()


def _has_material_change(matched: MeetingArtifactDocument, incoming: MeetingArtifactDocument) -> bool:
    if incoming.ownerText and incoming.ownerText != matched.ownerText:
        return True
    if (incoming.dueDateText or incoming.dueDateResolved) and (
        incoming.dueDateText != matched.dueDateText or incoming.dueDateResolved != matched.dueDateResolved
    ):
        return True
    if incoming.content and _normalize(incoming.content) != _normalize(matched.content or ""):
        incoming_tokens = _tokens(incoming.content)
        matched_tokens = _tokens(matched.content or "")
        if incoming_tokens - matched_tokens:
            return True
    if incoming.operation in {"UPDATE", "COMPLETE", "CANCEL"}:
        return True
    return False


def _same_scope(left: MeetingArtifactDocument, right: MeetingArtifactDocument) -> bool:
    if left.ownerText and right.ownerText and _normalize(left.ownerText) != _normalize(right.ownerText):
        return False
    left_due = left.dueDateResolved or left.dueDateText
    right_due = right.dueDateResolved or right.dueDateText
    if left_due and right_due and _normalize(left_due) != _normalize(right_due):
        return False
    return True


def _compatible_types(left: ArtifactType, right: ArtifactType) -> bool:
    if left == right:
        return True
    for group in RELATED_TYPE_GROUPS:
        if left in group and right in group:
            return True
    return False


def _looks_like_parent(possible_parent: MeetingArtifactDocument, child: MeetingArtifactDocument) -> bool:
    parent_tokens = _tokens(possible_parent.title)
    child_tokens = _tokens(child.title)
    if not parent_tokens or not child_tokens:
        return False
    parent_verb = _primary_verb(possible_parent.title)
    child_verb = _primary_verb(child.title)
    if parent_verb in {"fix", "finish", "complete"} and child_verb and child_verb != parent_verb:
        return parent_tokens <= child_tokens or bool(parent_tokens & child_tokens)
    return False


def _title_jaccard(left: str, right: str) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 1.0 if _normalize(left) == _normalize(right) and _normalize(left) else 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _object_jaccard(left: str, right: str) -> float:
    left_tokens = _object_tokens(left)
    right_tokens = _object_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _primary_verb(title: str) -> str | None:
    tokens = _normalize(title).split()
    for token in tokens:
        if token in ACTION_VERBS:
            return token
    return tokens[0] if tokens else None


def _object_tokens(text: str) -> set[str]:
    tokens = _tokens(text)
    return {token for token in tokens if token not in ACTION_VERBS}


def _tokens(text: str) -> set[str]:
    return {token for token in _normalize(text).split() if token and token not in STOPWORDS}


def _normalize(text: str) -> str:
    lowered = _NON_ALNUM_RE.sub(" ", (text or "").casefold())
    return _SPACE_RE.sub(" ", lowered).strip()


def _merge_evidence(left: list[EvidenceSpan], right: list[EvidenceSpan]) -> list[EvidenceSpan]:
    seen: set[str] = set()
    merged: list[EvidenceSpan] = []
    for span in [*left, *right]:
        key = f"{span.sequenceStart}:{span.sequenceEnd}:{_normalize(span.text)[:80]}"
        if key in seen:
            continue
        seen.add(key)
        merged.append(span)
    return merged[:8]


def _append_history(
    matched: MeetingArtifactDocument,
    incoming: MeetingArtifactDocument,
    resolution: ArtifactResolutionKind,
    previous_content: str | None,
) -> None:
    matched.history = [
        *matched.history[-7:],
        ArtifactHistoryEntry(
            at=utc_now(),
            sourceWindowId=str(incoming.sourceWindowId) if incoming.sourceWindowId is not None else None,
            resolution=resolution,
            change=incoming.content or incoming.title,
            previousContent=previous_content,
        ),
    ]


def _prefer_richer(current: str, incoming: str) -> str:
    if not incoming:
        return current
    if not current:
        return incoming
    if len(incoming) >= len(current) or _tokens(incoming) - _tokens(current):
        return incoming
    return current


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _min_optional(left: int | None, right: int | None) -> int | None:
    values = [value for value in (left, right) if value is not None]
    return min(values) if values else None


def _max_optional(left: int | None, right: int | None) -> int | None:
    values = [value for value in (left, right) if value is not None]
    return max(values) if values else None


def _replace_or_append(existing: list[MeetingArtifactDocument], artifact: MeetingArtifactDocument) -> None:
    for index, item in enumerate(existing):
        if str(item.id) == str(artifact.id) or item.identityKey == artifact.identityKey:
            existing[index] = artifact
            return
    existing.append(artifact)
