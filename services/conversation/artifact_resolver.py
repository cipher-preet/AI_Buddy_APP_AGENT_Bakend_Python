from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from apps.api_gateway.config.setting import settings
from services.conversation.models import (
    ArtifactHistoryEntry,
    ArtifactLifecycleStatus,
    ArtifactReconcileDecision,
    ArtifactResolutionKind,
    EvidenceSpan,
    MeetingArtifactDocument,
    ReconcileAction,
    utc_now,
)


_SPACE_RE = re.compile(r"\s+")
_TERMINAL_STATUSES = {
    ArtifactLifecycleStatus.MERGED,
    ArtifactLifecycleStatus.REJECTED,
}
_INACTIVE_STATUSES = {
    ArtifactLifecycleStatus.COMPLETED,
    ArtifactLifecycleStatus.CANCELLED,
    ArtifactLifecycleStatus.SUPERSEDED,
    ArtifactLifecycleStatus.MERGED,
    ArtifactLifecycleStatus.REJECTED,
}
_MODIFYING_ACTIONS = {
    ReconcileAction.UPDATE_EXISTING,
    ReconcileAction.COMPLETE_EXISTING,
    ReconcileAction.CANCEL_EXISTING,
    ReconcileAction.SUPERSEDE_EXISTING,
}


@dataclass(frozen=True)
class ResolutionResult:
    incoming: MeetingArtifactDocument
    resolution: ArtifactResolutionKind
    matched: MeetingArtifactDocument | None = None
    reconcileAction: ReconcileAction | None = None


def resolve_incoming_artifacts(
    existing: list[MeetingArtifactDocument],
    incoming: list[MeetingArtifactDocument],
    decisions: list[ArtifactReconcileDecision] | None = None,
) -> list[MeetingArtifactDocument]:
    """Apply explicit reconcile decisions. semanticHint is never identity."""
    if not incoming:
        return [_copy_artifact(artifact) for artifact in existing]
    resolved_decisions = decisions or [
        ArtifactReconcileDecision(incomingIndex=index, action=ReconcileAction.CREATE_NEW)
        for index, _ in enumerate(incoming)
    ]
    return apply_llm_decisions(existing, incoming, resolved_decisions)


async def reconcile_incoming_artifacts(
    router,
    existing: list[MeetingArtifactDocument],
    incoming: list[MeetingArtifactDocument],
    window_text: str,
) -> list[MeetingArtifactDocument]:
    if not incoming:
        return [_copy_artifact(artifact) for artifact in existing]
    if not existing:
        return resolve_incoming_artifacts(existing, incoming)
    from services.conversation import agents

    candidates = retrieve_relevant_artifacts(existing, incoming)
    try:
        response = await agents.reconcile_artifacts(router, candidates, incoming, window_text)
        invalid = invalid_modifying_decisions(response.decisions, incoming, existing)
        if invalid:
            response = await agents.reconcile_artifacts(
                router,
                candidates,
                incoming,
                window_text,
                repair={
                    "reason": "modifying actions require a valid targetArtifactId from validTargetArtifactIds",
                    "invalidDecisions": [item.model_dump() for item in invalid],
                    "validTargetArtifactIds": [str(item.id) for item in candidates],
                },
            )
    except Exception as error:
        print(
            "Artifact reconciliation fell back to CREATE_NEW after LLM failure:",
            {"incoming": len(incoming), "existing": len(existing), "error": str(error)[:500]},
        )
        return resolve_incoming_artifacts(existing, incoming)
    return resolve_incoming_artifacts(existing, incoming, response.decisions)


def retrieve_relevant_artifacts(
    existing: list[MeetingArtifactDocument],
    incoming: list[MeetingArtifactDocument],
    limit: int | None = None,
) -> list[MeetingArtifactDocument]:
    """Small active/session set. semanticHint is a retrieval boost, not a merge key."""
    cap = limit or settings.MEETING_MEMORY_RETRIEVAL_LIMIT
    selected: list[MeetingArtifactDocument] = []
    seen: set[str] = set()

    def _add(items: list[MeetingArtifactDocument]) -> bool:
        for artifact in items:
            key = str(artifact.id)
            if key in seen:
                continue
            seen.add(key)
            selected.append(artifact)
            if len(selected) >= cap:
                return True
        return False

    hints = {(item.semanticHint or "").strip() for item in incoming if (item.semanticHint or "").strip()}
    types = {item.artifactType for item in incoming}
    active = [artifact for artifact in existing if artifact.status not in _TERMINAL_STATUSES]
    unresolved = [artifact for artifact in active if artifact.status not in _INACTIVE_STATUSES]
    hinted = [artifact for artifact in active if (artifact.semanticHint or "").strip() in hints]
    same_type = [artifact for artifact in unresolved if artifact.artifactType in types]
    if _add(hinted):
        return selected
    if _add(list(reversed(same_type))):
        return selected
    if _add(list(reversed(unresolved))):
        return selected
    _add(list(reversed(active)))
    return selected


def apply_llm_decisions(
    existing: list[MeetingArtifactDocument],
    incoming: list[MeetingArtifactDocument],
    decisions: list[ArtifactReconcileDecision],
) -> list[MeetingArtifactDocument]:
    working = [_copy_artifact(artifact) for artifact in existing]
    by_index = {decision.incomingIndex: decision for decision in decisions}
    for index, item in enumerate(incoming):
        decision = by_index.get(index)
        action = decision.action if decision else ReconcileAction.CREATE_NEW
        target_id = decision.targetArtifactId if decision else None
        matched = _find_by_id(working, target_id) if target_id else None
        if action in _MODIFYING_ACTIONS and matched is None:
            print(
                "Skipping incoming unit; modifying action is missing a valid targetArtifactId:",
                {
                    "incomingIndex": index,
                    "action": action.value,
                    "targetArtifactId": target_id,
                },
            )
            continue
        if action == ReconcileAction.RELATED_BUT_DISTINCT and matched is None:
            action = ReconcileAction.CREATE_NEW
        incoming_item = _copy_artifact(item)
        if decision and decision.evidence:
            incoming_item.evidence = _merge_evidence(incoming_item.evidence, decision.evidence)
        result = _resolution_from_action(action, incoming_item, matched)
        applied = apply_resolution(working, result)
        if applied is not None:
            _replace_or_append(working, applied)
    return working


def resolve_one(existing: list[MeetingArtifactDocument], incoming: MeetingArtifactDocument) -> ResolutionResult:
    # Identity is decided by the reconciliation model with an explicit target
    # artifact ID. Keys, titles, and token overlap are not authoritative.
    return ResolutionResult(incoming=incoming, resolution=ArtifactResolutionKind.NEW)


def apply_resolution(existing: list[MeetingArtifactDocument], result: ResolutionResult) -> MeetingArtifactDocument | None:
    incoming = _copy_artifact(result.incoming)
    incoming.resolution = result.resolution
    incoming.updatedAt = utc_now()
    action_name = result.reconcileAction.value if result.reconcileAction else None
    if result.resolution == ArtifactResolutionKind.NEW or result.matched is None:
        incoming.status = incoming.status or ArtifactLifecycleStatus.PROVISIONAL
        return incoming

    matched = _copy_artifact(result.matched)
    if incoming.status in {ArtifactLifecycleStatus.CANCELLED, ArtifactLifecycleStatus.REJECTED} or incoming.operation == "CANCEL":
        _append_history(matched, incoming, ArtifactResolutionKind.CONTRADICTION, matched.content, action_name)
        matched.status = ArtifactLifecycleStatus.CANCELLED
        matched.operation = "CANCEL"
        matched.sourceWindowIds = _unique([*matched.sourceWindowIds, *incoming.sourceWindowIds])
        matched.sourceChunkIds = sorted(set(matched.sourceChunkIds + incoming.sourceChunkIds))
        matched.evidence = _merge_evidence(matched.evidence, incoming.evidence)
        matched.updatedAt = utc_now()
        return matched

    if result.resolution == ArtifactResolutionKind.RELATED:
        incoming.status = ArtifactLifecycleStatus.PROVISIONAL
        incoming.relatedArtifactIds = _unique([*incoming.relatedArtifactIds, str(matched.id)])
        matched.relatedArtifactIds = _unique([*matched.relatedArtifactIds, str(incoming.id)])
        _replace_or_append(existing, matched)
        return incoming

    if result.resolution == ArtifactResolutionKind.DUPLICATE:
        _append_history(matched, incoming, ArtifactResolutionKind.DUPLICATE, matched.content, action_name)
        matched.sourceWindowIds = _unique([*matched.sourceWindowIds, *incoming.sourceWindowIds])
        matched.sourceChunkIds = sorted(set(matched.sourceChunkIds + incoming.sourceChunkIds))
        matched.evidence = _merge_evidence(matched.evidence, incoming.evidence)
        matched.sourceStartIndex = _min_optional(matched.sourceStartIndex, incoming.sourceStartIndex)
        matched.sourceEndIndex = _max_optional(matched.sourceEndIndex, incoming.sourceEndIndex)
        matched.confidence = max(matched.confidence, incoming.confidence)
        matched.updatedAt = utc_now()
        return matched

    if result.resolution == ArtifactResolutionKind.COMPLETION:
        _append_history(matched, incoming, ArtifactResolutionKind.COMPLETION, matched.content, action_name)
        matched.status = ArtifactLifecycleStatus.COMPLETED
        matched.operation = "COMPLETE"
        matched.sourceWindowIds = _unique([*matched.sourceWindowIds, *incoming.sourceWindowIds])
        matched.sourceChunkIds = sorted(set(matched.sourceChunkIds + incoming.sourceChunkIds))
        matched.evidence = _merge_evidence(matched.evidence, incoming.evidence)
        matched.updatedAt = utc_now()
        return matched

    if result.resolution == ArtifactResolutionKind.CONTRADICTION:
        _append_history(matched, incoming, ArtifactResolutionKind.CONTRADICTION, matched.content, action_name)
        matched.status = ArtifactLifecycleStatus.SUPERSEDED
        matched.supersededBy = str(incoming.id)
        matched.updatedAt = utc_now()
        incoming.status = ArtifactLifecycleStatus.CONFIRMED
        incoming.supersedes = str(matched.id)
        incoming.reason = incoming.reason or incoming.content or incoming.title
        incoming.relatedArtifactIds = _unique([*incoming.relatedArtifactIds, str(matched.id)])
        _replace_or_append(existing, matched)
        return incoming

    _append_history(matched, incoming, ArtifactResolutionKind.UPDATE, matched.content, action_name)
    matched.title = incoming.title or matched.title
    matched.content = _prefer_richer(matched.content, incoming.content)
    matched.ownerText = incoming.ownerText or matched.ownerText
    matched.ownerUserId = incoming.ownerUserId or matched.ownerUserId
    matched.dueDateText = incoming.dueDateText or matched.dueDateText
    matched.dueDateResolved = incoming.dueDateResolved or matched.dueDateResolved
    if incoming.dueDateStatus != "none":
        matched.dueDateStatus = incoming.dueDateStatus
    matched.topic = incoming.topic or matched.topic
    if incoming.semanticHint and not matched.semanticHint:
        matched.semanticHint = incoming.semanticHint
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
    return False


def candidate_payload(artifact: MeetingArtifactDocument) -> dict:
    return {
        "artifactId": str(artifact.id),
        "semanticHint": artifact.semanticHint or "",
        "artifactType": artifact.artifactType.value,
        "title": artifact.title,
        "content": (artifact.content or "")[:400],
        "status": artifact.status.value,
        "ownerText": artifact.ownerText,
        "dueDateText": artifact.dueDateText or artifact.dueDateResolved,
        "evidence": [span.model_dump() for span in artifact.evidence],
        "sourceWindowIds": artifact.sourceWindowIds[:4],
    }


def incoming_payload(index: int, artifact: MeetingArtifactDocument) -> dict:
    return {
        "incomingIndex": index,
        "semanticHint": artifact.semanticHint or "",
        "artifactType": artifact.artifactType.value,
        "title": artifact.title,
        "content": artifact.content,
        "status": artifact.status.value,
        "ownerText": artifact.ownerText,
        "dueDateText": artifact.dueDateText or artifact.dueDateResolved,
        "operation": artifact.operation,
        "evidence": [span.model_dump() for span in artifact.evidence],
        "sourceWindowIds": artifact.sourceWindowIds[:4],
    }


def invalid_modifying_decisions(
    decisions: list[ArtifactReconcileDecision],
    incoming: list[MeetingArtifactDocument],
    existing: list[MeetingArtifactDocument],
) -> list[ArtifactReconcileDecision]:
    existing_ids = {str(artifact.id) for artifact in existing}
    invalid: list[ArtifactReconcileDecision] = []
    for decision in decisions:
        if decision.action not in _MODIFYING_ACTIONS:
            continue
        target = (decision.targetArtifactId or "").strip()
        if not target or target not in existing_ids:
            invalid.append(decision)
    return invalid


def _resolution_from_action(
    action: ReconcileAction,
    incoming: MeetingArtifactDocument,
    matched: MeetingArtifactDocument | None,
) -> ResolutionResult:
    if action == ReconcileAction.CREATE_NEW or matched is None:
        return ResolutionResult(incoming=incoming, resolution=ArtifactResolutionKind.NEW, reconcileAction=action)
    if action == ReconcileAction.UPDATE_EXISTING:
        return ResolutionResult(
            incoming=incoming,
            resolution=ArtifactResolutionKind.UPDATE,
            matched=matched,
            reconcileAction=action,
        )
    if action == ReconcileAction.COMPLETE_EXISTING:
        incoming.status = ArtifactLifecycleStatus.COMPLETED
        incoming.operation = "COMPLETE"
        return ResolutionResult(
            incoming=incoming,
            resolution=ArtifactResolutionKind.COMPLETION,
            matched=matched,
            reconcileAction=action,
        )
    if action == ReconcileAction.CANCEL_EXISTING:
        incoming.status = ArtifactLifecycleStatus.CANCELLED
        incoming.operation = "CANCEL"
        return ResolutionResult(
            incoming=incoming,
            resolution=ArtifactResolutionKind.CONTRADICTION,
            matched=matched,
            reconcileAction=action,
        )
    if action == ReconcileAction.SUPERSEDE_EXISTING:
        incoming.resolution = ArtifactResolutionKind.CONTRADICTION
        return ResolutionResult(
            incoming=incoming,
            resolution=ArtifactResolutionKind.CONTRADICTION,
            matched=matched,
            reconcileAction=action,
        )
    incoming.relatedArtifactIds = _unique([*incoming.relatedArtifactIds, str(matched.id)])
    return ResolutionResult(
        incoming=incoming,
        resolution=ArtifactResolutionKind.RELATED,
        matched=matched,
        reconcileAction=action,
    )


def _find_by_id(existing: list[MeetingArtifactDocument], artifact_id: str | None) -> MeetingArtifactDocument | None:
    if not artifact_id:
        return None
    wanted = str(artifact_id)
    return next((artifact for artifact in existing if str(artifact.id) == wanted), None)


def _copy_artifact(artifact: MeetingArtifactDocument) -> MeetingArtifactDocument:
    copied = MeetingArtifactDocument.model_validate(artifact.model_dump(by_alias=True))
    if str(copied.id) != str(artifact.id):
        copied = copied.model_copy(update={"id": artifact.id})
    return copied


def _append_history(
    matched: MeetingArtifactDocument,
    incoming: MeetingArtifactDocument,
    resolution: ArtifactResolutionKind,
    previous_content: str | None,
    reconcile_action: str | None = None,
) -> None:
    matched.history = [
        *matched.history[-7:],
        ArtifactHistoryEntry(
            at=utc_now(),
            sourceWindowId=str(incoming.sourceWindowId) if incoming.sourceWindowId is not None else None,
            resolution=resolution,
            change=incoming.content or incoming.title,
            previousContent=previous_content,
            evidence=list(incoming.evidence),
            reconcileAction=reconcile_action,
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


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", _normalize(text), flags=re.UNICODE))


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "").casefold()
    return _SPACE_RE.sub(" ", normalized).strip()


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
        if str(item.id) == str(artifact.id):
            existing[index] = artifact
            return
    existing.append(artifact)
