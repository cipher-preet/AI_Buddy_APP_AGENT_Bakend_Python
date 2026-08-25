from __future__ import annotations

from typing import Any, Iterable

from services.conversation.fingerprints import artifact_identity_key, note_fingerprint, task_fingerprint
from services.conversation.intelligence import score_and_filter_result
from services.conversation.models import (
    ArtifactLifecycleStatus,
    ArtifactResolutionKind,
    ArtifactType,
    ConversationWindowDocument,
    EvidenceSpan,
    ExtractedDecision,
    ExtractedIssue,
    ExtractedNote,
    ExtractedTask,
    MeetingArtifactDocument,
    Operation,
    SemanticUnit,
    WindowExtractionResult,
    new_id,
    utc_now,
)


TASK_ARTIFACT_TYPES = {
    ArtifactType.TASK,
    ArtifactType.FOLLOW_UP,
    ArtifactType.COMMITMENT,
    ArtifactType.DEADLINE,
}
NOTE_ARTIFACT_TYPES = {
    ArtifactType.NOTE,
    ArtifactType.FACT,
    ArtifactType.PREFERENCE,
    ArtifactType.IDEA,
    ArtifactType.REQUIREMENT,
    ArtifactType.REFERENCE,
    ArtifactType.ANSWER,
}
DECISION_ARTIFACT_TYPES = {ArtifactType.DECISION}
ISSUE_ARTIFACT_TYPES = {ArtifactType.QUESTION, ArtifactType.RISK, ArtifactType.BLOCKER}
ACTIVE_STATUSES = {
    ArtifactLifecycleStatus.PROVISIONAL,
    ArtifactLifecycleStatus.PROPOSED,
    ArtifactLifecycleStatus.ACTIVE,
    ArtifactLifecycleStatus.CONFIRMED,
    ArtifactLifecycleStatus.MODIFIED,
    ArtifactLifecycleStatus.ASSIGNED,
    ArtifactLifecycleStatus.BLOCKED,
    ArtifactLifecycleStatus.COMPLETED,
    ArtifactLifecycleStatus.UNRESOLVED,
}
TERMINAL_HIDDEN_STATUSES = {
    ArtifactLifecycleStatus.SUPERSEDED,
    ArtifactLifecycleStatus.MERGED,
    ArtifactLifecycleStatus.REJECTED,
    ArtifactLifecycleStatus.CANCELLED,
}


def artifacts_from_window(
    window: ConversationWindowDocument,
    result: WindowExtractionResult | None = None,
) -> list[MeetingArtifactDocument]:
    extraction = result or window.result
    if extraction is None:
        return []
    conversation_id = str(window.conversationId)
    window_id = str(window.id)
    fallback_evidence = _window_evidence(window)
    topic = extraction.topics[0] if extraction.topics else None
    artifacts: list[MeetingArtifactDocument] = []
    if getattr(extraction, "semanticUnits", None):
        artifacts.extend(
            _artifacts_from_semantic_units(extraction.semanticUnits, window, conversation_id, window_id, fallback_evidence, topic)
        )
    publishable = extraction
    if not getattr(extraction, "isCheckpoint", False) and not getattr(extraction, "semanticUnits", None):
        # Final user-facing conversion still uses the publish gate. Checkpoint
        # and reconstruction artifacts stay on the ledger when evidence exists.
        publishable = score_and_filter_result(extraction, window.text)
    for task in publishable.tasks:
        artifacts.append(
            _task_artifact(task, window, conversation_id, window_id, fallback_evidence, topic)
        )
    for note in publishable.notes:
        artifacts.append(
            _note_artifact(note, ArtifactType.NOTE, window, conversation_id, window_id, fallback_evidence, topic)
        )
    for decision in publishable.decisions:
        artifacts.append(_decision_artifact(decision, window, conversation_id, window_id, fallback_evidence, topic))
    for issue in publishable.issues:
        artifacts.append(_issue_artifact(issue, window, conversation_id, window_id, fallback_evidence, topic))
    return _dedupe_new_artifacts(
        [artifact for artifact in artifacts if artifact.evidence]
    )


def artifacts_from_windows(windows: Iterable[ConversationWindowDocument]) -> list[MeetingArtifactDocument]:
    captured: list[MeetingArtifactDocument] = []
    for window in windows:
        captured.extend(artifacts_from_window(window))
    return captured


def compact_artifact(artifact: MeetingArtifactDocument, content_limit: int = 240) -> dict[str, Any]:
    body_preview = (artifact.content or "")[:content_limit]
    return {
        "artifactId": str(artifact.id),
        "artifactType": artifact.artifactType.value,
        "title": artifact.title,
        "content": body_preview,
        # Kept as a derived presentation field so the persistence schema stays
        # unchanged.  It can never accidentally repeat the title unless the
        # validated body itself did so.
        "bodyPreview": body_preview,
        "status": artifact.status.value,
        "resolution": artifact.resolution.value,
        "ownerText": artifact.ownerText,
        "dueDateText": artifact.dueDateText or artifact.dueDateResolved,
        "topic": artifact.topic,
        "parentArtifactId": artifact.parentArtifactId,
        "relatedArtifactIds": artifact.relatedArtifactIds[:8],
        "confidence": artifact.confidence,
        "needsConfirmation": artifact.needsConfirmation,
        "sourceWindowId": str(artifact.sourceWindowId) if artifact.sourceWindowId is not None else None,
        "sourceWindowIds": artifact.sourceWindowIds[:8],
        "sourceChunkIds": artifact.sourceChunkIds[:16],
        "sourceStartIndex": artifact.sourceStartIndex,
        "sourceEndIndex": artifact.sourceEndIndex,
        "reason": artifact.reason,
        "supersededBy": artifact.supersededBy,
        "evidence": [span.model_dump() for span in artifact.evidence[:3]],
    }


def artifacts_to_extraction_result(
    artifacts: list[MeetingArtifactDocument],
    conversation_id: str,
    space_id: str,
    summary: str = "",
    topics: list[str] | None = None,
) -> WindowExtractionResult:
    tasks: list[ExtractedTask] = []
    notes: list[ExtractedNote] = []
    decisions: list[ExtractedDecision] = []
    issues: list[ExtractedIssue] = []
    facts: list[str] = []
    questions: list[str] = []
    for artifact in artifacts:
        if artifact.status in TERMINAL_HIDDEN_STATUSES:
            if artifact.artifactType == ArtifactType.DECISION and artifact.status == ArtifactLifecycleStatus.SUPERSEDED:
                continue
            continue
        if artifact.artifactType in TASK_ARTIFACT_TYPES:
            task = artifact_to_task(artifact, conversation_id, space_id)
            if task:
                tasks.append(task)
        elif artifact.artifactType in DECISION_ARTIFACT_TYPES:
            decision = artifact_to_decision(artifact, conversation_id)
            if decision:
                decisions.append(decision)
        elif artifact.artifactType in ISSUE_ARTIFACT_TYPES:
            issue = artifact_to_issue(artifact, conversation_id)
            if issue:
                issues.append(issue)
            if artifact.artifactType == ArtifactType.QUESTION:
                questions.append(artifact.title)
        else:
            note = artifact_to_note(artifact, conversation_id, space_id)
            if note:
                notes.append(note)
            if artifact.artifactType == ArtifactType.FACT:
                facts.append(artifact.content or artifact.title)
    return WindowExtractionResult(
        summary=summary,
        topics=topics or [],
        importantFacts=facts,
        tasks=tasks,
        notes=notes,
        decisions=decisions,
        issues=issues,
        openQuestions=questions,
    )


def artifact_to_task(artifact: MeetingArtifactDocument, conversation_id: str, space_id: str) -> ExtractedTask | None:
    evidence = list(artifact.evidence)
    if not evidence:
        return None
    operation = _task_operation(artifact)
    task = ExtractedTask(
        title=artifact.title,
        body=artifact.content or artifact.title,
        operation=operation,
        existingTaskId=artifact.existingTaskId,
        ownerText=artifact.ownerText,
        ownerUserId=artifact.ownerUserId,
        dueDateText=artifact.dueDateText,
        dueDateResolved=artifact.dueDateResolved,
        dueDateStatus=artifact.dueDateStatus,
        confidence=artifact.confidence,
        needsConfirmation=artifact.needsConfirmation or operation == "NEEDS_CONFIRMATION",
        sourceConversationId=conversation_id,
        fingerprint=artifact.fingerprint or artifact.identityKey,
        evidence=evidence,
        artifactId=str(artifact.id),
        parentTitle=None,
        sourceWindowId=str(artifact.sourceWindowId) if artifact.sourceWindowId is not None else None,
        changes={"artifactType": artifact.artifactType.value, "parentArtifactId": artifact.parentArtifactId},
        origin=artifact.reason if artifact.reason in {"explicit", "strongly_inferred"} else "unknown",
    )
    task.fingerprint = task.fingerprint or task_fingerprint(space_id, task)
    return task


def artifact_to_note(artifact: MeetingArtifactDocument, conversation_id: str, space_id: str) -> ExtractedNote | None:
    evidence = list(artifact.evidence)
    if not evidence:
        return None
    note = ExtractedNote(
        title=artifact.title,
        body=artifact.content or artifact.title,
        confidence=artifact.confidence,
        sourceConversationId=conversation_id,
        fingerprint=artifact.fingerprint or artifact.identityKey,
        evidence=evidence,
        artifactId=str(artifact.id),
        sourceWindowId=str(artifact.sourceWindowId) if artifact.sourceWindowId is not None else None,
        debug={"artifactType": artifact.artifactType.value, "relatedArtifactIds": artifact.relatedArtifactIds[:8]},
    )
    note.fingerprint = note.fingerprint or note_fingerprint(space_id, note)
    return note


def artifact_to_decision(artifact: MeetingArtifactDocument, conversation_id: str) -> ExtractedDecision | None:
    evidence = list(artifact.evidence)
    if not evidence:
        return None
    status = "confirmed_decision"
    if artifact.status == ArtifactLifecycleStatus.PROVISIONAL:
        status = "proposal"
    if artifact.resolution == ArtifactResolutionKind.CONTRADICTION and artifact.status != ArtifactLifecycleStatus.SUPERSEDED:
        status = "confirmed_decision"
    return ExtractedDecision(
        title=artifact.title,
        status=status,
        confidence=artifact.confidence,
        sourceConversationId=conversation_id,
        evidence=evidence,
        artifactId=str(artifact.id),
        sourceWindowId=str(artifact.sourceWindowId) if artifact.sourceWindowId is not None else None,
        superseded=artifact.status == ArtifactLifecycleStatus.SUPERSEDED,
        reason=artifact.reason,
    )


def artifact_to_issue(artifact: MeetingArtifactDocument, conversation_id: str) -> ExtractedIssue | None:
    evidence = list(artifact.evidence)
    if not evidence:
        return None
    kind = "open_question"
    if artifact.artifactType == ArtifactType.BLOCKER:
        kind = "blocker"
    elif artifact.artifactType == ArtifactType.RISK:
        kind = "risk"
    return ExtractedIssue(
        title=artifact.title,
        kind=kind,
        confidence=artifact.confidence,
        sourceConversationId=conversation_id,
        evidence=evidence,
        artifactId=str(artifact.id),
        sourceWindowId=str(artifact.sourceWindowId) if artifact.sourceWindowId is not None else None,
    )


def meaningful_artifacts(artifacts: list[MeetingArtifactDocument]) -> list[MeetingArtifactDocument]:
    return [
        artifact
        for artifact in artifacts
        if artifact.status not in TERMINAL_HIDDEN_STATUSES and (artifact.title or "").strip()
    ]


def source_chunk_ids(evidence: list[EvidenceSpan]) -> list[int]:
    values: set[int] = set()
    for span in evidence:
        values.add(span.sequenceStart)
        values.add(span.sequenceEnd)
    return sorted(values)


def _task_artifact(
    task: ExtractedTask,
    window: ConversationWindowDocument,
    conversation_id: str,
    window_id: str,
    fallback_evidence: EvidenceSpan,
    topic: str | None,
) -> MeetingArtifactDocument:
    evidence = list(task.evidence) or [fallback_evidence]
    artifact_type = ArtifactType.TASK
    return _base_artifact(
        conversation_id=conversation_id,
        user_id=window.userId,
        space_id=window.spaceId,
        artifact_type=artifact_type,
        title=task.title,
        content=task.body or task.title,
        window=window,
        window_id=window_id,
        evidence=evidence,
        owner_text=task.ownerText,
        owner_user_id=task.ownerUserId,
        due_date_text=task.dueDateText,
        due_date_resolved=task.dueDateResolved,
        due_date_status=task.dueDateStatus,
        confidence=task.confidence,
        needs_confirmation=task.needsConfirmation,
        operation=task.operation,
        existing_task_id=task.existingTaskId,
        fingerprint=task.fingerprint,
        topic=topic,
        status=_status_for_operation(task.operation),
        reason=task.origin,
        semantic_key=(task.changes or {}).get("semanticArtifactKey"),
    )


def _note_artifact(
    note: ExtractedNote,
    artifact_type: ArtifactType,
    window: ConversationWindowDocument,
    conversation_id: str,
    window_id: str,
    fallback_evidence: EvidenceSpan,
    topic: str | None,
) -> MeetingArtifactDocument:
    evidence = list(note.evidence) or [fallback_evidence]
    return _base_artifact(
        conversation_id=conversation_id,
        user_id=window.userId,
        space_id=window.spaceId,
        artifact_type=artifact_type,
        title=note.title,
        content=note.body or note.title,
        window=window,
        window_id=window_id,
        evidence=evidence,
        confidence=note.confidence,
        fingerprint=note.fingerprint,
        topic=topic,
        semantic_key=(getattr(note, "debug", None) or {}).get("semanticArtifactKey"),
    )


def _decision_artifact(
    decision: ExtractedDecision,
    window: ConversationWindowDocument,
    conversation_id: str,
    window_id: str,
    fallback_evidence: EvidenceSpan,
    topic: str | None,
) -> MeetingArtifactDocument:
    evidence = list(decision.evidence) or [fallback_evidence]
    status = ArtifactLifecycleStatus.CONFIRMED if decision.status == "confirmed_decision" else ArtifactLifecycleStatus.PROVISIONAL
    if decision.status == "idea":
        artifact_type = ArtifactType.IDEA
    else:
        artifact_type = ArtifactType.DECISION
    return _base_artifact(
        conversation_id=conversation_id,
        user_id=window.userId,
        space_id=window.spaceId,
        artifact_type=artifact_type,
        title=decision.title,
        content=decision.title,
        window=window,
        window_id=window_id,
        evidence=evidence,
        confidence=decision.confidence,
        topic=topic,
        status=status,
    )


def _issue_artifact(
    issue: ExtractedIssue,
    window: ConversationWindowDocument,
    conversation_id: str,
    window_id: str,
    fallback_evidence: EvidenceSpan,
    topic: str | None,
) -> MeetingArtifactDocument:
    evidence = list(issue.evidence) or [fallback_evidence]
    artifact_type = ArtifactType.QUESTION
    if issue.kind == "blocker":
        artifact_type = ArtifactType.BLOCKER
    elif issue.kind == "risk":
        artifact_type = ArtifactType.RISK
    return _base_artifact(
        conversation_id=conversation_id,
        user_id=window.userId,
        space_id=window.spaceId,
        artifact_type=artifact_type,
        title=issue.title,
        content=issue.title,
        window=window,
        window_id=window_id,
        evidence=evidence,
        confidence=issue.confidence,
        topic=topic,
    )


def _text_artifact(
    artifact_type: ArtifactType,
    title: str,
    content: str,
    window: ConversationWindowDocument,
    conversation_id: str,
    window_id: str,
    evidence: EvidenceSpan,
    topic: str | None,
) -> MeetingArtifactDocument:
    cleaned = " ".join(str(title or "").split())
    return _base_artifact(
        conversation_id=conversation_id,
        user_id=window.userId,
        space_id=window.spaceId,
        artifact_type=artifact_type,
        title=cleaned[:180] or artifact_type.value,
        content=content or cleaned,
        window=window,
        window_id=window_id,
        evidence=[evidence],
        confidence=0.55,
        topic=topic,
    )


def _base_artifact(
    *,
    conversation_id: str,
    user_id: Any,
    space_id: Any,
    artifact_type: ArtifactType,
    title: str,
    content: str,
    window: ConversationWindowDocument,
    window_id: str,
    evidence: list[EvidenceSpan],
    owner_text: str | None = None,
    owner_user_id: str | None = None,
    due_date_text: str | None = None,
    due_date_resolved: str | None = None,
    due_date_status: str = "none",
    confidence: float = 0.5,
    needs_confirmation: bool = False,
    operation: Operation | None = None,
    existing_task_id: str | None = None,
    fingerprint: str | None = None,
    topic: str | None = None,
    status: ArtifactLifecycleStatus = ArtifactLifecycleStatus.PROVISIONAL,
    reason: str | None = None,
    semantic_key: str | None = None,
) -> MeetingArtifactDocument:
    chunk_ids = source_chunk_ids(evidence)
    start_index = min((span.sequenceStart for span in evidence), default=window.sequenceStart)
    end_index = max((span.sequenceEnd for span in evidence), default=window.sequenceEnd)
    artifact_id = new_id()
    identity = artifact_identity_key(conversation_id, str(artifact_id))
    hint = (semantic_key or "").strip() or None
    return MeetingArtifactDocument(
        id=artifact_id,
        conversationId=window.conversationId,
        userId=user_id,
        spaceId=space_id,
        identityKey=identity,
        semanticHint=hint,
        artifactType=artifact_type,
        title=title.strip(),
        content=content.strip(),
        ownerText=owner_text,
        ownerUserId=owner_user_id,
        dueDateText=due_date_text,
        dueDateResolved=due_date_resolved,
        dueDateStatus=due_date_status,
        topic=topic,
        confidence=confidence,
        status=status,
        resolution=ArtifactResolutionKind.NEW,
        sourceWindowId=window.id,
        sourceWindowIds=[window_id],
        sourceChunkIds=chunk_ids,
        sourceStartIndex=start_index,
        sourceEndIndex=end_index,
        evidence=evidence,
        fingerprint=fingerprint or identity,
        needsConfirmation=needs_confirmation,
        operation=operation,
        existingTaskId=existing_task_id,
        reason=reason,
        createdAt=utc_now(),
        updatedAt=utc_now(),
    )


def _window_evidence(window: ConversationWindowDocument) -> EvidenceSpan:
    text = " ".join((window.text or "").split())[:400] or f"window {window.windowIndex}"
    return EvidenceSpan(sequenceStart=window.sequenceStart, sequenceEnd=window.sequenceEnd, text=text)


def _status_for_operation(operation: Operation | None) -> ArtifactLifecycleStatus:
    if operation == "COMPLETE":
        return ArtifactLifecycleStatus.COMPLETED
    if operation == "CANCEL":
        return ArtifactLifecycleStatus.CANCELLED
    if operation == "UPDATE":
        return ArtifactLifecycleStatus.MODIFIED
    return ArtifactLifecycleStatus.PROPOSED


def _task_operation(artifact: MeetingArtifactDocument) -> Operation:
    if artifact.operation:
        return artifact.operation
    if artifact.status == ArtifactLifecycleStatus.COMPLETED:
        return "COMPLETE"
    if artifact.status in {ArtifactLifecycleStatus.REJECTED, ArtifactLifecycleStatus.CANCELLED}:
        return "CANCEL"
    if artifact.resolution == ArtifactResolutionKind.UPDATE or artifact.status in {
        ArtifactLifecycleStatus.MODIFIED,
        ArtifactLifecycleStatus.ASSIGNED,
        ArtifactLifecycleStatus.ACTIVE,
    }:
        return "UPDATE"
    if artifact.needsConfirmation:
        return "NEEDS_CONFIRMATION"
    return "CREATE"


def _artifacts_from_semantic_units(
    units: list[SemanticUnit],
    window: ConversationWindowDocument,
    conversation_id: str,
    window_id: str,
    fallback_evidence: EvidenceSpan,
    topic: str | None,
) -> list[MeetingArtifactDocument]:
    artifacts: list[MeetingArtifactDocument] = []
    for unit in units:
        if not (unit.meaning or "").strip():
            continue
        evidence = list(unit.evidence) or [fallback_evidence]
        artifacts.append(
            _base_artifact(
                conversation_id=conversation_id,
                user_id=window.userId,
                space_id=window.spaceId,
                artifact_type=_artifact_type_for_unit(unit.kind),
                title=_unit_title(unit),
                content=unit.meaning.strip(),
                window=window,
                window_id=window_id,
                evidence=evidence,
                owner_text=unit.ownerText,
                due_date_text=unit.dueDateText,
                confidence=0.72 if unit.quality.get("grounded") else 0.6,
                topic=topic,
                status=_status_from_label(unit.state),
                reason=unit.kind,
                semantic_key=unit.semanticKey or None,  # retrieval hint only; not identity
            )
        )
    return artifacts


def _artifact_type_for_unit(kind: str) -> ArtifactType:
    mapping = {
        "commitment": ArtifactType.COMMITMENT,
        "action_candidate": ArtifactType.TASK,
        "note_candidate": ArtifactType.NOTE,
        "decision": ArtifactType.DECISION,
        "fact": ArtifactType.FACT,
        "assignment": ArtifactType.TASK,
        "deadline": ArtifactType.DEADLINE,
        "follow_up": ArtifactType.FOLLOW_UP,
        "question": ArtifactType.QUESTION,
        "thread": ArtifactType.QUESTION,
        "dependency": ArtifactType.REQUIREMENT,
        "blocker": ArtifactType.BLOCKER,
        "change": ArtifactType.FACT,
        "completion": ArtifactType.TASK,
        "cancellation": ArtifactType.TASK,
        "assumption": ArtifactType.FACT,
        "narrative": ArtifactType.NOTE,
    }
    return mapping.get((kind or "").strip().lower(), ArtifactType.FACT)


def _unit_title(unit: SemanticUnit) -> str:
    meaning = " ".join((unit.meaning or "").split())
    return meaning[:180] or unit.kind


def _status_from_label(label: str | None) -> ArtifactLifecycleStatus:
    try:
        return ArtifactLifecycleStatus((label or "unresolved").strip().lower())
    except ValueError:
        return ArtifactLifecycleStatus.UNRESOLVED


def _dedupe_new_artifacts(artifacts: list[MeetingArtifactDocument]) -> list[MeetingArtifactDocument]:
    seen: set[str] = set()
    unique: list[MeetingArtifactDocument] = []
    for artifact in artifacts:
        if artifact.identityKey in seen:
            continue
        seen.add(artifact.identityKey)
        unique.append(artifact)
    return unique
