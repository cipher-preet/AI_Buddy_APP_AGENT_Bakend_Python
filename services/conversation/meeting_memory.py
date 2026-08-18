from __future__ import annotations

from typing import Any

from apps.api_gateway.config.setting import settings
from services.conversation.artifacts import (
    DECISION_ARTIFACT_TYPES,
    ISSUE_ARTIFACT_TYPES,
    NOTE_ARTIFACT_TYPES,
    TASK_ARTIFACT_TYPES,
    meaningful_artifacts,
)
from services.conversation.fingerprints import stable_hash
from services.conversation.models import (
    ArtifactType,
    ConversationWindowDocument,
    MeetingArtifactDocument,
    MeetingMemoryDocument,
    MeetingMemoryItem,
    TopicMemory,
    utc_now,
)


def build_meeting_memory(
    conversation_id: str,
    user_id: Any,
    space_id: Any,
    artifacts: list[MeetingArtifactDocument],
    windows: list[ConversationWindowDocument],
    previous: MeetingMemoryDocument | None = None,
) -> MeetingMemoryDocument:
    active = meaningful_artifacts(artifacts)
    topics = _topics_from(windows, active)
    memory = previous.model_copy(deep=True) if previous else MeetingMemoryDocument(
        conversationId=conversation_id,
        userId=user_id,
        spaceId=space_id,
    )
    limit = settings.MEETING_MEMORY_GLOBAL_ITEM_LIMIT
    memory.activeTopics = topics
    memory.knownTasks = _items([item for item in active if item.artifactType in TASK_ARTIFACT_TYPES], limit)
    memory.knownNotes = _items([item for item in active if item.artifactType in NOTE_ARTIFACT_TYPES], limit)
    memory.decisions = _items([item for item in active if item.artifactType in DECISION_ARTIFACT_TYPES], limit)
    memory.requirements = _items([item for item in active if item.artifactType == ArtifactType.REQUIREMENT], limit)
    memory.commitments = _items([item for item in active if item.artifactType == ArtifactType.COMMITMENT], limit)
    memory.openQuestions = _items([item for item in active if item.artifactType == ArtifactType.QUESTION], limit)
    memory.deadlines = _items(
        [item for item in active if item.dueDateText or item.dueDateResolved or item.artifactType == ArtifactType.DEADLINE],
        limit,
    )
    memory.blockers = _items(
        [item for item in active if item.artifactType in {ArtifactType.BLOCKER, ArtifactType.RISK}],
        limit,
    )
    memory.importantFacts = _items([item for item in active if item.artifactType == ArtifactType.FACT], limit)
    memory.unresolvedReferences = [
        item.title for item in active if item.artifactType == ArtifactType.QUESTION
    ][:limit]
    memory.shortSummary = _short_summary(windows, topics)
    memory.artifactCount = len(active)
    memory.windowCount = len(windows)
    memory.version = (previous.version + 1) if previous else 1
    memory.updatedAt = utc_now()
    return memory


def select_context_for_window(
    memory: MeetingMemoryDocument | None,
    artifacts: list[MeetingArtifactDocument],
    window_text: str,
    window_topics: list[str] | None = None,
) -> dict[str, Any]:
    if memory is None:
        return {"meetingMemory": {}, "relevantArtifacts": [], "activeTopics": []}
    relevant = _relevant_artifacts(memory, artifacts, window_text, window_topics or [])
    return {
        "meetingMemory": {
            "shortSummary": memory.shortSummary,
            "activeTopics": [topic.model_dump() for topic in memory.activeTopics[:12]],
            "knownTasks": [item.model_dump() for item in memory.knownTasks[: settings.MEETING_MEMORY_GLOBAL_ITEM_LIMIT]],
            "decisions": [item.model_dump() for item in memory.decisions[: settings.MEETING_MEMORY_GLOBAL_ITEM_LIMIT]],
            "openQuestions": [item.model_dump() for item in memory.openQuestions[: settings.MEETING_MEMORY_GLOBAL_ITEM_LIMIT]],
            "blockers": [item.model_dump() for item in memory.blockers[: settings.MEETING_MEMORY_GLOBAL_ITEM_LIMIT]],
            "requirements": [item.model_dump() for item in memory.requirements[: settings.MEETING_MEMORY_GLOBAL_ITEM_LIMIT]],
            "deadlines": [item.model_dump() for item in memory.deadlines[: settings.MEETING_MEMORY_GLOBAL_ITEM_LIMIT]],
            "artifactCount": memory.artifactCount,
            "windowCount": memory.windowCount,
        },
        "relevantArtifacts": relevant,
        "activeTopics": [topic.label for topic in memory.activeTopics],
    }


def _relevant_artifacts(
    memory: MeetingMemoryDocument,
    artifacts: list[MeetingArtifactDocument],
    window_text: str,
    window_topics: list[str],
) -> list[dict[str, Any]]:
    window_tokens = _tokens(window_text)
    topic_labels = {_normalize(label) for label in [*window_topics, *[topic.label for topic in memory.activeTopics]]}
    scored: list[tuple[float, MeetingArtifactDocument]] = []
    for artifact in meaningful_artifacts(artifacts):
        score = 0.0
        artifact_tokens = _tokens(f"{artifact.title} {artifact.content} {artifact.topic or ''}")
        if window_tokens and artifact_tokens:
            score += len(window_tokens & artifact_tokens) / max(1, len(artifact_tokens))
        if artifact.topic and _normalize(artifact.topic) in topic_labels:
            score += 0.4
        if artifact.artifactType in {ArtifactType.TASK, ArtifactType.DECISION, ArtifactType.QUESTION, ArtifactType.BLOCKER}:
            score += 0.1
        scored.append((score, artifact))
    scored.sort(key=lambda item: item[0], reverse=True)
    selected = [artifact for score, artifact in scored if score > 0][: settings.MEETING_MEMORY_RETRIEVAL_LIMIT]
    if len(selected) < min(6, len(artifacts)):
        recent = meaningful_artifacts(artifacts)[-6:]
        seen = {str(item.id) for item in selected}
        for artifact in recent:
            if str(artifact.id) in seen:
                continue
            selected.append(artifact)
            if len(selected) >= settings.MEETING_MEMORY_RETRIEVAL_LIMIT:
                break
    return [
        {
            "artifactId": str(artifact.id),
            "artifactType": artifact.artifactType.value,
            "title": artifact.title,
            "content": (artifact.content or "")[:180],
            "status": artifact.status.value,
            "ownerText": artifact.ownerText,
            "dueDateText": artifact.dueDateText or artifact.dueDateResolved,
            "topic": artifact.topic,
        }
        for artifact in selected[: settings.MEETING_MEMORY_RETRIEVAL_LIMIT]
    ]


def _topics_from(windows: list[ConversationWindowDocument], artifacts: list[MeetingArtifactDocument]) -> list[TopicMemory]:
    topics: dict[str, TopicMemory] = {}
    for window in windows:
        labels = list(window.result.topics) if window.result else []
        for label in labels:
            normalized = _normalize(label)
            if not normalized:
                continue
            topic_id = stable_hash(normalized)
            topic = topics.setdefault(
                topic_id,
                TopicMemory(topicId=topic_id, label=label.strip()),
            )
            if window.windowIndex not in topic.windowIndexes:
                topic.windowIndexes.append(window.windowIndex)
            topic.lastSequence = window.sequenceEnd
    for artifact in artifacts:
        if not artifact.topic:
            continue
        topic_id = stable_hash(_normalize(artifact.topic))
        topic = topics.setdefault(topic_id, TopicMemory(topicId=topic_id, label=artifact.topic))
        topic.artifactCount += 1
    return list(topics.values())[:20]


def _items(artifacts: list[MeetingArtifactDocument], limit: int) -> list[MeetingMemoryItem]:
    items = [
        MeetingMemoryItem(
            artifactId=str(artifact.id),
            artifactType=artifact.artifactType.value,
            title=artifact.title,
            status=artifact.status.value,
            ownerText=artifact.ownerText,
            dueDateText=artifact.dueDateText or artifact.dueDateResolved,
            topic=artifact.topic,
            contentPreview=(artifact.content or "")[:160],
        )
        for artifact in artifacts
    ]
    return items[:limit]


def _short_summary(windows: list[ConversationWindowDocument], topics: list[TopicMemory]) -> str:
    labels = [topic.label for topic in topics[:6]]
    last_summary = ""
    for window in reversed(windows):
        if window.result and window.result.summary:
            last_summary = window.result.summary.strip()
            break
    topic_text = ", ".join(labels)
    if last_summary and topic_text:
        return f"{last_summary[:240]} Topics: {topic_text}."
    return last_summary[:280] or (f"Active topics: {topic_text}." if topic_text else "")


def _tokens(text: str) -> set[str]:
    return {token for token in _normalize(text).split() if len(token) > 2}


def _normalize(text: str) -> str:
    return " ".join((text or "").casefold().split())
