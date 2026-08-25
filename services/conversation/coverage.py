from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from apps.api_gateway.config.setting import settings
from services.conversation.artifact_resolver import item_is_represented
from services.conversation.artifacts import meaningful_artifacts
from services.conversation.models import (
    ConversationWindowDocument,
    MeetingArtifactDocument,
    WindowExtractionResult,
    WindowProcessingStatus,
)


class CoverageSignal(BaseModel):
    windowCount: int = 0
    processedWindowCount: int = 0
    meaningfulArtifactCount: int = 0
    finalArtifactCount: int = 0
    representedArtifactCount: int = 0
    compressionRatio: float = 0.0
    coverageScore: float = 1.0
    unrepresentedTitles: list[str] = Field(default_factory=list)
    weakWindowIndexes: list[int] = Field(default_factory=list)
    topicCoverage: dict[str, int] = Field(default_factory=dict)
    suspicious: bool = False
    reasons: list[str] = Field(default_factory=list)

    def as_checkpoint(self) -> dict[str, Any]:
        return self.model_dump()


def evaluate_coverage(
    windows: list[ConversationWindowDocument],
    artifacts: list[MeetingArtifactDocument],
    final_result: WindowExtractionResult,
) -> CoverageSignal:
    meaningful = meaningful_artifacts(artifacts)
    final_titles = _final_titles(final_result)
    unrepresented = [
        artifact.title
        for artifact in meaningful
        if not item_is_represented(artifact.title, final_titles)
    ]
    represented_count = len(meaningful) - len(unrepresented)
    final_count = _final_count(final_result)
    compression = 0.0
    if meaningful:
        compression = max(0.0, 1.0 - (final_count / max(len(meaningful), 1)))
    weak_windows = _weak_window_indexes(windows, artifacts)
    topics = _topic_counts(windows, artifacts)
    reasons: list[str] = []
    if (
        len(meaningful) >= settings.COVERAGE_MIN_PROVISIONAL_FOR_GUARD
        and compression >= settings.COVERAGE_COMPRESSION_RATIO_THRESHOLD
        and final_count < max(5, int(len(meaningful) * 0.35))
    ):
        reasons.append("suspicious_compression_ratio")
    if weak_windows:
        reasons.append("weak_windows")
    if unrepresented and len(unrepresented) >= max(3, int(len(meaningful) * 0.4)):
        reasons.append("unrepresented_artifacts")
    coverage_score = 1.0
    if meaningful:
        coverage_score = represented_count / len(meaningful)
    if weak_windows:
        coverage_score = min(coverage_score, max(0.0, 1.0 - (len(weak_windows) / max(len(windows), 1))))
    return CoverageSignal(
        windowCount=len(windows),
        processedWindowCount=sum(1 for window in windows if window.status == WindowProcessingStatus.COMPLETED),
        meaningfulArtifactCount=len(meaningful),
        finalArtifactCount=final_count,
        representedArtifactCount=represented_count,
        compressionRatio=round(compression, 4),
        coverageScore=round(coverage_score, 4),
        unrepresentedTitles=unrepresented[:40],
        weakWindowIndexes=weak_windows,
        topicCoverage=topics,
        suspicious=bool(reasons),
        reasons=reasons,
    )


def preserve_unrepresented(
    final_result: WindowExtractionResult,
    artifacts: list[MeetingArtifactDocument],
    conversation_id: str,
    space_id: str,
) -> WindowExtractionResult:
    from services.conversation.artifacts import artifacts_to_extraction_result

    carried = artifacts_to_extraction_result(artifacts, conversation_id, space_id, summary=final_result.summary, topics=final_result.topics)
    final_result.tasks = _union_by_title(final_result.tasks, carried.tasks, strict=True)
    final_result.notes = _union_by_title(final_result.notes, carried.notes, strict=True)
    final_result.decisions = _union_by_title(final_result.decisions, carried.decisions, strict=True)
    final_result.issues = _union_by_title(final_result.issues, carried.issues, strict=True)
    final_result.importantFacts = _union_strings(final_result.importantFacts, carried.importantFacts)
    final_result.openQuestions = _union_strings(final_result.openQuestions, carried.openQuestions)
    if not final_result.topics:
        final_result.topics = carried.topics
    if not final_result.summary:
        final_result.summary = carried.summary
    return final_result


def _final_titles(result: WindowExtractionResult) -> list[str]:
    titles = [item.title for item in result.tasks]
    titles.extend(item.title for item in result.notes)
    titles.extend(item.title for item in result.decisions)
    titles.extend(item.title for item in result.issues)
    titles.extend(result.importantFacts)
    titles.extend(result.openQuestions)
    return titles


def _final_count(result: WindowExtractionResult) -> int:
    return (
        len(result.tasks)
        + len(result.notes)
        + len(result.decisions)
        + len(result.issues)
        + len(result.importantFacts)
        + len(result.openQuestions)
    )


def _weak_window_indexes(
    windows: list[ConversationWindowDocument],
    artifacts: list[MeetingArtifactDocument],
) -> list[int]:
    artifact_windows: set[str] = set()
    for artifact in artifacts:
        artifact_windows.update(artifact.sourceWindowIds)
        if artifact.sourceWindowId is not None:
            artifact_windows.add(str(artifact.sourceWindowId))
    weak: list[int] = []
    for window in windows:
        if not _window_is_meaningful(window) or window.isFinalPartial or window.extractionSkipped:
            continue
        window_id = str(window.id)
        represented = window_id in artifact_windows
        extraction = window.result
        extracted_count = 0
        if extraction:
            extracted_count = (
                len(extraction.tasks)
                + len(extraction.notes)
                + len(extraction.decisions)
                + len(extraction.issues)
                + len(extraction.importantFacts)
            )
        if not represented and extracted_count == 0:
            weak.append(window.windowIndex)
    return weak


def _window_is_meaningful(window: ConversationWindowDocument) -> bool:
    if getattr(window, "nonEmptyChunkCount", 0):
        return True
    text = (window.text or "").strip()
    if not text:
        return False
    if not settings.COVERAGE_SPARSE_WINDOW_ENABLED and window.tokenCount < settings.COVERAGE_WEAK_WINDOW_MIN_TOKENS:
        return False
    return True


def _topic_counts(windows: list[ConversationWindowDocument], artifacts: list[MeetingArtifactDocument]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for window in windows:
        if not window.result:
            continue
        for topic in window.result.topics:
            counts[topic] = counts.get(topic, 0)
    for artifact in meaningful_artifacts(artifacts):
        if artifact.topic:
            counts[artifact.topic] = counts.get(artifact.topic, 0) + 1
    return counts


def _union_by_title(primary: list, extra: list, strict: bool = False) -> list:
    titles = [getattr(item, "title", "") for item in primary]
    merged = list(primary)
    for item in extra:
        title = getattr(item, "title", "")
        if item_is_represented(title, titles, strict=strict):
            continue
        merged.append(item)
        titles.append(title)
    return merged


def _union_strings(primary: list[str], extra: list[str]) -> list[str]:
    seen = {item.strip().casefold() for item in primary if item.strip()}
    merged = list(primary)
    for item in extra:
        key = item.strip().casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged
