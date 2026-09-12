from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from services.conversation.models import utc_now


PIPELINE_VERSION = "daily-briefing-v1"


class BriefingStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    READY = "READY"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class EvidenceRef(BaseModel):
    sourceType: str
    sourceId: str
    timestamp: datetime | None = None


class BriefingItem(BaseModel):
    id: str
    title: str
    detail: str = ""
    evidence: list[EvidenceRef] = Field(default_factory=list)


class TaskCard(BaseModel):
    id: str
    title: str
    meta: str = ""


class MeetingCard(BaseModel):
    id: str
    time: str
    title: str
    meta: str = ""


class InsightCard(BaseModel):
    id: str
    source: str
    sourceType: str
    title: str
    body: str
    excerpt: str = ""
    whyItMatters: str = ""
    capturedAt: str = ""
    space: str = ""
    tags: list[str] = Field(default_factory=list)


class SourceStats(BaseModel):
    transcriptCount: int = 0
    taskCount: int = 0
    noteCount: int = 0
    eventCount: int = 0
    reminderCount: int = 0
    pendingTranscriptCount: int = 0


class WindowIntelligence(BaseModel):
    summary: str = ""
    highlights: list[BriefingItem] = Field(default_factory=list)
    decisions: list[BriefingItem] = Field(default_factory=list)
    completedItems: list[BriefingItem] = Field(default_factory=list)
    pendingItems: list[BriefingItem] = Field(default_factory=list)
    followUps: list[BriefingItem] = Field(default_factory=list)
    importantMoments: list[BriefingItem] = Field(default_factory=list)
    people: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)


class DailyBriefingSynthesis(BaseModel):
    headline: str = ""
    overview: str = ""
    highlights: list[BriefingItem] = Field(default_factory=list)
    importantMoments: list[BriefingItem] = Field(default_factory=list)
    completed: list[BriefingItem] = Field(default_factory=list)
    pendingTasks: list[BriefingItem] = Field(default_factory=list)
    decisions: list[BriefingItem] = Field(default_factory=list)
    followUps: list[BriefingItem] = Field(default_factory=list)
    tomorrowFocus: list[BriefingItem] = Field(default_factory=list)
    insights: list[InsightCard] = Field(default_factory=list)
    people: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    tasks: list[TaskCard] = Field(default_factory=list)
    meetings: list[MeetingCard] = Field(default_factory=list)
    missedCandidates: list[BriefingItem] = Field(default_factory=list)


class ValidatorResult(BaseModel):
    accepted: bool
    reasons: list[str] = Field(default_factory=list)
    repaired: DailyBriefingSynthesis | None = None


class DailyBriefingDocument(BaseModel):
    userId: str
    dateKey: str
    timezone: str
    periodStartUtc: datetime
    periodEndUtc: datetime
    status: BriefingStatus = BriefingStatus.PENDING
    headline: str = ""
    overview: str = ""
    highlights: list[BriefingItem] = Field(default_factory=list)
    importantMoments: list[BriefingItem] = Field(default_factory=list)
    completed: list[BriefingItem] = Field(default_factory=list)
    pendingTasks: list[BriefingItem] = Field(default_factory=list)
    decisions: list[BriefingItem] = Field(default_factory=list)
    followUps: list[BriefingItem] = Field(default_factory=list)
    tomorrowFocus: list[BriefingItem] = Field(default_factory=list)
    insights: list[InsightCard] = Field(default_factory=list)
    people: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    tasks: list[TaskCard] = Field(default_factory=list)
    meetings: list[MeetingCard] = Field(default_factory=list)
    missedCandidates: list[BriefingItem] = Field(default_factory=list)
    sourceStats: SourceStats = Field(default_factory=SourceStats)
    pipelineVersion: str = PIPELINE_VERSION
    skipReason: str | None = None
    error: str | None = None
    claimedBy: str | None = None
    claimedAt: datetime | None = None
    generatedAt: datetime | None = None
    createdAt: datetime = Field(default_factory=utc_now)
    updatedAt: datetime = Field(default_factory=utc_now)


class ActivityBundle(BaseModel):
    transcripts: list[dict[str, Any]] = Field(default_factory=list)
    tasks: list[dict[str, Any]] = Field(default_factory=list)
    notes: list[dict[str, Any]] = Field(default_factory=list)
    reminders: list[dict[str, Any]] = Field(default_factory=list)
    events: list[dict[str, Any]] = Field(default_factory=list)
    pendingTranscriptCount: int = 0

    @property
    def has_activity(self) -> bool:
        return bool(self.transcripts or self.tasks or self.notes or self.reminders or self.events)


class TranscriptWindow(BaseModel):
    index: int
    items: list[dict[str, Any]]
    tokenCount: int = 0
