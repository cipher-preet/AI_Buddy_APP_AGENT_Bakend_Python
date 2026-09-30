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
    priority: str = ""
    due: str = ""
    dueStatus: str = ""
    space: str = ""
    reason: str = ""


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
    chatMessageCount: int = 0
    openTaskCount: int = 0


class FocusItem(BaseModel):
    id: str
    title: str
    why: str = ""
    timeHint: str = ""
    taskId: str = ""


class AgendaCard(BaseModel):
    id: str
    kind: str
    time: str = ""
    endTime: str = ""
    title: str
    location: str = ""
    prep: str = ""


class BriefingStats(BaseModel):
    completedTasks: int = 0
    openTasks: int = 0
    overdueTasks: int = 0
    dueToday: int = 0
    meetings: int = 0
    reminders: int = 0
    conversations: int = 0
    chatMessages: int = 0
    notes: int = 0


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
    focus: list[FocusItem] = Field(default_factory=list)
    agenda: list[AgendaCard] = Field(default_factory=list)
    risks: list[BriefingItem] = Field(default_factory=list)
    stats: BriefingStats = Field(default_factory=BriefingStats)
    planDateKey: str = ""


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
    focus: list[FocusItem] = Field(default_factory=list)
    agenda: list[AgendaCard] = Field(default_factory=list)
    risks: list[BriefingItem] = Field(default_factory=list)
    stats: BriefingStats = Field(default_factory=BriefingStats)
    planDateKey: str = ""
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
    chats: list[dict[str, Any]] = Field(default_factory=list)
    openTasks: list[dict[str, Any]] = Field(default_factory=list)
    planDateKey: str = ""
    planEvents: list[dict[str, Any]] = Field(default_factory=list)
    planReminders: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def has_activity(self) -> bool:
        return bool(self.transcripts or self.tasks or self.notes or self.reminders or self.events)

    @property
    def has_plan_context(self) -> bool:
        return bool(self.chats or self.planEvents or self.planReminders)


class TranscriptWindow(BaseModel):
    index: int
    items: list[dict[str, Any]]
    tokenCount: int = 0


# --- v2 LLM contracts: models cite short refs (T1, E2, S5...), never raw ids. ---


class DraftItem(BaseModel):
    title: str = ""
    detail: str = ""
    refs: list[str] = Field(default_factory=list)


class DraftWorkItem(DraftItem):
    status: str = ""
    owner: str = ""
    due: str = ""


class DraftFollowUp(DraftItem):
    person: str = ""


class WindowDigest(BaseModel):
    summary: str = ""
    workItems: list[DraftWorkItem] = Field(default_factory=list)
    decisions: list[DraftItem] = Field(default_factory=list)
    followUps: list[DraftFollowUp] = Field(default_factory=list)
    moments: list[DraftItem] = Field(default_factory=list)
    people: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)


class DraftFocus(BaseModel):
    title: str = ""
    why: str = ""
    timeHint: str = ""
    taskRef: str = ""


class DraftTaskPriority(BaseModel):
    ref: str = ""
    priority: str = ""
    reason: str = ""


class DraftAgendaNote(BaseModel):
    ref: str = ""
    prep: str = ""


class DraftInsight(BaseModel):
    title: str = ""
    body: str = ""
    whyItMatters: str = ""
    refs: list[str] = Field(default_factory=list)


class BriefingDraft(BaseModel):
    headline: str = ""
    overview: str = ""
    focus: list[DraftFocus] = Field(default_factory=list)
    taskPriorities: list[DraftTaskPriority] = Field(default_factory=list)
    agendaNotes: list[DraftAgendaNote] = Field(default_factory=list)
    followUps: list[DraftFollowUp] = Field(default_factory=list)
    risks: list[DraftItem] = Field(default_factory=list)
    highlights: list[DraftItem] = Field(default_factory=list)
    decisions: list[DraftItem] = Field(default_factory=list)
    completed: list[DraftItem] = Field(default_factory=list)
    moments: list[DraftItem] = Field(default_factory=list)
    insights: list[DraftInsight] = Field(default_factory=list)
    missedCandidates: list[DraftItem] = Field(default_factory=list)
    people: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
