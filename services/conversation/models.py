from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from bson import ObjectId
from pydantic import BaseModel, Field, field_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str | None = None) -> ObjectId:
    return ObjectId()


class ConversationStatus(str, Enum):
    RECORDING = "RECORDING"
    STOP_REQUESTED = "STOP_REQUESTED"
    WAITING_FOR_TRANSCRIPTS = "WAITING_FOR_TRANSCRIPTS"
    READY_FOR_PROCESSING = "READY_FOR_PROCESSING"
    PROCESSING = "PROCESSING"
    VALIDATING = "VALIDATING"
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    RETRY_PENDING = "RETRY_PENDING"
    FAILED = "FAILED"


VALID_TRANSITIONS: dict[ConversationStatus, set[ConversationStatus]] = {
    ConversationStatus.RECORDING: {
        ConversationStatus.STOP_REQUESTED,
        ConversationStatus.WAITING_FOR_TRANSCRIPTS,
        ConversationStatus.FAILED,
    },
    ConversationStatus.STOP_REQUESTED: {
        ConversationStatus.WAITING_FOR_TRANSCRIPTS,
        ConversationStatus.READY_FOR_PROCESSING,
        ConversationStatus.PARTIAL,
        ConversationStatus.RETRY_PENDING,
        ConversationStatus.FAILED,
    },
    ConversationStatus.WAITING_FOR_TRANSCRIPTS: {
        ConversationStatus.READY_FOR_PROCESSING,
        ConversationStatus.PARTIAL,
        ConversationStatus.RETRY_PENDING,
        ConversationStatus.FAILED,
    },
    ConversationStatus.READY_FOR_PROCESSING: {
        ConversationStatus.PROCESSING,
        ConversationStatus.RETRY_PENDING,
        ConversationStatus.FAILED,
    },
    ConversationStatus.PROCESSING: {
        ConversationStatus.VALIDATING,
        ConversationStatus.RETRY_PENDING,
        ConversationStatus.FAILED,
    },
    ConversationStatus.VALIDATING: {
        ConversationStatus.COMPLETED,
        ConversationStatus.PARTIAL,
        ConversationStatus.RETRY_PENDING,
        ConversationStatus.FAILED,
    },
    ConversationStatus.PARTIAL: {
        ConversationStatus.PROCESSING,
        ConversationStatus.RETRY_PENDING,
        ConversationStatus.FAILED,
        ConversationStatus.COMPLETED,
    },
    ConversationStatus.RETRY_PENDING: {
        ConversationStatus.WAITING_FOR_TRANSCRIPTS,
        ConversationStatus.READY_FOR_PROCESSING,
        ConversationStatus.PROCESSING,
        ConversationStatus.FAILED,
    },
    ConversationStatus.FAILED: {ConversationStatus.RETRY_PENDING},
    ConversationStatus.COMPLETED: set(),
}


def assert_valid_transition(current: ConversationStatus, target: ConversationStatus) -> None:
    if current == target:
        return
    if target not in VALID_TRANSITIONS.get(current, set()):
        raise ValueError(f"Invalid conversation transition: {current} -> {target}")


class STTStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TranscriptProcessingStatus(str, Enum):
    UNPROCESSED = "unprocessed"
    PROCESSED = "processed"
    ARCHIVED = "archived"
    EXPIRED = "expired"


class ExtractionRunStatus(str, Enum):
    PROCESSING = "PROCESSING"
    VALIDATING = "VALIDATING"
    READY_TO_PUBLISH = "READY_TO_PUBLISH"
    PUBLISHED = "PUBLISHED"
    FAILED = "FAILED"


class FailureClass(str, Enum):
    TRANSIENT = "TRANSIENT"
    RATE_LIMIT = "RATE_LIMIT"
    INVALID_INPUT = "INVALID_INPUT"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    SCHEMA_ERROR = "SCHEMA_ERROR"
    MISSING_DATA = "MISSING_DATA"
    CONFLICT = "CONFLICT"
    PERMANENT = "PERMANENT"


class ConversationDocument(BaseModel):
    id: Any = Field(default_factory=new_id, alias="_id")
    userId: Any
    spaceId: Any
    status: ConversationStatus = ConversationStatus.RECORDING
    startedAt: datetime = Field(default_factory=utc_now)
    stoppedAt: datetime | None = None
    stoppedAtClient: datetime | None = None
    expectedLastSequence: int | None = None
    receivedAudioChunkCount: int = 0
    completedTranscriptChunkCount: int = 0
    failedTranscriptChunkCount: int = 0
    processingVersion: int = 1
    activeExtractionRunId: Any | None = None
    missingSequences: list[int] = Field(default_factory=list)
    lastError: str | None = None
    lastActivityAt: datetime = Field(default_factory=utc_now)
    processedAt: datetime | None = None
    createdAt: datetime = Field(default_factory=utc_now)
    updatedAt: datetime = Field(default_factory=utc_now)

    class Config:
        populate_by_name = True


class AudioChunkMetadata(BaseModel):
    conversationId: Any
    userId: Any
    spaceId: Any
    chunkId: str
    sequenceNumber: int = Field(ge=0)
    capturedAt: datetime | None = None
    durationMs: int | None = Field(default=None, ge=0)
    filePath: str
    filename: str
    contentType: str | None = None
    createdAt: datetime = Field(default_factory=utc_now)


class TranscriptChunkDocument(BaseModel):
    id: Any = Field(default_factory=new_id, alias="_id")
    conversationId: Any
    userId: Any
    spaceId: Any
    chunkId: str
    sequenceNumber: int = Field(ge=0)
    rawText: str | None = None
    normalizedText: str | None = None
    languageCode: str | None = None
    sttProvider: str = "sarvam"
    sttRequestId: str | None = None
    sttStatus: STTStatus = STTStatus.PENDING
    processingStatus: TranscriptProcessingStatus = TranscriptProcessingStatus.UNPROCESSED
    startTimeMs: int | None = None
    endTimeMs: int | None = None
    audioFilePath: str | None = None
    sttAttempts: int = 0
    lastError: str | None = None
    archiveRef: str | None = None
    createdAt: datetime = Field(default_factory=utc_now)
    updatedAt: datetime = Field(default_factory=utc_now)
    expiresAt: datetime | None = None

    class Config:
        populate_by_name = True


class EvidenceSpan(BaseModel):
    sequenceStart: int
    sequenceEnd: int
    text: str

    @field_validator("text")
    @classmethod
    def evidence_not_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("evidence text is required")
        return value


Operation = Literal["CREATE", "UPDATE", "COMPLETE", "CANCEL", "NO_ACTION", "NEEDS_CONFIRMATION"]


class ExtractedTask(BaseModel):
    title: str
    operation: Operation
    existingTaskId: str | None = None
    ownerText: str | None = None
    ownerUserId: str | None = None
    dueDateText: str | None = None
    dueDateResolved: str | None = None
    dueDateStatus: Literal["resolved", "ambiguous", "none"] = "none"
    confidence: float = Field(ge=0, le=1)
    needsConfirmation: bool = False
    sourceConversationId: str
    fingerprint: str | None = None
    changes: dict[str, Any] = Field(default_factory=dict)
    evidence: list[EvidenceSpan]


class ExtractedNote(BaseModel):
    title: str
    body: str
    confidence: float = Field(ge=0, le=1)
    sourceConversationId: str
    fingerprint: str | None = None
    evidence: list[EvidenceSpan]


class ExtractedDecision(BaseModel):
    title: str
    status: Literal["confirmed_decision", "proposal", "idea", "unresolved_discussion"]
    confidence: float = Field(ge=0, le=1)
    sourceConversationId: str
    evidence: list[EvidenceSpan]


class ExtractedIssue(BaseModel):
    title: str
    kind: Literal["blocker", "risk", "open_question", "missing_information"]
    confidence: float = Field(ge=0, le=1)
    sourceConversationId: str
    evidence: list[EvidenceSpan]


class Segment(BaseModel):
    segmentId: str
    conversationId: str
    sequenceStart: int
    sequenceEnd: int
    text: str
    tokenCount: int


class SectionExtractionResult(BaseModel):
    segmentId: str
    tasks: list[ExtractedTask] = Field(default_factory=list)
    notes: list[ExtractedNote] = Field(default_factory=list)
    decisions: list[ExtractedDecision] = Field(default_factory=list)
    issues: list[ExtractedIssue] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CoverageItem(BaseModel):
    sequenceStart: int
    sequenceEnd: int
    label: Literal["covered", "small_talk", "duplicate", "non_actionable", "missing", "uncertain"]
    reason: str


class CoverageReport(BaseModel):
    score: float = Field(ge=0, le=1)
    criticalMissingCount: int = 0
    items: list[CoverageItem] = Field(default_factory=list)


class ExtractionRunDocument(BaseModel):
    id: Any = Field(default_factory=new_id, alias="_id")
    conversationId: Any
    userId: Any
    spaceId: Any
    processingVersion: int
    status: ExtractionRunStatus = ExtractionRunStatus.PROCESSING
    segmentCount: int = 0
    processedSegmentCount: int = 0
    coverageScore: float | None = None
    validationErrors: list[dict[str, Any]] = Field(default_factory=list)
    warningCount: int = 0
    provider: str
    model: str
    promptVersions: dict[str, str] = Field(default_factory=dict)
    tokenUsage: dict[str, int] = Field(default_factory=dict)
    checkpoints: dict[str, Any] = Field(default_factory=dict)
    stagedTasks: list[ExtractedTask] = Field(default_factory=list)
    stagedNotes: list[ExtractedNote] = Field(default_factory=list)
    stagedDecisions: list[ExtractedDecision] = Field(default_factory=list)
    stagedIssues: list[ExtractedIssue] = Field(default_factory=list)
    startedAt: datetime = Field(default_factory=utc_now)
    completedAt: datetime | None = None
    updatedAt: datetime = Field(default_factory=utc_now)

    class Config:
        populate_by_name = True


class ConversationSummaryDocument(BaseModel):
    id: Any = Field(default_factory=new_id, alias="_id")
    conversationId: Any
    userId: Any
    spaceId: Any
    summary: str
    topics: list[str] = Field(default_factory=list)
    importantFacts: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    openQuestions: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    taskIds: list[Any] = Field(default_factory=list)
    languageCodes: list[str] = Field(default_factory=list)
    processingVersion: int
    modelProvider: str
    modelName: str
    promptVersion: str
    createdAt: datetime = Field(default_factory=utc_now)

    class Config:
        populate_by_name = True


class SpaceMemoryDocument(BaseModel):
    id: Any = Field(default_factory=new_id, alias="_id")
    userId: Any
    spaceId: Any
    currentSummary: str = ""
    importantFacts: list[str] = Field(default_factory=list)
    importantDecisions: list[str] = Field(default_factory=list)
    openQuestionIds: list[Any] = Field(default_factory=list)
    activeTaskIds: list[Any] = Field(default_factory=list)
    blockerIds: list[Any] = Field(default_factory=list)
    recentConversationSummaryIds: list[Any] = Field(default_factory=list)
    lastUpdatedConversationId: Any | None = None
    version: int = 1
    updatedAt: datetime = Field(default_factory=utc_now)

    class Config:
        populate_by_name = True
