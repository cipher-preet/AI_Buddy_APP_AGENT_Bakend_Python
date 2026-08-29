from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from services.conversation.models import EvidenceSpan, ExtractedNote, ExtractedTask


class EventKind(str, Enum):
    REQUEST = "REQUEST"
    COMMITMENT = "COMMITMENT"
    ASSIGNMENT = "ASSIGNMENT"
    DECISION = "DECISION"
    PROPOSAL = "PROPOSAL"
    REQUIREMENT = "REQUIREMENT"
    ISSUE = "ISSUE"
    STATE = "STATE"
    RESULT = "RESULT"
    FACT = "FACT"
    FOLLOW_UP = "FOLLOW_UP"
    DEADLINE = "DEADLINE"
    COMPLETION = "COMPLETION"
    CANCELLATION = "CANCELLATION"
    CONTRADICTION = "CONTRADICTION"
    CONSTRAINT = "CONSTRAINT"
    IMPORTANT_CONTEXT = "IMPORTANT_CONTEXT"
    OPEN_QUESTION = "OPEN_QUESTION"
    NOISE = "NOISE"


class ThreadRelation(str, Enum):
    SAME_THREAD = "SAME_THREAD"
    UPDATES = "UPDATES"
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    COMPLETES = "COMPLETES"
    CANCELS = "CANCELS"
    SUPERSEDES = "SUPERSEDES"
    RELATED_BUT_DISTINCT = "RELATED_BUT_DISTINCT"
    UNRELATED = "UNRELATED"
    RELATED_TO = "RELATED_TO"


class EventDisposition(str, Enum):
    TASK = "TASK"
    NOTE = "NOTE"
    INTENTIONALLY_NON_PUBLISHABLE = "INTENTIONALLY_NON_PUBLISHABLE"
    SUPERSEDED = "SUPERSEDED"
    DUPLICATE = "DUPLICATE"
    REJECTED = "REJECTED"


class MemoryDisposition(str, Enum):
    PUBLISHED_NOTE = "PUBLISHED_NOTE"
    DUPLICATE = "DUPLICATE"
    SUPERSEDED = "SUPERSEDED"
    LOW_VALUE = "LOW_VALUE"
    UNSUPPORTED = "UNSUPPORTED"
    RELATED_CONTEXT_ONLY = "RELATED_CONTEXT_ONLY"
    REJECTED_WITH_REASON = "REJECTED_WITH_REASON"


class ActionDisposition(str, Enum):
    PUBLISHED_TASK = "PUBLISHED_TASK"
    DUPLICATE = "DUPLICATE"
    SUPERSEDED = "SUPERSEDED"
    UNSUPPORTED = "UNSUPPORTED"
    UNRESOLVED_OBJECT = "UNRESOLVED_OBJECT"
    AMBIGUOUS = "AMBIGUOUS"
    INTENTIONALLY_NONPUBLISHABLE = "INTENTIONALLY_NONPUBLISHABLE"
    VALIDATION_REJECTED = "VALIDATION_REJECTED"


class BlockDisposition(str, Enum):
    PRODUCED_EVENTS = "PRODUCED_EVENTS"
    NO_EVENT = "NO_EVENT"


class SemanticUnitDisposition(str, Enum):
    EVENT_CREATED = "EVENT_CREATED"
    MERGED_WITH_EVENT = "MERGED_WITH_EVENT"
    LOW_VALUE = "LOW_VALUE"
    NOISE = "NOISE"
    UNSUPPORTED = "UNSUPPORTED"
    DUPLICATE = "DUPLICATE"
    AMBIGUOUS = "AMBIGUOUS"


class ValidationAction(str, Enum):
    ACCEPT = "ACCEPT"
    REMOVE_BAD_EVIDENCE = "REMOVE_BAD_EVIDENCE"
    SPLIT = "SPLIT"
    REWRITE_FROM_EXISTING_EVENTS = "REWRITE_FROM_EXISTING_EVENTS"
    REJECT = "REJECT"


class NLILabel(str, Enum):
    ENTAILED = "entailed"
    NEUTRAL = "neutral"
    CONTRADICTED = "contradicted"


ACTION_EVENT_KINDS = frozenset(
    {
        EventKind.REQUEST,
        EventKind.COMMITMENT,
        EventKind.ASSIGNMENT,
        EventKind.FOLLOW_UP,
        EventKind.DEADLINE,
        EventKind.COMPLETION,
        EventKind.CANCELLATION,
    }
)
ACTION_ROLES = frozenset({"REQUEST", "COMMITMENT", "ASSIGNMENT", "INSTRUCTION", "FOLLOW_UP"})
ACTION_STRENGTHS = frozenset({"NONE", "POSSIBLE", "EXPLICIT"})
OBJECT_GROUNDING_TYPES = frozenset({"EXPLICIT", "LOCAL_COREFERENCE", "INFERRED", "UNRESOLVED"})
ABSTAIN_UNRESOLVED_OBJECT = "ABSTAIN_UNRESOLVED_OBJECT"
SEMANTIC_RELATIONS = frozenset({"SAME_THREAD", "RELATED_BUT_DISTINCT", "UNRELATED"})
MEMORY_EVENT_KINDS = frozenset(
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
NON_PUBLISHABLE_KINDS = frozenset({EventKind.NOISE})

GENERIC_TASK_TITLES = frozenset(
    {
        "complete pending task",
        "complete the pending task",
        "complete it",
        "complete this",
        "fix issue",
        "fix the issue",
        "fix it",
        "handle it",
        "handle problem",
        "handle the problem",
        "do the work",
        "do this",
        "do it",
        "check problem",
        "check the problem",
        "pending task",
    }
)
GENERIC_ACTION_OBJECTS = frozenset(
    {
        "it",
        "this",
        "that",
        "issue",
        "problem",
        "task",
        "work",
        "thing",
        "pending",
        "item",
        "stuff",
    }
)
DEICTIC_OR_TIME = frozenset(
    {
        "tomorrow",
        "today",
        "yesterday",
        "later",
        "soon",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "kal",
        "aaj",
    }
)
ACTION_PRONOUNS = frozenset(
    {
        "it",
        "this",
        "that",
        "them",
        "these",
        "those",
        "usko",
        "uska",
        "uski",
        "iski",
        "isko",
        "yeh",
        "woh",
        "unhe",
        "usko",
        "usse",
    }
)


class CleanedTranscriptRecord(BaseModel):
    sequenceId: int
    chunkId: str
    sourceId: str
    speaker: str | None = None
    rawText: str
    timestampMs: int | None = None
    sessionId: str
    spaceId: str
    userId: str
    languageCode: str | None = None
    excluded: bool = False
    exclusionReason: str | None = None


class CleaningLedger(BaseModel):
    totalSequences: int = 0
    usefulSequences: int = 0
    excludedStructuralSequences: int = 0
    records: list[CleanedTranscriptRecord] = Field(default_factory=list)
    useful: list[CleanedTranscriptRecord] = Field(default_factory=list)
    excluded: list[CleanedTranscriptRecord] = Field(default_factory=list)
    accountedSequenceIds: list[int] = Field(default_factory=list)

    @property
    def complete(self) -> bool:
        return self.totalSequences == self.usefulSequences + self.excludedStructuralSequences


class MicroBlock(BaseModel):
    microBlockId: str
    sequenceStart: int
    sequenceEnd: int
    sequenceIds: list[int] = Field(default_factory=list)
    sourceIds: list[str] = Field(default_factory=list)
    text: str
    tokenCount: int = 0
    embedding: list[float] | None = None
    overlapSequenceIds: list[int] = Field(default_factory=list)
    speakerIds: list[str] = Field(default_factory=list)
    informationDensity: float = 0.0


class LocalTopic(BaseModel):
    topicId: str
    label: str
    microBlockIds: list[str] = Field(default_factory=list)
    sequenceStart: int
    sequenceEnd: int
    sequenceIds: list[int] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    embedding: list[float] | None = None
    text: str = ""
    coherence: float = 0.0
    boundaryReason: str = ""
    tokenCount: int = 0


class ActionSignal(BaseModel):
    isActionable: bool = False
    role: str | None = None
    actionStrength: str | None = None
    verb: str | None = None
    object: str | None = None
    rawActionObject: str | None = None
    canonicalActionObject: str | None = None
    objectGroundingType: str | None = None
    actor: str | None = None
    deadline: str | None = None
    artifactStatus: str | None = None


class MemorySignal(BaseModel):
    isMemoryWorthy: bool = False
    importance: str | None = None
    reason: str | None = None


class FieldEvidence(BaseModel):
    actionVerb: list[EvidenceSpan] = Field(default_factory=list)
    actionObject: list[EvidenceSpan] = Field(default_factory=list)
    actor: list[EvidenceSpan] = Field(default_factory=list)
    deadline: list[EvidenceSpan] = Field(default_factory=list)


class AtomicEvent(BaseModel):
    eventId: str
    topicId: str
    kind: EventKind
    meaning: str
    actor: str | None = None
    object: str | None = None
    timeExpression: str | None = None
    entities: list[str] = Field(default_factory=list)
    uncertainty: list[str] = Field(default_factory=list)
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    microBlockIds: list[str] = Field(default_factory=list)
    sourceIds: list[str] = Field(default_factory=list)
    sequenceIds: list[int] = Field(default_factory=list)
    embedding: list[float] | None = None
    threadId: str | None = None
    channel: Literal["action", "memory", "other"] = "other"
    disposition: EventDisposition | None = None
    dispositionReason: str | None = None
    memoryDisposition: MemoryDisposition | None = None
    memoryDispositionReason: str | None = None
    actionDisposition: ActionDisposition | None = None
    actionDispositionReason: str | None = None
    conversationId: str = ""
    userId: str = ""
    spaceId: str = ""
    actionSignal: ActionSignal | None = None
    memorySignal: MemorySignal | None = None
    fieldEvidence: FieldEvidence | None = None


class ThreadLink(BaseModel):
    fromEventId: str
    toEventId: str
    relation: ThreadRelation
    score: float = 0.0
    crossWindow: bool = False


class GlobalThread(BaseModel):
    threadId: str
    label: str
    eventIds: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    latestState: str = ""
    sequenceStart: int = 0
    sequenceEnd: int = 0
    embedding: list[float] | None = None


class CoverageBlockRecord(BaseModel):
    microBlockId: str
    sequenceIds: list[int] = Field(default_factory=list)
    disposition: BlockDisposition
    reason: str = ""
    eventIds: list[str] = Field(default_factory=list)


class CoverageEventRecord(BaseModel):
    eventId: str
    disposition: EventDisposition
    reason: str = ""
    artifactTitle: str | None = None
    memoryDisposition: str | None = None
    actionDisposition: str | None = None


class CoverageSemanticUnitRecord(BaseModel):
    microBlockId: str = ""
    meaning: str = ""
    kind: str | None = None
    disposition: SemanticUnitDisposition | None = None
    eventId: str | None = None
    reason: str = ""
    sequenceIds: list[int] = Field(default_factory=list)


class CoverageLedger(BaseModel):
    total_raw_sequences: int = 0
    useful_sequences: int = 0
    excluded_structural_sequences: int = 0
    micro_blocks: int = 0
    topics: int = 0
    events: int = 0
    action_events: int = 0
    memory_events: int = 0
    other_events: int = 0
    tasks_generated: int = 0
    notes_generated: int = 0
    rejected_events: int = 0
    unaccounted_blocks: int = 0
    memoryPublished: int = 0
    memoryDuplicates: int = 0
    memorySuperseded: int = 0
    memoryLowValue: int = 0
    memoryUnsupported: int = 0
    memoryRelatedContext: int = 0
    memoryRejected: int = 0
    memoryUnaccounted: int = 0
    memoryUpdates: int = 0
    memoryCoverageFailure: bool = False
    actionPublished: int = 0
    actionDuplicates: int = 0
    actionSuperseded: int = 0
    actionUnsupported: int = 0
    actionUnresolved: int = 0
    actionAmbiguous: int = 0
    actionNonpublishable: int = 0
    actionRejected: int = 0
    actionUnaccounted: int = 0
    actionCoverageFailure: bool = False
    semanticUnitsDetected: int = 0
    semanticUnitsCreated: int = 0
    semanticUnitsMerged: int = 0
    semanticUnitsLowValue: int = 0
    semanticUnitsNoise: int = 0
    semanticUnitsAmbiguous: int = 0
    semanticUnitsUnsupported: int = 0
    semanticUnitsDuplicate: int = 0
    unaccountedSemanticUnits: int = 0
    semanticCoverage: float = 1.0
    semanticCoverageFailure: bool = False
    semanticReviewRan: bool = False
    blocks: list[CoverageBlockRecord] = Field(default_factory=list)
    eventRecords: list[CoverageEventRecord] = Field(default_factory=list)
    semanticUnitRecords: list[CoverageSemanticUnitRecord] = Field(default_factory=list)
    suspicious: list[str] = Field(default_factory=list)
    hardFailure: bool = False

    def as_metrics(self) -> dict[str, Any]:
        return {
            "total_raw_sequences": self.total_raw_sequences,
            "useful_sequences": self.useful_sequences,
            "excluded_structural_sequences": self.excluded_structural_sequences,
            "micro_blocks": self.micro_blocks,
            "topics": self.topics,
            "events": self.events,
            "action_events": self.action_events,
            "memory_events": self.memory_events,
            "other_events": self.other_events,
            "tasks_generated": self.tasks_generated,
            "notes_generated": self.notes_generated,
            "rejected_events": self.rejected_events,
            "unaccounted_blocks": self.unaccounted_blocks,
            "memoryPublished": self.memoryPublished,
            "memoryDuplicates": self.memoryDuplicates,
            "memorySuperseded": self.memorySuperseded,
            "memoryLowValue": self.memoryLowValue,
            "memoryUnsupported": self.memoryUnsupported,
            "memoryRelatedContext": self.memoryRelatedContext,
            "memoryRejected": self.memoryRejected,
            "memoryUnaccounted": self.memoryUnaccounted,
            "memoryUpdates": self.memoryUpdates,
            "memoryCoverageFailure": self.memoryCoverageFailure,
            "actionPublished": self.actionPublished,
            "actionDuplicates": self.actionDuplicates,
            "actionSuperseded": self.actionSuperseded,
            "actionUnsupported": self.actionUnsupported,
            "actionUnresolved": self.actionUnresolved,
            "actionAmbiguous": self.actionAmbiguous,
            "actionNonpublishable": self.actionNonpublishable,
            "actionRejected": self.actionRejected,
            "actionUnaccounted": self.actionUnaccounted,
            "actionCoverageFailure": self.actionCoverageFailure,
            "semanticUnitsDetected": self.semanticUnitsDetected,
            "semanticUnitsCreated": self.semanticUnitsCreated,
            "semanticUnitsMerged": self.semanticUnitsMerged,
            "semanticUnitsLowValue": self.semanticUnitsLowValue,
            "semanticUnitsNoise": self.semanticUnitsNoise,
            "semanticUnitsAmbiguous": self.semanticUnitsAmbiguous,
            "semanticUnitsUnsupported": self.semanticUnitsUnsupported,
            "semanticUnitsDuplicate": self.semanticUnitsDuplicate,
            "unaccountedSemanticUnits": self.unaccountedSemanticUnits,
            "semanticCoverage": self.semanticCoverage,
            "semanticCoverageFailure": self.semanticCoverageFailure,
            "semanticReviewRan": self.semanticReviewRan,
            "suspicious": list(self.suspicious),
            "hardFailure": self.hardFailure,
        }


class StageMetrics(BaseModel):
    name: str
    durationMs: int = 0
    llmCalls: int = 0
    embeddingCalls: int = 0
    embeddingItems: int = 0
    inputTokens: int = 0
    outputTokens: int = 0
    extra: dict[str, Any] = Field(default_factory=dict)


class ModelRouteRecord(BaseModel):
    stage: str
    capability: str
    provider: str = ""
    model: str = ""
    fallback: bool = False
    requested: str = ""


class PipelineObservability(BaseModel):
    stages: list[StageMetrics] = Field(default_factory=list)
    logs: list[str] = Field(default_factory=list)
    modelRoutes: list[ModelRouteRecord] = Field(default_factory=list)
    comparisonCount: int = 0
    embeddingRequests: int = 0
    embeddingItems: int = 0
    gemmaCalls: int = 0
    gptOss120bCalls: int = 0
    gptOss20bCalls: int = 0
    otherLlmCalls: int = 0
    inputTokens: int = 0
    outputTokens: int = 0
    fallbackCount: int = 0
    retryCount: int = 0
    providerErrors: int = 0
    asyncLifecycleErrors: int = 0
    estimatedCostUsd: float | None = None
    sessionId: str = ""
    workerId: str = ""
    mode: str = ""
    rawSequences: int = 0
    usefulSequences: int = 0
    microBlockCount: int = 0
    topicCount: int = 0
    eventCount: int = 0
    threadCount: int = 0
    actionEvents: int = 0
    memoryEvents: int = 0
    tasksPublished: int = 0
    notesPublished: int = 0
    genericTasks: int = 0
    mixedThreads: int = 0
    duplicates: int = 0
    unaccountedBlocks: int = 0
    memoryCoverageFailures: int = 0
    actionCoverageFailures: int = 0
    semanticCoverageFailures: int = 0
    unaccountedSemanticUnits: int = 0
    semanticUnitsDetected: int = 0
    semanticUnitsCreated: int = 0
    atomicEvents: int = 0
    actionableEvents: int = 0
    explicitActionEvents: int = 0
    groundedActionObjects: int = 0
    actionChannelEvents: int = 0
    taskSynthesisInputEvents: int = 0
    taskCandidates: int = 0
    taskValidationAccepted: int = 0
    taskValidationRejected: int = 0
    tasksPersisted: int = 0
    tasksReturnedByApi: int = 0
    alerts: list[dict[str, Any]] = Field(default_factory=list)
    pipelineVersion: str = ""
    eventSchemaVersion: str = ""
    promptVersion: str = ""
    artifactPipelineVersion: str = ""

    def llm_calls(self) -> int:
        staged = sum(stage.llmCalls for stage in self.stages)
        routed = self.gemmaCalls + self.gptOss120bCalls + self.gptOss20bCalls + self.otherLlmCalls
        return max(staged, routed)

    def embedding_calls(self) -> int:
        staged = sum(stage.embeddingCalls for stage in self.stages)
        return max(staged, self.embeddingRequests)

    def tokens(self) -> int:
        staged = sum(stage.inputTokens + stage.outputTokens for stage in self.stages)
        return max(staged, self.inputTokens + self.outputTokens)


class EventPipelineResult(BaseModel):
    tasks: list[ExtractedTask] = Field(default_factory=list)
    notes: list[ExtractedNote] = Field(default_factory=list)
    events: list[AtomicEvent] = Field(default_factory=list)
    threads: list[GlobalThread] = Field(default_factory=list)
    topics: list[LocalTopic] = Field(default_factory=list)
    microBlocks: list[MicroBlock] = Field(default_factory=list)
    cleaning: CleaningLedger | None = None
    coverage: CoverageLedger | None = None
    observability: PipelineObservability = Field(default_factory=PipelineObservability)
    snapshots: dict[str, Any] = Field(default_factory=dict)
    cost: dict[str, Any] = Field(default_factory=dict)
    provider: str = "event-pipeline"
    model: str = "hierarchical-v1"
    diagnostics: dict[str, Any] = Field(default_factory=dict)
