"""Structured schemas for the meeting extract-consolidate-verify path."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from services.conversation.models import ExtractedNote, ExtractedTask


def _require_every_property(schema: dict[str, Any]) -> None:
    properties = schema.get("properties") or {}
    if properties:
        schema["required"] = list(properties)


class WireModel(BaseModel):
    """LLM-facing response model.

    Constrained decoders (json_schema / guided decoding) routinely skip
    properties that are not listed as required, silently dropping fields that
    carry defaults. The wire schema therefore marks every property required,
    while Python-side defaults keep parsing lenient for providers that still
    omit a field.
    """

    model_config = ConfigDict(json_schema_extra=_require_every_property)


class CandidateKind(str, Enum):
    ACTION = "ACTION"
    REQUIREMENT = "REQUIREMENT"
    DECISION = "DECISION"
    FACT = "FACT"
    RATIONALE = "RATIONALE"
    ISSUE = "ISSUE"
    IDEA = "IDEA"
    QUESTION = "QUESTION"


class VerifierVerdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"


class TranscriptTurn(BaseModel):
    sequence_id: int
    speaker: str | None = None
    raw_text: str = ""


class ExtractionWindow(BaseModel):
    window_id: str
    window_index: int
    sequence_start: int
    sequence_end: int
    sequence_ids: list[int] = Field(default_factory=list)
    owned_sequence_ids: list[int] = Field(default_factory=list)
    overlap_sequence_ids: list[int] = Field(default_factory=list)
    text: str = ""
    token_count: int = 0


class MeetingCandidate(BaseModel):
    candidateId: str
    kind: CandidateKind = CandidateKind.FACT
    meaning: str
    evidenceSequences: list[int] = Field(default_factory=list)
    owner: str | None = None
    dueDate: str | None = None
    topic: str | None = None
    sourceWindowId: str = ""
    sourceWindowIndex: int = 0


class CandidateLLMItem(WireModel):
    candidateId: str | None = None
    kind: CandidateKind = CandidateKind.FACT
    topic: str | None = None
    meaning: str
    evidenceSequences: list[int] = Field(min_length=1)
    owner: str | None = None
    dueDate: str | None = None


class MeetingCandidateExtractorResponse(WireModel):
    candidates: list[CandidateLLMItem] = Field(default_factory=list)


class ConsolidatedTaskItem(WireModel):
    title: str
    description: str = ""
    acceptanceCriteria: str = ""
    priority: str | None = None
    topic: str | None = None
    owner: str | None = None
    dueDate: str | None = None
    sourceCandidateIds: list[str] = Field(default_factory=list)
    evidenceSequences: list[int] = Field(default_factory=list)


class ConsolidatedNoteItem(WireModel):
    title: str
    body: str
    topic: str | None = None
    sourceCandidateIds: list[str] = Field(default_factory=list)
    evidenceSequences: list[int] = Field(default_factory=list)


class MeetingConsolidatorResponse(WireModel):
    tasks: list[ConsolidatedTaskItem] = Field(default_factory=list)
    notes: list[ConsolidatedNoteItem] = Field(default_factory=list)
    discardedCandidateIds: list[str] = Field(default_factory=list)
    summary: str = ""
    topics: list[str] = Field(default_factory=list)


class CoverageAttachment(WireModel):
    artifactId: str
    candidateIds: list[str] = Field(default_factory=list)
    addition: str = ""


class MeetingCoverageResponse(WireModel):
    attachments: list[CoverageAttachment] = Field(default_factory=list)
    tasks: list[ConsolidatedTaskItem] = Field(default_factory=list)
    notes: list[ConsolidatedNoteItem] = Field(default_factory=list)
    discardedCandidateIds: list[str] = Field(default_factory=list)


class OutlineSection(WireModel):
    title: str = ""
    noteIds: list[str] = Field(default_factory=list)


class OutlineTaskMerge(WireModel):
    title: str = ""
    taskIds: list[str] = Field(default_factory=list)


class OutlinePriority(WireModel):
    taskId: str = ""
    priority: str = ""


class MeetingLogisticsResponse(WireModel):
    offTopicIds: list[str] = Field(default_factory=list)


class MeetingOutlineResponse(WireModel):
    offTopicIds: list[str] = Field(default_factory=list)
    rankedTaskIds: list[str] = Field(default_factory=list)
    priorities: list[OutlinePriority] = Field(default_factory=list)
    taskMerges: list[OutlineTaskMerge] = Field(default_factory=list)
    sections: list[OutlineSection] = Field(default_factory=list)


class MeetingTaskEligibilityResponse(WireModel):
    tasks: list[ConsolidatedTaskItem] = Field(default_factory=list)


class ArtifactClaim(BaseModel):
    artifactKey: str
    kind: Literal["task", "note"]
    title: str
    body: str = ""
    owner: str | None = None
    dueDate: str | None = None
    priority: str | None = None
    acceptanceCriteria: str = ""
    topic: str | None = None
    sourceCandidateIds: list[str] = Field(default_factory=list)
    evidenceSequences: list[int] = Field(default_factory=list)


class FieldSupport(BaseModel):
    title: bool | None = None
    description: bool | None = None
    owner: bool | None = None
    dueDate: bool | None = None


class VerifierFieldSupport(WireModel):
    title: bool
    description: bool
    owner: bool
    dueDate: bool


class VerifierItem(WireModel):
    artifactKey: str
    verdict: VerifierVerdict
    unsupportedFields: list[str] = Field(default_factory=list)
    fieldSupport: VerifierFieldSupport
    reason: str = ""

    @model_validator(mode="after")
    def verdict_consistent_with_reason(self):
        reason = (self.reason or "").strip().casefold()
        if reason == "supported" and self.verdict == VerifierVerdict.UNSUPPORTED:
            raise ValueError("reason=supported requires verdict=SUPPORTED")
        return self


class MeetingVerifierResponse(WireModel):
    items: list[VerifierItem] = Field(default_factory=list)


class MeetingArtifactRepairResponse(WireModel):
    title: str
    body: str = ""
    acceptanceCriteria: str | None = None
    owner: str | None = None
    dueDate: str | None = None


class VerifiedArtifact(BaseModel):
    kind: Literal["task", "note"]
    title: str
    body: str = ""
    owner: str | None = None
    dueDate: str | None = None
    priority: str | None = None
    acceptanceCriteria: str = ""
    topic: str | None = None
    sourceCandidateIds: list[str] = Field(default_factory=list)
    evidenceSequences: list[int] = Field(default_factory=list)
    verdict: VerifierVerdict = VerifierVerdict.UNSUPPORTED
    unsupportedFields: list[str] = Field(default_factory=list)
    fieldSupport: FieldSupport = Field(default_factory=FieldSupport)
    reason: str = ""
    repaired: bool = False
    artifactKey: str = ""


class MeetingPipelineResult(BaseModel):
    tasks: list[ExtractedTask] = Field(default_factory=list)
    notes: list[ExtractedNote] = Field(default_factory=list)
    candidates: list[MeetingCandidate] = Field(default_factory=list)
    windows: list[ExtractionWindow] = Field(default_factory=list)
    claims: list[ArtifactClaim] = Field(default_factory=list)
    verified: list[VerifiedArtifact] = Field(default_factory=list)
    rejected: list[VerifiedArtifact] = Field(default_factory=list)
    summary: str = ""
    topics: list[str] = Field(default_factory=list)
    provider: str = "meeting-pipeline"
    model: str = "extract-consolidate-verify-v1"
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    observability: dict[str, Any] = Field(default_factory=dict)
    usage: list[dict[str, Any]] = Field(default_factory=list)
    ledgerPayload: list[dict[str, Any]] = Field(default_factory=list)
