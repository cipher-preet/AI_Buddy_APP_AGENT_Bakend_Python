from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from services.conversation.models import (
    ConversationStatus,
    CoverageReport,
    ExtractedDecision,
    ExtractedIssue,
    ExtractedNote,
    ExtractedTask,
    SectionExtractionResult,
    Segment,
)


class ConversationGraphState(BaseModel):
    conversation_id: str
    user_id: str
    space_id: str
    processing_version: int
    extraction_run_id: str
    conversation_status: ConversationStatus
    raw_transcript: str = ""
    normalized_transcript: str = ""
    segments: list[Segment] = Field(default_factory=list)
    space_memory: dict[str, Any] = Field(default_factory=dict)
    relevant_previous_summaries: list[dict[str, Any]] = Field(default_factory=list)
    active_tasks: list[dict[str, Any]] = Field(default_factory=list)
    section_results: list[SectionExtractionResult] = Field(default_factory=list)
    merged_tasks: list[ExtractedTask] = Field(default_factory=list)
    merged_notes: list[ExtractedNote] = Field(default_factory=list)
    merged_decisions: list[ExtractedDecision] = Field(default_factory=list)
    merged_questions: list[ExtractedIssue] = Field(default_factory=list)
    merged_blockers: list[ExtractedIssue] = Field(default_factory=list)
    coverage_report: CoverageReport | None = None
    validation_errors: list[dict[str, Any]] = Field(default_factory=list)
    repair_round: int = 0
    warnings: list[str] = Field(default_factory=list)
