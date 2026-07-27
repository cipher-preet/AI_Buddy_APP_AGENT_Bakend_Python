"""Schemas for transcript-window memory analysis."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


Operation = Literal["create", "update", "complete", "cancel", "no_change"]


class ContextResolution(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    resolved_entities: dict[str, str] = Field(default_factory=dict)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class TaskOperation(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    operation: Operation
    existing_task_id: str | None = None
    title: str = Field(default="", max_length=200)
    description: str = Field(default="", max_length=1200)
    due_at: str | None = None
    status: str = "open"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source_chunk_ids: list[str] = Field(default_factory=list)

    @field_validator("title", "description", "status", mode="before")
    @classmethod
    def normalize_text(cls, value: object) -> str:
        return "" if value is None else str(value).strip()


class NoteOperation(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    operation: Operation
    existing_note_id: str | None = None
    title: str = Field(default="", max_length=200)
    content: str = Field(default="", max_length=4000)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source_chunk_ids: list[str] = Field(default_factory=list)

    @field_validator("title", "content", mode="before")
    @classmethod
    def normalize_text(cls, value: object) -> str:
        return "" if value is None else str(value).strip()


class SummaryUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    should_update: bool = False
    updated_summary: str = Field(default="", max_length=6000)


class TranscriptAnalysisOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    is_complete_enough: bool
    requires_more_context: bool
    context_resolution: ContextResolution
    task_operations: list[TaskOperation] = Field(default_factory=list)
    note_operations: list[NoteOperation] = Field(default_factory=list)
    summary_update: SummaryUpdate
