"""Pydantic schemas for memory analysis jobs and structured LLM output."""

from datetime import date
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TaskPriority(StrEnum):
    """Allowed task priority values."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AnalysisJob(BaseModel):
    """Redis job used to trigger memory analysis for one user space."""

    model_config = ConfigDict(extra="ignore")

    user_id: str = Field(min_length=1)
    space_id: str = Field(min_length=1)
    request_id: str | None = None


class GeneratedTask(BaseModel):
    """A task generated from unpublished memory text."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    title: str = Field(min_length=1, max_length=160)
    description: str = Field(default="", max_length=1000)
    priority: TaskPriority = TaskPriority.MEDIUM
    due_date: date | None = None
    source_chunk_ids: list[str] = Field(default_factory=list, alias="sourceChunkIds")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("title", "description", mode="before")
    @classmethod
    def normalize_text(cls, value: object) -> str:
        return "" if value is None else str(value).strip()


class GeneratedNote(BaseModel):
    """A durable note generated from unpublished memory text."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    title: str = Field(min_length=1, max_length=160)
    content: str = Field(min_length=1, max_length=4000)
    tags: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list, alias="sourceChunkIds")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("title", "content", mode="before")
    @classmethod
    def normalize_text(cls, value: object) -> str:
        return "" if value is None else str(value).strip()

    @field_validator("tags", mode="before")
    @classmethod
    def normalize_tags(cls, value: object) -> list[str]:
        if not value:
            return []
        if not isinstance(value, list):
            return []
        return [str(tag).strip().lower() for tag in value if str(tag).strip()]


class MemoryAnalysisOutput(BaseModel):
    """Strict structured output expected from the LLM."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    tasks: list[GeneratedTask] = Field(default_factory=list)
    notes: list[GeneratedNote] = Field(default_factory=list)
    should_publish_chunks: bool = Field(default=True, alias="shouldPublishChunks")


class ChunkFilterDecision(BaseModel):
    """LLM decision for whether a chunk is useful enough to keep."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    chunk_id: str = Field(alias="chunkId", min_length=1)
    is_useful: bool = Field(alias="isUseful")
    reason: str = Field(default="", max_length=500)
    confidence: float = Field(ge=0.0, le=1.0)


class ChunkFilterOutput(BaseModel):
    """Structured output for transcript chunk noise filtering."""

    model_config = ConfigDict(extra="forbid")

    decisions: list[ChunkFilterDecision] = Field(default_factory=list)


class RerankedChunk(BaseModel):
    """LLM relevance score for one chunk."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    chunk_id: str = Field(alias="chunkId", min_length=1)
    relevance_score: float = Field(alias="relevanceScore", ge=0.0, le=1.0)
    reason: str = Field(default="", max_length=500)


class ContextRerankOutput(BaseModel):
    """Structured output for context reranking."""

    model_config = ConfigDict(extra="forbid")

    chunks: list[RerankedChunk] = Field(default_factory=list)


class ContextQualityOutput(BaseModel):
    """Structured output for deciding whether generation is justified."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    context_quality_score: float = Field(alias="contextQualityScore", ge=0.0, le=1.0)
    should_generate: bool = Field(alias="shouldGenerate")
    has_clear_user_intent: bool = Field(alias="hasClearUserIntent")
    is_actionable_or_memorable: bool = Field(alias="isActionableOrMemorable")
    is_contradictory: bool = Field(alias="isContradictory")
    reasons: list[str] = Field(default_factory=list)


class GeneratedItemValidationDecision(BaseModel):
    """LLM validation decision for one generated task or note."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    item_type: Literal["task", "note"] = Field(alias="itemType")
    item_index: int = Field(alias="itemIndex", ge=0)
    is_valid: bool = Field(alias="isValid")
    reason: str = Field(default="", max_length=500)
    confidence: float = Field(ge=0.0, le=1.0)


class TaskNoteValidationOutput(BaseModel):
    """Structured validation output for generated tasks and notes."""

    model_config = ConfigDict(extra="forbid")

    decisions: list[GeneratedItemValidationDecision] = Field(default_factory=list)
