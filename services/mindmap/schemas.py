from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


PIPELINE_VERSION = "mindmap-v1"

MindmapTone = Literal[
    "hub",
    "cyan",
    "green",
    "pink",
    "purple",
    "yellow",
    "blue",
    "lavender",
    "magenta",
]
MindmapNodeKind = Literal["hub", "branch", "card", "junction"]
MindmapVariant = Literal["default", "alert", "compact", "priority"]
PriorityTone = Literal["high", "medium", "later"]


class MindmapStatus(str, Enum):
    QUEUED = "QUEUED"
    GATHERING = "GATHERING"
    GENERATING = "GENERATING"
    SAVING = "SAVING"
    READY = "READY"
    FAILED = "FAILED"


ACTIVE_STATUSES = {
    MindmapStatus.QUEUED.value,
    MindmapStatus.GATHERING.value,
    MindmapStatus.GENERATING.value,
    MindmapStatus.SAVING.value,
}

TERMINAL_STATUSES = {
    MindmapStatus.READY.value,
    MindmapStatus.FAILED.value,
}


class PrioritySection(BaseModel):
    label: str
    tone: PriorityTone = "medium"
    items: list[str] = Field(default_factory=list)


class MindmapNodeData(BaseModel):
    kind: MindmapNodeKind
    title: str
    subtitle: str | None = None
    items: list[str] = Field(default_factory=list)
    tone: MindmapTone | None = None
    tags: list[str] = Field(default_factory=list)
    noteCount: int | None = None
    variant: MindmapVariant | None = None
    prioritySections: list[PrioritySection] = Field(default_factory=list)
    edgeHint: str | None = None


class MindmapPosition(BaseModel):
    x: float = 0
    y: float = 0


class MindmapNode(BaseModel):
    id: str
    type: Literal["mindmapNode"] = "mindmapNode"
    position: MindmapPosition = Field(default_factory=MindmapPosition)
    data: MindmapNodeData


class MindmapMarkerEnd(BaseModel):
    type: Literal["arrowclosed"] = "arrowclosed"
    width: int = 16
    height: int = 16
    color: str = "#64748b"


class MindmapEdgeStyle(BaseModel):
    stroke: str = "#64748b"
    strokeWidth: float = 2
    strokeDasharray: str | None = None


class MindmapEdge(BaseModel):
    id: str
    source: str
    target: str
    type: Literal["smoothstep"] = "smoothstep"
    animated: bool = False
    label: str | None = None
    style: MindmapEdgeStyle | None = None
    markerEnd: MindmapMarkerEnd | None = None


class MindmapGraph(BaseModel):
    nodes: list[MindmapNode] = Field(default_factory=list)
    edges: list[MindmapEdge] = Field(default_factory=list)


class MindmapLLMResponse(BaseModel):
    """Structured OpenRouter output before layout normalization."""

    spaceTitle: str = "Space"
    hubSubtitle: str | None = "Key notes, tasks, and meeting themes"
    branches: list[dict[str, Any]] = Field(default_factory=list)


class SourceStats(BaseModel):
    taskCount: int = 0
    noteCount: int = 0
    transcriptCount: int = 0
    meetingCount: int = 0
    truncated: bool = False


class SpaceContextBundle(BaseModel):
    spaceId: str
    spaceName: str = "Space"
    tasks: list[dict[str, Any]] = Field(default_factory=list)
    notes: list[dict[str, Any]] = Field(default_factory=list)
    transcripts: list[dict[str, Any]] = Field(default_factory=list)
    sourceStats: SourceStats = Field(default_factory=SourceStats)
