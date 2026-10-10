from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


PIPELINE_VERSION = "document-v1"


class DocumentStatus(str, Enum):
    QUEUED = "QUEUED"
    GATHERING = "GATHERING"
    GENERATING = "GENERATING"
    RENDERING = "RENDERING"
    SAVING = "SAVING"
    READY = "READY"
    FAILED = "FAILED"


ACTIVE_STATUSES = {
    DocumentStatus.QUEUED.value,
    DocumentStatus.GATHERING.value,
    DocumentStatus.GENERATING.value,
    DocumentStatus.RENDERING.value,
    DocumentStatus.SAVING.value,
}

TERMINAL_STATUSES = {
    DocumentStatus.READY.value,
    DocumentStatus.FAILED.value,
}


class DocumentTable(BaseModel):
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)


class DocumentSection(BaseModel):
    heading: str
    body: str | None = None
    bullets: list[str] = Field(default_factory=list)
    table: DocumentTable | None = None


class GeneratedDocumentContent(BaseModel):
    title: str
    subtitle: str | None = None
    meta: dict[str, str] = Field(default_factory=dict)
    sections: list[DocumentSection] = Field(default_factory=list)


class DocumentSectionLLM(BaseModel):
    """Flat section shape — free OpenRouter models struggle with nested table/dict schemas."""

    heading: str = ""
    body: str = ""
    bullets: list[str] = Field(default_factory=list)
    tableHeaders: list[str] = Field(default_factory=list)
    # Each row is one pipe-separated line: "cell A | cell B | cell C"
    tableRows: list[str] = Field(default_factory=list)


class DocumentGenerateSchema(BaseModel):
    """LLM structured output optimized for openrouter free models."""

    title: str = "Document"
    subtitle: str = ""
    # Prefer "Key: Value" lines over object maps (avoids additionalProperties failures).
    metaLines: list[str] = Field(default_factory=list)
    sections: list[DocumentSectionLLM] = Field(default_factory=list)


class SourceStats(BaseModel):
    taskCount: int = 0
    noteCount: int = 0
    transcriptCount: int = 0
    meetingCount: int = 0
    truncated: bool = False


class TemplateSpec(BaseModel):
    code: str
    title: str
    tagline: str = ""
    description: str = ""
    requiredSections: list[str] = Field(default_factory=list)
    guidance: str = ""
    tableHints: dict[str, list[str]] = Field(default_factory=dict)


# Professional criteria researched from meeting-minutes / status-report best practices.
TEMPLATE_SPECS: dict[str, TemplateSpec] = {
    "meeting-recap": TemplateSpec(
        code="meeting-recap",
        title="Meeting Recap",
        requiredSections=["Summary", "Decisions", "Action Items", "Discussion Notes", "Next Steps"],
        guidance=(
            "Lead with a one-line summary and outcomes. Decisions are settled choices, not debate. "
            "Every action item needs one owner, a verb-led task, and a concrete due date. "
            "Keep discussion notes brief — never paste transcript."
        ),
        tableHints={
            "Action Items": ["Action", "Owner", "Due date", "Status"],
            "Decisions": ["Decision", "Owner", "Date"],
        },
    ),
    "meeting-prep": TemplateSpec(
        code="meeting-prep",
        title="Meeting Prep",
        requiredSections=["Agenda", "Open Questions", "Pending Tasks", "Risks", "Suggested Asks"],
        guidance="Build a focused prep pack from prior context. Surface unresolved decisions and risks early.",
        tableHints={"Pending Tasks": ["Task", "Owner", "Due date", "Status"]},
    ),
    "weekly-task-planner": TemplateSpec(
        code="weekly-task-planner",
        title="Weekly Task Planner",
        requiredSections=["This Week Overview", "High Priority", "Medium Priority", "Later / Backlog", "Blockers"],
        guidance="Group work by priority. Pair owner + deadline. Call out blocked/overdue items.",
        tableHints={
            "High Priority": ["Task", "Owner", "Due date", "Status"],
            "Medium Priority": ["Task", "Owner", "Due date", "Status"],
            "Later / Backlog": ["Task", "Owner", "Due date", "Status"],
        },
    ),
    "weekly-review": TemplateSpec(
        code="weekly-review",
        title="Weekly Review",
        requiredSections=["Completed", "Pending", "Blockers", "Key Decisions", "Next Week Priorities"],
        guidance="Reflect honestly on what shipped vs pending. Keep next-week priorities actionable and few.",
    ),
    "one-on-one-prep": TemplateSpec(
        code="one-on-one-prep",
        title="1:1 Meeting Prep",
        requiredSections=["Commitments", "Wins", "Blockers", "Discussion Questions", "Follow-ups"],
        guidance="Carry forward prior commitments. Celebrate wins. Surface blockers and useful questions.",
    ),
    "project-status": TemplateSpec(
        code="project-status",
        title="Project Status Report",
        requiredSections=["Progress Summary", "Milestones", "Completed", "Pending", "Risks & Blockers", "Next Steps"],
        guidance="Stakeholder-ready tone. Separate completed vs pending. Risks need impact + mitigation when known.",
        tableHints={"Risks & Blockers": ["Risk / Blocker", "Impact", "Owner", "Mitigation"]},
    ),
    "client-meeting-recap": TemplateSpec(
        code="client-meeting-recap",
        title="Client Meeting Recap",
        requiredSections=["Requirements", "Decisions", "Commitments", "Deliverables", "Follow-ups"],
        guidance="Client-ready wording. Requirements and commitments must be explicit. Follow-ups need owners and dates.",
        tableHints={"Follow-ups": ["Action", "Owner", "Due date", "Status"]},
    ),
    "daily-work-plan": TemplateSpec(
        code="daily-work-plan",
        title="Daily Work Plan",
        requiredSections=["Today's Priorities", "Schedule", "Overdue / Carryovers", "Focus Blocks"],
        guidance="Keep the day realistic. Rank priorities. Protect deep-focus blocks around meetings.",
        tableHints={"Today's Priorities": ["Priority", "Task", "Owner", "Status"]},
    ),
    "decision-log": TemplateSpec(
        code="decision-log",
        title="Decision Log",
        requiredSections=["Decisions"],
        guidance=(
            "Durable decision record: what was decided, why, alternatives considered, owner, date, and source. "
            "Do not invent decisions that are not evidenced in context."
        ),
        tableHints={
            "Decisions": ["Decision", "Reason", "Alternatives", "Owner", "Date", "Source"],
        },
    ),
    "action-item-tracker": TemplateSpec(
        code="action-item-tracker",
        title="Action Item Tracker",
        requiredSections=["Open Actions", "Due Soon", "Completed Recently"],
        guidance="Consolidate actions across meetings. Every row needs owner, deadline, status, and origin when known.",
        tableHints={
            "Open Actions": ["Action", "Owner", "Due date", "Status", "Origin"],
            "Due Soon": ["Action", "Owner", "Due date", "Status", "Origin"],
            "Completed Recently": ["Action", "Owner", "Completed", "Origin"],
        },
    ),
}


def get_template_spec(code: str, *, title: str | None = None) -> TemplateSpec:
    normalized = str(code or "").strip().lower()
    if normalized in TEMPLATE_SPECS:
        return TEMPLATE_SPECS[normalized]
    return TemplateSpec(
        code=normalized or "custom",
        title=title or "Document",
        requiredSections=["Summary", "Details", "Action Items", "Next Steps"],
        guidance="Produce a clear professional document grounded only in the provided context.",
    )


def public_source_stats(stats: SourceStats | dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(stats, SourceStats):
        return stats.model_dump()
    if hasattr(stats, "model_dump"):
        payload = stats.model_dump()  # type: ignore[union-attr]
        return SourceStats(
            **{key: payload.get(key, 0) for key in SourceStats.model_fields}
        ).model_dump()
    if isinstance(stats, dict):
        return SourceStats(
            **{key: stats.get(key, 0) for key in SourceStats.model_fields}
        ).model_dump()
    return SourceStats().model_dump()
