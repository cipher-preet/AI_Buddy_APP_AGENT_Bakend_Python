from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

EventKind = Literal["meeting", "call", "appointment", "deadline", "event"]
RepeatKind = Literal["once", "daily", "weekly", "weekdays", "monthly"]


def _clean(value: object) -> str:
    return " ".join(str(value or "").split())


def _require_all(schema: dict, _model: type) -> None:
    # Schema-guided decoding skips optional properties; defaults stay for lenient parsing.
    schema["required"] = list(schema.get("properties", {}))


class _Candidate(BaseModel):
    model_config = ConfigDict(json_schema_extra=_require_all)

    title: str = ""
    description: str = ""
    dateKey: str = ""
    dateText: str = ""
    timeText: str = ""
    evidence: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("title", "description", "dateKey", "dateText", "timeText", "evidence", mode="before")
    @classmethod
    def _strip(cls, value: object) -> str:
        return _clean(value)

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp(cls, value: object) -> float:
        try:
            return min(1.0, max(0.0, float(value)))
        except (TypeError, ValueError):
            return 0.0


class EventCandidate(_Candidate):
    kind: EventKind = "meeting"
    startTime: str = ""
    endTime: str = ""
    location: str = ""

    @field_validator("kind", mode="before")
    @classmethod
    def _kind(cls, value: object) -> str:
        text = _clean(value).casefold()
        return text if text in {"meeting", "call", "appointment", "deadline", "event"} else "event"

    @field_validator("startTime", "endTime", "location", mode="before")
    @classmethod
    def _strip_extra(cls, value: object) -> str:
        return _clean(value)


class ReminderCandidate(_Candidate):
    time: str = ""
    repeat: RepeatKind = "once"

    @field_validator("time", mode="before")
    @classmethod
    def _strip_time(cls, value: object) -> str:
        return _clean(value)

    @field_validator("repeat", mode="before")
    @classmethod
    def _repeat(cls, value: object) -> str:
        text = _clean(value).casefold()
        return text if text in {"once", "daily", "weekly", "weekdays", "monthly"} else "once"


class EventExtraction(BaseModel):
    model_config = ConfigDict(json_schema_extra=_require_all)
    events: list[EventCandidate] = Field(default_factory=list)


class ReminderExtraction(BaseModel):
    model_config = ConfigDict(json_schema_extra=_require_all)
    reminders: list[ReminderCandidate] = Field(default_factory=list)


@dataclass
class ScheduledEvent:
    kind: str
    title: str
    description: str
    location: str
    dateKey: str
    startTimeLabel: str
    endTimeLabel: str
    timeInferred: bool
    evidence: str
    confidence: float
    fingerprint: str = ""


@dataclass
class ScheduledReminder:
    title: str
    description: str
    dateKey: str
    timeLabel: str
    repeat: str
    timeInferred: bool
    evidence: str
    confidence: float
    fingerprint: str = ""


@dataclass
class PipelineOutcome:
    name: str
    items: list = field(default_factory=list)
    windows: int = 0
    failedWindows: int = 0
    rawCandidates: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    error: str | None = None

    @property
    def failed(self) -> bool:
        return self.windows > 0 and self.failedWindows >= self.windows

    def drop(self, reason: str) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1
