from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ReminderIntent = Literal["set_reminder", "out_of_context", "unclear"]
ReminderRepeat = Literal["once", "daily", "weekly", "weekdays", "monthly"]
TurnStatus = Literal["need_more", "ready", "out_of_context", "unclear"]


class ReminderCollected(BaseModel):
    title: str | None = None
    description: str | None = None
    dateKey: str | None = None
    dateLabel: str | None = None
    timeLabel: str | None = None
    repeat: ReminderRepeat | None = None

    language: Literal["en", "hi"] | None = None

    def missing_field(self) -> str | None:
        if not (self.title or "").strip():
            return "title"
        if not (self.dateKey or "").strip():
            return "date"
        if not (self.timeLabel or "").strip():
            return "time"
        return None

    def is_complete(self) -> bool:
        return self.missing_field() is None


class ReminderExtractorResponse(BaseModel):
    intent: ReminderIntent = "unclear"
    title: str | None = None
    description: str | None = None
    dateExpression: str | None = None
    timeExpression: str | None = None
    repeat: ReminderRepeat | None = None


class ReminderPayload(BaseModel):
    title: str
    description: str
    dateKey: str
    dateLabel: str
    timeLabel: str
    repeat: ReminderRepeat = "once"
    source: Literal["ai"] = "ai"
    aiCalling: bool = False
    notification: bool = True
    beeping: bool = False


class ReminderVoiceTurnResult(BaseModel):
    status: TurnStatus
    transcript: str = ""
    replyText: str
    replyKind: str
    collected: ReminderCollected = Field(default_factory=ReminderCollected)
    reminder: ReminderPayload | None = None
    sttProvider: str | None = None
    usedLlm: bool = False
    language: Literal["en", "hi"] = "en"
