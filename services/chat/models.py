from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from pydantic import BaseModel, Field


MAX_CHAT_MESSAGES = 100


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ChatSessionDocument(BaseModel):
    id: Any = Field(default_factory=ObjectId, alias="_id")
    userId: Any
    spaceId: Any | None = None
    title: str | None = None
    status: str = "active"
    messageCount: int = 0
    createdAt: datetime = Field(default_factory=utc_now)
    updatedAt: datetime = Field(default_factory=utc_now)

    model_config = {"populate_by_name": True}


class RetrievedContext(BaseModel):
    text: str
    score: float | None = None
    sourceId: str | None = None
    jobId: str | None = None
    chunkIndex: int | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
