"""Transcript + date anchor shared by the calendar and reminder pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from services.conversation.models import TranscriptChunkDocument
from services.conversation.repository import to_mongo_id
from services.conversation.transcript import normalize_chunk_text, segment_transcript
from services.daily_briefing.scheduler import resolve_user_timezone
from services.daily_briefing.timezones import ensure_utc, to_local

LOOKUP_DAYS = 14
_RELATIVE = {0: "today", 1: "tomorrow", 2: "day after tomorrow"}


@dataclass
class ExtractionContext:
    conversation_id: str
    user_id: str
    timezone_name: str
    recorded_at_local: datetime
    transcript: str
    windows: list[str] = field(default_factory=list)
    truncated: bool = False

    @property
    def recording_date_key(self) -> str:
        return self.recorded_at_local.date().isoformat()

    def anchor(self) -> dict[str, Any]:
        base = self.recorded_at_local.date()
        return {
            "timezone": self.timezone_name,
            "recordingDate": base.isoformat(),
            "recordingWeekday": base.strftime("%A"),
            "recordingTime": self.recorded_at_local.strftime("%I:%M %p").lstrip("0"),
            "calendar": [
                {
                    "date": (base + timedelta(days=offset)).isoformat(),
                    "weekday": (base + timedelta(days=offset)).strftime("%A"),
                    **({"relative": _RELATIVE[offset]} if offset in _RELATIVE else {}),
                }
                for offset in range(LOOKUP_DAYS)
            ],
        }


def _useful(chunks: list[TranscriptChunkDocument]) -> list[TranscriptChunkDocument]:
    return [chunk for chunk in chunks if (chunk.normalizedText or chunk.rawText or "").strip()]


def transcript_text(chunks: list[TranscriptChunkDocument]) -> str:
    lines: list[str] = []
    previous = ""
    for chunk in sorted(chunks, key=lambda item: item.sequenceNumber):
        text = normalize_chunk_text((chunk.normalizedText or chunk.rawText or "").strip(), previous)
        if text:
            lines.append(text)
            previous = text
    return "\n".join(lines)


def build_windows(
    conversation_id: str,
    chunks: list[TranscriptChunkDocument],
    window_tokens: int,
    max_windows: int,
) -> tuple[list[str], bool]:
    segments = segment_transcript(conversation_id, chunks, window_tokens, 0.1, max_windows + 1)
    return [segment.text for segment in segments[:max_windows]], len(segments) > max_windows


async def load_context(
    repository,
    database,
    conversation_id: str,
    window_tokens: int,
    max_windows: int,
) -> ExtractionContext | None:
    conversation = await repository.get_conversation(conversation_id)
    if conversation is None:
        return None
    chunks = _useful(await repository.list_transcript_chunks(conversation_id))
    user_id = str(conversation.userId)
    user = await database.users.find_one({"_id": to_mongo_id(user_id)}, {"_id": 1, "timezone": 1}) or {"_id": user_id}
    timezone_name = await resolve_user_timezone(database, user)
    recorded = conversation.startedAt or conversation.createdAt
    recorded_local = to_local(ensure_utc(recorded), timezone_name)
    windows, truncated = build_windows(conversation_id, chunks, window_tokens, max_windows) if chunks else ([], False)
    return ExtractionContext(
        conversation_id=conversation_id,
        user_id=user_id,
        timezone_name=timezone_name,
        recorded_at_local=recorded_local,
        transcript=transcript_text(chunks),
        windows=windows,
        truncated=truncated,
    )
