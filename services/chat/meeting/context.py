from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from motor.motor_asyncio import AsyncIOMotorDatabase

from services.conversation.repository import mongo_id_candidates
from services.db.mongo import get_database


@dataclass
class TranscriptSegment:
    id: str
    text: str
    start_ms: int | None
    end_ms: int | None
    speaker: str | None
    sequence: int


@dataclass
class MeetingItem:
    id: str
    title: str
    body: str
    status: str | None = None
    due: str | None = None
    owner: str | None = None


@dataclass
class MeetingContext:
    meeting_id: str
    user_id: str
    space_id: str | None
    title: str
    started_at: datetime | None
    duration_ms: int | None
    status: str | None
    segments: list[TranscriptSegment] = field(default_factory=list)
    summary: dict[str, Any] | None = None
    memory: dict[str, Any] | None = None
    tasks: list[MeetingItem] = field(default_factory=list)
    notes: list[MeetingItem] = field(default_factory=list)

    @property
    def speakers(self) -> list[str]:
        seen: list[str] = []
        for segment in self.segments:
            if segment.speaker and segment.speaker not in seen:
                seen.append(segment.speaker)
        return seen


class MeetingContextLoader:
    """Read-only loader for everything Buddy knows about one meeting."""

    def __init__(self, db: AsyncIOMotorDatabase | None = None):
        self.db = db if db is not None else get_database()

    async def load(self, user_id: str, meeting_id: str) -> MeetingContext:
        ids = mongo_id_candidates(meeting_id)
        session, conversation = await asyncio.gather(
            self.db.meeting_sessions.find_one({"_id": {"$in": ids}}),
            self.db.conversations.find_one({"_id": {"$in": ids}}),
        )
        owner_doc = session or conversation
        if not owner_doc:
            raise ValueError("Meeting not found")
        if str(owner_doc.get("userId")) != str(user_id):
            raise PermissionError("Meeting does not belong to this user")

        space_id = (session or {}).get("spaceId") or (conversation or {}).get("spaceId")
        started_at = (session or {}).get("startedAt") or (conversation or {}).get("createdAt")
        context = MeetingContext(
            meeting_id=str(meeting_id),
            user_id=str(user_id),
            space_id=str(space_id) if space_id else None,
            title=str((session or {}).get("meetingTitle") or (conversation or {}).get("title") or "Untitled meeting"),
            started_at=started_at if isinstance(started_at, datetime) else None,
            duration_ms=_as_int((session or {}).get("durationMs")),
            status=(session or {}).get("status") or (conversation or {}).get("status"),
        )

        segments, summary, memory, tasks, notes = await asyncio.gather(
            self._load_segments(ids),
            self.db.conversation_summaries.find_one({"conversationId": {"$in": ids}}),
            self.db.meeting_memory.find_one({"conversationId": {"$in": ids}}),
            self._load_items(ids, "tasks", "stagedTasks"),
            self._load_items(ids, "notes", "stagedNotes"),
        )
        context.segments = segments
        context.summary = summary
        context.memory = memory
        context.tasks = tasks
        context.notes = notes
        return context

    async def _load_segments(self, ids: list[Any]) -> list[TranscriptSegment]:
        cursor = self.db.transcript_chunks.find(
            {"conversationId": {"$in": ids}},
            {"segments": 1, "rawText": 1, "normalizedText": 1, "startTimeMs": 1, "endTimeMs": 1, "sequenceNumber": 1, "chunkId": 1},
        ).sort([("startTimeMs", 1), ("sequenceNumber", 1)])
        segments: list[TranscriptSegment] = []
        async for chunk in cursor:
            sequence = _as_int(chunk.get("sequenceNumber")) or 0
            nested = chunk.get("segments") if isinstance(chunk.get("segments"), list) else []
            if nested:
                for index, segment in enumerate(nested):
                    text = str((segment or {}).get("text") or "").strip()
                    if not text:
                        continue
                    segments.append(
                        TranscriptSegment(
                            id=str(segment.get("id") or f"{chunk.get('chunkId') or sequence}:{index}"),
                            text=text,
                            start_ms=_as_int(segment.get("startOffsetMs")),
                            end_ms=_as_int(segment.get("endOffsetMs")),
                            speaker=_speaker_label(segment),
                            sequence=sequence,
                        )
                    )
                continue
            text = str(chunk.get("rawText") or chunk.get("normalizedText") or "").strip()
            if text:
                segments.append(
                    TranscriptSegment(
                        id=str(chunk.get("chunkId") or chunk.get("_id")),
                        text=text,
                        start_ms=_as_int(chunk.get("startTimeMs")),
                        end_ms=_as_int(chunk.get("endTimeMs")),
                        speaker=None,
                        sequence=sequence,
                    )
                )
        segments.sort(key=lambda item: (item.start_ms if item.start_ms is not None else 0, item.sequence))
        return segments

    async def _load_items(self, ids: list[Any], published: str, staged: str) -> list[MeetingItem]:
        not_deleted = {"$or": [{"deletedAt": None}, {"deletedAt": {"$exists": False}}]}
        published_docs, staged_docs = await asyncio.gather(
            self.db[published].find({"sourceConversationId": {"$in": ids}, **not_deleted}).sort("createdAt", -1).to_list(200),
            self.db[staged]
            .find(
                {
                    "$and": [
                        {"$or": [{"sourceConversationId": {"$in": ids}}, {"conversationId": {"$in": ids}}]},
                        not_deleted,
                    ]
                }
            )
            .sort("createdAt", -1)
            .to_list(200),
        )
        merged: dict[str, MeetingItem] = {}
        for doc in [*staged_docs, *published_docs]:
            merged[str(doc.get("_id"))] = _item_from_doc(doc)
        return list(merged.values())


def _item_from_doc(doc: dict[str, Any]) -> MeetingItem:
    operation = str(doc.get("operation") or "").upper()
    status = str(doc.get("status") or "open").lower()
    if operation == "COMPLETE" or status in {"done", "completed"}:
        status = "done"
    elif operation == "CANCEL" or status in {"blocked", "cancelled"}:
        status = "blocked"
    else:
        status = "open"
    due = doc.get("dueDateResolved") or doc.get("dueDateText") or doc.get("dueDate") or doc.get("date")
    return MeetingItem(
        id=str(doc.get("_id")),
        title=str(doc.get("title") or "Untitled"),
        body=str(doc.get("body") or doc.get("description") or ""),
        status=status,
        due=str(due) if due else None,
        owner=str(doc.get("ownerText")) if doc.get("ownerText") else None,
    )


def _speaker_label(segment: dict[str, Any]) -> str | None:
    label = segment.get("speakerLabel")
    if label:
        return str(label)
    speaker_id = segment.get("speakerId")
    if speaker_id is None or speaker_id == "":
        return None
    return f"Speaker {speaker_id}"


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def format_timestamp(ms: int | None) -> str:
    if ms is None or ms < 0:
        return "--:--"
    total_seconds = ms // 1000
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"
