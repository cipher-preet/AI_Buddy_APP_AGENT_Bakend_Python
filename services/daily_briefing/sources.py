from __future__ import annotations

from datetime import datetime
from typing import Any
import time

from services.conversation.repository import mongo_id_candidates
from services.observability.diagnostics import diag_log
from services.daily_briefing.schemas import ActivityBundle, SourceStats

TRANSCRIPT_PROJECTION = {
    "_id": 1,
    "conversationId": 1,
    "userId": 1,
    "spaceId": 1,
    "chunkId": 1,
    "rawText": 1,
    "normalizedText": 1,
    "sttStatus": 1,
    "terminal": 1,
    "createdAt": 1,
    "capturedAt": 1,
}
TASK_PROJECTION = {
    "_id": 1,
    "title": 1,
    "body": 1,
    "description": 1,
    "status": 1,
    "dueDate": 1,
    "spaceId": 1,
    "createdAt": 1,
    "updatedAt": 1,
}
NOTE_PROJECTION = {
    "_id": 1,
    "title": 1,
    "body": 1,
    "spaceId": 1,
    "createdAt": 1,
    "updatedAt": 1,
}
REMINDER_PROJECTION = {
    "_id": 1,
    "title": 1,
    "description": 1,
    "dateKey": 1,
    "timeLabel": 1,
    "timezone": 1,
    "createdAt": 1,
}
EVENT_PROJECTION = {
    "_id": 1,
    "title": 1,
    "description": 1,
    "dateKey": 1,
    "startTimeLabel": 1,
    "endTimeLabel": 1,
    "location": 1,
    "createdAt": 1,
}


def _id_filter(user_id: str) -> dict[str, Any]:
    return {"userId": {"$in": mongo_id_candidates(user_id)}}


def _created_range(start: datetime, end: datetime) -> dict[str, Any]:
    return {"createdAt": {"$gte": start, "$lt": end}}


class ActivitySource:
    def __init__(self, database):
        self.db = database
        self._collections: set[str] | None = None

    async def _collection_names(self) -> set[str]:
        if self._collections is None:
            self._collections = set(await self.db.list_collection_names())
        return self._collections

    async def load(
        self,
        user_id: str,
        date_key: str,
        period_start: datetime,
        period_end: datetime,
    ) -> ActivityBundle:
        user_filter = _id_filter(user_id)
        created = _created_range(period_start, period_end)
        transcript_query = {**user_filter, **created}
        pending_query = {
            **transcript_query,
            "sttStatus": {"$in": ["pending", "processing"]},
            "$or": [{"terminal": {"$ne": True}}, {"terminal": {"$exists": False}}],
        }
        completed_query = {**transcript_query, "sttStatus": "completed"}
        date_key_query = {**user_filter, "dateKey": date_key}

        transcript_started = time.perf_counter()
        transcripts = await self.db.transcript_chunks.find(
            completed_query, TRANSCRIPT_PROJECTION
        ).sort("createdAt", 1).to_list(length=5000)
        diag_log(
            "mongo_query_timing",
            operation="daily_briefing_transcript_fetch",
            duration_ms=int((time.perf_counter() - transcript_started) * 1000),
            result_count=len(transcripts),
            user_id=user_id,
            date_key=date_key,
        )
        pending = await self.db.transcript_chunks.count_documents(pending_query)
        tasks = await self._merge_collections(
            ["tasks", "stagedTasks"],
            {**user_filter, "$or": [created, {"dueDate": date_key}]},
            TASK_PROJECTION,
        )
        notes = await self._merge_collections(
            ["notes", "stagedNotes"],
            {**user_filter, **created},
            NOTE_PROJECTION,
        )
        reminders = await self.db.reminders.find(date_key_query, REMINDER_PROJECTION).to_list(length=500)
        names = await self._collection_names()
        events = []
        if "calendar_events" in names:
            events = await self.db.calendar_events.find(date_key_query, EVENT_PROJECTION).to_list(length=500)

        mapped_transcripts = [
            {
                "id": str(item.get("_id") or item.get("chunkId")),
                "text": (item.get("normalizedText") or item.get("rawText") or "").strip(),
                "createdAt": item.get("createdAt") or item.get("capturedAt"),
                "spaceId": str(item.get("spaceId") or ""),
                "conversationId": str(item.get("conversationId") or ""),
            }
            for item in transcripts
        ]
        return ActivityBundle(
            transcripts=mapped_transcripts,
            tasks=[self._map_task(item) for item in tasks],
            notes=[self._map_note(item) for item in notes],
            reminders=[self._map_reminder(item) for item in reminders],
            events=[self._map_event(item) for item in events],
            pendingTranscriptCount=pending,
        )

    async def _merge_collections(
        self,
        names: list[str],
        query: dict[str, Any],
        projection: dict[str, Any],
    ) -> list[dict[str, Any]]:
        available = await self._collection_names()
        seen: set[str] = set()
        rows: list[dict[str, Any]] = []
        for name in names:
            if name not in available:
                continue
            docs = await self.db[name].find(query, projection).to_list(length=1000)
            for doc in docs:
                key = str(doc.get("_id"))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(doc)
        return rows

    @staticmethod
    def _map_task(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": str(item.get("_id")),
            "title": str(item.get("title") or "").strip(),
            "detail": str(item.get("body") or item.get("description") or "").strip(),
            "status": str(item.get("status") or ""),
            "dueDate": item.get("dueDate"),
            "spaceId": str(item.get("spaceId") or ""),
            "createdAt": item.get("createdAt"),
        }

    @staticmethod
    def _map_note(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": str(item.get("_id")),
            "title": str(item.get("title") or "").strip(),
            "detail": str(item.get("body") or "").strip(),
            "spaceId": str(item.get("spaceId") or ""),
            "createdAt": item.get("createdAt"),
        }

    @staticmethod
    def _map_reminder(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": str(item.get("_id")),
            "title": str(item.get("title") or "").strip(),
            "detail": str(item.get("description") or "").strip(),
            "timeLabel": str(item.get("timeLabel") or ""),
            "createdAt": item.get("createdAt"),
        }

    @staticmethod
    def _map_event(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": str(item.get("_id")),
            "title": str(item.get("title") or "").strip(),
            "detail": str(item.get("description") or "").strip(),
            "timeLabel": str(item.get("startTimeLabel") or ""),
            "meta": " · ".join(
                part for part in [item.get("endTimeLabel"), item.get("location")] if part
            ),
            "createdAt": item.get("createdAt"),
        }


def source_stats(bundle: ActivityBundle, transcript_count: int) -> SourceStats:
    return SourceStats(
        transcriptCount=transcript_count,
        taskCount=len(bundle.tasks),
        noteCount=len(bundle.notes),
        eventCount=len(bundle.events),
        reminderCount=len(bundle.reminders),
        pendingTranscriptCount=bundle.pendingTranscriptCount,
    )
