from __future__ import annotations

import asyncio
import json
import time
from datetime import date, datetime, timedelta
from typing import Any

from bson import ObjectId

from apps.api_gateway.config.setting import settings
from services.conversation.repository import mongo_id_candidates
from services.observability.diagnostics import diag_log
from services.daily_briefing.schemas import ActivityBundle, SourceStats
from services.daily_briefing.timezones import ensure_utc

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
    "operation": 1,
    "priority": 1,
    "dueDate": 1,
    "dueDateResolved": 1,
    "dueDateText": 1,
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
    "repeat": 1,
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
SPACE_PROJECTION = {"_id": 1, "spacename": 1}
CHAT_SESSION_PROJECTION = {"_id": 1, "title": 1, "spaceId": 1}

OPEN_TASK_SCAN_LIMIT = 300
REPEATING = ("daily", "weekdays", "weekly", "monthly")
# Assistant replies that only report infrastructure errors carry no user signal.
CHAT_NOISE_PREFIXES = ("could not reach buddy api", "something went wrong", "error:")


def _id_filter(user_id: str) -> dict[str, Any]:
    return {"userId": {"$in": mongo_id_candidates(user_id)}}


def _created_range(start: datetime, end: datetime) -> dict[str, Any]:
    return {"createdAt": {"$gte": start, "$lt": end}}


def _as_aware(value: Any, default: datetime) -> datetime:
    return ensure_utc(value) if isinstance(value, datetime) else default


def next_date_key(date_key: str) -> str:
    return (date.fromisoformat(date_key) + timedelta(days=1)).isoformat()


def reminder_occurs_on(reminder: dict[str, Any], target_key: str) -> bool:
    start_key = str(reminder.get("dateKey") or "")
    repeat = str(reminder.get("repeat") or "once").lower()
    if not start_key:
        return False
    if repeat not in REPEATING:
        return start_key == target_key
    try:
        start, target = date.fromisoformat(start_key), date.fromisoformat(target_key)
    except ValueError:
        return False
    if target < start:
        return False
    if repeat == "daily":
        return True
    if repeat == "weekdays":
        return target.weekday() < 5
    if repeat == "weekly":
        return target.weekday() == start.weekday()
    return target.day == start.day


def _is_open_task(item: dict[str, Any]) -> bool:
    status = str(item.get("status") or "").lower()
    operation = str(item.get("operation") or "").upper()
    return status not in {"completed", "done"} and operation != "DONE"


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
        include_plan_context: bool = False,
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
        names = await self._collection_names()

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
        task_scope = [created, {"dueDate": date_key}]
        if include_plan_context:
            task_scope.append({"updatedAt": {"$gte": period_start, "$lt": period_end}})
        tasks = await self._merge_collections(
            ["tasks", "stagedTasks"],
            {**user_filter, "$or": task_scope},
            TASK_PROJECTION,
        )
        notes = await self._merge_collections(
            ["notes", "stagedNotes"],
            {**user_filter, **created},
            NOTE_PROJECTION,
        )
        reminders = await self.db.reminders.find(date_key_query, REMINDER_PROJECTION).to_list(length=500)
        events = []
        if "calendar_events" in names:
            events = await self.db.calendar_events.find(date_key_query, EVENT_PROJECTION).to_list(length=500)

        plan_key = next_date_key(date_key)
        open_tasks: list[dict[str, Any]] = []
        plan_events: list[dict[str, Any]] = []
        plan_reminders: list[dict[str, Any]] = []
        chats: list[dict[str, Any]] = []
        space_names: dict[str, str] = {}
        if include_plan_context:
            open_tasks, (plan_events, plan_reminders), chats = await asyncio.gather(
                self._load_open_tasks(user_filter, names),
                self._load_plan_agenda(user_filter, plan_key, names),
                self._load_chats(user_id, period_start, period_end, names),
            )
            space_names = await self._space_names(
                {str(item.get("spaceId") or "") for item in [*tasks, *open_tasks]} - {""},
                names,
            )

        mapped_transcripts = [
            {
                "id": str(item.get("_id") or item.get("chunkId")),
                "text": (item.get("normalizedText") or item.get("rawText") or "").strip(),
                "createdAt": _as_aware(item.get("createdAt") or item.get("capturedAt"), period_start),
                "spaceId": str(item.get("spaceId") or ""),
                "conversationId": str(item.get("conversationId") or ""),
            }
            for item in transcripts
        ]
        return ActivityBundle(
            transcripts=mapped_transcripts,
            tasks=[self._map_task(item, space_names) for item in tasks],
            notes=[self._map_note(item) for item in notes],
            reminders=[self._map_reminder(item) for item in reminders],
            events=[self._map_event(item) for item in events],
            pendingTranscriptCount=pending,
            chats=chats,
            openTasks=[self._map_task(item, space_names) for item in open_tasks],
            planDateKey=plan_key,
            planEvents=[self._map_event(item) for item in plan_events],
            planReminders=[self._map_reminder(item) for item in plan_reminders],
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

    async def _load_open_tasks(self, user_filter: dict[str, Any], available: set[str]) -> list[dict[str, Any]]:
        seen_ids: set[str] = set()
        seen_titles: set[tuple[str, str]] = set()
        rows: list[dict[str, Any]] = []
        query = {**user_filter, "deletedAt": None}
        for name in ("tasks", "stagedTasks"):
            if name not in available:
                continue
            docs = await (
                self.db[name]
                .find(query, TASK_PROJECTION)
                .sort("updatedAt", -1)
                .limit(OPEN_TASK_SCAN_LIMIT)
                .to_list(length=OPEN_TASK_SCAN_LIMIT)
            )
            for doc in docs:
                title = str(doc.get("title") or "").strip()
                if not title or not _is_open_task(doc):
                    continue
                doc_id = str(doc.get("_id"))
                title_key = (title.casefold(), str(doc.get("spaceId") or ""))
                if doc_id in seen_ids or title_key in seen_titles:
                    continue
                seen_ids.add(doc_id)
                seen_titles.add(title_key)
                rows.append(doc)
        return rows

    async def _load_plan_agenda(
        self,
        user_filter: dict[str, Any],
        plan_key: str,
        names: set[str],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        events: list[dict[str, Any]] = []
        if "calendar_events" in names:
            events = await self.db.calendar_events.find(
                {**user_filter, "dateKey": plan_key}, EVENT_PROJECTION
            ).to_list(length=200)
        candidates = await self.db.reminders.find(
            {**user_filter, "$or": [{"dateKey": plan_key}, {"repeat": {"$in": list(REPEATING)}}]},
            REMINDER_PROJECTION,
        ).to_list(length=500)
        reminders = [item for item in candidates if reminder_occurs_on(item, plan_key)]
        return events, reminders

    async def _load_chats(
        self,
        user_id: str,
        period_start: datetime,
        period_end: datetime,
        names: set[str],
    ) -> list[dict[str, Any]]:
        limit = settings.DAILY_BRIEFING_CHAT_MESSAGE_LIMIT
        if limit <= 0 or not {"chat_sessions", "chat_message_store"} <= names:
            return []
        sessions = await self.db.chat_sessions.find(
            {**_id_filter(user_id), "updatedAt": {"$gte": period_start}},
            CHAT_SESSION_PROJECTION,
        ).to_list(length=200)
        if not sessions:
            return []
        titles = {str(item["_id"]): str(item.get("title") or "") for item in sessions}
        session_ids = [candidate for key in titles for candidate in mongo_id_candidates(key)]
        started = time.perf_counter()
        docs = await (
            self.db.chat_message_store.find(
                {
                    "SessionId": {"$in": session_ids},
                    "_id": {
                        "$gte": ObjectId.from_datetime(period_start),
                        "$lt": ObjectId.from_datetime(period_end),
                    },
                }
            )
            .sort("_id", 1)
            .limit(limit)
            .to_list(length=limit)
        )
        diag_log(
            "mongo_query_timing",
            operation="daily_briefing_chat_fetch",
            duration_ms=int((time.perf_counter() - started) * 1000),
            result_count=len(docs),
            user_id=user_id,
        )
        messages: list[dict[str, Any]] = []
        for doc in docs:
            parsed = self._parse_chat_message(doc)
            if parsed is None:
                continue
            parsed["sessionTitle"] = titles.get(str(doc.get("SessionId")), "")
            messages.append(parsed)
        return messages

    async def _space_names(self, space_ids: set[str], names: set[str]) -> dict[str, str]:
        if not space_ids or "spaces" not in names:
            return {}
        candidates = [candidate for key in space_ids for candidate in mongo_id_candidates(key)]
        docs = await self.db.spaces.find({"_id": {"$in": candidates}}, SPACE_PROJECTION).to_list(length=len(space_ids))
        return {str(doc["_id"]): str(doc.get("spacename") or "") for doc in docs}

    @staticmethod
    def _parse_chat_message(doc: dict[str, Any]) -> dict[str, Any] | None:
        raw = doc.get("History")
        try:
            payload = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except (TypeError, ValueError):
            return None
        kind = str(payload.get("type") or "").lower()
        content = str((payload.get("data") or {}).get("content") or "").strip()
        if kind not in {"human", "ai"} or not content:
            return None
        if kind == "ai" and content.casefold().startswith(CHAT_NOISE_PREFIXES):
            return None
        doc_id = doc.get("_id")
        created = doc_id.generation_time if isinstance(doc_id, ObjectId) else None
        return {
            "id": str(doc_id),
            "role": "user" if kind == "human" else "assistant",
            "text": content,
            "createdAt": created,
        }

    @staticmethod
    def _map_task(item: dict[str, Any], space_names: dict[str, str] | None = None) -> dict[str, Any]:
        space_id = str(item.get("spaceId") or "")
        return {
            "id": str(item.get("_id")),
            "title": str(item.get("title") or "").strip(),
            "detail": str(item.get("body") or item.get("description") or "").strip(),
            "status": str(item.get("status") or ""),
            "operation": str(item.get("operation") or ""),
            "priority": str(item.get("priority") or ""),
            "dueDate": item.get("dueDate") or item.get("dueDateResolved"),
            "dueText": str(item.get("dueDateText") or ""),
            "spaceId": space_id,
            "space": (space_names or {}).get(space_id, ""),
            "createdAt": item.get("createdAt"),
            "updatedAt": item.get("updatedAt"),
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
            "repeat": str(item.get("repeat") or "once"),
            "createdAt": item.get("createdAt"),
        }

    @staticmethod
    def _map_event(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": str(item.get("_id")),
            "title": str(item.get("title") or "").strip(),
            "detail": str(item.get("description") or "").strip(),
            "timeLabel": str(item.get("startTimeLabel") or ""),
            "endTimeLabel": str(item.get("endTimeLabel") or ""),
            "location": str(item.get("location") or ""),
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
        chatMessageCount=len(bundle.chats),
        openTaskCount=len(bundle.openTasks),
    )
