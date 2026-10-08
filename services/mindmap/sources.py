from __future__ import annotations

import time
from typing import Any

from apps.api_gateway.config.setting import settings
from services.conversation.repository import mongo_id_candidates
from services.mindmap.log import mindmap_log
from services.mindmap.schemas import SourceStats, SpaceContextBundle
from services.observability.diagnostics import diag_log

TASK_PROJECTION = {
    "_id": 1,
    "title": 1,
    "body": 1,
    "description": 1,
    "status": 1,
    "operation": 1,
    "priority": 1,
    "dueDate": 1,
    "spaceId": 1,
    "createdAt": 1,
    "updatedAt": 1,
    "deletedAt": 1,
}
NOTE_PROJECTION = {
    "_id": 1,
    "title": 1,
    "body": 1,
    "spaceId": 1,
    "createdAt": 1,
    "updatedAt": 1,
    "deletedAt": 1,
}
TRANSCRIPT_PROJECTION = {
    "_id": 1,
    "conversationId": 1,
    "spaceId": 1,
    "rawText": 1,
    "normalizedText": 1,
    "sttStatus": 1,
    "createdAt": 1,
    "capturedAt": 1,
    "meetingSessionId": 1,
}
SPACE_PROJECTION = {"_id": 1, "spacename": 1}


def _id_filter(user_id: str) -> dict[str, Any]:
    return {"userId": {"$in": mongo_id_candidates(user_id)}}


def _space_filter(space_id: str) -> dict[str, Any]:
    return {"spaceId": {"$in": mongo_id_candidates(space_id)}}


def _not_deleted() -> dict[str, Any]:
    return {"$or": [{"deletedAt": None}, {"deletedAt": {"$exists": False}}]}


def _clip(text: str, limit: int) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)].rstrip() + "…"


class MindmapSource:
    def __init__(self, database):
        self.db = database
        self._collections: set[str] | None = None

    async def _collection_names(self) -> set[str]:
        if self._collections is None:
            self._collections = set(await self.db.list_collection_names())
        return self._collections

    async def load(self, user_id: str, space_id: str) -> SpaceContextBundle:
        names = await self._collection_names()
        user_filter = _id_filter(user_id)
        space_filter = _space_filter(space_id)
        soft = _not_deleted()
        task_limit = settings.MINDMAP_MAX_TASKS
        note_limit = settings.MINDMAP_MAX_NOTES
        transcript_limit = settings.MINDMAP_MAX_TRANSCRIPTS
        body_chars = settings.MINDMAP_ITEM_BODY_CHARS

        started = time.perf_counter()
        space_name = await self._space_name(space_id, names)
        tasks = await self._merge_collections(
            ["tasks", "stagedTasks"],
            {**user_filter, **space_filter, **soft},
            TASK_PROJECTION,
            limit=task_limit,
            sort=[("updatedAt", -1)],
        )
        notes = await self._merge_collections(
            ["notes", "stagedNotes"],
            {**user_filter, **space_filter, **soft},
            NOTE_PROJECTION,
            limit=note_limit,
            sort=[("updatedAt", -1)],
        )
        transcripts = await self._load_transcripts(
            user_filter,
            space_filter,
            limit=transcript_limit,
            names=names,
        )
        diag_log(
            "mongo_query_timing",
            operation="mindmap_context_fetch",
            duration_ms=int((time.perf_counter() - started) * 1000),
            result_count=len(tasks) + len(notes) + len(transcripts),
            user_id=user_id,
            space_id=space_id,
        )

        mapped_tasks = [
            {
                "id": str(item.get("_id")),
                "title": _clip(str(item.get("title") or "Untitled task"), 120),
                "body": _clip(str(item.get("body") or item.get("description") or ""), body_chars),
                "priority": str(item.get("priority") or "") or None,
                "status": str(item.get("status") or item.get("operation") or "") or None,
                "dueDate": str(item.get("dueDate") or "") or None,
            }
            for item in tasks
            if str(item.get("title") or "").strip()
        ]
        mapped_notes = [
            {
                "id": str(item.get("_id")),
                "title": _clip(str(item.get("title") or "Untitled note"), 120),
                "body": _clip(str(item.get("body") or ""), body_chars),
            }
            for item in notes
            if str(item.get("title") or item.get("body") or "").strip()
        ]
        mapped_transcripts = [
            {
                "id": str(item.get("_id")),
                "conversationId": str(item.get("conversationId") or item.get("meetingSessionId") or ""),
                "text": _clip(
                    str(item.get("normalizedText") or item.get("rawText") or ""),
                    settings.MINDMAP_TRANSCRIPT_CHARS,
                ),
            }
            for item in transcripts
            if str(item.get("normalizedText") or item.get("rawText") or "").strip()
        ]

        truncated = (
            len(tasks) >= task_limit
            or len(notes) >= note_limit
            or len(transcripts) >= transcript_limit
        )
        stats = SourceStats(
            taskCount=len(mapped_tasks),
            noteCount=len(mapped_notes),
            transcriptCount=len(mapped_transcripts),
            meetingCount=len({item["conversationId"] for item in mapped_transcripts if item["conversationId"]}),
            truncated=truncated,
        )
        mindmap_log(
            "context_loaded",
            userId=user_id,
            spaceId=space_id,
            taskCount=stats.taskCount,
            noteCount=stats.noteCount,
            transcriptCount=stats.transcriptCount,
            truncated=truncated,
        )
        return SpaceContextBundle(
            spaceId=space_id,
            spaceName=space_name or "Space",
            tasks=mapped_tasks,
            notes=mapped_notes,
            transcripts=mapped_transcripts,
            sourceStats=stats,
        )

    async def _space_name(self, space_id: str, names: set[str]) -> str:
        if "spaces" not in names:
            return "Space"
        doc = await self.db.spaces.find_one(
            {"_id": {"$in": mongo_id_candidates(space_id)}},
            SPACE_PROJECTION,
        )
        return str((doc or {}).get("spacename") or "Space").strip() or "Space"

    async def _merge_collections(
        self,
        collection_names: list[str],
        query: dict[str, Any],
        projection: dict[str, Any],
        *,
        limit: int,
        sort: list[tuple[str, int]],
    ) -> list[dict[str, Any]]:
        available = await self._collection_names()
        seen: set[str] = set()
        rows: list[dict[str, Any]] = []
        for name in collection_names:
            if name not in available or len(rows) >= limit:
                break
            remaining = limit - len(rows)
            docs = await (
                self.db[name]
                .find(query, projection)
                .sort(sort)
                .limit(remaining)
                .to_list(length=remaining)
            )
            for doc in docs:
                key = str(doc.get("_id"))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(doc)
                if len(rows) >= limit:
                    break
        return rows

    async def _load_transcripts(
        self,
        user_filter: dict[str, Any],
        space_filter: dict[str, Any],
        *,
        limit: int,
        names: set[str],
    ) -> list[dict[str, Any]]:
        if "transcript_chunks" not in names:
            return []
        query = {
            **user_filter,
            **space_filter,
            "sttStatus": "completed",
        }
        return await (
            self.db.transcript_chunks.find(query, TRANSCRIPT_PROJECTION)
            .sort([("createdAt", -1)])
            .limit(limit)
            .to_list(length=limit)
        )
