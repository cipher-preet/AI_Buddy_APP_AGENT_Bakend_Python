from __future__ import annotations

from datetime import datetime, timedelta, timezone
from hashlib import sha1
from typing import Any

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ReturnDocument

from apps.api_gateway.config.setting import settings
from services.conversation.models import (
    AudioChunkMetadata,
    ConversationDocument,
    ConversationStatus,
    ConversationSummaryDocument,
    ExtractionRunStatus,
    ExtractionRunDocument,
    SpaceMemoryDocument,
    STTStatus,
    TranscriptChunkDocument,
    TranscriptProcessingStatus,
    assert_valid_transition,
    utc_now,
)


class ConversationRepository:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db

    async def create_conversation(self, user_id: str, space_id: str) -> ConversationDocument:
        conversation = ConversationDocument(userId=to_mongo_id(user_id), spaceId=to_mongo_id(space_id))
        await self.db.conversations.insert_one(conversation.model_dump(by_alias=True))
        return conversation

    async def get_conversation(self, conversation_id: str) -> ConversationDocument | None:
        data = await self.db.conversations.find_one(_id_query(conversation_id))
        return ConversationDocument.model_validate(data) if data else None

    async def transition(
        self,
        conversation_id: str,
        target: ConversationStatus,
        updates: dict[str, Any] | None = None,
    ) -> ConversationDocument:
        current = await self.get_conversation(conversation_id)
        if not current:
            raise ValueError(f"Conversation not found: {conversation_id}")
        assert_valid_transition(current.status, target)

        update_doc = {"status": target.value, "updatedAt": utc_now()}
        if updates:
            update_doc.update(updates)
        result = await self.db.conversations.find_one_and_update(
            {**_id_query(conversation_id), "status": current.status.value},
            {"$set": update_doc},
            return_document=ReturnDocument.AFTER,
        )
        if not result:
            raise ValueError(f"Conversation transition conflict: {conversation_id}")
        return ConversationDocument.model_validate(result)

    async def record_audio_chunk(self, metadata: AudioChunkMetadata) -> bool:
        doc = metadata.model_dump()
        doc["conversationId"] = to_mongo_id(metadata.conversationId)
        doc["userId"] = to_mongo_id(metadata.userId)
        doc["spaceId"] = to_mongo_id(metadata.spaceId)
        result = await self.db.audio_chunks.update_one(
            {
                "conversationId": doc["conversationId"],
                "sequenceNumber": metadata.sequenceNumber,
            },
            {"$setOnInsert": doc},
            upsert=True,
        )
        inserted = bool(result.upserted_id)
        update: dict[str, Any] = {"lastActivityAt": utc_now(), "updatedAt": utc_now()}
        inc = {"receivedAudioChunkCount": 1} if inserted else {}
        await self.db.conversations.update_one(
            {
                **_id_query(metadata.conversationId),
                "userId": doc["userId"],
                "spaceId": doc["spaceId"],
                "status": {"$in": [ConversationStatus.RECORDING.value, ConversationStatus.STOP_REQUESTED.value]},
            },
            {"$set": update, **({"$inc": inc} if inc else {})},
        )
        await self.db.transcript_chunks.update_one(
            {
                "conversationId": doc["conversationId"],
                "sequenceNumber": metadata.sequenceNumber,
            },
            {
                "$setOnInsert": TranscriptChunkDocument(
                    conversationId=doc["conversationId"],
                    userId=doc["userId"],
                    spaceId=doc["spaceId"],
                    chunkId=metadata.chunkId,
                    sequenceNumber=metadata.sequenceNumber,
                    audioFilePath=metadata.filePath,
                    startTimeMs=None,
                    endTimeMs=metadata.durationMs,
                ).model_dump(by_alias=True)
            },
            upsert=True,
        )
        return inserted

    async def complete_transcript_chunk(
        self,
        conversation_id: str,
        sequence_number: int,
        raw_text: str,
        language_code: str | None,
        request_id: str | None,
        provider: str,
    ) -> None:
        now = utc_now()
        result = await self.db.transcript_chunks.update_one(
            {
                "conversationId": to_mongo_id(conversation_id),
                "sequenceNumber": sequence_number,
                "sttStatus": {"$ne": STTStatus.COMPLETED.value},
            },
            {
                "$set": {
                    "rawText": raw_text,
                    "languageCode": language_code,
                    "sttRequestId": request_id,
                    "sttProvider": provider,
                    "sttStatus": STTStatus.COMPLETED.value,
                    "updatedAt": now,
                },
                "$inc": {"sttAttempts": 1},
            },
        )
        if result.modified_count:
            await self.db.conversations.update_one(
                _id_query(conversation_id),
                {
                    "$inc": {"completedTranscriptChunkCount": 1},
                    "$set": {"updatedAt": now, "lastActivityAt": now},
                },
            )

    async def fail_transcript_chunk(self, conversation_id: str, sequence_number: int, error: str) -> None:
        now = utc_now()
        result = await self.db.transcript_chunks.update_one(
            {"conversationId": to_mongo_id(conversation_id), "sequenceNumber": sequence_number},
            {
                "$set": {
                    "sttStatus": STTStatus.FAILED.value,
                    "lastError": error[:1000],
                    "updatedAt": now,
                },
                "$inc": {"sttAttempts": 1},
            },
        )
        if result.modified_count:
            await self.db.conversations.update_one(
                _id_query(conversation_id),
                {"$inc": {"failedTranscriptChunkCount": 1}, "$set": {"updatedAt": now}},
            )

    async def list_transcript_chunks(self, conversation_id: str) -> list[TranscriptChunkDocument]:
        cursor = self.db.transcript_chunks.find({"conversationId": to_mongo_id(conversation_id)}).sort("sequenceNumber", 1)
        return [TranscriptChunkDocument.model_validate(doc) async for doc in cursor]

    async def create_extraction_run(
        self,
        conversation: ConversationDocument,
        provider: str,
        model: str,
    ) -> ExtractionRunDocument:
        existing = await self.db.extraction_runs.find_one(
            {
                "conversationId": {"$in": mongo_id_candidates(conversation.id)},
                "processingVersion": conversation.processingVersion,
            }
        )
        if existing:
            return ExtractionRunDocument.model_validate(existing)

        run = ExtractionRunDocument(
            conversationId=conversation.id,
            userId=conversation.userId,
            spaceId=conversation.spaceId,
            processingVersion=conversation.processingVersion,
            provider=provider,
            model=model,
        )
        run_doc = extraction_run_doc(run)
        await self.db.extraction_runs.update_one(
            {
                "conversationId": run.conversationId,
                "processingVersion": run.processingVersion,
            },
            {"$setOnInsert": run_doc},
            upsert=True,
        )
        await self.db.conversations.update_one(
            {"_id": conversation.id},
            {"$set": {"activeExtractionRunId": run.id, "updatedAt": utc_now()}},
        )
        return run

    async def save_extraction_run(self, run: ExtractionRunDocument) -> None:
        run.updatedAt = utc_now()
        staged_tasks = [item.model_dump() for item in run.stagedTasks]
        staged_notes = [item.model_dump() for item in run.stagedNotes]
        staged_decisions = [item.model_dump() for item in run.stagedDecisions]
        staged_issues = [item.model_dump() for item in run.stagedIssues]
        doc = extraction_run_doc(run)
        await self.db.extraction_runs.update_one(
            {"_id": run.id},
            {
                "$set": doc,
                "$unset": {
                    "stagedTasks": "",
                    "stagedNotes": "",
                    "stagedDecisions": "",
                    "stagedIssues": "",
                },
            },
            upsert=True,
        )
        run_status = run.status.value if isinstance(run.status, ExtractionRunStatus) else str(run.status)
        should_replace_staged = bool(staged_tasks or staged_notes or staged_decisions or staged_issues) or run_status in {
            ExtractionRunStatus.FAILED.value,
            ExtractionRunStatus.PUBLISHED.value,
        }
        if should_replace_staged:
            await self._replace_staged_collection("stagedTasks", run, staged_tasks)
            await self._replace_staged_collection("stagedNotes", run, staged_notes)
            await self._replace_staged_collection("stagedDecisions", run, staged_decisions)
            await self._replace_staged_collection("stagedIssues", run, staged_issues)

    async def mark_extraction_run_failed(self, run_id: Any, error: Exception | str) -> None:
        await self.db.extraction_runs.update_one(
            {"_id": to_mongo_id(run_id)},
            {
                "$set": {
                    "status": ExtractionRunStatus.FAILED.value,
                    "updatedAt": utc_now(),
                },
                "$push": {
                    "validationErrors": {
                        "code": "WORKFLOW_EXCEPTION",
                        "message": str(error)[:1000],
                    }
                },
            },
        )

    async def mark_active_extraction_run_failed(self, conversation_id: str, error: Exception | str) -> None:
        conversation = await self.get_conversation(conversation_id)
        if conversation and conversation.activeExtractionRunId is not None:
            await self.mark_extraction_run_failed(conversation.activeExtractionRunId, error)

    async def get_extraction_run(self, run_id: Any) -> ExtractionRunDocument | None:
        data = await self.db.extraction_runs.find_one({"_id": to_mongo_id(run_id)})
        return ExtractionRunDocument.model_validate(data) if data else None

    async def mark_conversation_failed(self, conversation_id: str, error: Exception | str) -> None:
        await self.db.conversations.update_one(
            {
                **_id_query(conversation_id),
                "status": {"$ne": ConversationStatus.COMPLETED.value},
            },
            {
                "$set": {
                    "status": ConversationStatus.FAILED.value,
                    "lastError": str(error)[:1000],
                    "updatedAt": utc_now(),
                }
            },
        )

    async def _replace_staged_collection(
        self,
        collection_name: str,
        run: ExtractionRunDocument,
        items: list[dict[str, Any]],
    ) -> None:
        collection = self.db[collection_name]
        await collection.delete_many({"extractionRunId": run.id})
        if not items:
            return
        await collection.insert_many(
            [staged_collection_doc(collection_name, run, item, index) for index, item in enumerate(items)],
            ordered=False,
        )

    async def publish_outputs(
        self,
        run: ExtractionRunDocument,
        summary: ConversationSummaryDocument,
        memory: SpaceMemoryDocument,
    ) -> dict[str, list[Any]]:
        task_ids: list[Any] = []
        summary.taskIds = task_ids
        summary_doc = summary.model_dump(by_alias=True)
        summary_doc["conversationId"] = to_mongo_id(summary.conversationId)
        summary_doc["userId"] = to_mongo_id(summary.userId)
        summary_doc["spaceId"] = to_mongo_id(summary.spaceId)
        await self.db.conversation_summaries.update_one(
            {"conversationId": to_mongo_id(summary.conversationId)},
            {"$set": summary_doc},
            upsert=True,
        )

        memory_doc = memory.model_dump(by_alias=True)
        memory_doc["userId"] = to_mongo_id(memory.userId)
        memory_doc["spaceId"] = to_mongo_id(memory.spaceId)
        if memory.lastUpdatedConversationId is not None:
            memory_doc["lastUpdatedConversationId"] = to_mongo_id(memory.lastUpdatedConversationId)
        await self.db.space_memory.update_one(
            {"userId": to_mongo_id(memory.userId), "spaceId": to_mongo_id(memory.spaceId)},
            {"$set": memory_doc},
            upsert=True,
        )
        return {"taskIds": task_ids}

    async def schedule_transcript_expiry(self, conversation_id: str) -> None:
        expires_at = datetime.now(timezone.utc) + timedelta(days=settings.RAW_TRANSCRIPT_RETENTION_DAYS)
        await self.db.transcript_chunks.update_many(
            {
                "conversationId": to_mongo_id(conversation_id),
                "sttStatus": STTStatus.COMPLETED.value,
            },
            {
                "$set": {
                    "processingStatus": TranscriptProcessingStatus.ARCHIVED.value,
                    "expiresAt": expires_at,
                    "updatedAt": utc_now(),
                }
            },
        )

    async def get_space_memory(self, user_id: str, space_id: str) -> SpaceMemoryDocument:
        data = await self.db.space_memory.find_one({"userId": to_mongo_id(user_id), "spaceId": to_mongo_id(space_id)})
        if data:
            return SpaceMemoryDocument.model_validate(data)
        return SpaceMemoryDocument(userId=to_mongo_id(user_id), spaceId=to_mongo_id(space_id))

    async def list_active_tasks(self, user_id: str, space_id: str) -> list[dict[str, Any]]:
        cursor = self.db.tasks.find(
            {
                "userId": to_mongo_id(user_id),
                "spaceId": to_mongo_id(space_id),
                "status": {"$in": ["pending", "in_progress", "blocked"]},
            },
            {"audit": 0},
        ).limit(100)
        return [doc async for doc in cursor]

    async def list_recent_summaries(self, user_id: str, space_id: str, limit: int = 5) -> list[dict[str, Any]]:
        cursor = self.db.conversation_summaries.find(
            {"userId": to_mongo_id(user_id), "spaceId": to_mongo_id(space_id)},
            {"summary": 1, "topics": 1, "createdAt": 1},
        ).sort("createdAt", -1).limit(limit)
        return [doc async for doc in cursor]

    async def find_inactive_recording_conversations(self, before: datetime, limit: int = 100) -> list[ConversationDocument]:
        cursor = self.db.conversations.find(
            {
                "status": ConversationStatus.RECORDING.value,
                "lastActivityAt": {"$lte": before},
                "receivedAudioChunkCount": {"$gt": 0},
            }
        ).limit(limit)
        return [ConversationDocument.model_validate(doc) async for doc in cursor]

    async def infer_last_sequence(self, conversation_id: str) -> int | None:
        doc = await self.db.audio_chunks.find_one(
            {"conversationId": to_mongo_id(conversation_id)},
            sort=[("sequenceNumber", -1)],
            projection={"sequenceNumber": 1},
        )
        return int(doc["sequenceNumber"]) if doc else None


def to_mongo_id(value: Any) -> Any:
    if isinstance(value, ObjectId):
        return value
    if isinstance(value, str) and ObjectId.is_valid(value):
        return ObjectId(value)
    return value


def mongo_id_candidates(value: Any) -> list[Any]:
    mongo_id = to_mongo_id(value)
    candidates = [mongo_id]
    if isinstance(mongo_id, ObjectId):
        candidates.append(str(mongo_id))
    return candidates


def same_mongo_id(left: Any, right: Any) -> bool:
    return to_mongo_id(left) == to_mongo_id(right)


def extraction_run_doc(run: ExtractionRunDocument) -> dict[str, Any]:
    doc = run.model_dump(by_alias=True)
    doc.pop("stagedTasks", None)
    doc.pop("stagedNotes", None)
    doc.pop("stagedDecisions", None)
    doc.pop("stagedIssues", None)
    return doc


def staged_collection_doc(
    collection_name: str,
    run: ExtractionRunDocument,
    item: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    item = dict(item)
    item["_id"] = staged_object_id(collection_name, run.id, item, index)
    item["extractionRunId"] = run.id
    item["conversationId"] = to_mongo_id(run.conversationId)
    item["userId"] = to_mongo_id(run.userId)
    item["spaceId"] = to_mongo_id(run.spaceId)
    item["processingVersion"] = run.processingVersion
    item["updatedAt"] = run.updatedAt
    item.setdefault("createdAt", run.startedAt)
    if item.get("sourceConversationId") is not None:
        item["sourceConversationId"] = to_mongo_id(item["sourceConversationId"])
    for evidence in item.get("evidence", []):
        if isinstance(evidence, dict):
            evidence.setdefault("_id", embedded_object_id(evidence))
    return item


def staged_object_id(collection_name: str, run_id: Any, item: dict[str, Any], index: int) -> ObjectId:
    stable_source = "|".join(
        [
            str(run_id),
            collection_name,
            str(index),
            str(item.get("fingerprint") or ""),
            str(item.get("title") or ""),
            str(item.get("body") or ""),
        ]
    )
    return ObjectId(sha1(stable_source.encode("utf-8")).hexdigest()[:24])


def embedded_object_id(item: dict[str, Any]) -> ObjectId:
    stable_source = item.get("fingerprint") or "|".join(
        str(item.get(key, "")) for key in ("title", "sequenceStart", "sequenceEnd", "text")
    )
    if stable_source.strip("|"):
        return ObjectId(sha1(stable_source.encode("utf-8")).hexdigest()[:24])
    return ObjectId()


def _id_query(value: Any) -> dict[str, Any]:
    mongo_id = to_mongo_id(value)
    if mongo_id == value:
        return {"_id": value}
    return {"_id": mongo_id}
