from __future__ import annotations

from datetime import datetime, timedelta, timezone
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
    ExtractionRunDocument,
    SpaceMemoryDocument,
    STTStatus,
    TranscriptChunkDocument,
    TranscriptProcessingStatus,
    assert_valid_transition,
    new_id,
    utc_now,
)


class ConversationRepository:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db

    async def create_conversation(self, user_id: str, space_id: str) -> ConversationDocument:
        conversation = ConversationDocument(userId=user_id, spaceId=space_id)
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
        result = await self.db.audio_chunks.update_one(
            {
                "conversationId": metadata.conversationId,
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
                "userId": metadata.userId,
                "spaceId": metadata.spaceId,
                "status": {"$in": [ConversationStatus.RECORDING.value, ConversationStatus.STOP_REQUESTED.value]},
            },
            {"$set": update, **({"$inc": inc} if inc else {})},
        )
        await self.db.transcript_chunks.update_one(
            {
                "conversationId": metadata.conversationId,
                "sequenceNumber": metadata.sequenceNumber,
            },
            {
                "$setOnInsert": TranscriptChunkDocument(
                    conversationId=metadata.conversationId,
                    userId=metadata.userId,
                    spaceId=metadata.spaceId,
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
                "conversationId": conversation_id,
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
            {"conversationId": conversation_id, "sequenceNumber": sequence_number},
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
        cursor = self.db.transcript_chunks.find({"conversationId": conversation_id}).sort("sequenceNumber", 1)
        return [TranscriptChunkDocument.model_validate(doc) async for doc in cursor]

    async def create_extraction_run(
        self,
        conversation: ConversationDocument,
        provider: str,
        model: str,
    ) -> ExtractionRunDocument:
        existing = await self.db.extraction_runs.find_one(
            {
                "conversationId": str(conversation.id),
                "processingVersion": conversation.processingVersion,
            }
        )
        if existing:
            return ExtractionRunDocument.model_validate(existing)

        run = ExtractionRunDocument(
            conversationId=str(conversation.id),
            userId=conversation.userId,
            spaceId=conversation.spaceId,
            processingVersion=conversation.processingVersion,
            provider=provider,
            model=model,
        )
        await self.db.extraction_runs.update_one(
            {
                "conversationId": run.conversationId,
                "processingVersion": run.processingVersion,
            },
            {"$setOnInsert": run.model_dump(by_alias=True)},
            upsert=True,
        )
        await self.db.conversations.update_one(
            {"_id": conversation.id},
            {"$set": {"activeExtractionRunId": run.id, "updatedAt": utc_now()}},
        )
        return run

    async def save_extraction_run(self, run: ExtractionRunDocument) -> None:
        run.updatedAt = utc_now()
        await self.db.extraction_runs.update_one(
            {"_id": run.id},
            {"$set": run.model_dump(by_alias=True)},
            upsert=True,
        )

    async def publish_outputs(
        self,
        run: ExtractionRunDocument,
        summary: ConversationSummaryDocument,
        memory: SpaceMemoryDocument,
    ) -> dict[str, list[Any]]:
        task_ids: list[Any] = []
        for task in run.stagedTasks:
            if task.operation == "NO_ACTION":
                continue
            task_id = to_mongo_id(task.existingTaskId) if task.existingTaskId else new_id()
            task_doc = {
                "_id": task_id,
                "userId": run.userId,
                "spaceId": run.spaceId,
                "title": task.title,
                "status": "completed" if task.operation == "COMPLETE" else "pending",
                "operation": task.operation,
                "ownerText": task.ownerText,
                "dueDateText": task.dueDateText,
                "dueDateResolved": task.dueDateResolved,
                "sourceConversationId": run.conversationId,
                "fingerprint": task.fingerprint,
                "evidence": [item.model_dump() for item in task.evidence],
                "updatedAt": utc_now(),
                "createdAt": utc_now(),
            }
            await self.db.tasks.update_one(
                {"_id": task_id},
                {"$set": task_doc, "$push": {"audit": task_doc}},
                upsert=True,
            )
            task_ids.append(task_id)

        summary.taskIds = task_ids
        await self.db.conversation_summaries.update_one(
            {"conversationId": summary.conversationId},
            {"$setOnInsert": summary.model_dump(by_alias=True)},
            upsert=True,
        )

        await self.db.space_memory.update_one(
            {"userId": memory.userId, "spaceId": memory.spaceId, "version": memory.version - 1},
            {"$set": memory.model_dump(by_alias=True)},
            upsert=True,
        )
        return {"taskIds": task_ids}

    async def schedule_transcript_expiry(self, conversation_id: str) -> None:
        expires_at = datetime.now(timezone.utc) + timedelta(days=settings.RAW_TRANSCRIPT_RETENTION_DAYS)
        await self.db.transcript_chunks.update_many(
            {
                "conversationId": conversation_id,
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
        data = await self.db.space_memory.find_one({"userId": user_id, "spaceId": space_id})
        if data:
            return SpaceMemoryDocument.model_validate(data)
        return SpaceMemoryDocument(userId=user_id, spaceId=space_id)

    async def list_active_tasks(self, user_id: str, space_id: str) -> list[dict[str, Any]]:
        cursor = self.db.tasks.find(
            {"userId": user_id, "spaceId": space_id, "status": {"$in": ["pending", "in_progress", "blocked"]}},
            {"audit": 0},
        ).limit(100)
        return [doc async for doc in cursor]

    async def list_recent_summaries(self, user_id: str, space_id: str, limit: int = 5) -> list[dict[str, Any]]:
        cursor = self.db.conversation_summaries.find(
            {"userId": user_id, "spaceId": space_id},
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
            {"conversationId": conversation_id},
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


def _id_query(value: Any) -> dict[str, Any]:
    mongo_id = to_mongo_id(value)
    if mongo_id == value:
        return {"_id": value}
    return {"_id": mongo_id}
