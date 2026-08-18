from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from hashlib import sha1
from typing import Any

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ReturnDocument

from apps.api_gateway.config.setting import settings
from services.conversation.models import (
    AudioChunkMetadata,
    ConversationWindowDocument,
    ConversationDocument,
    ConversationStatus,
    ConversationSummaryDocument,
    ExtractionRunStatus,
    ExtractionRunDocument,
    MeetingArtifactDocument,
    MeetingMemoryDocument,
    SpaceMemoryDocument,
    STTStatus,
    TranscriptChunkDocument,
    TranscriptProcessingStatus,
    WindowExtractionResult,
    WindowProcessingStatus,
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

    async def get_transcript_chunk(self, conversation_id: str, sequence_number: int) -> TranscriptChunkDocument | None:
        data = await self.db.transcript_chunks.find_one(
            {
                "conversationId": to_mongo_id(conversation_id),
                "sequenceNumber": sequence_number,
            }
        )
        return TranscriptChunkDocument.model_validate(data) if data else None

    async def get_audio_chunk(self, conversation_id: str, sequence_number: int) -> dict[str, Any] | None:
        return await self.db.audio_chunks.find_one(
            {
                "conversationId": to_mongo_id(conversation_id),
                "sequenceNumber": sequence_number,
            }
        )

    async def mark_transcript_chunk_processing(self, conversation_id: str, sequence_number: int) -> bool:
        result = await self.db.transcript_chunks.update_one(
            {
                "conversationId": to_mongo_id(conversation_id),
                "sequenceNumber": sequence_number,
                "sttStatus": {"$ne": STTStatus.COMPLETED.value},
            },
            {
                "$set": {
                    "sttStatus": STTStatus.PROCESSING.value,
                    "updatedAt": utc_now(),
                }
            },
        )
        return bool(result.modified_count)

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

    async def list_transcript_chunks_in_range(
        self,
        conversation_id: str,
        sequence_start: int,
        sequence_end: int,
    ) -> list[TranscriptChunkDocument]:
        cursor = self.db.transcript_chunks.find(
            {
                "conversationId": to_mongo_id(conversation_id),
                "sequenceNumber": {"$gte": sequence_start, "$lte": sequence_end},
                "sttStatus": STTStatus.COMPLETED.value,
            }
        ).sort("sequenceNumber", 1)
        return [TranscriptChunkDocument.model_validate(doc) async for doc in cursor]

    async def list_completed_unwindowed_transcript_chunks(
        self,
        conversation_id: str,
        through_sequence: int | None = None,
    ) -> list[TranscriptChunkDocument]:
        query: dict[str, Any] = {
            "conversationId": to_mongo_id(conversation_id),
            "sttStatus": STTStatus.COMPLETED.value,
            "processingStatus": TranscriptProcessingStatus.UNPROCESSED.value,
        }
        if through_sequence is not None:
            query["sequenceNumber"] = {"$lte": through_sequence}
        cursor = self.db.transcript_chunks.find(query).sort("sequenceNumber", 1)
        return [TranscriptChunkDocument.model_validate(doc) async for doc in cursor]

    async def next_window_index(self, conversation_id: str) -> int:
        doc = await self.db.conversation_windows.find_one(
            {"conversationId": to_mongo_id(conversation_id)},
            sort=[("windowIndex", -1)],
            projection={"windowIndex": 1},
        )
        return int(doc["windowIndex"]) + 1 if doc else 0

    async def create_conversation_window(
        self,
        window: ConversationWindowDocument,
        sequence_numbers: list[int],
    ) -> ConversationWindowDocument:
        now = utc_now()
        doc = window.model_dump(by_alias=True)
        doc["conversationId"] = to_mongo_id(window.conversationId)
        doc["userId"] = to_mongo_id(window.userId)
        doc["spaceId"] = to_mongo_id(window.spaceId)
        doc["createdAt"] = window.createdAt
        doc["updatedAt"] = now
        result = await self.db.conversation_windows.find_one_and_update(
            {
                "conversationId": doc["conversationId"],
                "processingVersion": window.processingVersion,
                "sequenceStart": window.sequenceStart,
                "sequenceEnd": window.sequenceEnd,
            },
            {"$setOnInsert": doc},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        saved = ConversationWindowDocument.model_validate(result)
        if sequence_numbers:
            await self.db.transcript_chunks.update_many(
                {
                    "conversationId": doc["conversationId"],
                    "sequenceNumber": {"$in": sequence_numbers},
                    "processingStatus": TranscriptProcessingStatus.UNPROCESSED.value,
                },
                {
                    "$set": {
                        "processingStatus": TranscriptProcessingStatus.PROCESSED.value,
                        "processingWindowId": saved.id,
                        "processedAt": now,
                        "updatedAt": now,
                    }
                },
            )
        return saved

    async def get_conversation_window(self, window_id: Any) -> ConversationWindowDocument | None:
        data = await self.db.conversation_windows.find_one({"_id": to_mongo_id(window_id)})
        return ConversationWindowDocument.model_validate(data) if data else None

    async def list_conversation_windows(self, conversation_id: str) -> list[ConversationWindowDocument]:
        cursor = self.db.conversation_windows.find({"conversationId": to_mongo_id(conversation_id)}).sort("sequenceStart", 1)
        return [ConversationWindowDocument.model_validate(doc) async for doc in cursor]

    async def mark_window_processing(self, window_id: Any) -> ConversationWindowDocument | None:
        data = await self.db.conversation_windows.find_one_and_update(
            {
                "_id": to_mongo_id(window_id),
                "status": {"$in": [WindowProcessingStatus.PENDING.value, WindowProcessingStatus.FAILED.value]},
            },
            {
                "$set": {
                    "status": WindowProcessingStatus.PROCESSING.value,
                    "startedAt": utc_now(),
                    "updatedAt": utc_now(),
                    "lastError": None,
                },
                "$inc": {"attemptCount": 1},
            },
            return_document=ReturnDocument.AFTER,
        )
        return ConversationWindowDocument.model_validate(data) if data else None

    async def complete_window(
        self,
        window_id: Any,
        result: WindowExtractionResult,
        provider: str,
        model: str,
        token_usage: dict[str, int] | None = None,
    ) -> None:
        await self.db.conversation_windows.update_one(
            {"_id": to_mongo_id(window_id)},
            {
                "$set": {
                    "status": WindowProcessingStatus.COMPLETED.value,
                    "result": result.model_dump(),
                    "provider": provider,
                    "model": model,
                    "tokenUsage": token_usage or {},
                    "completedAt": utc_now(),
                    "updatedAt": utc_now(),
                }
            },
        )

    async def fail_window(self, window_id: Any, error: Exception | str) -> None:
        await self.db.conversation_windows.update_one(
            {"_id": to_mongo_id(window_id)},
            {
                "$set": {
                    "status": WindowProcessingStatus.FAILED.value,
                    "lastError": str(error)[:1000],
                    "updatedAt": utc_now(),
                }
            },
        )

    async def mark_window_queued(self, window_id: Any) -> None:
        await self.db.conversation_windows.update_one(
            {"_id": to_mongo_id(window_id)},
            {"$set": {"queuedAt": utc_now(), "updatedAt": utc_now()}},
        )

    async def list_meeting_artifacts(self, conversation_id: str) -> list[MeetingArtifactDocument]:
        cursor = self.db.meeting_artifacts.find({"conversationId": to_mongo_id(conversation_id)}).sort("createdAt", 1)
        return [MeetingArtifactDocument.model_validate(doc) async for doc in cursor]

    async def replace_meeting_artifacts(
        self,
        conversation_id: str,
        artifacts: list[MeetingArtifactDocument],
    ) -> None:
        collection = self.db.meeting_artifacts
        conversation_key = to_mongo_id(conversation_id)
        await collection.delete_many({"conversationId": conversation_key})
        if not artifacts:
            return
        docs = []
        for artifact in artifacts:
            artifact.updatedAt = utc_now()
            doc = artifact.model_dump(by_alias=True)
            doc["conversationId"] = conversation_key
            doc["userId"] = to_mongo_id(artifact.userId)
            doc["spaceId"] = to_mongo_id(artifact.spaceId)
            if artifact.sourceWindowId is not None:
                doc["sourceWindowId"] = to_mongo_id(artifact.sourceWindowId)
            docs.append(doc)
        await collection.insert_many(docs, ordered=False)

    async def upsert_meeting_artifacts(self, artifacts: list[MeetingArtifactDocument]) -> None:
        if not artifacts:
            return
        for artifact in artifacts:
            artifact.updatedAt = utc_now()
            doc = artifact.model_dump(by_alias=True)
            artifact_id = doc.pop("_id", artifact.id)
            doc["conversationId"] = to_mongo_id(artifact.conversationId)
            doc["userId"] = to_mongo_id(artifact.userId)
            doc["spaceId"] = to_mongo_id(artifact.spaceId)
            if artifact.sourceWindowId is not None:
                doc["sourceWindowId"] = to_mongo_id(artifact.sourceWindowId)
            await self.db.meeting_artifacts.find_one_and_update(
                {
                    "conversationId": doc["conversationId"],
                    "identityKey": artifact.identityKey,
                },
                {"$set": doc, "$setOnInsert": {"_id": to_mongo_id(artifact_id)}},
                upsert=True,
            )

    async def get_meeting_memory(self, conversation_id: str) -> MeetingMemoryDocument | None:
        data = await self.db.meeting_memory.find_one({"conversationId": to_mongo_id(conversation_id)})
        return MeetingMemoryDocument.model_validate(data) if data else None

    async def save_meeting_memory(self, memory: MeetingMemoryDocument) -> MeetingMemoryDocument:
        memory.updatedAt = utc_now()
        doc = memory.model_dump(by_alias=True)
        doc["conversationId"] = to_mongo_id(memory.conversationId)
        doc["userId"] = to_mongo_id(memory.userId)
        doc["spaceId"] = to_mongo_id(memory.spaceId)
        result = await self.db.meeting_memory.find_one_and_update(
            {"conversationId": doc["conversationId"]},
            {"$set": doc},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        return MeetingMemoryDocument.model_validate(result)

    async def append_meeting_debug_trace(
        self,
        conversation_id: str,
        user_id: Any,
        space_id: Any,
        stage: str,
        payload: dict[str, Any],
    ) -> None:
        if not debug_traces_enabled():
            return
        await self.db.meeting_debug_traces.insert_one(
            {
                "conversationId": to_mongo_id(conversation_id),
                "userId": to_mongo_id(user_id),
                "spaceId": to_mongo_id(space_id),
                "stage": stage,
                "payload": payload,
                "createdAt": utc_now(),
            }
        )

    async def mark_transcripts_published(self, conversation_id: str) -> None:
        await self.db.transcript_chunks.update_many(
            {
                "conversationId": to_mongo_id(conversation_id),
                "sttStatus": STTStatus.COMPLETED.value,
            },
            {"$set": {"publishedAt": utc_now(), "updatedAt": utc_now()}},
        )

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
        task_ids = await self._publish_tasks(run)
        note_ids = await self._publish_notes(run)
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
        return {"taskIds": task_ids, "noteIds": note_ids}

    async def _publish_tasks(self, run: ExtractionRunDocument) -> list[Any]:
        task_ids: list[Any] = []
        for task in run.stagedTasks:
            if task.operation == "NO_ACTION":
                continue
            doc = task.model_dump()
            doc["conversationId"] = to_mongo_id(run.conversationId)
            doc["sourceConversationId"] = to_mongo_id(task.sourceConversationId)
            doc["userId"] = to_mongo_id(run.userId)
            doc["spaceId"] = to_mongo_id(run.spaceId)
            doc["updatedAt"] = utc_now()
            doc.setdefault("createdAt", run.startedAt)
            doc["status"] = task_status_for_operation(task.operation, task.needsConfirmation)
            for evidence in doc.get("evidence", []):
                if isinstance(evidence, dict):
                    evidence.setdefault("_id", embedded_object_id(evidence))

            existing_task_id = task.existingTaskId.strip() if task.existingTaskId else None
            if existing_task_id:
                result = await self.db.tasks.find_one_and_update(
                    {"_id": to_mongo_id(existing_task_id), "userId": doc["userId"], "spaceId": doc["spaceId"]},
                    {"$set": {key: value for key, value in doc.items() if key != "createdAt"}},
                    return_document=ReturnDocument.AFTER,
                )
            else:
                task_id = task_object_id(run.id, doc, len(task_ids))
                doc["_id"] = task_id
                filter_doc = {"fingerprint": task.fingerprint} if task.fingerprint else {"_id": task_id}
                result = await self.db.tasks.find_one_and_update(
                    filter_doc,
                    {"$setOnInsert": doc},
                    upsert=True,
                    return_document=ReturnDocument.AFTER,
                )
            if result:
                task_ids.append(result["_id"])
        return task_ids

    async def _publish_notes(self, run: ExtractionRunDocument) -> list[Any]:
        note_ids: list[Any] = []
        for note in run.stagedNotes:
            doc = note.model_dump()
            doc["conversationId"] = to_mongo_id(run.conversationId)
            doc["sourceConversationId"] = to_mongo_id(note.sourceConversationId)
            doc["userId"] = to_mongo_id(run.userId)
            doc["spaceId"] = to_mongo_id(run.spaceId)
            doc["updatedAt"] = utc_now()
            doc.setdefault("createdAt", run.startedAt)
            for evidence in doc.get("evidence", []):
                if isinstance(evidence, dict):
                    evidence.setdefault("_id", embedded_object_id(evidence))

            note_id = note_object_id(run.id, doc, len(note_ids))
            doc["_id"] = note_id
            filter_doc = {"fingerprint": note.fingerprint} if note.fingerprint else {"_id": note_id}
            result = await self.db.notes.find_one_and_update(
                filter_doc,
                {"$setOnInsert": doc},
                upsert=True,
                return_document=ReturnDocument.AFTER,
            )
            if result:
                note_ids.append(result["_id"])
        return note_ids

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
        data = await self.db.space_memory.find_one(
            {"userId": {"$in": mongo_id_candidates(user_id)}, "spaceId": {"$in": mongo_id_candidates(space_id)}}
        )
        if data:
            return SpaceMemoryDocument.model_validate(data)
        return SpaceMemoryDocument(userId=to_mongo_id(user_id), spaceId=to_mongo_id(space_id))

    async def list_user_spaces(self, user_id: str, limit: int = 50) -> list[dict[str, Any]]:
        user_keys = mongo_id_candidates(user_id)
        spaces: dict[str, dict[str, Any]] = {}
        discovered_space_ids: list[Any] = []
        for collection_name in (
            "space_memory",
            "tasks",
            "notes",
            "conversation_summaries",
            "conversations",
            "chat_sessions",
            "stagedTasks",
            "stagedNotes",
            "stagedDecisions",
            "stagedIssues",
        ):
            try:
                values = await self.db[collection_name].distinct("spaceId", {"userId": {"$in": user_keys}})
            except Exception:
                values = []
            for value in values:
                if value is None:
                    continue
                discovered_space_ids.extend(mongo_id_candidates(value))
                key = str(value)
                spaces.setdefault(
                    key,
                    {
                        "spaceId": key,
                        "label": key,
                        "sources": [],
                    },
                )
                spaces[key]["sources"].append(collection_name)

        try:
            space_queries: list[dict[str, Any]] = [_space_owner_query(user_keys)]
            if discovered_space_ids:
                space_queries.append(_space_identity_query(discovered_space_ids))
            for collection_name in ("spaces", "space", "Spaces"):
                cursor = self.db[collection_name].find({"$or": space_queries}).limit(limit)
                async for item in cursor:
                    space_id = item.get("_id") or item.get("spaceId") or item.get("space_id") or item.get("id")
                    if space_id is None:
                        continue
                    key = str(space_id)
                    label = _space_document_label(item) or key
                    spaces.setdefault(key, {"spaceId": key, "label": str(label), "sources": []})
                    spaces[key]["label"] = str(label)
                    spaces[key]["sources"].append(collection_name)
        except Exception:
            pass

        ordered = sorted(spaces.values(), key=lambda item: (item["label"].lower(), item["spaceId"]))
        for item in ordered:
            if item["label"] == item["spaceId"]:
                item["label"] = _space_fallback_label(item["spaceId"])
        return ordered[:limit]

    async def list_active_tasks(self, user_id: str, space_id: str) -> list[dict[str, Any]]:
        cursor = self.db.tasks.find(
            {
                "userId": {"$in": mongo_id_candidates(user_id)},
                "spaceId": {"$in": mongo_id_candidates(space_id)},
                "status": {"$in": ["pending", "in_progress", "blocked", "needs_confirmation"]},
            },
            {"audit": 0},
        ).limit(100)
        return [doc async for doc in cursor]

    async def list_tasks(self, user_id: str, space_id: str, limit: int = 100) -> list[dict[str, Any]]:
        cursor = self.db.tasks.find(
            {"userId": {"$in": mongo_id_candidates(user_id)}, "spaceId": {"$in": mongo_id_candidates(space_id)}},
            {"audit": 0},
        ).sort("updatedAt", -1).limit(limit)
        return [doc async for doc in cursor]

    async def get_user_profile(self, user_id: str) -> dict[str, Any] | None:
        user_keys = mongo_id_candidates(user_id)
        data = await self.db.users.find_one(
            {"_id": {"$in": user_keys}},
            {"name": 1, "email": 1, "phone": 1, "provider": 1, "onboarding": 1},
        )
        return data if data else None

    async def get_space_stats(self, user_id: str, space_id: str) -> dict[str, int]:
        query = {"userId": {"$in": mongo_id_candidates(user_id)}, "spaceId": {"$in": mongo_id_candidates(space_id)}}
        (
            notes_count,
            tasks_count,
            done_tasks_count,
            staged_notes_count,
            staged_tasks_count,
            staged_done_tasks_count,
        ) = await _gather_counts(
            self.db.notes.count_documents(query),
            self.db.tasks.count_documents(query),
            self.db.tasks.count_documents({**query, "status": "completed"}),
            self.db["stagedNotes"].count_documents(query),
            self.db["stagedTasks"].count_documents(query),
            self.db["stagedTasks"].count_documents({**query, "operation": {"$in": ["DONE", "COMPLETE"]}}),
        )
        visible_task_count = tasks_count or staged_tasks_count
        visible_done_count = done_tasks_count if tasks_count else staged_done_tasks_count
        completion = 0 if visible_task_count == 0 else round((visible_done_count / visible_task_count) * 100)
        return {
            "notesCount": notes_count,
            "tasksCount": tasks_count,
            "doneTasksCount": done_tasks_count,
            "stagedNotesCount": staged_notes_count,
            "stagedTasksCount": staged_tasks_count,
            "stagedDoneTasksCount": staged_done_tasks_count,
            "completionPercentage": completion,
        }

    async def list_recent_notes(self, user_id: str, space_id: str, limit: int = 50) -> list[dict[str, Any]]:
        cursor = self.db.notes.find(
            {"userId": {"$in": mongo_id_candidates(user_id)}, "spaceId": {"$in": mongo_id_candidates(space_id)}},
            {"evidence": 0},
        ).sort("updatedAt", -1).limit(limit)
        return [doc async for doc in cursor]

    async def list_staged_tasks(self, user_id: str, space_id: str, limit: int = 50) -> list[dict[str, Any]]:
        cursor = self.db["stagedTasks"].find(
            {"userId": {"$in": mongo_id_candidates(user_id)}, "spaceId": {"$in": mongo_id_candidates(space_id)}},
            {"evidence": 0, "fingerprint": 0},
        ).sort("updatedAt", -1).limit(limit)
        return [doc async for doc in cursor]

    async def list_staged_notes(self, user_id: str, space_id: str, limit: int = 50) -> list[dict[str, Any]]:
        cursor = self.db["stagedNotes"].find(
            {"userId": {"$in": mongo_id_candidates(user_id)}, "spaceId": {"$in": mongo_id_candidates(space_id)}},
            {"evidence": 0, "fingerprint": 0},
        ).sort("updatedAt", -1).limit(limit)
        return [doc async for doc in cursor]

    async def list_staged_decisions(self, user_id: str, space_id: str, limit: int = 50) -> list[dict[str, Any]]:
        cursor = self.db["stagedDecisions"].find(
            {"userId": {"$in": mongo_id_candidates(user_id)}, "spaceId": {"$in": mongo_id_candidates(space_id)}},
            {"evidence": 0},
        ).sort("updatedAt", -1).limit(limit)
        return [doc async for doc in cursor]

    async def list_staged_issues(self, user_id: str, space_id: str, limit: int = 50) -> list[dict[str, Any]]:
        cursor = self.db["stagedIssues"].find(
            {"userId": {"$in": mongo_id_candidates(user_id)}, "spaceId": {"$in": mongo_id_candidates(space_id)}},
            {"evidence": 0},
        ).sort("updatedAt", -1).limit(limit)
        return [doc async for doc in cursor]

    async def list_recent_summaries(self, user_id: str, space_id: str, limit: int = 5) -> list[dict[str, Any]]:
        cursor = self.db.conversation_summaries.find(
            {"userId": {"$in": mongo_id_candidates(user_id)}, "spaceId": {"$in": mongo_id_candidates(space_id)}},
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

    async def find_stale_unfinalized_conversations(self, before: datetime, limit: int = 100) -> list[ConversationDocument]:
        cursor = self.db.conversations.find(
            {
                "status": {"$in": [ConversationStatus.STOP_REQUESTED.value, ConversationStatus.WAITING_FOR_TRANSCRIPTS.value]},
                "expectedLastSequence": {"$ne": None},
                "updatedAt": {"$lte": before},
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


def debug_traces_enabled() -> bool:
    return bool(settings.ENABLE_TRANSCRIPT_DEBUG_LOGS or settings.APP_ENV in {"local", "development"})


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


def task_object_id(run_id: Any, item: dict[str, Any], index: int) -> ObjectId:
    return _published_object_id("task", run_id, item, index)


def note_object_id(run_id: Any, item: dict[str, Any], index: int) -> ObjectId:
    return _published_object_id("note", run_id, item, index)


def _published_object_id(kind: str, run_id: Any, item: dict[str, Any], index: int) -> ObjectId:
    stable_source = "|".join(
        [
            str(run_id),
            kind,
            str(index),
            str(item.get("fingerprint") or ""),
            str(item.get("title") or ""),
            str(item.get("body") or ""),
        ]
    )
    return ObjectId(sha1(stable_source.encode("utf-8")).hexdigest()[:24])


def task_status_for_operation(operation: str, needs_confirmation: bool = False) -> str:
    if needs_confirmation or operation == "NEEDS_CONFIRMATION":
        return "needs_confirmation"
    if operation == "COMPLETE":
        return "completed"
    if operation == "CANCEL":
        return "cancelled"
    return "pending"


def _space_owner_query(user_keys: list[Any]) -> dict[str, Any]:
    return {
        "$or": [
            {"userId": {"$in": user_keys}},
            {"user_id": {"$in": user_keys}},
            {"ownerId": {"$in": user_keys}},
            {"owner_id": {"$in": user_keys}},
            {"createdBy": {"$in": user_keys}},
            {"createdById": {"$in": user_keys}},
            {"members.userId": {"$in": user_keys}},
            {"members.user_id": {"$in": user_keys}},
            {"users": {"$in": user_keys}},
            {"userIds": {"$in": user_keys}},
            {"user_ids": {"$in": user_keys}},
        ]
    }


def _space_identity_query(space_ids: list[Any]) -> dict[str, Any]:
    unique_ids = _unique_values(space_ids)
    return {
        "$or": [
            {"_id": {"$in": unique_ids}},
            {"spaceId": {"$in": unique_ids}},
            {"space_id": {"$in": unique_ids}},
            {"id": {"$in": unique_ids}},
        ]
    }


def _space_document_label(item: dict[str, Any]) -> str | None:
    for field in (
        "spaceName",
        "spacename",
        "space_name",
        "name",
        "title",
        "label",
        "workspaceName",
        "workspace_name",
    ):
        value = item.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _unique_values(values: list[Any]) -> list[Any]:
    seen: set[str] = set()
    unique: list[Any] = []
    for value in values:
        key = f"{type(value).__name__}:{value}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(value)
    return unique


def _space_fallback_label(space_id: Any) -> str:
    text = str(space_id or "").strip()
    if not text:
        return "Unnamed workspace"
    compact = "".join(char for char in text if char.isalnum())
    suffix = compact[-6:] if len(compact) > 6 else compact
    return f"Unnamed workspace ({suffix})" if suffix else "Unnamed workspace"


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


async def _gather_counts(*awaitables) -> list[int]:
    return [int(value) for value in await asyncio.gather(*awaitables)]
