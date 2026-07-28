from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import aiofiles
from fastapi import UploadFile

from apps.api_gateway.config.setting import settings
from services.conversation.models import AudioChunkMetadata, ConversationStatus, utc_now
from services.conversation.repository import ConversationRepository
from services.db.mongo import get_database
from services.queue.streams import EventEnvelope, RedisStreamProducer


UPLOAD_DIR = Path("resources/audio_jobs").resolve()


class ConversationService:
    def __init__(
        self,
        repository: ConversationRepository | None = None,
        producer: RedisStreamProducer | None = None,
    ):
        self.repository = repository or ConversationRepository(get_database())
        self.producer = producer or RedisStreamProducer()

    async def start(self, user_id: str, space_id: str) -> dict:
        conversation = await self.repository.create_conversation(user_id.strip(), space_id.strip())
        conversation_id = str(conversation.id)
        return {
            "conversationId": conversation_id,
            "userId": conversation.userId,
            "spaceId": conversation.spaceId,
            "status": conversation.status.value,
            "startedAt": conversation.startedAt,
        }

    async def ingest_audio(
        self,
        conversation_id: str,
        user_id: str,
        space_id: str,
        chunk_id: str,
        sequence_number: int,
        file: UploadFile,
        captured_at: datetime | None = None,
        duration_ms: int | None = None,
    ) -> dict:
        conversation = await self.repository.get_conversation(conversation_id)
        if not conversation:
            raise ValueError("Conversation not found")
        if conversation.userId != user_id or conversation.spaceId != space_id:
            raise PermissionError("Conversation does not belong to this user and space")
        if conversation.status not in {ConversationStatus.RECORDING, ConversationStatus.STOP_REQUESTED}:
            raise ValueError(f"Conversation is not accepting audio: {conversation.status.value}")

        conversation_dir = UPLOAD_DIR / conversation_id
        os.makedirs(conversation_dir, exist_ok=True)
        filename = file.filename or f"{chunk_id}.audio"
        extension = filename.rsplit(".", 1)[-1] if "." in filename else "audio"
        saved_filename = f"{sequence_number:08d}_{chunk_id}.{extension}"
        file_path = conversation_dir / saved_filename

        async with aiofiles.open(file_path, "wb") as out_file:
            await out_file.write(await file.read())

        metadata = AudioChunkMetadata(
            conversationId=conversation_id,
            userId=user_id,
            spaceId=space_id,
            chunkId=chunk_id,
            sequenceNumber=sequence_number,
            capturedAt=captured_at,
            durationMs=duration_ms,
            filePath=str(file_path),
            filename=filename,
            contentType=file.content_type,
        )
        inserted = await self.repository.record_audio_chunk(metadata)
        if inserted:
            event = EventEnvelope(
                eventType="stt.requested",
                correlationId=conversation_id,
                userId=user_id,
                spaceId=space_id,
                conversationId=conversation_id,
                payload=metadata.model_dump(mode="json"),
            )
            await self.producer.publish(settings.REDIS_AUDIO_STREAM, event)

        return {
            "conversationId": conversation_id,
            "chunkId": chunk_id,
            "sequenceNumber": sequence_number,
            "status": "queued" if inserted else "duplicate_ignored",
        }

    async def stop(
        self,
        conversation_id: str,
        user_id: str,
        space_id: str,
        last_sequence_number: int,
        stopped_at_client: datetime | None = None,
    ) -> dict:
        conversation = await self.repository.get_conversation(conversation_id)
        if not conversation:
            raise ValueError("Conversation not found")
        if conversation.userId != user_id or conversation.spaceId != space_id:
            raise PermissionError("Conversation does not belong to this user and space")

        stopped = await self.repository.transition(
            conversation_id,
            ConversationStatus.STOP_REQUESTED,
            {
                "expectedLastSequence": last_sequence_number,
                "stoppedAt": utc_now(),
                "stoppedAtClient": stopped_at_client,
            },
        )
        event = EventEnvelope(
            eventType="conversation.finalization.requested",
            correlationId=conversation_id,
            userId=user_id,
            spaceId=space_id,
            conversationId=conversation_id,
            payload={"expectedLastSequence": last_sequence_number},
        )
        await self.producer.publish(settings.REDIS_FINALIZATION_STREAM, event)
        return {
            "conversationId": str(stopped.id),
            "status": stopped.status.value,
            "accepted": True,
        }

    async def status(self, conversation_id: str, user_id: str, space_id: str) -> dict:
        conversation = await self.repository.get_conversation(conversation_id)
        if not conversation:
            raise ValueError("Conversation not found")
        if conversation.userId != user_id or conversation.spaceId != space_id:
            raise PermissionError("Conversation does not belong to this user and space")
        return _serialize_conversation(conversation.model_dump(by_alias=True))


def _serialize_conversation(data: dict) -> dict:
    for key in {"_id", "activeExtractionRunId"}:
        if data.get(key) is not None:
            data[key] = str(data[key])
    return data
