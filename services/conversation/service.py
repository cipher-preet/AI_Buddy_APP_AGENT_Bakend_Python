from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import aiofiles
from fastapi import UploadFile

from apps.api_gateway.config.setting import settings
from services.conversation.models import AudioChunkMetadata, ConversationStatus, utc_now
from services.conversation.repository import ConversationRepository, same_mongo_id
from services.db.mongo import get_database
from services.queue.streams import EventEnvelope, RedisStreamProducer
from services.storage.s3_audio_storage import (
    build_conversation_audio_object_key,
    get_s3_audio_storage,
    validate_allowed_audio_upload,
    validate_conversation_audio_object_key,
)


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
            "userId": str(conversation.userId),
            "spaceId": str(conversation.spaceId),
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
        if not same_mongo_id(conversation.userId, user_id) or not same_mongo_id(conversation.spaceId, space_id):
            raise PermissionError("Conversation does not belong to this user and space")
        if conversation.status not in {ConversationStatus.RECORDING, ConversationStatus.STOP_REQUESTED}:
            raise ValueError(f"Conversation is not accepting audio: {conversation.status.value}")
        _validate_stt_chunk_duration(duration_ms)

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
                eventType="audio.ingested",
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

    async def create_audio_upload_url(
        self,
        conversation_id: str,
        user_id: str,
        space_id: str,
        sequence_number: int,
        content_type: str,
        extension: str,
        expected_size_bytes: int,
        chunk_id: str | None = None,
    ) -> dict:
        conversation = await self._get_owned_open_conversation(conversation_id, user_id, space_id)
        content_type, extension = validate_allowed_audio_upload(
            content_type=content_type,
            extension=extension,
            expected_size_bytes=expected_size_bytes,
        )
        chunk_id = (chunk_id or str(uuid4())).strip()
        if not chunk_id:
            raise ValueError("chunkId is required")
        object_key = build_conversation_audio_object_key(
            user_id=str(conversation.userId),
            space_id=str(conversation.spaceId),
            conversation_id=str(conversation.id),
            sequence_number=sequence_number,
            chunk_id=chunk_id,
            extension=extension,
        )
        upload_url = await get_s3_audio_storage().create_presigned_upload_url(
            object_key=object_key,
            content_type=content_type,
        )
        return {
            "conversationId": str(conversation.id),
            "chunkId": chunk_id,
            "sequenceNumber": sequence_number,
            "bucket": get_s3_audio_storage().bucket,
            "objectKey": object_key,
            "uploadUrl": upload_url,
            "method": "PUT",
            "headers": {"Content-Type": content_type},
            "expiresInSeconds": settings.S3_PRESIGNED_URL_TTL_SECONDS,
            "maxSizeBytes": settings.S3_MAX_AUDIO_SIZE_BYTES,
        }

    async def complete_audio_upload(
        self,
        conversation_id: str,
        user_id: str,
        space_id: str,
        chunk_id: str,
        sequence_number: int,
        object_key: str,
        content_type: str,
        size_bytes: int,
        captured_at: datetime | None = None,
        duration_ms: int | None = None,
    ) -> dict:
        conversation = await self._get_owned_open_conversation(conversation_id, user_id, space_id)
        _validate_stt_chunk_duration(duration_ms)
        content_type, extension = validate_allowed_audio_upload(
            content_type=content_type,
            extension=Path(object_key).suffix.lstrip("."),
            expected_size_bytes=size_bytes,
        )
        object_key = validate_conversation_audio_object_key(
            object_key=object_key,
            user_id=str(conversation.userId),
            space_id=str(conversation.spaceId),
            conversation_id=str(conversation.id),
        )
        expected_key = build_conversation_audio_object_key(
            user_id=str(conversation.userId),
            space_id=str(conversation.spaceId),
            conversation_id=str(conversation.id),
            sequence_number=sequence_number,
            chunk_id=chunk_id,
            extension=extension,
        )
        if object_key != expected_key:
            raise PermissionError("S3 object key does not match the server-generated chunk key")

        s3_ref = await get_s3_audio_storage().head_object(get_s3_audio_storage().bucket, object_key)
        if s3_ref.size_bytes != size_bytes:
            raise ValueError("Uploaded audio size does not match registration request")
        if s3_ref.size_bytes is None or s3_ref.size_bytes <= 0:
            raise ValueError("Uploaded audio object is empty")
        if s3_ref.size_bytes > settings.S3_MAX_AUDIO_SIZE_BYTES:
            raise ValueError("Uploaded audio object exceeds configured limit")
        if s3_ref.content_type and s3_ref.content_type.split(";", 1)[0].lower() != content_type:
            raise ValueError("Uploaded audio content type does not match registration request")

        job_id = _stable_job_id(conversation_id, sequence_number, chunk_id)
        metadata = AudioChunkMetadata(
            conversationId=conversation_id,
            userId=user_id,
            spaceId=space_id,
            chunkId=chunk_id,
            sequenceNumber=sequence_number,
            capturedAt=captured_at,
            durationMs=duration_ms,
            filePath=f"s3://{s3_ref.bucket}/{s3_ref.object_key}",
            filename=f"{chunk_id}.{extension}",
            contentType=content_type,
            storageProvider="s3",
            s3Bucket=s3_ref.bucket,
            s3ObjectKey=s3_ref.object_key,
            sizeBytes=s3_ref.size_bytes,
            jobId=job_id,
        )
        inserted = await self.repository.record_audio_chunk(metadata)
        if inserted:
            await self.producer.publish(
                settings.REDIS_STT_STREAM,
                EventEnvelope(
                    eventType="stt.requested",
                    correlationId=conversation_id,
                    userId=user_id,
                    spaceId=space_id,
                    conversationId=conversation_id,
                    payload={
                        "jobId": job_id,
                        "conversationId": conversation_id,
                        "userId": user_id,
                        "spaceId": space_id,
                        "chunkId": chunk_id,
                        "sequenceNumber": sequence_number,
                        "storageProvider": "s3",
                        "bucket": s3_ref.bucket,
                        "objectKey": s3_ref.object_key,
                        "contentType": content_type,
                        "sizeBytes": s3_ref.size_bytes,
                    },
                ),
            )

        return {
            "conversationId": conversation_id,
            "chunkId": chunk_id,
            "sequenceNumber": sequence_number,
            "jobId": job_id,
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
        if not same_mongo_id(conversation.userId, user_id) or not same_mongo_id(conversation.spaceId, space_id):
            raise PermissionError("Conversation does not belong to this user and space")
        if conversation.status not in {ConversationStatus.RECORDING, ConversationStatus.STOP_REQUESTED}:
            return {
                "conversationId": str(conversation.id),
                "status": conversation.status.value,
                "accepted": True,
                "duplicate": True,
            }

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
        if not same_mongo_id(conversation.userId, user_id) or not same_mongo_id(conversation.spaceId, space_id):
            raise PermissionError("Conversation does not belong to this user and space")
        if _is_stale_processing(conversation):
            await self.repository.mark_active_extraction_run_failed(
                conversation_id,
                f"Conversation processing timed out after {settings.CONVERSATION_PROCESSING_TIMEOUT_SECONDS} seconds.",
            )
            await self.repository.mark_conversation_failed(
                conversation_id,
                f"Conversation processing timed out after {settings.CONVERSATION_PROCESSING_TIMEOUT_SECONDS} seconds.",
            )
            conversation = await self.repository.get_conversation(conversation_id) or conversation
        data = _serialize_conversation(conversation.model_dump(by_alias=True))
        if conversation.activeExtractionRunId is not None:
            run = await self.repository.get_extraction_run(conversation.activeExtractionRunId)
            if run:
                data["activeExtractionRun"] = {
                    "id": str(run.id),
                    "status": run.status.value,
                    "coverageScore": run.coverageScore,
                    "validationErrors": run.validationErrors,
                    "checkpoints": run.checkpoints,
                    "updatedAt": run.updatedAt,
                }
        return data

    async def _get_owned_open_conversation(self, conversation_id: str, user_id: str, space_id: str):
        conversation = await self.repository.get_conversation(conversation_id)
        if not conversation:
            raise ValueError("Conversation not found")
        if not same_mongo_id(conversation.userId, user_id) or not same_mongo_id(conversation.spaceId, space_id):
            raise PermissionError("Conversation does not belong to this user and space")
        if conversation.status not in {ConversationStatus.RECORDING, ConversationStatus.STOP_REQUESTED}:
            raise ValueError(f"Conversation is not accepting audio: {conversation.status.value}")
        return conversation


def _serialize_conversation(data: dict) -> dict:
    for key in {"_id", "activeExtractionRunId", "userId", "spaceId"}:
        if data.get(key) is not None:
            data[key] = str(data[key])
    return data


def _is_stale_processing(conversation) -> bool:
    if conversation.status != ConversationStatus.PROCESSING:
        return False
    timeout = timedelta(seconds=settings.CONVERSATION_PROCESSING_TIMEOUT_SECONDS)
    return utc_now() - conversation.updatedAt > timeout


def _stable_job_id(conversation_id: str, sequence_number: int, chunk_id: str) -> str:
    return f"{conversation_id}:{sequence_number}:{chunk_id}"


def _validate_stt_chunk_duration(duration_ms: int | None) -> None:
    if duration_ms is None:
        return
    if duration_ms > settings.SARVAM_STT_MAX_DURATION_MS:
        raise ValueError(
            f"Audio chunk duration must be <= {settings.SARVAM_STT_MAX_DURATION_MS} ms for speech-to-text. "
            "Split recording into smaller chunks before upload."
        )
