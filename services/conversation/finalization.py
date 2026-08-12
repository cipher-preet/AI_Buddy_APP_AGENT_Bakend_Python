from __future__ import annotations

from urllib.parse import urlparse

from apps.api_gateway.config.setting import settings
from services.conversation.models import ConversationStatus, STTStatus
from services.conversation.repository import ConversationRepository
from services.conversation.transcript import detect_missing_sequences
from services.queue.streams import EventEnvelope, RedisStreamProducer
from services.storage.s3_audio_storage import build_audio_object_key, use_s3_storage


class ConversationFinalizationCoordinator:
    def __init__(
        self,
        repository: ConversationRepository,
        producer: RedisStreamProducer | None = None,
    ):
        self.repository = repository
        self.producer = producer or RedisStreamProducer()

    async def finalize(self, conversation_id: str) -> None:
        conversation = await self.repository.get_conversation(conversation_id)
        if not conversation or conversation.expectedLastSequence is None:
            raise ValueError("Conversation is not ready for finalization")
        if conversation.status in {
            ConversationStatus.READY_FOR_PROCESSING,
            ConversationStatus.PROCESSING,
            ConversationStatus.VALIDATING,
            ConversationStatus.COMPLETED,
            ConversationStatus.PARTIAL,
        }:
            return

        chunks = await self.repository.list_transcript_chunks(conversation_id)
        sequence_numbers = [chunk.sequenceNumber for chunk in chunks]
        missing = detect_missing_sequences(sequence_numbers, conversation.expectedLastSequence)
        failed = [chunk for chunk in chunks if chunk.sttStatus == STTStatus.FAILED]
        pending = [chunk for chunk in chunks if chunk.sttStatus in {STTStatus.PENDING, STTStatus.PROCESSING}]

        if missing or pending or failed:
            retryable_count = 0
            for chunk in [*pending, *failed]:
                if not chunk.audioFilePath or chunk.sttStatus == STTStatus.PROCESSING:
                    continue
                audio_chunk = await self.repository.get_audio_chunk(
                    str(chunk.conversationId),
                    chunk.sequenceNumber,
                )
                audio_fields = _stt_payload_audio_fields(chunk, audio_chunk)
                if _is_permanent_audio_failure(chunk):
                    continue
                if chunk.sttAttempts >= settings.WORKER_MAX_RETRIES and not _is_recoverable_s3_retry(chunk, audio_fields):
                    continue
                retryable_count += 1
                await self.producer.publish(
                    settings.REDIS_STT_STREAM,
                    EventEnvelope(
                        eventType="stt.requested",
                        correlationId=conversation_id,
                        userId=str(chunk.userId),
                        spaceId=str(chunk.spaceId),
                        conversationId=conversation_id,
                        payload={
                            "conversationId": str(chunk.conversationId),
                            "userId": str(chunk.userId),
                            "spaceId": str(chunk.spaceId),
                            "chunkId": chunk.chunkId,
                            "sequenceNumber": chunk.sequenceNumber,
                            **audio_fields,
                        },
                    ),
                )
            target = ConversationStatus.WAITING_FOR_TRANSCRIPTS if retryable_count or pending else ConversationStatus.PARTIAL
            await self.repository.transition(
                conversation_id,
                target,
                {"missingSequences": missing},
            )
            if target == ConversationStatus.PARTIAL:
                await self.producer.publish(
                    settings.REDIS_PROCESSING_STREAM,
                    EventEnvelope(
                        eventType="conversation.processing.requested",
                        correlationId=conversation_id,
                        userId=str(conversation.userId),
                        spaceId=str(conversation.spaceId),
                        conversationId=conversation_id,
                        payload={
                            "processingVersion": conversation.processingVersion,
                            "partial": True,
                            "missingSequences": missing,
                        },
                    ),
                )
            return

        await self.repository.transition(
            conversation_id,
            ConversationStatus.READY_FOR_PROCESSING,
            {"missingSequences": []},
        )
        await self.producer.publish(
            settings.REDIS_PROCESSING_STREAM,
            EventEnvelope(
                eventType="conversation.processing.requested",
                correlationId=conversation_id,
                userId=str(conversation.userId),
                spaceId=str(conversation.spaceId),
                conversationId=conversation_id,
                payload={"processingVersion": conversation.processingVersion},
            ),
        )


def _stt_payload_audio_fields(chunk, audio_chunk: dict | None) -> dict:
    if audio_chunk:
        payload = {
            "filename": audio_chunk.get("filename") or f"{chunk.chunkId}.audio",
            "contentType": audio_chunk.get("contentType") or "audio/wav",
        }
        storage_provider = str(audio_chunk.get("storageProvider") or "").lower()
        s3_bucket = audio_chunk.get("s3Bucket")
        s3_object_key = audio_chunk.get("s3ObjectKey")
        if storage_provider == "s3" or s3_bucket or s3_object_key:
            payload.update(
                {
                    "storageProvider": "s3",
                    "bucket": s3_bucket,
                    "objectKey": s3_object_key,
                }
            )
            return payload
        if use_s3_storage():
            payload.update(_legacy_s3_reference_payload(chunk, audio_chunk))
            return payload
        payload.update(_audio_reference_payload(audio_chunk.get("filePath") or chunk.audioFilePath))
        return payload

    if use_s3_storage():
        return {
            "filename": f"{chunk.chunkId}.audio",
            "contentType": "audio/wav",
            **_legacy_s3_reference_payload(chunk, None),
        }
    return {
        "filename": f"{chunk.chunkId}.audio",
        "contentType": "audio/wav",
        **_audio_reference_payload(chunk.audioFilePath),
    }


def _audio_reference_payload(audio_file_path: str | None) -> dict:
    value = str(audio_file_path or "").strip()
    if value.startswith("s3://"):
        parsed = urlparse(value)
        bucket = parsed.netloc
        object_key = parsed.path.lstrip("/")
        return {
            "storageProvider": "s3",
            "bucket": bucket,
            "objectKey": object_key,
        }
    return {"filePath": value}


def _legacy_s3_reference_payload(chunk, audio_chunk: dict | None) -> dict:
    filename = str((audio_chunk or {}).get("filename") or f"{chunk.chunkId}.audio")
    object_key = build_audio_object_key(
        user_id=str(chunk.userId),
        space_id=str(chunk.spaceId),
        session_id=str(chunk.conversationId),
        job_id=str(chunk.chunkId),
        filename=filename,
    )
    return {
        "storageProvider": "s3",
        "bucket": settings.S3_AUDIO_BUCKET or settings.S3_BUCKET,
        "objectKey": object_key,
    }


def _is_recoverable_s3_retry(chunk, audio_fields: dict) -> bool:
    if str(audio_fields.get("storageProvider") or "").lower() != "s3":
        return False
    error = str(chunk.lastError or "").lower()
    return "missing or empty" in error or "no such file" in error or "resources/audio_jobs" in error


def _is_permanent_audio_failure(chunk) -> bool:
    error = str(chunk.lastError or "").lower()
    permanent_markers = (
        "audio duration exceeds",
        "exceeds the maximum limit",
        "failed to read the file",
        "audio format",
        "invalid audio",
        "file too large",
        "batch api",
        "unsupported audio content type",
    )
    return any(marker in error for marker in permanent_markers)
