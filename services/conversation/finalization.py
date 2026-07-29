from __future__ import annotations

from apps.api_gateway.config.setting import settings
from services.conversation.models import ConversationStatus, STTStatus
from services.conversation.repository import ConversationRepository
from services.conversation.transcript import detect_missing_sequences
from services.queue.streams import EventEnvelope, RedisStreamProducer


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
            retryable = [chunk for chunk in failed if chunk.audioFilePath and chunk.sttAttempts < settings.WORKER_MAX_RETRIES]
            for chunk in retryable:
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
                            "filePath": chunk.audioFilePath,
                            "filename": f"{chunk.chunkId}.audio",
                            "contentType": "audio/wav",
                        },
                    ),
                )
            target = ConversationStatus.WAITING_FOR_TRANSCRIPTS if retryable or pending else ConversationStatus.PARTIAL
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
