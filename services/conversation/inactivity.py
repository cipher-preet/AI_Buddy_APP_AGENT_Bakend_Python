from __future__ import annotations

from datetime import timedelta

from apps.api_gateway.config.setting import settings
from services.conversation.models import ConversationStatus, utc_now
from services.conversation.repository import ConversationRepository
from services.queue.streams import EventEnvelope, RedisStreamProducer


class ConversationInactivityScanner:
    def __init__(
        self,
        repository: ConversationRepository,
        producer: RedisStreamProducer | None = None,
    ):
        self.repository = repository
        self.producer = producer or RedisStreamProducer()

    async def scan_once(self) -> int:
        cutoff = utc_now() - timedelta(seconds=settings.CONVERSATION_INACTIVITY_TIMEOUT_SECONDS)
        conversations = await self.repository.find_inactive_recording_conversations(cutoff)
        finalized = 0
        for conversation in conversations:
            conversation_id = str(conversation.id)
            last_sequence = await self.repository.infer_last_sequence(conversation_id)
            if last_sequence is None:
                continue
            await self.repository.transition(
                conversation_id,
                ConversationStatus.WAITING_FOR_TRANSCRIPTS,
                {
                    "expectedLastSequence": last_sequence,
                    "stoppedAt": utc_now(),
                    "missingSequences": [],
                },
            )
            await self.producer.publish(
                settings.REDIS_FINALIZATION_STREAM,
                EventEnvelope(
                    eventType="conversation.finalization.requested",
                    correlationId=conversation_id,
                    userId=conversation.userId,
                    spaceId=conversation.spaceId,
                    conversationId=conversation_id,
                    payload={"expectedLastSequence": last_sequence, "source": "inactivity"},
                ),
            )
            finalized += 1
        stale_cutoff = utc_now() - timedelta(seconds=max(30, settings.REDIS_CLAIM_IDLE_MS // 1000))
        stale_conversations = await self.repository.find_stale_unfinalized_conversations(stale_cutoff)
        for conversation in stale_conversations:
            conversation_id = str(conversation.id)
            await self.producer.publish(
                settings.REDIS_FINALIZATION_STREAM,
                EventEnvelope(
                    eventType="conversation.finalization.requested",
                    correlationId=conversation_id,
                    userId=str(conversation.userId),
                    spaceId=str(conversation.spaceId),
                    conversationId=conversation_id,
                    payload={
                        "expectedLastSequence": conversation.expectedLastSequence,
                        "source": "stale-finalization-recovery",
                    },
                ),
            )
            finalized += 1
        return finalized
