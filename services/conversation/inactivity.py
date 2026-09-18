from __future__ import annotations

from datetime import timedelta
import time

from apps.api_gateway.config.setting import settings
from services.conversation.models import ConversationStatus, as_utc, utc_now
from services.conversation.repository import ConversationRepository
from services.observability.diagnostics import (
    conversation_republishes,
    diag_log,
    drain_stt_requeued,
)
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
        started = time.perf_counter()
        cutoff = utc_now() - timedelta(seconds=settings.CONVERSATION_INACTIVITY_TIMEOUT_SECONDS)
        conversations = await self.repository.find_inactive_recording_conversations(cutoff)
        finalized = 0
        finalization_events_published = 0
        processing_events_published = 0
        for conversation in conversations:
            if getattr(conversation, "sourceType", None) == "meeting_extension":
                continue
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
            event = EventEnvelope(
                eventType="conversation.finalization.requested",
                correlationId=conversation_id,
                userId=conversation.userId,
                spaceId=conversation.spaceId,
                conversationId=conversation_id,
                payload={"expectedLastSequence": last_sequence, "source": "inactivity"},
            )
            await self.producer.publish(settings.REDIS_FINALIZATION_STREAM, event)
            finalization_events_published += 1
            finalized += 1
        stale_cutoff = utc_now() - timedelta(seconds=max(30, settings.REDIS_CLAIM_IDLE_MS // 1000))
        stale_conversations = await self.repository.find_stale_unfinalized_conversations(stale_cutoff)
        for conversation in stale_conversations:
            conversation_id = str(conversation.id)
            republish_count = conversation_republishes.record(conversation_id)
            updated_at = as_utc(conversation.updatedAt) if conversation.updatedAt else None
            updated_age_seconds = (
                int((utc_now() - updated_at).total_seconds()) if updated_at is not None else None
            )
            if conversation.status == ConversationStatus.READY_FOR_PROCESSING:
                event = EventEnvelope(
                    eventType="conversation.processing.requested",
                    correlationId=conversation_id,
                    userId=str(conversation.userId),
                    spaceId=str(conversation.spaceId),
                    conversationId=conversation_id,
                    payload={
                        "expectedLastSequence": conversation.expectedLastSequence,
                        "processingVersion": conversation.processingVersion,
                        "source": "stale-processing-recovery",
                    },
                )
                await self.producer.publish(settings.REDIS_PROCESSING_STREAM, event)
                destination = settings.REDIS_PROCESSING_STREAM
                processing_events_published += 1
            else:
                event = EventEnvelope(
                    eventType="conversation.finalization.requested",
                    correlationId=conversation_id,
                    userId=str(conversation.userId),
                    spaceId=str(conversation.spaceId),
                    conversationId=conversation_id,
                    payload={
                        "expectedLastSequence": conversation.expectedLastSequence,
                        "source": "stale-finalization-recovery",
                    },
                )
                await self.producer.publish(settings.REDIS_FINALIZATION_STREAM, event)
                destination = settings.REDIS_FINALIZATION_STREAM
                finalization_events_published += 1
            diag_log(
                "conversation_republished",
                conversation_id=conversation_id,
                status=conversation.status.value if hasattr(conversation.status, "value") else str(conversation.status),
                updated_age_seconds=updated_age_seconds,
                destination_stream=destination,
                new_event_id=event.eventId,
                attempt=event.attempt,
                republish_count=republish_count,
            )
            finalized += 1
        diag_log(
            "conversation_inactivity_scan",
            scan_duration_ms=int((time.perf_counter() - started) * 1000),
            stale_found=len(stale_conversations),
            finalization_events_published=finalization_events_published,
            processing_events_published=processing_events_published,
            stt_jobs_requeued=drain_stt_requeued(),
        )
        diag_log(
            "mongo_query_timing",
            operation="conversation_inactivity_scan",
            duration_ms=int((time.perf_counter() - started) * 1000),
            result_count=len(conversations) + len(stale_conversations),
        )
        return finalized
