from __future__ import annotations

import time
from typing import Any

from apps.api_gateway.config.setting import settings
from services.conversation import agents
from services.conversation.context import load_space_context
from services.conversation.models import WindowProcessingStatus
from services.conversation.repository import ConversationRepository
from services.conversation.windowing import build_ready_windows
from services.llm.router import LLMRouter, get_llm_router
from services.queue.streams import EventEnvelope, RedisStreamProducer


class IncrementalMeetingProcessor:
    def __init__(
        self,
        repository: ConversationRepository,
        producer: RedisStreamProducer | None = None,
        router: LLMRouter | None = None,
    ):
        self.repository = repository
        self.producer = producer or RedisStreamProducer()
        self.router = router or get_llm_router()

    async def close_ready_windows(
        self,
        conversation_id: str,
        force_final: bool = False,
        through_sequence: int | None = None,
    ) -> list[str]:
        if not settings.ENABLE_INCREMENTAL_MEETING_PROCESSING:
            return []
        conversation = await self.repository.get_conversation(conversation_id)
        if not conversation:
            return []
        chunks = await self.repository.list_completed_unwindowed_transcript_chunks(
            conversation_id,
            through_sequence=through_sequence,
        )
        existing_windows = await self.repository.list_conversation_windows(conversation_id)
        start_index = len(existing_windows)
        expected_start = existing_windows[-1].sequenceEnd + 1 if existing_windows else 0
        chunks = [chunk for chunk in chunks if chunk.sequenceNumber >= expected_start]
        if chunks and chunks[0].sequenceNumber != expected_start:
            return []
        built_windows = build_ready_windows(conversation, chunks, start_index, force_final=force_final)
        window_ids: list[str] = []
        for built in built_windows:
            saved = await self.repository.create_conversation_window(built.window, built.sequence_numbers)
            window_ids.append(str(saved.id))
            if saved.status in {WindowProcessingStatus.PENDING, WindowProcessingStatus.FAILED}:
                await self.repository.mark_window_queued(saved.id)
                await self.producer.publish(
                    settings.REDIS_WINDOW_EXTRACTION_STREAM,
                    EventEnvelope(
                        eventType="conversation.window.extraction.requested",
                        correlationId=conversation_id,
                        userId=str(conversation.userId),
                        spaceId=str(conversation.spaceId),
                        conversationId=conversation_id,
                        payload={
                            "windowId": str(saved.id),
                            "windowIndex": saved.windowIndex,
                            "sequenceStart": saved.sequenceStart,
                            "sequenceEnd": saved.sequenceEnd,
                        },
                    ),
                )
        if built_windows:
            print(
                "Incremental windows closed:",
                {
                    "conversationId": conversation_id,
                    "windowCount": len(built_windows),
                    "forceFinal": force_final,
                },
            )
        return window_ids

    async def extract_window(self, window_id: str) -> None:
        started = time.perf_counter()
        window = await self.repository.mark_window_processing(window_id)
        if not window:
            existing = await self.repository.get_conversation_window(window_id)
            if existing and existing.status == WindowProcessingStatus.COMPLETED:
                print(
                    "Duplicate window extraction skipped:",
                    {"conversationId": str(existing.conversationId), "windowId": window_id},
                )
                return
            return
        try:
            context = await load_space_context(self.repository, window.userId, window.spaceId)
            result, provider, model = await agents.extract_window(self.router, window, context)
            await self.repository.complete_window(window.id, result, provider, model)
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            print(
                "Window extraction completed:",
                {
                    "conversationId": str(window.conversationId),
                    "windowId": str(window.id),
                    "windowIndex": window.windowIndex,
                    "tokenCount": window.tokenCount,
                    "elapsedMs": elapsed_ms,
                    "provider": provider,
                    "model": model,
                },
            )
        except Exception as error:
            await self.repository.fail_window(window.id, error)
            print(
                "Window extraction failed:",
                {
                    "conversationId": str(window.conversationId),
                    "windowId": str(window.id),
                    "windowIndex": window.windowIndex,
                    "error": str(error)[:500],
                },
            )
            raise


def compact_window_payload(window: Any) -> dict[str, Any]:
    result = window.result
    return {
        "windowIndex": window.windowIndex,
        "sequenceStart": window.sequenceStart,
        "sequenceEnd": window.sequenceEnd,
        "summary": result.summary if result else "",
        "topics": result.topics if result else [],
        "importantFacts": result.importantFacts if result else [],
        "tasks": [item.model_dump() for item in result.tasks] if result else [],
        "notes": [item.model_dump() for item in result.notes] if result else [],
        "decisions": [item.model_dump() for item in result.decisions] if result else [],
        "issues": [item.model_dump() for item in result.issues] if result else [],
        "openQuestions": result.openQuestions if result else [],
    }
