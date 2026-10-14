"""Regression: persist upserts must never $set immutable _id (Mongo error 66)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from bson import ObjectId

from services.conversation.models import (
    ConversationSummaryDocument,
    ExtractionRunDocument,
    MeetingMemoryDocument,
    SpaceMemoryDocument,
)
from services.conversation.repository import ConversationRepository


class _CapturingCollection:
    def __init__(self) -> None:
        self.update_calls: list[tuple[dict, dict]] = []
        self.find_one_and_update_calls: list[tuple[dict, dict]] = []

    async def update_one(self, filt, update, **kwargs):
        self.update_calls.append((filt, update))
        return SimpleNamespace(matched_count=1, modified_count=1, upserted_id=None)

    async def find_one_and_update(self, filt, update, **kwargs):
        self.find_one_and_update_calls.append((filt, update))
        # Return a document that validates as the requested model fields.
        base = dict(update.get("$set") or {})
        if "$setOnInsert" in update:
            base = {**update["$setOnInsert"], **base}
        return base


@pytest.mark.asyncio
async def test_publish_outputs_does_not_set_id_on_summary_or_space_memory():
    summaries = _CapturingCollection()
    space_memory = _CapturingCollection()
    repo = ConversationRepository(
        SimpleNamespace(
            conversation_summaries=summaries,
            space_memory=space_memory,
        )
    )
    repo._publish_tasks = AsyncMock(return_value=[])  # type: ignore[method-assign]
    repo._publish_notes = AsyncMock(return_value=[])  # type: ignore[method-assign]

    user_id = ObjectId()
    space_id = ObjectId()
    conversation_id = ObjectId()
    run = ExtractionRunDocument(
        conversationId=conversation_id,
        userId=user_id,
        spaceId=space_id,
        processingVersion=1,
        provider="test",
        model="test",
    )
    summary = ConversationSummaryDocument(
        conversationId=conversation_id,
        userId=user_id,
        spaceId=space_id,
        summary="hello",
        processingVersion=1,
        modelProvider="test",
        modelName="test",
        promptVersion="v1",
    )
    memory = SpaceMemoryDocument(userId=user_id, spaceId=space_id, currentSummary="mem")

    await repo.publish_outputs(run, summary, memory)

    assert len(summaries.update_calls) == 1
    _, summary_update = summaries.update_calls[0]
    assert "_id" not in summary_update["$set"]
    assert "_id" in summary_update["$setOnInsert"]

    assert len(space_memory.update_calls) == 1
    _, memory_update = space_memory.update_calls[0]
    assert "_id" not in memory_update["$set"]
    assert "_id" in memory_update["$setOnInsert"]


@pytest.mark.asyncio
async def test_save_meeting_memory_does_not_set_id():
    meeting_memory = _CapturingCollection()
    repo = ConversationRepository(SimpleNamespace(meeting_memory=meeting_memory))

    memory = MeetingMemoryDocument(
        conversationId=ObjectId(),
        userId=ObjectId(),
        spaceId=ObjectId(),
        shortSummary="s",
    )
    await repo.save_meeting_memory(memory)

    assert len(meeting_memory.find_one_and_update_calls) == 1
    _, update = meeting_memory.find_one_and_update_calls[0]
    assert "_id" not in update["$set"]
    assert "_id" in update["$setOnInsert"]


@pytest.mark.asyncio
async def test_publish_outputs_second_pass_with_existing_docs_excludes_id():
    """Simulates retry after summary/space_memory already exist — the AWS failure mode."""
    summaries = _CapturingCollection()
    space_memory = _CapturingCollection()
    repo = ConversationRepository(
        SimpleNamespace(
            conversation_summaries=summaries,
            space_memory=space_memory,
        )
    )
    repo._publish_tasks = AsyncMock(return_value=[])  # type: ignore[method-assign]
    repo._publish_notes = AsyncMock(return_value=[])  # type: ignore[method-assign]

    user_id = ObjectId()
    space_id = ObjectId()
    conversation_id = ObjectId()
    run = ExtractionRunDocument(
        conversationId=conversation_id,
        userId=user_id,
        spaceId=space_id,
        processingVersion=1,
        provider="test",
        model="test",
    )

    for _ in range(2):
        summary = ConversationSummaryDocument(
            conversationId=conversation_id,
            userId=user_id,
            spaceId=space_id,
            summary="retry summary",
            processingVersion=1,
            modelProvider="test",
            modelName="test",
            promptVersion="v1",
        )
        memory = SpaceMemoryDocument(userId=user_id, spaceId=space_id, currentSummary="retry mem")
        await repo.publish_outputs(run, summary, memory)

    assert len(summaries.update_calls) == 2
    for _, update in summaries.update_calls:
        assert "_id" not in update["$set"]
        assert "$setOnInsert" in update
    for _, update in space_memory.update_calls:
        assert "_id" not in update["$set"]
        assert "$setOnInsert" in update
