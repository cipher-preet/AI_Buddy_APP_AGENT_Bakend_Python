"""meeting_sessions should advance when conversation status changes."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from services.conversation.models import ConversationDocument, ConversationStatus
from services.conversation.repository import ConversationRepository


class _FakeSessions:
    def __init__(self):
        self.updates = []

    async def update_one(self, query, update):
        self.updates.append((query, update))
        return SimpleNamespace(modified_count=1)


def test_sync_meeting_session_marks_intelligence_completed():
    repo = ConversationRepository(SimpleNamespace(meeting_sessions=_FakeSessions()))
    conversation = ConversationDocument(
        userId="user_1",
        spaceId=None,
        status=ConversationStatus.COMPLETED,
        sourceType="meeting_extension",
    )
    asyncio.run(repo.sync_meeting_session_status(conversation))
    assert repo.db.meeting_sessions.updates
    patch = repo.db.meeting_sessions.updates[0][1]["$set"]
    assert patch["intelligenceStatus"] == "completed"
    assert patch["transcriptStatus"] == "completed"
    assert patch["processingStatus"] == "completed"


def test_sync_meeting_session_ignores_non_extension():
    repo = ConversationRepository(SimpleNamespace(meeting_sessions=_FakeSessions()))
    conversation = ConversationDocument(
        userId="user_1",
        spaceId="space_1",
        status=ConversationStatus.COMPLETED,
        sourceType="mobile",
    )
    asyncio.run(repo.sync_meeting_session_status(conversation))
    assert repo.db.meeting_sessions.updates == []
