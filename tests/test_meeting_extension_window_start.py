"""Regression: meeting_extension 1-based sequences must create windows."""

from __future__ import annotations

import asyncio

from services.conversation.incremental import IncrementalMeetingProcessor
from services.conversation.models import (
    ConversationDocument,
    ConversationStatus,
    STTStatus,
    TranscriptChunkDocument,
    TranscriptProcessingStatus,
    WindowProcessingStatus,
)
from services.conversation.transcript import first_sequence_for_conversation


def _meeting_chunk(sequence: int, text: str) -> TranscriptChunkDocument:
    return TranscriptChunkDocument(
        conversationId="6ab12893e28d5c4855045546",
        userId="user_1",
        spaceId=None,
        chunkId=f"meeting:muxed:{sequence}",
        sequenceNumber=sequence,
        rawText=text,
        normalizedText=text,
        sttStatus=STTStatus.COMPLETED,
        processingStatus=TranscriptProcessingStatus.UNPROCESSED,
        sourceType="meeting_extension",
        meetingSessionId="6ab12893e28d5c4855045546",
    )


class _FakeRepo:
    def __init__(self, conversation, chunks):
        self.conversation = conversation
        self.chunks = chunks
        self.windows = []
        self.created = []

    async def get_conversation(self, conversation_id):
        return self.conversation

    async def list_transcript_chunks(self, conversation_id):
        return list(self.chunks)

    async def list_conversation_windows(self, conversation_id):
        return list(self.windows)

    async def mark_transcripts_excluded(self, *args, **kwargs):
        return None

    async def create_conversation_window(self, window, owned, skipped_sequence_numbers=None):
        self.created.append(
            {
                "sequenceStart": window.sequenceStart,
                "sequenceEnd": window.sequenceEnd,
                "owned": list(owned),
                "skipped": list(skipped_sequence_numbers or []),
            }
        )
        saved = window.model_copy(deep=True)
        self.windows.append(saved)
        owned_set = set(owned)
        skipped_set = set(skipped_sequence_numbers or [])
        for chunk in self.chunks:
            if chunk.sequenceNumber in owned_set or chunk.sequenceNumber in skipped_set:
                chunk.processingStatus = TranscriptProcessingStatus.PROCESSED
        return saved

    async def complete_window(self, window_id, *args, **kwargs):
        for window in self.windows:
            if str(window.id) == str(window_id):
                window.status = WindowProcessingStatus.COMPLETED
                window.extractionSkipped = bool(kwargs.get("extraction_skipped", True))
                return

    async def mark_window_queued(self, *args, **kwargs):
        return None

    async def list_transcript_chunks_in_range(self, *args, **kwargs):
        return []

    async def append_meeting_debug_trace(self, *args, **kwargs):
        return None


class _FakeProducer:
    def __init__(self):
        self.events = []

    async def publish(self, stream, event):
        self.events.append((stream, event))


def test_first_sequence_for_meeting_extension_is_one():
    conversation = ConversationDocument(
        userId="user_1",
        spaceId=None,
        status=ConversationStatus.FINALIZING,
        expectedLastSequence=7,
        sourceType="meeting_extension",
    )
    assert first_sequence_for_conversation(conversation) == 1


def test_close_ready_windows_does_not_block_on_phantom_sequence_zero(monkeypatch):
    monkeypatch.setattr(
        "services.conversation.incremental.settings.ENABLE_INCREMENTAL_MEETING_PROCESSING",
        True,
    )
    conversation = ConversationDocument(
        userId="user_1",
        spaceId=None,
        status=ConversationStatus.FINALIZING,
        expectedLastSequence=3,
        sourceType="meeting_extension",
    )
    chunks = [
        _meeting_chunk(1, "Start leaves discussion."),
        _meeting_chunk(2, "Today we discuss hiring."),
        _meeting_chunk(3, "Please create the server ID."),
    ]
    repo = _FakeRepo(conversation, chunks)
    producer = _FakeProducer()
    processor = IncrementalMeetingProcessor(repo, producer)

    window_ids = asyncio.run(
        processor.close_ready_windows(
            str(conversation.id),
            force_final=True,
            through_sequence=3,
        )
    )

    assert window_ids, "meeting_extension 1..N must create windows without waiting for sequence 0"
    assert repo.created, "at least one window should be persisted"
    assert repo.created[0]["sequenceStart"] >= 1
    assert all(chunk.processingStatus == TranscriptProcessingStatus.PROCESSED for chunk in chunks)
