"""End-to-end: meeting_extension stop → windows → READY → processing request.

Regression for the two bugs that blocked tasks/notes:
1. 1-based sequences left a phantom hole at 0 (no windows).
2. force_final TOKEN_TARGET mid-closes stayed PENDING waiting for window extraction.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from apps.api_gateway.config.setting import settings
from services.conversation.finalization import ConversationFinalizationCoordinator
from services.conversation.models import (
    ConversationDocument,
    ConversationStatus,
    ConversationWindowDocument,
    STTStatus,
    TranscriptChunkDocument,
    TranscriptProcessingStatus,
    WindowProcessingStatus,
)


def _meeting_chunk(sequence: int, text: str) -> TranscriptChunkDocument:
    return TranscriptChunkDocument(
        conversationId="meet_conv_1",
        userId="user_1",
        spaceId=None,
        chunkId=f"meeting:muxed:{sequence}",
        sequenceNumber=sequence,
        rawText=text,
        normalizedText=text,
        sttStatus=STTStatus.COMPLETED,
        processingStatus=TranscriptProcessingStatus.UNPROCESSED,
        sourceType="meeting_extension",
        meetingSessionId="meet_conv_1",
        endTimeMs=30_000,
    )


class _MeetingRepo:
    def __init__(self, conversation, chunks):
        self.conversation = conversation
        self.chunks = list(chunks)
        self.windows: list[ConversationWindowDocument] = []
        self.db = SimpleNamespace(conversations=SimpleNamespace(update_one=_async_noop))

    async def get_conversation(self, conversation_id):
        return self.conversation

    async def list_transcript_chunks(self, conversation_id):
        return list(self.chunks)

    async def list_conversation_windows(self, conversation_id):
        return list(self.windows)

    async def mark_transcripts_excluded(self, *args, **kwargs):
        return None

    async def create_conversation_window(self, window, owned, skipped_sequence_numbers=None):
        saved = window.model_copy(deep=True)
        saved.status = WindowProcessingStatus.PENDING
        self.windows.append(saved)
        owned_set = set(owned)
        skipped_set = set(skipped_sequence_numbers or [])
        for chunk in self.chunks:
            if chunk.sequenceNumber in owned_set or chunk.sequenceNumber in skipped_set:
                chunk.processingStatus = TranscriptProcessingStatus.PROCESSED
                chunk.processingWindowId = saved.id
        return saved

    async def complete_window(self, window_id, result, provider, model, **kwargs):
        for window in self.windows:
            if str(window.id) == str(window_id):
                window.status = WindowProcessingStatus.COMPLETED
                window.result = result
                window.extractionSkipped = bool(kwargs.get("extraction_skipped"))
                window.checkpointKind = kwargs.get("checkpoint_kind")
                window.artifactPersistenceOk = bool(kwargs.get("artifact_persistence_ok", True))
                return

    async def mark_window_queued(self, *args, **kwargs):
        return None

    async def list_transcript_chunks_in_range(self, *args, **kwargs):
        return []

    async def append_meeting_debug_trace(self, *args, **kwargs):
        return None

    async def count_unwindowed_non_empty_transcripts(self, conversation_id, through_sequence=None):
        through = through_sequence
        return sum(
            1
            for chunk in self.chunks
            if chunk.sttStatus == STTStatus.COMPLETED
            and chunk.processingStatus == TranscriptProcessingStatus.UNPROCESSED
            and (chunk.rawText or "").strip()
            and (through is None or chunk.sequenceNumber <= through)
        )

    async def reclaim_stale_processing_windows(self, *args, **kwargs):
        return []

    async def reclaim_stale_stt_chunks(self, *args, **kwargs):
        return []

    async def get_audio_chunk(self, *args, **kwargs):
        return None

    async def get_extraction_run(self, *args, **kwargs):
        return None

    async def transition(self, conversation_id, status, updates=None):
        self.conversation.status = status
        if updates:
            for key, value in updates.items():
                setattr(self.conversation, key, value)
        return self.conversation

    async def sync_meeting_session_status(self, conversation):
        return None


class _FakeProducer:
    def __init__(self):
        self.events = []

    async def publish(self, stream, event):
        self.events.append((stream, event))
        return event.eventId


async def _async_noop(*args, **kwargs):
    return None


def test_meeting_extension_finalization_reaches_ready_without_window_extraction(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_INCREMENTAL_MEETING_PROCESSING", True)
    monkeypatch.setattr(settings, "ENABLE_MEETING_PIPELINE", True)
    # Force mid-session TOKEN_TARGET closes even under force_final.
    monkeypatch.setattr("services.conversation.windowing.settings.INCREMENTAL_WINDOW_TARGET_TOKENS", 12)
    monkeypatch.setattr("services.conversation.windowing.settings.INCREMENTAL_WINDOW_MAX_TOKENS", 20)

    conversation = ConversationDocument(
        _id="meet_conv_1",
        userId="user_1",
        spaceId=None,
        status=ConversationStatus.STOP_REQUESTED,
        expectedLastSequence=4,
        stoppedAt=datetime.now(timezone.utc),
        receivedAudioChunkCount=4,
        sourceType="meeting_extension",
    )
    chunks = [
        _meeting_chunk(1, " ".join(f"word{n}" for n in range(10))),
        _meeting_chunk(2, " ".join(f"hire{n}" for n in range(10))),
        _meeting_chunk(3, " ".join(f"task{n}" for n in range(10))),
        _meeting_chunk(4, "Please create the follow up note tomorrow."),
    ]
    repo = _MeetingRepo(conversation, chunks)
    producer = _FakeProducer()
    coordinator = ConversationFinalizationCoordinator(repo, producer=producer)

    asyncio.run(coordinator.finalize("meet_conv_1"))

    assert conversation.status == ConversationStatus.READY_FOR_PROCESSING
    assert repo.windows, "force_final must create windows for 1..N meeting chunks"
    assert all(window.status == WindowProcessingStatus.COMPLETED for window in repo.windows)
    assert all(getattr(window, "extractionSkipped", False) for window in repo.windows)
    assert all(chunk.processingStatus == TranscriptProcessingStatus.PROCESSED for chunk in chunks)

    processing_events = [
        event for _, event in producer.events if event.eventType == "conversation.processing.requested"
    ]
    assert processing_events, "READY must enqueue conversation.processing.requested"
    assert processing_events[0].spaceId == ""

    extraction_events = [
        event for _, event in producer.events if event.eventType == "conversation.window.extraction.requested"
    ]
    assert not extraction_events, "meeting pipeline must not wait on live window extraction at stop"


def test_meeting_extension_short_meeting_reaches_ready(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_INCREMENTAL_MEETING_PROCESSING", True)
    monkeypatch.setattr(settings, "ENABLE_MEETING_PIPELINE", True)

    conversation = ConversationDocument(
        _id="meet_conv_1",
        userId="user_1",
        spaceId=None,
        status=ConversationStatus.STOP_REQUESTED,
        expectedLastSequence=2,
        stoppedAt=datetime.now(timezone.utc),
        receivedAudioChunkCount=2,
        sourceType="meeting_extension",
    )
    chunks = [
        _meeting_chunk(1, "Today we discuss hiring."),
        _meeting_chunk(2, "Please create the server ID ticket."),
    ]
    repo = _MeetingRepo(conversation, chunks)
    producer = _FakeProducer()
    coordinator = ConversationFinalizationCoordinator(repo, producer=producer)

    asyncio.run(coordinator.finalize("meet_conv_1"))

    assert conversation.status == ConversationStatus.READY_FOR_PROCESSING
    assert any(event.eventType == "conversation.processing.requested" for _, event in producer.events)


def test_meeting_extension_skips_never_uploaded_leading_chunks_after_stop(monkeypatch):
    """Chunks 1-2 never arrived (AWS 6ab15dd7...); remaining STT must still publish."""
    monkeypatch.setattr(settings, "ENABLE_INCREMENTAL_MEETING_PROCESSING", True)
    monkeypatch.setattr(settings, "ENABLE_MEETING_PIPELINE", True)
    monkeypatch.setattr(settings, "MEETING_EXTENSION_MISSING_SEQUENCE_TIMEOUT_SECONDS", 45)

    conversation = ConversationDocument(
        _id="meet_conv_1",
        userId="user_1",
        spaceId=None,
        status=ConversationStatus.WAITING_FOR_TRANSCRIPTS,
        expectedLastSequence=9,
        stoppedAt=datetime.now(timezone.utc) - timedelta(seconds=60),
        receivedAudioChunkCount=7,
        sourceType="meeting_extension",
    )
    chunks = [
        _meeting_chunk(sequence, f"Useful speech from later chunk {sequence}.")
        for sequence in range(3, 10)
    ]
    repo = _MeetingRepo(conversation, chunks)
    producer = _FakeProducer()
    coordinator = ConversationFinalizationCoordinator(repo, producer=producer)

    asyncio.run(coordinator.finalize("meet_conv_1"))

    assert conversation.status == ConversationStatus.READY_FOR_PROCESSING
    assert repo.windows, "windows must start at first available transcript, not wait for seq 1-2"
    assert repo.windows[0].sequenceStart >= 3
    assert any(event.eventType == "conversation.processing.requested" for _, event in producer.events)
