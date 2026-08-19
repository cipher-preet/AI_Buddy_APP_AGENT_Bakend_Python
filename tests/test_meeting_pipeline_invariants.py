from datetime import datetime, timedelta, timezone

from services.conversation.agents import _needs_window_recovery
from services.conversation.artifact_resolver import item_is_represented
from services.conversation.coverage import _weak_window_indexes
from services.conversation.finalization import _is_lost_stt_job, _session_readiness, _should_publish_window_job
from services.conversation.models import (
    ConversationDocument,
    ConversationStatus,
    ConversationWindowDocument,
    STTStatus,
    TranscriptChunkDocument,
    TranscriptProcessingStatus,
    WindowExtractionResult,
    WindowProcessingStatus,
)
from services.conversation.windowing import (
    CLOSE_REASON_FORCED_FINAL,
    CLOSE_REASON_SPARSE_TIMEOUT,
    build_ready_windows,
    is_useful_chunk,
)


def _chunk(
    sequence: int,
    text: str = "",
    duration_ms: int = 30_000,
    status: STTStatus = STTStatus.COMPLETED,
    window_id=None,
    attempts: int = 0,
    error: str | None = None,
) -> TranscriptChunkDocument:
    return TranscriptChunkDocument(
        conversationId="conv_1",
        userId="user_1",
        spaceId="space_1",
        chunkId=f"chunk_{sequence}",
        sequenceNumber=sequence,
        rawText=text,
        sttStatus=status,
        processingStatus=TranscriptProcessingStatus.UNPROCESSED,
        processingWindowId=window_id,
        endTimeMs=duration_ms,
        sttAttempts=attempts,
        lastError=error,
    )


def _conversation(expected_last: int = 119, status: ConversationStatus = ConversationStatus.STOP_REQUESTED) -> ConversationDocument:
    return ConversationDocument(
        _id="conv_1",
        userId="user_1",
        spaceId="space_1",
        status=status,
        expectedLastSequence=expected_last,
        stoppedAt=datetime.now(timezone.utc),
        receivedAudioChunkCount=expected_last + 1,
    )


def test_dense_speech_still_closes_on_token_target(monkeypatch):
    monkeypatch.setattr("services.conversation.windowing.settings.INCREMENTAL_WINDOW_TARGET_TOKENS", 20)
    monkeypatch.setattr("services.conversation.windowing.settings.INCREMENTAL_WINDOW_MAX_TOKENS", 30)
    conversation = _conversation(5)
    chunks = [
        _chunk(index, " ".join(f"word{n}" for n in range(8)))
        for index in range(6)
    ]

    windows = build_ready_windows(conversation, chunks, start_index=0, force_final=False)

    assert len(windows) >= 1
    assert windows[0].window.closeReason in {"token_target", "token_max"}
    assert windows[0].window.emptyChunkCount == 0
    assert windows[0].window.nonEmptyChunkCount >= 1


def test_empty_chunks_do_not_create_or_close_semantic_windows(monkeypatch):
    monkeypatch.setattr("services.conversation.windowing.settings.INCREMENTAL_WINDOW_MAX_DURATION_MS", 5 * 60 * 1000)
    conversation = _conversation(9)
    chunks = [_chunk(index, "") for index in range(10)]

    windows = build_ready_windows(conversation, chunks, start_index=0, force_final=True)

    assert windows == []


def test_silence_heavy_meeting_windows_all_useful_transcripts(monkeypatch):
    monkeypatch.setattr("services.conversation.windowing.settings.INCREMENTAL_WINDOW_TARGET_TOKENS", 1200)
    monkeypatch.setattr("services.conversation.windowing.settings.SPARSE_WINDOW_MAX_WALL_CLOCK_MS", 60 * 60 * 1000)
    conversation = _conversation(119)
    even = set(range(0, 120, 2))
    extra_empty = set(range(1, 16, 2))
    empty_indexes = even | extra_empty
    useful_indexes = set(range(120)) - empty_indexes
    chunks = []
    for index in range(120):
        text = f"Speaker discussed action item {index} for the release." if index in useful_indexes else ""
        chunks.append(_chunk(index, text))

    windows = build_ready_windows(conversation, chunks, start_index=0, force_final=True)

    owned = [sequence for built in windows for sequence in built.owned_sequence_numbers]
    texts = "\n".join(built.window.text for built in windows)
    assert len(useful_indexes) == 52
    assert sorted(owned) == sorted(useful_indexes)
    assert all(is_useful_chunk(chunk) or chunk.sequenceNumber not in owned for chunk in chunks)
    assert "[0] " not in texts or 0 in useful_indexes
    for built in windows:
        assert built.window.nonEmptyChunkCount > 0
        assert built.window.text.strip()
        for sequence in built.skipped_sequence_numbers:
            assert sequence not in useful_indexes


def test_empty_placeholders_are_omitted_from_window_text():
    conversation = _conversation(4)
    chunks = [
        _chunk(0, ""),
        _chunk(1, ""),
        _chunk(2, "We should deploy the backend tonight."),
        _chunk(3, ""),
        _chunk(4, "Rahul should check pricing."),
    ]

    windows = build_ready_windows(conversation, chunks, start_index=0, force_final=True)

    assert len(windows) == 1
    text = windows[0].window.text
    assert "[2] We should deploy the backend tonight." in text
    assert "[4] Rahul should check pricing." in text
    assert "[0]" not in text
    assert "[1]" not in text
    assert "[3]" not in text
    assert windows[0].window.closeReason == CLOSE_REASON_FORCED_FINAL


def test_sparse_timeout_closes_small_useful_window(monkeypatch):
    monkeypatch.setattr("services.conversation.windowing.settings.SPARSE_WINDOW_MAX_WALL_CLOCK_MS", 120_000)
    monkeypatch.setattr("services.conversation.windowing.settings.SPARSE_WINDOW_MIN_USEFUL_TOKENS", 4)
    monkeypatch.setattr("services.conversation.windowing.settings.INCREMENTAL_WINDOW_TARGET_TOKENS", 1200)
    conversation = _conversation(5)
    chunks = [
        _chunk(0, ""),
        _chunk(1, "Rahul will deploy production tomorrow.", duration_ms=30_000),
        _chunk(2, "", duration_ms=30_000),
        _chunk(3, "", duration_ms=30_000),
        _chunk(4, "", duration_ms=30_000),
        _chunk(5, "", duration_ms=30_000),
    ]

    windows = build_ready_windows(conversation, chunks, start_index=0, force_final=False)

    assert len(windows) == 1
    assert windows[0].window.closeReason == CLOSE_REASON_SPARSE_TIMEOUT
    assert "Rahul will deploy production tomorrow." in windows[0].window.text
    assert windows[0].owned_sequence_numbers == [1]


def test_terminal_failed_gap_does_not_block_later_useful_chunks(monkeypatch):
    monkeypatch.setattr("services.conversation.windowing.settings.INCREMENTAL_WINDOW_TARGET_TOKENS", 1200)
    conversation = _conversation(5)
    chunks = [
        _chunk(1, "Kickoff is done."),
        _chunk(2, "We confirmed the owner."),
        _chunk(4, "Continue after the failed chunk."),
        _chunk(5, "Final action is assigned."),
    ]

    windows = build_ready_windows(conversation, chunks, start_index=0, force_final=True, skippable_sequences={3})

    owned = [sequence for built in windows for sequence in built.owned_sequence_numbers]
    assert 1 in owned
    assert 2 in owned
    assert 4 in owned
    assert 5 in owned
    assert 3 not in owned


def test_temporary_gap_blocks_later_chunks_until_resolved():
    conversation = _conversation(12)
    pending_gap = [
        _chunk(10, "First useful block."),
        _chunk(12, "Should wait for sequence 11."),
    ]
    windows = build_ready_windows(conversation, pending_gap, start_index=0, force_final=True)
    owned = [sequence for built in windows for sequence in built.owned_sequence_numbers]
    assert 12 not in owned

    resolved = [
        _chunk(10, "First useful block."),
        _chunk(11, "Gap filled with useful speech."),
        _chunk(12, "Should wait for sequence 11."),
    ]
    windows = build_ready_windows(conversation, resolved, start_index=0, force_final=True)
    owned = [sequence for built in windows for sequence in built.owned_sequence_numbers]
    assert owned == [10, 11, 12]


def test_session_readiness_waits_for_pending_stt():
    conversation = _conversation(4)
    chunks = [
        _chunk(0, "hello"),
        _chunk(1, "world"),
        _chunk(2, status=STTStatus.PENDING),
        _chunk(3, "later"),
        _chunk(4, ""),
    ]
    skippable, unresolved, accounting = _session_readiness(conversation, chunks, [])
    assert unresolved is True
    assert accounting["pendingTranscripts"] == 1
    assert accounting["validUnwindowed"] == 3
    assert 4 in skippable


def test_session_readiness_waits_for_incomplete_windows_via_accounting():
    conversation = _conversation(1)
    chunks = [
        _chunk(0, "hello", window_id="win_1"),
        _chunk(1, "world", window_id="win_1"),
    ]
    chunks[0].processingStatus = TranscriptProcessingStatus.PROCESSED
    chunks[1].processingStatus = TranscriptProcessingStatus.PROCESSED
    windows = [
        ConversationWindowDocument(
            conversationId="conv_1",
            userId="user_1",
            spaceId="space_1",
            windowIndex=0,
            sequenceStart=0,
            sequenceEnd=1,
            text="[0] hello\n[1] world",
            tokenCount=4,
            status=WindowProcessingStatus.PROCESSING,
            nonEmptyChunkCount=2,
        )
    ]
    _, unresolved, accounting = _session_readiness(conversation, chunks, windows)
    assert unresolved is False
    assert accounting["windowsProcessing"] == 1
    assert accounting["validUnwindowed"] == 0
    assert accounting["validWindowed"] == 2


def test_missing_sequences_are_not_terminal_immediately():
    conversation = _conversation(5)
    conversation.stoppedAt = datetime.now(timezone.utc)
    conversation.finalizationAttempts = 0
    chunks = [_chunk(index, f"text {index}") for index in range(4)]
    _, unresolved, accounting = _session_readiness(conversation, chunks, [])
    assert unresolved is True
    assert accounting["missingSequences"] == [4, 5]


def test_missing_sequences_become_skippable_after_timeout():
    conversation = _conversation(5)
    conversation.stoppedAt = datetime.now(timezone.utc) - timedelta(seconds=1000)
    chunks = [_chunk(index, f"text {index}") for index in range(4)]
    skippable, unresolved, accounting = _session_readiness(conversation, chunks, [])
    assert unresolved is False
    assert 4 in skippable
    assert 5 in skippable
    assert accounting["permanentFailures"] >= 2


def test_missing_sequences_do_not_timeout_from_event_count():
    conversation = _conversation(5)
    conversation.stoppedAt = datetime.now(timezone.utc)
    conversation.finalizationAttempts = 99
    chunks = [_chunk(index, f"text {index}") for index in range(4)]
    _, unresolved, accounting = _session_readiness(conversation, chunks, [])
    assert unresolved is True
    assert accounting["missingSequences"] == [4, 5]


def test_naive_stopped_at_does_not_crash_finalization():
    conversation = _conversation(5)
    conversation.stoppedAt = datetime.now(timezone.utc).replace(tzinfo=None)
    chunks = [_chunk(index, f"text {index}") for index in range(4)]
    _, unresolved, accounting = _session_readiness(conversation, chunks, [])
    assert unresolved is True
    assert accounting["missingSequences"] == [4, 5]


def test_naive_old_stopped_at_still_times_out():
    conversation = _conversation(5)
    conversation.stoppedAt = (datetime.now(timezone.utc) - timedelta(seconds=1000)).replace(tzinfo=None)
    chunks = [_chunk(index, f"text {index}") for index in range(4)]
    skippable, unresolved, accounting = _session_readiness(conversation, chunks, [])
    assert unresolved is False
    assert 4 in skippable
    assert 5 in skippable


def test_mongo_naive_conversation_document_is_utc_aware():
    conversation = ConversationDocument.model_validate(
        {
            "userId": "user_1",
            "spaceId": "space_1",
            "status": ConversationStatus.WAITING_FOR_TRANSCRIPTS,
            "stoppedAt": datetime(2026, 8, 19, 12, 0, 0),
            "updatedAt": datetime(2026, 8, 19, 12, 0, 0),
        }
    )
    assert conversation.stoppedAt is not None
    assert conversation.stoppedAt.tzinfo is not None
    assert conversation.updatedAt.tzinfo is not None


def test_lost_stt_job_helper_ignores_in_flight_work():
    stale_before = datetime.now(timezone.utc) - timedelta(seconds=10)
    fresh_pending = _chunk(0, status=STTStatus.PENDING)
    fresh_pending.updatedAt = datetime.now(timezone.utc)
    stale_pending = _chunk(1, status=STTStatus.PENDING)
    stale_pending.updatedAt = stale_before - timedelta(seconds=1)
    processing = _chunk(2, status=STTStatus.PROCESSING)
    processing.updatedAt = stale_before - timedelta(seconds=1)
    naive_stale = _chunk(3, status=STTStatus.PENDING)
    naive_stale.updatedAt = (datetime.now(timezone.utc) - timedelta(seconds=60)).replace(tzinfo=None)
    assert _is_lost_stt_job(fresh_pending, stale_before) is False
    assert _is_lost_stt_job(stale_pending, stale_before) is True
    assert _is_lost_stt_job(processing, stale_before) is False
    assert _is_lost_stt_job(naive_stale, stale_before) is True


def test_window_job_publish_skips_already_queued_pending():
    stale_before = datetime.now(timezone.utc) - timedelta(seconds=10)
    queued = ConversationWindowDocument(
        conversationId="conv_1",
        userId="user_1",
        spaceId="space_1",
        windowIndex=0,
        sequenceStart=0,
        sequenceEnd=1,
        text="hello",
        tokenCount=1,
        status=WindowProcessingStatus.PENDING,
        queuedAt=datetime.now(timezone.utc),
        updatedAt=datetime.now(timezone.utc),
    )
    unqueued = queued.model_copy(update={"queuedAt": None})
    failed = queued.model_copy(update={"status": WindowProcessingStatus.FAILED})
    assert _should_publish_window_job(queued, stale_before) is False
    assert _should_publish_window_job(unqueued, stale_before) is True
    assert _should_publish_window_job(failed, stale_before) is True
    conversation = _conversation(5)
    conversation.stoppedAt = datetime.now(timezone.utc)
    conversation.finalizationAttempts = 99
    chunks = [_chunk(index, f"text {index}") for index in range(4)]
    _, unresolved, accounting = _session_readiness(conversation, chunks, [])
    assert unresolved is True
    assert accounting["missingSequences"] == [4, 5]


def test_sparse_one_liner_triggers_window_recovery():
    result = WindowExtractionResult()
    text = "[12] Rahul will deploy production tomorrow."
    assert _needs_window_recovery(result, text) is True


def test_empty_window_text_does_not_trigger_recovery():
    result = WindowExtractionResult()
    assert _needs_window_recovery(result, "   ") is False


def test_coverage_includes_small_meaningful_windows():
    window = ConversationWindowDocument(
        conversationId="conv_1",
        userId="user_1",
        spaceId="space_1",
        windowIndex=7,
        sequenceStart=12,
        sequenceEnd=12,
        text="[12] Rahul will deploy production tomorrow.",
        tokenCount=7,
        nonEmptyChunkCount=1,
        usefulTokenCount=7,
    )
    assert _weak_window_indexes([window], []) == [7]


def test_coverage_ignores_empty_windows():
    window = ConversationWindowDocument(
        conversationId="conv_1",
        userId="user_1",
        spaceId="space_1",
        windowIndex=3,
        sequenceStart=8,
        sequenceEnd=10,
        text="",
        tokenCount=0,
        nonEmptyChunkCount=0,
    )
    assert _weak_window_indexes([window], []) == []


def test_related_titles_are_not_collapsed_when_preserving():
    assert item_is_represented("Deploy backend", ["Test backend deployment"], strict=True) is False
    assert item_is_represented("Check deployment logs", ["Deploy backend"], strict=True) is False
    assert item_is_represented("Deploy backend", ["Deploy backend"], strict=True) is True


def test_accounting_invariant_windowed_equals_useful(monkeypatch):
    monkeypatch.setattr("services.conversation.windowing.settings.INCREMENTAL_WINDOW_TARGET_TOKENS", 80)
    conversation = _conversation(9)
    chunks = [
        _chunk(0, ""),
        _chunk(1, "First useful statement about the launch."),
        _chunk(2, ""),
        _chunk(3, "Second useful statement about owners."),
        _chunk(4, ""),
        _chunk(5, "Third useful statement about deadlines."),
        _chunk(6, ""),
        _chunk(7, "Fourth useful statement about follow-up."),
        _chunk(8, ""),
        _chunk(9, "Fifth useful statement about testing."),
    ]
    windows = build_ready_windows(conversation, chunks, start_index=0, force_final=True)
    useful = [chunk.sequenceNumber for chunk in chunks if is_useful_chunk(chunk)]
    windowed = [sequence for built in windows for sequence in built.owned_sequence_numbers]
    assert sorted(windowed) == useful
    for chunk in chunks:
        chunk.processingWindowId = "win" if chunk.sequenceNumber in windowed else None
        if not is_useful_chunk(chunk):
            chunk.exclusionReason = "empty_transcript"
        else:
            chunk.processingStatus = TranscriptProcessingStatus.PROCESSED
    completed_windows = []
    for built in windows:
        built.window.status = WindowProcessingStatus.COMPLETED
        completed_windows.append(built.window)
    _, unresolved, accounting = _session_readiness(conversation, chunks, completed_windows)
    assert unresolved is False
    assert accounting["validUnwindowed"] == 0
    assert accounting["validTranscripts"] == len(useful)
    assert accounting["validWindowed"] == len(useful)
