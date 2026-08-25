import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

from bson import ObjectId

from apps.api_gateway.config.setting import settings
from apps.api_gateway.workers import conversation_workers
from services.conversation.finalization import ConversationFinalizationCoordinator, _session_readiness
from services.conversation.models import ConversationDocument, ConversationStatus, STTStatus, TranscriptChunkDocument
from services.conversation.repository import ConversationRepository
from services.conversation.semantic_input import assemble_semantic_window_input
from services.conversation.stt_failure import (
    FAILURE_CORRUPT_AUDIO,
    FAILURE_PROVIDER_5XX,
    FAILURE_RETRY_EXHAUSTED,
    FAILURE_S3_OBJECT_MISSING,
    TERMINAL_FAILED_PERMANENTLY,
    classify_stt_failure,
    is_terminal_failed_chunk,
)
from services.conversation.windowing import is_useful_chunk
from services.queue.streams import EventEnvelope, NonRetryableQueueError, RedisStreamConsumer
from services.speech.errors import STTPermanentAudioError, STTProviderTemporaryError
from services.storage.s3_audio_storage import PermanentS3StorageError


def _chunk(sequence, text="", status=STTStatus.COMPLETED, **kwargs):
    return TranscriptChunkDocument(
        conversationId=kwargs.get("conversationId", "conv_1"),
        userId="user_1",
        spaceId="space_1",
        chunkId=kwargs.get("chunkId", f"chunk_{sequence}"),
        sequenceNumber=sequence,
        rawText=text,
        sttStatus=status,
        sttAttempts=kwargs.get("sttAttempts", 0),
        lastError=kwargs.get("lastError"),
        failureType=kwargs.get("failureType"),
        failureStage=kwargs.get("failureStage"),
        terminal=kwargs.get("terminal", False),
        jobId=kwargs.get("jobId"),
        exclusionReason=kwargs.get("exclusionReason"),
        processingWindowId=kwargs.get("processingWindowId"),
    )


def _conversation(expected_last, status=ConversationStatus.STOP_REQUESTED):
    return ConversationDocument(
        _id="conv_1",
        userId="user_1",
        spaceId="space_1",
        status=status,
        expectedLastSequence=expected_last,
        stoppedAt=datetime.now(timezone.utc),
        receivedAudioChunkCount=expected_last + 1,
    )


def _stt_event(sequence=0, **payload):
    body = {
        "conversationId": "conv_1",
        "sequenceNumber": sequence,
        "jobId": payload.get("jobId", f"job-{sequence}"),
        "chunkId": payload.get("chunkId", f"chunk_{sequence}"),
        "filePath": payload.get("filePath", "/tmp/audio.wav"),
        "filename": "audio.wav",
        "contentType": "audio/wav",
        **payload,
    }
    return EventEnvelope(
        eventType="stt.requested",
        correlationId="conv_1",
        userId="user-1",
        spaceId="space-1",
        conversationId="conv_1",
        payload=body,
        attempt=payload.get("attempt", 0),
    )


class MemoryTranscriptRepository:
    def __init__(self, chunks=None, conversation=None):
        self.chunks = {int(chunk.sequenceNumber): chunk for chunk in chunks or []}
        self.conversation = conversation or _conversation(0)
        self.failed_count = 0
        self.fail_calls = []
        self.processing_calls = []

    async def get_transcript_chunk(self, conversation_id, sequence_number):
        return self.chunks.get(int(sequence_number))

    async def mark_transcript_chunk_processing(self, conversation_id, sequence_number):
        chunk = self.chunks.get(int(sequence_number))
        if chunk is None or chunk.sttStatus == STTStatus.COMPLETED or is_terminal_failed_chunk(chunk):
            return False
        chunk.sttStatus = STTStatus.PROCESSING
        self.processing_calls.append(int(sequence_number))
        return True

    async def fail_transcript_chunk(self, conversation_id, sequence_number, error, **kwargs):
        self.fail_calls.append((int(sequence_number), error, kwargs))
        chunk = self.chunks.get(int(sequence_number))
        if chunk is None:
            return False
        if chunk.sttStatus == STTStatus.COMPLETED or is_terminal_failed_chunk(chunk):
            return False
        was_failed = chunk.sttStatus == STTStatus.FAILED
        chunk.sttStatus = STTStatus.FAILED
        chunk.lastError = str(error)[:200]
        chunk.jobId = kwargs.get("job_id") or chunk.jobId
        chunk.failureStage = kwargs.get("failure_stage")
        chunk.failureType = kwargs.get("failure_type")
        chunk.terminal = bool(kwargs.get("terminal"))
        chunk.retryCount = kwargs.get("retry_count") or 0
        chunk.sttAttempts = int(chunk.sttAttempts or 0) + 1
        if kwargs.get("provider"):
            chunk.sttProvider = kwargs["provider"]
        if not was_failed:
            self.failed_count += 1
        return True

    async def complete_transcript_chunk(self, conversation_id, sequence_number, raw_text, language_code, request_id, provider):
        chunk = self.chunks.get(int(sequence_number))
        if chunk is None or is_terminal_failed_chunk(chunk):
            return
        chunk.sttStatus = STTStatus.COMPLETED
        chunk.rawText = raw_text
        chunk.terminal = False
        chunk.failureType = None

    async def get_conversation(self, conversation_id):
        return self.conversation

    async def list_transcript_chunks(self, conversation_id):
        return [self.chunks[key] for key in sorted(self.chunks)]

    async def list_conversation_windows(self, conversation_id):
        return []

    async def reclaim_stale_stt_chunks(self, *args, **kwargs):
        return []

    async def get_audio_chunk(self, *args, **kwargs):
        return None

    async def get_extraction_run(self, *args, **kwargs):
        return None

    async def transition(self, conversation_id, status, updates=None):
        self.conversation.status = status

    async def append_meeting_debug_trace(self, *args, **kwargs):
        return None

    async def count_unwindowed_non_empty_transcripts(self, *args, **kwargs):
        return 0


class FakeProducer:
    def __init__(self):
        self.events = []

    async def publish(self, stream, event):
        self.events.append((stream, event))
        return event.eventId


def _install_repo(monkeypatch, repo):
    monkeypatch.setattr(conversation_workers, "ConversationRepository", lambda db: repo)
    monkeypatch.setattr(conversation_workers, "get_database", lambda: object())
    monkeypatch.setattr(conversation_workers, "RedisStreamProducer", FakeProducer)


def test_s3_404_is_permanent_and_not_retryable():
    error = PermanentS3StorageError("S3 permanent error: 404")
    classification = classify_stt_failure(error)
    assert classification.permanent is True
    assert classification.failure_type == FAILURE_S3_OBJECT_MISSING


def test_corrupt_audio_error_is_permanent():
    error = STTPermanentAudioError(
        "failed to process audio: corrupt or unsupported data",
        provider="deepgram",
    )
    classification = classify_stt_failure(error)
    assert classification.permanent is True
    assert classification.failure_type == FAILURE_CORRUPT_AUDIO


def test_provider_500_is_retryable():
    error = STTProviderTemporaryError("Deepgram speech-to-text failed", provider="deepgram", status_code=500)
    classification = classify_stt_failure(error)
    assert classification.permanent is False
    assert classification.failure_type == FAILURE_PROVIDER_5XX


def test_retry_exhausted_becomes_terminal():
    error = STTProviderTemporaryError("timeout talking to provider", provider="deepgram", status_code=504)
    classification = classify_stt_failure(error, retry_exhausted=True)
    assert classification.permanent is True
    assert classification.failure_type == FAILURE_RETRY_EXHAUSTED


def test_s3_confirmed_404_marks_terminal_failed_without_retry(monkeypatch):
    repo = MemoryTranscriptRepository(chunks=[_chunk(5, status=STTStatus.PENDING, jobId="job-5")])
    _install_repo(monkeypatch, repo)
    transcribe_calls = []

    class Storage:
        async def download_file(self, bucket, object_key, destination):
            raise PermanentS3StorageError("S3 permanent error: 404")

    async def fail_transcribe(**kwargs):
        transcribe_calls.append(kwargs)
        raise AssertionError("permanent S3 404 must not reach STT")

    monkeypatch.setattr(conversation_workers, "get_s3_audio_storage", lambda: Storage())
    monkeypatch.setattr(conversation_workers, "transcribe_from_path_with_fallback", fail_transcribe)
    monkeypatch.setattr(conversation_workers, "safe_temp_audio_path", lambda job: __import__("pathlib").Path("/tmp/buddy/job-5/audio.wav"))
    monkeypatch.setattr(conversation_workers, "_cleanup_job_dir", lambda job_dir: None)
    monkeypatch.setattr(conversation_workers, "validate_conversation_audio_object_key", lambda **kwargs: kwargs["object_key"])

    event = _stt_event(
        5,
        jobId="job-5",
        storageProvider="s3",
        bucket="audio-bucket",
        objectKey="buddy/audio/user-1/space-1/conv_1/00000005-chunk.webm",
        filePath=None,
    )
    try:
        asyncio.run(conversation_workers.handle_stt_event(event))
    except NonRetryableQueueError as error:
        assert "S3_OBJECT_MISSING" in str(error)
    else:
        raise AssertionError("confirmed S3 404 must be a non-retryable failure")

    chunk = repo.chunks[5]
    assert chunk.sttStatus == STTStatus.FAILED
    assert chunk.terminal is True
    assert chunk.failureType == FAILURE_S3_OBJECT_MISSING
    assert transcribe_calls == []
    skippable, _, accounting = _session_readiness(_conversation(5), [_chunk(index, f"ok {index}") for index in range(5)] + [chunk], [])
    assert 5 in skippable
    assert accounting["permanentlyFailedSequences"] == 1
    assert accounting["retryingSequences"] == 0
    assert accounting["unresolvedExpected"] == 0


def test_corrupt_audio_marks_terminal_failed(monkeypatch):
    repo = MemoryTranscriptRepository(chunks=[_chunk(2, status=STTStatus.PENDING, jobId="job-2")])
    _install_repo(monkeypatch, repo)

    async def transcribe(**kwargs):
        raise STTPermanentAudioError("failed to process audio: corrupt or unsupported data", provider="deepgram")

    monkeypatch.setattr(conversation_workers, "transcribe_from_path_with_fallback", transcribe)

    try:
        asyncio.run(conversation_workers.handle_stt_event(_stt_event(2, jobId="job-2")))
    except NonRetryableQueueError:
        pass
    else:
        raise AssertionError("corrupt audio must be a non-retryable failure")

    chunk = repo.chunks[2]
    assert chunk.sttStatus == STTStatus.FAILED
    assert chunk.terminal is True
    assert chunk.failureType == FAILURE_CORRUPT_AUDIO
    assert is_terminal_failed_chunk(chunk) is True


def test_transient_provider_500_retries_then_succeeds(monkeypatch):
    repo = MemoryTranscriptRepository(chunks=[_chunk(1, status=STTStatus.PENDING)])
    _install_repo(monkeypatch, repo)
    calls = []

    async def transcribe(**kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise STTProviderTemporaryError("provider 500", provider="deepgram", status_code=500)
        return {"transcript": "hello there", "provider": "deepgram", "language_code": "en", "request_id": "r1"}

    monkeypatch.setattr(conversation_workers, "transcribe_from_path_with_fallback", transcribe)

    try:
        asyncio.run(conversation_workers.handle_stt_event(_stt_event(1)))
    except STTProviderTemporaryError:
        pass
    else:
        raise AssertionError("transient 500 must remain retryable")

    assert repo.chunks[1].sttStatus == STTStatus.FAILED
    assert repo.chunks[1].terminal is False
    asyncio.run(conversation_workers.handle_stt_event(_stt_event(1)))
    assert repo.chunks[1].sttStatus == STTStatus.COMPLETED
    assert repo.chunks[1].rawText == "hello there"
    assert len(calls) == 2


def test_exhausted_retryable_error_dlq_marks_terminal_failed(monkeypatch):
    repo = MemoryTranscriptRepository(chunks=[_chunk(3, status=STTStatus.PENDING, sttAttempts=5)])
    _install_repo(monkeypatch, repo)
    event = _stt_event(3, jobId="job-3", attempt=5)
    error = STTProviderTemporaryError("timeout talking to provider", provider="deepgram", status_code=504)

    asyncio.run(conversation_workers.handle_stt_dead_letter(event, error))

    chunk = repo.chunks[3]
    assert chunk.sttStatus == STTStatus.FAILED
    assert chunk.terminal is True
    assert chunk.failureType == FAILURE_RETRY_EXHAUSTED
    assert chunk.jobId == "job-3"


def test_duplicate_failure_events_are_idempotent(monkeypatch):
    repo = MemoryTranscriptRepository(chunks=[_chunk(4, status=STTStatus.PENDING, jobId="job-4")])
    _install_repo(monkeypatch, repo)

    async def transcribe(**kwargs):
        raise PermanentS3StorageError("S3 permanent error: NoSuchKey")

    monkeypatch.setattr(conversation_workers, "transcribe_from_path_with_fallback", transcribe)
    event = _stt_event(4, jobId="job-4")

    for _ in range(2):
        try:
            asyncio.run(conversation_workers.handle_stt_event(event))
        except NonRetryableQueueError:
            pass

    assert repo.failed_count == 1
    assert repo.chunks[4].sttStatus == STTStatus.FAILED
    assert repo.chunks[4].terminal is True
    assert len([call for call in repo.fail_calls if call[2].get("terminal")]) >= 1


def test_worker_restart_retains_failed_state():
    mongo_id = ObjectId()
    doc = {
        "_id": ObjectId(),
        "conversationId": mongo_id,
        "userId": ObjectId(),
        "spaceId": ObjectId(),
        "chunkId": "chunk_7",
        "sequenceNumber": 7,
        "sttStatus": STTStatus.FAILED.value,
        "terminal": True,
        "failureType": FAILURE_S3_OBJECT_MISSING,
        "failureStage": "s3_download",
        "jobId": "job-7",
        "sttAttempts": 1,
        "sttProvider": "unknown",
        "processingStatus": "unprocessed",
        "createdAt": datetime.now(timezone.utc),
        "updatedAt": datetime.now(timezone.utc),
    }
    repo = ConversationRepository(_MemoryMongo({"transcript_chunks": [doc], "conversations": [{"_id": mongo_id, "failedTranscriptChunkCount": 1}]}))
    assert asyncio.run(repo.mark_transcript_chunk_processing(str(mongo_id), 7)) is False
    chunk = asyncio.run(repo.get_transcript_chunk(str(mongo_id), 7))
    assert chunk.sttStatus == STTStatus.FAILED
    assert chunk.terminal is True
    assert asyncio.run(repo.fail_transcript_chunk(str(mongo_id), 7, "S3_OBJECT_MISSING", terminal=True, failure_type=FAILURE_S3_OBJECT_MISSING)) is False


def test_production_mix_reaches_finalization():
    chunks = []
    for sequence in range(93):
        chunks.append(_chunk(sequence, f"Useful statement number {sequence} about the launch plan."))
    for sequence in range(93, 125):
        chunks.append(_chunk(sequence, ""))
    chunks.append(_chunk(125, status=STTStatus.FAILED, terminal=True, failureType=FAILURE_S3_OBJECT_MISSING, jobId="job-125"))
    chunks.append(_chunk(126, status=STTStatus.FAILED, terminal=True, failureType=FAILURE_CORRUPT_AUDIO, jobId="job-126"))
    skippable, unresolved, accounting = _session_readiness(_conversation(126), chunks, [])
    assert unresolved is False
    assert accounting["expectedSequences"] == 127
    assert accounting["validTranscripts"] == 93
    assert accounting["emptyTranscripts"] == 32
    assert accounting["failedTranscripts"] == 2
    assert accounting["unresolvedExpected"] == 0
    assert accounting["successfulSequences"] == 93
    assert accounting["emptySequences"] == 32
    assert accounting["permanentlyFailedSequences"] == 2
    assert accounting["terminalSequences"] == 127
    assert accounting["pendingSequences"] == 0
    assert accounting["retryingSequences"] == 0
    assert 125 in skippable
    assert 126 in skippable


def test_failed_chunks_excluded_from_useful_chunks_but_accounted():
    chunks = [
        _chunk(0, "Keep this action item."),
        _chunk(1, status=STTStatus.FAILED, terminal=True, failureType=FAILURE_CORRUPT_AUDIO),
        _chunk(2, "Also keep this follow-up."),
    ]
    assembly = assemble_semantic_window_input(conversation_id="conv_1", chunks=chunks)
    _, unresolved, accounting = _session_readiness(_conversation(2), chunks, [])
    assert assembly.diagnostics["usefulChunks"] == [0, 2]
    assert 1 not in assembly.diagnostics["usefulChunks"]
    assert assembly.diagnostics["persistedSequenceNumbers"] == [0, 1, 2]
    assert accounting["validTranscripts"] == 2
    assert accounting["permanentlyFailedSequences"] == 1
    assert unresolved is False
    assert is_useful_chunk(chunks[1]) is False or chunks[1].sttStatus != STTStatus.COMPLETED


def test_one_failed_chunk_does_not_prevent_notes_from_remaining_transcript():
    chunks = [
        _chunk(0, "Rahul will open the retry ticket today."),
        _chunk(1, status=STTStatus.FAILED, terminal=True, failureType=FAILURE_S3_OBJECT_MISSING, jobId="job-1"),
        _chunk(2, "Mira will write the drain notes before Thursday."),
    ]
    assembly = assemble_semantic_window_input(conversation_id="conv_1", chunks=chunks)
    _, unresolved, accounting = _session_readiness(_conversation(2), chunks, [])
    assert unresolved is False
    assert accounting["validTranscripts"] == 2
    assert "Rahul will open the retry ticket today." in assembly.text
    assert "Mira will write the drain notes before Thursday." in assembly.text
    assert assembly.diagnostics["rejectionCounts"]["stt_failed"] == 1


def test_finalization_never_runs_while_sequence_pending(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_INCREMENTAL_MEETING_PROCESSING", False)
    conversation = _conversation(2)
    chunks = [
        _chunk(0, "hello"),
        _chunk(1, status=STTStatus.PENDING),
        _chunk(2, "world"),
    ]
    repo = MemoryTranscriptRepository(chunks=chunks, conversation=conversation)
    repo.db = SimpleNamespace(conversations=SimpleNamespace(update_one=_async_noop))
    producer = FakeProducer()
    coordinator = ConversationFinalizationCoordinator(repo, producer=producer)
    asyncio.run(coordinator.finalize("conv_1"))
    assert conversation.status != ConversationStatus.READY_FOR_PROCESSING
    assert all(event.eventType != "conversation.processing.requested" for _, event in producer.events)


def test_finalization_never_runs_while_sequence_retrying(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_INCREMENTAL_MEETING_PROCESSING", False)
    conversation = _conversation(1)
    chunks = [
        _chunk(0, "hello"),
        _chunk(1, status=STTStatus.FAILED, terminal=False, lastError="TIMEOUT", sttAttempts=1),
    ]
    repo = MemoryTranscriptRepository(chunks=chunks, conversation=conversation)
    repo.db = SimpleNamespace(conversations=SimpleNamespace(update_one=_async_noop))
    producer = FakeProducer()
    coordinator = ConversationFinalizationCoordinator(repo, producer=producer)
    asyncio.run(coordinator.finalize("conv_1"))
    _, unresolved, accounting = _session_readiness(conversation, chunks, [])
    assert unresolved is True
    assert accounting["retryingSequences"] == 1
    assert conversation.status != ConversationStatus.READY_FOR_PROCESSING


def test_dead_letter_callback_is_invoked_when_retry_budget_exhausted(monkeypatch):
    dead_letters = []

    async def on_dead_letter(event, error):
        dead_letters.append((event.payload.get("sequenceNumber"), str(error)))

    class Redis:
        def __init__(self):
            self.added = []

        async def xadd(self, stream, fields):
            self.added.append((stream, fields))
            return "1-0"

        async def xack(self, stream, group, message_id):
            return 1

    fake_redis = Redis()
    monkeypatch.setattr("services.queue.streams.redis_client", fake_redis)
    consumer = RedisStreamConsumer(
        stream=settings.REDIS_STT_STREAM,
        group=settings.REDIS_STT_GROUP,
        consumer_name="stt-test",
        handler=lambda event: None,
        max_retries=1,
        on_dead_letter=on_dead_letter,
    )
    event = _stt_event(9, jobId="job-9", attempt=1)
    asyncio.run(
        consumer._handle_failure(
            "1-0",
            {"event": event.model_dump_json()},
            STTProviderTemporaryError("timeout", provider="deepgram", status_code=504),
        )
    )
    assert dead_letters == [(9, "timeout")]
    assert fake_redis.added[0][0] == settings.REDIS_DEAD_LETTER_STREAM


async def _async_noop(*args, **kwargs):
    return None


class _MemoryMongo:
    def __init__(self, collections):
        self.transcript_chunks = _MemoryCollection(collections.get("transcript_chunks", []))
        self.conversations = _MemoryCollection(collections.get("conversations", []))


class _MemoryCollection:
    def __init__(self, docs):
        self.docs = [dict(doc) for doc in docs]

    async def find_one(self, filt):
        for doc in self.docs:
            if _matches(doc, filt):
                return dict(doc)
        return None

    async def update_one(self, filt, update):
        for doc in self.docs:
            if _matches(doc, filt):
                doc.update(update.get("$set") or {})
                for key, amount in (update.get("$inc") or {}).items():
                    doc[key] = int(doc.get(key) or 0) + int(amount)
                return SimpleNamespace(modified_count=1, upserted_id=None)
        return SimpleNamespace(modified_count=0, upserted_id=None)


def _matches(doc, filt):
    for key, expected in filt.items():
        if key == "$or":
            if not any(_matches(doc, clause) for clause in expected):
                return False
            continue
        actual = doc.get(key)
        if isinstance(expected, dict):
            if "$in" in expected:
                candidates = {str(item) for item in expected["$in"]}
                if str(actual) not in candidates:
                    return False
                continue
            if "$ne" in expected and actual == expected["$ne"]:
                return False
            if "$exists" in expected:
                exists = key in doc
                if bool(exists) is not bool(expected["$exists"]):
                    return False
            continue
        if str(actual) != str(expected):
            return False
    return True
