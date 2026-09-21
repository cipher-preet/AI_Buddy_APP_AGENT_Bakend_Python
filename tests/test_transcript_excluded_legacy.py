from datetime import datetime, timezone

from services.conversation.finalization import _session_readiness
from services.conversation.models import (
    ConversationDocument,
    ConversationStatus,
    STTStatus,
    TranscriptChunkDocument,
    TranscriptProcessingStatus,
)
from services.conversation.stt_failure import FAILURE_MISSING_UPLOAD, is_terminal_failed_chunk


def _excluded_mongo_doc(**overrides):
    base = {
        "_id": "chunk_excluded_1",
        "conversationId": "6ab0e8750c0e8fe47ae8ef81",
        "userId": "user_1",
        "spaceId": "space_1",
        "chunkId": "chunk:excluded:1",
        "sequenceNumber": 1,
        "rawText": "",
        "normalizedText": "",
        "sttStatus": "completed",
        "processingStatus": "excluded",
        "exclusionReason": "empty_transcript",
        "createdAt": datetime.now(timezone.utc),
        "updatedAt": datetime.now(timezone.utc),
    }
    base.update(overrides)
    return base


def test_legacy_excluded_processing_status_validates_as_processed():
    chunk = TranscriptChunkDocument.model_validate(_excluded_mongo_doc())
    assert chunk.processingStatus == TranscriptProcessingStatus.PROCESSED
    assert chunk.exclusionReason == "empty_transcript"


def test_legacy_excluded_without_reason_gets_default_exclusion():
    chunk = TranscriptChunkDocument.model_validate(
        _excluded_mongo_doc(exclusionReason=None),
    )
    assert chunk.processingStatus == TranscriptProcessingStatus.PROCESSED
    assert chunk.exclusionReason == "empty_transcript"


def test_missing_upload_chunk_is_terminal_and_skippable():
    chunk = TranscriptChunkDocument(
        conversationId="6ab1185658a050e7fca08d86",
        userId="user_1",
        spaceId=None,
        chunkId="meeting:6ab1185658a050e7fca08d86:missing:1",
        sequenceNumber=1,
        rawText=None,
        normalizedText=None,
        sttProvider="none",
        sttStatus=STTStatus.FAILED,
        processingStatus=TranscriptProcessingStatus.PROCESSED,
        lastError="Chunk never uploaded before upload wait timeout.",
        failureStage="upload",
        failureType=FAILURE_MISSING_UPLOAD,
        terminal=True,
        exclusionReason="sequence_missing",
        sourceType="meeting_extension",
    )
    assert is_terminal_failed_chunk(chunk) is True

    conversation = ConversationDocument(
        userId="user_1",
        spaceId=None,
        status=ConversationStatus.FINALIZING,
        expectedLastSequence=3,
        sourceType="meeting_extension",
    )
    other = TranscriptChunkDocument(
        conversationId=conversation.id,
        userId="user_1",
        spaceId=None,
        chunkId="ok-2",
        sequenceNumber=2,
        rawText="hello",
        normalizedText="hello",
        sttStatus=STTStatus.COMPLETED,
        processingStatus=TranscriptProcessingStatus.PROCESSED,
        sourceType="meeting_extension",
    )
    other3 = other.model_copy(update={"sequenceNumber": 3, "chunkId": "ok-3"})
    skippable, unresolved, accounting = _session_readiness(
        conversation,
        [chunk, other, other3],
        [],
    )
    assert 1 in skippable
    assert unresolved is False
    assert accounting["failedTranscripts"] >= 1
