from services.conversation.fingerprints import task_fingerprint
from services.conversation.models import (
    ConversationStatus,
    EvidenceSpan,
    ExtractedTask,
    TranscriptChunkDocument,
    assert_valid_transition,
)
from services.conversation.transcript import assemble_transcript, detect_missing_sequences, segment_transcript
from apps.api_gateway.main import app


def test_invalid_state_transition_rejected():
    try:
        assert_valid_transition(ConversationStatus.COMPLETED, ConversationStatus.PROCESSING)
    except ValueError:
        return
    raise AssertionError("completed conversations must not be reprocessed without explicit retry state")


def test_sequence_gap_detection():
    assert detect_missing_sequences([0, 2, 4], 4) == [1, 3]


def test_transcript_ordering_and_overlap_normalization():
    chunks = [
        TranscriptChunkDocument(
            conversationId="conv_1",
            userId="user_1",
            spaceId="space_1",
            chunkId="b",
            sequenceNumber=1,
            rawText="callback tomorrow. Also update notes.",
        ),
        TranscriptChunkDocument(
            conversationId="conv_1",
            userId="user_1",
            spaceId="space_1",
            chunkId="a",
            sequenceNumber=0,
            rawText="Rahul will test the payment callback tomorrow.",
        ),
    ]
    assembled = assemble_transcript(chunks)
    assert assembled.raw_transcript.startswith("[0]")
    assert "[1] Also update notes." in assembled.normalized_transcript


def test_segmenter_preserves_sequence_ranges():
    chunks = [
        TranscriptChunkDocument(
            conversationId="conv_1",
            userId="user_1",
            spaceId="space_1",
            chunkId=str(index),
            sequenceNumber=index,
            rawText=f"Sentence {index}. Another sentence {index}.",
        )
        for index in range(5)
    ]
    segments = segment_transcript("conv_1", chunks, target_tokens=8, overlap_ratio=0.1, max_segments=10)
    assert len(segments) > 1
    assert segments[0].sequenceStart == 0
    assert segments[-1].sequenceEnd == 4


def test_task_fingerprint_is_stable():
    task = ExtractedTask(
        title="Complete payment callback testing",
        operation="CREATE",
        ownerText="Rahul",
        dueDateResolved="2026-07-29",
        confidence=0.94,
        sourceConversationId="conv_1",
        evidence=[EvidenceSpan(sequenceStart=18, sequenceEnd=20, text="Rahul will test it tomorrow.")],
    )
    assert task_fingerprint("space_1", task) == task_fingerprint("space_1", task)


def test_legacy_speech_listening_end_route_is_registered():
    paths = {route.path for route in app.routes}
    assert "/api/v1/speech/listening/start" in paths
    assert "/api/v1/speech/listening/end" in paths
