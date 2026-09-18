"""Fixture-based meeting-extension path into the existing intelligence pipeline.

This is not a live STT/LLM test. It verifies that meeting_extension transcript
chunks are the same documents the meeting pipeline already consumes.
"""

from services.conversation.meeting_pipeline.flags import meeting_pipeline_enabled
from services.conversation.models import TranscriptChunkDocument
from services.meeting_extension import MEETING_SOURCE_TYPE


def test_meeting_pipeline_flag_still_default_on():
    assert meeting_pipeline_enabled() is True


def test_extension_transcript_chunk_is_compatible_with_pipeline_schema():
    chunk = TranscriptChunkDocument(
        conversationId="507f1f77bcf86cd799439011",
        userId="507f1f77bcf86cd799439012",
        spaceId="507f1f77bcf86cd799439013",
        chunkId="meeting:507f1f77bcf86cd799439011:chunk:1",
        sequenceNumber=1,
        rawText="We will ship the API tomorrow.",
        startTimeMs=1200,
        endTimeMs=6800,
        sourceType=MEETING_SOURCE_TYPE,
        source=MEETING_SOURCE_TYPE,
        meetingSessionId="507f1f77bcf86cd799439011",
        segments=[
            {
                "id": "seg-1",
                "text": "We will ship the API tomorrow.",
                "startOffsetMs": 1200,
                "endOffsetMs": 6800,
                "speakerId": None,
            }
        ],
    )
    assert chunk.sourceType == "meeting_extension"
    assert chunk.rawText
    assert chunk.sequenceNumber == 1
