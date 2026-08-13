from services.conversation.models import ConversationDocument, TranscriptChunkDocument
from services.conversation.windowing import build_ready_windows


def test_windowing_uses_only_contiguous_chunk_prefix(monkeypatch):
    monkeypatch.setattr("services.conversation.windowing.settings.INCREMENTAL_WINDOW_TARGET_TOKENS", 4)
    conversation = ConversationDocument(_id="conv_1", userId="user_1", spaceId="space_1")
    chunks = [
        TranscriptChunkDocument(
            conversationId="conv_1",
            userId="user_1",
            spaceId="space_1",
            chunkId="chunk_0",
            sequenceNumber=0,
            rawText="first chunk",
        ),
        TranscriptChunkDocument(
            conversationId="conv_1",
            userId="user_1",
            spaceId="space_1",
            chunkId="chunk_2",
            sequenceNumber=2,
            rawText="third chunk arrived before second",
        )
    ]

    windows = build_ready_windows(conversation, chunks, start_index=0)

    assert len(windows) == 1
    assert windows[0].window.sequenceStart == 0
    assert windows[0].window.sequenceEnd == 0


def test_windowing_closes_final_partial_window():
    conversation = ConversationDocument(_id="conv_1", userId="user_1", spaceId="space_1")
    chunks = [
        TranscriptChunkDocument(
            conversationId="conv_1",
            userId="user_1",
            spaceId="space_1",
            chunkId="chunk_0",
            sequenceNumber=0,
            rawText="short final meeting chunk",
        )
    ]

    windows = build_ready_windows(conversation, chunks, start_index=0, force_final=True)

    assert len(windows) == 1
    assert windows[0].window.isFinalPartial is True
    assert windows[0].sequence_numbers == [0]
