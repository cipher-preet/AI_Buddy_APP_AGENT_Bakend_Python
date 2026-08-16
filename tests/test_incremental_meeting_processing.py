from services.conversation.agents import (
    _merge_window_extraction_results,
    _needs_final_memory_recovery,
    _needs_window_recovery,
    _preserve_window_candidates_when_final_empty,
    _safe_task_from_payload,
)
from services.conversation.models import (
    ConversationDocument,
    EvidenceSpan,
    ExtractedIssue,
    ExtractedNote,
    ExtractedTask,
    TranscriptChunkDocument,
    WindowExtractionResult,
)
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


def test_empty_substantial_window_uses_llm_recovery_pass():
    result = WindowExtractionResult()
    text = " ".join(f"word{i}" for i in range(45))

    assert _needs_window_recovery(result, text) is True


def test_facts_without_memory_objects_still_use_llm_recovery_pass():
    result = WindowExtractionResult(importantFacts=["The model already found durable context."])
    text = " ".join(f"word{i}" for i in range(45))

    assert _needs_window_recovery(result, text) is True


def test_note_source_with_only_issue_still_uses_llm_recovery_pass():
    result = WindowExtractionResult(
        summary="The window has durable context but the first pass only returned an issue.",
        importantFacts=["The first pass found note-like context but did not create a note."],
        issues=[
            ExtractedIssue(
                title="Unresolved context",
                kind="open_question",
                confidence=0.6,
                sourceConversationId="conv_1",
                evidence=[EvidenceSpan(sequenceStart=0, sequenceEnd=0, text="Unresolved context")],
            )
        ],
    )
    text = " ".join(f"word{i}" for i in range(45))

    assert _needs_window_recovery(result, text) is True


def test_recovery_merge_deduplicates_generic_values():
    primary = WindowExtractionResult(topics=["planning"], importantFacts=[])
    recovery = WindowExtractionResult(topics=["Planning", "next steps"], importantFacts=["Remember the follow-up context."])

    merged = _merge_window_extraction_results(primary, recovery)

    assert merged.topics == ["planning", "next steps"]
    assert merged.importantFacts == ["Remember the follow-up context."]


def test_finalizer_preserves_window_notes_when_final_merge_drops_them():
    note = ExtractedNote(
        title="Durable note from window",
        body="The window extractor created a useful note that should survive finalization.",
        confidence=0.82,
        sourceConversationId="conv_1",
        evidence=[
            EvidenceSpan(
                sequenceStart=6,
                sequenceEnd=6,
                text="The window extractor created a useful note that should survive finalization.",
            )
        ],
    )
    window_payload = [
        {
            "windowIndex": 0,
            "sequenceStart": 0,
            "sequenceEnd": 6,
            "notes": [note.model_dump()],
            "tasks": [],
            "decisions": [],
            "issues": [],
        }
    ]

    finalized = _preserve_window_candidates_when_final_empty(
        WindowExtractionResult(),
        window_payload,
        "conv_1",
        "space_1",
    )

    assert len(finalized.notes) == 1
    assert finalized.notes[0].title == "Durable note from window"


def test_final_recovery_runs_when_finalizer_returns_no_memory_but_windows_have_facts():
    finalized = WindowExtractionResult(summary="The conversation had useful context.")
    window_payload = [
        {
            "windowIndex": 0,
            "summary": "The window includes durable learning that may be useful later.",
            "importantFacts": ["A durable fact was extracted by the window pass."],
            "tasks": [],
            "notes": [],
            "decisions": [],
            "issues": [],
        }
    ]

    assert _needs_final_memory_recovery(finalized, window_payload) is True


def test_final_recovery_runs_when_task_exists_but_notes_are_missing_from_facts():
    finalized = WindowExtractionResult(
        tasks=[
            ExtractedTask(
                title="Confirm follow up",
                body="A follow-up may be needed.",
                operation="NEEDS_CONFIRMATION",
                confidence=0.6,
                needsConfirmation=True,
                sourceConversationId="conv_1",
                evidence=[EvidenceSpan(sequenceStart=0, sequenceEnd=0, text="A follow-up may be needed.")],
            )
        ]
    )
    window_payload = [
        {
            "importantFacts": ["The window pass found note-like context that should not be hidden by a task."],
            "tasks": [],
            "notes": [],
            "decisions": [],
            "issues": [],
        }
    ]

    assert _needs_final_memory_recovery(finalized, window_payload) is True


def test_final_recovery_skips_when_finalizer_already_has_memory_objects():
    finalized = WindowExtractionResult(
        notes=[
            ExtractedNote(
                title="Existing note",
                body="The finalizer already produced a stored memory object.",
                confidence=0.8,
                sourceConversationId="conv_1",
                evidence=[
                    EvidenceSpan(
                        sequenceStart=0,
                        sequenceEnd=0,
                        text="The finalizer already produced a stored memory object.",
                    )
                ],
            )
        ]
    )
    window_payload = [{"importantFacts": ["A durable fact was extracted by the window pass."]}]

    assert _needs_final_memory_recovery(finalized, window_payload) is False


def test_no_action_task_payload_is_not_publishable():
    task = _safe_task_from_payload(
        {
            "title": "No action should be stored",
            "body": "This should not become a pending task.",
            "operation": "NO_ACTION",
            "confidence": 0.9,
            "evidence": [
                {
                    "sequenceStart": 0,
                    "sequenceEnd": 0,
                    "text": "This should not become a pending task.",
                }
            ],
        },
        "conv_1",
        "space_1",
    )

    assert task is None
