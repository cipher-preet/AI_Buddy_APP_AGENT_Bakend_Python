import asyncio
from types import SimpleNamespace

import pytest

from services.conversation import agents
from services.conversation.models import (
    ConversationStatus,
    ExtractionOutcome,
    ExtractionRunDocument,
    STTStatus,
    TranscriptChunkDocument,
    TranscriptProcessingStatus,
    WindowExtractionResult,
    WindowProcessingStatus,
)
from services.conversation.semantic_input import (
    SEMANTIC_INPUT_ASSEMBLY_FAILED,
    assemble_semantic_window_input,
    is_transcript_usable,
    parsed_semantic_sequences,
)
from services.conversation.workflow import ConversationProcessingWorkflow
from services.llm.router import LLMCapability
from tests.test_final_synthesis_persistence import FakeRepository, _chunks
from tests.test_zero_output_extraction import _grounded_payload, _run, _router
from apps.api_gateway.config.setting import settings


@pytest.fixture(autouse=True)
def _keep_legacy_short_session_path(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_EVENT_PIPELINE", False)
    monkeypatch.setattr(settings, "ENABLE_MEETING_PIPELINE", False)



def _chunk(sequence, text="", **kwargs):
    return TranscriptChunkDocument(
        conversationId=kwargs.get("conversationId", "conv_1"),
        userId="user_1",
        spaceId="space_1",
        chunkId=f"chunk_{sequence}",
        sequenceNumber=sequence,
        rawText=text,
        normalizedText=kwargs.get("normalizedText"),
        sttStatus=kwargs.get("sttStatus", STTStatus.COMPLETED),
        processingStatus=kwargs.get("processingStatus", TranscriptProcessingStatus.UNPROCESSED),
        processingWindowId=kwargs.get("processingWindowId"),
        publishedAt=kwargs.get("publishedAt"),
        exclusionReason=kwargs.get("exclusionReason"),
    )


def _window_doc(**kwargs):
    return SimpleNamespace(
        id=kwargs.get("id", "window_1"),
        conversationId="conv_1",
        windowIndex=kwargs.get("windowIndex", 0),
        sequenceStart=kwargs.get("sequenceStart", 1),
        sequenceEnd=kwargs.get("sequenceEnd", 2),
        text=kwargs.get("text", ""),
        isFinalPartial=kwargs.get("isFinalPartial", True),
        extractionSkipped=kwargs.get("extractionSkipped", True),
        checkpointKind=kwargs.get("checkpointKind", "raw_final"),
        status=kwargs.get("status", WindowProcessingStatus.COMPLETED),
        result=kwargs.get("result"),
        nonEmptyChunkCount=kwargs.get("nonEmptyChunkCount", 0),
    )


def test_basic_inclusion_all_non_empty_reach_useful_chunks():
    chunks = [
        _chunk(0, "hello"),
        _chunk(1, "we tested backend"),
        _chunk(2, "fix duplicate task tomorrow"),
    ]
    assembly = assemble_semantic_window_input(conversation_id="conv_1", chunks=chunks)
    assert assembly.diagnostics["usefulChunks"] == [0, 1, 2]
    assert assembly.diagnostics["usefulSequenceNumbers"] == [0, 1, 2]


def test_empty_removal_keeps_sequence_accounting():
    chunks = [_chunk(0, "hello"), _chunk(1, ""), _chunk(2, "test tomorrow")]
    assembly = assemble_semantic_window_input(conversation_id="conv_1", chunks=chunks)
    assert assembly.diagnostics["usefulChunks"] == [0, 2]
    assert assembly.diagnostics["persistedSequenceNumbers"] == [0, 1, 2]
    assert assembly.diagnostics["emptyTranscriptCount"] == 1
    assert assembly.diagnostics["rejectionCounts"]["empty_text"] == 1


def test_semantic_neutrality_includes_casual_and_vague_speech():
    chunks = [
        _chunk(0, "Hello, what is the current status?"),
        _chunk(1, "We tested it yesterday."),
        _chunk(2, "Maybe we could do something later."),
        _chunk(3, "We need to fix duplicate tasks tomorrow."),
        _chunk(4, "Raw transcript remains the source of truth."),
    ]
    assembly = assemble_semantic_window_input(conversation_id="conv_1", chunks=chunks)
    assert assembly.diagnostics["usefulChunks"] == [0, 1, 2, 3, 4]
    assert all(is_transcript_usable(chunk) for chunk in chunks)
    reasons = assembly.diagnostics["rejectionCounts"]
    assert reasons["empty_text"] == 0
    assert "not_actionable" not in reasons
    assert "low_value" not in reasons
    assert "not_important" not in reasons


def test_ordering_is_deterministic_ascending():
    chunks = [
        _chunk(5, "five"),
        _chunk(2, "two"),
        _chunk(4, "four"),
        _chunk(1, "one"),
        _chunk(3, "three"),
    ]
    assembly = assemble_semantic_window_input(conversation_id="conv_1", chunks=chunks)
    assert assembly.diagnostics["usefulChunks"] == [1, 2, 3, 4, 5]


def test_string_int_sequence_mismatch_does_not_exclude():
    chunks = [_chunk("5", "Stored as string five."), _chunk(6, "Integer six.")]
    assembly = assemble_semantic_window_input(
        conversation_id="conv_1",
        chunks=chunks,
        sequence_start=5,
        sequence_end=6,
        mode="window_range",
    )
    assert assembly.diagnostics["usefulChunks"] == [5, 6]


def test_final_short_session_six_chunks_all_reach_llm():
    chunks = [_chunk(i, f"speech turn {i}") for i in range(6)]
    assembly = assemble_semantic_window_input(conversation_id="conv_1", chunks=chunks, mode="final_raw")
    assert assembly.diagnostics["usefulChunks"] == [0, 1, 2, 3, 4, 5]
    assert assembly.diagnostics["usefulTranscriptCount"] == 6


def test_long_session_raw_remainder_excludes_checkpointed_history():
    chunks = [
        _chunk(1, "Historical checkpoint speech.", processingWindowId="hist", processingStatus=TranscriptProcessingStatus.PROCESSED),
        _chunk(2, "More history.", processingWindowId="hist", processingStatus=TranscriptProcessingStatus.PROCESSED),
        _chunk(3, "Current raw remainder after STOP."),
    ]
    history = _window_doc(
        id="hist",
        windowIndex=0,
        sequenceStart=1,
        sequenceEnd=2,
        text="[1] Historical checkpoint speech.\n[2] More history.",
        isFinalPartial=False,
        extractionSkipped=False,
        checkpointKind="semantic_checkpoint",
        result=WindowExtractionResult(isCheckpoint=True, semanticUnits=[], summary="history"),
    )
    remainder = _window_doc(
        id="raw",
        windowIndex=1,
        sequenceStart=3,
        sequenceEnd=3,
        text="[3] Current raw remainder after STOP.",
        isFinalPartial=True,
        extractionSkipped=True,
        checkpointKind="raw_final",
    )
    assembly = assemble_semantic_window_input(
        conversation_id="conv_1",
        chunks=chunks,
        windows=[history, remainder],
        mode="leftover",
    )
    assert assembly.diagnostics["rejectionCounts"]["outside_window_range"] == 2
    assert assembly.diagnostics["usefulChunks"] == [3]
    assert "remainder" in assembly.text
    assert "Historical checkpoint" not in assembly.text


def test_retry_reconstructs_identical_useful_chunks():
    first = [_chunk(1, "First."), _chunk(2, "Second."), _chunk(3, "Third.")]
    retried = [
        _chunk(1, "First.", processingStatus=TranscriptProcessingStatus.PROCESSED, processingWindowId="w1"),
        _chunk(2, "Second.", processingStatus=TranscriptProcessingStatus.PROCESSED, processingWindowId="w1"),
        _chunk(3, "Third.", processingStatus=TranscriptProcessingStatus.PROCESSED, processingWindowId="w1"),
    ]
    window = _window_doc(sequenceStart=1, sequenceEnd=3, text="[1] First.\n[2] Second.\n[3] Third.")
    first_assembly = assemble_semantic_window_input(conversation_id="conv_1", chunks=first, windows=[window])
    retry_assembly = assemble_semantic_window_input(conversation_id="conv_1", chunks=retried, windows=[window])
    assert first_assembly.diagnostics["usefulChunks"] == retry_assembly.diagnostics["usefulChunks"] == [1, 2, 3]


def test_invalid_integrity_state_is_assembly_failed_not_valid_empty():
    window = SimpleNamespace(
        conversationId="conv_1",
        userId="user_1",
        spaceId="space_1",
        id="broken",
        windowIndex=0,
        sequenceStart=1,
        sequenceEnd=2,
        text="",
        nonEmptyChunkCount=2,
        semanticInputDiagnostics={
            "persistedNonEmptyTranscriptCount": 2,
            "usefulSequenceNumbers": [],
            "usefulChunks": [],
            "semanticInputTranscriptCount": 0,
            "semanticInputAssemblyFailed": True,
        },
    )

    class _Router:
        def route(self, capability: LLMCapability):
            return SimpleNamespace(name="unused"), "unused"

    result, _, _ = asyncio.run(agents.extract_window(_Router(), window, context={}, meeting_context={}, mode="final"))
    assert result.extractionOutcome == ExtractionOutcome.SEMANTIC_INPUT_ASSEMBLY_FAILED
    assert result.extractionOutcome != ExtractionOutcome.VALID_EMPTY_EXTRACTION


def test_true_silence_is_legitimate_zero_useful_chunks():
    chunks = [_chunk(1, ""), _chunk(2, "   ")]
    assembly = assemble_semantic_window_input(conversation_id="conv_1", chunks=chunks)
    assert assembly.failed is False
    assert assembly.diagnostics["usefulChunks"] == []
    assert assembly.diagnostics["emptyTranscriptCount"] == 2

    def handler(route, schema):
        name = getattr(schema, "__name__", "")
        if name == "WindowExtractionLLMResponse":
            return schema(semanticUnits=[], supportedUnitVerdict="no_supported_units")
        return schema()

    result, _, _ = _run(_router(handler), text="")
    assert result.extractionOutcome == ExtractionOutcome.VALID_EMPTY_EXTRACTION


def test_stt_failed_and_damaged_are_technical_exclusions():
    chunks = [
        _chunk(0, "keep me"),
        _chunk(1, "failed forever", sttStatus=STTStatus.FAILED),
        _chunk(2, "", exclusionReason="stt_failed"),
        _chunk(3, "also keep"),
    ]
    # empty+exclusion without text → damaged/empty path
    chunks[2] = _chunk(2, "", exclusionReason="corrupted_audio", sttStatus=STTStatus.COMPLETED)
    assembly = assemble_semantic_window_input(conversation_id="conv_1", chunks=chunks)
    assert assembly.diagnostics["usefulChunks"] == [0, 3]
    assert assembly.diagnostics["rejectionCounts"]["stt_failed"] == 1
    assert assembly.diagnostics["rejectionCounts"]["damaged"] == 1


def test_unpublished_current_session_chunks_are_not_excluded():
    chunks = [
        _chunk(1, "Unpublished current session speech.", publishedAt=None),
        _chunk(2, "Still unpublished and useful.", publishedAt=None),
    ]
    assembly = assemble_semantic_window_input(conversation_id="conv_1", chunks=chunks)
    assert assembly.diagnostics["unpublishedFilterApplied"] is False
    assert "unpublished_filter" not in assembly.diagnostics["rejectionCounts"]
    assert assembly.diagnostics["usefulChunks"] == [1, 2]


def test_duplicate_sequence_is_rejected_once():
    chunks = [_chunk(1, "first copy"), _chunk(1, "duplicate copy"), _chunk(2, "ok")]
    assembly = assemble_semantic_window_input(conversation_id="conv_1", chunks=chunks)
    assert assembly.diagnostics["usefulChunks"] == [1, 2]
    assert assembly.diagnostics["duplicateTranscriptCount"] == 1


def test_empty_durable_window_text_is_rebuilt_from_persisted_chunks():
    chunks = [_chunk(1, "Persisted speech survived STOP."), _chunk(2, "Second persisted line.")]
    empty_window = _window_doc(sequenceStart=1, sequenceEnd=2, text="", nonEmptyChunkCount=2)
    assembly = assemble_semantic_window_input(conversation_id="conv_1", chunks=chunks, windows=[empty_window])
    assert assembly.failed is False
    assert assembly.diagnostics["usefulChunks"] == [1, 2]
    assert "Persisted speech" in assembly.text


def test_short_session_assembly_reaches_extraction_and_persists():
    def handler(route, schema):
        name = getattr(schema, "__name__", "")
        if name == "WindowExtractionLLMResponse":
            return schema(**_grounded_payload())
        if name == "FinalSynthesisLLMResponse":
            payload = _grounded_payload()
            return schema(summary=payload["summary"], tasks=payload["tasks"], notes=payload["notes"], publishVerdict="PUBLISH")
        if name == "MemoryUpdateResponse":
            return schema(currentSummary="updated")
        return schema()

    repo = FakeRepository(_chunks())
    workflow = ConversationProcessingWorkflow(repo, _router(handler))
    run = ExtractionRunDocument(
        conversationId=repo.conversation.id,
        userId=repo.conversation.userId,
        spaceId=repo.conversation.spaceId,
        processingVersion=1,
        provider="test",
        model="test",
    )
    asyncio.run(workflow._run_short_session_finalization(repo.conversation, run, []))
    diagnostics = run.checkpoints["short_raw_transcript"]
    assert diagnostics["usefulSequenceNumbers"]
    assert diagnostics["semanticInputTranscriptCount"] >= 1
    assert diagnostics["persistenceOutcome"] == "PERSISTED"
    assert repo.conversation.status == ConversationStatus.COMPLETED


def test_assembly_failure_raises_for_queue_retry():
    class _EmptyRepo(FakeRepository):
        async def list_transcript_chunks(self, conversation_id):
            return [_chunk(1, "Non-empty persisted speech.", conversationId=conversation_id)]

    repo = _EmptyRepo([])
    workflow = ConversationProcessingWorkflow(repo, _router(lambda route, schema: schema()))
    window = SimpleNamespace(
        conversationId=repo.conversation.id,
        userId=repo.conversation.userId,
        spaceId=repo.conversation.spaceId,
        id="broken",
        windowIndex=0,
        sequenceStart=1,
        sequenceEnd=1,
        text="",
        nonEmptyChunkCount=1,
        semanticInputDiagnostics={"persistedNonEmptyTranscriptCount": 1, "semanticInputAssemblyFailed": True},
    )
    result, _, _ = asyncio.run(agents.extract_window(_router(lambda route, schema: schema()), window, {}, {}, mode="final"))
    assert result.extractionOutcome == ExtractionOutcome.SEMANTIC_INPUT_ASSEMBLY_FAILED
    run = ExtractionRunDocument(
        conversationId=repo.conversation.id,
        userId=repo.conversation.userId,
        spaceId=repo.conversation.spaceId,
        processingVersion=1,
        provider="test",
        model="test",
    )
    original = agents.extract_from_raw_transcript

    async def _failing_extract(*args, **kwargs):
        return result, "none", "none"

    agents.extract_from_raw_transcript = _failing_extract
    try:
        with pytest.raises(RuntimeError, match=SEMANTIC_INPUT_ASSEMBLY_FAILED):
            asyncio.run(workflow._run_short_session_finalization(repo.conversation, run, []))
    finally:
        agents.extract_from_raw_transcript = original
    assert repo.conversation.status != ConversationStatus.COMPLETED
