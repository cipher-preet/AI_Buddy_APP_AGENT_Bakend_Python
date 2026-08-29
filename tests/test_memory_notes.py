"""Independent Task/Note eligibility, memory coverage, note identity, and generic domains."""

from __future__ import annotations

import asyncio

from services.conversation.event_pipeline.coverage import apply_memory_coverage, unpublished_memory_events
from services.conversation.event_pipeline.embeddings import CachedEmbedder, LexicalEmbedder
from services.conversation.event_pipeline.events import ScriptedEventExtractor
from services.conversation.event_pipeline.memory_identity import memory_relation
from services.conversation.event_pipeline.pipeline import run_event_pipeline
from services.conversation.event_pipeline.schemas import (
    ActionSignal,
    AtomicEvent,
    CoverageLedger,
    EventKind,
    MemoryDisposition,
    MemorySignal,
)
from services.conversation.models import EvidenceSpan, STTStatus, TranscriptChunkDocument
from tests.fixtures.generic_conversations import all_generic_conversations
from tests.fixtures.reviewed_meetings import build_meeting_c


def _embedder() -> CachedEmbedder:
    return CachedEmbedder(LexicalEmbedder())


def _chunk(sequence: int, text: str) -> TranscriptChunkDocument:
    return TranscriptChunkDocument(
        conversationId="conv",
        userId="u",
        spaceId="s",
        chunkId=f"chunk_{sequence}",
        sequenceNumber=sequence,
        rawText=text,
        sttStatus=STTStatus.COMPLETED,
    )


def _event(event_id: str, kind: EventKind, meaning: str, sequence: int, text: str, **kwargs) -> AtomicEvent:
    return AtomicEvent(
        eventId=event_id,
        topicId=kwargs.get("topicId", "T1"),
        kind=kind,
        meaning=meaning,
        object=kwargs.get("object"),
        entities=kwargs.get("entities") or [],
        evidence=[EvidenceSpan(sequenceStart=sequence, sequenceEnd=sequence, text=text)],
        sequenceIds=[sequence],
        sourceIds=[f"chunk_{sequence}"],
        conversationId="conv",
        userId="u",
        spaceId="s",
        actionSignal=kwargs.get("actionSignal"),
        memorySignal=kwargs.get("memorySignal"),
        threadId=kwargs.get("threadId"),
    )


def _blob(items) -> str:
    return " ".join(f"{item.title} {item.body}" for item in items).casefold()


def test_task_does_not_suppress_related_note_master_prompt():
    text = "The master prompt should contain elevation, frontend view, images and appearance. Please document these requirements."
    events = [
        _event(
            "e-req",
            EventKind.REQUIREMENT,
            "The master prompt should include elevation, frontend view, images and appearance.",
            0,
            text,
            object="master prompt",
            entities=["master"],
            memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH", reason="REQUIREMENT"),
        ),
        _event(
            "e-doc",
            EventKind.REQUEST,
            "Document the master-prompt requirements.",
            0,
            text,
            object="master-prompt requirements",
            entities=["master"],
            actionSignal=ActionSignal(
                isActionable=True,
                role="REQUEST",
                actionStrength="EXPLICIT",
                verb="document",
                object="master-prompt requirements",
                canonicalActionObject="master-prompt requirements",
                objectGroundingType="EXPLICIT",
            ),
        ),
    ]
    result = asyncio.run(
        run_event_pipeline(
            [_chunk(0, text)],
            "conv",
            "u",
            "s",
            event_extractor=ScriptedEventExtractor(events=events),
            embedder=_embedder(),
        )
    )
    assert result.tasks
    assert result.notes
    assert "document" in _blob(result.tasks) or "requirement" in _blob(result.tasks)
    assert any(token in _blob(result.notes) for token in ("elevation", "appearance", "master"))


def test_old_keys_note_and_replace_task_both_publish():
    events = [
        _event(
            "e-keys",
            EventKind.STATE,
            "Old keys are currently in use.",
            0,
            "Old keys are currently in use.",
            object="old keys",
            entities=["keys"],
            memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH", reason="STATUS"),
        ),
        _event(
            "e-replace",
            EventKind.REQUEST,
            "Replace the old keys Monday.",
            1,
            "We need to replace the old keys Monday.",
            object="old keys",
            entities=["keys"],
            actionSignal=ActionSignal(
                isActionable=True,
                role="REQUEST",
                actionStrength="EXPLICIT",
                verb="replace",
                object="old keys",
                canonicalActionObject="old keys",
                objectGroundingType="EXPLICIT",
                deadline="Monday",
            ),
        ),
    ]
    result = asyncio.run(
        run_event_pipeline(
            [_chunk(0, "Old keys are currently in use."), _chunk(1, "We need to replace the old keys Monday.")],
            "conv",
            "u",
            "s",
            event_extractor=ScriptedEventExtractor(events=events),
            embedder=_embedder(),
        )
    )
    assert result.notes and result.tasks
    assert "key" in _blob(result.notes)


def test_calendar_move_publishes_note_and_task():
    events = [
        _event(
            "e-move",
            EventKind.DECISION,
            "Meeting moved to Monday.",
            0,
            "We decided to move the meeting to Monday.",
            object="meeting",
            entities=["meeting"],
            memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH", reason="DECISION"),
        ),
        _event(
            "e-cal",
            EventKind.REQUEST,
            "Update the calendar.",
            1,
            "Please update the calendar.",
            object="calendar",
            entities=["calendar"],
            actionSignal=ActionSignal(
                isActionable=True,
                role="REQUEST",
                actionStrength="EXPLICIT",
                verb="update",
                object="calendar",
                canonicalActionObject="calendar",
                objectGroundingType="EXPLICIT",
            ),
        ),
    ]
    result = asyncio.run(
        run_event_pipeline(
            [_chunk(0, "We decided to move the meeting to Monday."), _chunk(1, "Please update the calendar.")],
            "conv",
            "u",
            "s",
            event_extractor=ScriptedEventExtractor(events=events),
            embedder=_embedder(),
        )
    )
    assert "monday" in _blob(result.notes) or "meeting" in _blob(result.notes)
    assert "calendar" in _blob(result.tasks)


def test_study_status_does_not_invent_task():
    events = [
        _event(
            "e-done",
            EventKind.RESULT,
            "Arrays revision is complete.",
            0,
            "Arrays revision is complete.",
            object="arrays",
            memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH", reason="RESULT"),
        ),
        _event(
            "e-pending",
            EventKind.STATE,
            "Graphs still need revision.",
            1,
            "Graphs still need revision.",
            object="graphs",
            memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH", reason="STATUS"),
        ),
    ]
    result = asyncio.run(
        run_event_pipeline(
            [_chunk(0, "Arrays revision is complete."), _chunk(1, "Graphs still need revision.")],
            "conv",
            "u",
            "s",
            event_extractor=ScriptedEventExtractor(events=events),
            embedder=_embedder(),
        )
    )
    assert result.notes
    assert result.tasks == []


def test_note_identity_paraphrase_vs_updates():
    failing = _event("e1", EventKind.ISSUE, "S3 is failing.", 1, "S3 is failing.", object="S3", entities=["S3"], threadId="t1")
    config = _event("e2", EventKind.FACT, "S3 configuration changed.", 2, "S3 configuration changed.", object="S3", entities=["S3"], threadId="t1")
    still = _event(
        "e3",
        EventKind.STATE,
        "S3 is still failing after the change.",
        3,
        "S3 is still failing after the change.",
        object="S3",
        entities=["S3"],
        threadId="t1",
    )
    insecure_a = _event("e4", EventKind.STATE, "Connection is insecure.", 4, "Connection is insecure.", object="connection", threadId="t2")
    insecure_b = _event(
        "e5",
        EventKind.STATE,
        "Current connection was reported insecure.",
        5,
        "Current connection was reported insecure.",
        object="connection",
        threadId="t2",
    )
    assert memory_relation(config, failing) in {"UPDATE", "DISTINCT"}
    assert memory_relation(still, failing) == "UPDATE"
    assert memory_relation(insecure_b, insecure_a) == "DUPLICATE"


def test_s3_status_updates_publish_separately():
    events = [
        _event("e1", EventKind.ISSUE, "S3 is failing.", 0, "S3 is failing.", object="S3", entities=["S3"], memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH")),
        _event("e2", EventKind.FACT, "S3 configuration changed.", 1, "S3 configuration changed.", object="S3", entities=["S3"], memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH")),
        _event(
            "e3",
            EventKind.STATE,
            "S3 is still failing after the change.",
            2,
            "S3 is still failing after the change.",
            object="S3",
            entities=["S3"],
            memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH"),
        ),
    ]
    result = asyncio.run(
        run_event_pipeline(
            [_chunk(0, "S3 is failing."), _chunk(1, "S3 configuration changed."), _chunk(2, "S3 is still failing after the change.")],
            "conv",
            "u",
            "s",
            event_extractor=ScriptedEventExtractor(events=events),
            embedder=_embedder(),
        )
    )
    assert len(result.notes) >= 3


def test_insecure_paraphrase_merges():
    events = [
        _event("e1", EventKind.STATE, "Connection is insecure.", 0, "Connection is insecure.", object="connection", memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH")),
        _event(
            "e2",
            EventKind.STATE,
            "Current connection was reported insecure.",
            1,
            "Current connection was reported insecure.",
            object="connection",
            memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH"),
        ),
    ]
    result = asyncio.run(
        run_event_pipeline(
            [_chunk(0, "Connection is insecure."), _chunk(1, "Current connection was reported insecure.")],
            "conv",
            "u",
            "s",
            event_extractor=ScriptedEventExtractor(events=events),
            embedder=_embedder(),
        )
    )
    assert len(result.notes) == 1
    assert any(event.memoryDisposition == MemoryDisposition.DUPLICATE for event in result.events)


def test_memory_coverage_invariant():
    meeting = build_meeting_c()
    result = asyncio.run(
        run_event_pipeline(
            meeting["chunks"],
            meeting["id"],
            "user_1",
            "space_1",
            event_extractor=ScriptedEventExtractor(events=meeting["events"]),
            embedder=_embedder(),
        )
    )
    assert result.coverage is not None
    coverage = result.coverage
    accounted = (
        coverage.memoryPublished
        + coverage.memoryDuplicates
        + coverage.memorySuperseded
        + coverage.memoryLowValue
        + coverage.memoryUnsupported
        + coverage.memoryRelatedContext
        + coverage.memoryRejected
    )
    assert accounted + coverage.memoryUnaccounted == coverage.memory_events
    assert coverage.memoryUnaccounted == 0
    assert coverage.memoryCoverageFailure is False
    assert unpublished_memory_events(result.events) == []


def test_memory_coverage_ledger_counts_terminal_states():
    events = [
        _event("e1", EventKind.ISSUE, "A useful issue.", 0, "A useful issue.", object="issue", memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH")),
    ]
    events[0].memoryDisposition = MemoryDisposition.PUBLISHED_NOTE
    ledger = CoverageLedger(memory_events=1)
    apply_memory_coverage(ledger, events)
    assert ledger.memoryPublished == 1
    assert ledger.memoryUnaccounted == 0
    assert ledger.memoryCoverageFailure is False


def test_meeting_c_scripted_publishes_all_gold_notes():
    from services.conversation.event_pipeline.gold_scoring import pipeline_benchmark

    meeting = build_meeting_c()
    transcript = "\n".join(f"[{seq}] {text}" for seq, text in meeting["lines"].items() if str(text).strip())
    result = asyncio.run(
        run_event_pipeline(
            meeting["chunks"],
            meeting["id"],
            "user_1",
            "space_1",
            event_extractor=ScriptedEventExtractor(events=meeting["events"]),
            embedder=_embedder(),
        )
    )
    report = pipeline_benchmark(
        result,
        meeting["goldTasks"],
        meeting["goldNotes"],
        case_id=meeting["id"],
        transcript=transcript,
        gold_complete=True,
    )
    assert report["noteRecall"] == 1.0
    assert report["taskRecall"] == 1.0
    blob = _blob(result.notes)
    for token in ("pdf", "webhook", "gst"):
        assert token in blob


def test_content_island_in_filler_is_recovered_when_extractor_skips():
    chunks = [_chunk(seq, f"umm {seq}") for seq in range(240, 255)]
    chunks[10] = _chunk(250, "GST field optional rakhne ka decision ho gaya")
    result = asyncio.run(
        run_event_pipeline(
            chunks,
            "conv",
            "u",
            "s",
            event_extractor=ScriptedEventExtractor(events=[]),
            embedder=_embedder(),
        )
    )
    blob = _blob(result.notes)
    assert "gst" in blob or "optional" in blob
    assert result.tasks == []
    from services.conversation.event_pipeline.channels import is_generic_task_text

    for case in all_generic_conversations():
        result = asyncio.run(
            run_event_pipeline(
                case["chunks"],
                case["id"],
                "user_1",
                "space_1",
                event_extractor=ScriptedEventExtractor(events=case["events"]),
                embedder=_embedder(),
            )
        )
        note_blob = _blob(result.notes)
        task_blob = _blob(result.tasks)
        if case.get("expectNoTask"):
            assert result.tasks == [], case["id"]
        if case.get("expectNoNote"):
            assert result.notes == [], case["id"]
        for token in case.get("expectNoteSubstrings") or []:
            assert token.casefold() in note_blob, (case["id"], token, note_blob)
        task_needles = case.get("expectTaskSubstrings") or []
        if task_needles:
            assert any(token.casefold() in task_blob for token in task_needles), (case["id"], task_blob)
        if case.get("forbidGenericTask"):
            assert all(not is_generic_task_text(task.title, task.body) for task in result.tasks)
        assert result.coverage is None or result.coverage.memoryUnaccounted == 0
