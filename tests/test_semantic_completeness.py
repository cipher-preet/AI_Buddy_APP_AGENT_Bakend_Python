"""Proposition-level coverage: one event must not consume a multi-meaning block."""

from __future__ import annotations

import asyncio

from services.conversation.event_pipeline.channels import is_generic_task_text
from services.conversation.event_pipeline.completeness import MissingSemanticUnit, ScriptedCompletenessReviewer
from services.conversation.event_pipeline.embeddings import CachedEmbedder, LexicalEmbedder
from services.conversation.event_pipeline.events import ScriptedEventExtractor
from services.conversation.event_pipeline.memory_identity import memory_relation
from services.conversation.event_pipeline.pipeline import run_event_pipeline
from services.conversation.event_pipeline.schemas import ActionSignal, AtomicEvent, EventKind, MemorySignal
from services.conversation.models import EvidenceSpan, STTStatus, TranscriptChunkDocument
from tests.fixtures.generic_conversations import (
    filler_does_not_invent_units,
    multi_meaning_conversations,
    related_distinct_same_subject,
)
from tests.fixtures.long_meeting_gold import build_gold_transcript
from tests.fixtures.reviewed_meetings import build_meeting_b


def _embedder() -> CachedEmbedder:
    return CachedEmbedder(LexicalEmbedder())


def _chunk(sequence: int, text: str, conversation_id: str = "conv") -> TranscriptChunkDocument:
    return TranscriptChunkDocument(
        conversationId=conversation_id,
        userId="u",
        spaceId="s",
        chunkId=f"chunk_{sequence}",
        sequenceNumber=sequence,
        rawText=text,
        sttStatus=STTStatus.COMPLETED,
    )


def _units(case: dict) -> list[MissingSemanticUnit]:
    return [MissingSemanticUnit(**item) for item in case.get("expectedUnits") or []]


def test_under_extraction_is_repaired_for_generic_multi_meaning_blocks():
    for case in multi_meaning_conversations():
        result = asyncio.run(
            run_event_pipeline(
                case["chunks"],
                case["id"],
                "user_1",
                "space_1",
                event_extractor=ScriptedEventExtractor(
                    events=case["firstPassEvents"],
                    repair_events=case.get("repairEvents") or [],
                ),
                completeness_reviewer=ScriptedCompletenessReviewer(_units(case)),
                embedder=_embedder(),
            )
        )
        non_noise = [event for event in result.events if event.kind != EventKind.NOISE]
        if case.get("expectNoNote") and case.get("expectNoTask"):
            assert result.tasks == []
            assert result.notes == []
            assert result.coverage is not None
            assert result.coverage.unaccountedSemanticUnits == 0
            continue
        assert len(non_noise) >= case["minEvents"], (case["id"], [event.meaning for event in non_noise])
        task_blob = " ".join(f"{item.title} {item.body}" for item in result.tasks).casefold()
        note_blob = " ".join(f"{item.title} {item.body}" for item in result.notes).casefold()
        for token in case.get("expectNoteSubstrings") or []:
            assert token.casefold() in note_blob or token.casefold() in " ".join(event.meaning for event in non_noise).casefold(), (
                case["id"],
                token,
            )
        needles = case.get("expectTaskSubstrings") or []
        if needles:
            assert any(token.casefold() in task_blob for token in needles), (case["id"], task_blob)
        assert all(not is_generic_task_text(task.title, task.body) for task in result.tasks)
        assert result.coverage is not None
        assert result.coverage.unaccountedSemanticUnits == 0
        assert result.coverage.semanticCoverageFailure is False
        assert result.coverage.semanticReviewRan is True
        assert result.coverage.semanticCoverage >= 0.90


def test_block_accounted_is_not_semantic_completeness_without_repair():
    case = next(item for item in multi_meaning_conversations() if item["id"] == "multi-personal")
    under = asyncio.run(
        run_event_pipeline(
            case["chunks"],
            case["id"],
            "user_1",
            "space_1",
            event_extractor=ScriptedEventExtractor(events=case["firstPassEvents"]),
            embedder=_embedder(),
        )
    )
    assert under.coverage is not None
    assert under.coverage.unaccounted_blocks == 0
    non_noise = [event for event in under.events if event.kind != EventKind.NOISE]
    assert len(non_noise) < case["minEvents"]

    repaired = asyncio.run(
        run_event_pipeline(
            case["chunks"],
            case["id"] + "-repaired",
            "user_1",
            "space_1",
            event_extractor=ScriptedEventExtractor(
                events=case["firstPassEvents"],
                repair_events=case["repairEvents"],
            ),
            completeness_reviewer=ScriptedCompletenessReviewer(_units(case)),
            embedder=_embedder(),
        )
    )
    recovered = [event for event in repaired.events if event.kind != EventKind.NOISE]
    assert len(recovered) >= case["minEvents"]
    assert repaired.coverage.unaccountedSemanticUnits == 0


def test_related_but_distinct_propositions_are_not_deduped():
    case = related_distinct_same_subject()
    generated = case["events"][0]
    opens = case["events"][1]
    generated.threadId = "t-link"
    opens.threadId = "t-link"
    assert memory_relation(opens, generated) == "DISTINCT"
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
    note_blob = " ".join(f"{item.title} {item.body}" for item in result.notes).casefold()
    assert "generated" in note_blob
    assert "form" in note_blob
    assert len(result.notes) >= 2


def test_filler_completeness_does_not_manufacture_artifacts():
    case = filler_does_not_invent_units()
    result = asyncio.run(
        run_event_pipeline(
            case["chunks"],
            case["id"],
            "user_1",
            "space_1",
            event_extractor=ScriptedEventExtractor(events=case["firstPassEvents"], repair_events=[]),
            completeness_reviewer=ScriptedCompletenessReviewer([]),
            embedder=_embedder(),
        )
    )
    assert result.tasks == []
    assert result.notes == []
    assert result.coverage.unaccountedSemanticUnits == 0
    assert all(not is_generic_task_text(task.title, task.body) for task in result.tasks)


def test_production_dense_block_recovers_independent_meanings():
    lines = {
        0: "HRMS तो हमें बनाना ही है.",
        1: "candidate को onboard करेंगे share candidate detail link generate होगा link candidate को जाएगा link पर form खुलेगा",
        2: "AI hiring थोड़ा बना लेंगे इसको emails automatic होंगे interviews automatic होंगे",
        3: "payroll उसको भी बनाएंगे.",
        4: "इसको पूरा देखना है कि direct integration क्या use करते हैं.",
    }
    first = [
        AtomicEvent(
            eventId="e-hrms",
            topicId="T1",
            kind=EventKind.COMMITMENT,
            meaning="Build HRMS.",
            object="HRMS",
            entities=["HRMS"],
            evidence=[EvidenceSpan(sequenceStart=0, sequenceEnd=0, text=lines[0])],
            sequenceIds=[0],
            conversationId="prod-complete",
            userId="u",
            spaceId="s",
            actionSignal=ActionSignal(
                isActionable=True,
                role="COMMITMENT",
                actionStrength="EXPLICIT",
                verb="build",
                object="HRMS",
                canonicalActionObject="HRMS",
                objectGroundingType="EXPLICIT",
            ),
        ),
        AtomicEvent(
            eventId="e-onboard-blob",
            topicId="T1",
            kind=EventKind.COMMITMENT,
            meaning="Onboard the candidate.",
            object="candidate",
            entities=["candidate"],
            evidence=[EvidenceSpan(sequenceStart=1, sequenceEnd=1, text=lines[1])],
            sequenceIds=[1],
            conversationId="prod-complete",
            userId="u",
            spaceId="s",
            actionSignal=ActionSignal(
                isActionable=True,
                role="COMMITMENT",
                actionStrength="EXPLICIT",
                verb="onboard",
                object="candidate",
                canonicalActionObject="candidate",
                objectGroundingType="EXPLICIT",
            ),
        ),
        AtomicEvent(
            eventId="e-ai-blob",
            topicId="T1",
            kind=EventKind.COMMITMENT,
            meaning="Build AI hiring.",
            object="AI hiring",
            entities=["AI"],
            evidence=[EvidenceSpan(sequenceStart=2, sequenceEnd=2, text=lines[2])],
            sequenceIds=[2],
            conversationId="prod-complete",
            userId="u",
            spaceId="s",
            actionSignal=ActionSignal(
                isActionable=True,
                role="COMMITMENT",
                actionStrength="EXPLICIT",
                verb="build",
                object="AI hiring",
                canonicalActionObject="AI hiring",
                objectGroundingType="EXPLICIT",
            ),
        ),
        AtomicEvent(
            eventId="e-payroll",
            topicId="T1",
            kind=EventKind.COMMITMENT,
            meaning="Build payroll.",
            object="payroll",
            entities=["payroll"],
            evidence=[EvidenceSpan(sequenceStart=3, sequenceEnd=3, text=lines[3])],
            sequenceIds=[3],
            conversationId="prod-complete",
            userId="u",
            spaceId="s",
            actionSignal=ActionSignal(
                isActionable=True,
                role="COMMITMENT",
                actionStrength="EXPLICIT",
                verb="build",
                object="payroll",
                canonicalActionObject="payroll",
                objectGroundingType="EXPLICIT",
            ),
        ),
        AtomicEvent(
            eventId="e-integ",
            topicId="T1",
            kind=EventKind.FOLLOW_UP,
            meaning="Investigate whether/how direct integration should be used.",
            object="direct integration",
            entities=["integration"],
            evidence=[EvidenceSpan(sequenceStart=4, sequenceEnd=4, text=lines[4])],
            sequenceIds=[4],
            conversationId="prod-complete",
            userId="u",
            spaceId="s",
            actionSignal=ActionSignal(
                isActionable=True,
                role="FOLLOW_UP",
                actionStrength="EXPLICIT",
                verb="investigate",
                object="direct integration",
                canonicalActionObject="direct integration",
                objectGroundingType="EXPLICIT",
            ),
            memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH", reason="OPEN_QUESTION"),
        ),
    ]
    repair = [
        AtomicEvent(
            eventId="e-share-detail",
            topicId="T1",
            kind=EventKind.REQUIREMENT,
            meaning="Candidate details are shared.",
            object="candidate detail",
            entities=["candidate"],
            evidence=[EvidenceSpan(sequenceStart=1, sequenceEnd=1, text=lines[1])],
            sequenceIds=[1],
            conversationId="prod-complete",
            userId="u",
            spaceId="s",
            memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH", reason="REQUIREMENT"),
        ),
        AtomicEvent(
            eventId="e-link-gen",
            topicId="T1",
            kind=EventKind.REQUIREMENT,
            meaning="A candidate link is generated.",
            object="candidate link",
            entities=["candidate", "link"],
            evidence=[EvidenceSpan(sequenceStart=1, sequenceEnd=1, text=lines[1])],
            sequenceIds=[1],
            conversationId="prod-complete",
            userId="u",
            spaceId="s",
            memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH", reason="REQUIREMENT"),
        ),
        AtomicEvent(
            eventId="e-link-form",
            topicId="T1",
            kind=EventKind.REQUIREMENT,
            meaning="The candidate link opens a form.",
            object="candidate form",
            entities=["candidate", "form"],
            evidence=[EvidenceSpan(sequenceStart=1, sequenceEnd=1, text=lines[1])],
            sequenceIds=[1],
            conversationId="prod-complete",
            userId="u",
            spaceId="s",
            memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH", reason="REQUIREMENT"),
        ),
        AtomicEvent(
            eventId="e-ai-email",
            topicId="T1",
            kind=EventKind.REQUIREMENT,
            meaning="AI hiring emails are intended to be automated.",
            object="AI hiring emails",
            entities=["emails"],
            evidence=[EvidenceSpan(sequenceStart=2, sequenceEnd=2, text=lines[2])],
            sequenceIds=[2],
            conversationId="prod-complete",
            userId="u",
            spaceId="s",
            memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH", reason="REQUIREMENT"),
        ),
        AtomicEvent(
            eventId="e-ai-interview",
            topicId="T1",
            kind=EventKind.REQUIREMENT,
            meaning="AI hiring interviews are intended to be automated.",
            object="AI hiring interviews",
            entities=["interviews"],
            evidence=[EvidenceSpan(sequenceStart=2, sequenceEnd=2, text=lines[2])],
            sequenceIds=[2],
            conversationId="prod-complete",
            userId="u",
            spaceId="s",
            memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH", reason="REQUIREMENT"),
        ),
    ]
    expected = [
        MissingSemanticUnit(meaning=event.meaning, kind=event.kind.value, sequenceStart=event.sequenceIds[0], sequenceEnd=event.sequenceIds[0], evidenceText=event.evidence[0].text)
        for event in [*first, *repair]
    ]
    chunks = [_chunk(seq, text, "prod-complete") for seq, text in lines.items()]
    result = asyncio.run(
        run_event_pipeline(
            chunks,
            "prod-complete",
            "u",
            "s",
            event_extractor=ScriptedEventExtractor(events=first, repair_events=repair),
            completeness_reviewer=ScriptedCompletenessReviewer(expected),
            embedder=_embedder(),
        )
    )
    meanings = " ".join(event.meaning for event in result.events).casefold()
    assert "onboard" in meanings
    assert "link" in meanings
    assert "form" in meanings
    assert "email" in meanings
    assert "interview" in meanings
    assert "payroll" in meanings
    note_blob = " ".join(f"{note.title} {note.body}" for note in result.notes).casefold()
    assert "is directly integrated" not in note_blob
    assert result.coverage.unaccountedSemanticUnits == 0
    assert result.coverage.semanticCoverageFailure is False
    assert result.coverage.unaccounted_blocks == 0
    assert result.coverage.actionCoverageFailure is False
    assert result.coverage.memoryCoverageFailure is False
    assert len(result.tasks) >= 3
    assert all(not is_generic_task_text(task.title, task.body) for task in result.tasks)
    print(
        "PRODUCTION_SEMANTIC_COVERAGE",
        {
            "microblocks": len(result.microBlocks),
            "semanticUnits": result.coverage.semanticUnitsDetected,
            "atomicEvents": len(result.events),
            "usefulMemory": result.coverage.memory_events,
            "actionEvents": result.coverage.action_events,
            "tasks": len(result.tasks),
            "notes": len(result.notes),
            "unaccountedSemanticUnits": result.coverage.unaccountedSemanticUnits,
        },
    )


def test_checkpoint_events_survive_stop_merge():
    checkpoint = AtomicEvent(
        eventId="chk-note",
        topicId="T0",
        kind=EventKind.FACT,
        meaning="Meeting tracking should retain the meeting owner.",
        object="meeting owner",
        entities=["meeting"],
        evidence=[EvidenceSpan(sequenceStart=0, sequenceEnd=0, text="meeting owner track karna padega")],
        sequenceIds=[0],
        conversationId="chk-merge",
        userId="u",
        spaceId="s",
        memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH", reason="REQUIREMENT"),
    )
    leftover = AtomicEvent(
        eventId="stop-note",
        topicId="T1",
        kind=EventKind.REQUIREMENT,
        meaning="Meeting-related notes should be retained.",
        object="meeting notes",
        entities=["notes"],
        evidence=[EvidenceSpan(sequenceStart=10, sequenceEnd=10, text="notes banane padenge")],
        sequenceIds=[10],
        conversationId="chk-merge",
        userId="u",
        spaceId="s",
        memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH", reason="REQUIREMENT"),
    )
    result = asyncio.run(
        run_event_pipeline(
            [_chunk(0, "meeting owner track karna padega", "chk-merge"), _chunk(10, "notes banane padenge", "chk-merge")],
            "chk-merge",
            "u",
            "s",
            checkpoint_events=[checkpoint],
            event_extractor=ScriptedEventExtractor(events=[leftover]),
            embedder=_embedder(),
        )
    )
    ids = {event.eventId for event in result.events}
    assert "chk-note" in ids
    assert "stop-note" in ids
    blob = " ".join(f"{note.title} {note.body}" for note in result.notes).casefold()
    assert "owner" in blob
    assert "note" in blob


def test_meeting_b_and_gold_keep_quality_after_completeness_ledger():
    from services.conversation.event_pipeline.gold_scoring import pipeline_benchmark

    meeting = build_meeting_b()
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
    assert report["taskRecall"] == 1.0
    assert report["genericTaskRate"] == 0
    assert report["unaccountedSemanticUnits"] == 0
    assert report["semanticCoverageFailure"] is False

    gold = build_gold_transcript()
    gold_transcript = "\n".join(f"[{seq}] {text}" for seq, text in gold["lines"].items() if str(text).strip())
    long_result = asyncio.run(
        run_event_pipeline(
            gold["chunks"],
            "gold-long-meeting",
            "user_1",
            "space_1",
            event_extractor=ScriptedEventExtractor(events=gold["events"]),
            embedder=_embedder(),
        )
    )
    gold_report = pipeline_benchmark(
        long_result,
        gold["goldTasks"],
        gold["goldNotes"],
        case_id="gold-long-meeting",
        transcript=gold_transcript,
        valid_additional_notes=gold.get("validAdditionalNotes"),
        valid_additional_tasks=gold.get("validAdditionalTasks"),
        gold_events=gold["events"],
        gold_threads=gold.get("goldThreads"),
        gold_complete=True,
        original_actionable_ids=gold.get("originalActionableEventIds"),
        reviewed_actionable_ids=gold.get("reviewedActionableEventIds"),
    )
    assert gold_report["groundedPrecisionTasks"] >= 0.90
    assert gold_report["requiredTaskRecall"] >= 0.85
    assert gold_report["noteUsefulnessPrecision"] >= 0.85
    assert gold_report["requiredNoteRecall"] >= 0.85
    assert gold_report["genericTaskRate"] == 0
    assert gold_report["mixedThreadRate"] <= 0.05
    assert gold_report["unaccountedSemanticUnits"] == 0
    assert gold_report["semanticCoverageFailure"] is False
    assert long_result.coverage.unaccounted_blocks == 0
    assert long_result.coverage.actionCoverageFailure is False
    assert long_result.coverage.memoryCoverageFailure is False
