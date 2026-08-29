"""Freeze the working Task path. Do not lower Task quality while fixing Notes."""

from __future__ import annotations

import asyncio

from services.conversation.event_pipeline.channels import is_generic_task_text
from services.conversation.event_pipeline.embeddings import CachedEmbedder, LexicalEmbedder
from services.conversation.event_pipeline.events import ScriptedEventExtractor
from services.conversation.event_pipeline.pipeline import run_event_pipeline
from services.conversation.event_pipeline.schemas import ABSTAIN_UNRESOLVED_OBJECT, ActionSignal, AtomicEvent, EventKind
from services.conversation.event_pipeline.validation import mixed_thread_rate
from services.conversation.models import EvidenceSpan, STTStatus, TranscriptChunkDocument
from tests.fixtures.long_meeting_gold import build_gold_transcript
from tests.fixtures.reviewed_meetings import all_reviewed_meetings


def _embedder() -> CachedEmbedder:
    return CachedEmbedder(LexicalEmbedder())


def _chunk(sequence: int, text: str, conversation_id: str = "conv") -> TranscriptChunkDocument:
    return TranscriptChunkDocument(
        conversationId=conversation_id,
        userId="u",
        spaceId="s",
        chunkId=f"{conversation_id}_{sequence}",
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


def test_specific_grounded_tasks_publish():
    result = asyncio.run(
        run_event_pipeline(
            [_chunk(0, "Server ID create karna hai")],
            "conv",
            "u",
            "s",
            event_extractor=ScriptedEventExtractor(
                events=[
                    _event(
                        "e1",
                        EventKind.REQUEST,
                        "Create server ID.",
                        0,
                        "Server ID create karna hai",
                        object="server ID",
                        entities=["Server", "ID"],
                        actionSignal=ActionSignal(
                            isActionable=True,
                            role="REQUEST",
                            actionStrength="EXPLICIT",
                            verb="create",
                            object="server ID",
                            canonicalActionObject="server ID",
                            objectGroundingType="EXPLICIT",
                        ),
                    )
                ]
            ),
            embedder=_embedder(),
        )
    )
    assert result.tasks
    assert any("server" in task.title.casefold() and "id" in task.title.casefold() for task in result.tasks)
    assert all(not is_generic_task_text(task.title, task.body) for task in result.tasks)


def test_generic_tasks_do_not_publish():
    result = asyncio.run(
        run_event_pipeline(
            [_chunk(0, "please complete pending work")],
            "conv",
            "u",
            "s",
            event_extractor=ScriptedEventExtractor(
                events=[
                    _event(
                        "e1",
                        EventKind.REQUEST,
                        "Complete pending work.",
                        0,
                        "please complete pending work",
                        object="pending work",
                        actionSignal=ActionSignal(
                            isActionable=True,
                            role="REQUEST",
                            actionStrength="EXPLICIT",
                            verb="complete",
                            object="pending work",
                            objectGroundingType="EXPLICIT",
                        ),
                    )
                ]
            ),
            embedder=_embedder(),
        )
    )
    assert result.tasks == [] or all(not is_generic_task_text(task.title, task.body) for task in result.tasks)
    assert not any("pending work" in f"{task.title} {task.body}".casefold() for task in result.tasks)


def test_unresolved_action_objects_abstain():
    result = asyncio.run(
        run_event_pipeline(
            [_chunk(0, "kal isko fix kar denge")],
            "conv",
            "u",
            "s",
            event_extractor=ScriptedEventExtractor(
                events=[
                    _event(
                        "e1",
                        EventKind.COMMITMENT,
                        "Fix it tomorrow.",
                        0,
                        "kal isko fix kar denge",
                        actionSignal=ActionSignal(
                            isActionable=True,
                            role="COMMITMENT",
                            actionStrength="EXPLICIT",
                            verb="fix",
                            object=None,
                            objectGroundingType="UNRESOLVED",
                            artifactStatus=ABSTAIN_UNRESOLVED_OBJECT,
                            deadline="kal",
                        ),
                    )
                ]
            ),
            embedder=_embedder(),
        )
    )
    assert result.tasks == []


def test_mixed_thread_evidence_does_not_attach():
    events = [
        _event(
            "e-server",
            EventKind.REQUEST,
            "Create server ID.",
            110,
            "Server ID create karna hai",
            object="server ID",
            entities=["Server", "ID"],
            actionSignal=ActionSignal(
                isActionable=True,
                role="REQUEST",
                actionStrength="EXPLICIT",
                verb="create",
                object="server ID",
                canonicalActionObject="server ID",
                objectGroundingType="EXPLICIT",
            ),
        ),
        _event(
            "e-conn",
            EventKind.ISSUE,
            "Connection string missing.",
            60,
            "database server connection string missing hai",
            object="connection string",
            entities=["database"],
        ),
    ]
    result = asyncio.run(
        run_event_pipeline(
            [_chunk(60, "database server connection string missing hai"), _chunk(110, "Server ID create karna hai")],
            "conv",
            "u",
            "s",
            event_extractor=ScriptedEventExtractor(events=events),
            embedder=_embedder(),
        )
    )
    assert result.tasks
    assert mixed_thread_rate(result.tasks, result.events) == 0
    server = next(task for task in result.tasks if "server" in task.title.casefold())
    sequences = {span.sequenceStart for span in server.evidence} | {span.sequenceEnd for span in server.evidence}
    assert 60 not in sequences


def test_create_server_id_paraphrase_dedupes_without_merging_track():
    events = [
        _event(
            "e-create-a",
            EventKind.REQUEST,
            "Create server ID.",
            0,
            "Create server ID",
            object="server ID",
            entities=["Server", "ID"],
            threadId="TH-server",
            actionSignal=ActionSignal(
                isActionable=True,
                role="REQUEST",
                actionStrength="EXPLICIT",
                verb="create",
                object="server ID",
                canonicalActionObject="server ID",
                objectGroundingType="EXPLICIT",
            ),
        ),
        _event(
            "e-create-b",
            EventKind.REQUEST,
            "Create the server ID.",
            1,
            "Create the server ID",
            object="server ID",
            entities=["Server", "ID"],
            threadId="TH-server",
            actionSignal=ActionSignal(
                isActionable=True,
                role="REQUEST",
                actionStrength="EXPLICIT",
                verb="create",
                object="the server ID",
                canonicalActionObject="server ID",
                objectGroundingType="EXPLICIT",
            ),
        ),
        _event(
            "e-track",
            EventKind.FOLLOW_UP,
            "Track server ID.",
            2,
            "Track server ID",
            object="server ID",
            entities=["Server", "ID"],
            threadId="TH-server",
            actionSignal=ActionSignal(
                isActionable=True,
                role="FOLLOW_UP",
                actionStrength="EXPLICIT",
                verb="track",
                object="server ID",
                canonicalActionObject="server ID",
                objectGroundingType="EXPLICIT",
            ),
        ),
    ]
    result = asyncio.run(
        run_event_pipeline(
            [_chunk(0, "Create server ID"), _chunk(1, "Create the server ID"), _chunk(2, "Track server ID")],
            "conv",
            "u",
            "s",
            event_extractor=ScriptedEventExtractor(events=events),
            embedder=_embedder(),
        )
    )
    create_tasks = [task for task in result.tasks if "create" in f"{task.title} {task.body}".casefold()]
    track_tasks = [task for task in result.tasks if "track" in f"{task.title} {task.body}".casefold()]
    assert len(create_tasks) == 1
    assert len(track_tasks) == 1


def test_task_recall_does_not_regress_on_scripted_gold():
    gold = build_gold_transcript()
    result = asyncio.run(
        run_event_pipeline(
            gold["chunks"],
            "gold-long-meeting",
            "user_1",
            "space_1",
            event_extractor=ScriptedEventExtractor(events=gold["events"]),
            embedder=_embedder(),
        )
    )
    from services.conversation.event_pipeline.gold_scoring import pipeline_benchmark

    transcript = "\n".join(f"[{seq}] {text}" for seq, text in gold["lines"].items() if str(text).strip())
    report = pipeline_benchmark(
        result,
        gold["goldTasks"],
        gold["goldNotes"],
        case_id="gold-long-meeting",
        transcript=transcript,
        valid_additional_notes=gold.get("validAdditionalNotes"),
        valid_additional_tasks=gold.get("validAdditionalTasks"),
        gold_complete=True,
    )
    assert report["groundedPrecisionTasks"] >= 0.90
    assert report["taskRecall"] >= 0.85
    assert report["genericTaskRate"] == 0
    assert report["mixedThreadRate"] == 0


def test_reviewed_meetings_keep_task_bar():
    from services.conversation.event_pipeline.gold_scoring import pipeline_benchmark

    for meeting in all_reviewed_meetings():
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
        assert report["groundedPrecisionTasks"] >= 0.90
        assert report["taskRecall"] >= 0.85
        assert report["genericTaskRate"] == 0
        assert report["mixedThreadRate"] <= 0.05
