"""Independent atomic-event gold. Scripted extractor verifies scoring; real Gemma is integration-only."""

from __future__ import annotations

import asyncio

from services.conversation.event_pipeline.channels import is_generic_task_text
from services.conversation.event_pipeline.embeddings import CachedEmbedder, LexicalEmbedder
from services.conversation.event_pipeline.events import ScriptedEventExtractor, materialize_events, AtomicEventLLMItem, AtomicEventLLMResponse
from services.conversation.event_pipeline.gold_scoring import score_events
from services.conversation.event_pipeline.pipeline import run_event_pipeline
from services.conversation.event_pipeline.schemas import AtomicEvent, EventKind, LocalTopic
from services.conversation.models import EvidenceSpan, STTStatus, TranscriptChunkDocument
from tests.fixtures.atomic_event_gold import SEGMENTS


def _chunk(sequence: int, text: str) -> TranscriptChunkDocument:
    return TranscriptChunkDocument(
        conversationId="atomic-gold",
        userId="u",
        spaceId="s",
        chunkId=f"chunk_{sequence}",
        sequenceNumber=sequence,
        rawText=text,
        sttStatus=STTStatus.COMPLETED,
    )


def _scripted_events(segment: dict, sequence: int) -> list[AtomicEvent]:
    events = []
    for index, item in enumerate(segment["expected"]):
        events.append(
            AtomicEvent(
                eventId=f"{segment['id']}-{index}",
                topicId="T1",
                kind=item["kind"],
                meaning=item["meaning"],
                object=item.get("object"),
                uncertainty=list(item.get("uncertainty") or []),
                evidence=[EvidenceSpan(sequenceStart=sequence, sequenceEnd=sequence, text=segment["text"])],
                sequenceIds=[sequence],
                conversationId="atomic-gold",
                userId="u",
                spaceId="s",
            )
        )
    return events


def test_atomic_event_gold_scripted_metrics():
    all_events = []
    chunks = []
    for index, segment in enumerate(SEGMENTS):
        chunks.append(_chunk(index, segment["text"]))
        all_events.extend(_scripted_events(segment, index))
    result = asyncio.run(
        run_event_pipeline(
            chunks,
            "atomic-gold",
            "u",
            "s",
            event_extractor=ScriptedEventExtractor(events=all_events),
            embedder=CachedEmbedder(LexicalEmbedder()),
        )
    )
    metrics = score_events(result.events, all_events)
    print("ATOMIC_EVENT_GOLD_SCRIPTED", metrics)
    assert metrics["eventRecall"] >= 0.9
    assert metrics["eventPrecision"] >= 0.8
    assert metrics["eventTypeAccuracy"] >= 0.9


def test_insecure_connection_does_not_become_a_task():
    segment = next(item for item in SEGMENTS if item["id"] == "insecure-no-task")
    events = _scripted_events(segment, 0)
    result = asyncio.run(
        run_event_pipeline(
            [_chunk(0, segment["text"])],
            "atomic-gold",
            "u",
            "s",
            event_extractor=ScriptedEventExtractor(events=events),
            embedder=CachedEmbedder(LexicalEmbedder()),
        )
    )
    assert result.tasks == []
    assert result.notes
    assert not any("fix connection" in f"{task.title} {task.body}".casefold() for task in result.tasks)


def test_kal_kar_denge_preserves_ambiguity_instead_of_generic_task():
    segment = next(item for item in SEGMENTS if item["id"] == "kal-kar-denge-ambiguous")
    events = _scripted_events(segment, 0)
    result = asyncio.run(
        run_event_pipeline(
            [_chunk(0, segment["text"])],
            "atomic-gold",
            "u",
            "s",
            event_extractor=ScriptedEventExtractor(events=events),
            embedder=CachedEmbedder(LexicalEmbedder()),
        )
    )
    assert not any(is_generic_task_text(task.title, task.body) for task in result.tasks)
    assert not any("complete pending task" in task.title.casefold() for task in result.tasks)
    assert any("missing_object" in event.uncertainty for event in events)


def test_materialize_events_drops_ungrounded_evidence_ids():
    topic = LocalTopic(
        topicId="T1",
        label="S3",
        sequenceStart=0,
        sequenceEnd=0,
        sequenceIds=[0],
        text="[0] S3 frontend nahi aa raha",
    )
    response = AtomicEventLLMResponse(
        events=[
            AtomicEventLLMItem(
                kind=EventKind.ISSUE,
                meaning="S3 is not reaching the frontend.",
                evidence=[EvidenceSpan(sequenceStart=99, sequenceEnd=99, text="invented")],
            )
        ]
    )
    events = materialize_events(response, topic, [], {0: "S3 frontend nahi aa raha"})
    assert events
    assert all(span.sequenceStart != 99 and span.sequenceEnd != 99 for event in events for span in event.evidence)
    assert events[0].sequenceIds == [0]
