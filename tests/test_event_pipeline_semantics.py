import asyncio

from services.conversation.event_pipeline.channels import is_generic_task_text, split_action_and_memory
from services.conversation.event_pipeline.embeddings import CachedEmbedder, LexicalEmbedder
from services.conversation.event_pipeline.events import ScriptedEventExtractor
from services.conversation.event_pipeline.pipeline import run_event_pipeline
from services.conversation.event_pipeline.schemas import ActionSignal, AtomicEvent, EventKind, ThreadRelation
from services.conversation.event_pipeline.threads import link_global_threads
from services.conversation.event_pipeline.validation import validate_artifact
from services.conversation.models import EvidenceSpan, ExtractedTask, STTStatus, TranscriptChunkDocument


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
        actor=kwargs.get("actor"),
        object=kwargs.get("object"),
        timeExpression=kwargs.get("timeExpression"),
        entities=kwargs.get("entities") or [],
        evidence=[EvidenceSpan(sequenceStart=sequence, sequenceEnd=sequence, text=text)],
        sequenceIds=[sequence],
        sourceIds=[f"chunk_{sequence}"],
        conversationId="conv",
        userId="u",
        spaceId="s",
    )


def test_global_threads_link_distant_s3_and_keep_pricing_separate():
    events = [
        _event("e1", EventKind.ISSUE, "S3 has a problem.", 10, "S3 has a problem.", object="S3", entities=["S3"]),
        _event("e2", EventKind.PROPOSAL, "Pricing should start around 200.", 90, "Pricing should start around 200.", object="pricing", entities=["Pricing"]),
        _event("e3", EventKind.STATE, "S3 frontend still failing.", 180, "S3 frontend still failing.", object="S3 frontend", entities=["S3", "frontend"]),
    ]
    threads, links, comparisons = asyncio.run(link_global_threads(events, _embedder()))
    s3_thread = next(thread for thread in threads if "e1" in thread.eventIds)
    assert "e3" in s3_thread.eventIds
    assert "e2" not in s3_thread.eventIds
    pricing = next(thread for thread in threads if "e2" in thread.eventIds)
    assert pricing.threadId != s3_thread.threadId
    assert comparisons <= max(8, len(events) * 8)


def test_false_positive_link_prevention_for_unrelated_high_similarity_vocab():
    events = [
        _event("e1", EventKind.REQUEST, "Create server ID.", 1, "Server ID create karna hai", object="server ID", entities=["Server", "ID"]),
        _event("e2", EventKind.ISSUE, "Connection string is missing.", 2, "connection string missing", object="connection string", entities=["connection"]),
    ]
    threads, _, _ = asyncio.run(link_global_threads(events, _embedder()))
    assert len(threads) >= 2


def test_related_but_distinct_server_objects_do_not_merge():
    events = [
        _event("e-id", EventKind.REQUEST, "Create server ID.", 1, "Server ID create karna hai", object="server ID", entities=["Server", "ID"]),
        _event("e-fail", EventKind.ISSUE, "Server connection failure.", 2, "server connection fail ho raha", object="server connection", entities=["server", "connection"]),
        _event("e-string", EventKind.ISSUE, "Database connection string is missing.", 3, "database connection string missing", object="connection string", entities=["database"]),
        _event("e-port", EventKind.FOLLOW_UP, "Track Port ID.", 4, "kal track kariye Port ID", object="Port ID", entities=["Port", "ID"]),
    ]
    for event in events:
        event.actionSignal = ActionSignal(isActionable=event.kind in {EventKind.REQUEST, EventKind.FOLLOW_UP}, actionStrength="EXPLICIT" if event.kind in {EventKind.REQUEST, EventKind.FOLLOW_UP} else "NONE", object=event.object)
    threads, links, _ = asyncio.run(link_global_threads(events, _embedder()))
    by_event = {event.eventId: event.threadId for event in events}
    assert len({by_event["e-id"], by_event["e-fail"], by_event["e-string"], by_event["e-port"]}) == 4
    assert all(link.relation != ThreadRelation.SAME_THREAD or link.fromEventId == link.toEventId for link in links if link.relation == ThreadRelation.RELATED_BUT_DISTINCT)


def test_issue_without_request_is_note_only():
    result = asyncio.run(
        run_event_pipeline(
            [_chunk(0, "Connection is insecure.")],
            "conv",
            "u",
            "s",
            event_extractor=ScriptedEventExtractor(
                events=[_event("e1", EventKind.STATE, "Connection is reported as insecure.", 0, "Connection is insecure.", object="connection")]
            ),
            embedder=_embedder(),
        )
    )
    assert result.tasks == []
    assert result.notes


def test_explicit_fix_request_creates_task_and_optional_note():
    events = [
        _event("e-state", EventKind.STATE, "Connection is insecure.", 0, "Please fix the insecure connection tomorrow.", object="connection"),
        _event(
            "e-req",
            EventKind.REQUEST,
            "Please fix the insecure connection tomorrow.",
            0,
            "Please fix the insecure connection tomorrow.",
            object="insecure connection",
            timeExpression="tomorrow",
            entities=["connection"],
        ),
    ]
    result = asyncio.run(
        run_event_pipeline(
            [_chunk(0, "Please fix the insecure connection tomorrow.")],
            "conv",
            "u",
            "s",
            event_extractor=ScriptedEventExtractor(events=events),
            embedder=_embedder(),
        )
    )
    assert result.tasks
    assert any("fix" in task.title.casefold() or "insecure" in task.title.casefold() for task in result.tasks)


def test_server_id_create_becomes_specific_task():
    result = asyncio.run(
        run_event_pipeline(
            [_chunk(0, "Server ID create.")],
            "conv",
            "u",
            "s",
            event_extractor=ScriptedEventExtractor(
                events=[_event("e1", EventKind.REQUEST, "Create server ID.", 0, "Server ID create.", object="server ID", entities=["Server", "ID"])]
            ),
            embedder=_embedder(),
        )
    )
    assert result.tasks
    assert any("server" in task.title.casefold() and "id" in task.title.casefold() for task in result.tasks)


def test_create_server_id_does_not_attach_unrelated_evidence():
    events = [
        _event("e-server", EventKind.REQUEST, "Create server ID.", 110, "Server ID create karna hai", object="server ID", entities=["Server", "ID"]),
        _event("e-conn", EventKind.ISSUE, "Connection string missing.", 60, "database server connection string missing hai", object="connection string", entities=["database"]),
        _event("e-net", EventKind.ISSUE, "Network error on staging.", 61, "network error", object="network error", entities=["network"]),
        _event("e-pnb", EventKind.FACT, "PNB setup is a separate topic.", 63, "PNB setup alag topic hai", object="PNB setup", entities=["PNB"]),
        _event("e-port", EventKind.STATE, "Port tracking is pending.", 64, "port tracking pending", object="port tracking", entities=["port"]),
    ]
    chunks = [
        _chunk(60, "database server connection string missing hai"),
        _chunk(61, "network error"),
        _chunk(63, "PNB setup alag topic hai"),
        _chunk(64, "port tracking pending"),
        _chunk(110, "Server ID create karna hai"),
    ]
    result = asyncio.run(run_event_pipeline(chunks, "conv", "u", "s", event_extractor=ScriptedEventExtractor(events=events), embedder=_embedder()))
    server_task = next(task for task in result.tasks if "server" in task.title.casefold())
    evidence_sequences = {span.sequenceStart for span in server_task.evidence}
    assert evidence_sequences.isdisjoint({60, 61, 63, 64})


def test_generic_tasks_are_rejected():
    assert is_generic_task_text("Complete pending task")
    assert is_generic_task_text("Fix issue")
    assert is_generic_task_text("Handle problem")
    assert is_generic_task_text("Do this")
    assert is_generic_task_text("Check it")
    assert is_generic_task_text("Resolve pending work")
    result = asyncio.run(
        run_event_pipeline(
            [_chunk(0, "Complete pending task"), _chunk(1, "Fix issue")],
            "conv",
            "u",
            "s",
            event_extractor=ScriptedEventExtractor(
                events=[
                    _event("e1", EventKind.REQUEST, "Complete pending task", 0, "Complete pending task"),
                    _event("e2", EventKind.REQUEST, "Fix issue", 1, "Fix issue"),
                ]
            ),
            embedder=_embedder(),
        )
    )
    assert result.tasks == []


def test_memory_events_cannot_result_in_zero_notes():
    events = [
        _event("e1", EventKind.STATE, "Old keys are currently in use.", 0, "old keys are currently in use", object="old keys"),
        _event("e2", EventKind.STATE, "Connection is insecure.", 1, "connection is insecure", object="connection"),
        _event("e3", EventKind.PROPOSAL, "Pricing was discussed.", 2, "pricing discussed", object="pricing", entities=["Pricing"]),
        _event("e4", EventKind.ISSUE, "Play Store issue exists.", 3, "Play Store issue exists", object="Play Store issue", entities=["Play", "Store"]),
    ]
    actions, memory, _ = split_action_and_memory(events)
    assert actions == []
    assert len(memory) == 4
    result = asyncio.run(
        run_event_pipeline(
            [_chunk(0, "old keys are currently in use"), _chunk(1, "connection is insecure"), _chunk(2, "pricing discussed"), _chunk(3, "Play Store issue exists")],
            "conv",
            "u",
            "s",
            event_extractor=ScriptedEventExtractor(events=events),
            embedder=_embedder(),
        )
    )
    assert result.notes
    assert result.coverage
    assert result.coverage.unaccounted_blocks == 0
    assert "memory_events_without_notes" not in result.coverage.suspicious or result.notes


def test_validator_removes_unrelated_evidence():
    task = ExtractedTask(
        title="Create server ID",
        body="Create the server ID as requested.",
        operation="CREATE",
        confidence=0.8,
        sourceConversationId="conv",
        evidence=[
            EvidenceSpan(sequenceStart=110, sequenceEnd=110, text="Server ID create karna hai"),
            EvidenceSpan(sequenceStart=60, sequenceEnd=60, text="database server connection string missing hai"),
        ],
        changes={"sourceSemanticUnitIds": ["e-server"]},
    )
    sequence_text = {
        110: "Server ID create karna hai",
        60: "database server connection string missing hai",
    }
    events = [
        _event("e-server", EventKind.REQUEST, "Create server ID.", 110, "Server ID create karna hai", object="server ID", entities=["Server", "ID"]),
    ]
    result = validate_artifact(task, sequence_text, events, artifact_kind="task")
    sequences = {span.sequenceStart for span in result.item.evidence}
    assert 110 in sequences
    assert 60 not in sequences
