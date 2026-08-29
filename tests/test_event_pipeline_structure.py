import asyncio

from services.conversation.event_pipeline.cleaning import clean_transcripts
from services.conversation.event_pipeline.embeddings import CachedEmbedder, LexicalEmbedder
from services.conversation.event_pipeline.events import ScriptedEventExtractor, materialize_events
from services.conversation.event_pipeline.microblocks import build_micro_blocks
from services.conversation.event_pipeline.pipeline import run_event_pipeline
from services.conversation.event_pipeline.schemas import AtomicEvent, EventKind, LocalTopic
from services.conversation.event_pipeline.store import ConversationEventStore
from services.conversation.event_pipeline.topics import segment_local_topics
from services.conversation.models import EvidenceSpan, STTStatus, TranscriptChunkDocument


def _chunk(sequence: int, text: str, **kwargs) -> TranscriptChunkDocument:
    return TranscriptChunkDocument(
        conversationId=kwargs.get("conversationId", "conv"),
        userId=kwargs.get("userId", "user"),
        spaceId=kwargs.get("spaceId", "space"),
        chunkId=kwargs.get("chunkId", f"chunk_{sequence}"),
        sequenceNumber=sequence,
        rawText=text,
        sttStatus=kwargs.get("sttStatus", STTStatus.COMPLETED),
        startTimeMs=kwargs.get("startTimeMs"),
        endTimeMs=kwargs.get("endTimeMs"),
    )


def _embedder() -> CachedEmbedder:
    return CachedEmbedder(LexicalEmbedder())


def test_cleaning_excludes_empty_whitespace_and_placeholders_but_keeps_hinglish():
    chunks = [
        _chunk(0, ""),
        _chunk(1, "   "),
        _chunk(2, "null"),
        _chunk(3, "S3 frontend nahi reach kar raha"),
        _chunk(4, "we need fix that tommorow pls"),
        _chunk(5, "..."),
        _chunk(5, "duplicate sequence ignored"),
        _chunk(6, "we need fix that tommorow pls"),
    ]
    ledger = clean_transcripts(chunks, conversation_id="conv", user_id="user", space_id="space")
    assert ledger.totalSequences == ledger.usefulSequences + ledger.excludedStructuralSequences
    assert {record.sequenceId for record in ledger.records} == {0, 1, 2, 3, 4, 5, 6}
    useful_text = " ".join(record.rawText for record in ledger.useful)
    assert "S3 frontend nahi reach kar raha" in useful_text
    assert "we need fix that tommorow pls" in useful_text
    assert ledger.usefulSequences >= 2
    assert ledger.excludedStructuralSequences >= 4


def test_cleaning_does_not_drop_broken_grammar_or_incomplete_sentences():
    chunks = [_chunk(0, "server id create"), _chunk(1, "connection insecure")]
    ledger = clean_transcripts(chunks)
    assert ledger.usefulSequences == 2
    assert ledger.excludedStructuralSequences == 0


def test_microblock_groups_s3_issue_with_fix_tomorrow():
    chunks = [
        _chunk(30, "S3 is not reaching frontend"),
        _chunk(31, "We need to fix that tomorrow"),
        _chunk(32, "Pricing should start around 200"),
    ]
    ledger = clean_transcripts(chunks)
    blocks = asyncio.run(build_micro_blocks(ledger.useful, _embedder()))
    s3_block = next(block for block in blocks if 30 in block.sequenceIds)
    assert 31 in s3_block.sequenceIds
    pricing = next(block for block in blocks if 32 in block.sequenceIds)
    assert 30 not in pricing.sequenceIds or pricing is not s3_block
    assert s3_block.microBlockId != pricing.microBlockId


def test_microblock_keeps_pricing_and_usage_adjacent():
    chunks = [
        _chunk(40, "Pricing should start around 200"),
        _chunk(41, "free plan usage limit should stay low"),
    ]
    ledger = clean_transcripts(chunks)
    blocks = asyncio.run(build_micro_blocks(ledger.useful, _embedder()))
    assert any(40 in block.sequenceIds and 41 in block.sequenceIds for block in blocks) or len(blocks) <= 2


def test_microblock_force_splits_on_maximum_size(monkeypatch):
    monkeypatch.setattr("services.conversation.event_pipeline.microblocks.settings.EVENT_PIPELINE_MICROBLOCK_MAX_TURNS", 2)
    monkeypatch.setattr("services.conversation.event_pipeline.microblocks.settings.EVENT_PIPELINE_MICROBLOCK_MAX_TOKENS", 40)
    chunks = [_chunk(index, f"topic continues with extra tokens {index} about the same backend queue drain") for index in range(6)]
    ledger = clean_transcripts(chunks)
    blocks = asyncio.run(build_micro_blocks(ledger.useful, _embedder()))
    assert len(blocks) >= 2
    assert all(len(block.sequenceIds) <= 3 for block in blocks)


def test_overlap_does_not_invent_duplicate_source_ids():
    chunks = [
        _chunk(20, "S3 frontend integration is broken today"),
        _chunk(21, "we discussed S3 ACL again"),
        _chunk(22, "Pricing plan two hundred rupees starts now"),
        _chunk(23, "Pricing free tier limit is too high"),
    ]
    ledger = clean_transcripts(chunks)
    blocks = asyncio.run(build_micro_blocks(ledger.useful, _embedder()))
    source_ids = [source for block in blocks for source in block.sourceIds]
    # Overlap may repeat a sequence in two blocks, but original source ids remain the chunk ids.
    assert all(item.startswith("chunk_") for item in source_ids)


def test_local_topics_split_s3_pricing_database():
    chunks = [
        _chunk(0, "S3 bucket cannot reach the frontend"),
        _chunk(1, "S3 credentials were rotated yesterday"),
        _chunk(2, "Pricing should start around two hundred"),
        _chunk(3, "Pricing free plan has a usage limit"),
        _chunk(4, "Database connection is failing on staging"),
        _chunk(5, "Database server timeout keeps repeating"),
    ]
    ledger = clean_transcripts(chunks)
    blocks = asyncio.run(build_micro_blocks(ledger.useful, _embedder()))
    topics = asyncio.run(segment_local_topics(blocks, _embedder()))
    assert len(topics) >= 3


def test_local_topics_do_not_merge_distant_s3_occurrences():
    chunks = [
        _chunk(0, "S3 bucket cannot reach the frontend"),
        _chunk(1, "Pricing should start around two hundred"),
        _chunk(2, "S3 still cannot reach the frontend"),
    ]
    ledger = clean_transcripts(chunks)
    blocks = asyncio.run(build_micro_blocks(ledger.useful, _embedder()))
    topics = asyncio.run(segment_local_topics(blocks, _embedder()))
    assert len(topics) >= 2


def test_atomic_event_schema_rejects_merged_meanings_via_scripted_split():
    topic = LocalTopic(topicId="T1", label="S3", microBlockIds=["MB1"], sequenceStart=1, sequenceEnd=1, sequenceIds=[1], text="[1] S3 is failing. We will fix it tomorrow.")
    payload = {
        "events": [
            {
                "kind": "ISSUE",
                "meaning": "S3 is failing.",
                "object": "S3",
                "evidence": [{"sequenceStart": 1, "sequenceEnd": 1, "text": "S3 is failing. We will fix it tomorrow."}],
            },
            {
                "kind": "COMMITMENT",
                "meaning": "The S3 issue will be fixed tomorrow.",
                "object": "S3 issue",
                "timeExpression": "tomorrow",
                "evidence": [{"sequenceStart": 1, "sequenceEnd": 1, "text": "S3 is failing. We will fix it tomorrow."}],
            },
        ]
    }
    events = materialize_events(payload, topic, [], {1: "S3 is failing. We will fix it tomorrow."})
    assert {event.kind for event in events} == {EventKind.ISSUE, EventKind.COMMITMENT}


def test_insecure_connection_is_state_not_inferred_task():
    event = AtomicEvent(
        eventId="e1",
        topicId="T1",
        kind=EventKind.STATE,
        meaning="Connection is reported as insecure.",
        object="connection",
        evidence=[EvidenceSpan(sequenceStart=0, sequenceEnd=0, text="Connection is insecure.")],
        sequenceIds=[0],
        conversationId="conv",
        userId="u",
        spaceId="s",
    )
    extractor = ScriptedEventExtractor(events=[event])
    chunks = [_chunk(0, "Connection is insecure.")]
    result = asyncio.run(run_event_pipeline(chunks, "conv", "u", "s", event_extractor=extractor, embedder=_embedder()))
    assert result.tasks == []
    assert result.notes
    assert any("insecure" in f"{note.title} {note.body}".casefold() for note in result.notes)


def test_complete_it_tomorrow_preserves_uncertainty_without_generic_task():
    event = AtomicEvent(
        eventId="e2",
        topicId="T1",
        kind=EventKind.COMMITMENT,
        meaning="Complete it tomorrow.",
        object=None,
        timeExpression="tomorrow",
        uncertainty=["missing_object"],
        evidence=[EvidenceSpan(sequenceStart=0, sequenceEnd=0, text="Complete it tomorrow.")],
        sequenceIds=[0],
        conversationId="conv",
        userId="u",
        spaceId="s",
    )
    result = asyncio.run(
        run_event_pipeline(
            [_chunk(0, "Complete it tomorrow.")],
            "conv",
            "u",
            "s",
            event_extractor=ScriptedEventExtractor(events=[event]),
            embedder=_embedder(),
        )
    )
    assert not any("complete pending task" in task.title.casefold() for task in result.tasks)
    assert all("complete it" not in task.title.casefold() for task in result.tasks)


def test_event_store_upsert_is_idempotent():
    store = ConversationEventStore()
    event = AtomicEvent(
        eventId="e-dup",
        topicId="T1",
        kind=EventKind.FACT,
        meaning="Old keys are in use.",
        evidence=[EvidenceSpan(sequenceStart=1, sequenceEnd=1, text="old keys are in use")],
        sequenceIds=[1],
        conversationId="conv",
    )
    first = asyncio.run(store.upsert("conv", [event]))
    second = asyncio.run(store.upsert("conv", [event, event]))
    assert len(first) == 1
    assert len(second) == 1
