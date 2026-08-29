import asyncio

from services.conversation.event_pipeline.embeddings import CachedEmbedder, LexicalEmbedder
from services.conversation.event_pipeline.events import LLMEventExtractor, ScriptedEventExtractor
from services.conversation.event_pipeline.pipeline import run_event_pipeline
from services.conversation.event_pipeline.schemas import AtomicEvent, EventKind
from services.conversation.event_pipeline.store import ConversationEventStore
from services.conversation.models import EvidenceSpan, STTStatus, TranscriptChunkDocument
from services.llm.errors import LLMProviderError, StructuredOutputError
from services.llm.schema_adapter import (
    ASYNC_LIFECYCLE_ERROR,
    HTTP_ERROR,
    MALFORMED_JSON,
    PROVIDER_TIMEOUT,
    RATE_LIMITED,
    classify_failure_class,
    classify_llm_failure,
)


class _BoomEmbedder:
    async def embed_many(self, texts):
        raise RuntimeError("embedding provider failure")


class _MalformedRouter:
    def route(self, capability):
        return self, "model"

    async def generate_structured(self, request, schema):
        raise ValueError("malformed model JSON")


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


def test_embedding_provider_failure_falls_back_to_lexical():
    cached = CachedEmbedder(_BoomEmbedder())
    vectors = asyncio.run(cached.embed_many(["S3 frontend", "pricing plan"]))
    assert len(vectors) == 2
    assert len(vectors[0]) >= 8


def test_extractor_timeout_is_accounted_not_silent():
    class Adapter:
        calls = 0

        async def extract(self, topic, blocks, sequence_text):
            self.calls += 1
            return await LLMEventExtractor(_MalformedRouter()).extract(topic, blocks, sequence_text)  # type: ignore[arg-type]

    result = asyncio.run(
        run_event_pipeline(
            [_chunk(0, "S3 is failing and we will discuss pricing later")],
            "conv",
            "u",
            "s",
            event_extractor=Adapter(),
            embedder=CachedEmbedder(LexicalEmbedder()),
        )
    )
    assert result.coverage
    assert result.coverage.unaccounted_blocks == 0
    assert result.events


def test_duplicate_queue_delivery_does_not_duplicate_events():
    event = AtomicEvent(
        eventId="stable",
        topicId="T1",
        kind=EventKind.REQUEST,
        meaning="Create server ID.",
        object="server ID",
        entities=["Server", "ID"],
        evidence=[EvidenceSpan(sequenceStart=0, sequenceEnd=0, text="Server ID create")],
        sequenceIds=[0],
        conversationId="conv",
        userId="u",
        spaceId="s",
    )
    store = ConversationEventStore()
    extractor = ScriptedEventExtractor(events=[event])
    chunks = [_chunk(0, "Server ID create")]
    first = asyncio.run(run_event_pipeline(chunks, "conv", "u", "s", event_extractor=extractor, event_store=store, embedder=CachedEmbedder(LexicalEmbedder())))
    second = asyncio.run(run_event_pipeline(chunks, "conv", "u", "s", event_extractor=extractor, event_store=store, embedder=CachedEmbedder(LexicalEmbedder())))
    assert {item.eventId for item in first.events} == {item.eventId for item in second.events}
    assert {task.fingerprint for task in first.tasks} == {task.fingerprint for task in second.tasks}


def test_partial_checkpoint_merges_with_leftover():
    checkpoint = AtomicEvent(
        eventId="chk-s3",
        topicId="T1",
        kind=EventKind.ISSUE,
        meaning="S3 has a problem.",
        object="S3",
        entities=["S3"],
        evidence=[EvidenceSpan(sequenceStart=10, sequenceEnd=10, text="S3 has a problem.")],
        sequenceIds=[10],
        microBlockIds=["MB-chk"],
        conversationId="conv",
        userId="u",
        spaceId="s",
    )
    leftover = AtomicEvent(
        eventId="left-s3",
        topicId="T2",
        kind=EventKind.STATE,
        meaning="S3 still does not reach frontend.",
        object="S3 frontend",
        entities=["S3", "frontend"],
        evidence=[EvidenceSpan(sequenceStart=40, sequenceEnd=40, text="S3 still does not reach frontend.")],
        sequenceIds=[40],
        conversationId="conv",
        userId="u",
        spaceId="s",
    )
    chunks = [_chunk(10, "S3 has a problem."), _chunk(40, "S3 still does not reach frontend.")]
    result = asyncio.run(
        run_event_pipeline(
            chunks,
            "conv",
            "u",
            "s",
            checkpoint_events=[checkpoint],
            event_extractor=ScriptedEventExtractor(events=[leftover]),
            embedder=CachedEmbedder(LexicalEmbedder()),
        )
    )
    s3 = [event for event in result.events if event.eventId in {"chk-s3", "left-s3"}]
    assert len(s3) == 2
    assert len({event.threadId for event in s3}) == 1


def test_cached_embedder_falls_back_when_inner_provider_fails():
    cached = CachedEmbedder(_BoomEmbedder())
    vectors = asyncio.run(cached.embed_many(["S3 frontend"]))
    assert vectors and vectors[0]


def test_overlap_window_does_not_duplicate_canonical_events():
    first = AtomicEvent(
        eventId="e-s3-window1",
        topicId="T1",
        kind=EventKind.ISSUE,
        meaning="S3 is not reaching the frontend.",
        object="S3 frontend",
        entities=["S3"],
        evidence=[EvidenceSpan(sequenceStart=20, sequenceEnd=20, text="S3 is not reaching frontend")],
        sequenceIds=[20],
        conversationId="conv",
        userId="u",
        spaceId="s",
    )
    overlap = AtomicEvent(
        eventId="e-s3-window2",
        topicId="T2",
        kind=EventKind.ISSUE,
        meaning="S3 is not reaching the frontend.",
        object="S3 frontend",
        entities=["S3"],
        evidence=[EvidenceSpan(sequenceStart=20, sequenceEnd=21, text="S3 is not reaching frontend")],
        sequenceIds=[20, 21],
        conversationId="conv",
        userId="u",
        spaceId="s",
    )
    store = ConversationEventStore()
    asyncio.run(store.upsert("conv", [first]))
    merged = asyncio.run(store.upsert("conv", [overlap]))
    s3 = [event for event in merged if "s3" in event.meaning.casefold()]
    assert len(s3) == 1
    assert set(s3[0].sequenceIds) >= {20}


def test_validator_failure_does_not_drop_grounded_artifacts():
    from services.conversation.event_pipeline.validation import LLMArtifactValidator

    class BoomRouter:
        def route(self, capability):
            return self, "gpt-oss-20b"

        async def generate_structured(self, request, schema):
            raise TimeoutError("validator timeout")

    event = AtomicEvent(
        eventId="e-server",
        topicId="T1",
        kind=EventKind.REQUEST,
        meaning="Create server ID.",
        object="server ID",
        entities=["Server", "ID"],
        evidence=[EvidenceSpan(sequenceStart=0, sequenceEnd=0, text="Please create the server ID.")],
        sequenceIds=[0],
        conversationId="conv",
        userId="u",
        spaceId="s",
    )
    result = asyncio.run(
        run_event_pipeline(
            [_chunk(0, "Please create the server ID.")],
            "conv",
            "u",
            "s",
            router=BoomRouter(),  # type: ignore[arg-type]
            event_extractor=ScriptedEventExtractor(events=[event]),
            embedder=CachedEmbedder(LexicalEmbedder()),
            polish_with_llm=True,
        )
    )
    assert result.tasks
    _ = LLMArtifactValidator


def test_synthesis_provider_failure_falls_back_to_deterministic():
    class BoomRouter:
        def route(self, capability):
            return self, "gpt-oss-120b"

        async def generate_structured(self, request, schema):
            raise TimeoutError("gpt-oss-120b timeout")

    event = AtomicEvent(
        eventId="e-server",
        topicId="T1",
        kind=EventKind.REQUEST,
        meaning="Create server ID.",
        object="server ID",
        entities=["Server", "ID"],
        evidence=[EvidenceSpan(sequenceStart=0, sequenceEnd=0, text="Please create the server ID.")],
        sequenceIds=[0],
        conversationId="conv",
        userId="u",
        spaceId="s",
    )
    result = asyncio.run(
        run_event_pipeline(
            [_chunk(0, "Please create the server ID.")],
            "conv",
            "u",
            "s",
            router=BoomRouter(),  # type: ignore[arg-type]
            event_extractor=ScriptedEventExtractor(events=[event]),
            embedder=CachedEmbedder(LexicalEmbedder()),
            polish_with_llm=True,
        )
    )
    assert {task.title for task in result.tasks}
    assert result.coverage
    assert result.coverage.unaccounted_blocks == 0


def test_checkpoint_replay_is_idempotent():
    checkpoint = AtomicEvent(
        eventId="chk-s3",
        topicId="T1",
        kind=EventKind.ISSUE,
        meaning="S3 has a problem.",
        object="S3",
        entities=["S3"],
        evidence=[EvidenceSpan(sequenceStart=10, sequenceEnd=10, text="S3 has a problem.")],
        sequenceIds=[10],
        conversationId="conv",
        userId="u",
        spaceId="s",
    )
    leftover = AtomicEvent(
        eventId="left-price",
        topicId="T2",
        kind=EventKind.PROPOSAL,
        meaning="Pricing should start around 200.",
        object="pricing",
        entities=["Pricing"],
        evidence=[EvidenceSpan(sequenceStart=40, sequenceEnd=40, text="Pricing should start around 200.")],
        sequenceIds=[40],
        conversationId="conv",
        userId="u",
        spaceId="s",
    )
    store = ConversationEventStore()
    chunks = [_chunk(10, "S3 has a problem."), _chunk(40, "Pricing should start around 200.")]
    first = asyncio.run(
        run_event_pipeline(
            chunks,
            "conv",
            "u",
            "s",
            checkpoint_events=[checkpoint],
            event_extractor=ScriptedEventExtractor(events=[leftover]),
            event_store=store,
            embedder=CachedEmbedder(LexicalEmbedder()),
        )
    )
    second = asyncio.run(
        run_event_pipeline(
            chunks,
            "conv",
            "u",
            "s",
            checkpoint_events=[checkpoint],
            event_extractor=ScriptedEventExtractor(events=[leftover]),
            event_store=store,
            embedder=CachedEmbedder(LexicalEmbedder()),
        )
    )
    assert {event.eventId for event in first.events} == {event.eventId for event in second.events}
    assert {task.fingerprint for task in first.tasks} == {task.fingerprint for task in second.tasks}
    assert {note.fingerprint for note in first.notes} == {note.fingerprint for note in second.notes}


def test_failure_classes_are_differentiated():
    assert classify_failure_class(LLMProviderError("timeout", retryable=True, failure_reason=PROVIDER_TIMEOUT)) == "TIMEOUT"
    assert classify_failure_class(LLMProviderError("rate", retryable=True, status_code=429, failure_reason=RATE_LIMITED)) == "RATE_LIMIT"
    assert classify_failure_class(LLMProviderError("reset", retryable=True, status_code=502, failure_reason=HTTP_ERROR)) == "PROVIDER_FAILURE"
    assert classify_failure_class(StructuredOutputError(MALFORMED_JSON, "bad json")) == "STRUCTURED_OUTPUT_FAILURE"
    assert classify_llm_failure(RuntimeError("Event loop is closed")) == ASYNC_LIFECYCLE_ERROR
    assert classify_failure_class(RuntimeError("Event loop is closed")) == ASYNC_LIFECYCLE_ERROR
