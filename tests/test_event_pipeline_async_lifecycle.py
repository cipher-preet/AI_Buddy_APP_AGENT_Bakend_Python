"""Runtime ownership, restart, and parallel-job safety for the event pipeline."""

from __future__ import annotations

import asyncio

import pytest

from services.conversation.event_pipeline.embeddings import CachedEmbedder, LexicalEmbedder
from services.conversation.event_pipeline.events import ScriptedEventExtractor
from services.conversation.event_pipeline.pipeline import run_event_pipeline
from services.conversation.event_pipeline.schemas import AtomicEvent, EventKind
from services.conversation.event_pipeline.store import ConversationEventStore
from services.conversation.models import EvidenceSpan, STTStatus, TranscriptChunkDocument
from services.llm.async_runtime import LoopBoundAsyncClient, current_loop_id, is_async_lifecycle_error
from services.llm.errors import AsyncLifecycleError, LLMProviderError
from services.llm.fallback import FallbackLLMProvider, LLMRouteCandidate
from services.llm.openai_compatible import OpenAICompatibleProvider
from services.llm.schema_adapter import ASYNC_LIFECYCLE_ERROR, classify_failure_class, classify_llm_failure


def _chunk(conversation_id: str, sequence: int, text: str) -> TranscriptChunkDocument:
    return TranscriptChunkDocument(
        conversationId=conversation_id,
        userId=f"user-{conversation_id}",
        spaceId="space",
        chunkId=f"{conversation_id}_{sequence}",
        sequenceNumber=sequence,
        rawText=text,
        sttStatus=STTStatus.COMPLETED,
    )


def _event(conversation_id: str, event_id: str, text: str) -> AtomicEvent:
    return AtomicEvent(
        eventId=event_id,
        topicId="T1",
        kind=EventKind.REQUEST,
        meaning=text,
        object="server ID",
        entities=["Server", "ID"],
        evidence=[EvidenceSpan(sequenceStart=0, sequenceEnd=0, text=text)],
        sequenceIds=[0],
        conversationId=conversation_id,
        userId=f"user-{conversation_id}",
        spaceId="space",
    )


def test_http_client_is_recreated_when_event_loop_changes():
    provider = OpenAICompatibleProvider(
        name="krutrim",
        api_key="test",
        base_url="http://localhost",
        default_model="gemma-4-31b-it",
        timeout_seconds=1,
        max_retries=0,
        max_concurrency=2,
    )

    async def use():
        client = await provider._http_client()
        return id(client), current_loop_id(), provider.transport_debug()

    first_id, first_loop, first_debug = asyncio.run(use())
    second_id, second_loop, second_debug = asyncio.run(use())
    assert first_loop != second_loop
    assert first_id != second_id
    assert first_debug["client_id"] != second_debug["client_id"]
    assert second_debug["closed"] is False
    asyncio.run(provider.aclose())


def test_stale_transport_replace_does_not_reuse_closed_client():
    created = []

    class FakeClient:
        def __init__(self):
            self.is_closed = False
            created.append(self)

        async def aclose(self):
            self.is_closed = True

    transport = LoopBoundAsyncClient(FakeClient)

    async def first():
        client = await transport.get()
        return id(client)

    async def second():
        assert transport.stale()
        client = await transport.get()
        return id(client), transport.debug()

    first_id = asyncio.run(first())
    second_id, debug = asyncio.run(second())
    assert first_id != second_id
    assert len(created) == 2
    assert debug["created_loop_id"] == debug["current_loop_id"]


def test_lifecycle_error_is_not_ordinary_provider_fallback():
    class ClosedLoopProvider:
        name = "krutrim"
        configured = True
        calls = 0

        async def generate_structured(self, request, schema):
            self.calls += 1
            raise RuntimeError("Event loop is closed")

    class BackupProvider:
        name = "mistral"
        configured = True
        calls = 0

        async def generate_structured(self, request, schema):
            self.calls += 1
            return schema()

    closed = ClosedLoopProvider()
    backup = BackupProvider()
    wrapper = FallbackLLMProvider(
        "krutrim",
        [
            LLMRouteCandidate(provider=closed, model="gemma-4-31b-it"),  # type: ignore[arg-type]
            LLMRouteCandidate(provider=backup, model="ministral-14b-latest"),  # type: ignore[arg-type]
        ],
    )
    with pytest.raises(RuntimeError, match="Event loop is closed"):
        asyncio.run(wrapper.generate_structured(_empty_request(), _EmptySchema))
    assert closed.calls == 1
    assert backup.calls == 0


def test_classify_async_lifecycle_error():
    error = RuntimeError("Event loop is closed")
    assert is_async_lifecycle_error(error)
    assert classify_llm_failure(error) == ASYNC_LIFECYCLE_ERROR
    assert classify_failure_class(error) == ASYNC_LIFECYCLE_ERROR
    typed = AsyncLifecycleError("closed")
    assert classify_failure_class(typed) == ASYNC_LIFECYCLE_ERROR
    timeout = LLMProviderError("timed out", retryable=True, failure_reason="PROVIDER_TIMEOUT")
    assert classify_failure_class(timeout) == "TIMEOUT"
    rate = LLMProviderError("slow down", retryable=True, status_code=429, failure_reason="RATE_LIMITED")
    assert classify_failure_class(rate) == "RATE_LIMIT"


def test_worker_restart_does_not_reuse_pipeline_state():
    chunks = [_chunk("restart", 0, "Please create the server ID.")]
    extractor = ScriptedEventExtractor(events=[_event("restart", "e-server", "Please create the server ID.")])

    async def run_once(store: ConversationEventStore):
        return await run_event_pipeline(
            chunks,
            "restart",
            "user-restart",
            "space",
            event_extractor=extractor,
            event_store=store,
            embedder=CachedEmbedder(LexicalEmbedder()),
        )

    first_store = ConversationEventStore()
    first = asyncio.run(run_once(first_store))
    second_store = ConversationEventStore()
    second = asyncio.run(run_once(second_store))
    assert {event.eventId for event in first.events} == {event.eventId for event in second.events}
    assert first.observability.asyncLifecycleErrors == 0
    assert second.observability.asyncLifecycleErrors == 0


def test_parallel_sessions_do_not_mix_evidence():
    async def run_one(name: str):
        chunks = [_chunk(name, 0, f"Please create the server ID for {name}.")]
        return await run_event_pipeline(
            chunks,
            name,
            f"user-{name}",
            "space",
            event_extractor=ScriptedEventExtractor(events=[_event(name, f"e-{name}", f"Please create the server ID for {name}.")]),
            event_store=ConversationEventStore(),
            embedder=CachedEmbedder(LexicalEmbedder()),
        )

    async def run_all():
        return await asyncio.gather(run_one("A"), run_one("B"), run_one("C"))

    results = asyncio.run(run_all())
    ids = [{event.conversationId for event in result.events} for result in results]
    assert ids == [{"A"}, {"B"}, {"C"}]
    for result in results:
        assert result.observability.asyncLifecycleErrors == 0
        assert result.coverage
        assert result.coverage.unaccounted_blocks == 0


class _EmptySchema:
    def __init__(self, **kwargs):
        pass


def _empty_request():
    from services.llm.models import LLMMessage, StructuredLLMRequest

    return StructuredLLMRequest(
        model="gemma-4-31b-it",
        schema_name="Empty",
        messages=[LLMMessage(role="user", content="x")],
    )
