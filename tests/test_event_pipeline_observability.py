import asyncio

from services.conversation.event_pipeline.alerts import evaluate_alerts
from services.conversation.event_pipeline.budget import PipelineBudget, PipelineBudgetExceeded
from services.conversation.event_pipeline.embeddings import CachedEmbedder, LexicalEmbedder
from services.conversation.event_pipeline.events import ScriptedEventExtractor
from services.conversation.event_pipeline.observability import job_record
from services.conversation.event_pipeline.pipeline import run_event_pipeline
from services.conversation.event_pipeline.publish_gate import publication_ready
from services.conversation.event_pipeline.schemas import AtomicEvent, EventKind, PipelineObservability
from services.conversation.models import EvidenceSpan, STTStatus, TranscriptChunkDocument


def _chunk(text: str = "Please create the server ID.") -> TranscriptChunkDocument:
    return TranscriptChunkDocument(
        conversationId="conv",
        userId="u",
        spaceId="s",
        chunkId="chunk_0",
        sequenceNumber=0,
        rawText=text,
        sttStatus=STTStatus.COMPLETED,
    )


def _event() -> AtomicEvent:
    return AtomicEvent(
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


def test_job_record_has_production_counters_without_transcript():
    result = asyncio.run(
        run_event_pipeline(
            [_chunk()],
            "conv",
            "u",
            "s",
            event_extractor=ScriptedEventExtractor(events=[_event()]),
            embedder=CachedEmbedder(LexicalEmbedder()),
        )
    )
    record = job_record(result.observability)
    assert record["sessionId"] == "conv"
    assert record["workerId"]
    assert "Please create" not in str(record)
    assert "asyncLifecycleErrors" in record
    assert record["asyncLifecycleErrors"] == 0
    assert "embeddingCalls" in record
    assert "GemmaCalls" in record
    assert result.diagnostics.get("pipelineVersion")
    assert result.diagnostics.get("eventSchemaVersion")
    assert record.get("pipelineMode") in {"event_pipeline", "legacy", "shadow", ""}


def test_publication_ready_blocks_unaccounted_and_lifecycle():
    result = asyncio.run(
        run_event_pipeline(
            [_chunk()],
            "conv",
            "u",
            "s",
            event_extractor=ScriptedEventExtractor(events=[_event()]),
            embedder=CachedEmbedder(LexicalEmbedder()),
        )
    )
    ok, reason = publication_ready(result)
    assert ok, reason
    result.observability.asyncLifecycleErrors = 1
    ok, reason = publication_ready(result)
    assert not ok
    assert reason == "async_lifecycle_error"


def test_budget_exceeded_fails_safely():
    budget = PipelineBudget(max_runtime=0.0001, max_model_calls=1, max_retries=0)
    budget.started -= 1
    try:
        budget.check()
        raised = False
    except PipelineBudgetExceeded:
        raised = True
    assert raised


def test_async_lifecycle_alert_fires():
    obs = PipelineObservability(asyncLifecycleErrors=1)
    alerts = evaluate_alerts(observability=obs, coverage=None, events=[], tasks=[], notes=[])
    names = {item["name"] for item in alerts}
    assert "ASYNC_LIFECYCLE_ERROR" in names
