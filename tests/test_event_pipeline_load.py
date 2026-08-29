import asyncio
import time

from services.conversation.event_pipeline.embeddings import CachedEmbedder, LexicalEmbedder
from services.conversation.event_pipeline.events import ScriptedEventExtractor
from services.conversation.event_pipeline.pipeline import run_event_pipeline
from services.conversation.event_pipeline.schemas import AtomicEvent, EventKind
from services.conversation.models import EvidenceSpan, STTStatus, TranscriptChunkDocument


def _chunks(count: int, sparse: bool = False) -> list[TranscriptChunkDocument]:
    items = []
    for index in range(count):
        if sparse and index % 17 == 0:
            text = ""
        elif index % 40 == 0:
            text = f"S3 frontend still failing at minute {index}"
        elif index % 40 == 1:
            text = f"Please create server ID {index}"
        elif index % 40 == 2:
            text = f"Pricing around 200 was discussed {index}"
        else:
            text = f"filler speech chunk {index} about hallway status"
        items.append(
            TranscriptChunkDocument(
                conversationId="load",
                userId="u",
                spaceId="s",
                chunkId=f"chunk_{index}",
                sequenceNumber=index,
                rawText=text,
                sttStatus=STTStatus.COMPLETED,
            )
        )
    return items


def _extractor(chunks: list[TranscriptChunkDocument]) -> ScriptedEventExtractor:
    events = []
    for chunk in chunks:
        text = (chunk.rawText or "").strip()
        if not text or text.startswith("filler"):
            continue
        kind = EventKind.REQUEST if "Please" in text or "create" in text.casefold() else EventKind.ISSUE if "failing" in text else EventKind.FACT
        obj = "S3" if "S3" in text else "server ID" if "server" in text.casefold() else "pricing" if "Pricing" in text else text[:40]
        entities = ["S3"] if "S3" in text else ["Server", "ID"] if "server" in text.casefold() else ["Pricing"] if "Pricing" in text else []
        events.append(
            AtomicEvent(
                eventId=f"e-{chunk.sequenceNumber}",
                topicId="T",
                kind=kind,
                meaning=text,
                object=obj,
                entities=entities,
                evidence=[EvidenceSpan(sequenceStart=chunk.sequenceNumber, sequenceEnd=chunk.sequenceNumber, text=text)],
                sequenceIds=[chunk.sequenceNumber],
                conversationId="load",
                userId="u",
                spaceId="s",
            )
        )
    return ScriptedEventExtractor(events=events)


def _run(count: int, sparse: bool = False) -> dict:
    chunks = _chunks(count, sparse=sparse)
    started = time.perf_counter()
    result = asyncio.run(
        run_event_pipeline(
            chunks,
            "load",
            "u",
            "s",
            event_extractor=_extractor(chunks),
            embedder=CachedEmbedder(LexicalEmbedder()),
        )
    )
    elapsed = time.perf_counter() - started
    return {
        "chunks": count,
        "runtimeSec": round(elapsed, 4),
        "llmCalls": result.observability.llm_calls(),
        "embeddingCalls": result.observability.embedding_calls(),
        "tokens": result.observability.tokens(),
        "comparisons": result.observability.comparisonCount,
        "microBlocks": len(result.microBlocks),
        "topics": len(result.topics),
        "events": len(result.events),
        "tasks": len(result.tasks),
        "notes": len(result.notes),
        "unaccounted": result.coverage.unaccounted_blocks if result.coverage else None,
        "asyncLifecycleErrors": result.observability.asyncLifecycleErrors,
    }


def test_concurrent_meetings_do_not_raise_async_lifecycle_errors():
    async def run_one(index: int):
        chunks = _chunks(50)
        return await run_event_pipeline(
            chunks,
            f"load-{index}",
            "u",
            "s",
            event_extractor=_extractor(chunks),
            embedder=CachedEmbedder(LexicalEmbedder()),
        )

    async def run_many(count: int):
        return await asyncio.gather(*(run_one(index) for index in range(count)))

    for count in (10, 25, 50):
        results = asyncio.run(run_many(count))
        assert len(results) == count
        for result in results:
            assert result.observability.asyncLifecycleErrors == 0
            assert result.coverage is None or result.coverage.unaccounted_blocks == 0


def test_soak_scripted_pipeline_stays_lifecycle_clean():
    reports = []
    for size in (50, 50, 150, 150, 50):
        reports.append(_run(size))
    for extra in range(45):
        reports.append(_run(50))
    assert len(reports) >= 50
    for report in reports:
        assert report["unaccounted"] == 0
        assert report["asyncLifecycleErrors"] == 0


def test_load_does_not_degrade_quadratically():
    reports = {
        50: _run(50),
        200: _run(200),
        300: _run(300),
        500: _run(500),
    }
    sparse = _run(240, sparse=True)
    print("EVENT_PIPELINE_LOAD", reports, "sparse8h", sparse)
    # Top-k retrieval: comparisons should grow near-linear with event count, not n^2 on raw chunks.
    assert reports[500]["comparisons"] <= reports[50]["comparisons"] * 40 + 5000
    assert reports[500]["runtimeSec"] < 30
    for report in reports.values():
        assert report["unaccounted"] == 0
        assert report["llmCalls"] == 0
        assert report["asyncLifecycleErrors"] == 0
