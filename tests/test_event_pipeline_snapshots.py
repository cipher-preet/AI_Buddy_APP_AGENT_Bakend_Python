import asyncio
import json
from pathlib import Path

from services.conversation.event_pipeline.embeddings import CachedEmbedder, LexicalEmbedder
from services.conversation.event_pipeline.events import ScriptedEventExtractor
from services.conversation.event_pipeline.pipeline import run_event_pipeline
from services.conversation.event_pipeline.snapshots import format_traces as snapshot_traces
from tests.fixtures.long_meeting_gold import build_gold_transcript


def test_pipeline_builds_stage_snapshots_and_traces():
    gold = build_gold_transcript()
    result = asyncio.run(
        run_event_pipeline(
            gold["chunks"],
            "gold-long-meeting",
            "user_1",
            "space_1",
            event_extractor=ScriptedEventExtractor(events=gold["events"]),
            embedder=CachedEmbedder(LexicalEmbedder()),
        )
    )
    snapshots = result.snapshots
    for key in (
        "CLEANED_SEQUENCES",
        "MICRO_BLOCKS",
        "TOPICS",
        "ATOMIC_EVENTS",
        "GLOBAL_THREADS",
        "ACTION_EVENTS",
        "MEMORY_EVENTS",
        "TASK_CANDIDATES",
        "NOTE_CANDIDATES",
        "VALIDATED_ARTIFACTS",
        "COVERAGE_LEDGER",
        "TRACES",
    ):
        assert key in snapshots, key
    traces = snapshot_traces(snapshots)
    assert "sequence" in traces
    assert "event" in traces
    server_trace = next((item for item in snapshots["TRACES"] if "server" in (item.get("title") or "").casefold()), None)
    assert server_trace is not None
    assert server_trace.get("sequence") in {110, 111}


def test_debug_inspect_prints_sequence_path(tmp_path: Path):
    gold = build_gold_transcript()
    result = asyncio.run(
        run_event_pipeline(
            gold["chunks"],
            "gold-long-meeting",
            "user_1",
            "space_1",
            event_extractor=ScriptedEventExtractor(events=gold["events"]),
            embedder=CachedEmbedder(LexicalEmbedder()),
        )
    )
    path = tmp_path / "snap.json"
    path.write_text(json.dumps(result.snapshots, default=str), encoding="utf-8")
    from services.conversation.event_pipeline.debug_inspect import main

    assert main([str(path), "--sequence", "110"]) == 0
