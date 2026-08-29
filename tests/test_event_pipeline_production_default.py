"""100% production default: EVENT_PIPELINE for every user, with instant rollback."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from apps.api_gateway.config.setting import settings
from services.conversation.event_pipeline.embeddings import CachedEmbedder, LexicalEmbedder
from services.conversation.event_pipeline.events import ScriptedEventExtractor
from services.conversation.event_pipeline.flags import (
    event_pipeline_mode,
    event_pipeline_publishes,
    event_pipeline_selected_for,
    event_pipeline_shadow,
    legacy_pipeline_publishes,
    legacy_pipeline_publishes_for,
)
from services.conversation.event_pipeline.gold_scoring import pipeline_benchmark
from services.conversation.event_pipeline.observability import job_record
from services.conversation.event_pipeline.pipeline import run_event_pipeline
from services.conversation.event_pipeline.publish_gate import EventPipelineHardFailure, publication_ready
from services.conversation.event_pipeline.schemas import ActionSignal, AtomicEvent, EventKind
from services.conversation.models import EvidenceSpan, STTStatus, TranscriptChunkDocument
from services.conversation.workflow import ConversationProcessingWorkflow
from tests.fixtures.generic_conversations import (
    casual_noise,
    code_switching,
    family_decision,
    personal_planning,
    study_status,
    travel_change,
    work_followup,
)
from tests.fixtures.long_meeting_gold import FILLERS, build_gold_transcript
from tests.fixtures.scale_meetings import TOPIC_SNIPPETS, build_scale_meeting


def _production_100(monkeypatch) -> None:
    monkeypatch.setattr(settings, "ENABLE_EVENT_PIPELINE", True)
    monkeypatch.setattr(settings, "EVENT_PIPELINE_MODE", "event_pipeline")
    monkeypatch.setattr(settings, "EVENT_PIPELINE_ROLLOUT_PERCENT", 100)


def _embedder() -> CachedEmbedder:
    return CachedEmbedder(LexicalEmbedder())


def _provenance(item) -> dict:
    return dict(getattr(item, "changes", None) or getattr(item, "debug", None) or {})


def _assert_event_pipeline_session(result) -> None:
    assert result.diagnostics.get("pipelineMode") == "event_pipeline"
    assert result.diagnostics.get("artifactPipelineVersion")
    assert result.diagnostics.get("eventSchemaVersion")
    assert result.diagnostics.get("promptVersion")
    record = job_record(result.observability)
    assert record["pipelineMode"] == "event_pipeline"
    assert record["asyncLifecycleErrors"] == 0
    assert "Please create" not in str(record)
    ok, reason = publication_ready(result)
    assert ok, reason
    for task in result.tasks:
        meta = _provenance(task)
        assert meta.get("pipelineMode") == "event_pipeline"
        assert meta.get("artifactPipelineVersion")
        assert meta.get("eventSchemaVersion")
        assert meta.get("promptVersion")
    for note in result.notes:
        meta = _provenance(note)
        assert meta.get("pipelineMode") == "event_pipeline"
        assert meta.get("artifactPipelineVersion")


def test_production_100_routes_every_user_to_event_pipeline(monkeypatch):
    _production_100(monkeypatch)
    assert event_pipeline_mode() == "event_pipeline"
    assert event_pipeline_publishes()
    assert not legacy_pipeline_publishes()
    assert not event_pipeline_shadow()
    users = [
        "user A",
        "user B",
        "user C",
        "user-1",
        "user-99",
        "arbitrary-uuid-aaaa",
        "arbitrary-uuid-zzzz",
        "preet",
        "anon",
    ]
    for user in users:
        assert event_pipeline_selected_for(user) is True, user
        assert legacy_pipeline_publishes_for(user) is False, user
    assert event_pipeline_selected_for(None, "session-x") is True
    assert event_pipeline_selected_for(None, None) is True


def test_rollout_zero_and_legacy_mode_are_instant_rollback(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_EVENT_PIPELINE", True)
    monkeypatch.setattr(settings, "EVENT_PIPELINE_MODE", "event_pipeline")
    monkeypatch.setattr(settings, "EVENT_PIPELINE_ROLLOUT_PERCENT", 0)
    assert event_pipeline_mode() == "event_pipeline"
    assert not event_pipeline_publishes()
    for user in ("user A", "user B", "user C"):
        assert event_pipeline_selected_for(user) is False
        assert legacy_pipeline_publishes_for(user) is True

    monkeypatch.setattr(settings, "EVENT_PIPELINE_MODE", "legacy")
    monkeypatch.setattr(settings, "EVENT_PIPELINE_ROLLOUT_PERCENT", 100)
    assert event_pipeline_mode() == "legacy"
    assert not event_pipeline_publishes()
    assert legacy_pipeline_publishes()
    for user in ("user A", "user B", "user C"):
        assert event_pipeline_selected_for(user) is False


def test_short_and_long_sessions_use_event_pipeline_at_100(monkeypatch):
    _production_100(monkeypatch)
    monkeypatch.setattr(settings, "ENABLE_MEETING_PIPELINE", False)
    used = []

    async def fake_event(self, conversation, run, windows, path: str):
        used.append(path)

    workflow = ConversationProcessingWorkflow(SimpleNamespace(), SimpleNamespace())
    monkeypatch.setattr(ConversationProcessingWorkflow, "_run_event_pipeline_finalization", fake_event)
    conversation = SimpleNamespace(userId="user A", id="sess-short")
    run = SimpleNamespace(checkpoints={})
    asyncio.run(workflow._run_short_session_finalization(conversation, run, []))
    asyncio.run(workflow._run_incremental_finalization(conversation, run, []))
    assert used == ["short_raw_transcript", "long_checkpoint_synthesis"]


def test_legacy_mode_skips_event_pipeline_even_at_100_percent(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_EVENT_PIPELINE", True)
    monkeypatch.setattr(settings, "EVENT_PIPELINE_MODE", "legacy")
    monkeypatch.setattr(settings, "EVENT_PIPELINE_ROLLOUT_PERCENT", 100)
    used = []

    async def fake_event(self, conversation, run, windows, path: str):
        used.append(path)

    monkeypatch.setattr(ConversationProcessingWorkflow, "_run_event_pipeline_finalization", fake_event)
    conversation = SimpleNamespace(userId="user A", id="sess-legacy")
    assert event_pipeline_selected_for(str(conversation.userId), str(conversation.id)) is False
    assert used == []
    assert not event_pipeline_publishes()
    assert legacy_pipeline_publishes()


def test_hard_failure_records_pipeline_fallback(monkeypatch):
    _production_100(monkeypatch)

    async def boom(self, conversation, run, windows, path: str):
        raise EventPipelineHardFailure("coverage_hard_failure")

    workflow = ConversationProcessingWorkflow(SimpleNamespace(), SimpleNamespace())
    monkeypatch.setattr(ConversationProcessingWorkflow, "_run_event_pipeline_finalization", boom)
    run = SimpleNamespace(checkpoints={})
    ok = asyncio.run(workflow._try_event_pipeline_finalization(SimpleNamespace(), run, [], "short_raw_transcript"))
    assert ok is False
    assert run.checkpoints["pipelineFallback"] is True
    assert run.checkpoints["fallbackReason"] == "coverage_hard_failure"
    failed = run.checkpoints["short_raw_transcript_event_pipeline_failed"]
    assert failed["pipelineFallback"] is True
    assert failed["publishedFrom"] == "legacy"


def test_production_smoke_representative_sessions(monkeypatch):
    _production_100(monkeypatch)
    assert event_pipeline_selected_for("smoke-user") is True
    cases = [
        ("short-english-task-note", personal_planning()),
        ("note-only", family_decision()),
        ("note-only-status", study_status()),
        ("task-note-travel", travel_change()),
        ("task-note-work", work_followup()),
        ("hinglish", code_switching()),
        ("casual-no-action", casual_noise()),
        ("task-only", _task_only_case()),
    ]
    for label, case in cases:
        assert event_pipeline_selected_for(case["id"]) is True, label
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
        _assert_event_pipeline_session(result)
        assert result.observability.asyncLifecycleErrors == 0
        assert result.coverage is None or result.coverage.unaccounted_blocks == 0
        assert result.coverage is None or not result.coverage.memoryCoverageFailure
        if case.get("expectNoTask"):
            assert result.tasks == [], label
        if case.get("expectNoNote"):
            assert result.notes == [], label
        if case.get("expectTaskSubstrings"):
            blob = " ".join(f"{item.title} {item.body}" for item in result.tasks).casefold()
            assert any(token.casefold() in blob for token in case["expectTaskSubstrings"]), (label, blob)
        if case.get("expectNoteSubstrings"):
            blob = " ".join(f"{item.title} {item.body}" for item in result.notes).casefold()
            for token in case["expectNoteSubstrings"]:
                assert token.casefold() in blob, (label, token, blob)


def test_production_smoke_medium_and_gold_long_meeting(monkeypatch):
    _production_100(monkeypatch)
    gold = build_gold_transcript()
    medium_chunks = gold["chunks"][:40]
    medium_events = [event for event in gold["events"] if min(event.sequenceIds or [0]) < 40]
    medium = asyncio.run(
        run_event_pipeline(
            medium_chunks,
            "gold-medium",
            "user_1",
            "space_1",
            event_extractor=ScriptedEventExtractor(events=medium_events),
            embedder=_embedder(),
        )
    )
    _assert_event_pipeline_session(medium)
    assert medium.observability.llm_calls() == 0

    transcript = "\n".join(f"[{seq}] {text}" for seq, text in gold["lines"].items() if str(text).strip())
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
    _assert_event_pipeline_session(long_result)
    assert long_result.observability.llm_calls() == 0
    assert long_result.observability.asyncLifecycleErrors == 0
    report = pipeline_benchmark(
        long_result,
        gold["goldTasks"],
        gold["goldNotes"],
        case_id="gold-long-meeting-100",
        transcript=transcript,
        valid_additional_notes=gold.get("validAdditionalNotes"),
        valid_additional_tasks=gold.get("validAdditionalTasks"),
        gold_events=gold["events"],
        gold_threads=gold.get("goldThreads"),
        gold_complete=True,
        original_actionable_ids=gold.get("originalActionableEventIds"),
        reviewed_actionable_ids=gold.get("reviewedActionableEventIds"),
    )
    assert report["genericTaskRate"] == 0
    assert long_result.coverage.unaccounted_blocks == 0
    assert report["matchedNotes"] == report["expectedNotes"] or report["noteRecall"] >= 0.85


def test_production_smoke_400_chunk_scale_meeting(monkeypatch):
    _production_100(monkeypatch)
    meeting = build_scale_meeting(400)
    assert 300 <= len(meeting["chunks"]) <= 500
    events = _events_for_scale(meeting)
    result = asyncio.run(
        run_event_pipeline(
            meeting["chunks"],
            meeting["id"],
            "user_1",
            "space_1",
            event_extractor=ScriptedEventExtractor(events=events),
            embedder=_embedder(),
        )
    )
    _assert_event_pipeline_session(result)
    assert result.observability.llm_calls() == 0
    assert result.observability.asyncLifecycleErrors == 0
    assert result.coverage is not None
    assert result.coverage.unaccounted_blocks == 0
    assert not result.coverage.memoryCoverageFailure
    assert result.tasks
    assert result.notes
    record = job_record(result.observability)
    assert record["rawSequences"] >= 300
    assert record["microBlocks"] > 0
    assert record["topics"] > 0
    assert record["events"] > 0
    assert record["threads"] > 0


def _task_only_case() -> dict:
    cid = "generic-task-only"
    text = "Please create the server ID."
    return {
        "id": cid,
        "chunks": [
            TranscriptChunkDocument(
                conversationId=cid,
                userId="user_1",
                spaceId="space_1",
                chunkId=f"{cid}_0",
                sequenceNumber=0,
                rawText=text,
                sttStatus=STTStatus.COMPLETED,
            )
        ],
        "events": [
            AtomicEvent(
                eventId="e-server",
                topicId="T1",
                kind=EventKind.REQUEST,
                meaning="Create the server ID.",
                object="server ID",
                entities=["Server", "ID"],
                evidence=[EvidenceSpan(sequenceStart=0, sequenceEnd=0, text=text)],
                sequenceIds=[0],
                conversationId=cid,
                userId="user_1",
                spaceId="space_1",
                actionSignal=ActionSignal(
                    isActionable=True,
                    role="REQUEST",
                    actionStrength="EXPLICIT",
                    verb="create",
                    object="server ID",
                    objectGroundingType="EXPLICIT",
                ),
            )
        ],
        "expectTaskSubstrings": ["server"],
    }


def _events_for_scale(meeting: dict) -> list[AtomicEvent]:
    gold = build_gold_transcript()
    events = list(gold["events"])
    for chunk in meeting["chunks"]:
        sequence = chunk.sequenceNumber
        if sequence < 221:
            continue
        text = (chunk.rawText or "").strip()
        if not text or any(text.startswith(filler) for filler in FILLERS):
            continue
        if not any(snippet in text for snippet in TOPIC_SNIPPETS):
            continue
        kind = EventKind.REQUEST if "create" in text.casefold() or "Please" in text else EventKind.FACT
        events.append(
            AtomicEvent(
                eventId=f"e-scale-{sequence}",
                topicId="T-scale",
                kind=kind,
                meaning=text,
                object=text[:48],
                entities=["S3"] if "S3" in text else ["server"] if "server" in text.casefold() else ["topic"],
                evidence=[EvidenceSpan(sequenceStart=sequence, sequenceEnd=sequence, text=text)],
                sequenceIds=[sequence],
                conversationId=meeting["id"],
                userId="user_1",
                spaceId="space_1",
                actionSignal=ActionSignal(
                    isActionable=True,
                    role="REQUEST",
                    actionStrength="EXPLICIT",
                    verb="create",
                    object=text[:48],
                    objectGroundingType="EXPLICIT",
                )
                if kind == EventKind.REQUEST
                else None,
            )
        )
    return events
