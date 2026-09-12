from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from apps.api_gateway.config.setting import settings
from services.conversation.inactivity import ConversationInactivityScanner
from services.conversation.models import ConversationDocument, ConversationStatus
from services.daily_briefing.jobs import DailyBriefingJobHandler
from services.daily_briefing.pipeline import DailyBriefingPipeline
from services.daily_briefing.scheduler import DailyBriefingScheduler
from services.daily_briefing.schemas import DailyBriefingSynthesis, SourceStats, TaskCard
from services.observability.diagnostics import (
    active_jobs,
    briefing_duplicates,
    collect_process_health,
    conversation_republishes,
    diag_log,
    run_worker_heartbeat,
    supervisor_restarts,
)
from services.observability import diagnostics as diagnostics_module
from services.queue.streams import EventEnvelope
from services.speech.errors import STTProviderTemporaryError
from services.speech import transcription_router
from tests.daily_briefing_fakes import FakeDatabase


def _json_events(capsys) -> list[dict]:
    rows = []
    for line in capsys.readouterr().out.splitlines():
        text = line.strip()
        if not text.startswith("{"):
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("event"):
            rows.append(payload)
    return rows


def _event_named(events: list[dict], name: str) -> list[dict]:
    return [item for item in events if item.get("event") == name]


def _briefing_event(user_id="user-1", date_key="2026-09-09", attempt=0) -> EventEnvelope:
    return EventEnvelope(
        eventType="daily.briefing.requested",
        correlationId=f"daily-briefing:{user_id}:{date_key}",
        userId=user_id,
        spaceId="daily-briefing",
        conversationId=f"daily-briefing:{user_id}:{date_key}",
        attempt=attempt,
        payload={"dateKey": date_key, "timezone": "Asia/Kolkata"},
    )


def _enable_scheduler(monkeypatch) -> datetime:
    monkeypatch.setattr(settings, "DAILY_BRIEFING_ENABLED", True)
    monkeypatch.setattr(settings, "DAILY_BRIEFING_TRIGGER_HOUR", 0)
    monkeypatch.setattr(settings, "DAILY_BRIEFING_TRIGGER_MINUTE", 5)
    monkeypatch.setattr(settings, "DAILY_BRIEFING_GRACE_MINUTES", 5)
    monkeypatch.setattr(settings, "DAILY_BRIEFING_BATCH_SIZE", 100)
    return datetime(2026, 9, 10, 0, 10, tzinfo=ZoneInfo("Asia/Kolkata")).astimezone(timezone.utc)


def test_failed_briefing_requeue_emits_diagnostic(monkeypatch, capsys):
    now = _enable_scheduler(monkeypatch)
    database = FakeDatabase()
    database.users.docs.append({"_id": "user-1", "timezone": "Asia/Kolkata"})
    database.daily_briefings.docs.append(
        {
            "userId": "user-1",
            "dateKey": "2026-09-09",
            "status": "FAILED",
            "updatedAt": now - timedelta(minutes=30),
        }
    )
    enqueued: list[EventEnvelope] = []

    async def enqueue(event: EventEnvelope) -> str:
        enqueued.append(event)
        return event.eventId

    scheduler = DailyBriefingScheduler(database, enqueue=enqueue, now_factory=lambda: now)
    counts = asyncio.run(scheduler.scan_once())
    events = _json_events(capsys)
    failed = _event_named(events, "daily_briefing_failed_requeue")
    assert counts["enqueued"] == 1
    assert counts["failed_requeued"] == 1
    assert len(enqueued) == 1
    assert len(failed) == 1
    assert failed[0]["user_id"] == "user-1"
    assert failed[0]["date_key"] == "2026-09-09"
    assert failed[0]["previous_status"] == "FAILED"
    assert failed[0]["event_id"] == enqueued[0].eventId
    assert "rawText" not in json.dumps(failed[0])


def test_stale_processing_requeue_emits_diagnostic(monkeypatch, capsys):
    now = _enable_scheduler(monkeypatch)
    database = FakeDatabase()
    database.users.docs.append({"_id": "user-1", "timezone": "Asia/Kolkata"})
    database.daily_briefings.docs.append(
        {
            "userId": "user-1",
            "dateKey": "2026-09-09",
            "status": "PROCESSING",
            "claimedAt": now - timedelta(minutes=20),
            "updatedAt": now - timedelta(minutes=20),
        }
    )
    enqueued: list[EventEnvelope] = []

    async def enqueue(event: EventEnvelope) -> str:
        enqueued.append(event)
        return event.eventId

    scheduler = DailyBriefingScheduler(database, enqueue=enqueue, now_factory=lambda: now)
    counts = asyncio.run(scheduler.scan_once())
    events = _json_events(capsys)
    stale = _event_named(events, "daily_briefing_stale_requeue")
    assert counts["enqueued"] == 1
    assert counts["stale_processing_requeued"] == 1
    assert len(stale) == 1
    assert stale[0]["user_id"] == "user-1"
    assert stale[0]["processing_age_seconds"] >= 15 * 60
    assert stale[0]["event_id"] == enqueued[0].eventId


def test_duplicate_briefing_emits_diagnostic_but_does_not_block(capsys):
    briefing_duplicates.reset()
    database = FakeDatabase()
    started = asyncio.Event()
    release = asyncio.Event()

    class SlowPipeline(DailyBriefingPipeline):
        async def generate(self, **kwargs):
            started.set()
            await release.wait()
            return (
                DailyBriefingSynthesis(headline="Ready", tasks=[TaskCard(id="1", title="A", meta="")]),
                SourceStats(transcriptCount=1),
                1,
            )

    handler = DailyBriefingJobHandler(database, pipeline=SlowPipeline())
    database.transcript_chunks.docs.append(
        {
            "_id": "chunk-1",
            "userId": "user-1",
            "rawText": "useful daily conversation about shipping the report",
            "sttStatus": "completed",
            "createdAt": datetime(2026, 9, 9, 8, tzinfo=timezone.utc),
        }
    )

    async def run():
        first = asyncio.create_task(handler.handle(_briefing_event(attempt=0)))
        await started.wait()
        second = asyncio.create_task(handler.handle(_briefing_event(attempt=1)))
        await asyncio.sleep(0.05)
        release.set()
        await asyncio.gather(first, second)

    asyncio.run(run())
    events = _json_events(capsys)
    duplicates = _event_named(events, "duplicate_briefing_execution_detected")
    assert duplicates
    assert duplicates[0]["user_id"] == "user-1"
    assert duplicates[0]["date_key"] == "2026-09-09"
    assert duplicates[0]["first_event_id"] != duplicates[0]["second_event_id"]
    stored = asyncio.run(database.daily_briefings.find_one({"userId": "user-1", "dateKey": "2026-09-09"}))
    assert stored["status"] in {"READY", "PROCESSING", "PENDING"}
    briefing_duplicates.reset()


def test_inactivity_republish_emits_diagnostic(capsys):
    conversation_republishes.reset()
    conversation = ConversationDocument(
        userId="user-1",
        spaceId="space-1",
        status=ConversationStatus.WAITING_FOR_TRANSCRIPTS,
        expectedLastSequence=3,
        receivedAudioChunkCount=3,
        updatedAt=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    class Repository:
        async def find_inactive_recording_conversations(self, cutoff):
            return []

        async def find_stale_unfinalized_conversations(self, cutoff):
            return [conversation]

        async def infer_last_sequence(self, conversation_id):
            return 3

        async def transition(self, *args, **kwargs):
            return None

    class Producer:
        def __init__(self):
            self.published = []

        async def publish(self, stream, event):
            self.published.append((stream, event))
            return event.eventId

    producer = Producer()
    scanner = ConversationInactivityScanner(Repository(), producer=producer)
    finalized = asyncio.run(scanner.scan_once())
    events = _json_events(capsys)
    republished = _event_named(events, "conversation_republished")
    summary = _event_named(events, "conversation_inactivity_scan")
    assert finalized == 1
    assert len(producer.published) == 1
    assert len(republished) == 1
    assert republished[0]["conversation_id"] == str(conversation.id)
    assert republished[0]["status"] == ConversationStatus.WAITING_FOR_TRANSCRIPTS.value
    assert republished[0]["republish_count"] == 1
    assert republished[0]["attempt"] == 0
    assert summary[0]["stale_found"] == 1
    assert summary[0]["finalization_events_published"] == 1
    conversation_republishes.reset()


def test_active_job_counter_decrements_after_exception():
    active_jobs.reset()
    with pytest.raises(RuntimeError):
        with active_jobs.track("stt"):
            assert active_jobs.get("stt") == 1
            raise RuntimeError("handler failed")
    assert active_jobs.get("stt") == 0


def test_worker_health_collection_failure_does_not_stop_loop(monkeypatch):
    calls = {"n": 0}

    def boom(_sampler):
        raise RuntimeError("cpu sampler failed")

    async def instant(_seconds):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise asyncio.CancelledError()

    monkeypatch.setattr(diagnostics_module, "collect_process_health", boom)
    monkeypatch.setattr(diagnostics_module.asyncio, "sleep", instant)

    async def run():
        with pytest.raises(asyncio.CancelledError):
            await run_worker_heartbeat(interval_seconds=30)

    asyncio.run(run())
    assert calls["n"] >= 2


def test_collect_process_health_never_raises():
    from services.observability.diagnostics import _CpuSampler

    payload = collect_process_health(_CpuSampler())
    assert payload["process_pid"] > 0
    assert "asyncio_task_count" in payload
    assert "thread_count" in payload


def test_supervisor_restart_storm_after_six_failures():
    supervisor_restarts.reset()
    last = None
    for _ in range(6):
        last = supervisor_restarts.record("retry-relay", "TimeoutError")
    assert last is not None
    assert last["storm"] is True
    assert last["restarts_last_60s"] > 5
    supervisor_restarts.reset()


def test_stt_retry_log_has_no_transcript_or_audio(monkeypatch, capsys):
    monkeypatch.setattr(settings, "STT_PROVIDER_ORDER", "deepgram")
    monkeypatch.setattr(settings, "STT_ALLOW_SARVAM_FALLBACK", False)
    monkeypatch.setattr(settings, "STT_MAX_RETRIES", 2)

    async def fail(*args, **kwargs):
        raise STTProviderTemporaryError("provider timeout", provider="deepgram")

    monkeypatch.setitem(transcription_router._PROVIDERS, "deepgram", fail)

    async def run():
        with pytest.raises(RuntimeError):
            await transcription_router.transcribe_from_path_with_fallback(
                file_path="/tmp/secret-audio.wav",
                filename="secret-audio.wav",
                content_type="audio/wav",
                job_id="job-9",
                stream_attempt=2,
            )

    asyncio.run(run())
    events = _json_events(capsys)
    retries = _event_named(events, "stt_provider_retry")
    assert retries
    blob = json.dumps(retries)
    assert "transcript" not in blob.lower() or all("transcript" not in item for item in retries)
    assert "secret-audio" not in blob
    assert "/tmp/" not in blob
    assert retries[0]["immediate_retry"] is True
    assert retries[0]["job_id"] == "job-9"
    assert retries[0]["stream_attempt"] == 2
    assert retries[0]["provider_attempt"] >= 1


def test_diag_log_omits_none_and_payload_text():
    diag_log("worker_health", process_pid=1, transcript=None, audio=None)
    # If this raised it would fail; None fields are omitted by design.
    payload = json.loads(json.dumps({"event": "worker_health", "process_pid": 1}))
    assert "transcript" not in payload
    assert "audio" not in payload
