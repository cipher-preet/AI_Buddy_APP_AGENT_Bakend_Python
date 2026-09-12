from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from apps.api_gateway.config.setting import Settings, settings
from services.daily_briefing.jobs import DailyBriefingJobHandler
from services.daily_briefing.pipeline import DailyBriefingPipeline, PendingTranscriptError
from services.daily_briefing.prepare import build_windows, filter_and_order_transcripts
from services.daily_briefing.scheduler import DailyBriefingScheduler
from services.daily_briefing.schemas import (
    BriefingItem,
    DailyBriefingSynthesis,
    TaskCard,
    ValidatorResult,
    WindowIntelligence,
)
from services.daily_briefing.sources import ActivityBundle, ActivitySource
from services.daily_briefing.store import DailyBriefingStore
from services.daily_briefing.timezones import (
    is_after_trigger,
    local_day_bounds_utc,
    previous_date_key,
)
from services.llm.router import LLMCapability
from services.prompts.loader import load_prompt
from services.queue.streams import EventEnvelope
from tests.daily_briefing_fakes import FakeDatabase


def test_india_timezone_daily_boundary():
    start, end = local_day_bounds_utc("2026-09-09", "Asia/Kolkata")
    assert start == datetime(2026, 9, 8, 18, 30, tzinfo=timezone.utc)
    assert end == datetime(2026, 9, 9, 18, 30, tzinfo=timezone.utc)
    assert end == start + timedelta(hours=24)


def test_us_timezone_daily_boundary():
    start, end = local_day_bounds_utc("2026-09-09", "America/New_York")
    assert start == datetime(2026, 9, 9, 4, 0, tzinfo=timezone.utc)
    assert end == datetime(2026, 9, 10, 4, 0, tzinfo=timezone.utc)


def test_dst_safe_timezone_calculation():
    spring_start, spring_end = local_day_bounds_utc("2026-03-08", "America/New_York")
    assert spring_end - spring_start == timedelta(hours=23)
    fall_start, fall_end = local_day_bounds_utc("2026-11-01", "America/New_York")
    assert fall_end - fall_start == timedelta(hours=25)


def test_previous_date_key_is_completed_local_day():
    now = datetime(2026, 9, 10, 0, 10, tzinfo=ZoneInfo("Asia/Kolkata")).astimezone(timezone.utc)
    assert previous_date_key(now, "Asia/Kolkata") == "2026-09-09"


def test_trigger_hour_minute_config():
    local_1330 = datetime(2026, 9, 9, 13, 30, tzinfo=ZoneInfo("Asia/Kolkata")).astimezone(timezone.utc)
    local_1329 = datetime(2026, 9, 9, 13, 29, tzinfo=ZoneInfo("Asia/Kolkata")).astimezone(timezone.utc)
    assert is_after_trigger(local_1330, "Asia/Kolkata", 13, 30, grace_minutes=0) is True
    assert is_after_trigger(local_1329, "Asia/Kolkata", 13, 30, grace_minutes=0) is False


def test_changing_env_trigger_time_without_code_changes(monkeypatch):
    monkeypatch.setenv("SERVICE_ROLE", "api")
    monkeypatch.setenv("DAILY_BRIEFING_TRIGGER_HOUR", "13")
    monkeypatch.setenv("DAILY_BRIEFING_TRIGGER_MINUTE", "30")
    loaded = Settings(_env_file=None)
    assert loaded.DAILY_BRIEFING_TRIGGER_HOUR == 13
    assert loaded.DAILY_BRIEFING_TRIGGER_MINUTE == 30
    local_ready = datetime(2026, 9, 9, 13, 30, tzinfo=ZoneInfo("Asia/Kolkata")).astimezone(timezone.utc)
    assert is_after_trigger(
        local_ready,
        "Asia/Kolkata",
        loaded.DAILY_BRIEFING_TRIGGER_HOUR,
        loaded.DAILY_BRIEFING_TRIGGER_MINUTE,
        grace_minutes=0,
    )


def test_disabled_scheduler_does_not_enqueue(monkeypatch):
    monkeypatch.setattr(settings, "DAILY_BRIEFING_ENABLED", False)
    database = FakeDatabase()
    enqueued: list[EventEnvelope] = []

    async def enqueue(event: EventEnvelope) -> str:
        enqueued.append(event)
        return event.eventId

    scheduler = DailyBriefingScheduler(database, enqueue=enqueue)
    counts = asyncio.run(scheduler.scan_once())
    assert counts["disabled"] == 1
    assert enqueued == []


def test_transcript_chronological_order_and_duplicate_filtering():
    later = datetime(2026, 9, 9, 10, tzinfo=timezone.utc)
    earlier = datetime(2026, 9, 9, 8, tzinfo=timezone.utc)
    items = [
        {"text": "same useful transcript text", "createdAt": later, "id": "b"},
        {"text": "same useful transcript text", "createdAt": earlier, "id": "a"},
        {"text": "short", "createdAt": earlier, "id": "c"},
        {"text": "another useful transcript here", "createdAt": later, "id": "d"},
    ]
    ordered = filter_and_order_transcripts(items)
    assert [item["id"] for item in ordered] == ["a", "d"]


def test_hindi_hinglish_transcripts_are_kept():
    items = [
        {"text": "कल मीटिंग में बजट फाइनल करना है", "createdAt": datetime(2026, 9, 9, 8, tzinfo=timezone.utc)},
        {"text": "Kal meeting mein budget final karna hai", "createdAt": datetime(2026, 9, 9, 9, tzinfo=timezone.utc)},
    ]
    ordered = filter_and_order_transcripts(items)
    assert len(ordered) == 2


def test_hundreds_of_transcripts_become_bounded_windows():
    items = [
        {
            "text": f"useful daily conversation chunk number {index} " * 40,
            "createdAt": datetime(2026, 9, 9, 8, 0, tzinfo=timezone.utc) + timedelta(minutes=index),
            "id": str(index),
        }
        for index in range(200)
    ]
    ordered = filter_and_order_transcripts(items)
    windows = build_windows(ordered, target_tokens=400, max_tokens=500)
    assert len(windows) > 1
    assert all(window.tokenCount <= 2000 for window in windows)


class RecordingCaller:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[tuple[LLMCapability, str, dict]] = []

    async def generate(self, capability, prompt_name, schema, payload):
        self.calls.append((capability, prompt_name, payload))
        if not self.responses:
            raise AssertionError("unexpected model call")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _event(user_id="user-1", date_key="2026-09-09", attempt=0) -> EventEnvelope:
    return EventEnvelope(
        eventType="daily.briefing.requested",
        correlationId=f"daily-briefing:{user_id}:{date_key}",
        userId=user_id,
        spaceId="daily-briefing",
        conversationId=f"daily-briefing:{user_id}:{date_key}",
        attempt=attempt,
        payload={"dateKey": date_key, "timezone": "Asia/Kolkata"},
    )


def test_no_activity_user_skipped_without_llm():
    database = FakeDatabase()
    caller = RecordingCaller([])
    handler = DailyBriefingJobHandler(database, pipeline=DailyBriefingPipeline(caller=caller))
    asyncio.run(handler.handle(_event()))
    assert caller.calls == []
    stored = asyncio.run(database.daily_briefings.find_one({"userId": "user-1", "dateKey": "2026-09-09"}))
    assert stored["status"] == "SKIPPED"


def test_pending_stt_uses_retry_instead_of_ready():
    database = FakeDatabase()
    start, end = local_day_bounds_utc("2026-09-09", "Asia/Kolkata")
    database.transcript_chunks.docs.append(
        {
            "_id": "chunk-1",
            "userId": "user-1",
            "rawText": "still processing this useful transcript",
            "sttStatus": "pending",
            "createdAt": start + timedelta(hours=2),
        }
    )
    caller = RecordingCaller([])
    handler = DailyBriefingJobHandler(database, pipeline=DailyBriefingPipeline(caller=caller))

    async def run():
        try:
            await handler.handle(_event(attempt=0))
        except PendingTranscriptError:
            return "retried"
        return "completed"

    assert asyncio.run(run()) == "retried"
    assert caller.calls == []
    stored = asyncio.run(database.daily_briefings.find_one({"userId": "user-1", "dateKey": "2026-09-09"}))
    assert stored["status"] != "READY"


def test_tasks_and_notes_included_as_structured_context():
    bundle = ActivityBundle(
        transcripts=[{"id": "t1", "text": "We agreed to ship the weekly report tomorrow morning.", "createdAt": datetime(2026, 9, 9, 10, tzinfo=timezone.utc)}],
        tasks=[{"id": "task-1", "title": "Ship weekly report", "status": "open"}],
        notes=[{"id": "note-1", "title": "Report outline", "detail": "Include blockers"}],
    )
    caller = RecordingCaller(
        [
            WindowIntelligence(summary="Talked about the weekly report."),
            DailyBriefingSynthesis(headline="Report day", overview="Ship the weekly report."),
            ValidatorResult(accepted=True),
        ]
    )
    pipeline = DailyBriefingPipeline(caller=caller)
    asyncio.run(
        pipeline.generate(
            user_id="user-1",
            date_key="2026-09-09",
            timezone_name="Asia/Kolkata",
            bundle=bundle,
        )
    )
    window_payload = caller.calls[0][2]
    synthesis_payload = caller.calls[1][2]
    assert window_payload["structuredContext"]["tasks"][0]["title"] == "Ship weekly report"
    assert synthesis_payload["structuredContext"]["notes"][0]["title"] == "Report outline"


def test_daily_briefing_cannot_create_canonical_tasks():
    database = FakeDatabase()
    start, end = local_day_bounds_utc("2026-09-09", "Asia/Kolkata")
    database.transcript_chunks.docs.append(
        {
            "_id": "chunk-1",
            "userId": "user-1",
            "normalizedText": "Please create a task to call the vendor tomorrow.",
            "sttStatus": "completed",
            "createdAt": start + timedelta(hours=3),
        }
    )
    caller = RecordingCaller(
        [
            WindowIntelligence(
                summary="Vendor follow-up mentioned.",
                pendingItems=[BriefingItem(id="p1", title="Call vendor")],
            ),
            DailyBriefingSynthesis(
                headline="Follow up",
                missedCandidates=[BriefingItem(id="c1", title="Call vendor")],
            ),
            ValidatorResult(accepted=True),
        ]
    )
    handler = DailyBriefingJobHandler(database, pipeline=DailyBriefingPipeline(caller=caller))
    asyncio.run(handler.handle(_event()))
    assert database.tasks.inserts == []
    assert database.tasks.docs == []
    prompt = load_prompt("daily-briefing-synthesis-v1")
    assert "Do not create canonical Tasks" in prompt


def test_llm_malformed_schema_falls_back():
    bundle = ActivityBundle(
        transcripts=[{"id": "t1", "text": "Decision: keep the current launch window.", "createdAt": datetime(2026, 9, 9, 10, tzinfo=timezone.utc)}],
        tasks=[{"id": "task-1", "title": "Keep launch window", "dueDate": "today"}],
    )
    caller = RecordingCaller(
        [
            WindowIntelligence(summary="Launch window stays."),
            ValueError("malformed synthesis"),
            ValidatorResult(accepted=True),
        ]
    )
    pipeline = DailyBriefingPipeline(caller=caller)
    synthesis, _stats, _windows = asyncio.run(
        pipeline.generate(
            user_id="user-1",
            date_key="2026-09-09",
            timezone_name="Asia/Kolkata",
            bundle=bundle,
        )
    )
    assert synthesis.headline
    assert synthesis.tasks[0].title == "Keep launch window"


def test_validator_rejection_does_not_mark_ready():
    database = FakeDatabase()
    start, _end = local_day_bounds_utc("2026-09-09", "Asia/Kolkata")
    database.transcript_chunks.docs.append(
        {
            "_id": "chunk-1",
            "userId": "user-1",
            "normalizedText": "We decided to pause hiring this week.",
            "sttStatus": "completed",
            "createdAt": start + timedelta(hours=1),
        }
    )
    caller = RecordingCaller(
        [
            WindowIntelligence(summary="Hiring pause."),
            DailyBriefingSynthesis(headline="Invented CEO meeting"),
            ValidatorResult(accepted=False, reasons=["invented people"]),
        ]
    )
    pipeline = DailyBriefingPipeline(caller=caller)
    pipeline.max_retries = 0
    handler = DailyBriefingJobHandler(database, pipeline=pipeline)

    async def run():
        try:
            await handler.handle(_event())
        except ValueError as error:
            return str(error)
        return "ok"

    assert asyncio.run(run()) == "validator_rejected"
    stored = asyncio.run(database.daily_briefings.find_one({"userId": "user-1", "dateKey": "2026-09-09"}))
    assert stored["status"] == "FAILED"


def test_retry_behavior_revalidates_without_infinite_loop():
    bundle = ActivityBundle(
        transcripts=[{"id": "t1", "text": "Follow up with finance on invoices tomorrow.", "createdAt": datetime(2026, 9, 9, 10, tzinfo=timezone.utc)}]
    )
    caller = RecordingCaller(
        [
            WindowIntelligence(summary="Finance follow-up."),
            DailyBriefingSynthesis(headline="Needs repair"),
            ValidatorResult(accepted=False, reasons=["schema"]),
            DailyBriefingSynthesis(headline="Finance follow-up"),
            ValidatorResult(accepted=True),
        ]
    )
    pipeline = DailyBriefingPipeline(caller=caller)
    pipeline.max_retries = 1
    synthesis, _stats, _windows = asyncio.run(
        pipeline.generate(
            user_id="user-1",
            date_key="2026-09-09",
            timezone_name="Asia/Kolkata",
            bundle=bundle,
        )
    )
    assert synthesis.headline == "Finance follow-up"
    assert len([call for call in caller.calls if call[0] == LLMCapability.VALIDATION]) == 2
    assert caller.responses == []


def test_userid_datekey_idempotency_and_concurrent_claim():
    database = FakeDatabase()
    store = DailyBriefingStore(database)
    start, end = local_day_bounds_utc("2026-09-09", "Asia/Kolkata")

    async def race():
        first, second = await asyncio.gather(
            store.claim("user-1", "2026-09-09", "Asia/Kolkata", start, end, "job-a"),
            store.claim("user-1", "2026-09-09", "Asia/Kolkata", start, end, "job-b"),
        )
        return first[0], second[0]

    states = asyncio.run(race())
    assert "claimed" in states
    assert "busy" in states or "exists" in states
    assert len(database.daily_briefings.docs) == 1


def test_worker_retry_does_not_create_duplicate_record():
    database = FakeDatabase()
    start, end = local_day_bounds_utc("2026-09-09", "Asia/Kolkata")
    store = DailyBriefingStore(database)
    synthesis = DailyBriefingSynthesis(headline="Ready", tasks=[TaskCard(id="1", title="A", meta="")])
    from services.daily_briefing.schemas import SourceStats

    async def run():
        await store.claim("user-1", "2026-09-09", "Asia/Kolkata", start, end, "job-1")
        await store.save_ready("user-1", "2026-09-09", "Asia/Kolkata", start, end, synthesis, SourceStats())
        await store.claim("user-1", "2026-09-09", "Asia/Kolkata", start, end, "job-2")
        await store.save_ready("user-1", "2026-09-09", "Asia/Kolkata", start, end, synthesis, SourceStats())

    asyncio.run(run())
    assert len(database.daily_briefings.docs) == 1
    assert database.daily_briefings.docs[0]["status"] == "READY"


def test_api_authorization_isolation():
    database = FakeDatabase()
    store = DailyBriefingStore(database)
    start, end = local_day_bounds_utc("2026-09-09", "Asia/Kolkata")
    synthesis = DailyBriefingSynthesis(headline="User A")
    from services.daily_briefing.schemas import SourceStats

    async def run():
        await store.claim("user-a", "2026-09-09", "Asia/Kolkata", start, end, "job-a")
        await store.save_ready("user-a", "2026-09-09", "Asia/Kolkata", start, end, synthesis, SourceStats())
        other = await store.get("user-b", "2026-09-09")
        own = await store.get("user-a", "2026-09-09")
        return other, own

    other, own = asyncio.run(run())
    assert other is None
    assert own["headline"] == "User A"


def test_scheduler_enqueues_after_configured_trigger(monkeypatch):
    monkeypatch.setattr(settings, "DAILY_BRIEFING_ENABLED", True)
    monkeypatch.setattr(settings, "DAILY_BRIEFING_TRIGGER_HOUR", 0)
    monkeypatch.setattr(settings, "DAILY_BRIEFING_TRIGGER_MINUTE", 5)
    monkeypatch.setattr(settings, "DAILY_BRIEFING_GRACE_MINUTES", 5)
    monkeypatch.setattr(settings, "DAILY_BRIEFING_BATCH_SIZE", 100)
    database = FakeDatabase()
    database.users.docs.append({"_id": "user-1", "timezone": "Asia/Kolkata"})
    enqueued: list[EventEnvelope] = []

    async def enqueue(event: EventEnvelope) -> str:
        enqueued.append(event)
        return event.eventId

    now = datetime(2026, 9, 10, 0, 10, tzinfo=ZoneInfo("Asia/Kolkata")).astimezone(timezone.utc)
    scheduler = DailyBriefingScheduler(database, enqueue=enqueue, now_factory=lambda: now)
    counts = asyncio.run(scheduler.scan_once())
    assert counts["enqueued"] == 1
    assert enqueued[0].payload["dateKey"] == "2026-09-09"


def test_scheduler_requeues_stale_pending(monkeypatch):
    monkeypatch.setattr(settings, "DAILY_BRIEFING_ENABLED", True)
    monkeypatch.setattr(settings, "DAILY_BRIEFING_TRIGGER_HOUR", 0)
    monkeypatch.setattr(settings, "DAILY_BRIEFING_TRIGGER_MINUTE", 5)
    monkeypatch.setattr(settings, "DAILY_BRIEFING_GRACE_MINUTES", 5)
    monkeypatch.setattr(settings, "DAILY_BRIEFING_BATCH_SIZE", 100)
    database = FakeDatabase()
    now = datetime(2026, 9, 10, 0, 10, tzinfo=ZoneInfo("Asia/Kolkata")).astimezone(timezone.utc)
    database.users.docs.append({"_id": "user-1", "timezone": "Asia/Kolkata"})
    database.daily_briefings.docs.append(
        {
            "userId": "user-1",
            "dateKey": "2026-09-09",
            "status": "PENDING",
            "updatedAt": now - timedelta(minutes=20),
            "createdAt": now - timedelta(minutes=20),
        }
    )
    enqueued: list[EventEnvelope] = []

    async def enqueue(event: EventEnvelope) -> str:
        enqueued.append(event)
        return event.eventId

    scheduler = DailyBriefingScheduler(database, enqueue=enqueue, now_factory=lambda: now)
    counts = asyncio.run(scheduler.scan_once())
    assert counts["enqueued"] == 1
    assert enqueued[0].payload["dateKey"] == "2026-09-09"


def test_scheduler_does_not_requeue_fresh_pending(monkeypatch):
    monkeypatch.setattr(settings, "DAILY_BRIEFING_ENABLED", True)
    monkeypatch.setattr(settings, "DAILY_BRIEFING_TRIGGER_HOUR", 0)
    monkeypatch.setattr(settings, "DAILY_BRIEFING_TRIGGER_MINUTE", 5)
    monkeypatch.setattr(settings, "DAILY_BRIEFING_GRACE_MINUTES", 5)
    monkeypatch.setattr(settings, "DAILY_BRIEFING_BATCH_SIZE", 100)
    database = FakeDatabase()
    now = datetime(2026, 9, 10, 0, 10, tzinfo=ZoneInfo("Asia/Kolkata")).astimezone(timezone.utc)
    database.users.docs.append({"_id": "user-1", "timezone": "Asia/Kolkata"})
    database.daily_briefings.docs.append(
        {
            "userId": "user-1",
            "dateKey": "2026-09-09",
            "status": "PENDING",
            "updatedAt": now - timedelta(minutes=2),
            "createdAt": now - timedelta(minutes=2),
        }
    )
    enqueued: list[EventEnvelope] = []

    async def enqueue(event: EventEnvelope) -> str:
        enqueued.append(event)
        return event.eventId

    scheduler = DailyBriefingScheduler(database, enqueue=enqueue, now_factory=lambda: now)
    counts = asyncio.run(scheduler.scan_once())
    assert counts["enqueued"] == 0
    assert enqueued == []


def test_sources_query_uses_utc_bounds_not_server_calendar():
    database = FakeDatabase()
    start, end = local_day_bounds_utc("2026-09-09", "Asia/Kolkata")
    database.transcript_chunks.docs.extend(
        [
            {
                "_id": "inside",
                "userId": "user-1",
                "normalizedText": "inside the local day transcript text",
                "sttStatus": "completed",
                "createdAt": start + timedelta(hours=1),
            },
            {
                "_id": "outside",
                "userId": "user-1",
                "normalizedText": "outside the local day transcript text",
                "sttStatus": "completed",
                "createdAt": end,
            },
        ]
    )
    bundle = asyncio.run(ActivitySource(database).load("user-1", "2026-09-09", start, end))
    assert [item["id"] for item in bundle.transcripts] == ["inside"]
