from __future__ import annotations

import asyncio
import copy
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from bson import ObjectId

from services.conversation.models import STTStatus, TranscriptChunkDocument
from services.schedule_extraction.job import RetryableExtractionError, ScheduleExtractionJob
from services.schedule_extraction.reminder_schedule import ReminderScheduler, build_schedule_fields
from services.schedule_extraction.schemas import (
    EventCandidate,
    EventExtraction,
    PipelineOutcome,
    ReminderCandidate,
    ReminderExtraction,
)
from services.schedule_extraction.validation import CandidateValidator

NOW = datetime(2026, 9, 30, 4, 30, tzinfo=timezone.utc)  # 10:00 AM Asia/Kolkata
RECORDED_LOCAL = datetime(2026, 9, 30, 10, 0)
USER_ID = ObjectId()
CONVERSATION_ID = str(ObjectId())
TRANSCRIPT = [
    "Okay team, let's do the pricing review with Rahul tomorrow at 3 PM on Zoom.",
    "Also remind me to call the HDFC bank tomorrow at 11 AM about the loan.",
    "Yesterday we had the retro, that went fine.",
]


# ---------- in-memory fakes ----------


def _matches(doc: dict, query: dict) -> bool:
    for key, expected in query.items():
        actual = doc.get(key)
        if isinstance(expected, dict) and "$in" in expected:
            if actual not in expected["$in"]:
                return False
        elif actual != expected:
            return False
    return True


class _Cursor:
    def __init__(self, docs):
        self.docs = docs

    async def to_list(self, length=None):
        return [copy.deepcopy(doc) for doc in self.docs[:length]]


class FakeCollection:
    def __init__(self):
        self.docs: list[dict] = []

    def find(self, query=None, projection=None):
        return _Cursor([doc for doc in self.docs if _matches(doc, query or {})])

    async def find_one(self, query=None, projection=None, sort=None):
        for doc in self.docs:
            if _matches(doc, query or {}):
                return copy.deepcopy(doc)
        return None

    async def insert_one(self, doc):
        self.docs.append(copy.deepcopy(doc))

    async def update_one(self, query, update, upsert=False):
        for doc in self.docs:
            if _matches(doc, query):
                doc.update(copy.deepcopy(update.get("$set", {})))
                return

    async def find_one_and_update(self, query, update, upsert=False, return_document=None):
        for doc in self.docs:
            if _matches(doc, query):
                return copy.deepcopy(doc)
        if not upsert:
            return None
        created = {**query, **copy.deepcopy(update.get("$setOnInsert", {}))}
        self.docs.append(created)
        return copy.deepcopy(created)


class FakeDatabase:
    def __init__(self):
        self.users = FakeCollection()
        self.reminders = FakeCollection()
        self.calendar_events = FakeCollection()
        self.schedule_extractions = FakeCollection()
        self.users.docs.append({"_id": USER_ID, "timezone": "Asia/Kolkata"})


class FakeRepository:
    def __init__(self, lines: list[str]):
        self.lines = lines

    async def get_conversation(self, conversation_id):
        return SimpleNamespace(userId=USER_ID, startedAt=NOW, createdAt=NOW)

    async def list_transcript_chunks(self, conversation_id):
        return [
            TranscriptChunkDocument(
                conversationId=conversation_id,
                userId=USER_ID,
                spaceId=None,
                chunkId=f"c{index}",
                sequenceNumber=index,
                rawText=text,
                sttStatus=STTStatus.COMPLETED,
            )
            for index, text in enumerate(self.lines)
        ]


class FakePipe:
    def __init__(self, redis):
        self.redis = redis
        self.ops: list = []

    def zadd(self, key, mapping):
        self.ops.append(("zadd", key, mapping))

    def set(self, key, value, ex=None):
        self.ops.append(("set", key, value))

    async def execute(self):
        if self.redis.fail:
            raise ConnectionError("redis down")
        for op in self.ops:
            if op[0] == "zadd":
                self.redis.schedule.update(op[2])
            else:
                self.redis.payloads[op[1]] = op[2]


class FakeRedis:
    def __init__(self):
        self.schedule: dict[str, int] = {}
        self.payloads: dict[str, str] = {}
        self.fail = False

    def pipeline(self):
        return FakePipe(self)


MEETING = EventCandidate(
    kind="meeting",
    title="Pricing review with Rahul",
    dateKey="2026-10-01",
    startTime="3:00 PM",
    location="Zoom",
    evidence="let's do the pricing review with Rahul tomorrow at 3 PM on Zoom",
    confidence=0.9,
)
BANK = ReminderCandidate(
    title="Call HDFC bank",
    dateKey="2026-10-01",
    time="11:00 AM",
    evidence="remind me to call the HDFC bank tomorrow at 11 AM",
    confidence=0.95,
)


class FakeCaller:
    def __init__(self, events=None, reminders=None, fail: set[str] | None = None):
        self.events = events if events is not None else [MEETING]
        self.reminders = reminders if reminders is not None else [BANK]
        self.fail = fail or set()
        self.calls: list[str] = []

    async def generate(self, prompt_name, schema, payload):
        self.calls.append(prompt_name)
        if any(key in prompt_name for key in self.fail):
            raise RuntimeError("llm down")
        if schema is EventExtraction:
            return EventExtraction(events=self.events)
        return ReminderExtraction(reminders=self.reminders)


def _job(database, caller, redis, lines=TRANSCRIPT):
    return ScheduleExtractionJob(
        FakeRepository(lines),
        database,
        caller=caller,
        scheduler=ReminderScheduler(redis=redis),
        now=NOW,
    )


def _validator(lines=TRANSCRIPT):
    return CandidateValidator(CONVERSATION_ID, "\n".join(lines), RECORDED_LOCAL, "Asia/Kolkata", 0.6, 20, now=NOW)


# ---------- validator ----------


def test_validator_keeps_grounded_future_event_and_normalizes_time():
    outcome = PipelineOutcome(name="calendar_events")
    events = _validator().events([MEETING.model_copy(update={"startTime": "15:00"})], outcome)

    assert len(events) == 1
    assert events[0].startTimeLabel == "3:00 PM"
    assert events[0].endTimeLabel == "4:00 PM"
    assert events[0].timeInferred is False


@pytest.mark.parametrize(
    ("update", "reason"),
    [
        ({"evidence": "we will launch the rocket on mars next month"}, "ungrounded"),
        ({"confidence": 0.3}, "low_confidence"),
        ({"dateKey": "2026-09-29", "dateText": ""}, "invalid_date"),
        ({"dateKey": "2026-09-30", "startTime": "9:00 AM"}, "in_past"),
        ({"title": ""}, "no_title"),
    ],
)
def test_validator_drops_unsafe_events(update, reason):
    outcome = PipelineOutcome(name="calendar_events")
    events = _validator().events([MEETING.model_copy(update=update)], outcome)

    assert events == []
    assert outcome.dropped == {reason: 1}


def test_validator_dedupes_and_prefers_explicit_time():
    outcome = PipelineOutcome(name="calendar_events")
    vague = MEETING.model_copy(update={"startTime": "", "confidence": 0.95})
    events = _validator().events([vague, MEETING], outcome)

    assert len(events) == 1
    assert events[0].startTimeLabel == "3:00 PM"
    assert outcome.dropped == {"duplicate": 1}


def test_validator_falls_back_to_date_text_and_default_time():
    outcome = PipelineOutcome(name="reminders")
    reminders = _validator().reminders([BANK.model_copy(update={"dateKey": "", "dateText": "tomorrow", "time": ""})], outcome)

    assert reminders[0].dateKey == "2026-10-01"
    assert reminders[0].timeLabel == "9:00 AM"
    assert reminders[0].timeInferred is True


# ---------- scheduling ----------


def test_schedule_fields_match_node_rules():
    future = build_schedule_fields(
        date_key="2026-10-01", time_label="11:00 AM", repeat="once", timezone_name="Asia/Kolkata",
        ai_calling=True, beeping=False, now=NOW,
    )
    past = build_schedule_fields(
        date_key="2026-09-29", time_label="11:00 AM", repeat="once", timezone_name="Asia/Kolkata",
        ai_calling=True, beeping=False, now=NOW,
    )

    assert future["deliveryType"] == "AI_CALL"
    assert future["deliveryStatus"] == "SCHEDULED"
    assert future["nextTriggerAtUtc"] == datetime(2026, 10, 1, 5, 30, tzinfo=timezone.utc)
    assert past["deliveryStatus"] == "FAILED"


# ---------- end-to-end job ----------


def test_job_writes_event_with_linked_ai_call_and_separate_reminder():
    database, redis = FakeDatabase(), FakeRedis()
    result = asyncio.run(_job(database, FakeCaller(), redis).run(CONVERSATION_ID, "job-1"))

    assert result["status"] == "READY"
    [event] = database.calendar_events.docs
    assert event["title"] == "Pricing review with Rahul"
    assert event["aiReminder"] is True and event["aiCalling"] is True
    assert event["remindBeforeMinutes"] == 15
    assert event["location"] == "Zoom"

    reminders = {doc["title"]: doc for doc in database.reminders.docs}
    linked = reminders["Pricing review with Rahul"]
    assert linked["_id"] == event["reminderId"]
    assert linked["timeLabel"] == "2:45 PM"
    bank = reminders["Call HDFC bank"]
    assert bank["aiCalling"] is True and bank["deliveryType"] == "AI_CALL"
    assert bank["source"] == "ai" and bank["deliveryStatus"] == "SCHEDULED"
    assert set(redis.schedule) == {linked["scheduledOccurrenceId"], bank["scheduledOccurrenceId"]}
    assert bank["scheduledOccurrenceId"] == f"{bank['_id']}:2026-10-01T05:30:00Z"


def test_rerun_is_idempotent():
    database, redis = FakeDatabase(), FakeRedis()
    asyncio.run(_job(database, FakeCaller(), redis).run(CONVERSATION_ID, "job-1"))
    database.schedule_extractions.docs[0]["status"] = "RETRY_PENDING"
    asyncio.run(_job(database, FakeCaller(), redis).run(CONVERSATION_ID, "job-2"))

    assert len(database.calendar_events.docs) == 1
    assert len(database.reminders.docs) == 2


def test_finished_job_is_not_rerun():
    database, redis = FakeDatabase(), FakeRedis()
    asyncio.run(_job(database, FakeCaller(), redis).run(CONVERSATION_ID, "job-1"))
    caller = FakeCaller()
    result = asyncio.run(_job(database, caller, redis).run(CONVERSATION_ID, "job-2"))

    assert result == {"status": "exists"}
    assert caller.calls == []


def test_reminder_about_extracted_meeting_is_not_duplicated():
    database, redis = FakeDatabase(), FakeRedis()
    duplicate = ReminderCandidate(
        title="Pricing review with Rahul",
        dateKey="2026-10-01",
        time="2:30 PM",
        evidence="let's do the pricing review with Rahul tomorrow at 3 PM",
        confidence=0.9,
    )
    asyncio.run(_job(database, FakeCaller(reminders=[duplicate]), redis).run(CONVERSATION_ID, "job-1"))

    assert len(database.reminders.docs) == 1
    assert database.reminders.docs[0]["_id"] == database.calendar_events.docs[0]["reminderId"]


def test_existing_user_reminder_is_not_duplicated():
    database, redis = FakeDatabase(), FakeRedis()
    database.reminders.docs.append({"_id": ObjectId(), "userId": USER_ID, "title": "Call HDFC bank", "dateKey": "2026-10-01"})
    asyncio.run(_job(database, FakeCaller(events=[]), redis).run(CONVERSATION_ID, "job-1"))

    assert len(database.reminders.docs) == 1


def test_one_pipeline_failing_keeps_other_and_retry_skips_finished_pipeline():
    database, redis = FakeDatabase(), FakeRedis()
    with pytest.raises(RetryableExtractionError):
        asyncio.run(_job(database, FakeCaller(fail={"reminder"}), redis).run(CONVERSATION_ID, "job-1"))

    assert len(database.calendar_events.docs) == 1
    job_doc = database.schedule_extractions.docs[0]
    assert job_doc["status"] == "RETRY_PENDING"
    assert job_doc["completedPipelines"] == ["calendar_events"]

    caller = FakeCaller()
    result = asyncio.run(_job(database, caller, redis).run(CONVERSATION_ID, "job-1"))

    assert result["status"] == "READY"
    assert caller.calls == ["meeting-reminder-extractor-v1"]
    assert len(database.calendar_events.docs) == 1
    assert {doc["title"] for doc in database.reminders.docs} == {"Pricing review with Rahul", "Call HDFC bank"}


def test_final_attempt_marks_failed_without_raising():
    database, redis = FakeDatabase(), FakeRedis()
    result = asyncio.run(
        _job(database, FakeCaller(fail={"calendar", "reminder"}), redis).run(CONVERSATION_ID, "job-1", final_attempt=True)
    )

    assert result["status"] == "FAILED"
    assert database.calendar_events.docs == [] and database.reminders.docs == []


def test_redis_outage_raises_and_retry_schedules_existing_reminder():
    database, redis = FakeDatabase(), FakeRedis()
    redis.fail = True
    with pytest.raises(RuntimeError):
        asyncio.run(_job(database, FakeCaller(events=[]), redis).run(CONVERSATION_ID, "job-1"))
    assert len(database.reminders.docs) == 1 and redis.schedule == {}

    redis.fail = False
    asyncio.run(_job(database, FakeCaller(events=[]), redis).run(CONVERSATION_ID, "job-1"))

    assert len(database.reminders.docs) == 1
    assert list(redis.schedule) == [database.reminders.docs[0]["scheduledOccurrenceId"]]


def test_empty_transcript_is_skipped_without_llm_calls():
    database, redis = FakeDatabase(), FakeRedis()
    caller = FakeCaller()
    result = asyncio.run(_job(database, caller, redis, lines=[]).run(CONVERSATION_ID, "job-1"))

    assert result == {"status": "SKIPPED", "reason": "empty_transcript"}
    assert caller.calls == []
