from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

from bson import ObjectId

from apps.api_gateway.config.setting import settings
from services.daily_briefing.briefing_llm import STAGE_SYNTHESIS, STAGE_WINDOW, parse_model_route
from services.daily_briefing.context import build_context
from services.daily_briefing.jobs import DailyBriefingJobHandler
from services.daily_briefing.pipeline_v2 import PIPELINE_VERSION_V2, DailyBriefingPipelineV2
from services.daily_briefing.schemas import (
    BriefingDraft,
    DraftAgendaNote,
    DraftFocus,
    DraftFollowUp,
    DraftInsight,
    DraftItem,
    DraftTaskPriority,
    WindowDigest,
)
from services.daily_briefing.sources import ActivitySource, reminder_occurs_on
from services.daily_briefing.timezones import local_day_bounds_utc
from services.prompts.loader import load_prompt
from services.queue.streams import EventEnvelope
from tests.daily_briefing_fakes import FakeCollection, FakeDatabase

DATE_KEY = "2026-09-09"
PLAN_KEY = "2026-09-10"
TZ = "Asia/Kolkata"
START, END = local_day_bounds_utc(DATE_KEY, TZ)


def _at(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 9, 9, hour, minute, tzinfo=timezone.utc)


class BriefingFakeDatabase(FakeDatabase):
    def __init__(self):
        super().__init__()
        self.chat_sessions = FakeCollection()
        self.chat_message_store = FakeCollection()
        self.spaces = FakeCollection()

    async def list_collection_names(self):
        return [*(await super().list_collection_names()), "chat_sessions", "chat_message_store", "spaces"]


def _chat(session_id: str, kind: str, content: str, at: datetime) -> dict:
    return {
        "_id": ObjectId.from_datetime(at),
        "SessionId": session_id,
        "History": json.dumps({"type": kind, "data": {"content": content}}),
    }


def seeded_database() -> BriefingFakeDatabase:
    db = BriefingFakeDatabase()
    db.spaces.docs = [{"_id": "space-1", "spacename": "Launch"}]
    db.transcript_chunks.docs = [
        {
            "_id": "chunk-1",
            "userId": "user-1",
            "normalizedText": "Speaker 0: Rahul said the pricing deck must reach the client by Thursday.",
            "sttStatus": "completed",
            "conversationId": "conv-1",
            "createdAt": _at(5),
        },
        {
            "_id": "chunk-2",
            "userId": "user-1",
            "normalizedText": "We decided to ship the onboarding flow behind a feature flag.",
            "sttStatus": "completed",
            "conversationId": "conv-1",
            "createdAt": _at(6),
        },
    ]
    db.tasks.docs = [
        {
            "_id": "task-overdue",
            "userId": "user-1",
            "title": "Send pricing deck",
            "status": "pending",
            "priority": "High",
            "dueDate": "2026-09-08",
            "spaceId": "space-1",
            "createdAt": datetime(2026, 9, 1, tzinfo=timezone.utc),
            "updatedAt": datetime(2026, 9, 2, tzinfo=timezone.utc),
        },
        {
            "_id": "task-later",
            "userId": "user-1",
            "title": "Refactor settings page",
            "status": "pending",
            "createdAt": datetime(2026, 9, 3, tzinfo=timezone.utc),
            "updatedAt": datetime(2026, 9, 3, tzinfo=timezone.utc),
        },
        {
            "_id": "task-done",
            "userId": "user-1",
            "title": "Review QA report",
            "status": "completed",
            "createdAt": _at(4),
            "updatedAt": _at(7),
        },
        {
            "_id": "task-deleted",
            "userId": "user-1",
            "title": "Old idea",
            "status": "pending",
            "deletedAt": datetime(2026, 9, 5, tzinfo=timezone.utc),
            "createdAt": datetime(2026, 9, 1, tzinfo=timezone.utc),
            "updatedAt": datetime(2026, 9, 5, tzinfo=timezone.utc),
        },
    ]
    db.notes.docs = [
        {"_id": "note-1", "userId": "user-1", "title": "Onboarding metrics", "body": "Drop-off at step 3 is 40%.", "createdAt": _at(8)}
    ]
    db.calendar_events.docs = [
        {"_id": "event-1", "userId": "user-1", "title": "Client sync", "dateKey": PLAN_KEY, "startTimeLabel": "11:00 AM", "endTimeLabel": "11:30 AM"},
        {"_id": "event-0", "userId": "user-1", "title": "Standup", "dateKey": PLAN_KEY, "startTimeLabel": "9:30 AM"},
    ]
    db.reminders.docs = [
        {"_id": "rem-daily", "userId": "user-1", "title": "Drink water", "dateKey": "2026-09-01", "repeat": "daily", "timeLabel": "10:00 AM"},
        {"_id": "rem-weekly-off", "userId": "user-1", "title": "Weekly report", "dateKey": "2026-09-07", "repeat": "weekly", "timeLabel": "5:00 PM"},
    ]
    db.chat_sessions.docs = [{"_id": "session-1", "userId": "user-1", "title": "Launch plan", "updatedAt": _at(9)}]
    db.chat_message_store.docs = [
        _chat("session-1", "human", "Remind me to follow up with Priya about the contract", _at(9, 0)),
        _chat("session-1", "ai", "Could not reach Buddy API right now", _at(9, 1)),
        _chat("session-1", "ai", "Sure, I noted the Priya contract follow-up.", _at(9, 2)),
    ]
    return db


def load_bundle(db, include_plan_context=True):
    return asyncio.run(ActivitySource(db).load("user-1", DATE_KEY, START, END, include_plan_context=include_plan_context))


class ScriptedCaller:
    def __init__(self, draft=None, digest=None, fail_synthesis=False):
        self.draft = draft
        self.digest = digest
        self.fail_synthesis = fail_synthesis
        self.calls: list[tuple[str, str, dict]] = []

    async def generate(self, stage, prompt_name, schema, payload):
        self.calls.append((stage, prompt_name, payload))
        if stage == STAGE_WINDOW:
            return self.digest or WindowDigest(summary="window")
        if self.fail_synthesis:
            raise RuntimeError("krutrim down")
        return self.draft or BriefingDraft()


def test_reminder_recurrence_rules():
    assert reminder_occurs_on({"dateKey": "2026-09-01", "repeat": "daily"}, PLAN_KEY)
    assert reminder_occurs_on({"dateKey": "2026-09-03", "repeat": "weekly"}, PLAN_KEY)
    assert not reminder_occurs_on({"dateKey": "2026-09-07", "repeat": "weekly"}, PLAN_KEY)
    assert reminder_occurs_on({"dateKey": "2026-09-07", "repeat": "weekdays"}, PLAN_KEY)
    assert not reminder_occurs_on({"dateKey": "2026-09-12", "repeat": "daily"}, PLAN_KEY)
    assert reminder_occurs_on({"dateKey": PLAN_KEY, "repeat": "once"}, PLAN_KEY)


def test_sources_plan_context_is_opt_in():
    db = seeded_database()
    plain = load_bundle(db, include_plan_context=False)
    assert plain.chats == [] and plain.openTasks == [] and plain.planEvents == []

    bundle = load_bundle(db)
    assert {task["id"] for task in bundle.openTasks} == {"task-overdue", "task-later"}
    assert {item["id"] for item in bundle.planEvents} == {"event-0", "event-1"}
    assert [item["id"] for item in bundle.planReminders] == ["rem-daily"]
    assert [item["role"] for item in bundle.chats] == ["user", "assistant"]
    assert bundle.chats[0]["createdAt"].tzinfo is not None
    overdue = next(task for task in bundle.openTasks if task["id"] == "task-overdue")
    assert overdue["space"] == "Launch"
    assert bundle.planDateKey == PLAN_KEY


def test_context_ranks_backlog_and_sorts_agenda():
    ctx = build_context(load_bundle(seeded_database()), DATE_KEY, TZ, open_task_limit=10)
    assert ctx.open_tasks[0]["id"] == "task-overdue"
    assert ctx.open_tasks[0]["dueStatus"] == "overdue"
    assert [item["title"] for item in ctx.agenda] == ["Standup", "Drink water", "Client sync"]
    assert [item["ref"][0] for item in ctx.timeline] == ["S", "S", "C", "C"]
    assert ctx.stats.overdueTasks == 1
    assert ctx.stats.meetings == 2 and ctx.stats.reminders == 1
    assert ctx.stats.completedTasks == 1
    assert ctx.stats.chatMessages == 1
    facts = ctx.facts()
    assert facts["planWeekday"] == "Thursday"
    assert "timeline" not in facts


def test_v2_grounding_drops_invented_refs_people_and_speaker_labels():
    ctx = build_context(load_bundle(seeded_database()), DATE_KEY, TZ, open_task_limit=10)
    task_ref = {task["id"]: task["ref"] for task in ctx.open_tasks}
    voice_ref = ctx.timeline[0]["ref"]
    event_ref = next(item["ref"] for item in ctx.agenda if item["title"] == "Client sync")
    draft = BriefingDraft(
        headline="Speaker 0 needs the pricing deck today",
        overview="Ship the deck before the client sync.",
        focus=[
            DraftFocus(title="", why="Overdue and promised", timeHint="Before 11:00 AM", taskRef=task_ref["task-overdue"]),
            DraftFocus(title="Send pricing deck", why="duplicate"),
        ],
        taskPriorities=[
            DraftTaskPriority(ref=task_ref["task-later"], priority="low", reason="Can wait"),
            DraftTaskPriority(ref="T99", priority="high", reason="invented"),
        ],
        agendaNotes=[DraftAgendaNote(ref=event_ref, prep="Bring the updated pricing deck")],
        followUps=[
            DraftFollowUp(title="Send deck to Rahul", person="Rahul", refs=[voice_ref]),
            DraftFollowUp(title="Call Vikram", person="Vikram", refs=["S99"]),
        ],
        decisions=[DraftItem(title="Onboarding behind a feature flag", refs=[ctx.timeline[1]["ref"]])],
        insights=[DraftInsight(title="Pricing keeps slipping", body="The deck is overdue.", refs=[task_ref["task-overdue"]])],
        people=["Rahul", "Vikram", "Speaker 1"],
    )
    result = asyncio.run(
        DailyBriefingPipelineV2(caller=ScriptedCaller(draft=draft)).generate(
            user_id="user-1", date_key=DATE_KEY, timezone_name=TZ, bundle=load_bundle(seeded_database())
        )
    )
    synthesis, stats, window_count = result
    assert window_count == 0
    assert "Speaker" not in synthesis.headline
    assert [item.title for item in synthesis.focus] == ["Send pricing deck"]
    assert synthesis.focus[0].taskId == "task-overdue"
    assert synthesis.tasks[0].id == "task-later" and synthesis.tasks[0].priority == "low"
    assert synthesis.tasks[1].dueStatus == "overdue" and synthesis.tasks[1].space == "Launch"
    assert next(card for card in synthesis.agenda if card.id == "event-1").prep == "Bring the updated pricing deck"
    assert [item.title for item in synthesis.followUps] == ["Send deck to Rahul"]
    assert synthesis.followUps[0].evidence[0].sourceId == "chunk-1"
    assert synthesis.people == ["Rahul"]
    assert synthesis.insights[0].sourceType == "task"
    assert synthesis.risks and synthesis.risks[0].title.startswith("Overdue")
    assert [item.title for item in synthesis.completed] == ["Review QA report"]
    assert synthesis.planDateKey == PLAN_KEY
    assert stats.chatMessageCount == 2 and stats.openTaskCount == 2


def test_v2_falls_back_deterministically_when_llm_fails():
    synthesis, _stats, _count = asyncio.run(
        DailyBriefingPipelineV2(caller=ScriptedCaller(fail_synthesis=True)).generate(
            user_id="user-1", date_key=DATE_KEY, timezone_name=TZ, bundle=load_bundle(seeded_database())
        )
    )
    assert synthesis.headline.startswith("Thursday")
    assert synthesis.focus[0].taskId == "task-overdue"
    assert synthesis.risks[0].evidence[0].sourceId == "task-overdue"
    assert synthesis.overview
    assert len(synthesis.agenda) == 3


def test_v2_uses_window_digests_for_long_days(monkeypatch):
    monkeypatch.setattr(settings, "DAILY_BRIEFING_DIRECT_SYNTHESIS_TOKENS", 0)
    caller = ScriptedCaller(digest=WindowDigest(summary="Pricing discussion"))
    _synthesis, _stats, window_count = asyncio.run(
        DailyBriefingPipelineV2(caller=caller).generate(
            user_id="user-1", date_key=DATE_KEY, timezone_name=TZ, bundle=load_bundle(seeded_database())
        )
    )
    stages = [call[0] for call in caller.calls]
    assert window_count >= 1
    assert stages.count(STAGE_WINDOW) == window_count and stages[-1] == STAGE_SYNTHESIS
    synthesis_payload = caller.calls[-1][2]
    assert "windowDigests" in synthesis_payload and "timeline" not in synthesis_payload


def test_v2_job_persists_plan_fields_and_counts_chat_only_days():
    db = BriefingFakeDatabase()
    db.chat_sessions.docs = [{"_id": "session-1", "userId": "user-1", "title": "Plan", "updatedAt": _at(9)}]
    db.chat_message_store.docs = [_chat("session-1", "human", "Plan the investor update for tomorrow", _at(9))]
    handler = DailyBriefingJobHandler(db, pipeline=DailyBriefingPipelineV2(caller=ScriptedCaller()))
    event = EventEnvelope(
        eventId="evt-1",
        eventType="daily_briefing.generate",
        correlationId="corr-1",
        userId="user-1",
        spaceId="",
        conversationId="",
        payload={"dateKey": DATE_KEY, "timezone": TZ},
    )
    asyncio.run(handler.handle(event))
    doc = db.daily_briefings.docs[0]
    assert doc["status"] == "READY"
    assert doc["pipelineVersion"] == PIPELINE_VERSION_V2
    assert doc["planDateKey"] == PLAN_KEY
    assert doc["stats"]["chatMessages"] == 1
    assert "focus" in doc and "agenda" in doc and "risks" in doc


def test_clean_text_strips_refs_and_speaker_labels():
    from services.daily_briefing.grounding import clean_text

    assert clean_text("You fixed the sheet (T1) and checked rates [S3, C2].", 200) == "You fixed the sheet and checked rates."
    assert clean_text("Speaker 0 will call back", 200) == "A participant will call back"


def test_krutrim_caller_skips_truncated_model_and_scopes_reasoning_effort():
    from services.daily_briefing.briefing_llm import KrutrimBriefingCaller

    class FakeProvider:
        name = "krutrim"

        def __init__(self):
            self.requests = []
            self.last_structured_diagnostics = {}

        async def generate_structured(self, request, schema):
            self.requests.append(request)
            truncated = request.model == "gpt-oss-120b"
            self.last_structured_diagnostics = {
                "completionTokens": request.max_tokens if truncated else 50,
                "finishReason": "length" if truncated else "stop",
            }
            return schema(headline=request.model)

    class FakeRouter:
        def __init__(self):
            self.providers = {"krutrim": FakeProvider()}

    router = FakeRouter()
    result = asyncio.run(
        KrutrimBriefingCaller(router=router).generate(STAGE_SYNTHESIS, "daily-briefing-synthesis-v2", BriefingDraft, {})
    )
    requests = router.providers["krutrim"].requests
    assert result.headline == "gemma-4-31b-it"
    assert requests[0].metadata["extra_body"] == {"reasoning_effort": settings.DAILY_BRIEFING_REASONING_EFFORT}
    assert requests[1].metadata["extra_body"] == {}
    assert [request.model for request in requests] == ["gpt-oss-120b", "gemma-4-31b-it"]


def test_v2_prompts_and_model_route():
    for name in ("daily-briefing-window-v2", "daily-briefing-synthesis-v2"):
        prompt = load_prompt(name)
        assert "Do not create canonical Tasks" in prompt
        assert "Speaker 0" in prompt
        assert "Hinglish" in prompt
    assert parse_model_route("krutrim:gpt-oss-120b, krutrim:gpt-oss-120b,bad,krutrim:gemma-4-31b-it") == [
        ("krutrim", "gpt-oss-120b"),
        ("krutrim", "gemma-4-31b-it"),
    ]
