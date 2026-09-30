from __future__ import annotations

import asyncio

import pytest

from apps.api_gateway.config.setting import settings
from services.conversation.models import utc_now
from services.daily_briefing import force as force_module
from tests.daily_briefing_fakes import FakeDatabase

USER = "user-1"
DATE = "2026-09-29"


class _FakeProducer:
    calls: list = []
    fail = False

    async def publish(self, stream, event):
        if _FakeProducer.fail:
            raise ConnectionError("queue down")
        _FakeProducer.calls.append((stream, event))
        return event.eventId


@pytest.fixture
def launched(monkeypatch):
    _FakeProducer.calls = []
    _FakeProducer.fail = False
    monkeypatch.setattr(force_module, "RedisStreamProducer", _FakeProducer)
    return _FakeProducer.calls


def _seed(database: FakeDatabase, **fields) -> None:
    now = utc_now()
    database.daily_briefings.docs.append(
        {"userId": USER, "dateKey": DATE, "createdAt": now, "updatedAt": now, "claimedAt": now, **fields}
    )


async def _force(database: FakeDatabase) -> dict:
    result = await force_module.force_generate_daily_briefing(database, user_id=USER, date_key=DATE)
    await asyncio.sleep(0)
    return result


def test_ready_briefing_is_not_regenerated(launched):
    database = FakeDatabase()
    _seed(database, status="READY", headline="keep me")

    result = asyncio.run(_force(database))

    assert result["forced"] is False
    assert launched == []
    assert database.daily_briefings.docs[0]["headline"] == "keep me"


def test_in_flight_briefing_does_not_start_second_job(launched):
    database = FakeDatabase()
    _seed(database, status="PROCESSING")

    result = asyncio.run(_force(database))

    assert result["forced"] is False
    assert launched == []


def test_failed_briefing_regenerates_and_counts(launched):
    database = FakeDatabase()
    _seed(database, status="FAILED")

    result = asyncio.run(_force(database))

    assert result["forced"] is True
    assert len(launched) == 1
    doc = database.daily_briefings.docs[0]
    assert doc["status"] == "PENDING"
    assert doc["forceCount"] == 1


def test_force_limit_blocks_extra_paid_runs(launched):
    database = FakeDatabase()
    _seed(database, status="FAILED", forceCount=settings.DAILY_BRIEFING_FORCE_DAILY_LIMIT)

    with pytest.raises(PermissionError):
        asyncio.run(_force(database))
    assert launched == []


def test_skipped_briefing_keeps_count_after_regenerate(launched):
    database = FakeDatabase()
    _seed(database, status="SKIPPED", forceCount=1)

    asyncio.run(_force(database))

    assert len(launched) == 1
    assert database.daily_briefings.docs[0]["forceCount"] == 2


def test_force_job_is_queued_to_briefing_worker(launched):
    database = FakeDatabase()
    _seed(database, status="FAILED")

    asyncio.run(_force(database))

    [(stream, event)] = launched
    assert stream == settings.REDIS_DAILY_BRIEFING_STREAM
    assert event.eventType == "daily.briefing.requested"
    assert event.payload["dateKey"] == DATE


def test_publish_failure_falls_back_to_in_process_run(launched, monkeypatch):
    ran: list = []

    async def fake_job(database, event):
        ran.append(event)

    monkeypatch.setattr(force_module, "_run_force_job", fake_job)
    _FakeProducer.fail = True
    database = FakeDatabase()

    asyncio.run(_force(database))

    assert len(ran) == 1


def test_missing_briefing_defaults_to_yesterday(launched):
    database = FakeDatabase()

    result = asyncio.run(force_module.force_generate_daily_briefing(database, user_id=USER))

    assert result["forced"] is True
    today = force_module.date_key_for(force_module.datetime.now(force_module.timezone.utc), result["timezone"])
    assert result["dateKey"] < today
