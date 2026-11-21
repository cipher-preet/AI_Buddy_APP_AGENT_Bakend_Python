from __future__ import annotations

import asyncio

import pytest

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
def launched():
    _FakeProducer.calls = []
    _FakeProducer.fail = False
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


def test_force_generate_does_not_queue_a_briefing(launched):
    database = FakeDatabase()
    _seed(database, status="FAILED", headline="keep")

    result = asyncio.run(_force(database))

    assert result["status"] == "SKIPPED"
    assert result["forced"] is False
    assert launched == []
    assert database.daily_briefings.docs[0]["headline"] == "keep"
    assert database.daily_briefings.docs[0]["status"] == "FAILED"
