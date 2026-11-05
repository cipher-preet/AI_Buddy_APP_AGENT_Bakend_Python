"""Redis 7 retention: pending and unread entries survive a burst past the old cap of 200."""

from __future__ import annotations

import uuid

import pytest
from redis.asyncio import Redis

from apps.api_gateway.config.setting import settings
from services.queue.maintenance import _dead_letter_pending, maintain_conversation_redis, reap_stuck_stream
from services.queue.streams import EventEnvelope, RedisStreamConsumer, RedisStreamProducer

pytestmark = pytest.mark.integration

REDIS_HOST = "127.0.0.1"
REDIS_PORT = 6399
BURST = 500


async def _redis() -> Redis:
    client = Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True, socket_connect_timeout=2, socket_timeout=5)
    try:
        info = await client.info("server")
    except Exception as error:
        await client.aclose()
        pytest.fail(f"Redis 7 is not available on {REDIS_HOST}:{REDIS_PORT}: {type(error).__name__}")
    version = str(info.get("redis_version") or "")
    port = int(info.get("tcp_port") or 0)
    if not version.startswith("7.") or port != REDIS_PORT:
        await client.aclose()
        pytest.fail(f"refusing Redis {version} on port {port}")
    return client


def _suffix() -> str:
    return uuid.uuid4().hex[:10]


def _event(marker: str) -> EventEnvelope:
    return EventEnvelope(
        eventType="retention.probe",
        correlationId=marker,
        userId="user-retention",
        spaceId="space-retention",
        conversationId=marker,
        payload={"marker": marker},
    )


@pytest.fixture
async def redis():
    client = await _redis()
    created: list[str] = []
    yield client, created
    if created:
        await client.delete(*created)
    await client.aclose()


async def _burst(producer: RedisStreamProducer, stream: str, marker: str) -> None:
    for index in range(BURST):
        await producer.publish(stream, _event(f"{marker}-{index}"), maxlen=200)


async def test_pending_entry_survives_a_burst_past_the_old_cap(redis):
    client, created = redis
    stream = f"phase2e1:pending:{_suffix()}"
    created.append(stream)
    group = "retention-a"
    await client.xgroup_create(stream, group, id="0", mkstream=True)
    producer = RedisStreamProducer(client, force_direct=True)
    original = _event("original")
    await producer.publish(stream, original, maxlen=200)
    delivered = await client.xreadgroup(group, "worker-1", {stream: ">"}, count=1)
    message_id = delivered[0][1][0][0]
    await _burst(producer, stream, "later")
    pending = await client.xpending(stream, group)
    claimed = await client.xclaim(stream, group, "worker-2", min_idle_time=0, message_ids=[message_id])
    assert int(pending["pending"]) == 1
    assert claimed and claimed[0][0] == message_id
    body = claimed[0][1]["event"]
    assert "original" in body
    assert await client.xlen(stream) == BURST + 1


async def test_undelivered_backlog_survives_for_every_group(redis):
    client, created = redis
    stream = f"phase2e1:unread:{_suffix()}"
    created.append(stream)
    await client.xgroup_create(stream, "slow", id="0", mkstream=True)
    await client.xgroup_create(stream, "other", id="0", mkstream=True)
    producer = RedisStreamProducer(client, force_direct=True)
    first = _event("first-unread")
    await producer.publish(stream, first, maxlen=200)
    await _burst(producer, stream, "rest")
    for group in ("slow", "other"):
        rows = await client.xreadgroup(group, "worker", {stream: ">"}, count=1)
        body = rows[0][1][0][1]["event"]
        assert "first-unread" in body
        assert int((await client.xpending(stream, group))["pending"]) == 1


async def test_maintenance_does_not_trim_retry_or_a_quiet_group(redis, monkeypatch):
    client, created = redis
    suffix = _suffix()
    names = {
        "REDIS_AUDIO_STREAM": f"phase2e1:audio:{suffix}",
        "REDIS_STT_STREAM": f"phase2e1:stt:{suffix}",
        "REDIS_TRANSCRIPT_READY_STREAM": f"phase2e1:transcript:{suffix}",
        "REDIS_WINDOW_EXTRACTION_STREAM": f"phase2e1:window:{suffix}",
        "REDIS_FINALIZATION_STREAM": f"phase2e1:final:{suffix}",
        "REDIS_PROCESSING_STREAM": f"phase2e1:processing:{suffix}",
        "REDIS_DAILY_BRIEFING_STREAM": f"phase2e1:brief:{suffix}",
        "REDIS_MEETING_MERGE_STREAM": f"phase2e1:merge:{suffix}",
        "REDIS_SCHEDULE_EXTRACTION_STREAM": f"phase2e1:schedule:{suffix}",
        "REDIS_MINDMAP_STREAM": f"phase2e1:mindmap:{suffix}",
        "REDIS_DOCUMENT_STREAM": f"phase2e1:document:{suffix}",
        "REDIS_RETRY_STREAM": f"phase2e1:retry:{suffix}",
        "REDIS_DEAD_LETTER_STREAM": f"phase2e1:dlq-maintain:{suffix}",
    }
    for name, value in names.items():
        monkeypatch.setattr(settings, name, value)
        created.append(value)
    retry = names["REDIS_RETRY_STREAM"]
    for index in range(BURST):
        await client.xadd(retry, {"event": f"retry-{index}"})
    audio = names["REDIS_AUDIO_STREAM"]
    await client.xgroup_create(audio, settings.REDIS_AUDIO_GROUP, id="0", mkstream=True)
    await client.xadd(audio, {"event": "still-there"})
    result = await maintain_conversation_redis(client)
    assert result["trimmed"] == 0
    assert await client.xlen(retry) == BURST
    assert await client.xlen(audio) == 1
    ids = [row[0] for row in await client.xrange(retry)]
    assert ids == sorted(ids)


async def test_reaper_leaves_a_healthy_backlog_in_place(redis):
    client, created = redis
    stream = f"phase2e1:reap:{_suffix()}"
    created.append(stream)
    await client.xgroup_create(stream, "reap-group", id="0", mkstream=True)
    first = await client.xadd(stream, {"event": "keep"})
    delivered = await client.xreadgroup("reap-group", "worker", {stream: ">"}, count=1)
    assert delivered[0][1][0][0] == first
    for index in range(BURST):
        await client.xadd(stream, {"event": str(index)})
    result = await reap_stuck_stream(client, stream, "reap-group", maxlen=200)
    assert result["reaped"] == 0
    assert result["trimmed"] == 0
    assert await client.xlen(stream) == BURST + 1
    assert await client.xrange(stream, min=first, max=first)


async def test_dead_letter_keeps_entries_past_the_old_cap_of_50(redis, monkeypatch):
    client, created = redis
    source = f"phase2e1:src:{_suffix()}"
    dead = f"phase2e1:dead:{_suffix()}"
    created.extend([source, dead])
    monkeypatch.setattr(settings, "REDIS_DEAD_LETTER_STREAM", dead)
    message_id = await client.xadd(source, {"event": '{"eventType":"stt.requested"}'})
    for _ in range(60):
        await _dead_letter_pending(client, stream=source, message_id=message_id, idle_ms=1, deliveries=9)
    assert await client.xlen(dead) == 60
    first = await client.xrange(dead, count=1)
    assert first


async def test_acknowledged_id_can_be_deleted_without_dropping_a_neighbor(redis):
    client, created = redis
    stream = f"phase2e1:ack:{_suffix()}"
    created.append(stream)
    group = "ack-group"
    await client.xgroup_create(stream, group, id="0", mkstream=True)
    producer = RedisStreamProducer(client, force_direct=True)
    await producer.publish(stream, _event("done"), maxlen=200)
    await producer.publish(stream, _event("still-pending"), maxlen=200)
    rows = await client.xreadgroup(group, "worker", {stream: ">"}, count=1)
    done_id = rows[0][1][0][0]
    assert int((await client.xpending(stream, group))["pending"]) == 1
    consumer = RedisStreamConsumer(
        stream,
        group,
        "worker",
        lambda event: None,
        redis=client,
        delete_after_ack=True,
        stream_maxlen=200,
    )
    await consumer._ack_and_cleanup(done_id)
    assert await client.xrange(stream, min=done_id, max=done_id) == []
    remaining = await client.xrange(stream)
    assert len(remaining) == 1
    assert "still-pending" in remaining[0][1]["event"]
    assert int((await client.xpending(stream, group))["pending"]) == 0
