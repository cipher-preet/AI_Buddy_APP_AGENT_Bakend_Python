"""Real Redis 7 checks. This file connects only to 127.0.0.1:6399."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

import pytest
from redis.asyncio import Redis
from apps.api_gateway.config.setting import settings
from services.queue.job_lease import (
    StaleOwnershipError,
    acquire_lease,
    logical_job_key,
    push_lease,
    remember_completion,
    require_active_lease,
    reset_lease,
)
from services.queue.mongo_fence import FENCE_FIELD, fenced_update_one
from services.queue.streams import EventEnvelope, RedisStreamConsumer

pytestmark = pytest.mark.integration

HOST = "127.0.0.1"
PORT = 6399


def _envelope(event_type: str, **payload) -> EventEnvelope:
    conversation_id = str(payload.get("conversationId") or f"conv-{uuid.uuid4().hex[:8]}")
    return EventEnvelope(
        eventType=event_type,
        correlationId=conversation_id,
        userId=str(payload.pop("userId", "user-redis7")),
        spaceId="space-redis7",
        conversationId=conversation_id,
        payload=payload,
    )


async def _local_redis() -> Redis:
    client = Redis(host=HOST, port=PORT, decode_responses=True, socket_connect_timeout=2)
    try:
        info = await client.info("server")
    except Exception:
        await client.aclose()
        pytest.skip(f"Redis 7 is not accepting connections on {HOST}:{PORT}")
    version = str(info.get("redis_version") or "")
    pool = client.connection_pool.connection_kwargs
    if pool.get("host") != HOST or int(pool.get("port")) != PORT or not version.startswith("7."):
        await client.aclose()
        pytest.fail(f"refusing Redis {version} at {pool.get('host')}:{pool.get('port')}")
    return client


@pytest.fixture
async def redis():
    client = await _local_redis()
    created: list[str] = []
    client.test_streams = created  # type: ignore[attr-defined]
    yield client
    cleaner = client
    try:
        await cleaner.ping()
    except Exception:
        cleaner = Redis(host=HOST, port=PORT, decode_responses=True, socket_connect_timeout=2, socket_timeout=2)
    try:
        cursor = 0
        while True:
            cursor, keys = await cleaner.scan(cursor=cursor, match="buddy:*", count=200)
            if keys:
                await cleaner.delete(*keys)
            if cursor == 0:
                break
        for stream in created:
            await cleaner.delete(stream)
    finally:
        await cleaner.aclose()
        if cleaner is not client:
            await client.aclose()


def _stream(client: Redis) -> tuple[str, str]:
    name = f"phase2a2:{uuid.uuid4().hex}"
    client.test_streams.append(name)  # type: ignore[attr-defined]
    return name, "phase2a2-workers"


def _consumer(client: Redis, stream: str, group: str, name: str, handler, *, max_retries: int = 5) -> RedisStreamConsumer:
    consumer = RedisStreamConsumer(
        stream=stream,
        group=group,
        consumer_name=name,
        handler=handler,
        concurrency=2,
        max_retries=max_retries,
        redis=client,
    )
    if max_retries == 0:
        consumer.max_retries = 0
    return consumer


async def test_redis7_acquire_script_allows_one_owner(redis: Redis):
    event = _envelope("stt.requested", jobId=f"job-{uuid.uuid4().hex}")
    decisions = await asyncio.gather(
        *[acquire_lease(redis, event, owner=f"worker-{index}", message_id=f"{index}-0") for index in range(8)]
    )
    runs = [item for item in decisions if item.action == "run"]
    assert len(runs) == 1
    assert sorted(item.action for item in decisions).count("busy") == 7
    fence_ttl = await redis.pttl("buddy:job-fence:global")
    assert fence_ttl == -1
    renewed = await runs[0].lease.redis.eval(
        "return redis.call('PTTL', KEYS[1])",
        1,
        f"buddy:job-lease:{runs[0].lease.logical_key}",
    )
    assert int(renewed) > 0
    assert await remember_completion(redis, runs[0].lease) is True
    replay = await acquire_lease(redis, event.model_copy(update={"eventId": "replay"}), owner="worker-replay", message_id="9-0")
    assert replay.action == "completed"


async def test_stream_commands_keep_delivery_count_and_ack(redis: Redis):
    stream, group = _stream(redis)
    await redis.xgroup_create(stream, group, id="0", mkstream=True)
    message_id = await redis.xadd(stream, {"event": "{}"})
    read = await redis.xreadgroup(groupname=group, consumername="worker-a", streams={stream: ">"}, count=1)
    assert read[0][1][0][0] == message_id
    pending = await redis.xpending_range(stream, group, min=message_id, max=message_id, count=1)
    assert pending[0]["times_delivered"] == 1
    assert pending[0]["consumer"] == "worker-a"
    claimed = await redis.xclaim(stream, group, "worker-a", 0, [message_id], retrycount=1, justid=True)
    assert message_id in claimed
    pending = await redis.xpending_range(stream, group, min=message_id, max=message_id, count=1)
    assert pending[0]["times_delivered"] == 1
    assert pending[0]["consumer"] == "worker-a"
    await redis.xautoclaim(stream, group, "worker-b", 0, start_id="0-0", count=1)
    pending = await redis.xpending_range(stream, group, min=message_id, max=message_id, count=1)
    assert pending[0]["times_delivered"] == 2
    assert pending[0]["consumer"] == "worker-b"
    assert await redis.xack(stream, group, message_id) == 1
    assert await redis.xpending_range(stream, group, min=message_id, max=message_id, count=1) == []


async def test_heartbeat_does_not_steal_or_increase_deliveries(redis: Redis, monkeypatch):
    monkeypatch.setattr(settings, "JOB_LEASE_HEARTBEAT_SECONDS", 0.05)
    monkeypatch.setattr(settings, "JOB_LEASE_TTL_SECONDS", 30)
    stream, group = _stream(redis)
    started = asyncio.Event()
    release = asyncio.Event()

    async def handler(event: EventEnvelope) -> None:
        started.set()
        await release.wait()

    consumer = _consumer(redis, stream, group, "worker-a", handler)
    await redis.xgroup_create(stream, group, id="0", mkstream=True)
    event = _envelope("stt.requested", jobId=f"job-{uuid.uuid4().hex}")
    await redis.xadd(stream, {"event": event.model_dump_json()})
    messages = await consumer.read_batch(block_ms=10)
    message_id = messages[0][1][0][0]
    task = asyncio.create_task(consumer.dispatch_messages(messages))
    await asyncio.wait_for(started.wait(), timeout=2)
    await asyncio.sleep(0.25)
    pending = await redis.xpending_range(stream, group, min=message_id, max=message_id, count=1)
    assert pending[0]["consumer"] == "worker-a"
    assert pending[0]["times_delivered"] == 1
    release.set()
    await asyncio.wait_for(task, timeout=2)
    assert await redis.xpending_range(stream, group, min=message_id, max=message_id, count=1) == []


async def test_one_worker_runs_a_job_and_a_crash_can_be_recovered(redis: Redis, monkeypatch):
    monkeypatch.setattr(settings, "JOB_LEASE_TTL_SECONDS", 1)
    monkeypatch.setattr(settings, "REDIS_CLAIM_IDLE_MS", 1000)
    stream, group = _stream(redis)
    active = 0
    max_active = 0
    calls = 0
    started = asyncio.Event()
    hold_first = True

    async def handler(event: EventEnvelope) -> None:
        nonlocal active, max_active, calls, hold_first
        active += 1
        max_active = max(max_active, active)
        calls += 1
        started.set()
        try:
            if hold_first and calls == 1:
                await asyncio.Event().wait()
        finally:
            active -= 1

    first = _consumer(redis, stream, group, "worker-a", handler)
    second = _consumer(redis, stream, group, "worker-b", handler)
    await redis.xgroup_create(stream, group, id="0", mkstream=True)
    event = _envelope("stt.requested", jobId=f"job-{uuid.uuid4().hex}")
    await redis.xadd(stream, {"event": event.model_dump_json()})
    messages = await first.read_batch(block_ms=10)
    dispatch = asyncio.create_task(first.dispatch_messages(messages))
    await asyncio.wait_for(started.wait(), timeout=2)
    worker = next(iter(first._handler_tasks))
    worker.cancel()
    with pytest.raises(asyncio.CancelledError):
        await worker
    await asyncio.wait_for(dispatch, timeout=2)
    message_id = messages[0][1][0][0]
    assert message_id in {
        row["message_id"]
        for row in await redis.xpending_range(stream, group, min="-", max="+", count=10)
    }
    hold_first = False
    await asyncio.sleep(1.3)
    await second.claim_stale()
    assert calls == 2
    assert max_active == 1
    assert await redis.xpending_range(stream, group, min=message_id, max=message_id, count=1) == []


def _redis_binary() -> tuple[str, str]:
    root = rf"{__import__('os').environ['TEMP']}\kukunotes-redis749\Redis-7.4.9-Windows-x64-msys2"
    return root, root + r"\redis-server.exe"


async def _restart_local_redis() -> None:
    import subprocess

    root, binary = _redis_binary()
    conf = "redis-6399-aof.conf"
    with open(root + "\\" + conf, "w", encoding="ascii") as handle:
        handle.write("bind 127.0.0.1\nport 6399\nprotected-mode yes\nappendonly yes\ndir ./data\n")
    subprocess.Popen(
        [binary, conf],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(30):
        try:
            probe = Redis(host=HOST, port=PORT, socket_connect_timeout=0.2, socket_timeout=0.2)
            await probe.ping()
            await probe.aclose()
            return
        except Exception:
            await asyncio.sleep(0.15)
    raise RuntimeError("disposable Redis did not return on 127.0.0.1:6399")


async def test_redis_outage_leaves_the_message_pending(redis: Redis):
    import subprocess

    stream, group = _stream(redis)
    worker = Redis(
        host=HOST,
        port=PORT,
        decode_responses=True,
        socket_connect_timeout=1,
        socket_timeout=1,
    )
    started = asyncio.Event()
    continue_handler = asyncio.Event()

    async def handler(event: EventEnvelope) -> None:
        started.set()
        await continue_handler.wait()
        await require_active_lease()

    consumer = _consumer(worker, stream, group, "worker-a", handler)
    await redis.xgroup_create(stream, group, id="0", mkstream=True)
    event = _envelope("stt.requested", jobId=f"job-{uuid.uuid4().hex}")
    message_id = await redis.xadd(stream, {"event": event.model_dump_json()})
    messages = await consumer.read_batch(block_ms=10)
    await consumer._semaphore.acquire()
    task = asyncio.create_task(consumer._handle_message(messages[0][1][0][0], messages[0][1][0][1]))
    await asyncio.wait_for(started.wait(), timeout=2)
    await redis.config_set("appendonly", "yes")
    listener = subprocess.run(
        ["netstat", "-ano"],
        check=False,
        capture_output=True,
        text=True,
    )
    pid = None
    for line in listener.stdout.splitlines():
        if f"127.0.0.1:{PORT}" in line and "LISTENING" in line:
            pid = line.split()[-1]
            break
    if pid is None:
        raise RuntimeError("no Redis listener on 127.0.0.1:6399")
    subprocess.run(["taskkill", "/PID", pid, "/F"], check=False, capture_output=True)
    continue_handler.set()
    try:
        await asyncio.wait_for(task, timeout=5)
    finally:
        await _restart_local_redis()
        await worker.aclose()
    fresh = Redis(host=HOST, port=PORT, decode_responses=True, socket_connect_timeout=2, socket_timeout=2)
    try:
        pending = await fresh.xpending_range(stream, group, min=message_id, max=message_id, count=1)
    finally:
        await fresh.aclose()
    assert pending
    assert pending[0]["message_id"] == message_id


async def test_retry_and_dead_letter_are_written_before_ack(redis: Redis):
    stream, group = _stream(redis)
    ops: list[tuple[str, str]] = []

    class Recording(Redis):
        async def xadd(self, name, fields, *args, **kwargs):
            ops.append(("xadd", str(name)))
            return await super().xadd(name, fields, *args, **kwargs)

        async def xack(self, name, groupname, *ids, **kwargs):
            ops.append(("xack", str(name)))
            return await super().xack(name, groupname, *ids, **kwargs)

    client = Recording(host=HOST, port=PORT, decode_responses=True, socket_connect_timeout=2)
    await client.info("server")

    async def fail(event: EventEnvelope) -> None:
        raise RuntimeError("try again")

    consumer = _consumer(client, stream, group, "worker-a", fail, max_retries=5)
    consumer._uses_default_redis = True
    await redis.xgroup_create(stream, group, id="0", mkstream=True)
    event = _envelope("stt.requested", jobId=f"job-{uuid.uuid4().hex}")
    message_id = await redis.xadd(stream, {"event": event.model_dump_json()})
    messages = await consumer.read_batch(block_ms=10)
    await consumer._semaphore.acquire()
    await consumer._handle_message(messages[0][1][0][0], messages[0][1][0][1])
    retry_at = ops.index(("xadd", settings.REDIS_RETRY_STREAM))
    ack_at = ops.index(("xack", stream))
    assert retry_at < ack_at
    assert await redis.xpending_range(stream, group, min=message_id, max=message_id, count=1) == []
    assert await redis.xlen(settings.REDIS_RETRY_STREAM) >= 1

    ops.clear()

    async def fatal(event: EventEnvelope) -> None:
        raise RuntimeError("permanent")

    dead = _consumer(client, stream, group, "worker-b", fatal, max_retries=0)
    event = _envelope("stt.requested", jobId=f"job-{uuid.uuid4().hex}")
    message_id = await redis.xadd(stream, {"event": event.model_dump_json()})
    messages = await dead.read_batch(block_ms=10)
    await dead._semaphore.acquire()
    await dead._handle_message(messages[0][1][0][0], messages[0][1][0][1])
    dead_at = ops.index(("xadd", settings.REDIS_DEAD_LETTER_STREAM))
    ack_at = ops.index(("xack", stream))
    assert dead_at < ack_at
    assert await redis.xpending_range(stream, group, min=message_id, max=message_id, count=1) == []
    await client.aclose()


async def test_stages_reprocess_and_force_do_not_share_completion_or_a_lower_fence(redis: Redis):
    conversation_id = f"conv-{uuid.uuid4().hex[:8]}"
    processing = _envelope(
        "conversation.processing.requested",
        conversationId=conversation_id,
        processingVersion=1,
    )
    finalization = _envelope("conversation.finalization.requested", conversationId=conversation_id)
    reprocess = _envelope(
        "conversation.processing.requested",
        conversationId=conversation_id,
        processingVersion=1,
        reprocess=True,
    )
    scheduled = _envelope(
        "daily.briefing.requested",
        userId=f"user-{uuid.uuid4().hex[:8]}",
        dateKey="2026-11-04",
        conversationId=f"daily-{uuid.uuid4().hex[:8]}",
    )
    forced = _envelope(
        "daily.briefing.requested",
        userId=scheduled.userId,
        dateKey="2026-11-04",
        conversationId=scheduled.conversationId,
        forced=True,
    )
    forced.createdAt = datetime(2026, 11, 4, 18, 30, tzinfo=timezone.utc)
    assert len({
        logical_job_key(processing),
        logical_job_key(finalization),
        logical_job_key(reprocess),
        logical_job_key(scheduled),
        logical_job_key(forced),
    }) == 5

    first = await acquire_lease(redis, processing, owner="worker-a", message_id="1-0")
    scope = push_lease(first.lease)
    try:
        await require_active_lease()
    finally:
        reset_lease(scope)
    await remember_completion(redis, first.lease)
    assert (await acquire_lease(redis, processing.model_copy(update={"eventId": "same"}), owner="worker-b", message_id="2-0")).action == "completed"

    stage = await acquire_lease(redis, finalization, owner="worker-c", message_id="3-0")
    later = await acquire_lease(redis, reprocess, owner="worker-d", message_id="4-0")
    assert stage.action == "run"
    assert later.action == "run"
    assert int(later.token) > int(stage.token) > int(first.token)
    assert await redis.pttl("buddy:job-fence:global") == -1

    from motor.motor_asyncio import AsyncIOMotorClient

    mongo = AsyncIOMotorClient("mongodb://127.0.0.1:27017", serverSelectionTimeoutMS=2000, directConnection=True)
    database_name = "kukunotes_phase2a2_verify"
    try:
        hello = await mongo.admin.command("hello")
        if mongo.address is None or mongo.address[0] not in {"127.0.0.1", "::1"} or hello.get("setName"):
            pytest.fail("refusing non-local MongoDB for the fence check")
        collection = mongo[database_name].conversations
        await collection.insert_one({"_id": conversation_id, "status": "PROCESSING"})
        scope = push_lease(later.lease)
        try:
            result = await fenced_update_one(
                collection,
                {"_id": conversation_id},
                {"$set": {"status": "PROCESSING", "stage": "reprocess"}},
            )
        finally:
            reset_lease(scope)
        assert result.matched_count == 1
        saved = await collection.find_one({"_id": conversation_id})
        assert saved["stage"] == "reprocess"
        assert saved[FENCE_FIELD] == int(later.token)
        stale = first.lease
        stale.released = False
        stale.lost = False
        nested = push_lease(stale)
        try:
            with pytest.raises(StaleOwnershipError) as caught:
                await fenced_update_one(
                    collection,
                    {"_id": conversation_id},
                    {"$set": {"stage": "stale"}},
                )
            assert caught.value.reason == "stale_fence"
        finally:
            reset_lease(nested)
        saved = await collection.find_one({"_id": conversation_id})
        assert saved["stage"] == "reprocess"
    finally:
        await mongo.drop_database(database_name)
        mongo.close()

    scheduled_lease = await acquire_lease(redis, scheduled, owner="worker-a", message_id="5-0")
    assert await remember_completion(redis, scheduled_lease.lease) is True
    assert (await acquire_lease(redis, forced, owner="worker-b", message_id="6-0")).action == "run"
    assert (
        await acquire_lease(redis, scheduled.model_copy(update={"eventId": "replay"}), owner="worker-c", message_id="7-0")
    ).action == "completed"
