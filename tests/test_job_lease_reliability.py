"""Redis stream ownership: long jobs, crashes, stale commits, and ack safety."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest
from redis.exceptions import ConnectionError

from apps.api_gateway.config.setting import settings
from services.conversation.models import ConversationStatus
from services.conversation.workflow import ConversationProcessingWorkflow
from services.queue.job_lease import (
    Lease,
    StaleOwnershipError,
    acquire_lease,
    logical_job_key,
    push_lease,
    reclaim_min_idle_ms,
    refresh_pending_idle,
    release_lease,
    reset_lease,
)
from services.queue.maintenance import reap_stuck_stream
from services.queue.streams import EventEnvelope, RedisStreamConsumer

STREAM = "buddy:stt:jobs"
GROUP = "stt-workers"


def _event(
    *,
    event_type: str = "stt.requested",
    job_id: str = "job-1",
    attempt: int = 0,
    conversation_id: str = "conv-1",
    event_id: str | None = None,
) -> EventEnvelope:
    payload = {"jobId": job_id, "conversationId": conversation_id, "sequenceNumber": 1}
    kwargs = {}
    if event_id is not None:
        kwargs["eventId"] = event_id
    return EventEnvelope(
        eventType=event_type,
        correlationId=conversation_id,
        userId="user-1",
        spaceId="space-1",
        conversationId=conversation_id,
        attempt=attempt,
        payload=payload,
        **kwargs,
    )


class MemoryRedis:
    """In-process Redis stream and string commands for lease tests. No network."""

    def __init__(self) -> None:
        self.kv: dict[str, tuple[str, float | None]] = {}
        self.logs: dict[str, list[tuple[str, dict]]] = {}
        self.pending: dict[tuple[str, str], dict[str, dict]] = {}
        self.cursors: dict[tuple[str, str], int] = {}
        self.ops: list[tuple[str, str]] = []
        self.autoclaim_hits = 0
        self.xclaim_retrycounts: list[int] = []
        self._seq = 0
        self.broken = False

    def _check(self) -> None:
        if self.broken:
            raise ConnectionError("redis down")

    def _now(self) -> float:
        return time.monotonic()

    def _read(self, key: str) -> str | None:
        item = self.kv.get(key)
        if item is None:
            return None
        value, expiry = item
        if expiry is not None and expiry <= self._now():
            self.kv.pop(key, None)
            return None
        return value

    async def set(self, name, value, nx=False, px=None, ex=None):
        self._check()
        if nx and self._read(name) is not None:
            return None
        if isinstance(value, bytes):
            value = value.decode()
        ttl_ms = px if px is not None else (int(ex * 1000) if ex is not None else None)
        expiry = None if ttl_ms is None else self._now() + (int(ttl_ms) / 1000)
        self.kv[name] = (str(value), expiry)
        return True

    async def get(self, name):
        self._check()
        return self._read(name)

    async def delete(self, *names):
        self._check()
        removed = 0
        for name in names:
            if name in self.kv:
                del self.kv[name]
                removed += 1
        return removed

    async def incr(self, name):
        self._check()
        current = int(self._read(name) or 0) + 1
        expiry = self.kv.get(name, (None, None))[1] if name in self.kv else None
        self.kv[name] = (str(current), expiry)
        return current

    async def pexpire(self, name, ttl):
        self._check()
        current = self._read(name)
        if current is None:
            return False
        self.kv[name] = (current, self._now() + (int(ttl) / 1000))
        return True

    async def eval(self, script, numkeys, *args):
        self._check()
        if "job-lease-" in str(script):
            from services.queue.job_lease import apply_job_lease_eval

            def write(key: str, value: str, px_ms: int) -> None:
                self.kv[key] = (str(value), self._now() + (int(px_ms) / 1000))

            return apply_job_lease_eval(
                self._read,
                write,
                lambda key: self.kv.pop(key, None),
                self._incr_sync,
                self._pexpire_sync,
                script,
                numkeys,
                args,
            )
        key = args[0]
        token = args[1] if len(args) > 1 else None
        current = self._read(key)
        if current == token:
            self.kv.pop(key, None)
            return 1
        return 0

    def _incr_sync(self, name: str) -> int:
        current = int(self._read(name) or 0) + 1
        expiry = self.kv.get(name, (None, None))[1] if name in self.kv else None
        self.kv[name] = (str(current), expiry)
        return current

    def _pexpire_sync(self, name: str, ttl: int) -> bool:
        current = self._read(name)
        if current is None:
            return False
        self.kv[name] = (current, self._now() + (int(ttl) / 1000))
        return True

    async def xgroup_create(self, stream, group, id="0", mkstream=True):
        self.logs.setdefault(stream, [])
        self.pending.setdefault((stream, group), {})
        self.cursors.setdefault((stream, group), 0)
        return True

    async def xadd(self, stream, fields, maxlen=None, approximate=True):
        self._check()
        self._seq += 1
        message_id = f"{self._seq}-0"
        self.logs.setdefault(stream, []).append((message_id, dict(fields)))
        self.ops.append(("xadd", stream))
        return message_id

    async def xack(self, stream, group, message_id):
        self.ops.append(("xack", stream))
        self.pending.get((stream, group), {}).pop(str(message_id), None)
        return 1

    async def xdel(self, stream, *ids):
        self.ops.append(("xdel", stream))
        return len(ids)

    async def xtrim(self, stream, maxlen=0, approximate=True):
        return 0

    async def xrange(self, stream, min="-", max="+", count=None):
        rows = []
        for message_id, fields in self.logs.get(stream, []):
            if min not in {"-", None} and message_id != min:
                continue
            rows.append((message_id, fields))
        return rows[: count or len(rows)]

    async def xreadgroup(self, groupname, consumername, streams, count=10, block=None):
        self._check()
        delivered = []
        for stream, cursor in streams.items():
            if cursor not in {">", b">"}:
                continue
            key = (stream, groupname)
            self.logs.setdefault(stream, [])
            start = self.cursors.get(key, 0)
            batch = self.logs[stream][start : start + count]
            self.cursors[key] = start + len(batch)
            entries = []
            for message_id, fields in batch:
                self.pending.setdefault(key, {})[message_id] = {
                    "consumer": consumername,
                    "fields": fields,
                    "deliveries": 1,
                    "delivered_at": self._now(),
                }
                entries.append((message_id, fields))
            if entries:
                delivered.append((stream, entries))
        return delivered

    async def xpending_range(self, name, groupname, min="-", max="+", count=10, **kwargs):
        self._check()
        rows = []
        for message_id, entry in self.pending.get((name, groupname), {}).items():
            if min not in {"-", None} and message_id != min:
                continue
            rows.append(
                {
                    "message_id": message_id,
                    "consumer": entry["consumer"],
                    "time_since_delivered": int((self._now() - entry["delivered_at"]) * 1000),
                    "times_delivered": entry["deliveries"],
                }
            )
        return rows[:count]

    async def xclaim(self, name, groupname, consumername, min_idle_time, message_ids, retrycount=None, justid=False, **kwargs):
        self._check()
        claimed = []
        pending = self.pending.setdefault((name, groupname), {})
        for message_id in message_ids:
            entry = pending.get(str(message_id))
            if entry is None:
                continue
            idle_ms = (self._now() - entry["delivered_at"]) * 1000
            if idle_ms < int(min_idle_time or 0):
                continue
            if retrycount is None:
                entry["deliveries"] += 1
            else:
                entry["deliveries"] = int(retrycount)
                self.xclaim_retrycounts.append(int(retrycount))
            entry["consumer"] = consumername
            entry["delivered_at"] = self._now()
            claimed.append(str(message_id) if justid else (str(message_id), entry["fields"]))
        return claimed

    async def xautoclaim(self, name, groupname, consumername, min_idle_time, start_id="0-0", count=10):
        self._check()
        stolen = []
        pending = self.pending.get((name, groupname), {})
        for message_id, entry in list(pending.items()):
            idle_ms = (self._now() - entry["delivered_at"]) * 1000
            if idle_ms < int(min_idle_time):
                continue
            entry["deliveries"] += 1
            entry["consumer"] = consumername
            entry["delivered_at"] = self._now()
            stolen.append((message_id, entry["fields"]))
            if len(stolen) >= int(count):
                break
        self.autoclaim_hits += len(stolen)
        return ["0-0", stolen, []]

    def force_idle(self, stream: str, group: str, message_id: str, idle_ms: int) -> None:
        entry = self.pending[(stream, group)][str(message_id)]
        entry["delivered_at"] = self._now() - (idle_ms / 1000)

    def drop_leases(self) -> None:
        for key in list(self.kv):
            if key.startswith("buddy:job-lease:"):
                self.kv.pop(key, None)


def _consumer(redis, name: str, handler, *, max_retries: int | None = None) -> RedisStreamConsumer:
    return RedisStreamConsumer(
        stream=STREAM,
        group=GROUP,
        consumer_name=name,
        handler=handler,
        concurrency=4,
        max_retries=max_retries if max_retries is not None else 5,
        redis=redis,
    )


async def _read(consumer: RedisStreamConsumer, redis: MemoryRedis):
    await redis.xgroup_create(STREAM, GROUP)
    return await consumer.read_batch(block_ms=0)


def test_logical_job_key_ignores_event_id_and_separates_sequences():
    first = _event(event_id="event-a")
    retry = _event(event_id="event-b", attempt=2)
    other = _event(job_id="job-2", event_id="event-c")
    processing = _event(event_type="conversation.processing.requested", job_id="ignored", conversation_id="conv-9")
    again = _event(event_type="conversation.processing.requested", job_id="other", conversation_id="conv-9", event_id="new")
    assert logical_job_key(first) == logical_job_key(retry)
    assert logical_job_key(first) != logical_job_key(other)
    assert logical_job_key(processing) == logical_job_key(again)
    assert logical_job_key(processing) != logical_job_key(first)


def test_reclaim_idle_cannot_undercut_the_lease(monkeypatch):
    monkeypatch.setattr(settings, "JOB_LEASE_TTL_SECONDS", 90)
    monkeypatch.setattr(settings, "REDIS_CLAIM_IDLE_MS", 15000)
    assert reclaim_min_idle_ms() == 90000
    monkeypatch.setattr(settings, "REDIS_CLAIM_IDLE_MS", 120000)
    assert reclaim_min_idle_ms() == 120000


def test_pending_refresh_does_not_increase_delivery_count():
    redis = MemoryRedis()

    async def run():
        await redis.xgroup_create(STREAM, GROUP)
        await redis.xadd(STREAM, {"event": "{}"})
        consumer = _consumer(redis, "worker-a", lambda event: None)
        messages = await _read(consumer, redis)
        message_id = messages[0][1][0][0]
        await redis.xautoclaim(STREAM, GROUP, "worker-b", 0, count=10)
        before = redis.pending[(STREAM, GROUP)][message_id]["deliveries"]
        assert before == 2
        refreshed = await refresh_pending_idle(
            redis,
            stream=STREAM,
            group=GROUP,
            consumer_name="worker-a",
            message_id=message_id,
        )
        assert refreshed is True
        assert redis.pending[(STREAM, GROUP)][message_id]["deliveries"] == before
        assert redis.xclaim_retrycounts[-1] == before

    asyncio.run(run())


def test_reaper_dead_letters_before_ack_and_stops_when_that_write_fails():
    class Redis:
        def __init__(self, fail_add: bool = False):
            self.fail_add = fail_add
            self.ops: list[str] = []

        async def xpending_range(self, stream, group, min="-", max="+", count=50):
            return [{"message_id": "9-0", "time_since_delivered": 20 * 60 * 1000, "times_delivered": 9}]

        async def xrange(self, stream, min="-", max="+", count=1):
            return [("9-0", {"event": '{"eventType":"stt.requested"}'})]

        async def xadd(self, stream, fields, maxlen=None, approximate=True):
            if self.fail_add:
                raise ConnectionError("dead-letter write failed")
            self.ops.append(f"xadd:{stream}")
            return "1-0"

        async def xack(self, stream, group, message_id):
            self.ops.append("xack")
            return 1

        async def xdel(self, stream, message_id):
            self.ops.append("xdel")
            return 1

    async def run():
        redis = Redis()
        result = await reap_stuck_stream(redis, STREAM, GROUP)
        assert result["reaped"] == 1
        assert redis.ops[0].startswith("xadd:")
        assert redis.ops[1:] == ["xack", "xdel"]
        failed = Redis(fail_add=True)
        result = await reap_stuck_stream(failed, STREAM, GROUP)
        assert result["reaped"] == 0
        assert failed.ops == []

    asyncio.run(run())


@pytest.fixture
def fast_lease(monkeypatch):
    """Shorter than a real meeting, long enough that the old 15s idle rule would steal it."""
    monkeypatch.setattr(settings, "JOB_LEASE_TTL_SECONDS", 1)
    monkeypatch.setattr(settings, "JOB_LEASE_HEARTBEAT_SECONDS", 0.05)
    monkeypatch.setattr(settings, "REDIS_CLAIM_IDLE_MS", 1000)


async def test_long_running_job_is_not_reclaimed_or_run_twice(fast_lease):
    redis = MemoryRedis()
    calls: list[str] = []
    commits: list[str] = []
    started = asyncio.Event()

    async def handler(event: EventEnvelope) -> None:
        calls.append(current_owner())
        started.set()
        await asyncio.sleep(1.3)
        commits.append(event.payload["jobId"])

    owner = _consumer(redis, "worker-a", handler)
    rival = _consumer(redis, "worker-b", handler)
    await redis.xadd(STREAM, {"event": _event().model_dump_json()})
    messages = await _read(owner, redis)
    message_id = messages[0][1][0][0]
    task = asyncio.create_task(owner.dispatch_messages(messages))
    await started.wait()
    await asyncio.sleep(1.15)
    await rival.claim_stale()
    # Ack removes the pending entry, so read the delivery count while it is still owned.
    assert redis.autoclaim_hits == 0
    assert redis.pending[(STREAM, GROUP)][message_id]["deliveries"] == 1
    assert redis.xclaim_retrycounts
    assert all(count == 1 for count in redis.xclaim_retrycounts)
    await task

    assert calls == ["worker-a"]
    assert commits == ["job-1"]
    assert any(op == ("xack", STREAM) for op in redis.ops)


async def test_duplicate_delivery_runs_one_handler_then_retired_after_completion():
    redis = MemoryRedis()
    calls: list[str] = []
    release = asyncio.Event()

    async def handler(event: EventEnvelope) -> None:
        calls.append(event.eventId)
        from services.queue.job_lease import mark_current_job_completed

        mark_current_job_completed()
        await release.wait()

    consumer = _consumer(redis, "worker-a", handler)
    first = _event(event_id="one")
    second = _event(event_id="two")
    await redis.xadd(STREAM, {"event": first.model_dump_json()})
    await redis.xadd(STREAM, {"event": second.model_dump_json()})
    messages = await _read(consumer, redis)
    task = asyncio.create_task(consumer.dispatch_messages([(STREAM, [messages[0][1][0]])]))
    while not calls:
        await asyncio.sleep(0.01)
    duplicate = asyncio.create_task(consumer.dispatch_messages([(STREAM, [messages[0][1][1]])]))
    await duplicate
    assert calls == ["one"]
    release.set()
    await task
    assert [op for op in redis.ops if op[0] == "xack"]
    third = asyncio.create_task(consumer.dispatch_messages([(STREAM, [messages[0][1][1]])]))
    await third
    assert calls == ["one"]


async def test_two_workers_only_one_acquires_the_lease():
    redis = MemoryRedis()
    calls: list[str] = []

    async def handler(event: EventEnvelope) -> None:
        calls.append(current_owner())
        await asyncio.sleep(0.05)

    first = _event(event_id="one")
    second = _event(event_id="two")
    worker_a = _consumer(redis, "worker-a", handler)
    worker_b = _consumer(redis, "worker-b", handler)
    await redis.xgroup_create(STREAM, GROUP)
    await asyncio.gather(
        _deliver(worker_a, redis, first),
        _deliver(worker_b, redis, second),
    )
    assert calls == ["worker-a"] or calls == ["worker-b"]
    assert len(calls) == 1


async def test_crashed_worker_leaves_the_message_pending_until_the_lease_expires():
    redis = MemoryRedis()
    calls: list[str] = []
    started = asyncio.Event()

    async def handler(event: EventEnvelope) -> None:
        calls.append(current_owner())
        if current_owner() == "worker-a":
            started.set()
            await asyncio.Event().wait()
        calls.append("recovered")

    worker_a = _consumer(redis, "worker-a", handler)
    worker_b = _consumer(redis, "worker-b", handler)
    await redis.xadd(STREAM, {"event": _event().model_dump_json()})
    messages = await _read(worker_a, redis)
    message_id = messages[0][1][0][0]
    task = asyncio.create_task(worker_a.dispatch_messages(messages))
    await started.wait()
    handler_task = next(iter(worker_a._handler_tasks))
    handler_task.cancel()
    await asyncio.wait([task, handler_task])
    assert not any(op[0] == "xack" for op in redis.ops)
    assert message_id in redis.pending[(STREAM, GROUP)]
    redis.drop_leases()
    redis.force_idle(STREAM, GROUP, message_id, reclaim_min_idle_ms() + 1000)
    await worker_b.claim_stale()
    assert "recovered" in calls
    assert any(op[0] == "xack" for op in redis.ops)


async def test_stale_worker_cannot_commit_after_a_newer_owner():
    redis = MemoryRedis()
    commits: list[str] = []
    started = asyncio.Event()
    release = asyncio.Event()

    async def handler(event: EventEnvelope) -> None:
        owner = current_owner()
        if owner == "worker-a":
            started.set()
            await release.wait()
        try:
            from services.queue.job_lease import require_active_lease

            await require_active_lease()
        except StaleOwnershipError:
            commits.append(f"{owner}:rejected")
            raise
        commits.append(f"{owner}:committed")

    worker_a = _consumer(redis, "worker-a", handler)
    worker_b = _consumer(redis, "worker-b", handler)
    await redis.xadd(STREAM, {"event": _event().model_dump_json()})
    messages = await _read(worker_a, redis)
    message_id = messages[0][1][0][0]
    task = asyncio.create_task(worker_a.dispatch_messages(messages))
    await started.wait()
    redis.drop_leases()
    redis.force_idle(STREAM, GROUP, message_id, reclaim_min_idle_ms() + 1000)
    await worker_b.claim_stale()
    release.set()
    await task
    assert "worker-b:committed" in commits
    assert "worker-a:committed" not in commits
    assert "worker-a:rejected" in commits


async def test_redis_disconnect_blocks_unsafe_commit_without_a_retry_storm(monkeypatch):
    monkeypatch.setattr(settings, "JOB_LEASE_HEARTBEAT_SECONDS", 0.05)
    redis = MemoryRedis()
    commits: list[str] = []
    started = asyncio.Event()
    proceed = asyncio.Event()

    async def handler(event: EventEnvelope) -> None:
        from services.queue.job_lease import current_lease

        lease = current_lease()
        started.set()
        await proceed.wait()
        if lease is None or lease.lost:
            raise StaleOwnershipError(lease.logical_key if lease else "missing")
        commits.append("committed")

    consumer = _consumer(redis, "worker-a", handler)
    consumer._uses_default_redis = True
    await redis.xadd(STREAM, {"event": _event().model_dump_json()})
    messages = await _read(consumer, redis)
    task = asyncio.create_task(consumer.dispatch_messages(messages))
    await started.wait()
    redis.broken = True
    await asyncio.sleep(0.35)
    proceed.set()
    await task
    assert commits == []
    assert not any(op[0] == "xack" for op in redis.ops)
    assert not any(op[1] == settings.REDIS_RETRY_STREAM for op in redis.ops)
    assert not any(op[1] == settings.REDIS_DEAD_LETTER_STREAM for op in redis.ops)


async def test_completed_job_redelivery_does_not_run_again():
    redis = MemoryRedis()
    calls = 0

    async def handler(event: EventEnvelope) -> None:
        nonlocal calls
        calls += 1
        from services.queue.job_lease import mark_current_job_completed

        mark_current_job_completed()

    consumer = _consumer(redis, "worker-a", handler)
    await redis.xadd(STREAM, {"event": _event(event_id="one").model_dump_json()})
    first = await _read(consumer, redis)
    await consumer.dispatch_messages(first)
    await redis.xadd(STREAM, {"event": _event(event_id="two").model_dump_json()})
    # The cursor already passed the first entry. Read the new one directly.
    consumer_b = _consumer(redis, "worker-b", handler)
    second = await consumer_b.read_batch(block_ms=0)
    await consumer_b.dispatch_messages(second)
    assert calls == 1
    assert sum(1 for op in redis.ops if op[0] == "xack") == 2


async def test_retryable_failure_acks_only_after_retry_handoff():
    redis = MemoryRedis()

    async def handler(event: EventEnvelope) -> None:
        raise RuntimeError("provider timeout")

    consumer = _consumer(redis, "worker-a", handler, max_retries=5)
    consumer._uses_default_redis = True
    await redis.xadd(STREAM, {"event": _event(attempt=0).model_dump_json()})
    messages = await _read(consumer, redis)
    await consumer.dispatch_messages(messages)
    kinds = [op[0] + ":" + op[1] for op in redis.ops]
    retry_at = kinds.index(f"xadd:{settings.REDIS_RETRY_STREAM}")
    ack_at = kinds.index(f"xack:{STREAM}")
    assert retry_at < ack_at


async def test_retry_exhaustion_is_dead_lettered_before_ack():
    redis = MemoryRedis()

    async def handler(event: EventEnvelope) -> None:
        raise RuntimeError("still failing")

    consumer = _consumer(redis, "worker-a", handler, max_retries=1)
    await redis.xadd(STREAM, {"event": _event(attempt=1).model_dump_json()})
    messages = await _read(consumer, redis)
    message_id = messages[0][1][0][0]
    await consumer.dispatch_messages(messages)
    kinds = [op[0] + ":" + op[1] for op in redis.ops]
    dead_at = kinds.index(f"xadd:{settings.REDIS_DEAD_LETTER_STREAM}")
    ack_at = kinds.index(f"xack:{STREAM}")
    assert dead_at < ack_at
    assert message_id not in redis.pending[(STREAM, GROUP)]
    assert not any(op[0] == "xdel" and op[1] == STREAM and kinds.index(f"xadd:{settings.REDIS_DEAD_LETTER_STREAM}") > ack_at for op in redis.ops)


async def test_graceful_shutdown_stops_new_work_and_keeps_unfinished_work_recoverable(monkeypatch):
    monkeypatch.setattr(settings, "WORKER_GRACEFUL_SHUTDOWN_SECONDS", 0.3)
    redis = MemoryRedis()
    started = asyncio.Event()
    calls: list[str] = []

    async def handler(event: EventEnvelope) -> None:
        owner = current_owner()
        calls.append(owner)
        if owner == "worker-a":
            started.set()
            await asyncio.Event().wait()

    worker = _consumer(redis, "worker-a", handler)
    rival = _consumer(redis, "worker-b", handler)
    await redis.xadd(STREAM, {"event": _event().model_dump_json()})
    await redis.xgroup_create(STREAM, GROUP)
    background = asyncio.create_task(worker.run_forever())
    await asyncio.wait_for(started.wait(), timeout=2)
    await redis.xadd(STREAM, {"event": _event(job_id="job-2", event_id="other").model_dump_json()})
    await worker.stop()
    await asyncio.wait_for(background, timeout=2)
    assert calls == ["worker-a"]
    assert not any(op[0] == "xack" for op in redis.ops)
    message_id = next(iter(redis.pending[(STREAM, GROUP)]))
    redis.drop_leases()
    redis.force_idle(STREAM, GROUP, message_id, reclaim_min_idle_ms() + 1000)
    await rival.claim_stale()
    assert calls == ["worker-a", "worker-b"]


async def test_lost_lease_does_not_mark_a_processing_conversation_failed():
    calls: list[str] = []

    class Repo:
        async def get_conversation(self, conversation_id):
            calls.append("get")
            return SimpleNamespace(status=ConversationStatus.PROCESSING)

        async def transition(self, *args, **kwargs):
            calls.append("transition")
            raise AssertionError("stale worker must not transition the conversation")

    lease = Lease(
        redis=None,
        logical_key="conversation.processing.requested|conversation:conv-1",
        owner="worker-a",
        token="1",
        message_id="1-0",
        event_id="event-1",
        event_type="conversation.processing.requested",
        retry_count=0,
        lost=True,
        reason="expired",
    )
    token = push_lease(lease)
    try:
        with pytest.raises(StaleOwnershipError):
            await ConversationProcessingWorkflow(Repo(), router=SimpleNamespace()).run("conv-1")
    finally:
        reset_lease(token)
    assert calls == ["get"]


async def test_fence_token_increases_when_a_new_owner_acquires():
    redis = MemoryRedis()
    event = _event()
    first = await acquire_lease(redis, event, owner="worker-a", message_id="1-0")
    assert first.action == "run"
    await release_lease(redis, first.lease)
    second = await acquire_lease(redis, event, owner="worker-b", message_id="1-0")
    assert second.action == "run"
    assert int(second.token) > int(first.token)
    busy = await acquire_lease(redis, _event(event_id="dup"), owner="worker-c", message_id="2-0")
    assert busy.action == "busy"
    assert busy.owner == "worker-b"


def current_owner() -> str:
    from services.queue.job_lease import current_lease

    lease = current_lease()
    assert lease is not None
    return lease.owner


async def _deliver(consumer: RedisStreamConsumer, redis: MemoryRedis, event: EventEnvelope) -> None:
    message_id = await redis.xadd(STREAM, {"event": event.model_dump_json()})
    key = (STREAM, GROUP)
    redis.pending.setdefault(key, {})[message_id] = {
        "consumer": consumer.consumer_name,
        "fields": {"event": event.model_dump_json()},
        "deliveries": 1,
        "delivered_at": time.monotonic(),
    }
    await consumer.dispatch_messages([(STREAM, [(message_id, {"event": event.model_dump_json()})])])
