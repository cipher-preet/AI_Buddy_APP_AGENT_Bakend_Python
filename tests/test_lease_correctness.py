"""Phase 2A.1 correctness: fencing races, job identity, and ack survival.

These tests pause after an ownership check, expire the old short lock, and
crash around XACK. They do not call live models.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from redis.exceptions import ConnectionError

from apps.api_gateway.config.setting import settings
from services.conversation.models import WindowExtractionResult
from services.conversation.repository import ConversationRepository
from services.queue.job_lease import (
    Lease,
    StaleOwnershipError,
    acquire_lease,
    logical_job_key,
    preserve_pending_delivery,
    push_lease,
    release_lease,
    remember_completion,
    reset_lease,
    spawn_heartbeat,
)
from services.queue.mongo_fence import FENCE_FIELD, fenced_update_one, push_fenced_write_hook, reset_fenced_write_hook
from services.queue.streams import EventEnvelope
from tests.test_job_lease_reliability import GROUP, STREAM, MemoryRedis, _consumer, _event, _read

WINDOW_ID = "window-race-1"


def _envelope(event_type: str, **payload) -> EventEnvelope:
    conversation_id = str(payload.get("conversationId") or "conv-1")
    return EventEnvelope(
        eventType=event_type,
        correlationId=conversation_id,
        userId=str(payload.pop("userId", "user-1")),
        spaceId="space-1",
        conversationId=conversation_id,
        payload=payload,
    )


class _Docs:
    """Small Mongo stand-in that honors the fence operators."""

    def __init__(self) -> None:
        self.docs: dict[str, dict] = {}

    def _matches(self, doc: dict, query: dict) -> bool:
        for key, expected in query.items():
            if key == "$and":
                if not all(self._matches(doc, item) for item in expected):
                    return False
                continue
            if key == "$or":
                if not any(self._matches(doc, item) for item in expected):
                    return False
                continue
            if isinstance(expected, dict) and "$exists" in expected:
                if (key in doc) != bool(expected["$exists"]):
                    return False
                continue
            if isinstance(expected, dict) and "$lte" in expected:
                value = doc.get(key)
                if value is None or value > expected["$lte"]:
                    return False
                continue
            if doc.get(key) != expected:
                return False
        return True

    async def find_one(self, query):
        for doc in self.docs.values():
            if self._matches(doc, query):
                return dict(doc)
        return None

    async def update_one(self, query, update, upsert=False):
        for key, doc in self.docs.items():
            if self._matches(doc, query):
                doc.update(update.get("$set") or {})
                self.docs[key] = doc
                return SimpleNamespace(matched_count=1, modified_count=1)
        if upsert:
            inserted = {"_id": f"new-{len(self.docs)}"}
            inserted.update(update.get("$set") or {})
            self.docs[inserted["_id"]] = inserted
            return SimpleNamespace(matched_count=0, modified_count=0, upserted_id=inserted["_id"])
        return SimpleNamespace(matched_count=0, modified_count=0)


def _repo(docs: _Docs) -> ConversationRepository:
    database = SimpleNamespace(conversation_windows=docs)
    return ConversationRepository(database)  # type: ignore[arg-type]


async def _owned_lease(redis: MemoryRedis, token_owner: str = "worker-a") -> Lease:
    decision = await acquire_lease(
        redis,
        _event(event_type="conversation.window.extraction.requested"),
        owner=token_owner,
        message_id="1-0",
    )
    assert decision.action == "run"
    assert decision.lease is not None
    return decision.lease


async def test_stale_worker_cannot_overwrite_a_newer_window_after_the_lease_check():
    redis = MemoryRedis()
    docs = _Docs()
    docs.docs[WINDOW_ID] = {"_id": WINDOW_ID, "status": "PROCESSING", "provider": "owner-b", FENCE_FIELD: 1}
    repo = _repo(docs)
    lease = await _owned_lease(redis)
    scope = push_lease(lease)

    async def newer_owner_commits() -> None:
        docs.docs[WINDOW_ID][FENCE_FIELD] = 2
        docs.docs[WINDOW_ID]["provider"] = "owner-b"
        docs.docs[WINDOW_ID]["status"] = "PROCESSING"

    hook = push_fenced_write_hook(newer_owner_commits)
    try:
        with pytest.raises(StaleOwnershipError) as caught:
            await repo.complete_window(WINDOW_ID, WindowExtractionResult(summary="stale"), "owner-a", "model-a")
        assert caught.value.reason == "stale_fence"
    finally:
        reset_fenced_write_hook(hook)
        reset_lease(scope)

    saved = docs.docs[WINDOW_ID]
    assert saved["provider"] == "owner-b"
    assert saved["status"] == "PROCESSING"
    assert saved[FENCE_FIELD] == 2
    assert "summary" not in str(saved.get("result"))


async def test_unleased_window_write_still_commits():
    docs = _Docs()
    docs.docs[WINDOW_ID] = {"_id": WINDOW_ID, "status": "PROCESSING", "provider": "old"}
    repo = _repo(docs)
    await repo.complete_window(WINDOW_ID, WindowExtractionResult(summary="done"), "worker", "model")
    assert docs.docs[WINDOW_ID]["provider"] == "worker"
    assert docs.docs[WINDOW_ID]["status"] == "COMPLETED"
    assert FENCE_FIELD not in docs.docs[WINDOW_ID]


async def test_newer_fence_can_replace_an_older_artifact_and_upsert_does_not_duplicate():
    redis = MemoryRedis()
    docs = _Docs()
    docs.docs["briefing"] = {"_id": "briefing", "userId": "user-1", "dateKey": "2026-11-04", "body": "old", FENCE_FIELD: 1}
    def held(token: str, owner: str) -> Lease:
        return Lease(
            redis=redis,
            logical_key="daily.briefing.requested|user:user-1|date:2026-11-04",
            owner=owner,
            token=token,
            message_id="1-0",
            event_id="e",
            event_type="daily.briefing.requested",
            retry_count=0,
        )

    scope = push_lease(held("2", "worker-b"))
    try:
        await fenced_update_one(
            docs,
            {"userId": "user-1", "dateKey": "2026-11-04"},
            {"$set": {"body": "new"}},
        )
        with pytest.raises(StaleOwnershipError):
            nested = push_lease(held("1", "worker-a"))
            try:
                await fenced_update_one(
                    docs,
                    {"userId": "user-1", "dateKey": "2026-11-04"},
                    {"$set": {"body": "stale"}},
                    upsert=True,
                )
            finally:
                reset_lease(nested)
    finally:
        reset_lease(scope)
    assert docs.docs["briefing"]["body"] == "new"
    assert len(docs.docs) == 1


def test_stages_retries_force_and_reprocess_have_distinct_identities():
    created = datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc)
    processing = _envelope("conversation.processing.requested", conversationId="conv-9", processingVersion=1)
    processing_retry = processing.model_copy(update={"eventId": "retry-event", "causationId": processing.eventId, "attempt": 2})
    processing_next = _envelope("conversation.processing.requested", conversationId="conv-9", processingVersion=2)
    processing_redo = _envelope(
        "conversation.processing.requested",
        conversationId="conv-9",
        processingVersion=1,
        reprocess=True,
    )
    processing_redo.createdAt = created
    finalization = _envelope("conversation.finalization.requested", conversationId="conv-9")
    transcript = _envelope("conversation.transcript.ready", conversationId="conv-9")
    schedule_a = _envelope(
        "conversation.schedule_extraction.requested",
        conversationId="conv-9",
        sourceEventId="processing-event-1",
    )
    schedule_retry = schedule_a.model_copy(update={"eventId": "schedule-retry", "attempt": 1})
    schedule_b = _envelope(
        "conversation.schedule_extraction.requested",
        conversationId="conv-9",
        sourceEventId="processing-event-2",
    )
    merge = _envelope("meeting.video.merge.requested", conversationId="conv-9", meetingSessionId="conv-9")
    window_a = _envelope("conversation.window.extraction.requested", conversationId="conv-9", windowId="w-1")
    window_b = _envelope("conversation.window.extraction.requested", conversationId="conv-9", windowId="w-2")
    scheduled = _envelope("daily.briefing.requested", userId="user-9", dateKey="2026-11-04", conversationId="daily")
    forced = _envelope(
        "daily.briefing.requested",
        userId="user-9",
        dateKey="2026-11-04",
        conversationId="daily",
        forced=True,
    )
    forced.createdAt = created
    forced_retry = forced.model_copy(update={"eventId": "force-retry", "causationId": forced.eventId, "attempt": 1})
    forced_again = _envelope(
        "daily.briefing.requested",
        userId="user-9",
        dateKey="2026-11-04",
        conversationId="daily",
        forced=True,
    )
    forced_again.createdAt = created + timedelta(seconds=30)

    keys = {
        "processing": logical_job_key(processing),
        "finalization": logical_job_key(finalization),
        "transcript": logical_job_key(transcript),
        "schedule": logical_job_key(schedule_a),
        "merge": logical_job_key(merge),
        "window": logical_job_key(window_a),
        "scheduled": logical_job_key(scheduled),
        "forced": logical_job_key(forced),
    }
    assert len(keys.values()) == len(set(keys.values()))
    assert logical_job_key(processing) == logical_job_key(processing_retry)
    assert logical_job_key(processing) != logical_job_key(processing_next)
    assert logical_job_key(processing) != logical_job_key(processing_redo)
    assert logical_job_key(schedule_a) == logical_job_key(schedule_retry)
    assert logical_job_key(schedule_a) != logical_job_key(schedule_b)
    assert logical_job_key(window_a) != logical_job_key(window_b)
    assert logical_job_key(forced) == logical_job_key(forced_retry)
    assert logical_job_key(forced) != logical_job_key(forced_again)
    assert logical_job_key(scheduled) != logical_job_key(forced)


async def test_completed_scheduled_briefing_does_not_block_force_regeneration():
    redis = MemoryRedis()
    scheduled = _envelope("daily.briefing.requested", userId="user-9", dateKey="2026-11-04", conversationId="daily")
    forced = _envelope(
        "daily.briefing.requested",
        userId="user-9",
        dateKey="2026-11-04",
        conversationId="daily",
        forced=True,
    )
    forced.createdAt = datetime(2026, 11, 4, 15, 0, tzinfo=timezone.utc)
    first = await acquire_lease(redis, scheduled, owner="worker-a", message_id="1-0")
    assert first.action == "run"
    assert await remember_completion(redis, first.lease)
    blocked = await acquire_lease(redis, scheduled.model_copy(update={"eventId": "replay"}), owner="worker-b", message_id="2-0")
    assert blocked.action == "completed"
    again = await acquire_lease(redis, forced, owner="worker-b", message_id="3-0")
    assert again.action == "run"


async def test_acquire_stays_single_owner_when_the_script_is_delayed_past_the_old_lock_ttl():
    redis = MemoryRedis()
    original = redis.eval

    async def delayed(script, numkeys, *args):
        if "job-lease-acquire" in str(script):
            await asyncio.sleep(0.05)
        return await original(script, numkeys, *args)

    redis.eval = delayed  # type: ignore[method-assign]
    event = _event()
    first, second = await asyncio.gather(
        acquire_lease(redis, event, owner="worker-a", message_id="1-0"),
        acquire_lease(redis, event, owner="worker-b", message_id="2-0"),
    )
    actions = sorted([first.action, second.action])
    assert actions == ["busy", "run"]
    assert len({item.owner for item in (first, second) if item.action == "run"}) == 1


async def test_removed_check_then_set_lock_can_split_ownership():
    """Documents the race the 2s SET NX lock had. The live path no longer does this."""
    redis = MemoryRedis()
    owners: list[str] = []

    async def old_acquire(name: str) -> None:
        while await redis.set("lock", name, nx=True, px=20) is None:
            await asyncio.sleep(0.005)
        seen = await redis.get("lease")
        await asyncio.sleep(0.05)
        if seen is None:
            await redis.set("lease", name)
            owners.append(name)

    await asyncio.gather(old_acquire("worker-a"), old_acquire("worker-b"))
    assert sorted(owners) == ["worker-a", "worker-b"]


async def test_preserve_pending_does_not_steal_or_reset_delivery_count():
    redis = MemoryRedis()
    await redis.xgroup_create(STREAM, GROUP)
    redis.pending[(STREAM, GROUP)]["9-0"] = {
        "consumer": "winner",
        "fields": {},
        "deliveries": 5,
        "delivered_at": redis._now(),
    }
    await preserve_pending_delivery(
        redis,
        stream=STREAM,
        group=GROUP,
        consumer_name="loser",
        message_id="9-0",
    )
    entry = redis.pending[(STREAM, GROUP)]["9-0"]
    assert entry["consumer"] == "winner"
    assert entry["deliveries"] == 5
    assert redis.xclaim_retrycounts == [5]


async def test_heartbeat_does_not_claim_after_the_token_changes(monkeypatch):
    monkeypatch.setattr(settings, "JOB_LEASE_HEARTBEAT_SECONDS", 0.05)
    redis = MemoryRedis()
    lease = await _owned_lease(redis, "winner")
    await redis.xgroup_create(STREAM, GROUP)
    redis.pending[(STREAM, GROUP)][lease.message_id] = {
        "consumer": "winner",
        "fields": {},
        "deliveries": 4,
        "delivered_at": redis._now(),
    }
    await redis.set(f"buddy:job-lease:{lease.logical_key}", "v1|other|99|2-0|e|stt.requested|0|1", px=60_000)
    task = spawn_heartbeat(
        redis,
        lease,
        stream=STREAM,
        group=GROUP,
        consumer_name="winner",
        message_id=lease.message_id,
    )
    await asyncio.wait_for(task, timeout=1)
    assert lease.lost is True
    assert redis.xclaim_retrycounts == []
    assert redis.pending[(STREAM, GROUP)][lease.message_id]["consumer"] == "winner"
    assert redis.pending[(STREAM, GROUP)][lease.message_id]["deliveries"] == 4


async def test_completion_before_ack_redelivers_without_running_again(monkeypatch):
    monkeypatch.setattr(settings, "JOB_LEASE_TTL_SECONDS", 90)
    redis = MemoryRedis()
    calls: list[str] = []

    async def handler(event: EventEnvelope) -> None:
        calls.append(event.eventId)

    event = _event(event_id="once")
    await redis.xadd(STREAM, {"event": event.model_dump_json()})
    owner = _consumer(redis, "worker-a", handler)
    messages = await _read(owner, redis)
    message_id = messages[0][1][0][0]
    decision = await acquire_lease(redis, event, owner="worker-a", message_id=message_id)
    assert await remember_completion(redis, decision.lease)
    redis.drop_leases()
    redis.force_idle(STREAM, GROUP, message_id, 120_000)
    rival = _consumer(redis, "worker-b", handler)
    await rival.claim_stale()
    assert calls == []
    assert message_id not in redis.pending[(STREAM, GROUP)]


async def test_completion_marker_expiry_allows_a_later_delivery(monkeypatch):
    monkeypatch.setattr(settings, "JOB_COMPLETION_TTL_SECONDS", 1)
    redis = MemoryRedis()
    event = _event()
    first = await acquire_lease(redis, event, owner="worker-a", message_id="1-0")
    assert await remember_completion(redis, first.lease)
    blocked = await acquire_lease(redis, event, owner="worker-b", message_id="2-0")
    assert blocked.action == "completed"
    await asyncio.sleep(1.15)
    redis.drop_leases()
    restored = await acquire_lease(redis, event, owner="worker-c", message_id="3-0")
    assert restored.action == "run"


async def test_retry_write_survives_an_ack_failure_and_duplicate_success_is_idle():
    redis = MemoryRedis()

    async def fail_ack(stream, group, message_id):
        redis.ops.append(("xack-fail", stream))
        raise ConnectionError("ack failed")

    redis.xack = fail_ack  # type: ignore[method-assign]
    calls: list[int] = []

    async def handler(event: EventEnvelope) -> None:
        calls.append(1)
        raise RuntimeError("try again")

    consumer = _consumer(redis, "worker-a", handler, max_retries=5)
    consumer._uses_default_redis = True
    event = _event()
    await redis.xadd(STREAM, {"event": event.model_dump_json()})
    messages = await _read(consumer, redis)
    message_id, fields = messages[0][1][0]
    await consumer._semaphore.acquire()
    with pytest.raises(ConnectionError):
        await consumer._handle_message(message_id, fields)
    assert calls == [1]
    assert message_id in redis.pending[(STREAM, GROUP)]
    assert any(op == ("xadd", settings.REDIS_RETRY_STREAM) for op in redis.ops)
    assert not any(op[0] == "xack" for op in redis.ops)


async def test_dead_letter_write_failure_leaves_the_pending_entry():
    redis = MemoryRedis()
    original = redis.xadd

    async def fail_dlq(stream, fields, maxlen=None, approximate=True):
        if stream == settings.REDIS_DEAD_LETTER_STREAM:
            raise ConnectionError("dlq down")
        return await original(stream, fields, maxlen=maxlen, approximate=approximate)

    redis.xadd = fail_dlq  # type: ignore[method-assign]

    async def handler(event: EventEnvelope) -> None:
        raise RuntimeError("permanent")

    consumer = _consumer(redis, "worker-a", handler, max_retries=0)
    consumer.max_retries = 0
    await redis.xadd(STREAM, {"event": _event().model_dump_json()})
    messages = await _read(consumer, redis)
    message_id, fields = messages[0][1][0]
    await consumer._semaphore.acquire()
    with pytest.raises(ConnectionError):
        await consumer._handle_message(message_id, fields)
    assert message_id in redis.pending[(STREAM, GROUP)]
    assert not any(op[0] == "xack" for op in redis.ops)


async def test_ack_failure_after_durable_completion_keeps_the_marker():
    redis = MemoryRedis()

    async def fail_ack(stream, group, message_id):
        raise ConnectionError("ack failed")

    redis.xack = fail_ack  # type: ignore[method-assign]

    async def handler(event: EventEnvelope) -> None:
        from services.queue.job_lease import mark_current_job_completed

        mark_current_job_completed()

    consumer = _consumer(redis, "worker-a", handler)
    event = _event(event_id="done-once")
    await redis.xadd(STREAM, {"event": event.model_dump_json()})
    messages = await _read(consumer, redis)
    message_id, fields = messages[0][1][0]
    await consumer._semaphore.acquire()
    with pytest.raises(ConnectionError):
        await consumer._handle_message(message_id, fields)
    done = [key for key in redis.kv if key.startswith("buddy:job-done:")]
    assert done
    assert message_id in redis.pending[(STREAM, GROUP)]
    redis.xack = MemoryRedis.xack.__get__(redis, MemoryRedis)  # type: ignore[method-assign]
    calls: list[str] = []

    async def second(event: EventEnvelope) -> None:
        calls.append(event.eventId)

    rival = _consumer(redis, "worker-b", second)
    redis.drop_leases()
    redis.force_idle(STREAM, GROUP, message_id, 120_000)
    await rival.claim_stale()
    assert calls == []
    assert message_id not in redis.pending[(STREAM, GROUP)]


async def test_fence_counter_has_no_ttl_and_later_work_can_raise_it():
    """A per-job counter or a 7-day TTL restarts at 1 and loses to a stored fence."""
    redis = MemoryRedis()
    docs = _Docs()
    docs.docs["conv"] = {"_id": "conv", "status": "PROCESSING"}
    processing = _envelope(
        "conversation.processing.requested",
        conversationId="conv-9",
        processingVersion=1,
    )
    finalization = _envelope("conversation.finalization.requested", conversationId="conv-9")
    reprocess = _envelope(
        "conversation.processing.requested",
        conversationId="conv-9",
        processingVersion=1,
        reprocess=True,
    )
    scheduled = _envelope(
        "daily.briefing.requested",
        userId="user-9",
        dateKey="2026-11-04",
        conversationId="daily",
    )
    forced = _envelope(
        "daily.briefing.requested",
        userId="user-9",
        dateKey="2026-11-04",
        conversationId="daily",
        forced=True,
    )
    forced.createdAt = datetime(2026, 11, 4, 18, 0, tzinfo=timezone.utc)

    first = await acquire_lease(redis, processing, owner="worker-a", message_id="1-0")
    assert first.action == "run"
    scope = push_lease(first.lease)
    try:
        await fenced_update_one(docs, {"_id": "conv"}, {"$set": {"stage": "processing"}})
    finally:
        reset_lease(scope)
    await release_lease(redis, first.lease)
    assert redis.kv["buddy:job-fence:global"][1] is None

    recovered = await acquire_lease(redis, processing, owner="worker-b", message_id="2-0")
    assert int(recovered.token) > int(first.token)
    scope = push_lease(recovered.lease)
    try:
        await fenced_update_one(docs, {"_id": "conv"}, {"$set": {"stage": "recovered"}})
    finally:
        reset_lease(scope)
        await release_lease(redis, recovered.lease)

    stage = await acquire_lease(redis, finalization, owner="worker-c", message_id="3-0")
    assert logical_job_key(finalization) != logical_job_key(processing)
    assert int(stage.token) > int(recovered.token)
    scope = push_lease(stage.lease)
    try:
        await fenced_update_one(docs, {"_id": "conv"}, {"$set": {"stage": "finalization"}})
    finally:
        reset_lease(scope)
        await release_lease(redis, stage.lease)

    again = await acquire_lease(redis, reprocess, owner="worker-d", message_id="4-0")
    assert logical_job_key(reprocess) != logical_job_key(processing)
    assert int(again.token) > int(stage.token)
    scope = push_lease(again.lease)
    try:
        await fenced_update_one(docs, {"_id": "conv"}, {"$set": {"stage": "reprocess"}})
    finally:
        reset_lease(scope)
    assert docs.docs["conv"]["stage"] == "reprocess"
    assert docs.docs["conv"][FENCE_FIELD] == int(again.token)

    stale = Lease(
        redis=redis,
        logical_key=first.lease.logical_key,
        owner="worker-a",
        token=first.token,
        message_id="1-0",
        event_id="e",
        event_type="conversation.processing.requested",
        retry_count=0,
    )
    nested = push_lease(stale)
    try:
        with pytest.raises(StaleOwnershipError) as caught:
            await fenced_update_one(docs, {"_id": "conv"}, {"$set": {"stage": "stale"}})
        assert caught.value.reason == "stale_fence"
    finally:
        reset_lease(nested)
    assert docs.docs["conv"]["stage"] == "reprocess"

    scheduled_lease = await acquire_lease(redis, scheduled, owner="worker-a", message_id="5-0")
    assert await remember_completion(redis, scheduled_lease.lease)
    forced_lease = await acquire_lease(redis, forced, owner="worker-b", message_id="6-0")
    assert forced_lease.action == "run"
    assert logical_job_key(scheduled) != logical_job_key(forced)
    replay = await acquire_lease(redis, scheduled.model_copy(update={"eventId": "replay"}), owner="worker-c", message_id="7-0")
    assert replay.action == "completed"


async def test_a_deleted_fence_counter_is_lower_than_a_stored_document_fence():
    """Redis flush or eviction can still restart the counter. That must not overwrite."""
    redis = MemoryRedis()
    docs = _Docs()
    docs.docs["conv"] = {"_id": "conv", "status": "PROCESSING", FENCE_FIELD: 4}
    event = _envelope("conversation.processing.requested", conversationId="conv-9", processingVersion=2)
    redis.kv.pop("buddy:job-fence:global", None)
    reset = await acquire_lease(redis, event, owner="worker-a", message_id="1-0")
    assert reset.token == "1"
    scope = push_lease(reset.lease)
    try:
        with pytest.raises(StaleOwnershipError) as caught:
            await fenced_update_one(docs, {"_id": "conv"}, {"$set": {"stage": "reset"}})
        assert caught.value.reason == "stale_fence"
    finally:
        reset_lease(scope)
    assert docs.docs["conv"].get("stage") != "reset"
    assert docs.docs["conv"][FENCE_FIELD] == 4
