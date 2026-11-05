"""Goal crash recovery against disposable local MongoDB and Redis 7."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient
from redis.asyncio import Redis

from apps.api_gateway.config.setting import settings
from apps.api_gateway.workers.goal_worker import GoalStreamsRunner
from services.goals.jobs import GoalJobHandler
from services.goals.store import GoalStore
from services.queue.atomic_publish import atomic_publish, dedupe_key, submission_fingerprint
from services.queue.job_lease import Lease, StaleOwnershipError, acquire_lease, logical_job_key, push_lease, reset_lease
from services.queue.streams import EventEnvelope

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[1]
MONGO_HOST = "127.0.0.1"
MONGO_PORT = 27017
REDIS_HOST = "127.0.0.1"
REDIS_PORT = 6399


class _Renew:
    async def eval(self, script, numkeys, *args):
        return 1


async def _mongo():
    client = AsyncIOMotorClient(
        f"mongodb://{MONGO_HOST}:{MONGO_PORT}",
        serverSelectionTimeoutMS=2000,
        uuidRepresentation="standard",
    )
    try:
        info = await client.admin.command("hello")
    except Exception as error:
        client.close()
        pytest.fail(f"local MongoDB is not available on {MONGO_HOST}:{MONGO_PORT}: {type(error).__name__}")
    address = client.address
    host = "" if address is None else str(address[0])
    port = None if address is None else int(address[1])
    me = str(info.get("me") or "")
    local_names = {"127.0.0.1", "localhost", "::1"}
    if host not in local_names or port != MONGO_PORT:
        client.close()
        pytest.fail(f"refusing non-local MongoDB at {host}:{port}")
    if me and not any(name in me for name in local_names):
        client.close()
        pytest.fail(f"refusing non-local MongoDB at {me}")
    return client, str(info.get("maxWireVersion")), info


async def _redis() -> Redis:
    client = Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True, socket_connect_timeout=2, socket_timeout=2)
    try:
        info = await client.info("server")
    except Exception as error:
        await client.aclose()
        pytest.fail(f"Redis 7 is not available on {REDIS_HOST}:{REDIS_PORT}: {type(error).__name__}")
    version = str(info.get("redis_version") or "")
    if not version.startswith("7."):
        await client.aclose()
        pytest.fail(f"refusing Redis {version}")
    return client


def _lease(token: int, owner: str = "worker-a") -> Lease:
    return Lease(
        redis=_Renew(),
        logical_key=f"goal.clarify|job:{token}",
        owner=owner,
        token=str(token),
        message_id="1-0",
        event_id="event-1",
        event_type="goal.clarify",
        retry_count=0,
    )


def _event(goal_id: str, job_id: str, job_type: str = "clarify") -> EventEnvelope:
    return EventEnvelope(
        eventId=job_id,
        eventType=f"goal.{job_type}",
        correlationId=goal_id,
        userId="user-1",
        spaceId="",
        conversationId=goal_id,
        payload={"goalId": goal_id, "jobId": job_id, "jobType": job_type},
    )


@pytest.fixture
async def world():
    client, _wire, info = await _mongo()
    redis = await _redis()
    name = f"phase2c-{uuid.uuid4().hex[:10]}"
    database = client[name]
    version = str((await client.server_info()).get("version") or "")
    yield database, redis, version, str(info.get("me") or "")
    await client.drop_database(name)
    client.close()
    await redis.aclose()


async def _seed(database, *, status: str = "queued", fence: int | None = None, durable: dict | None = None):
    goal_id = ObjectId()
    job_id = ObjectId()
    await database.goals.insert_one({"_id": goal_id, "ownerId": "user-1", "status": "DRAFT", "latestStep": None})
    job = {
        "_id": job_id,
        "goalId": goal_id,
        "ownerId": "user-1",
        "jobType": "clarify",
        "status": status,
        "error": "",
        "model": "",
        "payload": {},
    }
    if fence is not None:
        job["processingFence"] = fence
    if durable is not None:
        job["durableStep"] = durable
    await database.goal_jobs.insert_one(job)
    return str(goal_id), str(job_id)


async def test_queued_goal_is_claimed_processed_and_not_run_twice(world, monkeypatch):
    database, _redis, _version, _me = world
    goal_id, job_id = await _seed(database)
    calls = []

    async def build(goal, messages):
        calls.append(job_id)
        return {
            "step": {"stepId": "ask-1", "prompt": "When?", "control": "text"},
            "draft": {},
            "status": "CLARIFY",
            "model": "test",
        }

    monkeypatch.setattr("services.goals.jobs.build_clarify_step", build)
    token = push_lease(_lease(1))
    try:
        await GoalJobHandler(database).handle(_event(goal_id, job_id))
        await GoalJobHandler(database).handle(_event(goal_id, job_id))
    finally:
        reset_lease(token)
    assert calls == [job_id]
    stored = await database.goal_jobs.find_one({"_id": ObjectId(job_id)})
    assert stored["status"] == "ready"
    assert stored["processingFence"] == 1
    messages = await database.goal_messages.count_documents({"jobId": ObjectId(job_id)})
    assert messages == 1


async def test_abandoned_running_goal_is_recovered_instead_of_done(world):
    database, _redis, _version, _me = world
    goal_id, job_id = await _seed(database, status="running", fence=1)
    token = push_lease(_lease(2, owner="worker-b"))
    try:
        outcome, _job = await GoalStore(database).claim(job_id)
    finally:
        reset_lease(token)
    assert outcome == "recovered"
    stored = await database.goal_jobs.find_one({"_id": ObjectId(job_id)})
    assert stored["status"] == "running"
    assert stored["processingFence"] == 2
    assert stored["processingOwner"] == "worker-b"


async def test_same_fence_resumes_and_newer_fence_stays_busy(world):
    database, _redis, _version, _me = world
    _goal_id, job_id = await _seed(database, status="running", fence=4)
    same = push_lease(_lease(4, owner="worker-a"))
    try:
        outcome, _job = await GoalStore(database).claim(job_id)
    finally:
        reset_lease(same)
    assert outcome == "owned"
    newer = push_lease(_lease(3, owner="worker-c"))
    try:
        blocked, _job = await GoalStore(database).claim(job_id)
    finally:
        reset_lease(newer)
    assert blocked == "busy"


async def test_two_workers_cannot_both_recover_one_running_goal(world, monkeypatch):
    database, _redis, _version, _me = world
    _goal_id, job_id = await _seed(database, status="running", fence=1)
    from motor.motor_asyncio import AsyncIOMotorCollection

    original = AsyncIOMotorCollection.find_one
    state = {"n": 0}
    release = asyncio.Event()

    async def gated(self, *args, **kwargs):
        result = await original(self, *args, **kwargs)
        if self.name != "goal_jobs":
            return result
        state["n"] += 1
        if state["n"] == 1:
            await release.wait()
        elif state["n"] == 2:
            release.set()
        return result

    monkeypatch.setattr(AsyncIOMotorCollection, "find_one", gated)

    async def recover(token_value: int, owner: str):
        token = push_lease(_lease(token_value, owner=owner))
        try:
            return await GoalStore(database).claim(job_id)
        finally:
            reset_lease(token)

    first, second = await asyncio.wait_for(
        asyncio.gather(recover(5, "worker-a"), recover(6, "worker-b")),
        timeout=5,
    )
    outcomes = sorted([first[0], second[0]])
    assert outcomes == ["busy", "recovered"]
    stored = await database.goal_jobs.find_one({"_id": ObjectId(job_id)})
    assert stored["processingFence"] in {5, 6}
    assert stored["processingAttempt"] == 1


async def test_crash_after_running_recovers_and_runs_the_step(world, monkeypatch):
    database, _redis, _version, _me = world
    goal_id, job_id = await _seed(database, status="running", fence=1)
    calls = []

    async def build(goal, messages):
        calls.append("ran")
        return {
            "step": {"stepId": "ask-1", "prompt": "When?", "control": "text"},
            "draft": {},
            "status": "CLARIFY",
            "model": "test",
        }

    monkeypatch.setattr("services.goals.jobs.build_clarify_step", build)
    token = push_lease(_lease(2, owner="worker-b"))
    try:
        await GoalJobHandler(database).handle(_event(goal_id, job_id))
    finally:
        reset_lease(token)
    assert calls == ["ran"]
    stored = await database.goal_jobs.find_one({"_id": ObjectId(job_id)})
    assert stored["status"] == "ready"


async def test_persisted_step_is_resumed_without_a_second_model_call(world, monkeypatch):
    database, _redis, _version, _me = world
    durable = {
        "step": {"stepId": "ask-1", "prompt": "When?", "control": "text"},
        "draft": {},
        "status": "CLARIFY",
        "assistantText": "When?",
        "model": "test",
        "plan": None,
    }
    goal_id, job_id = await _seed(database, status="running", fence=1, durable=durable)
    calls = []

    async def build(goal, messages):
        calls.append("ran")
        return {"step": {"stepId": "other"}, "draft": {}, "status": "CLARIFY", "model": "test"}

    monkeypatch.setattr("services.goals.jobs.build_clarify_step", build)
    token = push_lease(_lease(2, owner="worker-b"))
    try:
        await GoalJobHandler(database).handle(_event(goal_id, job_id))
    finally:
        reset_lease(token)
    assert calls == []
    stored = await database.goal_jobs.find_one({"_id": ObjectId(job_id)})
    assert stored["status"] == "ready"
    goal = await database.goals.find_one({"_id": ObjectId(goal_id)})
    assert goal["latestStep"]["stepId"] == "ask-1"
    assert await database.goal_messages.count_documents({"jobId": ObjectId(job_id), "stepId": "ask-1"}) == 1


async def test_stale_owner_cannot_overwrite_a_newer_fence(world):
    database, _redis, _version, _me = world
    goal_id, job_id = await _seed(database, status="running", fence=2)
    from services.queue.job_lease import StaleOwnershipError

    token = push_lease(_lease(1, owner="worker-a"))
    try:
        with pytest.raises(StaleOwnershipError):
            await GoalStore(database).save_step(
                goal_id,
                job_id,
                step={"stepId": "stale", "prompt": "no", "control": "text"},
                draft_updates={},
                status="CLARIFY",
                assistant_text="no",
                model="test",
            )
    finally:
        reset_lease(token)
    stored = await database.goal_jobs.find_one({"_id": ObjectId(job_id)})
    assert stored["status"] == "running"
    assert stored["processingFence"] == 2
    assert "durableStep" not in stored


async def test_ready_goal_is_not_executed_again(world, monkeypatch):
    database, _redis, _version, _me = world
    goal_id, job_id = await _seed(database, status="ready", fence=2)
    calls = []

    async def build(goal, messages):
        calls.append("ran")
        return {"step": {"stepId": "x"}, "draft": {}, "status": "CLARIFY", "model": "test"}

    monkeypatch.setattr("services.goals.jobs.build_clarify_step", build)
    token = push_lease(_lease(3))
    try:
        await GoalJobHandler(database).handle(_event(goal_id, job_id))
    finally:
        reset_lease(token)
    assert calls == []
    stored = await database.goal_jobs.find_one({"_id": ObjectId(job_id)})
    assert stored["status"] == "ready"


async def test_failed_goal_is_not_restarted(world, monkeypatch):
    database, _redis, _version, _me = world
    goal_id, job_id = await _seed(database, status="failed", fence=2)
    calls = []
    monkeypatch.setattr("services.goals.jobs.build_clarify_step", lambda *args, **kwargs: calls.append("ran"))
    token = push_lease(_lease(3))
    try:
        await GoalJobHandler(database).handle(_event(goal_id, job_id))
    finally:
        reset_lease(token)
    assert calls == []
    stored = await database.goal_jobs.find_one({"_id": ObjectId(job_id)})
    assert stored["status"] == "failed"


async def test_transient_failure_marks_failed_before_the_handler_returns(world, monkeypatch):
    database, _redis, _version, _me = world
    goal_id, job_id = await _seed(database)

    async def build(goal, messages):
        raise RuntimeError("model down")

    monkeypatch.setattr("services.goals.jobs.build_clarify_step", build)
    token = push_lease(_lease(1))
    try:
        await GoalJobHandler(database).handle(_event(goal_id, job_id))
    finally:
        reset_lease(token)
    stored = await database.goal_jobs.find_one({"_id": ObjectId(job_id)})
    assert stored["status"] == "failed"


async def test_missing_goal_is_failed_before_dead_letter_ack(world, monkeypatch):
    database, redis, _version, _me = world
    goal_id, job_id = await _seed(database)
    await database.goals.delete_one({"_id": ObjectId(goal_id)})
    stream = f"phase2c:goals:{uuid.uuid4().hex[:8]}"
    group = "goal-workers"
    monkeypatch.setattr("apps.api_gateway.workers.goal_worker.get_goal_redis_client", lambda: redis)
    monkeypatch.setattr(settings, "REDIS_DEAD_LETTER_STREAM", f"phase2c:dlq:{uuid.uuid4().hex[:8]}")

    async def handle(event):
        await GoalJobHandler(database).handle(event)

    monkeypatch.setattr("apps.api_gateway.workers.goal_worker.handle_goal_event", handle)
    runner = GoalStreamsRunner()
    runner.redis = redis
    event = _event(goal_id, job_id)
    await redis.xadd(stream, {"event": event.model_dump_json()})
    await redis.xgroup_create(stream, group, id="0", mkstream=True)
    messages = await redis.xreadgroup(group, runner.consumer_name, {stream: ">"}, count=1)
    message_id, fields = messages[0][1][0]
    await runner._handle_message(stream, group, message_id, fields)
    stored = await database.goal_jobs.find_one({"_id": ObjectId(job_id)})
    assert stored["status"] == "failed"
    assert _pending_count(await redis.xpending(stream, group)) == 0
    dlq = settings.REDIS_DEAD_LETTER_STREAM
    assert await redis.xlen(dlq) == 1
    await redis.delete(stream, dlq)


async def test_active_redis_owner_does_not_let_the_rival_run(world, monkeypatch):
    database, redis, _version, _me = world
    goal_id, job_id = await _seed(database)
    stream = f"phase2c:goals:{uuid.uuid4().hex[:8]}"
    group = "goal-workers"
    calls = []

    async def build(goal, messages):
        calls.append("ran")
        await asyncio.sleep(0.4)
        return {
            "step": {"stepId": "ask-1", "prompt": "When?", "control": "text"},
            "draft": {},
            "status": "CLARIFY",
            "model": "test",
        }

    monkeypatch.setattr("services.goals.jobs.build_clarify_step", build)
    monkeypatch.setattr("apps.api_gateway.workers.goal_worker.get_goal_redis_client", lambda: redis)

    async def handle(event):
        await GoalJobHandler(database).handle(event)

    monkeypatch.setattr("apps.api_gateway.workers.goal_worker.handle_goal_event", handle)
    event = _event(goal_id, job_id)
    await redis.xadd(stream, {"event": event.model_dump_json()})
    await redis.xgroup_create(stream, group, id="0", mkstream=True)
    owner = GoalStreamsRunner()
    rival = GoalStreamsRunner()
    owner.redis = redis
    rival.redis = redis
    rival.consumer_name = f"{owner.consumer_name}-b"
    messages = await redis.xreadgroup(group, owner.consumer_name, {stream: ">"}, count=1)
    message_id, fields = messages[0][1][0]
    task = asyncio.create_task(owner._handle_message(stream, group, message_id, fields))
    await asyncio.sleep(0.05)
    await rival._handle_message(stream, group, message_id, fields)
    await task
    assert calls == ["ran"]
    stored = await database.goal_jobs.find_one({"_id": ObjectId(job_id)})
    assert stored["status"] == "ready"
    assert _pending_count(await redis.xpending(stream, group)) == 0
    await _forget_job(redis, event)
    await redis.delete(stream)


async def test_completion_without_ack_does_not_rerun_the_step(world, monkeypatch):
    database, redis, _version, _me = world
    goal_id, job_id = await _seed(database)
    stream = f"phase2c:goals:{uuid.uuid4().hex[:8]}"
    group = "goal-workers"
    calls = []

    async def build(goal, messages):
        calls.append("ran")
        return {
            "step": {"stepId": "ask-1", "prompt": "When?", "control": "text"},
            "draft": {},
            "status": "CLARIFY",
            "model": "test",
        }

    monkeypatch.setattr("services.goals.jobs.build_clarify_step", build)
    monkeypatch.setattr("apps.api_gateway.workers.goal_worker.get_goal_redis_client", lambda: redis)

    async def handle(event):
        await GoalJobHandler(database).handle(event)

    monkeypatch.setattr("apps.api_gateway.workers.goal_worker.handle_goal_event", handle)
    runner = GoalStreamsRunner()
    runner.redis = redis
    event = _event(goal_id, job_id)
    await redis.xadd(stream, {"event": event.model_dump_json()})
    await redis.xgroup_create(stream, group, id="0", mkstream=True)
    messages = await redis.xreadgroup(group, runner.consumer_name, {stream: ">"}, count=1)
    message_id, fields = messages[0][1][0]
    original = runner._ack

    async def fail_ack(stream_name, group_name, message):
        raise ConnectionError("ack interrupted")

    runner._ack = fail_ack
    await runner._handle_message(stream, group, message_id, fields)
    runner._ack = original
    assert calls == ["ran"]
    assert (await database.goal_jobs.find_one({"_id": ObjectId(job_id)}))["status"] == "ready"
    await _forget_job(redis, event)
    await runner._handle_message(stream, group, message_id, fields)
    assert calls == ["ran"]
    assert _pending_count(await redis.xpending(stream, group)) == 0
    await _forget_job(redis, event)
    await redis.delete(stream)


async def test_busy_claim_does_not_acknowledge(world, monkeypatch):
    database, redis, _version, _me = world
    goal_id, job_id = await _seed(database, status="running", fence=10**15)
    stream = f"phase2c:goals:{uuid.uuid4().hex[:8]}"
    group = "goal-workers"
    monkeypatch.setattr("apps.api_gateway.workers.goal_worker.get_goal_redis_client", lambda: redis)
    monkeypatch.setattr(settings, "JOB_LEASE_TTL_SECONDS", 30)
    monkeypatch.setattr(settings, "JOB_LEASE_HEARTBEAT_SECONDS", 5)

    async def handle(event):
        await GoalJobHandler(database).handle(event)

    monkeypatch.setattr("apps.api_gateway.workers.goal_worker.handle_goal_event", handle)
    runner = GoalStreamsRunner()
    runner.redis = redis
    event = _event(goal_id, job_id)
    await redis.xadd(stream, {"event": event.model_dump_json()})
    await redis.xgroup_create(stream, group, id="0", mkstream=True)
    messages = await redis.xreadgroup(group, runner.consumer_name, {stream: ">"}, count=1)
    message_id, fields = messages[0][1][0]
    await runner._handle_message(stream, group, message_id, fields)
    assert _pending_count(await redis.xpending(stream, group)) == 1
    stored = await database.goal_jobs.find_one({"_id": ObjectId(job_id)})
    assert stored["status"] == "running"
    assert stored["processingFence"] == 10**15
    await _forget_job(redis, event)
    await redis.delete(stream)


async def test_running_without_a_fence_recovers_only_with_a_lease(world):
    database, _redis, _version, _me = world
    _goal_id, job_id = await _seed(database, status="running")
    blocked, _job = await GoalStore(database).claim(job_id)
    assert blocked == "busy"
    token = push_lease(_lease(1, owner="worker-b"))
    try:
        outcome, _job = await GoalStore(database).claim(job_id)
    finally:
        reset_lease(token)
    assert outcome == "recovered"


async def test_stale_lease_cannot_commit_after_a_newer_owner(world):
    database, redis, _version, _me = world
    goal_id, job_id = await _seed(database)
    event = _event(goal_id, job_id)
    first = await acquire_lease(redis, event, owner="worker-a", message_id="1-0")
    assert first.action == "run"
    token = push_lease(first.lease)
    try:
        assert (await GoalStore(database).claim(job_id))[0] == "claimed"
    finally:
        reset_lease(token)
    await redis.delete(f"buddy:job-lease:{logical_job_key(event)}")
    second = await acquire_lease(redis, event, owner="worker-b", message_id="1-0")
    assert second.action == "run"
    token = push_lease(second.lease)
    try:
        assert (await GoalStore(database).claim(job_id))[0] == "recovered"
    finally:
        reset_lease(token)
    token = push_lease(first.lease)
    try:
        with pytest.raises(StaleOwnershipError):
            await GoalStore(database).save_step(
                goal_id,
                job_id,
                step={"stepId": "stale", "prompt": "no", "control": "text"},
                draft_updates={},
                status="CLARIFY",
                assistant_text="no",
                model="test",
            )
    finally:
        reset_lease(token)
    stored = await database.goal_jobs.find_one({"_id": ObjectId(job_id)})
    assert stored["status"] == "running"
    assert stored["processingFence"] == int(second.lease.token)
    assert "durableStep" not in stored
    await _forget_job(redis, event)


async def test_transient_failure_is_durable_before_ack_and_is_not_rerun(world, monkeypatch):
    database, redis, _version, _me = world
    goal_id, job_id = await _seed(database)
    stream = f"phase2c:goals:{uuid.uuid4().hex[:8]}"
    group = "goal-workers"
    calls = []

    async def build(goal, messages):
        calls.append("ran")
        raise RuntimeError("model down")

    monkeypatch.setattr("services.goals.jobs.build_clarify_step", build)
    monkeypatch.setattr("apps.api_gateway.workers.goal_worker.get_goal_redis_client", lambda: redis)

    async def handle(event):
        await GoalJobHandler(database).handle(event)

    monkeypatch.setattr("apps.api_gateway.workers.goal_worker.handle_goal_event", handle)
    runner = GoalStreamsRunner()
    runner.redis = redis
    event = _event(goal_id, job_id)
    await redis.xadd(stream, {"event": event.model_dump_json()})
    await redis.xgroup_create(stream, group, id="0", mkstream=True)
    messages = await redis.xreadgroup(group, runner.consumer_name, {stream: ">"}, count=1)
    message_id, fields = messages[0][1][0]
    original = runner._ack

    async def ack(stream_name, group_name, message):
        stored = await database.goal_jobs.find_one({"_id": ObjectId(job_id)})
        assert stored["status"] == "failed"
        await original(stream_name, group_name, message)

    runner._ack = ack
    await runner._handle_message(stream, group, message_id, fields)
    assert calls == ["ran"]
    await _forget_job(redis, event)
    await runner._handle_message(stream, group, message_id, fields)
    assert calls == ["ran"]
    assert _pending_count(await redis.xpending(stream, group)) == 0
    await redis.delete(stream)


async def test_dead_letter_failure_leaves_the_message_pending(world, monkeypatch):
    database, redis, _version, _me = world
    goal_id, job_id = await _seed(database)
    await database.goals.delete_one({"_id": ObjectId(goal_id)})
    stream = f"phase2c:goals:{uuid.uuid4().hex[:8]}"
    group = "goal-workers"
    dlq = f"phase2c:dlq:{uuid.uuid4().hex[:8]}"
    monkeypatch.setattr("apps.api_gateway.workers.goal_worker.get_goal_redis_client", lambda: redis)
    monkeypatch.setattr(settings, "REDIS_DEAD_LETTER_STREAM", dlq)

    async def handle(event):
        await GoalJobHandler(database).handle(event)

    monkeypatch.setattr("apps.api_gateway.workers.goal_worker.handle_goal_event", handle)
    runner = GoalStreamsRunner()
    runner.redis = redis
    event = _event(goal_id, job_id)
    await redis.xadd(stream, {"event": event.model_dump_json()})
    await redis.xgroup_create(stream, group, id="0", mkstream=True)
    original = redis.xadd

    async def xadd(name, *args, **kwargs):
        if name == dlq:
            raise ConnectionError("dead letter unavailable")
        return await original(name, *args, **kwargs)

    redis.xadd = xadd
    messages = await redis.xreadgroup(group, runner.consumer_name, {stream: ">"}, count=1)
    message_id, fields = messages[0][1][0]
    await runner._handle_message(stream, group, message_id, fields)
    redis.xadd = original
    assert (await database.goal_jobs.find_one({"_id": ObjectId(job_id)}))["status"] == "failed"
    assert _pending_count(await redis.xpending(stream, group)) == 1
    assert await redis.xlen(dlq) == 0
    await _forget_job(redis, event)
    await redis.delete(stream, dlq)


async def test_killed_goal_worker_is_recovered_by_a_replacement(world, monkeypatch):
    database, redis, _version, _me = world
    goal_id, job_id = await _seed(database)
    stream = f"phase2c:goals:{uuid.uuid4().hex[:8]}"
    group = "goal-workers"
    enter = f"phase2c:enter:{uuid.uuid4().hex[:8]}"
    monkeypatch.setattr(settings, "JOB_LEASE_TTL_SECONDS", 30)
    monkeypatch.setattr(settings, "JOB_LEASE_HEARTBEAT_SECONDS", 5)
    event = _event(goal_id, job_id)
    await redis.xadd(stream, {"event": event.model_dump_json()})
    script = r"""
import asyncio, os, sys
sys.path.insert(0, os.environ["PHASE2C_ROOT"])

async def main():
    from redis.asyncio import Redis
    from motor.motor_asyncio import AsyncIOMotorClient
    from apps.api_gateway.config.setting import settings
    from services.goals.jobs import GoalJobHandler
    from services.queue.job_lease import acquire_lease
    from services.queue.streams import EventEnvelope
    settings.JOB_LEASE_TTL_SECONDS = 30
    settings.JOB_LEASE_HEARTBEAT_SECONDS = 5
    redis = Redis(host="127.0.0.1", port=6399, decode_responses=True)
    client = AsyncIOMotorClient("mongodb://127.0.0.1:27017", serverSelectionTimeoutMS=2000)
    database = client[os.environ["PHASE2C_DB"]]
    stream = os.environ["PHASE2C_STREAM"]
    group = os.environ["PHASE2C_GROUP"]
    await redis.xgroup_create(stream, group, id="0", mkstream=True)
    messages = await redis.xreadgroup(group, "goal-child", {stream: ">"}, count=1)
    message_id, fields = messages[0][1][0]
    event = EventEnvelope.model_validate_json(fields["event"])
    decision = await acquire_lease(redis, event, owner="goal-child", message_id=str(message_id))
    assert decision.action == "run"
    from services.queue.job_lease import push_lease, reset_lease
    token = push_lease(decision.lease)
    try:
        outcome, _job = await __import__("services.goals.store", fromlist=["GoalStore"]).GoalStore(database).claim(os.environ["PHASE2C_JOB"])
        await redis.set(os.environ["PHASE2C_ENTER"], outcome)
        await asyncio.sleep(40)
    finally:
        reset_lease(token)
    await redis.aclose()
    client.close()

asyncio.run(main())
"""
    env = os.environ.copy()
    env.update(
        {
            "PHASE2C_ROOT": str(ROOT),
            "PHASE2C_DB": database.name,
            "PHASE2C_STREAM": stream,
            "PHASE2C_GROUP": group,
            "PHASE2C_JOB": job_id,
            "PHASE2C_ENTER": enter,
            "REDIS_URL": "redis://127.0.0.1:6399/0",
        }
    )
    redis_pid = _listener_pid()
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        for _ in range(40):
            if proc.poll() is not None:
                stderr = proc.stderr.read().decode("utf-8", errors="replace")[-2000:]
                raise RuntimeError(f"goal worker exited early: {stderr}")
            if await redis.get(enter) == "claimed":
                break
            await asyncio.sleep(0.25)
        else:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/F"], check=False, capture_output=True)
            stderr = proc.stderr.read().decode("utf-8", errors="replace")[-2000:]
            raise RuntimeError(f"goal worker never claimed: {stderr}")
        assert _pending_count(await redis.xpending(stream, group)) == 1
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/F"], check=False, capture_output=True)
        proc.wait(timeout=8)
        assert await redis.ping()
        assert _listener_pid() == redis_pid
        await asyncio.sleep(35)
        messages = await redis.xrange(stream, min="-", max="+")
        message_id, fields = messages[0]
        event = EventEnvelope.model_validate_json(fields["event"])
        decision = await acquire_lease(redis, event, owner="goal-replacement", message_id=message_id)
        assert decision.action == "run"
        calls = []

        async def build(goal, messages):
            calls.append("ran")
            return {
                "step": {"stepId": "ask-1", "prompt": "When?", "control": "text"},
                "draft": {},
                "status": "CLARIFY",
                "model": "test",
            }

        monkeypatch.setattr("services.goals.jobs.build_clarify_step", build)
        token = push_lease(decision.lease)
        try:
            await GoalJobHandler(database).handle(event)
        finally:
            reset_lease(token)
        assert calls == ["ran"]
        stored = await database.goal_jobs.find_one({"_id": ObjectId(job_id)})
        assert stored["status"] == "ready"
        assert stored["processingFence"] == int(decision.lease.token)
        assert await database.goal_messages.count_documents({"jobId": ObjectId(job_id)}) == 1
        await redis.xack(stream, group, message_id)
        assert _pending_count(await redis.xpending(stream, group)) == 0
    finally:
        if proc.poll() is None:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/F"], check=False, capture_output=True)
            proc.wait(timeout=8)
        await _forget_job(redis, event)
        await redis.delete(stream, enter)


async def test_one_published_goal_event_is_processed_once(world, monkeypatch):
    database, redis, _version, _me = world
    goal_id, job_id = await _seed(database)
    event = _event(goal_id, job_id)
    stream = f"phase2c:goals:{uuid.uuid4().hex[:8]}"
    payload = event.model_dump_json()
    fingerprint = submission_fingerprint(event)

    async def publish():
        return await atomic_publish(
            redis,
            event_id=event.eventId,
            queue=stream,
            mode="stream",
            payload=payload,
            fingerprint=fingerprint,
            correlation_id=goal_id,
            event_type=event.eventType,
        )

    results = await asyncio.gather(publish(), publish())
    assert sorted(item.outcome for item in results) == ["duplicate", "published"]
    assert await redis.xlen(stream) == 1
    calls = []

    async def build(goal, messages):
        calls.append("ran")
        return {
            "step": {"stepId": "ask-1", "prompt": "When?", "control": "text"},
            "draft": {},
            "status": "CLARIFY",
            "model": "test",
        }

    monkeypatch.setattr("services.goals.jobs.build_clarify_step", build)
    group = "goal-workers"
    monkeypatch.setattr("apps.api_gateway.workers.goal_worker.get_goal_redis_client", lambda: redis)

    async def handle(incoming):
        await GoalJobHandler(database).handle(incoming)

    monkeypatch.setattr("apps.api_gateway.workers.goal_worker.handle_goal_event", handle)
    runner = GoalStreamsRunner()
    runner.redis = redis
    await redis.xgroup_create(stream, group, id="0", mkstream=True)
    messages = await redis.xreadgroup(group, runner.consumer_name, {stream: ">"}, count=1)
    message_id, fields = messages[0][1][0]
    await runner._handle_message(stream, group, message_id, fields)
    assert calls == ["ran"]
    assert _pending_count(await redis.xpending(stream, group)) == 0
    await _forget_job(redis, event)
    await redis.delete(stream, dedupe_key(event.eventId))


async def test_ingress_dedupe_does_not_block_goal_recovery(world, monkeypatch):
    database, redis, _version, _me = world
    goal_id, job_id = await _seed(database, status="running", fence=1)
    event = _event(goal_id, job_id)
    stream = f"phase2c:goals:{uuid.uuid4().hex[:8]}"
    published = await atomic_publish(
        redis,
        event_id=event.eventId,
        queue=stream,
        mode="stream",
        payload=event.model_dump_json(),
        fingerprint=submission_fingerprint(event),
        correlation_id=goal_id,
        event_type=event.eventType,
    )
    assert published.outcome == "published"
    calls = []

    async def build(goal, messages):
        calls.append("ran")
        return {
            "step": {"stepId": "ask-1", "prompt": "When?", "control": "text"},
            "draft": {},
            "status": "CLARIFY",
            "model": "test",
        }

    monkeypatch.setattr("services.goals.jobs.build_clarify_step", build)
    token = push_lease(_lease(2, owner="worker-b"))
    try:
        await GoalJobHandler(database).handle(event)
    finally:
        reset_lease(token)
    assert calls == ["ran"]
    assert await redis.xlen(stream) == 1
    again = await atomic_publish(
        redis,
        event_id=event.eventId,
        queue=stream,
        mode="stream",
        payload=event.model_dump_json(),
        fingerprint=submission_fingerprint(event),
        correlation_id=goal_id,
        event_type=event.eventType,
    )
    assert again.outcome == "duplicate"
    assert await redis.xlen(stream) == 1
    await redis.delete(stream, dedupe_key(event.eventId))


def _pending_count(raw) -> int:
    if isinstance(raw, dict):
        return int(raw.get("pending") or 0)
    if isinstance(raw, (list, tuple)):
        return int(raw[0] or 0)
    return int(raw or 0)


async def _forget_job(redis, event: EventEnvelope) -> None:
    logical = logical_job_key(event)
    await redis.delete(f"buddy:job-lease:{logical}", f"buddy:job-done:{logical}")


def _listener_pid() -> str | None:
    listener = subprocess.run(["netstat", "-ano"], check=False, capture_output=True, text=True)
    for line in listener.stdout.splitlines():
        if f"127.0.0.1:{REDIS_PORT}" in line and "LISTENING" in line:
            return line.split()[-1]
    return None
