"""Real Redis 7 speech and vector transport. Connects only to 127.0.0.1:6399."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
from redis.asyncio import Redis
from redis.exceptions import ResponseError

from apps.api_gateway.config.setting import settings
from apps.api_gateway.workers import speech_worker, vector_worker
from services.queue import legacy_list_migration, redis_queue
from services.queue.job_lease import LeavePending
from services.queue.legacy_list_migration import (
    export_unacked_stream_to_list,
    migrate_speech_and_vector_backlog,
    parse_legacy_speech_item,
    rollback_speech_vector_to_lists,
)
from services.queue.streams import EventEnvelope

pytestmark = pytest.mark.integration

HOST = "127.0.0.1"
PORT = 6399
ROOT = Path(__file__).resolve().parents[1]


def _names(redis: Redis) -> tuple[str, str]:
    suffix = uuid.uuid4().hex[:10]
    speech = f"phase2b:speech:{suffix}"
    vector = f"phase2b:vector:{suffix}"
    redis.test_streams.append(speech)  # type: ignore[attr-defined]
    redis.test_streams.append(vector)  # type: ignore[attr-defined]
    return speech, vector


async def _local_redis() -> Redis:
    client = Redis(host=HOST, port=PORT, decode_responses=True, socket_connect_timeout=2, socket_timeout=2)
    try:
        info = await client.info("server")
    except Exception:
        await client.aclose()
        pytest.fail(f"Redis 7 is not accepting connections on {HOST}:{PORT}")
    version = str(info.get("redis_version") or "")
    pool = client.connection_pool.connection_kwargs
    if pool.get("host") != HOST or int(pool.get("port")) != PORT or not version.startswith("7."):
        await client.aclose()
        pytest.fail(f"refusing Redis {version} at {pool.get('host')}:{pool.get('port')}")
    return client


async def _delete_test_keys(client: Redis) -> None:
    for pattern in ("buddy:*", "phase2b:*", "speech_job:*", "speech_vector_done:*"):
        cursor = 0
        while True:
            cursor, keys = await client.scan(cursor=cursor, match=pattern, count=200)
            if keys:
                await client.delete(*keys)
            if cursor == 0:
                break


@pytest.fixture
async def redis():
    client = await _local_redis()
    client.test_streams = []  # type: ignore[attr-defined]
    dbsize = int(await client.dbsize())
    if dbsize > 200:
        await client.aclose()
        pytest.fail(f"refusing non-empty disposable Redis dbsize={dbsize}")
    await _delete_test_keys(client)
    yield client
    cleaner = client
    try:
        await cleaner.ping()
    except Exception:
        cleaner = Redis(host=HOST, port=PORT, decode_responses=True, socket_connect_timeout=2, socket_timeout=2)
    try:
        await _delete_test_keys(cleaner)
        for stream in client.test_streams:  # type: ignore[attr-defined]
            await cleaner.delete(stream)
    finally:
        await cleaner.aclose()
        if cleaner is not client:
            await client.aclose()


def _job(job_id: str, sequence: int) -> dict:
    return {
        "job_id": job_id,
        "user_id": "user-1",
        "space_id": "space-1",
        "conversation_id": "conv-1",
        "sequence_number": sequence,
        "filename": "audio.webm",
        "content_type": "audio/webm",
        "file_path": "resources/audio_jobs/audio.webm",
        "status": "queued",
    }


def _use_streams(monkeypatch, redis: Redis, speech: str, vector: str) -> None:
    monkeypatch.setattr(settings, "SPEECH_VECTOR_QUEUE_TRANSPORT", "stream")
    monkeypatch.setattr(settings, "REDIS_SPEECH_STREAM", speech)
    monkeypatch.setattr(settings, "REDIS_VECTOR_STREAM", vector)
    monkeypatch.setattr(settings, "REDIS_SPEECH_GROUP", "speech-workers")
    monkeypatch.setattr(settings, "REDIS_VECTOR_GROUP", "vector-workers")
    monkeypatch.setattr(redis_queue, "redis_client", redis)


async def _read_events(redis: Redis, stream: str) -> list[EventEnvelope]:
    rows = await redis.xrange(stream, min="-", max="+")
    return [EventEnvelope.model_validate_json(fields["event"]) for _, fields in rows]


async def test_speech_and_vector_publish_on_redis_7(redis, monkeypatch):
    speech, vector = _names(redis)
    _use_streams(monkeypatch, redis, speech, vector)
    job = _job("job-a", 1)
    assert await redis_queue.publish_speech_job(job, redis=redis) == speech
    await redis.hset(f"speech_job:{job['job_id']}", mapping={"user_id": "user-1", "space_id": "space-1"})
    assert await redis_queue.publish_completed_speech_job(job["job_id"], redis=redis) == vector
    speech_events = await _read_events(redis, speech)
    vector_events = await _read_events(redis, vector)
    assert speech_events[0].eventType == "speech.transcribe.requested"
    assert speech_events[0].payload["jobId"] == "job-a"
    assert vector_events[0].eventType == "vector.index.requested"
    assert vector_events[0].payload["jobId"] == "job-a"
    info = await redis.info("server")
    assert str(info["redis_version"]).startswith("7.")


async def test_speech_processing_acks_only_after_durable_result(redis, monkeypatch):
    speech, vector = _names(redis)
    _use_streams(monkeypatch, redis, speech, vector)
    calls = []

    async def transcribe(**kwargs):
        calls.append(kwargs["job_id"])
        return {"transcript": "hello", "provider": "test", "language_code": "en"}

    monkeypatch.setattr(speech_worker, "transcribe_from_path_with_fallback", transcribe)
    await redis_queue.publish_speech_job(_job("job-stt", 3), redis=redis)
    consumer = speech_worker.build_speech_stream_consumer(redis=redis)
    await consumer.ensure_group()
    messages = await consumer.read_batch(block_ms=50)
    await consumer.dispatch_messages(messages)
    assert calls == ["job-stt"]
    stored = await redis.hgetall("speech_job:job-stt")
    assert stored["status"] == "completed"
    assert json.loads(stored["result"])["transcript"] == "hello"
    assert await redis.xpending(speech, settings.REDIS_SPEECH_GROUP)
    pending = await redis.xpending(speech, "speech-workers")
    assert int(pending["pending"] if isinstance(pending, dict) else pending[0]) == 0
    vector_events = await _read_events(redis, vector)
    assert vector_events[0].payload["jobId"] == "job-stt"


async def test_vector_processing_acks_after_the_embedding_is_stored(redis, monkeypatch):
    speech, vector = _names(redis)
    _use_streams(monkeypatch, redis, speech, vector)
    stored = []

    async def get_result(job_id):
        data = await redis.hgetall(f"speech_job:{job_id}")
        if data.get("result"):
            data["result"] = json.loads(data["result"])
        return data or None

    class Repository:
        async def complete_transcript_chunk(self, **kwargs):
            stored.append(("chunk", kwargs["sequence_number"], kwargs["raw_text"]))

        async def get_conversation(self, conversation_id):
            return None

    async def embed(**kwargs):
        stored.append(("embed", kwargs["job_id"]))

    monkeypatch.setattr(vector_worker, "get_job_result", get_result)
    monkeypatch.setattr(vector_worker, "get_database", lambda: None)
    monkeypatch.setattr(vector_worker, "ConversationRepository", lambda database: Repository())
    monkeypatch.setattr(vector_worker, "store_transcript_in_vector_db", embed)
    await redis.hset(
        "speech_job:job-vec",
        mapping={
            "status": "completed",
            "user_id": "user-1",
            "space_id": "space-1",
            "conversation_id": "conv-1",
            "sequence_number": "4",
            "result": json.dumps({"transcript": "hello vector", "provider": "test"}),
        },
    )
    await redis_queue.publish_completed_speech_job("job-vec", redis=redis)
    consumer = vector_worker.build_vector_stream_consumer(redis=redis)
    await consumer.ensure_group()
    await consumer.dispatch_messages(await consumer.read_batch(block_ms=50))
    assert ("chunk", 4, "hello vector") in stored
    assert ("embed", "job-vec") in stored
    pending = await redis.xpending(vector, "vector-workers")
    assert int(pending["pending"] if isinstance(pending, dict) else pending[0]) == 0
    assert await redis.exists("speech_job:job-vec") == 0


async def test_crash_before_persistence_leaves_the_speech_message_pending(redis, monkeypatch):
    speech, vector = _names(redis)
    _use_streams(monkeypatch, redis, speech, vector)

    async def transcribe(**kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr(speech_worker, "transcribe_from_path_with_fallback", transcribe)
    await redis_queue.publish_speech_job(_job("job-crash", 1), redis=redis)
    consumer = speech_worker.build_speech_stream_consumer(redis=redis)
    await consumer.ensure_group()
    messages = await consumer.read_batch(block_ms=50)
    await consumer.dispatch_messages(messages)
    pending = await redis.xpending(speech, "speech-workers")
    assert int(pending["pending"] if isinstance(pending, dict) else pending[0]) == 1
    assert await redis.hget("speech_job:job-crash", "status") in {None, "processing"}


async def test_crash_after_persistence_does_not_transcribe_again(redis, monkeypatch):
    speech, vector = _names(redis)
    _use_streams(monkeypatch, redis, speech, vector)
    calls = []

    async def transcribe(**kwargs):
        calls.append(kwargs["job_id"])
        return {"transcript": "once", "provider": "test"}

    real_save = redis_queue.save_job_result

    async def save_then_crash(job_id, result):
        await real_save(job_id, result)
        raise asyncio.CancelledError()

    monkeypatch.setattr(speech_worker, "transcribe_from_path_with_fallback", transcribe)
    monkeypatch.setattr(speech_worker, "save_job_result", save_then_crash)
    await redis_queue.publish_speech_job(_job("job-once", 1), redis=redis)
    consumer = speech_worker.build_speech_stream_consumer(redis=redis)
    await consumer.ensure_group()
    first = await consumer.read_batch(block_ms=50)
    await consumer.dispatch_messages(first)
    assert calls == ["job-once"]
    pending = await redis.xpending_range(speech, "speech-workers", min="-", max="+", count=10)
    assert len(pending) == 1
    message_id = pending[0]["message_id"]
    await redis.delete(f"buddy:job-lease:{redis_queue.SPEECH_EVENT_TYPE}|job:job-once")
    fields = (await redis.xrange(speech, min=message_id, max=message_id))[0][1]
    monkeypatch.setattr(speech_worker, "save_job_result", real_save)
    await consumer._semaphore.acquire()
    await consumer._handle_message(message_id, fields)
    assert calls == ["job-once"]
    assert await redis.xpending_range(speech, "speech-workers", min="-", max="+", count=10) == []


async def test_two_copies_of_one_speech_job_have_one_owner(redis, monkeypatch):
    speech, vector = _names(redis)
    _use_streams(monkeypatch, redis, speech, vector)
    entered = 0
    release = asyncio.Event()

    async def transcribe(**kwargs):
        nonlocal entered
        entered += 1
        await release.wait()
        return {"transcript": "one", "provider": "test"}

    monkeypatch.setattr(speech_worker, "transcribe_from_path_with_fallback", transcribe)
    job = _job("job-race", 1)
    await redis_queue.publish_speech_job(job, redis=redis)
    await redis_queue.publish_speech_job(job, redis=redis)
    first = speech_worker.build_speech_stream_consumer(redis=redis)
    second = speech_worker.build_speech_stream_consumer(redis=redis)
    await first.ensure_group()
    await second.ensure_group()
    first_messages = await first.read_batch(block_ms=50)
    await first._semaphore.acquire()
    first_task = asyncio.create_task(
        first._handle_message(first_messages[0][1][0][0], first_messages[0][1][0][1])
    )
    for _ in range(50):
        if entered:
            break
        await asyncio.sleep(0.05)
    await second.dispatch_messages(await second.read_batch(block_ms=50))
    assert entered == 1
    release.set()
    await asyncio.wait_for(first_task, timeout=3)
    assert entered == 1


async def test_heartbeat_blocks_reclaim_during_a_long_transcription(redis, monkeypatch):
    speech, vector = _names(redis)
    _use_streams(monkeypatch, redis, speech, vector)
    monkeypatch.setattr(settings, "JOB_LEASE_TTL_SECONDS", 2)
    monkeypatch.setattr(settings, "JOB_LEASE_HEARTBEAT_SECONDS", 0.05)
    monkeypatch.setattr(settings, "REDIS_CLAIM_IDLE_MS", 500)
    calls = []

    async def transcribe(**kwargs):
        calls.append(kwargs["job_id"])
        await asyncio.sleep(3)
        return {"transcript": "slow", "provider": "test"}

    monkeypatch.setattr(speech_worker, "transcribe_from_path_with_fallback", transcribe)
    await redis_queue.publish_speech_job(_job("job-slow", 1), redis=redis)
    owner = speech_worker.build_speech_stream_consumer(redis=redis)
    rival = speech_worker.build_speech_stream_consumer(redis=redis)
    await owner.ensure_group()
    messages = await owner.read_batch(block_ms=50)
    await owner._semaphore.acquire()
    task = asyncio.create_task(owner._handle_message(messages[0][1][0][0], messages[0][1][0][1]))
    await asyncio.sleep(2.4)
    await rival.claim_stale()
    await asyncio.wait_for(task, timeout=5)
    assert calls == ["job-slow"]
    pending = await redis.xpending(speech, "speech-workers")
    assert int(pending["pending"] if isinstance(pending, dict) else pending[0]) == 0


async def test_vector_retry_is_written_before_ack_and_exhaustion_dead_letters(redis, monkeypatch):
    speech, vector = _names(redis)
    _use_streams(monkeypatch, redis, speech, vector)
    ops: list[str] = []
    failures = []

    class Recording(Redis):
        async def xadd(self, name, fields, *args, **kwargs):
            ops.append(f"xadd:{name}")
            return await super().xadd(name, fields, *args, **kwargs)

        async def xack(self, name, groupname, *ids, **kwargs):
            ops.append(f"xack:{name}")
            return await super().xack(name, groupname, *ids, **kwargs)

    recording = Recording(host=HOST, port=PORT, decode_responses=True, socket_connect_timeout=2, socket_timeout=2)
    monkeypatch.setattr(redis_queue, "redis_client", recording)

    async def get_result(job_id):
        data = await recording.hgetall(f"speech_job:{job_id}")
        if data.get("result"):
            data["result"] = json.loads(data["result"])
        return data or None

    async def embed(**kwargs):
        raise RuntimeError("embedding down")

    async def record_failure(*args, **kwargs):
        failures.append(args[0])

    monkeypatch.setattr(vector_worker, "get_job_result", get_result)
    monkeypatch.setattr(vector_worker, "get_database", lambda: None)
    monkeypatch.setattr(vector_worker, "ConversationRepository", lambda database: None)
    monkeypatch.setattr(vector_worker, "store_transcript_in_vector_db", embed)
    monkeypatch.setattr(vector_worker, "_record_vector_failure", record_failure)
    await recording.hset(
        "speech_job:job-retry",
        mapping={
            "status": "completed",
            "user_id": "user-1",
            "space_id": "space-1",
            "result": json.dumps({"transcript": "keep me", "provider": "test"}),
        },
    )
    await redis_queue.publish_completed_speech_job("job-retry", redis=recording)
    ops.clear()
    consumer = vector_worker.build_vector_stream_consumer(redis=recording)
    consumer._uses_default_redis = True
    await consumer.ensure_group()
    await consumer.dispatch_messages(await consumer.read_batch(block_ms=50))
    assert f"xadd:{settings.REDIS_RETRY_STREAM}" in ops
    assert ops.index(f"xadd:{settings.REDIS_RETRY_STREAM}") < ops.index(f"xack:{vector}")
    assert await recording.xlen(settings.REDIS_RETRY_STREAM) >= 1

    consumer.max_retries = 0
    consumer._uses_default_redis = True
    retry_rows = await recording.xrange(settings.REDIS_RETRY_STREAM, min="-", max="+")
    event = EventEnvelope.model_validate_json(retry_rows[-1][1]["event"])
    event = event.model_copy(update={"attempt": vector_worker.VECTOR_MAX_ATTEMPTS - 1})
    await recording.xadd(vector, {"event": event.model_dump_json()})
    ops.clear()
    await consumer.dispatch_messages(await consumer.read_batch(block_ms=50))
    assert failures == ["job-retry"]
    assert any(item.startswith(f"xadd:{settings.REDIS_DEAD_LETTER_STREAM}") for item in ops)
    dead_letter = next(index for index, item in enumerate(ops) if item.startswith("xadd:"))
    ack = next(index for index, item in enumerate(ops) if item.startswith("xack:"))
    assert dead_letter < ack
    await recording.aclose()


async def test_legacy_backlog_migration_is_resumable(redis, monkeypatch):
    speech, vector = _names(redis)
    _use_streams(monkeypatch, redis, speech, vector)
    speech_list = f"phase2b:list:speech:{uuid.uuid4().hex[:8]}"
    vector_list = f"phase2b:list:vector:{uuid.uuid4().hex[:8]}"
    redis.test_streams.extend([speech_list, vector_list, f"{speech_list}:migrating", f"{vector_list}:migrating"])  # type: ignore[attr-defined]
    first = _job("job-m1", 1)
    second = _job("job-m2", 2)
    await redis.lpush(speech_list, json.dumps(second))
    await redis.lpush(speech_list, json.dumps(first))
    await redis.lpush(vector_list, "job-m1")
    dry = await migrate_speech_and_vector_backlog(
        redis,
        execute=False,
        limit=10,
        speech_queue=speech_list,
        completed_queue=vector_list,
    )
    assert dry["speech"]["migrated"] == 0
    assert await redis.llen(speech_list) == 2
    partial = await migrate_speech_and_vector_backlog(
        redis,
        execute=True,
        limit=1,
        speech_queue=speech_list,
        completed_queue=vector_list,
    )
    assert partial["speech"]["migrated"] == 1
    assert await redis.llen(speech_list) == 1
    await redis.lmove(speech_list, f"{speech_list}:migrating", "RIGHT", "LEFT")
    assert await redis.llen(f"{speech_list}:migrating") == 1
    finished = await migrate_speech_and_vector_backlog(
        redis,
        execute=True,
        limit=10,
        speech_queue=speech_list,
        completed_queue=vector_list,
    )
    assert finished["speech"]["processing_depth"] == 0
    assert await redis.llen(speech_list) == 0
    assert await redis.llen(vector_list) == 0
    events = await _read_events(redis, speech)
    assert {event.payload["jobId"] for event in events} == {"job-m1", "job-m2"}
    assert parse_legacy_speech_item(json.dumps(first))["job_id"] == "job-m1"
    vector_events = await _read_events(redis, vector)
    assert [event.payload["jobId"] for event in vector_events] == ["job-m1"]


async def test_stopped_stream_consumer_leaves_the_job_recoverable(redis, monkeypatch):
    speech, vector = _names(redis)
    _use_streams(monkeypatch, redis, speech, vector)
    await redis_queue.publish_speech_job(_job("job-rollback", 1), redis=redis)
    assert await redis.xlen(speech) == 1
    assert await redis.llen("speech_transcribe_queue") == 0
    exported = await export_unacked_stream_to_list(
        redis,
        stream=speech,
        group="speech-workers",
        destination=f"{speech}:legacy",
        to_list_item=lambda raw: __import__(
            "services.queue.legacy_list_migration", fromlist=["_speech_list_item"]
        )._speech_list_item(raw),
        execute=True,
        limit=10,
    )
    assert exported["exported"] == 1
    restored = json.loads(await redis.rpop(f"{speech}:legacy"))
    assert restored["jobId"] == "job-rollback"
    assert await redis.xlen(speech) == 1


async def test_redis_interruption_keeps_the_speech_message_pending(redis, monkeypatch):
    import subprocess

    speech, vector = _names(redis)
    _use_streams(monkeypatch, redis, speech, vector)
    started = asyncio.Event()
    continue_handler = asyncio.Event()

    async def transcribe(**kwargs):
        started.set()
        await continue_handler.wait()
        return {"transcript": "after", "provider": "test"}

    monkeypatch.setattr(speech_worker, "transcribe_from_path_with_fallback", transcribe)
    await redis_queue.publish_speech_job(_job("job-outage", 1), redis=redis)
    worker = Redis(host=HOST, port=PORT, decode_responses=True, socket_connect_timeout=1, socket_timeout=1)
    monkeypatch.setattr(redis_queue, "redis_client", worker)
    consumer = speech_worker.build_speech_stream_consumer(redis=worker)
    await consumer.ensure_group()
    messages = await consumer.read_batch(block_ms=50)
    await consumer._semaphore.acquire()
    task = asyncio.create_task(consumer._handle_message(messages[0][1][0][0], messages[0][1][0][1]))
    await asyncio.wait_for(started.wait(), timeout=2)
    await redis.config_set("appendonly", "yes")
    for _ in range(50):
        persistence = await redis.info("persistence")
        if int(persistence.get("aof_enabled") or 0) == 1 and int(persistence.get("aof_rewrite_in_progress") or 0) == 0:
            break
        await asyncio.sleep(0.1)
    listener = subprocess.run(["netstat", "-ano"], check=False, capture_output=True, text=True)
    pid = None
    for line in listener.stdout.splitlines():
        if f"127.0.0.1:{PORT}" in line and "LISTENING" in line:
            pid = line.split()[-1]
            break
    if pid is None:
        raise RuntimeError("no Redis listener on 127.0.0.1:6399")
    message_id = messages[0][1][0][0]
    subprocess.run(["taskkill", "/PID", pid, "/F"], check=False, capture_output=True)
    continue_handler.set()
    try:
        await asyncio.wait_for(task, timeout=8)
    except Exception:
        # Redis is down, so the retry write cannot finish. The message must stay pending.
        pass
    finally:
        await _restart_local_redis()
        await worker.aclose()
    fresh = Redis(host=HOST, port=PORT, decode_responses=True, socket_connect_timeout=2, socket_timeout=2)
    try:
        pending = await fresh.xpending_range(speech, "speech-workers", min=message_id, max=message_id, count=1)
    finally:
        await fresh.aclose()
    assert pending
    assert pending[0]["message_id"] == message_id


async def _pending(redis: Redis, stream: str, group: str) -> int:
    try:
        info = await redis.xpending(stream, group)
    except ResponseError:
        return 0
    if isinstance(info, dict):
        return int(info.get("pending") or 0)
    return int(info[0] or 0)


async def _rehandle(consumer, redis: Redis, stream: str, group: str, lease_key: str) -> None:
    pending = await redis.xpending_range(stream, group, min="-", max="+", count=10)
    assert len(pending) == 1
    message_id = pending[0]["message_id"]
    await redis.delete(lease_key)
    fields = (await redis.xrange(stream, min=message_id, max=message_id))[0][1]
    await consumer._semaphore.acquire()
    await consumer._handle_message(message_id, fields)


def _listener_pid() -> str | None:
    listener = subprocess.run(["netstat", "-ano"], check=False, capture_output=True, text=True)
    for line in listener.stdout.splitlines():
        if f"127.0.0.1:{PORT}" in line and "LISTENING" in line:
            return line.split()[-1]
    return None


async def test_messages_published_before_the_group_are_consumed(redis, monkeypatch):
    speech, vector = _names(redis)
    _use_streams(monkeypatch, redis, speech, vector)
    consumed = []

    async def transcribe(**kwargs):
        consumed.append(kwargs["job_id"])
        return {"transcript": "kept", "provider": "test"}

    for index in range(3):
        await redis_queue.publish_speech_job(_job(f"job-pre-{index}", index), redis=redis)
    assert await redis.xlen(speech) == 3
    assert await redis.xinfo_groups(speech) == []
    monkeypatch.setattr(speech_worker, "transcribe_from_path_with_fallback", transcribe)
    consumer = speech_worker.build_speech_stream_consumer(redis=redis)
    await consumer.ensure_group()
    await consumer.dispatch_messages(await consumer.read_batch(block_ms=50))
    assert sorted(consumed) == ["job-pre-0", "job-pre-1", "job-pre-2"]
    assert await _pending(redis, speech, "speech-workers") == 0
    for index in range(3):
        assert await redis.hget(f"speech_job:job-pre-{index}", "status") == "completed"


async def test_rollback_copies_retries_migrating_and_pauses_publishers(redis, monkeypatch):
    speech, vector = _names(redis)
    speech_list = f"phase2b:speech-list:{uuid.uuid4().hex[:8]}"
    vector_list = f"phase2b:vector-list:{uuid.uuid4().hex[:8]}"
    retry_stream = f"phase2b:retry:{uuid.uuid4().hex[:8]}"
    _use_streams(monkeypatch, redis, speech, vector)
    await redis_queue.publish_speech_job(_job("job-lag", 1), redis=redis)
    event = redis_queue.speech_job_event(_job("job-retry", 2))
    await redis.xadd(retry_stream, {"event": event.model_dump_json(), "targetStream": speech, "notBefore": "0"})
    await redis.xadd(retry_stream, {"event": event.model_dump_json(), "targetStream": "buddy:other:jobs", "notBefore": "0"})
    await redis.lpush(f"{speech_list}:migrating", json.dumps(_job("job-migrating", 3)))
    release = asyncio.Event()
    entered = asyncio.Event()
    real_export = export_unacked_stream_to_list

    async def pause_first_export(*args, **kwargs):
        if not entered.is_set():
            entered.set()
            await release.wait()
        return await real_export(*args, **kwargs)

    monkeypatch.setattr(legacy_list_migration, "export_unacked_stream_to_list", pause_first_export)
    task = asyncio.create_task(
        rollback_speech_vector_to_lists(
            redis,
            execute=True,
            speech_stream=speech,
            vector_stream=vector,
            speech_list=speech_list,
            vector_list=vector_list,
            retry_stream=retry_stream,
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=2)
    with pytest.raises(redis_queue.SpeechVectorCutoverLocked):
        await redis_queue.publish_speech_job(_job("job-during", 4), redis=redis)
    with pytest.raises(redis_queue.SpeechVectorCutoverLocked):
        await redis_queue.publish_completed_speech_job("job-during", redis=redis)
    release.set()
    result = await task
    assert result["publishers_paused"] == 0
    assert result["retry_left"] == 0
    assert await redis.exists(redis_queue.EXPORT_LOCK_KEY) == 0
    exported = await redis.lrange(speech_list, 0, -1)
    assert any("job-lag" in item for item in exported)
    assert any("job-retry" in item for item in exported)
    assert any("job-migrating" in item for item in exported)
    assert await redis.llen(f"{speech_list}:migrating") == 0
    assert await redis.xlen(retry_stream) == 1
    await redis_queue.publish_speech_job(_job("job-after", 5), redis=redis)
    assert await redis.xlen(speech) >= 1


async def test_rollback_leaves_a_live_lease_pending_then_exports_it_once(redis, monkeypatch):
    speech, vector = _names(redis)
    speech_list = f"phase2b:speech-list:{uuid.uuid4().hex[:8]}"
    vector_list = f"phase2b:vector-list:{uuid.uuid4().hex[:8]}"
    _use_streams(monkeypatch, redis, speech, vector)

    async def transcribe(**kwargs):
        return {"transcript": "busy", "provider": "test"}

    monkeypatch.setattr(speech_worker, "transcribe_from_path_with_fallback", transcribe)
    monkeypatch.setattr(legacy_list_migration, "reclaim_min_idle_ms", lambda: 60000)
    await redis_queue.publish_speech_job(_job("job-live", 1), redis=redis)
    consumer = speech_worker.build_speech_stream_consumer(redis=redis)
    await consumer.ensure_group()
    messages = await consumer.read_batch(block_ms=50)
    assert messages
    first = await export_unacked_stream_to_list(
        redis,
        stream=speech,
        group="speech-workers",
        destination=speech_list,
        to_list_item=legacy_list_migration._speech_list_item,
        execute=True,
    )
    assert int(first["still_pending"]) == 1
    assert await redis.llen(speech_list) == 0
    monkeypatch.setattr(legacy_list_migration, "reclaim_min_idle_ms", lambda: 0)
    second = await export_unacked_stream_to_list(
        redis,
        stream=speech,
        group="speech-workers",
        destination=speech_list,
        to_list_item=legacy_list_migration._speech_list_item,
        execute=True,
    )
    third = await export_unacked_stream_to_list(
        redis,
        stream=speech,
        group="speech-workers",
        destination=speech_list,
        to_list_item=legacy_list_migration._speech_list_item,
        execute=True,
    )
    assert int(second["still_pending"]) == 0
    assert await redis.llen(speech_list) == 1
    assert int(third["exported"]) == 0
    assert await redis.llen(speech_list) == 1
    assert "job-live" in (await redis.lrange(speech_list, 0, -1))[0]


async def test_interrupted_export_restarts_without_a_duplicate_list_item(redis, monkeypatch):
    speech, vector = _names(redis)
    speech_list = f"phase2b:speech-list:{uuid.uuid4().hex[:8]}"
    _use_streams(monkeypatch, redis, speech, vector)
    await redis_queue.publish_speech_job(_job("job-interrupt", 1), redis=redis)
    original = redis.xack
    calls = {"n": 0}

    async def flaky_ack(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("export interrupted")
        return await original(*args, **kwargs)

    redis.xack = flaky_ack
    with pytest.raises(ConnectionError):
        await export_unacked_stream_to_list(
            redis,
            stream=speech,
            group="speech-workers",
            destination=speech_list,
            to_list_item=legacy_list_migration._speech_list_item,
            execute=True,
        )
    assert await redis.llen(speech_list) == 1
    redis.xack = original
    again = await export_unacked_stream_to_list(
        redis,
        stream=speech,
        group="speech-workers",
        destination=speech_list,
        to_list_item=legacy_list_migration._speech_list_item,
        execute=True,
    )
    assert int(again["still_pending"]) == 0
    assert await redis.llen(speech_list) == 1
    assert await redis.sismember(legacy_list_migration.EXPORT_LEDGER_KEY, "job-interrupt") == 1


async def test_killed_worker_process_is_reclaimed_once(redis, monkeypatch):
    speech, vector = _names(redis)
    job_id = f"job-kill-{uuid.uuid4().hex[:8]}"
    enter = f"phase2b:enter:{uuid.uuid4().hex[:8]}"
    _use_streams(monkeypatch, redis, speech, vector)
    monkeypatch.setattr(settings, "JOB_LEASE_TTL_SECONDS", 30)
    monkeypatch.setattr(settings, "JOB_LEASE_HEARTBEAT_SECONDS", 5)
    monkeypatch.setattr(settings, "REDIS_CLAIM_IDLE_MS", 1000)
    script = f"""
import asyncio, os, sys
sys.path.insert(0, os.environ["PHASE2B_ROOT"])

async def main():
    from redis.asyncio import Redis
    from apps.api_gateway.config.setting import settings
    from apps.api_gateway.workers import speech_worker
    from services.queue import redis_queue
    client = Redis(host="127.0.0.1", port=6399, decode_responses=True)
    redis_queue.redis_client = client
    settings.SPEECH_VECTOR_QUEUE_TRANSPORT = "stream"
    settings.REDIS_SPEECH_STREAM = os.environ["PHASE2B_SPEECH"]
    settings.REDIS_VECTOR_STREAM = os.environ["PHASE2B_VECTOR"]
    settings.REDIS_SPEECH_GROUP = "speech-workers"
    settings.REDIS_VECTOR_GROUP = "vector-workers"
    settings.JOB_LEASE_TTL_SECONDS = 30
    settings.JOB_LEASE_HEARTBEAT_SECONDS = 5
    settings.REDIS_CLAIM_IDLE_MS = 1000
    enter = os.environ["PHASE2B_ENTER"]
    job_id = os.environ["PHASE2B_JOB"]

    async def transcribe(**kwargs):
        await client.incr(enter)
        await asyncio.sleep(30)
        return {{"transcript": "should-not-save", "provider": "test"}}

    speech_worker.transcribe_from_path_with_fallback = transcribe
    job = {{
        "job_id": job_id,
        "user_id": "user-1",
        "space_id": "space-1",
        "conversation_id": "conv-1",
        "sequence_number": 1,
        "filename": "audio.webm",
        "content_type": "audio/webm",
        "file_path": "resources/audio_jobs/audio.webm",
        "status": "queued",
    }}
    await redis_queue.publish_speech_job(job, redis=client)
    await speech_worker.build_speech_stream_consumer(redis=client).run_forever()

asyncio.run(main())
"""
    env = os.environ.copy()
    env.update(
        {
            "PHASE2B_ROOT": str(ROOT),
            "PHASE2B_SPEECH": speech,
            "PHASE2B_VECTOR": vector,
            "PHASE2B_ENTER": enter,
            "PHASE2B_JOB": job_id,
            "REDIS_URL": "redis://127.0.0.1:6399/0",
            "SPEECH_VECTOR_QUEUE_TRANSPORT": "stream",
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
        for _ in range(80):
            if proc.poll() is not None:
                stderr = proc.stderr.read().decode("utf-8", errors="replace")[-2000:]
                raise RuntimeError(f"worker exited early: {stderr}")
            if int(await redis.get(enter) or 0) >= 1:
                break
            await asyncio.sleep(0.25)
        else:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/F"], check=False, capture_output=True)
            stderr = proc.stderr.read().decode("utf-8", errors="replace")[-2000:]
            raise RuntimeError(f"worker never started the speech job: {stderr}")
        assert await _pending(redis, speech, "speech-workers") == 1
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/F"], check=False, capture_output=True)
        proc.wait(timeout=8)
        assert await redis.ping()
        assert _listener_pid() == redis_pid
        assert await _pending(redis, speech, "speech-workers") == 1
        await asyncio.sleep(32)
        assert await redis.exists(f"buddy:job-lease:{redis_queue.SPEECH_EVENT_TYPE}|job:{job_id}") == 0

        async def recover(**kwargs):
            await redis.incr(enter)
            return {"transcript": "recovered", "provider": "test"}

        monkeypatch.setattr(speech_worker, "transcribe_from_path_with_fallback", recover)
        replacement = speech_worker.build_speech_stream_consumer(redis=redis)
        await replacement.claim_stale()
        assert int(await redis.get(enter) or 0) == 2
        assert await _pending(redis, speech, "speech-workers") == 0
        saved = json.loads(await redis.hget(f"speech_job:{job_id}", "result"))
        assert saved["transcript"] == "recovered"
        assert await redis.hget(f"speech_job:{job_id}", "status") == "completed"
    finally:
        if proc.poll() is None:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/F"], check=False, capture_output=True)
            proc.wait(timeout=8)


async def test_crash_after_transcript_before_vector_publish_is_recovered(redis, monkeypatch):
    speech, vector = _names(redis)
    _use_streams(monkeypatch, redis, speech, vector)
    calls = []

    async def transcribe(**kwargs):
        calls.append(kwargs["job_id"])
        return {"transcript": "durable", "provider": "test"}

    async def push_crash(*args, **kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr(speech_worker, "transcribe_from_path_with_fallback", transcribe)
    monkeypatch.setattr(speech_worker, "push_completed_speech_job", push_crash)
    job = _job("job-handoff-a", 1)
    await redis.hset(f"speech_job:{job['job_id']}", mapping={key: str(value) for key, value in job.items()})
    await redis_queue.publish_speech_job(job, redis=redis)
    consumer = speech_worker.build_speech_stream_consumer(redis=redis)
    await consumer.ensure_group()
    await consumer.dispatch_messages(await consumer.read_batch(block_ms=50))
    assert calls == ["job-handoff-a"]
    assert await redis.hget("speech_job:job-handoff-a", "status") == "completed"
    assert await redis.xlen(vector) == 0
    assert await _pending(redis, speech, "speech-workers") == 1
    monkeypatch.setattr(speech_worker, "push_completed_speech_job", redis_queue.publish_completed_speech_job)
    await _rehandle(
        consumer,
        redis,
        speech,
        "speech-workers",
        f"buddy:job-lease:{redis_queue.SPEECH_EVENT_TYPE}|job:job-handoff-a",
    )
    assert calls == ["job-handoff-a"]
    assert await _pending(redis, speech, "speech-workers") == 0
    assert len(await _read_events(redis, vector)) == 1
    embeds = await _embed_once(redis, vector, "job-handoff-a")
    assert embeds == ["job-handoff-a"]


async def test_crash_after_vector_publish_before_speech_ack_embeds_once(redis, monkeypatch):
    speech, vector = _names(redis)
    _use_streams(monkeypatch, redis, speech, vector)
    calls = []

    async def transcribe(**kwargs):
        calls.append(kwargs["job_id"])
        return {"transcript": "durable", "provider": "test"}

    def crash_mark():
        raise asyncio.CancelledError()

    monkeypatch.setattr(speech_worker, "transcribe_from_path_with_fallback", transcribe)
    monkeypatch.setattr(speech_worker, "mark_current_job_completed", crash_mark)
    job = _job("job-handoff-b", 1)
    await redis.hset(f"speech_job:{job['job_id']}", mapping={key: str(value) for key, value in job.items()})
    await redis_queue.publish_speech_job(job, redis=redis)
    consumer = speech_worker.build_speech_stream_consumer(redis=redis)
    await consumer.ensure_group()
    await consumer.dispatch_messages(await consumer.read_batch(block_ms=50))
    assert calls == ["job-handoff-b"]
    assert await redis.xlen(vector) == 1
    assert await _pending(redis, speech, "speech-workers") == 1
    monkeypatch.setattr(speech_worker, "mark_current_job_completed", lambda: None)
    await _rehandle(
        consumer,
        redis,
        speech,
        "speech-workers",
        f"buddy:job-lease:{redis_queue.SPEECH_EVENT_TYPE}|job:job-handoff-b",
    )
    assert calls == ["job-handoff-b"]
    assert await redis.xlen(vector) == 2
    assert await _pending(redis, speech, "speech-workers") == 0
    embeds = await _embed_once(redis, vector, "job-handoff-b")
    assert embeds == ["job-handoff-b"]


async def test_crash_after_embedding_before_vector_ack_does_not_embed_again(redis, monkeypatch):
    speech, vector = _names(redis)
    _use_streams(monkeypatch, redis, speech, vector)
    await redis.hset(
        "speech_job:job-handoff-c",
        mapping={
            "status": "completed",
            "user_id": "user-1",
            "space_id": "space-1",
            "result": json.dumps({"transcript": "embedded", "provider": "test"}),
        },
    )
    await redis_queue.publish_completed_speech_job("job-handoff-c", redis=redis, job={"user_id": "user-1", "space_id": "space-1"})
    embeds = []

    async def embed(**kwargs):
        embeds.append(kwargs["job_id"])

    def crash_mark():
        raise asyncio.CancelledError()

    monkeypatch.setattr(vector_worker, "store_transcript_in_vector_db", embed)
    monkeypatch.setattr(vector_worker, "mark_current_job_completed", crash_mark)
    consumer = vector_worker.build_vector_stream_consumer(redis=redis)
    await consumer.ensure_group()
    await consumer.dispatch_messages(await consumer.read_batch(block_ms=50))
    assert embeds == ["job-handoff-c"]
    assert await redis.exists("speech_job:job-handoff-c") == 0
    assert await redis.get(f"{redis_queue.VECTOR_DONE_PREFIX}job-handoff-c") == "1"
    assert await _pending(redis, vector, "vector-workers") == 1
    monkeypatch.setattr(vector_worker, "mark_current_job_completed", lambda: None)
    await _rehandle(
        consumer,
        redis,
        vector,
        "vector-workers",
        f"buddy:job-lease:{redis_queue.VECTOR_EVENT_TYPE}|job:job-handoff-c",
    )
    assert embeds == ["job-handoff-c"]
    assert await _pending(redis, vector, "vector-workers") == 0


async def _embed_once(redis: Redis, vector: str, job_id: str) -> list[str]:
    embeds = []

    async def embed(**kwargs):
        embeds.append(kwargs["job_id"])

    class Repository:
        async def complete_transcript_chunk(self, **kwargs):
            return None

        async def get_conversation(self, conversation_id):
            return None

    previous_embed = vector_worker.store_transcript_in_vector_db
    previous_database = vector_worker.get_database
    previous_repository = vector_worker.ConversationRepository
    vector_worker.store_transcript_in_vector_db = embed
    vector_worker.get_database = lambda: None
    vector_worker.ConversationRepository = lambda database: Repository()
    try:
        consumer = vector_worker.build_vector_stream_consumer(redis=redis)
        await consumer.ensure_group()
        for _ in range(4):
            if not await _pending(redis, vector, "vector-workers") and not await _group_lag(redis, vector, "vector-workers"):
                break
            await consumer.dispatch_messages(await consumer.read_batch(block_ms=50))
        return embeds
    finally:
        vector_worker.store_transcript_in_vector_db = previous_embed
        vector_worker.get_database = previous_database
        vector_worker.ConversationRepository = previous_repository


async def _group_lag(redis: Redis, stream: str, group: str) -> int:
    for row in await redis.xinfo_groups(stream):
        if row.get("name") == group:
            return int(row.get("lag") or 0)
    return 0


async def test_paused_vector_publish_keeps_the_transcript_and_list_rollback_queues_it(redis, monkeypatch):
    speech, vector = _names(redis)
    vector_list = f"phase2b:vector-list:{uuid.uuid4().hex[:8]}"
    _use_streams(monkeypatch, redis, speech, vector)
    monkeypatch.setattr(redis_queue, "COMPLETED_SPEECH_QUEUE", vector_list)
    job = _job("job-list-handoff", 1)
    await redis.hset(f"speech_job:{job['job_id']}", mapping={key: str(value) for key, value in job.items()})
    await redis_queue.save_job_result(job["job_id"], {"transcript": "kept", "provider": "test"})
    await redis.set(redis_queue.EXPORT_LOCK_KEY, "rollback")
    calls = []

    async def transcribe(**kwargs):
        calls.append(kwargs["job_id"])
        return {"transcript": "again", "provider": "test"}

    monkeypatch.setattr(speech_worker, "transcribe_from_path_with_fallback", transcribe)
    with pytest.raises(LeavePending):
        await speech_worker.process_speech_job(job)
    assert calls == []
    assert await redis.hget(f"speech_job:{job['job_id']}", "status") == "completed"
    await redis.delete(redis_queue.EXPORT_LOCK_KEY)
    monkeypatch.setattr(settings, "SPEECH_VECTOR_QUEUE_TRANSPORT", "list")
    await speech_worker.process_speech_job(job)
    assert calls == []
    assert await redis.lrange(vector_list, 0, -1) == [job["job_id"]]


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
    for _ in range(40):
        try:
            probe = Redis(host=HOST, port=PORT, socket_connect_timeout=0.2, socket_timeout=0.2)
            await probe.ping()
            await probe.aclose()
            return
        except Exception:
            await asyncio.sleep(0.15)
    raise RuntimeError("disposable Redis did not return on 127.0.0.1:6399")
