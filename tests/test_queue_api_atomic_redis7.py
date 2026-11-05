"""Queue API atomic dedupe against disposable Redis 7 on 127.0.0.1:6399."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

import httpx
import pytest
from redis.asyncio import Redis
from redis.exceptions import ResponseError

from apps.api_gateway.config.setting import settings
from apps.api_gateway import queue_api
from services.queue.atomic_publish import (
    _PUBLISH_LUA,
    QueuePayloadConflict,
    atomic_publish,
    dedupe_key,
    dedupe_ttl_seconds,
    submission_fingerprint,
)
from services.queue.streams import EventEnvelope, RedisStreamProducer

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[1]
REDIS_HOST = "127.0.0.1"
REDIS_PORT = 6399


async def _redis() -> Redis:
    client = Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        decode_responses=True,
        socket_connect_timeout=2,
        socket_timeout=2,
    )
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


def _event(event_id: str, *, event_type: str = "stt.requested", payload: dict | None = None, user_id: str = "user-1") -> EventEnvelope:
    return EventEnvelope(
        eventId=event_id,
        eventType=event_type,
        correlationId="conv-1",
        userId=user_id,
        spaceId="space-1",
        conversationId="conv-1",
        payload=payload or {"objectKey": f"buddy/audio/{event_id}.webm"},
    )


@pytest.fixture
async def redis_api(monkeypatch):
    client = await _redis()
    suffix = uuid.uuid4().hex[:8]
    names = {
        "REDIS_STT_STREAM": f"phase2c:stt:{suffix}",
        "REDIS_AUDIO_STREAM": f"phase2c:audio:{suffix}",
        "REDIS_RETRY_STREAM": f"phase2c:retry:{suffix}",
        "REDIS_SPEECH_STREAM": f"phase2c:speech-stream:{suffix}",
        "REDIS_DAILY_BRIEFING_STREAM": f"phase2c:briefing:{suffix}",
    }
    for key, value in names.items():
        monkeypatch.setattr(settings, key, value)
    speech_list = f"phase2c:speech-list:{suffix}"
    lock = f"phase2c:export-lock:{suffix}"
    monkeypatch.setattr(queue_api, "SPEECH_QUEUE", speech_list)
    monkeypatch.setattr("services.queue.atomic_publish.EXPORT_LOCK_KEY", lock)
    monkeypatch.setattr(queue_api, "redis_client", client)
    monkeypatch.setattr(settings, "QUEUE_API_SERVICE_TOKEN", "phase2c-token")
    monkeypatch.setattr(settings, "QUEUE_API_HMAC_SECRET", "")
    created: list[str] = []
    yield client, names, speech_list, lock, created
    keys = [key for key in [*created, *names.values(), speech_list, lock] if key]

    async def _delete(connection):
        if keys:
            await connection.delete(*keys)

    try:
        await _delete(client)
        await client.aclose()
    except Exception:
        await client.aclose()
        fresh = await _redis()
        try:
            await _delete(fresh)
        finally:
            await fresh.aclose()


def _headers() -> dict[str, str]:
    return {"authorization": "Bearer phase2c-token"}


async def _post(payload: dict, headers: dict | None = None):
    transport = httpx.ASGITransport(app=queue_api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://queue") as client:
        return await client.post("/internal/events", json=payload, headers=headers if headers is not None else _headers())


def _body(event_id: str, *, event_type: str = "stt.requested", payload: dict | None = None, user_id: str = "user-1") -> dict:
    return {
        "eventId": event_id,
        "eventType": event_type,
        "correlationId": "conv-1",
        "userId": user_id,
        "spaceId": "space-1",
        "conversationId": "conv-1",
        "payload": payload or {"objectKey": f"buddy/audio/{event_id}.webm"},
    }


async def test_one_request_publishes_one_event_and_records_dedupe(redis_api):
    client, names, _speech, _lock, created = redis_api
    event_id = f"phase2c-{uuid.uuid4().hex}"
    created.append(dedupe_key(event_id))
    started = time.perf_counter()
    response = await _post(_body(event_id))
    elapsed_ms = (time.perf_counter() - started) * 1000
    assert response.status_code == 202
    body = response.json()
    assert body["accepted"] is True
    assert body["duplicate"] is False
    assert body["stream"] == names["REDIS_STT_STREAM"]
    assert await client.xlen(names["REDIS_STT_STREAM"]) == 1
    record = await client.hgetall(dedupe_key(event_id))
    assert record["state"] == "published"
    assert record["redisMessageId"]
    assert record["payloadHash"]
    ttl = await client.ttl(dedupe_key(event_id))
    assert 80000 <= ttl <= dedupe_ttl_seconds()
    try:
        used = await client.memory_usage(dedupe_key(event_id))
    except Exception:
        used = None
    print(f"phase2c_publish_ms={elapsed_ms:.2f} phase2c_dedupe_bytes={used}")


async def test_sequential_duplicate_returns_the_same_publication(redis_api):
    client, names, _speech, _lock, created = redis_api
    event_id = f"phase2c-{uuid.uuid4().hex}"
    created.append(dedupe_key(event_id))
    payload = _body(event_id)
    first = await _post(payload)
    second = await _post(payload)
    assert first.status_code == 202 and second.status_code == 202
    assert first.json()["duplicate"] is False
    assert second.json()["duplicate"] is True
    assert second.json()["eventId"] == event_id
    assert await client.xlen(names["REDIS_STT_STREAM"]) == 1


async def test_concurrent_publishers_in_one_process_publish_once(redis_api):
    client, names, _speech, _lock, created = redis_api
    other = Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    event_id = f"phase2c-{uuid.uuid4().hex}"
    created.append(dedupe_key(event_id))
    event = _event(event_id)
    payload = event.model_dump_json()
    fingerprint = submission_fingerprint(event)

    async def publish(connection):
        return await atomic_publish(
            connection,
            event_id=event_id,
            queue=names["REDIS_STT_STREAM"],
            mode="stream",
            payload=payload,
            fingerprint=fingerprint,
            correlation_id="conv-1",
            event_type=event.eventType,
        )

    try:
        results = await asyncio.gather(publish(client), publish(other))
    finally:
        await other.aclose()
    assert sorted(item.outcome for item in results) == ["duplicate", "published"]
    assert await client.xlen(names["REDIS_STT_STREAM"]) == 1


async def test_two_processes_publish_one_event(redis_api):
    client, names, _speech, _lock, created = redis_api
    event_id = f"phase2c-{uuid.uuid4().hex}"
    created.append(dedupe_key(event_id))
    event = _event(event_id)
    script = (
        "import asyncio,os,sys\n"
        "sys.path.insert(0, os.environ['PHASE2C_ROOT'])\n"
        "async def main():\n"
        "    from redis.asyncio import Redis\n"
        "    from services.queue.atomic_publish import atomic_publish\n"
        "    redis = Redis(host='127.0.0.1', port=6399, decode_responses=True)\n"
        "    result = await atomic_publish(redis, event_id=os.environ['PHASE2C_EVENT'], queue=os.environ['PHASE2C_QUEUE'], mode='stream', payload=os.environ['PHASE2C_PAYLOAD'], fingerprint=os.environ['PHASE2C_FP'], correlation_id='conv-1', event_type='stt.requested')\n"
        "    print(result.outcome)\n"
        "    await redis.aclose()\n"
        "asyncio.run(main())\n"
    )
    env = os.environ.copy()
    env.update(
        {
            "PHASE2C_ROOT": str(ROOT),
            "PHASE2C_EVENT": event_id,
            "PHASE2C_QUEUE": names["REDIS_STT_STREAM"],
            "PHASE2C_PAYLOAD": event.model_dump_json(),
            "PHASE2C_FP": submission_fingerprint(event),
            "REDIS_URL": "redis://127.0.0.1:6399/0",
        }
    )
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", script],
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    outputs = []
    for proc in procs:
        stdout, stderr = proc.communicate(timeout=30)
        if proc.returncode != 0:
            raise AssertionError(stderr[-2000:])
        outputs.append(stdout.strip().splitlines()[-1])
    assert sorted(outputs) == ["duplicate", "published"]
    assert await client.xlen(names["REDIS_STT_STREAM"]) == 1


async def test_different_event_ids_both_publish(redis_api):
    client, names, _speech, _lock, created = redis_api
    first_id = f"phase2c-{uuid.uuid4().hex}"
    second_id = f"phase2c-{uuid.uuid4().hex}"
    created.extend([dedupe_key(first_id), dedupe_key(second_id)])
    assert (await _post(_body(first_id))).status_code == 202
    assert (await _post(_body(second_id))).status_code == 202
    assert await client.xlen(names["REDIS_STT_STREAM"]) == 2


async def test_same_event_id_with_a_different_payload_conflicts(redis_api):
    client, names, _speech, _lock, created = redis_api
    event_id = f"phase2c-{uuid.uuid4().hex}"
    created.append(dedupe_key(event_id))
    first = await _post(_body(event_id, payload={"objectKey": "buddy/audio/one.webm"}))
    second = await _post(_body(event_id, user_id="user-2", payload={"objectKey": "buddy/audio/two.webm"}))
    assert first.status_code == 202
    assert second.status_code == 409
    assert second.json() == {"detail": "event payload conflict"}
    assert "user-1" not in second.text
    assert await client.xlen(names["REDIS_STT_STREAM"]) == 1


async def test_old_dedupe_record_without_a_hash_stays_a_duplicate(redis_api):
    client, names, _speech, _lock, created = redis_api
    event_id = f"phase2c-{uuid.uuid4().hex}"
    created.append(dedupe_key(event_id))
    await client.hset(dedupe_key(event_id), mapping={"state": "published", "stream": names["REDIS_STT_STREAM"], "redisMessageId": "1-0"})
    await client.expire(dedupe_key(event_id), 3600)
    response = await _post(_body(event_id))
    assert response.status_code == 202
    assert response.json()["duplicate"] is True
    assert await client.xlen(names["REDIS_STT_STREAM"]) == 0


async def test_script_rejects_bad_input_before_writing_and_does_not_roll_back(redis_api):
    client, _names, _speech, lock, created = redis_api
    event_id = f"phase2c-{uuid.uuid4().hex}"
    queue = f"phase2c:bad:{uuid.uuid4().hex[:8]}"
    created.extend([dedupe_key(event_id), queue, f"{queue}:partial"])
    with pytest.raises(ResponseError):
        await client.eval(
            _PUBLISH_LUA,
            3,
            dedupe_key(event_id),
            queue,
            lock,
            "nope",
            "{}",
            "hash",
            "86400",
            "conv-1",
            "0",
        )
    assert await client.exists(dedupe_key(event_id)) == 0
    assert await client.xlen(queue) == 0
    with pytest.raises(ResponseError):
        await client.eval(
            "redis.call('LPUSH', KEYS[1], 'orphan')\nreturn redis.error_reply('after-write')\n",
            1,
            f"{queue}:partial",
        )
    assert await client.llen(f"{queue}:partial") == 1


async def test_lost_response_retries_the_existing_publication(redis_api):
    client, names, _speech, _lock, created = redis_api
    event_id = f"phase2c-{uuid.uuid4().hex}"
    created.append(dedupe_key(event_id))
    payload = _body(event_id)
    first = await _post(payload)
    assert first.status_code == 202
    retried = await _post(payload)
    assert retried.status_code == 202
    assert retried.json()["duplicate"] is True
    assert await client.xlen(names["REDIS_STT_STREAM"]) == 1


async def test_closed_redis_port_is_not_a_successful_publish(redis_api):
    _client, names, _speech, _lock, _created = redis_api
    dead = Redis(host=REDIS_HOST, port=6398, decode_responses=True, socket_connect_timeout=0.3, socket_timeout=0.3)
    event = _event(f"phase2c-{uuid.uuid4().hex}")
    try:
        with pytest.raises(Exception):
            await atomic_publish(
                dead,
                event_id=event.eventId,
                queue=names["REDIS_STT_STREAM"],
                mode="stream",
                payload=event.model_dump_json(),
                fingerprint=submission_fingerprint(event),
                correlation_id="conv-1",
                event_type=event.eventType,
            )
    finally:
        await dead.aclose()


async def test_retry_event_with_a_new_id_is_not_suppressed(redis_api):
    client, names, _speech, _lock, created = redis_api
    original_id = f"phase2c-{uuid.uuid4().hex}"
    retry_id = f"phase2c-{uuid.uuid4().hex}"
    created.extend([dedupe_key(original_id), dedupe_key(retry_id)])
    original = await _post(_body(original_id))
    assert original.status_code == 202
    retry = await _post(
        _body(
            retry_id,
            event_type="job.retry.requested",
            payload={"jobId": original_id, "targetStream": names["REDIS_STT_STREAM"]},
        )
    )
    assert retry.status_code == 202
    assert retry.json()["duplicate"] is False
    assert retry.json()["stream"] == names["REDIS_RETRY_STREAM"]
    assert await client.xlen(names["REDIS_RETRY_STREAM"]) == 1
    replay = await _post(_body(original_id))
    assert replay.json()["duplicate"] is True
    assert await client.xlen(names["REDIS_STT_STREAM"]) == 1
    producer = RedisStreamProducer(client, force_direct=True)
    relayed = _event(f"phase2c-{uuid.uuid4().hex}")
    await producer.publish(names["REDIS_STT_STREAM"], relayed)
    assert await client.xlen(names["REDIS_STT_STREAM"]) == 2


async def test_speech_list_and_stream_modes_do_not_double_write(redis_api, monkeypatch):
    client, names, speech_list, lock, created = redis_api
    list_id = f"phase2c-{uuid.uuid4().hex}"
    created.append(dedupe_key(list_id))
    monkeypatch.setattr(settings, "SPEECH_VECTOR_QUEUE_TRANSPORT", "list")
    listed = await _post(
        _body(
            list_id,
            event_type="speech.transcribe.requested",
            payload={"job_id": list_id, "filename": "audio.webm", "content_type": "audio/webm"},
        )
    )
    assert listed.status_code == 202
    assert listed.json()["stream"] == speech_list
    assert await client.llen(speech_list) == 1
    assert await client.xlen(names["REDIS_SPEECH_STREAM"]) == 0
    assert await client.hget(f"speech_job:{list_id}", "status") == "queued"
    created.append(f"speech_job:{list_id}")

    stream_id = f"phase2c-{uuid.uuid4().hex}"
    created.append(dedupe_key(stream_id))
    monkeypatch.setattr(settings, "SPEECH_VECTOR_QUEUE_TRANSPORT", "stream")
    streamed = await _post(
        _body(
            stream_id,
            event_type="speech.transcribe.requested",
            payload={"job_id": stream_id, "filename": "audio.webm", "content_type": "audio/webm"},
        )
    )
    assert streamed.status_code == 202
    assert streamed.json()["stream"] == names["REDIS_SPEECH_STREAM"]
    assert await client.xlen(names["REDIS_SPEECH_STREAM"]) == 1
    assert await client.llen(speech_list) == 1
    created.append(f"speech_job:{stream_id}")

    await client.set(lock, "rollback")
    locked_id = f"phase2c-{uuid.uuid4().hex}"
    created.append(dedupe_key(locked_id))
    locked = await _post(
        _body(
            locked_id,
            event_type="speech.transcribe.requested",
            payload={"job_id": locked_id, "filename": "audio.webm", "content_type": "audio/webm"},
        )
    )
    assert locked.status_code == 503
    assert await client.exists(dedupe_key(locked_id)) == 0
    assert await client.xlen(names["REDIS_SPEECH_STREAM"]) == 1
    duplicate = await _post(
        _body(
            stream_id,
            event_type="speech.transcribe.requested",
            payload={"job_id": stream_id, "filename": "audio.webm", "content_type": "audio/webm"},
        )
    )
    assert duplicate.status_code == 202
    assert duplicate.json()["duplicate"] is True
    monkeypatch.setattr(settings, "SPEECH_VECTOR_QUEUE_TRANSPORT", "list")


async def test_end_user_bearer_cannot_publish(redis_api):
    client, names, _speech, _lock, created = redis_api
    event_id = f"phase2c-{uuid.uuid4().hex}"
    created.append(dedupe_key(event_id))
    rejected = await _post(_body(event_id), headers={"authorization": "Bearer end-user-token"})
    assert rejected.status_code == 401
    missing = await _post(_body(event_id), headers={})
    assert missing.status_code == 401
    assert await client.xlen(names["REDIS_STT_STREAM"]) == 0
    assert await client.exists(dedupe_key(event_id)) == 0


async def test_unrelated_event_types_publish_independently(redis_api):
    client, names, _speech, _lock, created = redis_api
    audio_id = f"phase2c-{uuid.uuid4().hex}"
    briefing_id = f"phase2c-{uuid.uuid4().hex}"
    created.extend([dedupe_key(audio_id), dedupe_key(briefing_id)])
    audio = _body(audio_id, event_type="audio.ingested", payload={"conversationId": "conv-1", "sequenceNumber": 1})
    audio["targetStream"] = names["REDIS_AUDIO_STREAM"]
    assert (await _post(audio)).status_code == 202
    daily = _body(briefing_id, event_type="daily.briefing.requested", payload={"dateKey": "2026-11-05"})
    assert (await _post(daily)).status_code == 202
    assert await client.xlen(names["REDIS_AUDIO_STREAM"]) == 1
    assert await client.xlen(names["REDIS_DAILY_BRIEFING_STREAM"]) == 1


async def test_direct_publish_conflict_does_not_leave_a_marker(redis_api):
    client, names, _speech, _lock, created = redis_api
    event_id = f"phase2c-{uuid.uuid4().hex}"
    created.append(dedupe_key(event_id))
    event = _event(event_id)
    first = await atomic_publish(
        client,
        event_id=event_id,
        queue=names["REDIS_STT_STREAM"],
        mode="stream",
        payload=event.model_dump_json(),
        fingerprint=submission_fingerprint(event),
        correlation_id="conv-1",
        event_type=event.eventType,
    )
    assert first.outcome == "published"
    other = _event(event_id, payload={"objectKey": "buddy/audio/other.webm"})
    with pytest.raises(QueuePayloadConflict):
        await atomic_publish(
            client,
            event_id=event_id,
            queue=names["REDIS_STT_STREAM"],
            mode="stream",
            payload=other.model_dump_json(),
            fingerprint=submission_fingerprint(other),
            correlation_id="conv-1",
            event_type=other.eventType,
        )
    assert await client.xlen(names["REDIS_STT_STREAM"]) == 1
    assert (await client.hget(dedupe_key(event_id), "payloadHash")) == submission_fingerprint(event)


def test_timestamps_are_not_part_of_the_submission_fingerprint():
    first = _event("same")
    second = first.model_copy(update={"attempt": 4})
    assert submission_fingerprint(first) == submission_fingerprint(second)


def _listener_command(pid: str) -> str:
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", f"(Get-CimInstance Win32_Process -Filter \"ProcessId={pid}\").CommandLine"],
        check=False,
        capture_output=True,
        text=True,
    )
    return (result.stdout or "").strip()


def _listener_pid() -> str | None:
    listener = subprocess.run(["netstat", "-ano"], check=False, capture_output=True, text=True)
    for line in listener.stdout.splitlines():
        if f"127.0.0.1:{REDIS_PORT}" in line and "LISTENING" in line:
            return line.split()[-1]
    return None


async def test_disposable_redis_restart_does_not_report_false_success(redis_api):
    """Restart only the disposable Redis 7.4 listener. Never the application worker."""
    client, names, _speech, _lock, created = redis_api
    watched = (
        settings.REDIS_URL,
        settings.REMINDER_REDIS_URL,
        settings.MINDMAP_REDIS_URL,
        settings.GOAL_REDIS_URL,
    )
    if any(urlparse(str(value or "")).port == REDIS_PORT for value in watched):
        pytest.skip("an application Redis URL points at the disposable port; not restarting it")
    pid = _listener_pid()
    command = _listener_command(pid or "")
    if "Redis-7.4.9" not in command or "redis-server" not in command.lower():
        pytest.skip(f"6399 is not the disposable Redis 7.4.9 server ({command[:120]})")
    event_id = f"phase2c-{uuid.uuid4().hex}"
    created.append(dedupe_key(event_id))
    event = _event(event_id)
    queue = names["REDIS_STT_STREAM"]
    before = await client.xlen(queue)

    async def publish():
        return await atomic_publish(
            client,
            event_id=event_id,
            queue=queue,
            mode="stream",
            payload=event.model_dump_json(),
            fingerprint=submission_fingerprint(event),
            correlation_id="conv-1",
            event_type=event.eventType,
        )

    inflight = asyncio.create_task(publish())
    await asyncio.sleep(0.01)
    subprocess.run(["taskkill", "/PID", pid, "/F"], check=False, capture_output=True)
    outcome: object
    try:
        outcome = await inflight
    except Exception as error:
        outcome = error
    if isinstance(outcome, Exception):
        assert not isinstance(outcome, AssertionError)
    exe = command.split()[0].strip('"')
    conf = command.split()[1] if len(command.split()) > 1 else ""
    restarted = subprocess.Popen(
        [exe, conf] if conf else [exe],
        cwd=str(Path(exe).parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        recovered = None
        for _ in range(40):
            try:
                recovered = Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True, socket_connect_timeout=1, socket_timeout=1)
                if await recovered.ping():
                    break
            except Exception:
                if recovered is not None:
                    await recovered.aclose()
                    recovered = None
            await asyncio.sleep(0.5)
        else:
            raise AssertionError("disposable Redis did not come back")
        retry = await atomic_publish(
            recovered,
            event_id=event_id,
            queue=queue,
            mode="stream",
            payload=event.model_dump_json(),
            fingerprint=submission_fingerprint(event),
            correlation_id="conv-1",
            event_type=event.eventType,
        )
        again = await atomic_publish(
            recovered,
            event_id=event_id,
            queue=queue,
            mode="stream",
            payload=event.model_dump_json(),
            fingerprint=submission_fingerprint(event),
            correlation_id="conv-1",
            event_type=event.eventType,
        )
        assert retry.outcome in {"published", "duplicate"}
        assert again.outcome == "duplicate"
        length = await recovered.xlen(queue)
        print(f"phase2c_restart_xlen={length} phase2c_restart_first={type(outcome).__name__} phase2c_before={before}")
        assert length >= 1
        assert length - before <= 2
    finally:
        if restarted.poll() is None and _listener_pid() is None:
            subprocess.run(["taskkill", "/PID", str(restarted.pid), "/F"], check=False, capture_output=True)
        if recovered is not None:
            await recovered.aclose()
