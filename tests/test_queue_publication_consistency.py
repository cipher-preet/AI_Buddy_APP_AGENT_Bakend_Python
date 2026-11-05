"""Phase 2C.1 Redis publication failure cases on disposable Redis 7."""

from __future__ import annotations

import time
import uuid

import pytest
from fastapi.testclient import TestClient
from redis.asyncio import Redis
from redis.exceptions import ResponseError

from apps.api_gateway import queue_api
from apps.api_gateway.config.setting import settings
from services.queue.atomic_publish import atomic_publish, dedupe_key
from services.queue.redis_queue import SPEECH_JOB_TTL_SECONDS
from services.queue.streams import EventEnvelope

pytestmark = pytest.mark.integration

REDIS_HOST = "127.0.0.1"
REDIS_PORT = 6399


async def _redis() -> Redis:
    client = Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True, socket_connect_timeout=2, socket_timeout=2)
    try:
        info = await client.info("server")
        persistence = await client.info("persistence")
    except Exception as error:
        await client.aclose()
        pytest.fail(f"Redis 7 is not available on {REDIS_HOST}:{REDIS_PORT}: {type(error).__name__}")
    version = str(info.get("redis_version") or "")
    if not version.startswith("7."):
        await client.aclose()
        pytest.fail(f"refusing Redis {version}")
    print(
        f"phase2c1_redis={version} phase2c1_aof={persistence.get('aof_enabled')} "
        f"phase2c1_rdb_changes={persistence.get('rdb_changes_since_last_save')}"
    )
    return client


def _suffix() -> str:
    return uuid.uuid4().hex


async def _publish(client: Redis, event_id: str, queue: str, mode: str, *, speech: dict | None = None):
    return await atomic_publish(
        client,
        event_id=event_id,
        queue=queue,
        mode=mode,
        payload='{"job_id":"phase2c1"}',
        fingerprint="hash-a",
        correlation_id="corr-1",
        event_type="speech.transcribe.requested",
        speech_job_key=f"speech_job:{speech['job_id']}" if speech else "",
        speech_job_fields=speech,
        speech_job_ttl=SPEECH_JOB_TTL_SECONDS if speech else 0,
    )


@pytest.fixture
async def redis():
    client = await _redis()
    created: list[str] = []
    yield client, created
    if created:
        await client.delete(*created)
    await client.aclose()


async def test_publish_is_atomic_and_a_duplicate_does_not_write_again(redis):
    client, created = redis
    event_id = f"phase2c1-{_suffix()}"
    queue = f"phase2c1:list:{_suffix()}"
    created.extend([dedupe_key(event_id), queue])
    started = time.perf_counter()
    first = None
    for _ in range(20):
        sample = f"phase2c1-{_suffix()}"
        sample_queue = f"phase2c1:list:{_suffix()}"
        created.extend([dedupe_key(sample), sample_queue])
        first = await _publish(client, sample, sample_queue, "list")
        assert first.outcome == "published"
    elapsed = (time.perf_counter() - started) / 20
    print(f"phase2c1_publish_mean_s={elapsed:.6f}")
    again = await _publish(client, sample, sample_queue, "list")
    assert again.outcome == "duplicate"
    assert await client.llen(sample_queue) == 1
    record = await client.hgetall(dedupe_key(sample))
    assert record["payloadHash"] == "hash-a"
    assert record["state"] == "published"


async def test_concurrent_duplicates_publish_one_message(redis):
    client, created = redis
    event_id = f"phase2c1-{_suffix()}"
    queue = f"phase2c1:list:{_suffix()}"
    created.extend([dedupe_key(event_id), queue])

    async def once():
        return await _publish(client, event_id, queue, "list")

    import asyncio

    results = await asyncio.gather(once(), once(), once())
    assert sorted(item.outcome for item in results) == ["duplicate", "duplicate", "published"]
    assert await client.llen(queue) == 1


async def test_wrong_key_type_is_rejected_before_a_queue_write(redis):
    client, created = redis
    event_id = f"phase2c1-{_suffix()}"
    queue = f"phase2c1:list:{_suffix()}"
    created.extend([dedupe_key(event_id), queue])
    await client.set(dedupe_key(event_id), "not-a-hash")
    with pytest.raises(ResponseError, match="bad-dedupe-type"):
        await _publish(client, event_id, queue, "list")
    assert await client.llen(queue) == 0
    await client.delete(dedupe_key(event_id))
    await client.set(queue, "not-a-list")
    with pytest.raises(ResponseError, match="bad-queue-type"):
        await _publish(client, event_id, queue, "list")
    assert await client.exists(dedupe_key(event_id)) == 0
    assert await client.get(queue) == "not-a-list"


async def test_redis_keeps_a_write_when_a_later_script_command_fails(redis):
    client, created = redis
    queue = f"phase2c1:partial:{_suffix()}"
    meta = f"phase2c1:partial-meta:{_suffix()}"
    created.extend([queue, meta])
    await client.set(meta, "string")
    with pytest.raises(ResponseError, match="WRONGTYPE"):
        await client.eval(
            "redis.call('LPUSH', KEYS[1], 'item'); redis.call('HSET', KEYS[2], 'a', 'b'); return 1",
            2,
            queue,
            meta,
        )
    assert await client.llen(queue) == 1
    assert await client.type(meta) == "string"


async def test_duplicate_retry_repairs_a_missing_speech_hash_without_a_second_copy(redis):
    client, created = redis
    event_id = f"phase2c1-{_suffix()}"
    queue = f"phase2c1:list:{_suffix()}"
    job_id = f"phase2c1-{_suffix()}"
    speech_key = f"speech_job:{job_id}"
    created.extend([dedupe_key(event_id), queue, speech_key])
    speech = {"job_id": job_id, "status": "queued", "filename": "audio.webm"}
    first = await _publish(client, event_id, queue, "list", speech=speech)
    assert first.outcome == "published"
    assert await client.hget(speech_key, "status") == "queued"
    assert await client.llen(queue) == 1
    payload = await client.lindex(queue, 0)
    assert "job_id" in payload
    await client.delete(speech_key)
    assert await client.exists(speech_key) == 0
    again = await _publish(client, event_id, queue, "list", speech=speech)
    assert again.outcome == "duplicate"
    assert await client.llen(queue) == 1
    assert await client.hget(speech_key, "status") == "queued"
    assert await client.hget(speech_key, "filename") == "audio.webm"


async def test_stream_publish_and_list_default_contract(redis):
    client, created = redis
    event_id = f"phase2c1-{_suffix()}"
    queue = f"phase2c1:stream:{_suffix()}"
    created.extend([dedupe_key(event_id), queue])
    published = await _publish(client, event_id, queue, "stream")
    assert published.outcome == "published"
    assert await client.xlen(queue) == 1
    duplicate = await _publish(client, event_id, queue, "stream")
    assert duplicate.outcome == "duplicate"
    assert await client.xlen(queue) == 1
    assert settings.SPEECH_VECTOR_QUEUE_TRANSPORT == "list"


async def test_a_new_event_id_is_a_new_publication(redis):
    client, created = redis
    queue = f"phase2c1:list:{_suffix()}"
    first = f"phase2c1-{_suffix()}"
    second = f"phase2c1-{_suffix()}"
    created.extend([dedupe_key(first), dedupe_key(second), queue])
    assert (await _publish(client, first, queue, "list")).outcome == "published"
    assert (await _publish(client, second, queue, "list")).outcome == "published"
    assert await client.llen(queue) == 2


def test_queue_authentication_rejects_before_publication(monkeypatch):
    monkeypatch.setattr(settings, "QUEUE_API_SERVICE_TOKEN", "test-token")
    monkeypatch.setattr(settings, "QUEUE_API_HMAC_SECRET", "")
    client = TestClient(queue_api.app)
    response = client.post(
        "/internal/events",
        json={
            "eventId": "phase2c1-auth",
            "eventType": "stt.requested",
            "correlationId": "conv-1",
            "userId": "user-1",
            "spaceId": "space-1",
            "conversationId": "conv-1",
            "payload": {},
        },
        headers={"authorization": "Bearer not-the-token"},
    )
    assert response.status_code == 401


def test_speech_and_vector_transport_default_remains_list():
    assert settings.SPEECH_VECTOR_QUEUE_TRANSPORT == "list"
    envelope = EventEnvelope(
        eventId="phase2c1-shape",
        eventType="speech.transcribe.requested",
        correlationId="job-1",
        userId="user-1",
        spaceId="space-1",
        conversationId="conv-1",
        payload={"job_id": "job-1"},
    )
    assert envelope.eventType == "speech.transcribe.requested"
