"""Speech and vector transport behavior that does not need a Redis server."""

from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path

import pytest

from apps.api_gateway.config.setting import settings
from apps.api_gateway.workers import speech_worker, vector_worker
from services.queue.job_lease import Lease, logical_job_key, push_lease, reset_lease
from services.queue.redis_queue import (
    SPEECH_EVENT_TYPE,
    VECTOR_EVENT_TYPE,
    publish_speech_job,
    speech_job_event,
    speech_vector_transport,
    vector_job_event,
)

ROOT = Path(__file__).resolve().parents[1]


class _FakeRedis:
    def __init__(self):
        self.lists: list[tuple[str, str]] = []
        self.streams: list[tuple[str, dict]] = []
        self.hashes: dict[str, dict] = {}

    async def lpush(self, key, value):
        self.lists.append((key, value))
        return 1

    async def eval(self, script, numkeys, *keys_and_args):
        key = keys_and_args[1]
        mode = keys_and_args[2]
        payload = keys_and_args[3]
        if mode == "list":
            await self.lpush(key, payload)
            return "list"
        return await self.xadd(key, {"event": payload})

    async def xadd(self, stream, fields, maxlen=None, approximate=True):
        self.streams.append((stream, fields))
        return "1-0"

    async def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    async def hset(self, key, mapping):
        self.hashes.setdefault(key, {}).update(mapping)
        return 1

    async def expire(self, key, ttl):
        return True


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


def test_default_transport_stays_on_the_legacy_list():
    assert speech_vector_transport() == "list"


def test_list_publish_does_not_write_a_stream(monkeypatch):
    monkeypatch.setattr(settings, "SPEECH_VECTOR_QUEUE_TRANSPORT", "list")
    fake = _FakeRedis()
    target = asyncio.run(publish_speech_job(_job("job-1", 1), redis=fake))
    assert target == "speech_transcribe_queue"
    assert fake.streams == []
    assert json.loads(fake.lists[0][1])["job_id"] == "job-1"


def test_stream_publish_keeps_a_stable_logical_job_per_chunk(monkeypatch):
    monkeypatch.setattr(settings, "SPEECH_VECTOR_QUEUE_TRANSPORT", "stream")
    monkeypatch.setattr(settings, "REDIS_SPEECH_STREAM", "buddy:speech:jobs")
    fake = _FakeRedis()
    first = speech_job_event(_job("job-1", 1))
    second = speech_job_event(_job("job-2", 2))
    vector = vector_job_event("job-1", _job("job-1", 1))
    assert logical_job_key(first) == f"{SPEECH_EVENT_TYPE}|job:job-1"
    assert logical_job_key(second) == f"{SPEECH_EVENT_TYPE}|job:job-2"
    assert logical_job_key(vector) == f"{VECTOR_EVENT_TYPE}|job:job-1"
    assert logical_job_key(first) != logical_job_key(vector)

    target = asyncio.run(publish_speech_job(_job("job-1", 1), redis=fake))
    assert target == "buddy:speech:jobs"
    assert fake.lists == []
    event = json.loads(fake.streams[0][1]["event"])
    assert event["eventType"] == SPEECH_EVENT_TYPE
    assert event["payload"]["jobId"] == "job-1"
    assert event["payload"]["sequence_number"] == 1


def test_stream_consumers_do_not_raise_concurrency(monkeypatch):
    monkeypatch.setattr(settings, "WORKER_CONCURRENCY", 4)
    speech = speech_worker.build_speech_stream_consumer(redis=object())
    vector = vector_worker.build_vector_stream_consumer(redis=object())
    assert speech.concurrency == 1
    assert vector.concurrency == 1
    assert vector.max_retries == vector_worker.VECTOR_MAX_ATTEMPTS - 1


def test_completed_speech_does_not_finalize_before_the_transcript_is_durable(monkeypatch):
    calls = []

    async def get_result(job_id):
        return {"status": "queued", "job_id": job_id}

    class Repository:
        async def complete_transcript_chunk(self, **kwargs):
            calls.append(kwargs)

        async def get_conversation(self, conversation_id):
            calls.append(conversation_id)
            return None

    async def not_settled(job_id):
        return False

    monkeypatch.setattr(vector_worker, "get_job_result", get_result)
    monkeypatch.setattr(vector_worker, "vector_stage_settled", not_settled)
    monkeypatch.setattr(vector_worker, "get_database", lambda: None)
    monkeypatch.setattr(vector_worker, "ConversationRepository", lambda database: Repository())
    lease = Lease(
        redis=None,
        logical_key="vector.index.requested|job:job-1",
        owner="vector-test",
        token="1",
        message_id="1-0",
        event_id="event-1",
        event_type=VECTOR_EVENT_TYPE,
        retry_count=0,
    )
    token = push_lease(lease)
    try:
        with pytest.raises(RuntimeError, match="not durable"):
            asyncio.run(vector_worker.process_completed_speech_job("job-1"))
    finally:
        reset_lease(token)
    assert calls == []


def test_chunk_sequences_stay_on_their_own_jobs(monkeypatch):
    stored = []

    async def get_result(job_id):
        sequence = 1 if job_id == "job-1" else 2
        return {
            "status": "completed",
            "user_id": "user-1",
            "space_id": "space-1",
            "conversation_id": "conv-1",
            "sequence_number": sequence,
            "result": {"transcript": f"line {sequence}", "provider": "test"},
        }

    class Repository:
        async def complete_transcript_chunk(self, **kwargs):
            stored.append((kwargs["sequence_number"], kwargs["raw_text"]))

        async def get_conversation(self, conversation_id):
            return None

    async def store(**kwargs):
        stored.append(("vector", kwargs["job_id"]))

    async def delete_job(job_id):
        return None

    async def mark_stored(job_id):
        return None

    async def not_settled(job_id):
        return False

    monkeypatch.setattr(vector_worker, "get_job_result", get_result)
    monkeypatch.setattr(vector_worker, "vector_stage_settled", not_settled)
    monkeypatch.setattr(vector_worker, "get_database", lambda: None)
    monkeypatch.setattr(vector_worker, "ConversationRepository", lambda database: Repository())
    monkeypatch.setattr(vector_worker, "store_transcript_in_vector_db", store)
    monkeypatch.setattr(vector_worker, "delete_speech_job", delete_job)
    monkeypatch.setattr(vector_worker, "mark_vector_stored", mark_stored)

    asyncio.run(vector_worker.process_completed_speech_job("job-2"))
    asyncio.run(vector_worker.process_completed_speech_job("job-1"))
    assert stored == [(2, "line 2"), ("vector", "job-2"), (1, "line 1"), ("vector", "job-1")]


def test_worker_startup_does_not_drain_legacy_lists():
    source = (ROOT / "apps" / "api_gateway" / "workers" / "main.py").read_text(encoding="utf-8")
    assert "legacy_list_migration" not in source
    assert "migrate_speech_and_vector_backlog" not in source


def test_no_producer_left_on_the_legacy_lists():
    """List writes to the migrated queues may exist only in the list-mode branch."""
    allowed = {
        ROOT / "services" / "queue" / "redis_queue.py",
        ROOT / "services" / "queue" / "legacy_list_migration.py",
    }
    offenders: list[str] = []
    for path in ROOT.rglob("*.py"):
        if any(part in {"venv", ".venv", "node_modules", "__pycache__"} for part in path.parts):
            continue
        if "tests" in path.parts or path in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        if "lpush" not in text and "rpush" not in text:
            continue
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else ""
            if name not in {"lpush", "rpush"}:
                continue
            rendered = ast.get_source_segment(text, node) or ""
            if "SPEECH_QUEUE" in rendered or "COMPLETED_SPEECH_QUEUE" in rendered:
                offenders.append(f"{path}:{node.lineno}")
            if "speech_transcribe_queue" in rendered or "completed_speech_queue" in rendered:
                offenders.append(f"{path}:{node.lineno}")
    assert offenders == []

    queue = (ROOT / "services" / "queue" / "redis_queue.py").read_text(encoding="utf-8")
    speech_push = queue.index('await guarded_queue_write(client, SPEECH_QUEUE, "list"')
    vector_push = queue.index('await guarded_queue_write(client, COMPLETED_SPEECH_QUEUE, "list"')
    assert "await client.lpush(SPEECH_QUEUE" not in queue
    assert "await client.lpush(COMPLETED_SPEECH_QUEUE" not in queue
    assert queue.rfind('if speech_vector_transport() == "stream"', 0, speech_push) != -1
    assert queue.rfind('if speech_vector_transport() == "stream"', 0, vector_push) != -1
