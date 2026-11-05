"""Event-loop, lease, and provider-contract checks for file-read offloading."""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from pathlib import Path

import httpx
import pytest
from redis.asyncio import Redis

from apps.api_gateway.config.setting import settings
from apps.api_gateway.workers.speech_worker import build_speech_stream_consumer
from apps.api_gateway.workers.vector_worker import build_vector_stream_consumer
from services.meeting_extension.ffmpeg_audio import WEBM_EBML_ID, write_standalone_webm
from services.meeting_extension.processor import _ensure_standalone_webm
from services.queue.job_lease import acquire_lease, release_lease, spawn_heartbeat
from services.queue.streams import EventEnvelope
from services.runtime import file_io
from services.runtime.file_io import read_file_bytes, read_file_prefix
from services.speech.providers import deepgram_provider


def _event(job_id: str) -> EventEnvelope:
    return EventEnvelope(
        eventId=job_id,
        eventType="speech.transcribe.requested",
        correlationId="conv-perf",
        userId="user-perf",
        spaceId="space-perf",
        conversationId="conv-perf",
        payload={"jobId": job_id},
    )


async def _redis() -> Redis:
    client = Redis(host="127.0.0.1", port=6399, decode_responses=True, socket_connect_timeout=2, socket_timeout=2)
    try:
        info = await client.info("server")
    except Exception as error:
        await client.aclose()
        pytest.fail(f"Redis 7 is not available on 127.0.0.1:6399: {type(error).__name__}")
    if not str(info.get("redis_version") or "").startswith("7."):
        await client.aclose()
        pytest.fail("refusing non-Redis-7")
    return client


def test_blocking_read_delays_the_loop_and_the_helper_does_not(tmp_path, monkeypatch):
    path = tmp_path / "audio.bin"
    path.write_bytes(b"audio-bytes")

    def slow_read(self):
        time.sleep(0.35)
        return b"audio-bytes"

    monkeypatch.setattr(Path, "read_bytes", slow_read)

    async def max_gap(offload: bool) -> float:
        loop = asyncio.get_running_loop()
        gaps: list[float] = []

        async def probe() -> None:
            while True:
                started = loop.time()
                await asyncio.sleep(0.02)
                gaps.append(loop.time() - started - 0.02)

        watcher = asyncio.create_task(probe())
        await asyncio.sleep(0.01)
        if offload:
            await read_file_bytes(path)
        else:
            path.read_bytes()
        await asyncio.sleep(0.05)
        watcher.cancel()
        try:
            await watcher
        except asyncio.CancelledError:
            pass
        return max(gaps) if gaps else 0.0

    blocking_gap = asyncio.run(max_gap(False))
    offloaded_gap = asyncio.run(max_gap(True))
    assert blocking_gap >= 0.25
    assert offloaded_gap < 0.15


async def test_slow_read_keeps_the_lease_and_a_rival_out(monkeypatch, tmp_path):
    redis = await _redis()
    path = tmp_path / "audio.bin"
    path.write_bytes(b"audio")
    job_id = f"phase2d-{uuid.uuid4().hex}"
    event = _event(job_id)
    monkeypatch.setattr(settings, "JOB_LEASE_TTL_SECONDS", 30)
    monkeypatch.setattr(settings, "JOB_LEASE_HEARTBEAT_SECONDS", 0.2)
    renewals = []
    original = __import__("services.queue.job_lease", fromlist=["renew_lease"]).renew_lease

    async def counting(redis_client, lease):
        renewals.append(time.perf_counter())
        return await original(redis_client, lease)

    monkeypatch.setattr("services.queue.job_lease.renew_lease", counting)

    def slow_read(self):
        time.sleep(0.75)
        return b"audio"

    monkeypatch.setattr(Path, "read_bytes", slow_read)
    decision = await acquire_lease(redis, event, owner="perf-a", message_id="1-0")
    assert decision.action == "run"
    heartbeat = spawn_heartbeat(
        redis,
        decision.lease,
        stream="phase2d:speech",
        group="speech-workers",
        consumer_name="perf-a",
        message_id="1-0",
    )
    committed: list[bytes] = []

    async def read_and_commit():
        payload = await read_file_bytes(path)
        committed.append(payload)

    reader = asyncio.create_task(read_and_commit())
    await asyncio.sleep(0.05)
    try:
        rival = await acquire_lease(redis, event, owner="perf-b", message_id="1-0")
        assert rival.action == "busy"
        await reader
    finally:
        heartbeat.cancel()
        try:
            await heartbeat
        except asyncio.CancelledError:
            pass
        if not reader.done():
            reader.cancel()
        await release_lease(redis, decision.lease)
        logical = decision.lease.logical_key
        await redis.delete(f"buddy:job-lease:{logical}", f"buddy:job-done:{logical}")
        await redis.aclose()
    assert committed == [b"audio"]
    assert len(renewals) >= 1


async def test_cancelled_read_does_not_commit(monkeypatch, tmp_path):
    path = tmp_path / "audio.bin"
    path.write_bytes(b"audio")
    committed: list[str] = []

    def slow_read(self):
        time.sleep(0.4)
        return b"audio"

    monkeypatch.setattr(Path, "read_bytes", slow_read)

    async def job():
        payload = await read_file_bytes(path)
        committed.append(payload.decode("ascii"))

    task = asyncio.create_task(job())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.5)
    assert committed == []


async def test_concurrent_reads_stay_inside_the_shared_executor(tmp_path):
    source = Path(file_io.__file__).read_text(encoding="utf-8")
    assert "ThreadPoolExecutor(" not in source
    paths = []
    for index in range(12):
        path = tmp_path / f"{index}.bin"
        path.write_bytes(b"x" * 4096)
        paths.append(path)
    before = threading.active_count()
    await asyncio.gather(*(read_file_bytes(path) for path in paths))
    assert threading.active_count() - before < 40


async def test_header_read_does_not_allocate_the_whole_file(tmp_path):
    import tracemalloc

    path = tmp_path / "chunk.webm"
    path.write_bytes(WEBM_EBML_ID + (b"\x00" * (2 * 1024 * 1024)))
    tracemalloc.start()
    try:
        tracemalloc.clear_traces()
        header = await read_file_prefix(path, 4)
        _, prefix_peak = tracemalloc.get_traced_memory()
        tracemalloc.clear_traces()
        payload = await read_file_bytes(path)
        _, full_peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert header == WEBM_EBML_ID
    assert len(payload) > 2_000_000
    assert prefix_peak < 64_000
    assert full_peak > 1_000_000


async def test_deepgram_http_body_and_params_stay_the_same(monkeypatch, tmp_path):
    path = tmp_path / "audio.wav"
    body = b"RIFF-audio"
    path.write_bytes(body)
    monkeypatch.setattr(settings, "DEEPGRAM_API_KEY", "test-key")
    monkeypatch.setattr(deepgram_provider, "_sdk_v3", lambda: None)
    captured: dict = {}

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, params=None, headers=None, content=None):
            captured["url"] = url
            captured["params"] = params
            captured["content"] = content
            captured["content_type"] = headers["Content-Type"]
            request = httpx.Request("POST", url)
            return httpx.Response(200, json={"metadata": {"request_id": "req-1"}, "results": {}}, request=request)

    monkeypatch.setattr(deepgram_provider.httpx, "AsyncClient", Client)
    result = await deepgram_provider.deepgram_transcribe_from_path(str(path), "audio.wav", "audio/wav")
    assert captured["url"] == "https://api.deepgram.com/v1/listen"
    assert captured["content"] == body
    assert captured["content_type"] == "audio/wav"
    assert captured["params"]["model"] == settings.DEEPGRAM_MODEL
    assert result["provider"] == "deepgram"
    assert result["model"] == settings.DEEPGRAM_MODEL


async def test_webm_header_check_does_not_read_the_whole_chunk(monkeypatch, tmp_path):
    video = tmp_path / "chunk.webm"
    video.write_bytes(WEBM_EBML_ID + (b"\x11" * 300_000))
    calls = []

    def forbidden(self):
        calls.append("full")
        raise AssertionError("full read")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    result = await _ensure_standalone_webm(
        video,
        sequence=3,
        user_id="user-1",
        meeting_session_id="meeting-1",
        media_kind="video",
        bucket="bucket",
        job_dir=tmp_path,
    )
    assert result == video
    assert calls == []


def test_write_standalone_webm_can_reuse_loaded_bytes(tmp_path):
    chunk = tmp_path / "chunk.webm"
    chunk.write_bytes(b"\x00\x00")
    output = tmp_path / "out.webm"
    result = write_standalone_webm(chunk, WEBM_EBML_ID, output, b"\x00\x00")
    assert result == output
    assert output.read_bytes().startswith(WEBM_EBML_ID)


def test_speech_and_vector_concurrency_and_transport_defaults():
    speech = build_speech_stream_consumer(redis=object())
    vector = build_vector_stream_consumer(redis=object())
    assert speech.concurrency == 1
    assert vector.concurrency == 1
    assert settings.SPEECH_VECTOR_QUEUE_TRANSPORT == "list"


async def test_fast_work_proceeds_during_an_offloaded_read(monkeypatch, tmp_path):
    path = tmp_path / "audio.bin"
    path.write_bytes(b"audio")
    order: list[str] = []

    def slow_read(self):
        time.sleep(0.3)
        return b"audio"

    monkeypatch.setattr(Path, "read_bytes", slow_read)

    async def fast():
        await asyncio.sleep(0.05)
        order.append("fast")

    async def slow():
        await read_file_bytes(path)
        order.append("slow")

    await asyncio.gather(slow(), fast())
    assert order == ["fast", "slow"]


async def test_mixed_local_jobs_finish_without_growing_tasks(tmp_path):
    paths = []
    for index in range(6):
        path = tmp_path / f"job-{index}.bin"
        path.write_bytes(b"a" * 8192)
        paths.append(path)
    before = len(asyncio.all_tasks())

    async def speech(path: Path) -> int:
        payload = await read_file_bytes(path)
        await asyncio.sleep(0.01)
        return len(payload)

    async def briefing() -> str:
        await asyncio.sleep(0.01)
        return "ready"

    sizes, notes = await asyncio.gather(
        asyncio.gather(*(speech(path) for path in paths)),
        asyncio.gather(*(briefing() for _ in range(6))),
    )
    assert sizes == [8192] * 6
    assert notes == ["ready"] * 6
    assert len(asyncio.all_tasks()) - before <= 2
