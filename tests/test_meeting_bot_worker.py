import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess
from unittest.mock import AsyncMock
import wave

import httpx
import pytest

from apps.api_gateway.workers import meeting_bot_worker as bot
from apps.api_gateway.workers.meeting_bot_media import audio_chunk_duration, extract_chunks, safe_download_url
from services.meeting_extension.ffmpeg_audio import probe_ffmpeg


@pytest.mark.parametrize("url, accepted", [
    ("https://recall-data.s3.amazonaws.com/meeting.mp4?sig=abc", True),
    ("https://example.cloudfront.net/meeting.mp4", True),
    ("https://169.254.169.254/meeting.mp4", False),
    ("https://example.amazonaws.com.evil.test/meeting.mp4", False),
    ("https://key@example.amazonaws.com/meeting.mp4", False),
    ("http://example.amazonaws.com/meeting.mp4", False),
])
def test_recording_download_host_constraints(url, accepted):
    assert safe_download_url(url) is accepted


def test_extracts_independently_readable_audio_chunks(tmp_path):
    ffmpeg = probe_ffmpeg()
    if not ffmpeg:
        pytest.skip("FFmpeg is unavailable")
    video = tmp_path / "meeting.mp4"
    subprocess.run([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=black:s=64x64:r=10",
        "-f", "lavfi", "-i", "sine=frequency=400:sample_rate=16000", "-t", "5.5",
        "-c:v", "mpeg4", "-c:a", "aac", str(video),
    ], check=True, timeout=30)
    audio = asyncio.run(extract_chunks(video, 2, tmp_path))
    assert len(audio) == 3
    assert abs(sum(map(audio_chunk_duration, audio)) - 5500) < 100
    for chunk in audio:
        with wave.open(str(chunk)) as file:
            assert file.getframerate() == 16000 and file.getnchannels() == 1


def test_import_resumes_without_republishing_registered_chunks(tmp_path, monkeypatch):
    worker = object.__new__(bot.MeetingBotWorker)
    worker.http = None
    worker.save = AsyncMock()
    worker.wait_for_audio_capacity = AsyncMock()
    worker.terminal = AsyncMock()
    worker.node = AsyncMock(side_effect=[{}, {}, {"chunkMaxDurationMs": 60000, "chunkMaxSizeBytes": 10_000_000}, {}, {}])
    paths = [tmp_path / f"{index}.wav" for index in (1, 2)]
    for path in paths:
        with wave.open(str(path), "wb") as file:
            file.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            file.writeframes(b"\0" * 32000)
    monkeypatch.setenv("MEETING_BOT_TEMP_DIR", str(tmp_path))
    stored = AsyncMock()
    monkeypatch.setattr(bot, "store_video", stored)
    monkeypatch.setattr(bot, "extract_chunks", AsyncMock(return_value=paths))
    audio = AsyncMock()
    monkeypatch.setattr(bot, "store_audio_chunk", audio)
    started = datetime(2026, 11, 24, 9, 0, tzinfo=timezone.utc)
    run = {
        "_id": "run-id", "userId": "user-id", "joinAt": started,
        "videoStored": True, "registeredSequence": 1, "chunkSeconds": 45,
    }
    asyncio.run(worker.import_recording(run))
    assert stored.await_args.args[-1] is True
    audio.assert_awaited_once_with(paths[1], "user-id", "run-id", 2)
    assert [call.args[1] for call in worker.node.await_args_list] == ["live", "ended", "open", "chunk", "finish"]
    assert worker.node.await_args_list[3].args[2]["startOffsetMs"] == 1000
    assert worker.node.await_args_list[4].args[2]["durationMs"] == 2000
    worker.terminal.assert_awaited_once_with(run, "processing")
    assert worker.save.await_args.kwargs["importCompletion"]["finalSequence"] == 2


def test_retry_of_finish_only_retries_finalization(monkeypatch):
    worker = object.__new__(bot.MeetingBotWorker)
    worker.node = AsyncMock()
    worker.save = AsyncMock()
    worker.terminal = AsyncMock()
    media = AsyncMock()
    monkeypatch.setattr(bot, "store_video", media)
    run = {"importCompletion": {"finalSequence": 2, "durationMs": 90000, "endedAt": "2026-11-24T09:01:30Z"}}
    asyncio.run(worker.import_recording(run))
    assert [call.args[1] for call in worker.node.await_args_list] == ["live", "ended", "finish"]
    worker.terminal.assert_awaited_once_with(run, "processing")
    media.assert_not_awaited()


def test_stop_preserves_recording_when_the_recording_callback_was_missed():
    worker = object.__new__(bot.MeetingBotWorker)
    mock_response = httpx.Response(200, json={
        "status_changes": [{"code": "in_call_recording"}, {"code": "done"}], "recordings": [],
    }, request=httpx.Request("GET", "https://us-west-2.recall.ai/api/v1/bot/test/"))
    worker.recall = AsyncMock(return_value=mock_response)
    worker.node = AsyncMock()
    worker.save = AsyncMock()
    worker.terminal = AsyncMock()
    run = {"userId": "user-id", "_id": "run-id", "joinAt": bot.now(), "providerId": "test", "stopRequested": True}
    with pytest.raises(RuntimeError, match="media is pending"):
        asyncio.run(worker.import_recording(run))
    worker.terminal.assert_not_awaited()


def test_attendance_publishes_the_live_meeting_on_the_dashboard():
    worker = object.__new__(bot.MeetingBotWorker)
    worker.save = AsyncMock()
    worker.node = AsyncMock()
    worker.defer_attendance = AsyncMock()
    worker.terminal = AsyncMock()
    provider_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    response = httpx.Response(200, json={
        "status_changes": [{"code": "in_call_recording"}],
    }, request=httpx.Request("GET", f"https://us-east-1.recall.ai/api/v1/bot/{provider_id}/"))
    worker.recall = AsyncMock(return_value=response)
    run = {
        "_id": "run-id",
        "providerId": provider_id,
        "joinAt": bot.now() - timedelta(minutes=2),
        "providerStatus": "in_waiting_room",
        "providerStatusAt": bot.now() - timedelta(seconds=30),
    }
    asyncio.run(worker.attendance(run))
    worker.node.assert_awaited_once()
    assert worker.node.await_args.args[1] == "live"
    assert worker.save.await_args.kwargs["sessionPublished"] is True
    assert worker.save.await_args.kwargs["providerStatus"] == "in_call_recording"
    worker.defer_attendance.assert_not_awaited()


def test_unconfirmed_provider_request_is_recovered_without_creating_a_second_bot():
    worker = object.__new__(bot.MeetingBotWorker)
    worker.save = AsyncMock()
    worker.node = AsyncMock()
    response = httpx.Response(200, json={"results": [
        {"id": "original-provider-id", "metadata": {"kukunotes_run": "run-id"}},
    ]}, request=httpx.Request("GET", "https://us-west-2.recall.ai/api/v1/bot/"))
    worker.recall = AsyncMock(return_value=response)
    run = {"_id": "run-id", "dispatchStartedAt": bot.now() - timedelta(seconds=20)}
    asyncio.run(worker.dispatch(run))
    worker.recall.assert_awaited_once_with("GET", "/bot/", params={"metadata__kukunotes_run": "run-id"})
    assert worker.save.await_args.kwargs["providerId"] == "original-provider-id"
    worker.node.assert_not_awaited()


def test_card_thumbnail_keeps_the_largest_frame_and_skips_an_existing_one(tmp_path, monkeypatch):
    video = tmp_path / "meeting.mp4"
    video.write_bytes(b"video")
    uploaded = {}

    async def fake_extract(_source, destination, seek_seconds=1.0):
        destination = Path(destination)
        destination.write_bytes(b"x" * (1000 if seek_seconds == 60 else 120))
        return destination

    class Storage:
        async def upload_file(self, path, key, content_type=None):
            uploaded["bytes"] = Path(path).stat().st_size
            uploaded["key"] = key
            uploaded["type"] = content_type

    class Sessions:
        def __init__(self):
            self.doc = None

        async def find_one(self, query, projection=None):
            if self.doc and self.doc.get("_id") == query["_id"]:
                return dict(self.doc)
            return None

        async def update_one(self, query, update):
            self.doc = {"_id": query["_id"], **update["$set"]}

    sessions = Sessions()
    monkeypatch.setattr("services.meeting_extension.ffmpeg_audio.extract_video_thumbnail", fake_extract)
    monkeypatch.setattr("services.storage.s3_audio_storage.get_s3_audio_storage", lambda: Storage())
    asyncio.run(bot.store_meeting_card_thumbnail(type("DB", (), {"meeting_sessions": sessions})(), "meeting-1", "user-1", video))
    assert uploaded["bytes"] == 1000
    assert uploaded["type"] == "image/jpeg"
    assert uploaded["key"].endswith("/thumbnail.jpg")
    assert sessions.doc["thumbnailS3Key"] == uploaded["key"]
    uploaded.clear()
    asyncio.run(bot.store_meeting_card_thumbnail(type("DB", (), {"meeting_sessions": sessions})(), "meeting-1", "user-1", video))
    assert uploaded == {}
