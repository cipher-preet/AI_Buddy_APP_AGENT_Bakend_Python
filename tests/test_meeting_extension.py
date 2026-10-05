import asyncio
from pathlib import Path

from services.meeting_extension.ffmpeg_audio import (
    MeetingAudioExtractionError,
    extract_audio_for_stt,
    extract_webm_init,
    has_webm_header,
    reset_ffmpeg_bin_cache,
    resolve_ffmpeg_bin,
    WEBM_CLUSTER_ID,
    WEBM_EBML_ID,
    write_standalone_webm,
)
from services.meeting_extension.processor import process_meeting_video_chunk
from services.meeting_extension.segments import extract_stt_segments
from services.meeting_extension.timestamps import meeting_relative_ms
from services.queue.streams import EventEnvelope, NonRetryableQueueError
from services.speech.errors import STTPermanentAudioError
from apps.api_gateway.workers import speech_worker


def test_timestamp_normalization_example():
    assert meeting_relative_ms(600000, 1.2) == 601200
    assert meeting_relative_ms(600000, 6.4) == 606400


def test_timestamp_rounding_is_stable():
    assert meeting_relative_ms(1000, 0.0014) == 1001
    assert meeting_relative_ms(1000, 0.0015) == 1002


def test_webm_init_is_bytes_before_first_cluster(tmp_path):
    header = WEBM_EBML_ID + b"\x01\x02\x03"
    cluster = WEBM_CLUSTER_ID + b"cluster-payload"
    assert extract_webm_init(header + cluster) == header
    assert has_webm_header(header + cluster)
    assert not has_webm_header(cluster)
    fragment = tmp_path / "later.webm"
    fragment.write_bytes(cluster)
    repaired = write_standalone_webm(fragment, header)
    assert has_webm_header(repaired.read_bytes())
    assert repaired.read_bytes().endswith(cluster)


def test_segment_extraction_uses_chunk_offsets_not_sequence_math():
    result = {
        "transcript": "hello there",
        "results": {
            "utterances": [
                {"transcript": "hello there", "start": 1.2, "end": 6.4, "speaker": 0},
            ]
        },
    }
    segments = extract_stt_segments(
        result,
        chunk_start_offset_ms=600000,
        chunk_end_offset_ms=620000,
        started_at="2026-01-01T00:00:00+00:00",
        sequence=99,
        chunk_id="meeting:abc:chunk:99",
    )
    assert len(segments) == 1
    assert segments[0]["startOffsetMs"] == 601200
    assert segments[0]["endOffsetMs"] == 606400
    assert segments[0]["speakerId"] == 0
    assert segments[0]["text"] == "hello there"


def test_ffmpeg_failure_and_temp_cleanup(monkeypatch, tmp_path):
    cleaned = []

    class FakeProcess:
        returncode = 1

        async def communicate(self):
            return b"", b"Invalid data found when processing input"

        def kill(self):
            return None

        async def wait(self):
            return 1

    async def fake_exec(*command, **kwargs):
        return FakeProcess()

    monkeypatch.setattr("services.meeting_extension.ffmpeg_audio.asyncio.create_subprocess_exec", fake_exec)
    monkeypatch.setattr("services.meeting_extension.ffmpeg_audio.resolve_ffmpeg_bin", lambda: "ffmpeg")
    source = tmp_path / "chunk.webm"
    source.write_bytes(b"not-a-video")
    output = tmp_path / "out.wav"
    try:
        asyncio.run(extract_audio_for_stt(source, output))
        raise AssertionError("corrupt media must fail")
    except MeetingAudioExtractionError as error:
        assert error.corrupt is True
    assert source.exists()


def test_meeting_job_payload_is_parsed(monkeypatch, tmp_path):
    events = []
    updates = []

    class FakeRepo:
        async def get_transcript_chunk(self, conversation_id, sequence):
            return None

        async def mark_transcript_chunk_processing(self, conversation_id, sequence):
            return True

        async def complete_transcript_chunk(self, **kwargs):
            updates.append(kwargs)

        async def fail_transcript_chunk(self, *args, **kwargs):
            return True

        async def get_conversation(self, conversation_id):
            return None

    class FakeStorage:
        bucket = "bucket"

        async def download_file(self, bucket, object_key, destination):
            path = Path(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"video")
            return path

    async def fake_extract(input_path, output_path):
        path = Path(output_path).with_suffix(".wav")
        path.write_bytes(b"audio")
        return path

    async def fake_stt(**kwargs):
        return {
            "transcript": "hello",
            "language_code": "en",
            "request_id": "req-1",
            "provider": "deepgram",
            "results": {"utterances": [{"transcript": "hello", "start": 1.2, "end": 6.4}]},
        }

    class FakeProducer:
        async def publish(self, stream, event):
            events.append((stream, event.eventType, event.payload))

    monkeypatch.setattr("services.meeting_extension.processor.ConversationRepository", lambda db: FakeRepo())
    monkeypatch.setattr("services.meeting_extension.processor.get_s3_audio_storage", lambda: FakeStorage())
    monkeypatch.setattr("services.meeting_extension.processor.extract_audio_for_stt", fake_extract)
    monkeypatch.setattr("services.meeting_extension.processor.transcribe_from_path_with_fallback", fake_stt)
    monkeypatch.setattr("services.meeting_extension.processor.RedisStreamProducer", FakeProducer)
    monkeypatch.setattr("services.meeting_extension.processor.get_database", lambda: object())
    monkeypatch.setattr("services.meeting_extension.processor.temp_audio_root", lambda: tmp_path)
    monkeypatch.setattr("services.meeting_extension.processor.validate_meeting_object_key", lambda **kwargs: kwargs["object_key"])

    async def noop_mark(*args, **kwargs):
        return None

    monkeypatch.setattr("services.meeting_extension.processor._mark_chunk_processed", noop_mark)

    event = EventEnvelope(
        eventType="meeting.video.chunk.ready",
        correlationId="507f1f77bcf86cd799439011",
        userId="507f1f77bcf86cd799439012",
        spaceId="507f1f77bcf86cd799439013",
        conversationId="507f1f77bcf86cd799439011",
        payload={
            "jobType": "meeting_video_chunk_ready",
            "meetingSessionId": "507f1f77bcf86cd799439011",
            "chunkId": "meeting:507f1f77bcf86cd799439011:chunk:12",
            "sequence": 12,
            "userId": "507f1f77bcf86cd799439012",
            "spaceId": "507f1f77bcf86cd799439013",
            "s3Key": "meetings/507f1f77bcf86cd799439012/507f1f77bcf86cd799439011/chunks/000012.webm",
            "startOffsetMs": 600000,
            "endOffsetMs": 620000,
            "sourceType": "meeting_extension",
        },
    )
    asyncio.run(process_meeting_video_chunk(event))
    assert updates
    assert updates[0]["extra_fields"]["segments"][0]["startOffsetMs"] == 601200
    assert events[0][1] == "conversation.transcript.ready"


def test_duplicate_completed_transcript_is_skipped(monkeypatch):
    class Completed:
        sttStatus = type("S", (), {"COMPLETED": "completed"}) 

    class Chunk:
        sttStatus = __import__("services.conversation.models", fromlist=["STTStatus"]).STTStatus.COMPLETED

    class FakeRepo:
        async def get_transcript_chunk(self, conversation_id, sequence):
            return Chunk()

    monkeypatch.setattr("services.meeting_extension.processor.ConversationRepository", lambda db: FakeRepo())
    monkeypatch.setattr("services.meeting_extension.processor.get_database", lambda: object())

    async def fail_download(*args, **kwargs):
        raise AssertionError("duplicate jobs must not download")

    monkeypatch.setattr("services.meeting_extension.processor.get_s3_audio_storage", fail_download)
    event = EventEnvelope(
        eventType="meeting.video.chunk.ready",
        correlationId="c1",
        userId="u1",
        spaceId="s1",
        conversationId="c1",
        payload={"meetingSessionId": "c1", "sequence": 1, "s3Key": "meetings/u1/c1/chunks/000001.webm"},
    )
    asyncio.run(process_meeting_video_chunk(event))


def test_corrupt_media_is_not_retried(monkeypatch, tmp_path):
    class FakeRepo:
        async def get_transcript_chunk(self, conversation_id, sequence):
            return None

        async def mark_transcript_chunk_processing(self, conversation_id, sequence):
            return True

        async def fail_transcript_chunk(self, *args, **kwargs):
            self.failed = kwargs
            return True

        async def get_conversation(self, conversation_id):
            return None

    repo = FakeRepo()

    class FakeStorage:
        bucket = "bucket"

        async def download_file(self, bucket, object_key, destination):
            path = Path(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"bad")
            return path

    async def boom(input_path, output_path):
        raise MeetingAudioExtractionError("Invalid data found", corrupt=True)

    monkeypatch.setattr("services.meeting_extension.processor.ConversationRepository", lambda db: repo)
    monkeypatch.setattr("services.meeting_extension.processor.get_s3_audio_storage", lambda: FakeStorage())
    monkeypatch.setattr("services.meeting_extension.processor.extract_audio_for_stt", boom)
    monkeypatch.setattr("services.meeting_extension.processor.get_database", lambda: object())
    monkeypatch.setattr("services.meeting_extension.processor.temp_audio_root", lambda: tmp_path)
    monkeypatch.setattr("services.meeting_extension.processor.validate_meeting_object_key", lambda **kwargs: kwargs["object_key"])

    async def noop_mark(*args, **kwargs):
        return None

    monkeypatch.setattr("services.meeting_extension.processor._mark_chunk_processed", noop_mark)

    event = EventEnvelope(
        eventType="meeting.video.chunk.ready",
        correlationId="c1",
        userId="u1",
        spaceId="s1",
        conversationId="c1",
        payload={
            "meetingSessionId": "c1",
            "sequence": 1,
            "s3Key": "meetings/u1/c1/chunks/000001.webm",
            "startOffsetMs": 0,
            "endOffsetMs": 1000,
        },
    )
    try:
        asyncio.run(process_meeting_video_chunk(event))
        raise AssertionError("corrupt media must raise NonRetryableQueueError")
    except NonRetryableQueueError:
        pass


def test_existing_mobile_speech_job_normalizer_unchanged():
    job = speech_worker._normalize_speech_job(
        {
            "jobId": "job-1",
            "userId": "user-1",
            "spaceId": "space-1",
            "filePath": "/tmp/audio.wav",
            "contentType": "audio/wav",
            "filename": "audio.wav",
        }
    )
    assert job["job_id"] == "job-1"
    assert job["user_id"] == "user-1"
    assert "sourceType" not in job or job.get("sourceType") != "meeting_extension"


def test_out_of_order_chunks_keep_their_offsets():
    first = extract_stt_segments(
        {"transcript": "late", "results": {"utterances": [{"transcript": "late", "start": 0.0, "end": 1.0}]}},
        chunk_start_offset_ms=40000,
        chunk_end_offset_ms=60000,
        started_at=None,
        sequence=3,
        chunk_id="c3",
    )
    second = extract_stt_segments(
        {"transcript": "early", "results": {"utterances": [{"transcript": "early", "start": 0.0, "end": 1.0}]}},
        chunk_start_offset_ms=0,
        chunk_end_offset_ms=20000,
        started_at=None,
        sequence=1,
        chunk_id="c1",
    )
    assert first[0]["startOffsetMs"] == 40000
    assert second[0]["startOffsetMs"] == 0


def test_stt_permanent_failure_does_not_fail_other_chunks(monkeypatch, tmp_path):
    class FakeRepo:
        async def get_transcript_chunk(self, conversation_id, sequence):
            return None

        async def mark_transcript_chunk_processing(self, conversation_id, sequence):
            return True

        async def fail_transcript_chunk(self, *args, **kwargs):
            return True

        async def get_conversation(self, conversation_id):
            return None

    class FakeStorage:
        bucket = "b"

        async def download_file(self, bucket, object_key, destination):
            path = Path(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"video")
            return path

    async def fake_extract(input_path, output_path):
        path = Path(output_path).with_suffix(".wav")
        path.write_bytes(b"audio")
        return path

    async def fail_stt(**kwargs):
        raise STTPermanentAudioError("bad audio", provider="deepgram")

    monkeypatch.setattr("services.meeting_extension.processor.ConversationRepository", lambda db: FakeRepo())
    monkeypatch.setattr("services.meeting_extension.processor.get_s3_audio_storage", lambda: FakeStorage())
    monkeypatch.setattr("services.meeting_extension.processor.extract_audio_for_stt", fake_extract)
    monkeypatch.setattr("services.meeting_extension.processor.transcribe_from_path_with_fallback", fail_stt)
    monkeypatch.setattr("services.meeting_extension.processor.get_database", lambda: object())
    monkeypatch.setattr("services.meeting_extension.processor.temp_audio_root", lambda: tmp_path)
    monkeypatch.setattr("services.meeting_extension.processor.validate_meeting_object_key", lambda **kwargs: kwargs["object_key"])
    monkeypatch.setattr("services.meeting_extension.processor._mark_chunk_processed", lambda *args, **kwargs: asyncio.sleep(0))

    event = EventEnvelope(
        eventType="meeting.video.chunk.ready",
        correlationId="c1",
        userId="u1",
        spaceId="s1",
        conversationId="c1",
        payload={"meetingSessionId": "c1", "sequence": 2, "s3Key": "meetings/u1/c1/chunks/000002.webm", "startOffsetMs": 0, "endOffsetMs": 1},
    )
    try:
        asyncio.run(process_meeting_video_chunk(event))
        raise AssertionError("expected non-retryable STT failure")
    except NonRetryableQueueError:
        pass


def test_audio_chunk_skips_ffmpeg(monkeypatch, tmp_path):
    updates = []

    class FakeRepo:
        async def get_transcript_chunk(self, conversation_id, sequence):
            return None

        async def mark_transcript_chunk_processing(self, conversation_id, sequence):
            return True

        async def complete_transcript_chunk(self, **kwargs):
            updates.append(kwargs)

        async def fail_transcript_chunk(self, *args, **kwargs):
            return True

        async def get_conversation(self, conversation_id):
            return None

    class FakeStorage:
        bucket = "bucket"

        async def download_file(self, bucket, object_key, destination):
            path = Path(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"audio-bytes")
            return path

    async def boom_extract(*args, **kwargs):
        raise AssertionError("audio chunks must skip FFmpeg")

    async def fake_stt(**kwargs):
        assert kwargs["content_type"] == "audio/webm"
        return {
            "transcript": "hello",
            "provider": "deepgram",
            "results": {"utterances": [{"transcript": "hello", "start": 0.0, "end": 1.0}]},
        }

    class FakeProducer:
        async def publish(self, stream, event):
            return None

    monkeypatch.setattr("services.meeting_extension.processor.ConversationRepository", lambda db: FakeRepo())
    monkeypatch.setattr("services.meeting_extension.processor.get_s3_audio_storage", lambda: FakeStorage())
    monkeypatch.setattr("services.meeting_extension.processor.extract_audio_for_stt", boom_extract)
    monkeypatch.setattr("services.meeting_extension.processor.transcribe_from_path_with_fallback", fake_stt)
    monkeypatch.setattr("services.meeting_extension.processor.RedisStreamProducer", FakeProducer)
    monkeypatch.setattr("services.meeting_extension.processor.get_database", lambda: object())
    monkeypatch.setattr("services.meeting_extension.processor.temp_audio_root", lambda: tmp_path)
    monkeypatch.setattr("services.meeting_extension.processor.validate_meeting_object_key", lambda **kwargs: kwargs["object_key"])
    monkeypatch.setattr("services.meeting_extension.processor._mark_chunk_processed", lambda *args, **kwargs: asyncio.sleep(0))

    event = EventEnvelope(
        eventType="meeting.video.chunk.ready",
        correlationId="c1",
        userId="u1",
        spaceId="s1",
        conversationId="c1",
        payload={
            "meetingSessionId": "c1",
            "sequence": 1,
            "mediaKind": "audio",
            "mimeType": "audio/webm",
            "s3Key": "meetings/u1/c1/audio/000001.webm",
            "startOffsetMs": 0,
            "endOffsetMs": 1000,
        },
    )
    asyncio.run(process_meeting_video_chunk(event))
    assert updates


def test_later_muxed_chunk_prepends_webm_init(monkeypatch, tmp_path):
    extracted = []
    header = WEBM_EBML_ID + b"\x01\x02"
    cluster = WEBM_CLUSTER_ID + b"later"

    class FakeRepo:
        async def get_transcript_chunk(self, conversation_id, sequence):
            return None

        async def mark_transcript_chunk_processing(self, conversation_id, sequence):
            return True

        async def complete_transcript_chunk(self, **kwargs):
            return None

        async def fail_transcript_chunk(self, *args, **kwargs):
            return True

        async def get_conversation(self, conversation_id):
            return None

    class FakeStorage:
        bucket = "bucket"

        async def download_file(self, bucket, object_key, destination):
            path = Path(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(header + cluster if object_key.endswith("000001.webm") else cluster)
            return path

    async def fake_extract(input_path, output_path):
        extracted.append(Path(input_path).read_bytes())
        path = Path(output_path).with_suffix(".wav")
        path.write_bytes(b"audio")
        return path

    async def fake_stt(**kwargs):
        return {
            "transcript": "hello",
            "language_code": "en",
            "request_id": "r1",
            "provider": "deepgram",
            "results": {"utterances": [{"transcript": "hello", "start": 0.0, "end": 1.0}]},
        }

    class FakeProducer:
        async def publish(self, stream, event):
            return None

    monkeypatch.setattr("services.meeting_extension.processor.ConversationRepository", lambda db: FakeRepo())
    monkeypatch.setattr("services.meeting_extension.processor.get_s3_audio_storage", lambda: FakeStorage())
    monkeypatch.setattr("services.meeting_extension.processor.extract_audio_for_stt", fake_extract)
    monkeypatch.setattr("services.meeting_extension.processor.transcribe_from_path_with_fallback", fake_stt)
    monkeypatch.setattr("services.meeting_extension.processor.RedisStreamProducer", FakeProducer)
    monkeypatch.setattr("services.meeting_extension.processor.get_database", lambda: object())
    monkeypatch.setattr("services.meeting_extension.processor.temp_audio_root", lambda: tmp_path)
    monkeypatch.setattr(
        "services.meeting_extension.processor.validate_meeting_object_key",
        lambda **kwargs: kwargs["object_key"],
    )

    async def noop_mark(*args, **kwargs):
        return None

    monkeypatch.setattr("services.meeting_extension.processor._mark_chunk_processed", noop_mark)

    event = EventEnvelope(
        eventType="meeting.video.chunk.ready",
        correlationId="c1",
        userId="u1",
        spaceId="s1",
        conversationId="c1",
        payload={
            "meetingSessionId": "c1",
            "sequence": 4,
            "mediaKind": "muxed",
            "mimeType": "video/webm",
            "s3Key": "meetings/u1/c1/chunks/000004.webm",
            "startOffsetMs": 0,
            "endOffsetMs": 1000,
        },
    )
    asyncio.run(process_meeting_video_chunk(event))
    assert extracted
    assert has_webm_header(extracted[0])
    assert extracted[0].endswith(cluster)


def test_merge_prefers_video_folder_then_muxed(monkeypatch, tmp_path):
    from services.meeting_extension.video_merge import _download_merge_chunk

    class FakeStorage:
        bucket = "b"
        keys = []

        async def download_file(self, bucket, object_key, destination):
            self.keys.append(object_key)
            if "/video/" in object_key:
                raise FileNotFoundError(object_key)
            path = Path(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"ok")
            return path

    storage = FakeStorage()
    dest = tmp_path / "000001.webm"
    asyncio.run(_download_merge_chunk(storage, "u1", "c1", 1, dest))
    assert any("/video/" in key for key in storage.keys)
    assert any("/chunks/" in key for key in storage.keys)
    assert dest.exists()


def test_meeting_s3_keys_separate_audio_and_video():
    from services.meeting_extension.s3_keys import meeting_chunk_object_key

    audio = meeting_chunk_object_key("u1", "c1", 3, media_kind="audio")
    video = meeting_chunk_object_key("u1", "c1", 3, media_kind="video")
    muxed = meeting_chunk_object_key("u1", "c1", 3)
    assert "/audio/" in audio
    assert "/video/" in video
    assert "/chunks/" in muxed
    assert audio.endswith("000003.webm")
    assert video.endswith("000003.webm")
    assert muxed.endswith("000003.webm")


def test_meeting_audio_jobs_share_app_stt_stream():
    from apps.api_gateway.config.setting import settings
    from apps.api_gateway.workers.meeting_extension_worker import build_meeting_video_consumer

    assert settings.REDIS_MEETING_VIDEO_STREAM == settings.REDIS_STT_STREAM
    assert settings.REDIS_STT_STREAM == "buddy:stt:jobs"
    assert build_meeting_video_consumer() is None


def test_stt_handler_routes_meeting_jobs_without_touching_mobile_path(monkeypatch):
    from apps.api_gateway.workers import conversation_workers

    called = {}

    async def fake_process(event):
        called["event"] = event

    monkeypatch.setattr(conversation_workers, "process_meeting_video_chunk", fake_process)

    event = EventEnvelope(
        eventType="meeting.video.chunk.ready",
        correlationId="c1",
        userId="u1",
        spaceId="s1",
        conversationId="c1",
        payload={
            "jobType": "meeting_video_chunk_ready",
            "sourceType": "meeting_extension",
            "meetingSessionId": "c1",
            "sequence": 1,
        },
    )
    asyncio.run(conversation_workers.handle_stt_event(event))
    assert called["event"] is event
    assert conversation_workers.is_meeting_extension_stt_event(event) is True

    mobile = EventEnvelope(
        eventType="stt.requested",
        correlationId="c2",
        userId="u2",
        spaceId="s2",
        conversationId="c2",
        payload={"conversationId": "c2", "sequenceNumber": 1},
    )
    assert conversation_workers.is_meeting_extension_stt_event(mobile) is False


def test_resolve_ffmpeg_uses_configured_binary(monkeypatch, tmp_path):
    binary = tmp_path / "ffmpeg.exe"
    binary.write_bytes(b"ffmpeg")
    monkeypatch.setattr(
        "services.meeting_extension.ffmpeg_audio.settings.MEETING_FFMPEG_BIN",
        str(binary),
    )
    reset_ffmpeg_bin_cache()
    try:
        assert resolve_ffmpeg_bin() == str(binary)
    finally:
        reset_ffmpeg_bin_cache()


def test_resolve_ffmpeg_missing_raises(monkeypatch):
    monkeypatch.setattr("services.meeting_extension.ffmpeg_audio.settings.MEETING_FFMPEG_BIN", "")
    monkeypatch.delenv("IMAGEIO_FFMPEG_EXE", raising=False)
    monkeypatch.setattr("services.meeting_extension.ffmpeg_audio.shutil.which", lambda name: None)
    monkeypatch.setattr(
        "services.meeting_extension.ffmpeg_audio.Path.is_file",
        lambda self: False,
    )

    class MissingImageio:
        @staticmethod
        def get_ffmpeg_exe():
            raise RuntimeError("no bundled ffmpeg")

    monkeypatch.setitem(__import__("sys").modules, "imageio_ffmpeg", MissingImageio)
    reset_ffmpeg_bin_cache()
    try:
        try:
            resolve_ffmpeg_bin()
            raise AssertionError("missing ffmpeg must fail")
        except MeetingAudioExtractionError as error:
            assert "not installed" in str(error)
            assert error.corrupt is False
    finally:
        reset_ffmpeg_bin_cache()


def test_concat_single_webm_chunk_skips_ffmpeg(tmp_path, monkeypatch):
    from services.meeting_extension.ffmpeg_audio import concat_webm_chunks

    async def fail_ffmpeg(*args, **kwargs):
        raise AssertionError("single complete WebM must not invoke ffmpeg")

    monkeypatch.setattr("services.meeting_extension.ffmpeg_audio._run_ffmpeg", fail_ffmpeg)
    chunk = tmp_path / "000001.webm"
    chunk.write_bytes(WEBM_EBML_ID + b"\x00" * 10_000)
    out = tmp_path / "meeting.mp4"
    result = asyncio.run(concat_webm_chunks([chunk], out))
    assert result.suffix == ".webm"
    assert result.exists()
    assert has_webm_header(result.read_bytes()[:4])
    assert result.stat().st_size > 8_192


def test_concat_uses_stream_copy_without_h264_remux(tmp_path, monkeypatch):
    from services.meeting_extension import ffmpeg_audio

    calls: list[list[str]] = []

    async def fake_run(command, *, timeout, allow_failure):
        calls.append(command)
        Path(command[-1]).write_bytes(WEBM_EBML_ID + b"\x00" * 10_000)
        return None

    monkeypatch.setattr(ffmpeg_audio, "_run_ffmpeg", fake_run)
    monkeypatch.setattr(ffmpeg_audio.settings, "MEETING_MERGE_ALLOW_REENCODE", False)
    a = tmp_path / "a.webm"
    b = tmp_path / "b.webm"
    a.write_bytes(WEBM_EBML_ID + WEBM_CLUSTER_ID + b"\x01" * 100)
    b.write_bytes(WEBM_CLUSTER_ID + b"\x02" * 100)
    out = tmp_path / "meeting.webm"
    result = asyncio.run(ffmpeg_audio.concat_webm_chunks([a, b], out))
    assert result.exists()
    assert has_webm_header(result.read_bytes()[:4])
    assert calls
    joined = " ".join(calls[0])
    assert "libx264" not in joined
    assert "-c" in calls[0] and "copy" in calls[0]


def test_unify_resolution_reencodes_only_mismatched_fragments(tmp_path, monkeypatch):
    from services.meeting_extension import ffmpeg_audio

    sizes = {"a": ("vp9", 1280, 720), "b": ("vp9", 1920, 1080), "c": ("vp9", 1280, 720)}
    encoded: list[list[str]] = []

    async def fake_probe(path):
        return sizes[path.name.split(".")[0]]

    async def fake_run(command, *, timeout, allow_failure):
        encoded.append(command)
        Path(command[-1]).write_bytes(WEBM_EBML_ID + b"\x00" * 4_096)
        return None

    monkeypatch.setattr(ffmpeg_audio, "_probe_video_stream", fake_probe)
    monkeypatch.setattr(ffmpeg_audio, "_run_ffmpeg", fake_run)
    fragments = []
    for name in ("a", "b", "c"):
        path = tmp_path / f"{name}.webm"
        path.write_bytes(WEBM_EBML_ID + b"\x01" * 100)
        fragments.append(path)

    result = asyncio.run(ffmpeg_audio._unify_fragment_resolution(fragments))

    assert [path.name for path in result] == ["a.webm", "b.1280x720.webm", "c.webm"]
    assert len(encoded) == 1
    joined = " ".join(encoded[0])
    assert "libvpx-vp9" in joined
    assert "pad=1280:720" in joined


def test_concat_drops_corrupt_fragment_and_keeps_merging(tmp_path, monkeypatch):
    from services.meeting_extension import ffmpeg_audio
    from services.meeting_extension.ffmpeg_audio import MeetingAudioExtractionError

    concat_inputs: list[str] = []

    async def fake_run(command, *, timeout, allow_failure):
        source = command[command.index("-i") + 1]
        if source.endswith("b.webm"):
            return MeetingAudioExtractionError("bad", corrupt=True)
        if "concat" in command:
            concat_inputs.append(Path(source).read_text(encoding="utf-8"))
        Path(command[-1]).write_bytes(WEBM_EBML_ID + b"\x00" * 10_000)
        return None

    monkeypatch.setattr(ffmpeg_audio, "_run_ffmpeg", fake_run)
    monkeypatch.setattr(ffmpeg_audio.settings, "MEETING_MERGE_ALLOW_REENCODE", False)
    chunks = []
    for name in ("a", "b", "c"):
        path = tmp_path / f"{name}.webm"
        path.write_bytes(WEBM_EBML_ID + WEBM_CLUSTER_ID + b"\x01" * 100)
        chunks.append(path)
    result = asyncio.run(ffmpeg_audio.concat_webm_chunks(chunks, tmp_path / "meeting.webm"))
    assert result.exists()
    assert concat_inputs
    assert "a.clean.webm" in concat_inputs[0]
    assert "c.clean.webm" in concat_inputs[0]
    assert "b." not in concat_inputs[0]


def test_ffmpeg_timeout_on_one_fragment_does_not_abort(monkeypatch):
    from services.meeting_extension.ffmpeg_audio import MeetingAudioExtractionError, _run_ffmpeg

    class Hung:
        def __init__(self):
            self.returncode = None

        def kill(self):
            raise ProcessLookupError

        async def wait(self):
            return -9

        async def communicate(self):
            await asyncio.sleep(30)

    async def fake_exec(*args, **kwargs):
        return Hung()

    monkeypatch.setattr("services.meeting_extension.ffmpeg_audio.asyncio.create_subprocess_exec", fake_exec)
    monkeypatch.setattr("services.meeting_extension.ffmpeg_audio.resolve_ffmpeg_bin", lambda: "ffmpeg")
    failure = asyncio.run(_run_ffmpeg(["ffmpeg", "-i", "x"], timeout=0.05, allow_failure=True))
    assert isinstance(failure, MeetingAudioExtractionError)
    assert "timed out" in str(failure)


def test_killed_merge_is_resumed_and_repeated_timeouts_stop():
    from datetime import datetime, timezone

    from services.meeting_extension import video_merge
    from services.meeting_extension.video_merge import _merge_should_stand_down
    from services.queue.streams import NonRetryableQueueError

    class Sessions:
        def __init__(self, doc):
            self.doc = doc

        async def find_one(self, query, projection=None):
            return self.doc

    class DB:
        def __init__(self, doc):
            self.meeting_sessions = Sessions(doc)

    session_id = "6ac3f69712362e18f6c689a6"
    now = datetime.now(timezone.utc)
    video_merge._ACTIVE_MERGES.discard(session_id)
    # A RUNNING row is what a killed worker leaves behind. The next job must finish it.
    resumed = asyncio.run(
        _merge_should_stand_down(
            DB({"videoMergeStatus": "RUNNING", "updatedAt": now, "videoMergeAttempts": 1}),
            session_id,
        )
    )
    assert resumed is False
    video_merge._ACTIVE_MERGES.add(session_id)
    try:
        busy = asyncio.run(
            _merge_should_stand_down(
                DB({"videoMergeStatus": "RUNNING", "updatedAt": now, "videoMergeAttempts": 1}),
                session_id,
            )
        )
    finally:
        video_merge._ACTIVE_MERGES.discard(session_id)
    assert busy is True
    try:
        asyncio.run(
            _merge_should_stand_down(
                DB(
                    {
                        "videoMergeStatus": "FAILED",
                        "updatedAt": now,
                        "videoMergeAttempts": 2,
                        "lastErrorMessage": "FFmpeg timed out",
                    }
                ),
                session_id,
            )
        )
        raise AssertionError("repeated timeouts must stop")
    except NonRetryableQueueError:
        pass


def test_stuck_running_merge_is_published_again(monkeypatch):
    from services.meeting_extension import video_merge

    published = []

    class Cursor:
        async def to_list(self, limit):
            return [
                {
                    "_id": "6ac3f69712362e18f6c689a6",
                    "userId": "6a95868e4b6225ff9933a2b7",
                    "expectedFinalSequence": 24,
                }
            ][:limit]

    class Sessions:
        def find(self, query, projection=None):
            assert query["videoMergeStatus"] == "RUNNING"
            return Cursor()

    class DB:
        meeting_sessions = Sessions()

    class Producer:
        async def publish(self, stream, event):
            published.append((stream, event))
            return event.eventId

    monkeypatch.setattr(video_merge, "get_database_safe", lambda: DB())
    monkeypatch.setattr(video_merge, "RedisStreamProducer", Producer)
    monkeypatch.setattr(video_merge.settings, "MEETING_VIDEO_FINALIZATION_ENABLED", True)
    count = asyncio.run(video_merge.resume_stuck_video_merges())
    assert count == 1
    assert published[0][1].eventType == "meeting.video.merge.requested"
    assert published[0][1].payload["meetingSessionId"] == "6ac3f69712362e18f6c689a6"
    assert published[0][1].payload["expectedFinalSequence"] == 24
