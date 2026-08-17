import asyncio
import hashlib
import hmac
import json
import time

import httpx
from fastapi.testclient import TestClient

from apps.api_gateway.config.setting import settings
from apps.api_gateway.workers import conversation_workers, speech_worker, vector_worker
from apps.api_gateway import queue_api
from services.conversation.models import STTStatus, TranscriptChunkDocument
from services.queue.streams import EventEnvelope
from services.speech import transcription_router
from services.speech.errors import STTPermanentAudioError, STTProviderBillingError
from services.speech.providers import sarvam_provider
from services.storage.s3_audio_storage import (
    PermanentS3StorageError,
    build_audio_object_key,
    build_conversation_audio_object_key,
    safe_temp_audio_path,
    sanitize_filename,
    validate_allowed_audio_upload,
    validate_conversation_audio_object_key,
)


def test_duplicate_speech_job_does_not_call_processor(monkeypatch):
    async def completed(job_id):
        return {"status": "completed"}

    async def fail_transcribe(**kwargs):
        raise AssertionError("duplicate completed jobs must not be transcribed again")

    monkeypatch.setattr(speech_worker, "get_job_result", completed)
    monkeypatch.setattr(speech_worker, "transcribe_from_path_with_fallback", fail_transcribe)

    asyncio.run(speech_worker.process_speech_job({"job_id": "job-1"}))


def test_missing_local_file_path_is_permanent(monkeypatch):
    monkeypatch.setattr(settings, "STORAGE_PROVIDER", "local")
    failures = []

    async def missing_result(job_id):
        return None

    async def mark_failed(job_id, error):
        failures.append((job_id, error))

    monkeypatch.setattr(speech_worker, "get_job_result", missing_result)
    monkeypatch.setattr(speech_worker, "mark_job_failed", mark_failed)

    try:
        asyncio.run(speech_worker.process_speech_job({"job_id": "job-1"}))
    except ValueError as error:
        assert "file_path" in str(error)
    else:
        raise AssertionError("Malformed local speech jobs must be permanent failures")

    assert failures and failures[0][0] == "job-1"


def test_speech_job_normalizes_external_payload_aliases():
    job = speech_worker._normalize_speech_job(
        {
            "jobId": "job-1",
            "userId": " user-1 ",
            "spaceId": " space-1 ",
            "filePath": " /tmp/audio.wav ",
            "contentType": " audio/wav ",
            "storageProvider": " local ",
            "s3Bucket": " bucket ",
            "s3ObjectKey": " key.wav ",
            "filename": " audio.wav ",
        }
    )

    assert job["job_id"] == "job-1"
    assert job["user_id"] == "user-1"
    assert job["space_id"] == "space-1"
    assert job["file_path"] == "/tmp/audio.wav"
    assert job["content_type"] == "audio/wav"
    assert job["storage_provider"] == "local"
    assert job["s3_bucket"] == "bucket"
    assert job["s3_object_key"] == "key.wav"


def test_s3_key_and_temp_path_are_sanitized(monkeypatch):
    monkeypatch.setattr(settings, "S3_AUDIO_PREFIX", "buddy/audio")
    monkeypatch.setattr(settings, "WORKER_TEMP_AUDIO_ROOT", "/tmp/buddy")

    key = build_audio_object_key(
        user_id="../user one",
        space_id="space/one",
        session_id=None,
        job_id="job-1",
        filename="../hello world.wav",
    )
    temp_path = safe_temp_audio_path({"job_id": "../job-1", "filename": "../hello world.wav"})

    assert key == "buddy/audio/user-one/space-one/no-session/job-1/hello-world.wav"
    assert str(temp_path).replace("\\", "/").endswith("/tmp/buddy/job-1/hello-world.wav")
    assert sanitize_filename("../../bad name.mp3") == "bad-name.mp3"


def test_cleanup_job_dir_removes_only_inside_allowed_root(monkeypatch, tmp_path):
    root = tmp_path / "buddy-temp"
    job_dir = root / "job-1"
    job_dir.mkdir(parents=True)
    (job_dir / "audio.wav").write_bytes(b"audio")
    monkeypatch.setattr(settings, "WORKER_TEMP_AUDIO_ROOT", str(root))

    speech_worker._cleanup_job_dir(job_dir)

    assert not job_dir.exists()
    assert root.exists()


def test_cleanup_job_dir_refuses_outside_allowed_root(monkeypatch, tmp_path):
    root = tmp_path / "buddy-temp"
    outside = tmp_path / "outside-job"
    outside.mkdir()
    monkeypatch.setattr(settings, "WORKER_TEMP_AUDIO_ROOT", str(root))

    try:
        speech_worker._cleanup_job_dir(outside)
    except ValueError as error:
        assert "outside temporary audio root" in str(error)
    else:
        raise AssertionError("cleanup must reject directories outside the configured temp root")

    assert outside.exists()


def test_cleanup_job_dir_refuses_temp_root_itself(monkeypatch, tmp_path):
    root = tmp_path / "buddy-temp"
    root.mkdir()
    monkeypatch.setattr(settings, "WORKER_TEMP_AUDIO_ROOT", str(root))

    try:
        speech_worker._cleanup_job_dir(root)
    except ValueError as error:
        assert "temporary audio root" in str(error)
    else:
        raise AssertionError("cleanup must not delete the configured temp root itself")

    assert root.exists()


def test_direct_upload_key_scope_and_validation(monkeypatch):
    monkeypatch.setattr(settings, "S3_AUDIO_PREFIX", "buddy/audio")
    monkeypatch.setattr(settings, "S3_MAX_AUDIO_SIZE_BYTES", 1000)

    key = build_conversation_audio_object_key(
        user_id="user/1",
        space_id="space 1",
        conversation_id="conv-1",
        sequence_number=2,
        chunk_id="chunk-1",
        extension=".webm",
    )

    assert key == "buddy/audio/user-1/space-1/conv-1/00000002-chunk-1.webm"
    assert validate_conversation_audio_object_key(
        object_key=key,
        user_id="user/1",
        space_id="space 1",
        conversation_id="conv-1",
    ) == key
    assert validate_allowed_audio_upload(
        content_type="audio/webm;codecs=opus",
        extension="webm",
        expected_size_bytes=1000,
    ) == ("audio/webm", "webm")

    try:
        validate_conversation_audio_object_key(
            object_key="buddy/audio/other/space-1/conv-1/00000002-chunk-1.webm",
            user_id="user/1",
            space_id="space 1",
            conversation_id="conv-1",
        )
    except PermanentS3StorageError as error:
        assert "outside" in str(error)
    else:
        raise AssertionError("object keys outside the conversation scope must be rejected")


def test_s3_speech_job_downloads_and_injects_file_path(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "STORAGE_PROVIDER", "s3")
    monkeypatch.setattr(settings, "S3_DELETE_AFTER_PROCESSING", False)
    monkeypatch.setattr(settings, "WORKER_TEMP_AUDIO_ROOT", str(tmp_path))

    async def missing_result(job_id):
        return None

    async def noop(*args, **kwargs):
        return None

    injected_paths = []

    async def transcribe(file_path, filename, content_type):
        injected_paths.append(file_path)
        assert filename == "audio.wav"
        assert content_type == "audio/wav"
        return {"transcript": "hello"}

    class Storage:
        async def download_file(self, bucket, object_key, destination):
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"audio")
            return destination

    monkeypatch.setattr(speech_worker, "get_job_result", missing_result)
    monkeypatch.setattr(speech_worker, "mark_job_processing", noop)
    monkeypatch.setattr(speech_worker, "save_job_result", noop)
    monkeypatch.setattr(speech_worker, "push_completed_speech_job", noop)
    monkeypatch.setattr(speech_worker, "transcribe_from_path_with_fallback", transcribe)
    monkeypatch.setattr(speech_worker, "get_s3_audio_storage", lambda: Storage())

    asyncio.run(
        speech_worker.process_speech_job(
            {
                "job_id": "job-1",
                "storage_provider": "s3",
                "s3_bucket": "bucket",
                "s3_object_key": "buddy/audio/job-1/audio.wav",
                "filename": "audio.wav",
                "content_type": "audio/wav",
            }
        )
    )

    assert injected_paths
    assert not (tmp_path / "job-1").exists()


def test_sarvam_empty_transcript_is_successful_empty_result(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "STT_ALLOW_SARVAM_FALLBACK", True)
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")

    async def post(*args, **kwargs):
        return httpx.Response(
            200,
            json={"transcript": ""},
            request=httpx.Request("POST", "https://api.test/speech-to-text"),
        )

    monkeypatch.setattr(sarvam_provider, "_post_with_retries", post)

    result = asyncio.run(
        sarvam_provider.sarvam_transcribe_from_path(
            file_path=str(audio_path),
            filename="audio.wav",
            content_type="audio/wav",
        )
    )

    assert result["transcript"] == ""
    assert result["provider"] == "sarvam"
    assert result["is_empty_transcript"] is True


def test_sarvam_upload_normalizes_browser_content_type(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "STT_ALLOW_SARVAM_FALLBACK", True)
    audio_path = tmp_path / "audio"
    audio_path.write_bytes(b"audio")
    uploaded = {}

    async def post(url, headers, data, files):
        filename, file_obj, content_type = files["file"]
        uploaded["filename"] = filename
        uploaded["content_type"] = content_type
        uploaded["file_bytes"] = file_obj.read()
        return httpx.Response(
            200,
            json={"transcript": "hello"},
            request=httpx.Request("POST", "https://api.test/speech-to-text"),
        )

    monkeypatch.setattr(sarvam_provider, "_post_with_retries", post)

    asyncio.run(
        sarvam_provider.sarvam_transcribe_from_path(
            file_path=str(audio_path),
            filename="audio",
            content_type="audio/webm;codecs=opus",
        )
    )

    assert uploaded == {
        "filename": "audio.webm",
        "content_type": "audio/webm",
        "file_bytes": b"audio",
    }


def test_sarvam_invalid_audio_error_is_permanent(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "STT_ALLOW_SARVAM_FALLBACK", True)
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"not really audio")

    async def post(*args, **kwargs):
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": "Failed to read the file, please check the audio format.",
                    "code": "invalid_request_error",
                }
            },
            request=httpx.Request("POST", "https://api.test/speech-to-text"),
        )

    monkeypatch.setattr(sarvam_provider, "_post_with_retries", post)

    try:
        asyncio.run(
            sarvam_provider.sarvam_transcribe_from_path(
                file_path=str(audio_path),
                filename="audio.wav",
                content_type="audio/wav",
            )
        )
    except ValueError as error:
        assert "Failed to read the file" in str(error)
    else:
        raise AssertionError("Sarvam invalid audio must be treated as permanent")


def test_sarvam_audio_duration_limit_error_is_permanent(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "STT_ALLOW_SARVAM_FALLBACK", True)
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"long audio")

    async def post(*args, **kwargs):
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": (
                        "Audio duration exceeds the maximum limit of 30 seconds. "
                        "Please use the batch API for longer audio files."
                    ),
                    "code": "invalid_request_error",
                }
            },
            request=httpx.Request("POST", "https://api.test/speech-to-text"),
        )

    monkeypatch.setattr(sarvam_provider, "_post_with_retries", post)

    try:
        asyncio.run(
            sarvam_provider.sarvam_transcribe_from_path(
                file_path=str(audio_path),
                filename="audio.wav",
                content_type="audio/wav",
            )
        )
    except ValueError as error:
        assert "Audio duration exceeds" in str(error)
    else:
        raise AssertionError("Sarvam duration limit errors must be treated as permanent")


def test_stt_router_uses_deepgram_first(monkeypatch, tmp_path):
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")
    calls = []

    async def deepgram(file_path, filename, content_type):
        calls.append("deepgram")
        return {
            "transcript": "namaste",
            "provider": "deepgram",
            "model": "nova-3",
            "language_code": "hi",
        }

    async def sarvam(file_path, filename, content_type):
        calls.append("sarvam")
        return {"transcript": "should not run", "provider": "sarvam"}

    monkeypatch.setattr(settings, "STT_PROVIDER_ORDER", "deepgram,sarvam")
    monkeypatch.setattr(settings, "STT_ALLOW_SARVAM_FALLBACK", True)
    monkeypatch.setitem(transcription_router._PROVIDERS, "deepgram", deepgram)
    monkeypatch.setitem(transcription_router._PROVIDERS, "sarvam", sarvam)

    result = asyncio.run(
        transcription_router.transcribe_from_path_with_fallback(
            file_path=str(audio_path),
            filename="audio.wav",
            content_type="audio/wav",
        )
    )

    assert calls == ["deepgram"]
    assert result["provider"] == "deepgram"
    assert result["transcript"] == "namaste"
    assert result["is_empty_transcript"] is False


def test_stt_router_falls_back_from_deepgram_billing_to_sarvam(monkeypatch, tmp_path):
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")
    calls = []

    async def deepgram(file_path, filename, content_type):
        calls.append("deepgram")
        raise STTProviderBillingError("Deepgram billing quota exceeded", provider="deepgram")

    async def sarvam(file_path, filename, content_type):
        calls.append("sarvam")
        return {
            "transcript": "mera kaam ho gaya",
            "provider": "sarvam",
            "model": "saaras:v3",
        }

    monkeypatch.setattr(settings, "STT_PROVIDER_ORDER", "deepgram,sarvam")
    monkeypatch.setattr(settings, "STT_ALLOW_SARVAM_FALLBACK", True)
    monkeypatch.setitem(transcription_router._PROVIDERS, "deepgram", deepgram)
    monkeypatch.setitem(transcription_router._PROVIDERS, "sarvam", sarvam)

    result = asyncio.run(
        transcription_router.transcribe_from_path_with_fallback(
            file_path=str(audio_path),
            filename="audio.wav",
            content_type="audio/wav",
        )
    )

    assert calls == ["deepgram", "sarvam"]
    assert result["provider"] == "sarvam"
    assert result["fallback_attempts"][0]["provider"] == "deepgram"
    assert result["fallback_attempts"][0]["status"] == "fallback"


def test_stt_router_does_not_fallback_for_permanent_audio_errors(monkeypatch, tmp_path):
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"bad audio")
    calls = []

    async def deepgram(file_path, filename, content_type):
        calls.append("deepgram")
        raise STTPermanentAudioError("Invalid audio format", provider="deepgram")

    async def sarvam(file_path, filename, content_type):
        calls.append("sarvam")
        return {"transcript": "should not run", "provider": "sarvam"}

    monkeypatch.setattr(settings, "STT_PROVIDER_ORDER", "deepgram,sarvam")
    monkeypatch.setattr(settings, "STT_ALLOW_SARVAM_FALLBACK", True)
    monkeypatch.setitem(transcription_router._PROVIDERS, "deepgram", deepgram)
    monkeypatch.setitem(transcription_router._PROVIDERS, "sarvam", sarvam)

    try:
        asyncio.run(
            transcription_router.transcribe_from_path_with_fallback(
                file_path=str(audio_path),
                filename="audio.wav",
                content_type="audio/wav",
            )
        )
    except ValueError as error:
        assert "Invalid audio format" in str(error)
    else:
        raise AssertionError("Permanent audio errors must not fallback to another provider")

    assert calls == ["deepgram"]


def test_stt_provider_order_filters_sarvam_unless_explicitly_enabled(monkeypatch):
    monkeypatch.setattr(settings, "STT_PROVIDER_ORDER", "deepgram,sarvam")
    monkeypatch.setattr(settings, "STT_ALLOW_SARVAM_FALLBACK", False)

    assert settings.stt_provider_order_list == ["deepgram"]

    monkeypatch.setattr(settings, "STT_ALLOW_SARVAM_FALLBACK", True)

    assert settings.stt_provider_order_list == ["deepgram", "sarvam"]


def test_empty_completed_speech_job_skips_vector_and_cleans_up(monkeypatch):
    completed = {
        "status": "completed",
        "user_id": "user-1",
        "space_id": "space-1",
        "result": {"transcript": "", "is_empty_transcript": True},
    }
    deleted = []

    async def get_result(job_id):
        return completed

    async def fail_store(*args, **kwargs):
        raise AssertionError("empty transcripts must not be indexed")

    async def delete_job(job_id):
        deleted.append(job_id)

    monkeypatch.setattr(vector_worker, "get_job_result", get_result)
    monkeypatch.setattr(vector_worker, "store_transcript_in_vector_db", fail_store)
    monkeypatch.setattr(vector_worker, "delete_speech_job", delete_job)

    asyncio.run(vector_worker.process_completed_speech_job("job-1"))

    assert deleted == ["job-1"]


def test_missing_s3_object_is_permanent(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "STORAGE_PROVIDER", "s3")
    monkeypatch.setattr(settings, "WORKER_TEMP_AUDIO_ROOT", str(tmp_path))
    failures = []

    async def missing_result(job_id):
        return None

    async def mark_failed(job_id, error):
        failures.append((job_id, error))

    class Storage:
        async def download_file(self, bucket, object_key, destination):
            raise PermanentS3StorageError("S3 permanent error: NoSuchKey")

    monkeypatch.setattr(speech_worker, "get_job_result", missing_result)
    monkeypatch.setattr(speech_worker, "mark_job_failed", mark_failed)
    monkeypatch.setattr(speech_worker, "get_s3_audio_storage", lambda: Storage())

    try:
        asyncio.run(
            speech_worker.process_speech_job(
                {
                    "job_id": "job-1",
                    "storage_provider": "s3",
                    "s3_bucket": "bucket",
                    "s3_object_key": "missing.wav",
                    "filename": "audio.wav",
                }
            )
        )
    except ValueError as error:
        assert "NoSuchKey" in str(error)
    else:
        raise AssertionError("Missing S3 object must be treated as a permanent failure")

    assert failures and failures[0][0] == "job-1"


def test_original_s3_error_survives_final_cleanup_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "STORAGE_PROVIDER", "s3")
    monkeypatch.setattr(settings, "WORKER_TEMP_AUDIO_ROOT", str(tmp_path))
    failures = []
    cleanup_calls = 0

    async def missing_result(job_id):
        return None

    async def mark_failed(job_id, error):
        failures.append((job_id, error))

    class Storage:
        async def download_file(self, bucket, object_key, destination):
            raise PermanentS3StorageError("S3 permanent error: NoSuchKey")

    original_cleanup = speech_worker._cleanup_job_dir

    def cleanup_then_fail(job_dir):
        nonlocal cleanup_calls
        cleanup_calls += 1
        if cleanup_calls == 1:
            return original_cleanup(job_dir)
        raise ValueError("cleanup exploded")

    monkeypatch.setattr(speech_worker, "get_job_result", missing_result)
    monkeypatch.setattr(speech_worker, "mark_job_failed", mark_failed)
    monkeypatch.setattr(speech_worker, "get_s3_audio_storage", lambda: Storage())
    monkeypatch.setattr(speech_worker, "_cleanup_job_dir", cleanup_then_fail)

    try:
        asyncio.run(
            speech_worker.process_speech_job(
                {
                    "job_id": "job-1",
                    "storage_provider": "s3",
                    "s3_bucket": "bucket",
                    "s3_object_key": "missing.wav",
                    "filename": "audio.wav",
                }
            )
        )
    except ValueError as error:
        assert "NoSuchKey" in str(error)
        assert "cleanup exploded" not in str(error)
    else:
        raise AssertionError("Missing S3 object must remain the raised error")

    assert cleanup_calls == 2
    assert failures and "NoSuchKey" in failures[0][1]


def test_missing_s3_bucket_is_permanent_and_marked_failed(monkeypatch):
    monkeypatch.setattr(settings, "STORAGE_PROVIDER", "s3")
    failures = []

    async def missing_result(job_id):
        return None

    async def mark_failed(job_id, error):
        failures.append((job_id, error))

    monkeypatch.setattr(speech_worker, "get_job_result", missing_result)
    monkeypatch.setattr(speech_worker, "mark_job_failed", mark_failed)

    try:
        asyncio.run(
            speech_worker.process_speech_job(
                {
                    "job_id": "job-1",
                    "storage_provider": "s3",
                    "s3_object_key": "audio.wav",
                    "filename": "audio.wav",
                }
            )
        )
    except ValueError as error:
        assert "s3_bucket" in str(error)
    else:
        raise AssertionError("Missing S3 bucket must be treated as a permanent failure")

    assert failures and failures[0] == ("job-1", "Speech S3 job is missing s3_bucket")


def test_queue_api_accepts_hmac_event_and_replays_duplicate(monkeypatch):
    monkeypatch.setattr(settings, "QUEUE_API_SERVICE_TOKEN", "")
    monkeypatch.setattr(settings, "QUEUE_API_HMAC_SECRET", "test-secret")
    monkeypatch.setattr(settings, "QUEUE_API_SIGNATURE_TOLERANCE_SECONDS", 300)

    class Redis:
        def __init__(self):
            self.records = {}
            self.xadded = []

        async def hgetall(self, key):
            return self.records.get(key, {})

        async def xadd(self, stream, fields):
            self.xadded.append((stream, fields))
            return "1-0"

        async def hset(self, key, mapping):
            self.records[key] = dict(mapping)

        async def expire(self, key, ttl):
            return True

    fake_redis = Redis()
    monkeypatch.setattr(queue_api, "redis_client", fake_redis)
    client = TestClient(queue_api.app)
    payload = {
        "eventId": "event-1",
        "eventType": "stt.requested",
        "correlationId": "conv-1",
        "userId": "user-1",
        "spaceId": "space-1",
        "conversationId": "conv-1",
        "payload": {"objectKey": "buddy/audio/user-1/space-1/conv-1/00000000-chunk.webm"},
    }
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    timestamp = str(int(time.time()))
    signature = hmac.new(b"test-secret", timestamp.encode("utf-8") + b"." + body, hashlib.sha256).hexdigest()
    headers = {"x-buddy-timestamp": timestamp, "x-buddy-signature": f"sha256={signature}"}

    first = client.post("/internal/events", content=body, headers=headers)
    second = client.post("/internal/events", content=body, headers=headers)

    assert first.status_code == 202
    assert first.json()["duplicate"] is False
    assert second.status_code == 202
    assert second.json()["duplicate"] is True
    assert len(fake_redis.xadded) == 1
    assert fake_redis.xadded[0][0] == settings.REDIS_STT_STREAM


def test_queue_api_accepts_audio_stream_target(monkeypatch):
    monkeypatch.setattr(settings, "QUEUE_API_SERVICE_TOKEN", "test-token")
    monkeypatch.setattr(settings, "QUEUE_API_HMAC_SECRET", "")

    class Redis:
        def __init__(self):
            self.records = {}
            self.xadded = []

        async def hgetall(self, key):
            return self.records.get(key, {})

        async def xadd(self, stream, fields):
            self.xadded.append((stream, fields))
            return "1-0"

        async def hset(self, key, mapping):
            self.records[key] = dict(mapping)

        async def expire(self, key, ttl):
            return True

    fake_redis = Redis()
    monkeypatch.setattr(queue_api, "redis_client", fake_redis)
    client = TestClient(queue_api.app)
    payload = {
        "eventId": "event-1",
        "eventType": "audio.ingested",
        "correlationId": "conv-1",
        "userId": "user-1",
        "spaceId": "space-1",
        "conversationId": "conv-1",
        "targetStream": settings.REDIS_AUDIO_STREAM,
        "payload": {"conversationId": "conv-1", "sequenceNumber": 0},
    }

    response = client.post(
        "/internal/events",
        json=payload,
        headers={"authorization": "Bearer test-token"},
    )

    assert response.status_code == 202
    assert response.json()["stream"] == settings.REDIS_AUDIO_STREAM
    assert fake_redis.xadded[0][0] == settings.REDIS_AUDIO_STREAM


def test_queue_api_accepts_legacy_speech_job(monkeypatch):
    monkeypatch.setattr(settings, "QUEUE_API_SERVICE_TOKEN", "test-token")
    monkeypatch.setattr(settings, "QUEUE_API_HMAC_SECRET", "")

    class Redis:
        def __init__(self):
            self.records = {}
            self.pushed = []

        async def hgetall(self, key):
            return self.records.get(key, {})

        async def hset(self, key, mapping):
            self.records[key] = dict(mapping)

        async def lpush(self, queue, payload):
            self.pushed.append((queue, json.loads(payload)))

        async def expire(self, key, ttl):
            return True

    fake_redis = Redis()
    monkeypatch.setattr(queue_api, "redis_client", fake_redis)
    client = TestClient(queue_api.app)
    payload = {
        "eventId": "event-1",
        "eventType": "speech.transcribe.requested",
        "correlationId": "job-1",
        "userId": "user-1",
        "spaceId": "space-1",
        "conversationId": "conv-1",
        "payload": {
            "job_id": "job-1",
            "filename": "audio.webm",
            "content_type": "audio/webm",
            "storage_provider": "s3",
            "s3_bucket": "bucket",
            "s3_object_key": "buddy/audio/audio.webm",
        },
    }

    response = client.post(
        "/internal/events",
        json=payload,
        headers={"authorization": "Bearer test-token"},
    )

    assert response.status_code == 202
    assert response.json()["stream"] == "speech_transcribe_queue"
    assert fake_redis.records["speech_job:job-1"]["status"] == "queued"
    assert fake_redis.pushed[0][1]["job_id"] == "job-1"


def test_queue_api_rejects_stale_hmac_timestamp(monkeypatch):
    monkeypatch.setattr(settings, "QUEUE_API_SERVICE_TOKEN", "")
    monkeypatch.setattr(settings, "QUEUE_API_HMAC_SECRET", "test-secret")
    monkeypatch.setattr(settings, "QUEUE_API_SIGNATURE_TOLERANCE_SECONDS", 30)
    client = TestClient(queue_api.app)
    body = b"{}"
    timestamp = str(int(time.time()) - 3600)
    signature = hmac.new(b"test-secret", timestamp.encode("utf-8") + b"." + body, hashlib.sha256).hexdigest()

    response = client.post(
        "/internal/events",
        content=body,
        headers={"x-buddy-timestamp": timestamp, "x-buddy-signature": f"sha256={signature}"},
    )

    assert response.status_code == 401


def test_duplicate_stt_event_does_not_call_sarvam(monkeypatch):
    class Repository:
        async def get_transcript_chunk(self, conversation_id, sequence_number):
            return TranscriptChunkDocument(
                conversationId=conversation_id,
                userId="user-1",
                spaceId="space-1",
                chunkId="chunk-1",
                sequenceNumber=sequence_number,
                sttStatus=STTStatus.COMPLETED,
            )

        async def mark_transcript_chunk_processing(self, *args, **kwargs):
            raise AssertionError("completed transcript chunks must not be marked processing")

    async def fail_transcribe(*args, **kwargs):
        raise AssertionError("completed transcript chunks must not be transcribed again")

    monkeypatch.setattr(conversation_workers, "ConversationRepository", lambda db: Repository())
    monkeypatch.setattr(conversation_workers, "get_database", lambda: object())
    monkeypatch.setattr(conversation_workers, "transcribe_from_path_with_fallback", fail_transcribe)

    event = EventEnvelope(
        eventType="stt.requested",
        correlationId="conv-1",
        userId="user-1",
        spaceId="space-1",
        conversationId="conv-1",
        payload={"conversationId": "conv-1", "sequenceNumber": 0},
    )

    asyncio.run(conversation_workers.handle_stt_event(event))
