import asyncio
import base64
import json

import httpx
from fastapi.testclient import TestClient

from apps.api_gateway.config.setting import settings
from apps.api_gateway.workers import http_app, speech_worker, vector_worker
from services.speech.providers import sarvam_provider
from services.queue.pubsub import (
    InvalidPubSubEnvelope,
    InvalidPubSubPayload,
    PubSubMessagePublisher,
    decode_pubsub_push_envelope,
)
from services.storage.s3_audio_storage import (
    PermanentS3StorageError,
    build_audio_object_key,
    safe_temp_audio_path,
    sanitize_filename,
)


def envelope(payload, attributes=None):
    return {
        "message": {
            "data": base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii"),
            "messageId": "msg-1",
            "attributes": attributes or {},
        },
        "subscription": "projects/test/subscriptions/sub-1",
        "deliveryAttempt": 2,
    }


def test_valid_pubsub_envelope_decoding_preserves_fields():
    payload = {
        "job_id": "job-1",
        "user_id": "user-1",
        "space_id": "space-1",
        "request_id": "request-1",
        "file_path": "resources/audio_jobs/job-1.wav",
        "filename": "audio.wav",
        "content_type": "audio/wav",
        "transcript": "hello",
        "language_code": "en-IN",
        "nested": {"kept": True},
    }
    decoded = decode_pubsub_push_envelope(envelope(payload, {"event_type": "speech.transcription.requested"}))

    assert decoded.payload == payload
    assert decoded.message_id == "msg-1"
    assert decoded.attributes["event_type"] == "speech.transcription.requested"
    assert decoded.delivery_attempt == 2


def test_missing_message_rejected():
    try:
        decode_pubsub_push_envelope({})
    except InvalidPubSubEnvelope as error:
        assert "missing message" in str(error)
        return
    raise AssertionError("missing message must be rejected")


def test_missing_data_rejected():
    try:
        decode_pubsub_push_envelope({"message": {}})
    except InvalidPubSubEnvelope as error:
        assert "missing data" in str(error)
        return
    raise AssertionError("missing data must be rejected")


def test_invalid_base64_rejected():
    try:
        decode_pubsub_push_envelope({"message": {"data": "not base64!!!"}})
    except InvalidPubSubPayload as error:
        assert "base64" in str(error)
        return
    raise AssertionError("invalid base64 must be rejected")


def test_invalid_json_rejected():
    encoded = base64.b64encode(b"{").decode("ascii")
    try:
        decode_pubsub_push_envelope({"message": {"data": encoded}})
    except InvalidPubSubPayload as error:
        assert "JSON" in str(error)
        return
    raise AssertionError("invalid JSON must be rejected")


def test_duplicate_speech_job_does_not_call_processor(monkeypatch):
    async def completed(job_id):
        return {"status": "completed"}

    async def fail_transcribe(**kwargs):
        raise AssertionError("duplicate completed jobs must not be transcribed again")

    monkeypatch.setattr(speech_worker, "get_job_result", completed)
    monkeypatch.setattr(speech_worker, "sarvam_transcribe_from_path", fail_transcribe)

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


def test_speech_job_normalizes_pubsub_aliases():
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


def test_s3_speech_job_downloads_and_injects_file_path(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "STORAGE_PROVIDER", "s3")
    monkeypatch.setattr(settings, "S3_DELETE_AFTER_PROCESSING", False)

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
    monkeypatch.setattr(speech_worker, "sarvam_transcribe_from_path", transcribe)
    monkeypatch.setattr(speech_worker, "get_s3_audio_storage", lambda: Storage())
    monkeypatch.setattr(speech_worker, "safe_temp_audio_path", lambda job: tmp_path / "job-1" / "audio.wav")

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
    monkeypatch.setattr(speech_worker, "safe_temp_audio_path", lambda job: tmp_path / "job-1" / "audio.wav")

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


def test_speech_route_calls_processor_success(monkeypatch):
    monkeypatch.setattr(settings, "PUBSUB_VERIFY_PUSH_AUTH", False)
    called = []

    async def processor(payload):
        called.append(payload)

    monkeypatch.setattr(http_app, "process_speech_job", processor)
    client = TestClient(http_app.app)
    payload = {"job_id": "job-1", "user_id": "user-1", "space_id": "space-1"}

    response = client.post("/pubsub/speech", json=envelope(payload))

    assert response.status_code == 204
    assert called == [payload]


def test_temporary_processor_failure_returns_non_2xx(monkeypatch):
    monkeypatch.setattr(settings, "PUBSUB_VERIFY_PUSH_AUTH", False)

    async def processor(payload):
        raise RuntimeError("provider down")

    monkeypatch.setattr(http_app, "process_speech_job", processor)
    client = TestClient(http_app.app)

    response = client.post("/pubsub/speech", json=envelope({"job_id": "job-1"}))

    assert response.status_code == 500


def test_permanent_malformed_vector_payload_is_acked(monkeypatch):
    monkeypatch.setattr(settings, "PUBSUB_VERIFY_PUSH_AUTH", False)
    client = TestClient(http_app.app)

    response = client.post("/pubsub/vector", json=envelope({"not_job_id": "job-1"}))

    assert response.status_code == 204


def test_orchestration_route_selects_processing_handler(monkeypatch):
    monkeypatch.setattr(settings, "PUBSUB_VERIFY_PUSH_AUTH", False)
    called = []

    async def processing(event):
        called.append(event.eventType)

    monkeypatch.setattr(http_app, "handle_processing_event", processing)
    client = TestClient(http_app.app)
    payload = {
        "eventType": "conversation.processing.requested",
        "correlationId": "conv-1",
        "userId": "user-1",
        "spaceId": "space-1",
        "conversationId": "conv-1",
        "payload": {},
    }

    response = client.post(
        "/pubsub/orchestration",
        json=envelope(payload, {"event_type": "conversation.processing.requested"}),
    )

    assert response.status_code == 204
    assert called == ["conversation.processing.requested"]


def test_pubsub_publisher_sends_json_and_safe_attributes(monkeypatch):
    published = {}

    class Future:
        def result(self, timeout=None):
            published["timeout"] = timeout
            return "pubsub-message-1"

    class Client:
        def topic_path(self, project_id, topic):
            return f"projects/{project_id}/topics/{topic}"

        def publish(self, topic_path, data, **attributes):
            published["topic_path"] = topic_path
            published["data"] = data
            published["attributes"] = attributes
            return Future()

    publisher = PubSubMessagePublisher(project_id="project-1", timeout_seconds=3)
    publisher._client = Client()
    payload = {"job_id": "job-1", "user_id": "user-1", "space_id": "space-1", "secret": "kept-in-body"}

    message_id = asyncio.run(publisher.publish("speech-topic", payload, {"event_type": "speech.requested"}))

    assert message_id == "pubsub-message-1"
    assert published["topic_path"] == "projects/project-1/topics/speech-topic"
    assert json.loads(published["data"].decode("utf-8")) == payload
    assert published["attributes"]["job_id"] == "job-1"
    assert published["attributes"]["event_type"] == "speech.requested"
    assert "secret" not in published["attributes"]
