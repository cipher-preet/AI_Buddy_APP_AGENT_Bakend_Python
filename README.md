# Buddy Conversation Processing Service

Buddy keeps the public FastAPI API on Google Cloud Run and runs background processing on AWS. Audio bytes are uploaded directly from mobile clients to S3; Cloud Run handles authentication, presigned upload creation, upload registration, chat, status, and result reads.

## Runtime Entry Points

Public API:

```powershell
uvicorn apps.api_gateway.main:app --reload --host 0.0.0.0 --port 8000
```

AWS Queue API:

```powershell
uvicorn apps.api_gateway.queue_api:app --host 0.0.0.0 --port 8080
```

AWS worker:

```powershell
python -m apps.api_gateway.workers.main
```

## Conversation API

```text
POST /api/v1/conversations/start
POST /api/v1/conversations/{conversationId}/audio/upload-url
POST /api/v1/conversations/{conversationId}/audio/complete
POST /api/v1/conversations/{conversationId}/audio
POST /api/v1/conversations/{conversationId}/stop
GET  /api/v1/conversations/{conversationId}/status
```

The multipart `/audio` endpoint is retained for compatibility. New mobile clients should use the direct S3 upload flow.

## Direct Upload Flow

```text
Mobile app
  -> Cloud Run API: request upload URL
  -> S3: upload audio bytes directly
  -> Cloud Run API: register uploaded object metadata
  -> AWS Queue API: authenticated small event
  -> Redis Streams: XADD
  -> AWS worker: download from S3, transcribe, persist, finalize, process
```

Cloud Run should not receive, download, or forward audio bytes in the new flow.

## Queue Modes

Supported queue providers:

```text
redis
queue_api
```

Use `queue_api` for Cloud Run production so the API sends small HTTPS events to AWS. Use `redis` for local development or AWS-internal services that can reach Redis privately.

## Redis Streams

Streams:

```text
buddy:audio:ingestion
buddy:stt:jobs
buddy:transcript:ready
buddy:conversation:finalization
buddy:conversation:processing
buddy:conversation:retry
buddy:dead-letter
```

Consumer groups:

```text
audio-workers
stt-workers
transcript-workers
finalization-workers
conversation-processing-workers
```

Workers use consumer groups, stale pending-message recovery, bounded retries, retry backoff, and dead-letter routing. MongoDB remains the durable source of truth.

## Storage

Set:

```text
STORAGE_PROVIDER=s3
S3_AUDIO_BUCKET=your-audio-bucket
S3_AUDIO_PREFIX=buddy/audio
S3_PRESIGNED_URL_TTL_SECONDS=300
S3_MAX_AUDIO_SIZE_BYTES=26214400
```

S3 object keys are generated server-side and scoped by user, space, conversation, sequence number, and chunk id.

## Deployment

Cloud Run API:

```powershell
deploy.bat
```

AWS stack:

```bash
docker compose -f docker-compose.aws.yml --env-file .env.aws up -d --build
```

More details are in `DockerDeployee.md` and `AWS_QUEUE_MIGRATION.md`.

## Tests

```powershell
python -m pytest tests
```

Tests use mocks for paid services. Do not use real Sarvam, OpenAI, S3, MongoDB, or Qdrant credentials in tests.
