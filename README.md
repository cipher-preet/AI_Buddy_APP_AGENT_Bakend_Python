# Buddy Conversation Processing Service

Buddy keeps the public FastAPI API on Google Cloud Run and runs background processing on AWS. Audio bytes are uploaded directly from mobile clients to S3; Cloud Run handles authentication, presigned upload creation, upload registration, chat, status, and result reads.

## Runtime Entry Points

# For activating Enviroment 
PS D:\AI_Personal_Buddy\AI_Orchestration> (Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& d:\AI_Personal_Buddy\AI_Orchestration\venv\Scripts\Activate.ps1)

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
buddy:speech:jobs
buddy:vector:jobs
```

`SPEECH_VECTOR_QUEUE_TRANSPORT` defaults to `list`, which keeps `speech_transcribe_queue` and `completed_speech_queue` on LPUSH/BRPOP. Set it to `stream` only during the coordinated cutover. Do not drain those lists from worker startup. Inventory them with:

```powershell
python -m services.queue.legacy_list_migration
```

Moving the backlog requires `SPEECH_VECTOR_QUEUE_TRANSPORT=stream` and `LEGACY_QUEUE_MIGRATION_CONFIRM=yes` on that command. Stop list consumers first, set the flag on the Queue API and the worker together, run the migration, then start the stream consumers. Rollback copies pending entries, undelivered entries, retry-stream copies, and `:migrating` leftovers back to the lists. It pauses publishers until that copy finishes:

```powershell
$env:LEGACY_QUEUE_MIGRATION_CONFIRM='yes'
python -m services.queue.legacy_list_migration --export-to-lists
```

Goal jobs left `running` after a crash are recoverable once the Redis lease expires. `ready` and `failed` stay terminal. Queue API dedupe and the queue write share one Redis script. The dedupe key remains `queue_api:event:{eventId}` for one day. A repeated submission returns the existing publication. The same event id with a different payload returns 409. `SPEECH_VECTOR_QUEUE_TRANSPORT` stays `list` until its own cutover. Replace Queue API replicas together, then replace goal workers, so an old process cannot acknowledge an abandoned goal or publish around the script.

Speech and meeting file reads go through `services/runtime/file_io.py`, which uses the shared asyncio thread pool. That keeps the event loop free for lease heartbeats while a large file is read. Provider requests, worker concurrency, and the speech/vector transport flag are unchanged. Loading a full audio buffer for STT still uses memory proportional to the file. A local comparison is `python -m scripts.benchmark_worker_perf`.

A goal plan version is written before the job is marked `ready`. Recovery replays that write from `durableStep` and does not call the planner again. A redelivered `ready` job inserts a missing version only when that job's own durable step still has the plan; it does not replace a version that is already stored, and startup does not scan history. Research sessions for the same goal job upsert on `goalId` + `operationId`. Baseline `evidenceVersion` advances once per baseline job. The speech-job hash is written inside the queue publish script. Redis 7 does not roll a Lua script back after a successful command, so the script checks key types before `LPUSH` or `XADD`. The AWS compose file starts Redis with AOF and a 60-second RDB snapshot. Create the goal idempotency indexes with `ensure_idempotency_indexes` in maintenance before the new workers run. Do not run old and new goal workers together during the rollout.

Consumer groups:

```text
audio-workers
stt-workers
transcript-workers
finalization-workers
conversation-processing-workers
speech-workers
vector-workers
```

Workers use consumer groups, bounded retries, retry backoff, and dead-letter routing. MongoDB remains the durable source of truth.

A logical job has one active owner. The worker heartbeats a Redis lease and refreshes the stream pending entry without counting that refresh as a new delivery. Abandoned work is reclaimed only after the lease expires (`JOB_LEASE_TTL_SECONDS`, default 90). `REDIS_CLAIM_IDLE_MS` cannot make reclaim faster than that lease. A worker that loses the lease does not acknowledge the message and does not commit final artifacts.

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

## Goal consultant research

Goal workers can call Parallel Search (`POST https://api.parallel.ai/v1/search`) while building a plan.

```text
PARALLEL_API_KEY=...
PARALLEL_SEARCH_MODE=fast
PARALLEL_MAX_RESULTS=8
GOAL_SEARCH_BUDGET_PER_GOAL=4
GOAL_SEARCH_CACHE_TTL_SECONDS=86400
PARALLEL_EXTRACT_ENABLED=false
```

Do not put the API key in client apps, logs, or generated plans. Default unit tests mock Parallel and do not make paid calls.

Live smoke (opt-in):

```powershell
$env:PARALLEL_API_KEY="..."
python -m pytest tests/test_parallel_search_live.py -m integration
```

Full evaluation (fixture rubrics, not a production-ready stamp):

```powershell
python -m scripts.eval.goal_consultant_eval
```
