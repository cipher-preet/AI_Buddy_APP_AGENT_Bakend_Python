# Phase 2E architecture inventory

Inspection date: 2026-11-05. This is a read of repositories and deployment files. Live Cloud Run revisions, the EC2 host, and production Redis were not queried. Items marked UNVERIFIED were not visible from the files that were opened.

No secret values are recorded here.

## Repository commits

| Repository | Path | HEAD | What HEAD contains |
| --- | --- | --- | --- |
| AI_Orchestration | `d:\AI_Personal_Buddy\AI_Orchestration` | `d606a98` | The Phase 2 worker, queue, goal, and file-I/O changes are uncommitted modifications and untracked files on top of this commit. Deploying HEAD alone does not ship them. |
| Node_Backend | `d:\AI_Personal_Buddy\Node_Backend` | `6d77dfb` | Node auth, home ownership, and the daily-briefing proxy are in this tree. |
| BuddyDesktopApp | `d:\AI_Personal_Buddy\BuddyDesktopApp` | `cffd49c` | Web dashboard and Electron renderer. API client sets `Authorization: Bearer` when a token exists. |
| AIAssistantApp | `d:\AI_Personal_Buddy\AIAssistantApp` | `244a32f` | React Native Android app. `baseApi` sets `Authorization: Bearer` when a token exists. |
| chromeExtension | `d:\AI_Personal_Buddy\chromeExtension` | `9250437` | Meeting recorder calls the API with `Authorization: Bearer`. |
| Buddywebsite | `d:\AI_Personal_Buddy\Buddywebsite` | not recorded | No `BUDDY_API`, `buddy-ai-api`, or `Authorization` usage was found. Out of scope for this release. |

## Service table

| Service | Runtime | Deployment target | Dependencies | Environment source | Rollout dependency |
| --- | --- | --- | --- | --- | --- |
| Node API | Node 22, Express, TypeScript. Container `node:22-alpine`, `npm start`, port 5000. | Google Cloud Run service `buddy-ai-api` in `asia-south1`, as named in `DockerDeployee.md`. The live revision was not queried. UNVERIFIED. | MongoDB database `buddy`, session store, S3, the Queue API, Python `BUDDY_API_BASE`, reminder Redis, goal Redis. | Gitignored `Node_Backend/cloud-run-env.yaml` on the workstation. Not in the git index. Cloud Run may also have a console revision that differs. UNVERIFIED. | Must be able to sign the same HMAC bearer Python verifies before Python auth is relied on. |
| Python API | Python 3.12, FastAPI, uvicorn `apps.api_gateway.main:app`, port 8080 (`Dockerfile`). | Google Cloud Run, same service name in the deploy notes. `SERVICE_ROLE=api`. Live revision UNVERIFIED. | MongoDB `buddy`, user-token secret, Queue API for internal publish, provider credentials. | `.env` locally. Cloud Run env was not read from Google. | Rejects user calls with no bearer, a bad signature, a wrong secret, or an expired token. Deploy in the same window as Node, after Node can forward or mint that bearer. |
| Queue API | Same image, command `uvicorn apps.api_gateway.queue_api:app` port 8080. | AWS EC2 via `docker-compose.aws.yml` service `queue-api`. Caddy 2.8 publishes ports 80 and 443 and proxies `/internal/events` and `/health/*`. | Compose Redis database 0, service token and/or HMAC secret. | `env_file: .env.aws`, then compose overrides `SERVICE_ROLE`, `QUEUE_PROVIDER`, and `REDIS_URL`. | Publish path used by Node and the Python API. New atomic publisher must not run beside an old check-then-publish replica. |
| Worker | Same image, command `python -m apps.api_gateway.workers.main`. | AWS EC2 service `buddy-worker`. | Compose Redis database 0, MongoDB, S3, ffmpeg, optional `./apps/secrets` for FCM. | `.env.aws`, then compose forces Redis database 0 and `WORKER_CONCURRENCY` default 1. | Must be the only goal/speech consumer on that Redis. Old and new goal workers together are unsupported. |
| Redis (worker and Queue API) | `redis:7-alpine`. | Compose service `redis`. Port 6379 is exposed only inside the compose network. | Volume `redis_data`. Password from `REDIS_PASSWORD`. | Compose command, not the application env URL. | Queue API and worker must use this instance. Node must not be given this URL by accident, and this instance must not be the production instance when the deploy is staging. |
| Reminder Redis | Hosted Redis. Key `REMINDER_REDIS_URL`. | Separate from the compose Redis. Port class seen in local settings is not the compose port. | Reminder schedule keys. | Node env file. | Not the speech/goal queue. Do not point the worker at it. |
| Goal Redis | Hosted Redis. Key `GOAL_REDIS_URL`. | Separate host from both compose Redis and reminder Redis. | Goal coordination keys used by Node. | Node env file. | Not the worker stream Redis unless an operator proves they are the same instance. The files say they are not. |
| MongoDB | Application database name `buddy` on both Node (`MONGO_URI` path) and Python (`MONGODB_DATABASE` default `buddy`). | A remote host in the Node env file. Python worker role rejects a localhost URL. Live topology UNVERIFIED beyond that file. | Collections listed below. | Node `MONGO_URI`. Python `MONGODB_URL` / `MONGODB_URI`. | Staging needs its own database. The name may stay `buddy` only on that isolated server. |
| Object storage | S3, region variable `AWS_REGION`. | Bucket named by `AWS_S3_BUCKET` in the Node env file, with a CloudFront distribution variable. | Meeting chunks, speech audio, documents. | Node env and Python storage settings. | Staging needs another bucket or a prefix that production workers do not read. |
| Desktop / web | Electron renderer in BuddyDesktopApp. | Installed clients plus the web dashboard build. | Node API base URL. | Client configuration. | Current source sends a bearer token when one is stored. A binary that never sends it will receive 401 on protected routes. |
| Android | React Native, AIAssistantApp. | Installed APKs. | Node API. | Client configuration. | Current source sends a bearer token. Older installed builds were not executed. |
| Chrome extension | `buddy-meeting-recorder`. | Installed extension. | Node upload API with bearer token. | Extension connection state. | Current source sends the connection token. |

## Process and event path

```text
Desktop / Android / Chrome extension
        |  bearer user token, or a Node session cookie
        v
Node API (Cloud Run) ---- MongoDB buddy ---- S3
        |  forwards the verified bearer, or mints a 10-minute session token
        |  Queue API calls use the service token and HMAC, not the user token
        v
Python API (Cloud Run) ---- MongoDB buddy
        |
        |  POST /internal/events
        v
Caddy --> Queue API ---- Redis db 0 (compose)
                              |
                              v
                         buddy-worker ---- MongoDB buddy
                                       \-- S3
                                       \-- speech / meeting providers
```

Reminder delivery uses `REMINDER_REDIS_URL`. Node goal features that read `GOAL_REDIS_URL` use the other hosted Redis. Those two are not the compose Redis that the worker consumes.

## Redis names (code defaults)

Streams and groups:

| Stream | Group |
| --- | --- |
| `buddy:audio:ingestion` | `audio-workers` |
| `buddy:stt:jobs` | `stt-workers` and, for meeting video, `meeting-video-workers` |
| `buddy:transcript:ready` | `transcript-workers` |
| `buddy:conversation:window-extraction` | `window-extraction-workers` |
| `buddy:conversation:finalization` | `finalization-workers` |
| `buddy:conversation:processing` | `conversation-processing-workers` |
| `buddy:conversation:retry` | retry stream |
| `buddy:dead-letter` | dead letter |
| `buddy:daily-briefing:jobs` | `daily-briefing-workers` |
| `buddy:meeting:video-merge` | `meeting-merge-workers` |
| `buddy:mindmap:jobs` | `mindmap-workers` |
| `buddy:document:jobs` | `document-workers` |
| `buddy:speech:jobs` | `speech-workers` |
| `buddy:vector:jobs` | `vector-workers` |
| `buddy:goals:interactive` | `goal-interactive-workers` |
| `buddy:goals:jobs` | `goal-workers` |
| `buddy:conversation:schedule-extraction` | `schedule-extraction-workers` |

Lists used while `SPEECH_VECTOR_QUEUE_TRANSPORT=list`:

- `speech_transcribe_queue`
- `completed_speech_queue`
- `speech_transcribe_queue:migrating`
- `speech_transcribe_queue:migration_failed`

Other keys: `queue_api:event:{eventId}` dedupe hash, `speech_job:{jobId}` hash, `buddy:job-lease:*`, `buddy:job-done:*`, `buddy:job-fence:global`, `buddy:speech-vector:export-lock`, reminder keys `buddy:reminder:schedule`, `buddy:reminder:processing`, `buddy:reminder:retry`, `buddy:reminder:dead-letter`.

Mindmap, document, and goal streams have a default `MAXLEN` of 200. The conversation streams do not set that default in `setting.py`.

## Shared configuration

These values must be the same pair of services, and must be different from production when the environment is staging:

| Shared item | Who must match |
| --- | --- |
| HMAC bearer secret | Node `AUTH_TOKEN_SECRET` or else `SESSION_SECRET`, and Python `AUTH_TOKEN_SECRET` or else `SESSION_SECRET`. The effective values must be equal. |
| Queue service token and HMAC | Node `QUEUE_API_SERVICE_TOKEN` / `QUEUE_API_HMAC_SECRET` and the Queue API process. |
| Redis | Queue API and worker. Compose replaces both with `redis://:<password>@redis:6379/0`. |
| MongoDB database | Node, Python API, and worker, one staging database. |
| Stream names and group names | Queue API publishers and the worker. Defaults above. |
| `SPEECH_VECTOR_QUEUE_TRANSPORT` | Queue API and worker. Both stay `list` for the first staging deploy. |
| S3 bucket | Node uploads and the worker that reads them. |

## Startup commands

- Python API image: `uvicorn apps.api_gateway.main:app` on port 8080.
- Queue API: `uvicorn apps.api_gateway.queue_api:app` on port 8080.
- Worker: `python -m apps.api_gateway.workers.main`.
- Node image: `npm start` on port 5000.
- Compose Redis: `redis-server --appendonly yes --save 60 1 --requirepass <password>`. `appendfsync` is not set, so Redis uses `everysec`. `maxmemory` is not set.

## Evidence gaps

- The current Cloud Run revision, its env vars, and its image digest were not read.
- `.env.aws` contents were not printed. Compose shows which variables it overrides.
- EC2 Redis persistence was not queried. Only the compose command and a local disposable Redis on `127.0.0.1:6399` were inspected.
- Installed Android and Electron binaries were not launched.
