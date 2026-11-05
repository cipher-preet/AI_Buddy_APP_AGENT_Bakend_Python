# Phase 2E.1 staging environment template

Placeholders only. Generate real values in the operator's secret store. Do not copy the workstation `cloud-run-env.yaml` or `.env`.

Staging is a separate Cloud Run service, a separate compose project, a separate Redis volume, a separate MongoDB, and a separate S3 bucket. If any of those is the production resource, do not start the process.

## Shared secrets

| Name | Who | Rule |
| --- | --- | --- |
| `AUTH_TOKEN_SECRET` | Node and Python API | One new value, identical. Node signs with this when it is set, else `SESSION_SECRET`. Python verifies with the same precedence. |
| `SESSION_SECRET` | Node sessions. Python only if `AUTH_TOKEN_SECRET` is empty | Set `AUTH_TOKEN_SECRET` so the fallback is unused. |
| `QUEUE_API_SERVICE_TOKEN` | Node, Python, Queue API | One new value. Not `replace-with-long-random-token`. |
| `QUEUE_API_HMAC_SECRET` | same | One new value. Not the example placeholder. |

Invalid signatures fail closed. The Queue API token is not a user bearer.

## Node staging

```text
AUTH_TOKEN_SECRET=<new-shared-hmac-secret>
SESSION_SECRET=<new-session-secret>
MONGO_URI=mongodb://<staging-user>:<staging-password>@<staging-mongo-host>:27017/buddy?authSource=buddy
BUDDY_API_BASE=https://<staging-python-host>/api/v1
QUEUE_API_BASE_URL=https://<staging-queue-host>
QUEUE_API_SERVICE_TOKEN=<new-queue-token>
QUEUE_API_HMAC_SECRET=<new-queue-hmac>
REDIS_URL=redis://:<staging-redis-password>@<staging-redis-host>:6379/0
REMINDER_REDIS_URL=redis://:<password>@<staging-reminder-redis>:6379/0
GOAL_REDIS_URL=redis://:<password>@<staging-goal-redis>:6379/0
AWS_REGION=<region>
AWS_S3_BUCKET=<staging-bucket>
AWS_ACCESS_KEY_ID=<staging-key-id>
AWS_SECRET_ACCESS_KEY=<staging-key>
REDIS_STT_STREAM=buddy:stt:jobs
REDIS_MEETING_MERGE_STREAM=buddy:meeting:video-merge
REDIS_FINALIZATION_STREAM=buddy:conversation:finalization
REDIS_GOAL_INTERACTIVE_STREAM=buddy:goals:interactive
REDIS_GOAL_JOBS_STREAM=buddy:goals:jobs
```

`QUEUE_API_BASE_URL` must not be `http://` plus a public address. The process rejects that before it sends the bearer.

## Python API staging

```text
APP_ENV=staging
SERVICE_ROLE=api
QUEUE_PROVIDER=queue_api
QUEUE_API_BASE_URL=https://<staging-queue-host>
QUEUE_API_SERVICE_TOKEN=<new-queue-token>
QUEUE_API_HMAC_SECRET=<new-queue-hmac>
AUTH_TOKEN_SECRET=<new-shared-hmac-secret>
SESSION_SECRET=<new-session-secret>
MONGODB_URL=mongodb://<staging-user>:<staging-password>@<staging-mongo-host>:27017
MONGODB_DATABASE=buddy
SPEECH_VECTOR_QUEUE_TRANSPORT=list
```

Provider keys for Deepgram, Sarvam, and LLMs stay unset or point at a sandbox account. This template does not invent provider variable names beyond the ones in `.env.aws.example` (`DEEPGRAM_API_KEY`, `SARVAM_API_KEY`, `OPENAI_API_KEY`, and the other keys already listed there).

## Queue API and worker staging

Compose project name must not be the production project. New volume. `REDIS_PASSWORD` is new.

```text
APP_ENV=staging
QUEUE_API_DOMAIN=<staging-queue-hostname>
CADDY_ACME_EMAIL=<operator-email>
REDIS_PASSWORD=<new-redis-password>
REDIS_URL=redis://:<new-redis-password>@redis:6379/0
SPEECH_VECTOR_QUEUE_TRANSPORT=list
WORKER_CONCURRENCY=1
JOB_LEASE_TTL_SECONDS=90
JOB_LEASE_HEARTBEAT_SECONDS=20
QUEUE_API_SERVICE_TOKEN=<new-queue-token>
QUEUE_API_HMAC_SECRET=<new-queue-hmac>
MONGODB_URL=mongodb://<staging-user>:<staging-password>@<staging-mongo-host>:27017
MONGODB_DATABASE=buddy
AUTH_TOKEN_SECRET=<new-shared-hmac-secret>
STORAGE_PROVIDER=s3
S3_AUDIO_BUCKET=<staging-bucket>
AWS_REGION=<region>
AWS_ACCESS_KEY_ID=<staging-key-id>
AWS_SECRET_ACCESS_KEY=<staging-key>
FCM_ENABLED=false
```

`QUEUE_API_DOMAIN` is a hostname, not `:80`. Caddy then terminates TLS. Clients use `https://` that same name. Inside compose, `http://queue-api:8080` remains the proxy target and is not the client URL.

`WORKER_CONCURRENCY=1` matches the compose default. Do not add a second worker replica. Speech and vector consumers stay at concurrency 1 in code.

`REMINDER_REDIS_URL` and `GOAL_REDIS_URL` are not the compose Redis unless the operator writes down that they collapsed them on purpose. The current production-shaped files use three different Redis endpoints.

## Flag

`SPEECH_VECTOR_QUEUE_TRANSPORT=list` on the Queue API and the worker. The stream cutover stays in the Phase 2E runbook and is not part of the first staging start.

## Test users

Two synthetic accounts created in the staging database. No production user id.
