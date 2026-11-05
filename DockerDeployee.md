# Buddy AI Orchestration Deployment Guide

The public API runs on Google Cloud Run. Background processing runs on AWS with the Queue API, private Redis Streams, and Python workers.

## Cloud Run API

Build and deploy from this repository:

```bat
cd /d D:\AI_Personal_Buddy\AI_Orchestration
deploy.bat
```

The script:

1. Builds the shared Docker image.
2. Pushes it to Artifact Registry.
3. Deploys only `apps.api_gateway.main:app` to Cloud Run.

Cloud Run uses `cloud-run-api.env`. Set production secrets through a secure runtime mechanism before deployment.

## AWS Worker Stack

Do not run `docker compose build` on the EC2 instance. The instance is too small for `apt` and the Python dependency install. `queue-api` and `buddy-worker` share one image, `buddy-app:latest`. Build that image on this PC, copy it to the instance, then start the stack without a build.

On this PC, from `D:\AI_Personal_Buddy\AI_Orchestration`:

```bat
docker build -t buddy-app:latest .
docker save buddy-app:latest -o buddy-app.tar
scp buddy-app.tar ubuntu@3.7.237.120:~/buddy/buddy-app.tar
```

On the EC2 host, in `~/buddy`, after `.env.aws` is already there:

```bash
docker load -i buddy-app.tar
docker compose -f docker-compose.aws.yml --env-file .env.aws up -d --no-build
docker compose -f docker-compose.aws.yml --env-file .env.aws logs -f --tail=80 buddy-worker
```

`--no-build` is required. Without it, Compose sees `build: .` and compiles on the instance again. A cancelled `docker compose build` does not need to be resumed.

`.env.aws` stays on the server. It is not copied into the image. The worker must log `meeting ffmpeg: /usr/bin/ffmpeg`. If it logs `ffmpeg is not installed`, the loaded image is stale — rebuild it on this PC and `docker load` again.

The worker must log `meeting ffmpeg: /usr/bin/ffmpeg`. If it logs `ffmpeg is not installed`, the image is stale — rebuild with `--no-cache` as above.

### Stream job ownership

Long jobs keep a Redis lease so a slow transcription or meeting is not reclaimed at 15 seconds. Defaults are safe to leave unset:

```text
JOB_LEASE_TTL_SECONDS=90
JOB_LEASE_HEARTBEAT_SECONDS=20
JOB_COMPLETION_TTL_SECONDS=86400
```

`JOB_LEASE_HEARTBEAT_SECONDS` must be at most one third of `JOB_LEASE_TTL_SECONDS`. Reclaim waits at least the lease TTL even if `REDIS_CLAIM_IDLE_MS` is lower. `WORKER_GRACEFUL_SHUTDOWN_SECONDS` is how long in-flight jobs may finish after SIGTERM.

### Reminder / FCM (required when `FCM_ENABLED=true`)

1. Put the Firebase service-account JSON on the host at `apps/secrets/firebase-admin.json` (gitignored; not baked into the image).
2. In `.env.aws` set:
   - `FCM_ENABLED=true`
   - `FIREBASE_SERVICE_ACCOUNT_JSON=apps/secrets/firebase-admin.json`
3. Compose mounts `./apps/secrets` into the worker as `/app/apps/secrets`.

If `firebase-admin` is missing or credentials are wrong, the reminder worker logs a fatal config error and stops retrying (other workers keep running).

The AWS stack contains:

- `reverse-proxy`
- `queue-api`
- `redis`
- `buddy-worker`

Redis is private to the Docker network and must not expose port `6379` publicly.

## Verify

Cloud Run API:

```bat
gcloud run services describe buddy-ai-api --region asia-south1 --format="value(status.url)"
gcloud run services logs read buddy-ai-api --region asia-south1 --freshness 10m --limit 100
```

AWS Queue API:

```bash
curl https://queue-api.example.com/health/live
curl https://queue-api.example.com/health/ready
docker compose -f docker-compose.aws.yml --env-file .env.aws ps
```

## Important

The old Cloud Run worker deployment is removed from the local deployment script. Workers should be run on AWS so audio downloads, Sarvam calls, Redis Streams, MongoDB updates, and Qdrant writes happen outside Cloud Run.
