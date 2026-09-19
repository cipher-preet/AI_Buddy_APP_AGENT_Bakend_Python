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

On the AWS host, create `.env.aws` from `.env.aws.example`, then run:

```bash
docker compose -f docker-compose.aws.yml --env-file .env.aws build
docker compose -f docker-compose.aws.yml --env-file .env.aws up -d
```

After pulling new code, rebuild the worker so Python deps (including `firebase-admin`) and FFmpeg are installed. Pip can sit on `Collecting cryptography` / `boto3` for a few minutes on a small EC2 — do not Ctrl+C.

Prefer a dedicated build so Redis/Caddy are not competing for RAM:

```bash
docker compose -f docker-compose.aws.yml --env-file .env.aws build --no-cache buddy-worker
docker compose -f docker-compose.aws.yml --env-file .env.aws up -d
docker compose -f docker-compose.aws.yml --env-file .env.aws logs -f --tail=80 buddy-worker
```

The worker must log `meeting ffmpeg: /usr/bin/ffmpeg`. If it logs `ffmpeg is not installed`, the image is stale — rebuild with `--no-cache` as above.

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
