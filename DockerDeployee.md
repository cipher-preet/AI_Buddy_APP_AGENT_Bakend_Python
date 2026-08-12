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

On the AWS host, create `.env.aws` from `.env.example`, then run:

```bash
docker compose -f docker-compose.aws.yml --env-file .env.aws build
docker compose -f docker-compose.aws.yml --env-file .env.aws up -d
```

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
