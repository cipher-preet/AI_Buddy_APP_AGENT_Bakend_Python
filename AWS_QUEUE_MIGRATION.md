# Buddy AWS Queue Migration

This migration keeps the public FastAPI API on Cloud Run and moves audio bytes plus background processing to AWS. Cloud Run requests presigned S3 upload URLs, receives only upload-complete metadata, and publishes small authenticated HTTPS events to the AWS Queue API.

## Required AWS Infrastructure

- One x86 EC2 or Lightsail instance in `ap-south-1`, initially 2 vCPU, 4 GB RAM, encrypted 25-30 GB disk.
- One private S3 bucket for temporary audio objects.
- DNS record for the Queue API domain pointing at the instance.
- Security group inbound rules: `80/tcp` and `443/tcp` from the internet, `22/tcp` only from trusted admin IPs.
- No inbound rule for Redis `6379`.

Use x86 `t3` or `t3a` first. Treat ARM `t4g` as unverified until all Python, audio, and crypto dependencies are tested on ARM.

## Environment

Create `.env.aws` on the AWS host from `.env.example`. Do not commit it.

Required AWS-side values:

```text
SERVICE_ROLE=worker
REDIS_PASSWORD=<long random password>
REDIS_URL=redis://:<REDIS_PASSWORD>@redis:6379/0
MONGODB_URL=<mongodb connection string>
STORAGE_PROVIDER=s3
AWS_ACCESS_KEY_ID=<least privilege key>
AWS_SECRET_ACCESS_KEY=<least privilege secret>
AWS_REGION=ap-south-1
S3_AUDIO_BUCKET=<bucket>
QUEUE_API_SERVICE_TOKEN=<long random token>
QUEUE_API_HMAC_SECRET=<long random hmac secret>
SARVAM_API_KEY=<sarvam key>
QDRANT_URL=<qdrant url>
```

Cloud Run API values after cutover:

```text
SERVICE_ROLE=api
QUEUE_PROVIDER=queue_api
QUEUE_API_BASE_URL=https://queue-api.example.com
QUEUE_API_SERVICE_TOKEN=<same token>
QUEUE_API_HMAC_SECRET=<same hmac secret>
STORAGE_PROVIDER=s3
S3_AUDIO_BUCKET=<bucket>
S3_PRESIGNED_URL_TTL_SECONDS=300
```

## Start AWS Stack

Manual commands:

```bash
docker compose -f docker-compose.aws.yml --env-file .env.aws build
docker compose -f docker-compose.aws.yml --env-file .env.aws up -d
docker compose -f docker-compose.aws.yml --env-file .env.aws ps
```

Health checks:

```bash
curl https://queue-api.example.com/health/live
curl https://queue-api.example.com/health/ready
docker compose -f docker-compose.aws.yml --env-file .env.aws exec redis redis-cli -a "$REDIS_PASSWORD" XINFO STREAM buddy:stt:jobs
```

Redis is exposed only on the internal Docker network. The Compose file intentionally has no public `6379` port mapping, enables AOF, and stores data in a named volume.

## Cloud Run API

The Cloud Run API should continue to run only:

```text
uvicorn apps.api_gateway.main:app --host 0.0.0.0 --port 8080
```

Manual update example:

```bash
gcloud run services update buddy-ai-api \
  --region asia-south1 \
  --set-env-vars SERVICE_ROLE=api,QUEUE_PROVIDER=queue_api,QUEUE_API_BASE_URL=https://queue-api.example.com,STORAGE_PROVIDER=s3
```

Set secrets through Secret Manager or your existing secure env workflow. Do not place secret values in shell history.

## Mobile Integration

1. Call `POST /api/v1/conversations/{conversationId}/audio/upload-url`.
2. Upload the audio bytes directly to the returned S3 `uploadUrl` with the returned `Content-Type` header.
3. Call `POST /api/v1/conversations/{conversationId}/audio/complete` with the returned `objectKey`, `chunkId`, `sequenceNumber`, `contentType`, and `sizeBytes`.
4. Continue using the existing stop/status endpoints.

The legacy multipart `/audio` endpoint remains available during migration.

## Migration Order

1. Audit current implementation.
2. Implement direct S3 upload while preserving the old endpoint.
3. Implement Queue API.
4. Harden Redis consumers.
5. Add Docker Compose.
6. Run local integration tests.
7. Deploy AWS stack manually after approval.
8. Test with a staging conversation.
9. Switch Cloud Run API publishing to AWS Queue API.
10. Verify MongoDB and Qdrant results.
11. Verify no duplicate processing.
12. Verify Cloud Run sent bytes decrease.
13. Verify the old Cloud Run worker service is no longer receiving traffic.
14. Keep the legacy multipart upload endpoint temporarily for client rollback.

## Rollback

- Set Cloud Run `QUEUE_PROVIDER=redis` only for local or private-network emergency testing.
- Keep the legacy multipart upload endpoint available for temporary client rollback.
- For production rollback, point mobile clients back to the multipart endpoint and run workers where they can reach the configured Redis instance.

## Monitoring

Track Queue API accepts/duplicates, Redis stream length and pending counts, dead-letter count, STT latency, LLM latency, S3 downloaded bytes, and Cloud Run response egress. Logs should include identifiers such as `eventId`, `jobId`, `correlationId`, `conversationId`, `sequenceNumber`, stage, attempt, duration, and failure class, but not presigned URLs, tokens, transcripts, or audio content.
