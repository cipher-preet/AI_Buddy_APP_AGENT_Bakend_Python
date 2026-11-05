# Phase 2E.2 infrastructure options

Date: 2026-11-05. Nothing in this document was deployed. Production hosts were not queried. Live revisions, the real EC2 size, and live DNS are UNVERIFIED.

## What the repositories actually deploy

| Component | Current production topology | Staging requirement | Isolation method |
| --- | --- | --- | --- |
| Node API | `Node_Backend/deploy.bat` deploys Cloud Run service `buddy-node-backend` in project `buddy-505709`, region `asia-south1`, image in Artifact Registry repo `buddy-backend-repo`. Flags present in the script: `--min-instances 0`, `--cpu-throttling`, `--allow-unauthenticated`, port 5000 from the Dockerfile. No memory or CPU flag. Live revision UNVERIFIED. | `node-api` on the staging VM, reached only through Caddy. A later Cloud Run service is a different name and is not the first step. | New process, new `AUTH_TOKEN_SECRET` / `SESSION_SECRET`, staging `MONGO_URI` only. |
| Python API | `AI_Orchestration/deploy.bat` deploys Cloud Run service `buddy-ai-api` in project `python-microservice-hub`, region `asia-south1`. Flags: `--memory 1Gi`, `--cpu 1`, `--timeout 300`, `--concurrency 80`, `--min-instances 0`, `--max-instances 5`, port 8080. The script does not pass `--cpu-throttling`, so the live CPU allocation mode is UNVERIFIED. `DockerDeployee.md` describes this service, not the Node service. | `python-api` on the same staging VM. | Same new user-token secret as Node. Staging `MONGODB_URL` only. |
| Queue API | `docker-compose.aws.yml` service `queue-api`. `deploy/aws/Caddyfile` publishes `/internal/events` and `/health/*` for `{$QUEUE_API_DOMAIN}`. `AWS_QUEUE_MIGRATION.md` describes one EC2 or Lightsail in `ap-south-1`, initially 2 vCPU / 4 GB / 25–30 GB, security group 80/443 public and 22 admin-only, Redis port not public. The live instance size is UNVERIFIED. | New VM, compose project `kukunotes-staging`, Caddy only after the operator supplies DNS names. | Different host, volume names, passwords, and tokens. |
| Python worker | Compose service `buddy-worker`, command `python -m apps.api_gateway.workers.main`, `SERVICE_ROLE=worker`. Compose default `WORKER_CONCURRENCY` is 1. The Settings default in code is 4. Staging keeps 1. | One replica. Staging tmpfs is 256 MB, smaller than the production compose tmpfs of 1 GB, so large audio fixtures need a bigger staging volume only if a test requires it. | `REDIS_URL` is forced to the staging `redis` service. |
| Worker Redis | Compose `redis:7-alpine`, `--appendonly yes`, `--save 60 1`, `--requirepass`, volume `redis_data`, port 6379 not published. Queue API and worker are forced to database 0. | Volume `staging_redis_data`. No host port. | Compose project name `kukunotes-staging` plus a distinct volume name. |
| Reminder Redis | `REMINDER_REDIS_URL`. Not the compose Redis. Mind maps use `MINDMAP_REDIS_URL` and otherwise fall back to the reminder URL (`services/mindmap/redis_client.py`). Live host UNVERIFIED and not contacted. | Container `redis-reminder`. `MINDMAP_REDIS_URL` is set to that same staging container so the production fallback pairing stays, on staging data. | Own password and volume. |
| Goal Redis | `GOAL_REDIS_URL`. Separate from the worker Redis. Goal streams `buddy:goals:interactive` and `buddy:goals:jobs` use this client. Live host UNVERIFIED. | Container `redis-goals`. | Own password and volume. |
| MongoDB | Application setting `MONGODB_URL` / Node `MONGO_URI`. The worker rejects a blank URL and the exact string `mongodb://localhost:27017`. `mongodb://staging-mongo:27017/...` passes. Live cluster UNVERIFIED and not contacted. | `mongo:7` as `staging-mongo`, database `buddy`, root user created by the image. No host port. | New user and password. No production dump. |
| Object storage | Python `services/storage/s3_audio_storage.py` builds a boto3 client with no custom endpoint, so MinIO is not a Python option without a code change. Node `Config/s3.ts` reads `AWS_S3_BUCKET`, `S3_AUDIO_BUCKET`, or `S3_BUCKET`, and can take `AWS_S3_ENDPOINT`. | `STORAGE_PROVIDER=local` and empty bucket variables until an operator creates a new bucket. | The new bucket, if created later, must not be able to read production objects. |
| Auth secrets | Both sides use `AUTH_TOKEN_SECRET`, then `SESSION_SECRET`. Queue API auth is a different pair: `QUEUE_API_SERVICE_TOKEN` and `QUEUE_API_HMAC_SECRET`. | New staging values. The user secret and the queue token must differ. | Generated on the staging host. Never copied from a production env file. |

`docker-compose.local.yml` is only Redis and publishes host port 6379. It is not a staging stack. `docker-compose.aws.yml` is the production-shaped stack and must not be started for staging.

## Option A — local integration staging

`docker-compose.staging.yml` with `STAGING_MODE=local`. Caddy binds `127.0.0.1:8088` and `127.0.0.1:8443`. Node, Python, the Queue API, the worker, three Redis 7 containers, MongoDB 7, and Qdrant share the compose network. `QUEUE_API_BASE_URL=http://queue-api:8080` is a single-label Docker name, which the transport check allows. Redis and Mongo publish no host ports.

This is the daily correctness loop. It does not reproduce Cloud Run cold starts, a public certificate, or the production EC2 memory ceiling. A passing local run is not staging acceptance.

## Option B — minimal cloud staging

One new EC2 instance in `ap-south-1`, type `t3.medium` (2 vCPU, 4 GiB), stopped between sessions. The same compose file runs there with `STAGING_MODE=cloud`, `STAGING_BIND_IP` set to the instance address, and ports 80 and 443. Caddy obtains certificates only after the operator points three DNS names at that address.

Node and Python stay on that VM for this phase. They need MongoDB, goal Redis, and reminder Redis. Those ports stay closed. Putting the APIs on Cloud Run would require either publishing Redis and MongoDB or adding a VPC connector. Publishing them is rejected. The connector is extra cost and is not required to test HTTPS, service authentication, or worker recovery.

`t3.small` (2 GiB) is not the host. The worker, MongoDB, three Redis processes, two APIs, Caddy, and Qdrant do not fit beside a 256 MB tmpfs with usable headroom. `t3.micro` is smaller still.

### The production EC2 is not the staging host

Sharing it is rejected even with a second compose project:

- Production Caddy already binds 80 and 443. The staging cloud profile needs those ports for a real certificate.
- `AWS_QUEUE_MIGRATION.md` sizes that host for the production queue and worker, initially 4 GB. A second MongoDB, three Redis processes, and two API containers on that host can stall the production worker. The live size was not measured.
- The production volume is `redis_data` in the default compose project. The staging file uses project `kukunotes-staging` and `staging_redis_data`. Different names are necessary and still not sufficient: a port clash or an out-of-memory kill hits the production process on the same kernel.
- Container names do not isolate disks, credentials, or the host firewall.

## Recommendation

Use Option A on the developer machine for repeated tests. Use one stopped-between-sessions `t3.medium` for the cross-service HTTPS and crash-recovery acceptance. Do not share the production EC2. Do not create the Cloud Run staging services until a private path to MongoDB and Redis exists.

Speech/vector transport stays `list`. Worker concurrency stays 1. One worker replica.
