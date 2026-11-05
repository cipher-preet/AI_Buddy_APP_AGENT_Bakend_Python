# Phase 2E.2 staging environment

Templates only. The filled file is gitignored. No production value is copied into these files.

## Files

| File | Role |
| --- | --- |
| `deploy/staging/staging.env.example` | Placeholder env. Safe to read. Not a secret store. |
| `deploy/staging/staging.env` | Operator copy. Gitignored. Never created by this phase. |
| `docker-compose.staging.yml` | Project name `kukunotes-staging`. Does not start on its own. |
| `deploy/staging/Caddyfile` | Three hostnames. Queue paths limited. |
| `deploy/staging/cloud-run-*.staging.*` | Not deploy files. See the network document. |
| `scripts/staging/phase2e2_preflight.py` | Reads one local file. Refuses any other flag, including `--apply`. |

## Values the template pins

| Name | Staging value | Why this name |
| --- | --- | --- |
| `AUTH_TOKEN_SECRET` | new, shared by Node and Python | `utils/authToken.ts` and the Python verifier. Falls through to `SESSION_SECRET` only when the first is empty. Set both to the same new staging value so the fallback cannot pick up a different secret. |
| `SESSION_SECRET` | same new value | Node session cookie and the auth fallback. |
| `QUEUE_API_SERVICE_TOKEN` | a different new value | `MeetingExtension/config.ts` and the Queue API bearer check. |
| `QUEUE_API_HMAC_SECRET` | a different new value | HMAC of timestamp and body. Not the user token. |
| `QUEUE_API_BASE_URL` | `http://queue-api:8080` locally. `https://<operator queue name>` in cloud mode. | Transport check in `services/queue/transport_security.py` and `MeetingExtension/queueTransport.ts`. |
| `QUEUE_API_DOMAIN` | operator DNS, or `queue.example.invalid` in the template | Caddy site address. |
| `PYTHON_API_DOMAIN`, `NODE_API_DOMAIN` | operator DNS | Caddy site addresses. |
| `REDIS_URL` | `redis:6379/0` | Compose overwrites the worker and Queue API with this host even if the env file is edited. |
| `REMINDER_REDIS_URL` | `redis-reminder` | Reminder keys `buddy:reminder:schedule`, `processing`, `retry`, `dead-letter`. |
| `GOAL_REDIS_URL` | `redis-goals` | Goal streams. Not collapsed into the worker Redis. |
| `MINDMAP_REDIS_URL` | same staging reminder URL | Production falls back to the reminder URL when this is empty. Staging sets it so a leftover production URL cannot be inherited. |
| `MONGO_URI` | `staging-mongo:27017/buddy` | Node `Config/db.ts`. |
| `MONGODB_URL` | the same staging URI | Python settings. Worker rejects the exact localhost default. |
| `SPEECH_VECTOR_QUEUE_TRANSPORT` | `list` | Forced again on the worker, Queue API, Python API, and Node service. |
| `WORKER_CONCURRENCY` | `1` | Compose production default. Code default is 4. Staging does not raise it. |
| `JOB_LEASE_TTL_SECONDS` | `90` | Current default. |
| `JOB_LEASE_HEARTBEAT_SECONDS` | `20` | Current default. Heartbeat times three stays under the TTL. |
| `STORAGE_PROVIDER` | `local` | Python accepts only `local` or `s3`, and the S3 client has no custom endpoint. |
| `S3_BUCKET`, `S3_AUDIO_BUCKET`, `AWS_S3_BUCKET` | empty | Node treats any one of these as a configured bucket. Leave them empty until a new bucket exists. |
| `FCM_ENABLED` | `false` | No real push. |
| `DAILY_BRIEFING_ENABLED` | `true` | The scheduler can run against synthetic users. No paid model calls. |
| Provider API keys | empty | Tests use mocks. |

`QUEUE_PROVIDER` is `queue_api` for the Python API process and is forced to `redis` for `queue-api` and `buddy-worker`. That matches the production compose split.

## Password rules

Staging passwords are generated on the staging host. Use letters and digits only so the Mongo URI and Redis URL do not need extra encoding. The preflight never prints the values.

## What the operator changes for cloud mode

Copy the example to `deploy/staging/staging.env` on the staging host. Then set:

- `STAGING_ENV_FILE=deploy/staging/staging.env`
- `STAGING_MODE=cloud`
- `STAGING_BIND_IP` to the instance address that should receive 80 and 443
- `STAGING_HTTP_PORT=80` and `STAGING_HTTPS_PORT=443`
- the three DNS names and `CADDY_ACME_EMAIL`
- `QUEUE_API_BASE_URL=https://` plus the queue name
- new secrets, replacing every `replace-with` value

Run the preflight with `--expect ready` before any start command. A template file must fail that check.

## Commands that are validation, not a start

```
python scripts/staging/phase2e2_preflight.py --env deploy/staging/staging.env.example --expect template
docker compose --env-file deploy/staging/staging.env.example -f docker-compose.staging.yml config --services
```

`docker compose up` is not a validation command. `deploy.bat` is not a validation command.
