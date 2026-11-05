# Phase 2E environment matrix

No passwords, tokens, or connection strings are stored in this file. Values below are names, defaults, and the safe staging behavior.

The workstation file `Node_Backend/cloud-run-env.yaml` is gitignored and is not a staging template. Do not copy it. It selects the existing MongoDB database `buddy`, a localhost `REDIS_URL`, separate hosted reminder and goal Redis endpoints, the existing S3 bucket, and an `http://` Queue API base URL. `AUTH_TOKEN_SECRET` is not a key in that file. `QUEUE_API_SERVICE_TOKEN` and `QUEUE_API_HMAC_SECRET` are the example placeholders from `.env.example`, not generated secrets.

## Authentication

Node `utils/authToken.ts` and Python `services/auth/node_token.py` implement the same contract:

- Body is `base64url(json) + "." + base64url(hmac_sha256)`. This is not a JWT.
- Signed bytes are the encoded payload segment.
- JSON key order is `userId`, `provider`, `iat`, `exp`, with no extra spaces.
- Default lifetime is 30 days when the caller does not pass an expiry.
- Python rejects a token whose `exp` is less than `floor(time)`.
- A missing secret raises. Node throws from `createAuthToken`. Python raises `AuthNotConfigured`. Neither side treats an unsigned payload as a user.
- Precedence on both sides is `AUTH_TOKEN_SECRET`, then `SESSION_SECRET`.

Safe staging rule: set `AUTH_TOKEN_SECRET` to one new staging value on Node and on the Python API. Do not rely on `SESSION_SECRET` being the only key on one side and a different `AUTH_TOKEN_SECRET` on the other. A mismatch fails closed: Python returns 401. It does not authorize the request.

The daily-briefing proxy (`Buddy/dailyBriefingAuth.ts`) forwards a presented bearer only after it verifies as the authenticated user. A session with no bearer gets a token that expires in 600 seconds and is not returned to the browser. The body `userId` is not an input.

Queue API authentication is separate (`apps/api_gateway/queue_api.py`):

- If neither service token nor HMAC secret is set, the process returns 401.
- If the service token is set, the `Authorization` header must equal `Bearer <service token>` by a constant-time compare of the whole header.
- If the HMAC secret is set, `x-buddy-timestamp` and `x-buddy-signature` must match, inside `QUEUE_API_SIGNATURE_TOLERANCE_SECONDS`.
- A user HMAC bearer does not satisfy the service-token compare, and a service token does not satisfy `require_user`.

## Redis

| Variable | Who reads it | Safe staging behavior |
| --- | --- | --- |
| `REDIS_URL` | Python API, and Node for some meeting publishes. | Staging URL for that service only. Compose overrides the Queue API and the worker to `redis:6379/0` and ignores a host URL in `.env.aws` for those two services. |
| `REDIS_PASSWORD` | Compose Redis and the override URL. | A new password. Required by the compose command. |
| `REMINDER_REDIS_URL` | Node reminders. | A staging reminder instance. Node tests show reminder code ignores `REDIS_URL` when this is set. |
| `GOAL_REDIS_URL` | Node goals. | A staging goal instance, not the worker Redis, unless an operator deliberately collapses them and documents it. |
| `JOB_LEASE_TTL_SECONDS` | Worker. Code default 90, minimum 30. | Leave at 90 unless a measured change is approved. |
| `JOB_LEASE_HEARTBEAT_SECONDS` | Worker. Code default 20. Must be at most one third of the lease TTL. | Leave at 20. |
| `REDIS_CLAIM_IDLE_MS` | Reclaim. Code default 15000. | Do not tune in this release. |
| `SPEECH_JOB_TTL_SECONDS` | Speech job hash. Code default 86400. | Leave at the default. |
| `WORKER_CONCURRENCY` | Worker. Code default 4. Compose default `${WORKER_CONCURRENCY:-1}`. | Leave the compose default at 1. Do not add replicas. |
| Speech and vector stream consumers | Hardcoded `concurrency=1` in the speech and vector workers. | Do not change. |

Lease keys use prefix `buddy:job-lease:`. Completion keys use `buddy:job-done:`. The fence counter `buddy:job-fence:global` has no TTL in the application code. Losing it and starting again at a lower value would let a stale owner write. Staging Redis must be persistent and must not be flushed.

## MongoDB

| Variable | Who | Safe staging behavior |
| --- | --- | --- |
| `MONGO_URI` | Node. | A staging URI. Database name may be `buddy` only if the server is not the production server. |
| `MONGODB_URL` / `MONGODB_URI` | Python. Default URL `mongodb://localhost:27017`. If `MONGODB_URI` is set and the URL is still the localhost default, the URI replaces it. | Staging URI. Worker role refuses a blank or localhost URL. |
| `MONGODB_DATABASE` | Python. Default `buddy`. | Same database Node uses, on the staging server. |

The worker does not call `ensure_idempotency_indexes`. Creating those indexes is a maintenance step in the runbook.

## Storage and providers

| Concern | Staging behavior |
| --- | --- |
| S3 `AWS_S3_BUCKET`, `AWS_REGION`, access keys | New bucket or a prefix no production worker lists. New keys with access only to that bucket. |
| Deepgram, Sarvam, LLM, embeddings | Unset, or sandbox keys that cannot bill the production account. This audit did not call them. |
| Email, SMS, Razorpay, Zoho, FCM | Disabled or sandbox. `FCM_ENABLED` should stay false unless a test project is used. The worker mounts `./apps/secrets` only for FCM. |

## Feature flags

| Flag | Required value for the first staging deploy |
| --- | --- |
| `SPEECH_VECTOR_QUEUE_TRANSPORT` | `list`. Code default is `list`. Allowed values are `list` and `stream` only. |
| `QUEUE_PROVIDER` | `redis` on the Queue API and the worker, as compose sets it. |
| `SERVICE_ROLE` | `api` on Cloud Run, `queue_api` on the Queue API, `worker` on the worker. |

Do not set the speech/vector transport to `stream` during the consolidated deploy. The separate cutover is in the runbook and stays unauthorized until the list-mode staging run is accepted.

## Matrix

| Variable | Node | Python API | Queue API | Worker | Must match? | Safe staging value / behavior |
| --- | --- | --- | --- | --- | --- | --- |
| `AUTH_TOKEN_SECRET` | Signs if set, else `SESSION_SECRET` | Verifies if set, else `SESSION_SECRET` | No | No | Yes, the effective secret with the Python API | One new secret on Node and Python API |
| `SESSION_SECRET` | Cookie session, and HMAC fallback | HMAC fallback only | No | No | Fallback must not disagree with `AUTH_TOKEN_SECRET` | New session secret. Set `AUTH_TOKEN_SECRET` explicitly so fallback is unused |
| `QUEUE_API_SERVICE_TOKEN` | Sends `Bearer` | Sends `Bearer` when it publishes | Constant-time compare if set | No | Yes with the Queue API | New random token. Reject the example placeholder |
| `QUEUE_API_HMAC_SECRET` | Signs body | Signs body when it publishes | Checks timestamp and signature if set | No | Yes with the Queue API | New random secret. Reject the example placeholder |
| `QUEUE_API_BASE_URL` | Meeting and other publishers | Internal publishers | n/a | n/a | Same staging Queue API | HTTPS name that reaches staging Caddy, not the production IP |
| `BUDDY_API_BASE` | Daily briefing and other Python calls | n/a | n/a | n/a | The staging Python API | Staging Cloud Run URL ending in `/api/v1` |
| `REDIS_URL` | Some meeting publishes | API-side Redis if used | Overridden to compose db 0 | Overridden to compose db 0 | Queue API and worker yes | Staging compose Redis only |
| `REMINDER_REDIS_URL` | Reminders | No | No | No | Isolated | Staging reminder Redis |
| `GOAL_REDIS_URL` | Node goals | No | No | No | Isolated | Staging goal Redis |
| `MONGO_URI` / `MONGODB_URL` | Yes | Yes | If the role reads it | Yes | Same staging database | New server or new database. Not production |
| `MONGODB_DATABASE` | Path inside `MONGO_URI` | Default `buddy` | Same rule | Same rule | Yes | `buddy` only on the staging server |
| `SPEECH_VECTOR_QUEUE_TRANSPORT` | No | Publisher must agree | Yes | Yes | Queue API and worker | `list` |
| `WORKER_CONCURRENCY` | No | No | No | Compose default 1 | n/a | Leave at 1 |
| `JOB_LEASE_TTL_SECONDS` | No | No | No | Default 90 | n/a | 90 |
| `JOB_LEASE_HEARTBEAT_SECONDS` | No | No | No | Default 20 | n/a | 20 |
| `AWS_S3_BUCKET` | Uploads | Reads | No | Reads | Node and worker | Staging bucket |
| Provider API keys | Some | Yes | No | Yes | Per environment | Sandbox or unset |

## Isolation checks before any staging process starts

1. Staging Redis host is not the compose host that production uses, not Redis Cloud hosts named in the workstation env file, and not `127.0.0.1:6399` (that port is the local disposable test instance).
2. Staging Mongo host is not the host in the workstation `MONGO_URI`.
3. Staging bucket is not the bucket in the workstation env file.
4. Queue credentials are not the `.env.example` placeholders.
5. `SPEECH_VECTOR_QUEUE_TRANSPORT` is `list` on every staging process that reads it.
