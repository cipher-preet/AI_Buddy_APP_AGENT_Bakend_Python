# Phase 2E.2 staging deployment runbook

Do not execute this runbook as part of preparation. Each phase that creates a resource or starts a container waits for a separate operator approval. Speech/vector transport stays `list` through Phase 12. The stream cutover is not a phase here.

Dependency order used below: MongoDB indexes before workers write goals, Redis persistence before the Queue API accepts events, Queue API before Node and Python publish, worker after Redis and MongoDB are healthy. Node and Python can start together once the Queue API health URL answers. Caddy is last among the processes so it does not route to a dead upstream, and it is first from the client's point of view.

## Phase 0 — approval and isolation

- Written approval to create one new `t3.medium` in `ap-south-1`, or to stay on local mode only.
- Confirmation the target is not the production EC2, not project `buddy-505709`'s live Node service, and not project `python-microservice-hub`'s live `buddy-ai-api`.
- Confirmation no production Redis, MongoDB, S3 bucket, or secret will be copied.
- Operator DNS names, or an explicit decision to remain on local mode.

Stop if any of those are missing.

## Phase 1 — release commits

- Create `release/phase-2e1-staging` only after approval.
- Apply the commit split in `phase2e2-release-artifacts.md`.
- Do not push until a second approval.
- Build the Python and Node images from those commits. Do not build from dirty `main` if the commits exist.

## Phase 2 — provision, after a fresh approval

Documented commands, not run here:

- Create the instance, 30 GB gp3, security group 80/443 public and 22 from the operator address.
- Allocate an Elastic IP if the names must survive a stop. Otherwise record that DNS will change every stop.
- Install Docker on that instance. Copy this repository and `Node_Backend` as siblings, because the Node build context is `../Node_Backend`.
- Do not run `deploy.bat`. Do not run `docker compose -f docker-compose.aws.yml`.

Local mode skips the instance and uses the developer machine. It still uses only `docker-compose.staging.yml`.

## Phase 3 — DNS, TLS, secrets, storage

- Copy `deploy/staging/staging.env.example` to `deploy/staging/staging.env`.
- Generate new secrets on that host. Alphanumeric. User token and queue token differ. Node and Python share the user token.
- Cloud mode: set the three DNS names to the staging address, ports 80 and 443, and `QUEUE_API_BASE_URL` to `https://` plus the queue name.
- Leave bucket variables empty and `STORAGE_PROVIDER=local`.
- `python scripts/staging/phase2e2_preflight.py --env deploy/staging/staging.env --expect ready`
- In that copy, set `STAGING_ENV_FILE=deploy/staging/staging.env`. Compose loads `${STAGING_ENV_FILE}` and otherwise stays on the example file.

## Phase 4 — MongoDB indexes

Start only `staging-mongo` if a partial start is used, or start the data services and stop before the worker. Follow `phase2e2-redis-mongo-preparation.md`. Stop on duplicate groups. Do not delete rows.

## Phase 5 — Redis

Start `redis`, `redis-reminder`, and `redis-goals`. Run the persistence checks in that same document. Confirm no host port is published:

```
docker compose -p kukunotes-staging -f docker-compose.staging.yml ps
```

Stop if 6379 or 27017 appears in the host port column.

## Phase 6 — Queue API

Start `queue-api`. Local: `http://queue-api:8080/health/ready` from another container. Cloud: `https://<operator queue name>/health/ready` with certificate verification left on. A TLS error stops the run. Do not switch the client to `http://` on the public name.

## Phase 7 — worker

Start `buddy-worker`. Confirm the log line for consumer-group creation or `BUSYGROUP`, concurrency 1, and transport `list`. Confirm it is not connected to a Redis host other than `redis`.

## Phase 8 — Node and Python

Start `python-api` and `node-api`, then `caddy`. Python uses `QUEUE_PROVIDER=queue_api`. Node uses the same queue URL and the same user secret. Order between Node and Python does not matter. Caddy after both are listening.

## Phase 9 — authentication smoke

- Mint a staging user token with the staging secret. Call a Node route and a Python route that require a user. Both accept it.
- Flip one character. Both reject it.
- Send that user token to the Queue API. It is rejected.
- Send the queue service token without a valid HMAC. It is rejected.
- Send the queue token and a valid HMAC to `/internal/events`. It is accepted once.

Use a synthetic user id. Do not use a production account.

## Phase 10 — product smoke, synthetic only

- Speech job with a tiny local file and a mock provider. Transport remains `list`.
- Meeting event through the Queue API, then the worker.
- Goal create and plan. Confirm a plan version row exists before status `ready`.
- Daily briefing scheduler with a synthetic clock. No real push (`FCM_ENABLED=false`).
- One reminder on `redis-reminder` only.

No paid STT, LLM, or embedding call. If a code path cannot run without a provider key, record it as not executed.

## Phase 11 — crash and recovery

On the staging worker Redis only:

- Stop the worker after a job is leased and before ack. Start it again. The job is reclaimed after the lease TTL, not lost.
- Publish the same queue event id twice. One payload is stored.
- Confirm a restart does not `XTRIM` the staging streams.
- Do not run these against the disposable test port if that port is shared with another process, and never against a hosted Redis.

## Phase 12 — acceptance and rollback

Acceptance is the Phase 9 through 11 results written down, plus the preflight `ready` result, plus the index `getIndexes()` output. That record is what can later support a staging-validated claim. This preparation document does not make that claim.

Rollback on the staging host: stop project `kukunotes-staging`, start the previous staging image, or remove the staging volumes if the data is synthetic. Production services stay untouched. DNS, if it points at the staging Elastic IP, is removed or left pointed at a stopped instance. Do not retarget production names.

## Later, not in this runbook

`SPEECH_VECTOR_QUEUE_TRANSPORT=stream` is a separate change with its own approval, after list-mode staging has passed. It is not a flag flip during Phase 6.
