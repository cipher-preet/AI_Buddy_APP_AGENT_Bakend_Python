# Phase 2E staging deployment runbook

Phase 2E.1 (2026-11-05) changed two rules this runbook now assumes:

- Work, retry, and dead-letter streams are not trimmed with `MAXLEN`. Do not turn that trimming back on. `XDEL` still removes only an id that this consumer has acknowledged.
- A public `http://` Queue API URL is rejected. Staging `QUEUE_API_BASE_URL` must be `https://` on a hostname the operator controls. `QUEUE_API_DOMAIN=:80` serves cleartext and is not acceptable for that URL. Loopback, RFC1918, and single-label Docker names may stay on HTTP.

This runbook is not an authorization to deploy. Phase 2E did not deploy. Complete Stage 0 and get a named person's approval before any command that creates a cloud resource, an index, or a container.

Placeholders use angle brackets. Do not paste production URIs, passwords, or the example Queue API tokens.

## Stage 0 — Preconditions

Stop if any box is unchecked.

- [ ] AI_Orchestration changes for Phase 2 are committed, and the commit id is written here: `<ORCH_SHA>`. HEAD `d606a98` does not contain them. They were uncommitted on 2026-11-05.
- [ ] Node commit id is written here: `<NODE_SHA>`. Audited HEAD was `6d77dfb`.
- [ ] Desktop, Android, and extension commit ids are recorded if those clients ship in the same window.
- [ ] Images are built from those commits, not from a dirty tree.
- [ ] Staging Redis is a new compose volume. Host and volume name are recorded.
- [ ] Staging MongoDB is a new server or a new database. Host is recorded and is not the host in the workstation Node env file.
- [ ] Staging S3 bucket or prefix is recorded and is not the production bucket.
- [ ] Two synthetic users exist. No production user id is used.
- [ ] `AUTH_TOKEN_SECRET` is a new value, identical on Node and the Python API.
- [ ] `QUEUE_API_SERVICE_TOKEN` and `QUEUE_API_HMAC_SECRET` are new values, identical on Node, the Python API, and the Queue API. They are not the `.env.example` placeholders.
- [ ] `SPEECH_VECTOR_QUEUE_TRANSPORT=list` on the Queue API and the worker.
- [ ] `WORKER_CONCURRENCY` stays at the compose default 1. Replica count stays 1.
- [ ] Provider keys are sandbox or unset. FCM, SMS, and email to real users are off.
- [ ] A backup of the empty staging Mongo and the empty Redis volume procedure is written down before data is inserted.
- [ ] Baseline log command is known (container logs, Cloud Run logs). No new monitoring product is required.
- [ ] Named approver: `<NAME>` date: `<DATE>`.

## Stage 1 — Schema and Redis

1. From a shell that has only the staging URI, run the duplicate queries in `data-migration-checklist.md`. On an empty database the counts are zero.
2. If any count is greater than zero, stop. Do not delete rows.
3. Run `ensure_idempotency_indexes` against the staging database only, using the command in that checklist.
4. Confirm the four index names and `unique: true`.
5. Start staging Redis with the compose command: `--appendonly yes --save 60 1`. Do not add `maxmemory` or change `appendfsync` in this release.
6. `INFO persistence` shows `aof_enabled:1`. Record it.
7. Confirm the worker and Queue API environment print `list` for the speech/vector transport. Do not print the Redis URL.

## Stage 2 — Credentials and order

Deploy in this order. Do not start the next row while the previous health check is failing.

| Step | Process | Why this position |
| --- | --- | --- |
| 1 | Staging Redis | Empty volume, AOF on. |
| 2 | Queue API, one replica, new image | Atomic publisher. No old replica left on the same Redis. |
| 3 | Worker, one replica, new image | Lease, fence, and goal recovery. Stop any older worker first. Do not ack or delete pending entries. On a new volume there are none. |
| 4 | Node API | Sends user bearers and mints the 600-second briefing token. Uses the new Queue API credentials. |
| 5 | Python API | Rejects calls that lack the bearer. Node is already able to send one. |

Python before Node fails closed and breaks briefing and other Node-to-Python calls until Node is updated. That is safe and still an outage, so Node comes first.

Queue API health: `GET /health/ready` through staging Caddy returns success. Then one authenticated synthetic publish returns 202, and the same `eventId` returns duplicate without a second `LLEN` or `XLEN` increment.

Worker health: the process stays up, ffmpeg is present, and a log line shows the consumer groups. Pending counts stay 0 until a test publishes work.

## Stage 3 — Worker

On a new staging volume there is nothing to drain.

If this runbook is ever used against a Redis volume that already has entries:

- Stop the old worker with the process manager and wait until it exits. Do not `XACK`. Do not `DEL` streams.
- Leave consumer groups in place.
- Start one new worker.
- `XPENDING` on `buddy:goals:jobs` for group `goal-workers` still shows the previous entries, now reclaimable after the idle time.
- Confirm heartbeats in the logs about every 20 seconds and the lease TTL of 90 seconds.

Speech and vector consumers stay at concurrency 1 even if `WORKER_CONCURRENCY` is left at 1.

## Stage 4 — Application

Run the cases in `staging-e2e-test-plan.md` sections E1, E2, E3, E5, E6, and E7 in list mode. E4 stream mode is not part of this stage.

## Stage 5 — Failure cases

Run E8 and E9 only on the staging compose project. Do not point them at production Redis, the workstation Redis Cloud hosts, or the local disposable Redis on port 6399 while a developer is using it.

## Stage 6 — Watch

For the duration of Stage 4 and Stage 5, record:

- HTTP 401 and 403 counts on Node and Python, split into expected negative tests and unexpected failures.
- Queue API 401 count.
- Redis pending count per group.
- `XLEN` of `buddy:conversation:retry` and `buddy:dead-letter`.
- `goal_jobs` in `running` longer than 90 seconds after the worker is healthy.
- Meeting jobs that stay in processing after the worker is healthy.
- Duplicate `goal_plan_versions` or `goal_research_sessions` for one operation id.
- Worker memory, CPU, and event-loop lag warnings. The existing monitor logs lag above 250 ms on an 8 second interval (`services/observability/diagnostics.py`).
- Redis `used_memory` and MongoDB operation latency from the existing server logs.

## Stage 7 — Acceptance

The checklist at the end of `readiness-report.md` is filled from this run. A completed runbook file is not a production approval.

## Speech/vector stream cutover (separate, not part of Stage 2)

Do this only after list-mode Stage 7 passes, and only on staging, with a second approval.

1. Record `LLEN` of `speech_transcribe_queue`, `completed_speech_queue`, `:migrating`, and `:migration_failed`.
2. Stop the worker so no `BRPOP` is in flight. Confirm the processing lists are empty.
3. Confirm no publisher still has `SPEECH_VECTOR_QUEUE_TRANSPORT=list` or an unset value that defaults to `list` while the worker is about to switch. Switch publishers and the worker in one window.
4. Set the Queue API and the worker to `stream`.
5. Run the existing migration utility in its dry-run mode first. A dry-run that would delete keys is a stop. The utility was not run in this audit.
6. Start the worker. Confirm groups `speech-workers` and `vector-workers` exist on `buddy:speech:jobs` and `buddy:vector:jobs`.
7. Publish one synthetic speech job. Expect one stream entry, one transcript, one vector result, and an empty legacy list.
8. Kill the staging worker once, wait past the lease, and confirm the pending entry is reclaimed once.
9. Rollback of this cutover is in `rollback-plan.md`. Do not delete the stream or the group.

## Commands that are intentionally absent

This file does not contain `gcloud run deploy`, `docker compose up` against the production project, `FLUSHALL`, or index creation against the production URI. Add the real staging hostnames in the operator's shell, not in git.
