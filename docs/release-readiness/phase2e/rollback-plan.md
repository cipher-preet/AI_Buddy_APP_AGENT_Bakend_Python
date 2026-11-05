# Phase 2E rollback plan

A code rollback can restore a bug that drops or duplicates work. Pause the affected path and roll forward when a rollback would acknowledge, skip, or double-write data. Do not delete user documents or Redis entries to make a rollback look clean.

The audited Phase 2 worker code is uncommitted. Until it is tagged, the rollback artifact for the worker does not exist. Do not roll back to `d606a98` and expect the new goal recovery to remain.

## 1. Node API

- Trigger: staging Node returns 5xx on login or home, or it sends Python a token Python rejects for every user while the secret was verified equal.
- Stop: Cloud Run traffic back to the previous Node revision. In-flight requests finish or fail. Nothing is drained in Redis by Node.
- Artifact: the previous Cloud Run revision id, recorded at deploy time.
- Data risk: sessions in staging Mongo remain valid if `SESSION_SECRET` did not change. If the secret changed with the deploy, old cookies fail closed. Users sign in again. Do not point the rollback revision at production Mongo.
- Redis: Node does not ack worker streams. Meeting publishes that already returned 202 stay in Redis for the worker.
- After: one synthetic login, one space list, one briefing force-generate.

## 2. Python API

- Trigger: all valid user bearers receive 401, or a bad token receives 200.
- Stop: Cloud Run traffic back to the previous Python revision. Prefer a forward fix if the previous revision has no user checks: leaving an unauthenticated revision in front of real data is worse than a 401 outage.
- Artifact: previous revision id.
- Data risk: documents already written stay. The old API may authorize by a client-supplied id. On staging with synthetic data that is a test defect. On any database that holds real users, do not route traffic to that revision.
- Redis: already published events stay. The old API may publish without the atomic script if traffic is rolled back that far. Pause publishes if an old and a new Queue API would both be active. This rollback is the API, not the Queue API.
- After: user A can read A's goal and cannot read B's.

## 3. Queue API

- Trigger: publishes return 202 and the corresponding list or stream length does not increase, or two concurrent publishes of a new `eventId` both return published.
- Stop: one replica. Do not run the old check-then-publish image beside the new one.
- Drain: no. Entries already in Redis must stay.
- Artifact: the previous Queue API image digest.
- Data risk: rolling back to the check-then-publish code reopens the duplicate-publish race. Prefer a forward fix. If the new process cannot start, stop publishers, leave the entries, and restore the last image that still authenticated requests.
- Redis: do not delete `queue_api:event:*` keys. A deleted dedupe key makes the next publish look new.
- After: one new `eventId` returns 202 once, and a repeat is a duplicate. A user bearer without the service HMAC returns 401.

## 4. Worker

- Trigger: jobs stay `running` with a live heartbeat that never finishes, fences reject every write including the current owner, or the process crash-loops.
- Stop: one worker. Let it exit. Do not ack its pending entries.
- Artifact: the image built from `<ORCH_SHA>`. There is no such image until Stage 0 of the runbook.
- Unsafe rollback: an older goal worker can mark a job `ready` without a plan version and can acknowledge an abandoned `running` job. A list-only worker ignores `buddy:speech:jobs` if the flag was set to `stream`. Do not start that worker while stream entries are pending.
- Preferred action: leave the entries pending, fix forward, start one new worker.
- If the new worker cannot stay up: keep publishers paused, keep the Redis volume, and start the previous worker only when the flag is still `list` and no goal job is mid-flight. Accept that the old recovery bug returns until the forward fix is back.
- After: pending count is not lower just because of the restart. A synthetic job completes once. `goal_plan_versions` exists before `ready`.

## 5. MongoDB indexes

- Trigger: `create_index` fails because of duplicates or a name conflict.
- Stop: do not drop the collection and do not drop an index that the current application uses.
- Data risk: the unique indexes reject a second research session or plan version for the same key. That rejection is the desired contract. Rolling the index back by dropping it allows duplicates again.
- Preferred action: stop the worker, inspect the conflicting documents, get approval for a repair, then create the index. Do not delete either document as part of this rollback.
- After: `getIndexes` matches the table in `data-migration-checklist.md`, or the worker stays stopped.

## 6. Speech/vector transport

- Trigger: after a stream cutover, jobs remain only on `speech_transcribe_queue`, or the worker reads the stream and the publisher still pushes the list.
- Stop publishers and the worker before changing the flag.
- Rollback to `list`: set both the Queue API and the worker to `list`. Leave `buddy:speech:jobs`, `buddy:vector:jobs`, and their groups in place. A later stream consumer still needs those entries. Do not `XGROUP DESTROY` and do not `DEL` the streams.
- Entries that were copied to a stream and also remain on the list can be processed twice if both consumers run. Do not run a list consumer and a stream consumer for the same job family at the same time.
- Export completeness: before the reverse switch, record `XLEN`, `XPENDING`, `XLEN` of the retry and dead-letter streams, and the four legacy `LLEN` values. The rollback is incomplete if any of those numbers were not written down.
- After: one new list publish is consumed once, and the stream pending count is unchanged by that list job.

## What a rollback does not fix

- A plan version already stored stays stored.
- A research session with `writeComplete: true` stays the session the next attempt reuses.
- An `evidenceVersion` that moved twice on the review or space-evidence path stays at that number. Do not decrement it by hand.
- The global fence stays. Restoring an older Redis RDB over the live volume can move the fence backward. Do not do that.
