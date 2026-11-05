# Phase 2E data migration checklist

Do not run this checklist against production. Do not delete documents. Do not run `FLUSHALL`, `XTRIM`, or stream-group deletion. The helper `scripts/staging/phase2e_preflight.py` only prints this list. It exits 2 if given any other flag, and it never opens a connection.

## C1. MongoDB indexes

`ensure_idempotency_indexes` in `services/goals/store.py` is not called from worker startup. A new worker assumes the unique indexes exist when concurrent writes race. On a brand-new staging database the collections are empty and creation is safe. On a copy of historical data, create the indexes only after the duplicate query returns zero rows.

| Collection | Keys | Unique | Partial filter | Index name |
| --- | --- | --- | --- | --- |
| `goal_research_sessions` | `goalId`, `operationId` | yes | `operationId` is a string | `goal_research_operation` |
| `goal_research_evidence` | `goalId`, `operationId`, `evidenceId` | yes | `operationId` is a string | `goal_research_evidence_operation` |
| `goal_plan_versions` | `goalId`, `version` | yes | none | `goal_plan_version` |
| `goal_evidence` | `goalId`, `sourceType`, `sourceId`, `sourceRevision`, `evidenceKey` | yes | none | `goal_evidence_identity` |

Historical session and evidence rows with no `operationId` are outside the partial filter. They do not block index creation and they are not deleted.

Name collisions: if an index already uses one of those names with different keys, `create_index` will fail. Stop and inspect `getIndexes`. Do not drop an index whose name is shared with application queries until the key spec is compared.

### Duplicate detection (read only)

Connect to the staging database with a read-only user. For each grouping, count groups whose size is greater than 1.

- `goal_research_sessions` where `operationId` is a string, group by `goalId` and `operationId`.
- `goal_research_evidence` where `operationId` is a string, group by `goalId`, `operationId`, and `evidenceId`.
- `goal_plan_versions` group by `goalId` and `version`.
- `goal_evidence` group by `goalId`, `sourceType`, `sourceId`, `sourceRevision`, and `evidenceKey`.

Also list `goal_jobs` with `status: "ready"` whose `durableStep.plan.milestones` is non-empty and that have no `goal_plan_versions` document for that `goalId` and version. The worker can insert a missing version from `durableStep` when a job is claimed again. It does not scan all ready jobs at startup. Rows that never get another claim stay without a version until a maintenance reconcile. That is a data-quality item, not a delete.

If any duplicate count is greater than zero, stop. Repair needs a separate approval. This release does not delete either copy.

### Staging creation procedure

Run only after the duplicate counts are zero and the shell prompt shows the staging host.

```text
python -c "import asyncio; from motor.motor_asyncio import AsyncIOMotorClient; from services.goals.store import ensure_idempotency_indexes; async def main():
    client = AsyncIOMotorClient('<STAGING_MONGODB_URL>')
    await ensure_idempotency_indexes(client['<STAGING_DB>'])
    client.close()
asyncio.run(main())"
```

Then `getIndexes` on the four collections and confirm the names and `unique: true`.

## C2. Redis persistence

Compose Redis is started with `--appendonly yes --save 60 1`. `appendfsync` is unset, so Redis uses `everysec`. A crash can drop about one second of writes. AOF plus a successful Lua script is not the same as a disk fsync.

| Event | What survives a restart when AOF is on | What can be lost |
| --- | --- | --- |
| Acknowledged stream entry | The entry is gone. The work must already be in MongoDB or S3. | The last second before a crash, if fsync has not run. |
| Pending stream entry | Still pending. The new worker reclaims it after the idle time. | The last second of `XADD`s. |
| Lease key | Present until its TTL. A lost lease allows reclaim. The fence must still reject the old owner. | A lease written in the last second. |
| Fence counter | Present. It has no TTL. | A restart from an empty volume resets it. Do not restore an older RDB over a newer fence. |
| Completion key `buddy:job-done:*` | Present until its TTL. | A lost key allows a duplicate attempt. Fenced Mongo writes are what stop a second durable effect. |
| List push that returned, then the process died before the dedupe hash | The list entry remains. A retry can push another. | This is the remaining at-least-once window. Lua does not undo a successful `LPUSH` or `XADD`. |

`maxmemory` is not set in the compose command. Redis will not evict keys until the host is out of memory. That is safer for queues than `allkeys-lru`, and it means a full disk is an incident, not a silent drop. Do not change the policy in this release.

The local disposable Redis used for tests was `127.0.0.1:6399`, Redis 7.4.9, `aof_enabled` 1. That process is not staging and not production.

Live EC2 AOF status is UNVERIFIED.

## C3. Read-only queue inspection

Use `XINFO`, `XLEN`, `XPENDING`, `LLEN`, and `TYPE` on the staging instance after it exists. Do not use a wildcard delete. Do not scan production.

Record, before and after the worker starts:

- `XINFO GROUPS` on each stream in `architecture-inventory.md`
- Pending count and the oldest idle time
- `LLEN speech_transcribe_queue`
- `LLEN completed_speech_queue`
- `LLEN speech_transcribe_queue:migrating`
- `LLEN speech_transcribe_queue:migration_failed`
- `XLEN buddy:conversation:retry` and `XLEN buddy:dead-letter`
- `XLEN buddy:goals:jobs` and pending in `goal-workers`
- Count of `goal_jobs` in status `running` older than the lease TTL

On the first staging deploy these should be empty because the Redis volume is new. Non-zero values on a shared volume mean the instance is not isolated. Stop.

## C4. Idempotency classification

| Finding | Class |
| --- | --- |
| Review path in `services/goals/jobs.py` `_review` increments `evidenceVersion` with a fence and no operation id | Acceptable documented at-least-once limitation for staging. A retried review can increment twice before `ready`. |
| Space-evidence path `_evidence` increments `evidenceVersion` on every fenced execution | Same class. Not an operation-idempotent update. |
| `services/goals/reviews.py` also increments `evidenceVersion` without an operation id | Same class. |
| Baseline findings use `appliedEvidenceOps` and increment once per job | Required behavior. Covered by `tests/test_goal_persistence_consistency.py`. |
| Research sessions upsert on `goal-research:{goalId}:{jobId}` and reuse only when `writeComplete` is true | Required behavior. A crash before `writeComplete` may call the model again. At-least-once, not exactly-once. |
| Callers that save research without `job_id` still `insert_one` | Future improvement. The job path passes `job_id`. |
| Plan version is written before status `ready` | Required behavior. |
| Ready jobs with no `durableStep.plan` are not backfilled | Future improvement. Reconcile inserts only when milestones exist and the job is claimed again. |
| Historical research rows without `operationId` | Allowed by the partial index. Do not delete them. |
| Lua returns after a successful push and then the process dies before the dedupe hash | Acceptable at-least-once limitation. A wrong Redis type is rejected before the push. A crash is not. |
| AOF `everysec` | Acceptable if the operator accepts about one second of loss. Not a reason to set `appendfsync always` in this release. |
| Goal document and plan-version document are not one multi-document transaction | Staging Mongo is a standalone server in the current design. Fenced ordered writes are the contract. |
| Indexes missing on the database the new worker uses | Staging release blocker until the creation procedure above has run on that database. |

## Speech/vector flag

Leave `SPEECH_VECTOR_QUEUE_TRANSPORT=list`. Do not run `legacy_list_migration.py`. Inventory of the list keys is part of C3 so a later, separate stream test can see that the staging lists started empty.
