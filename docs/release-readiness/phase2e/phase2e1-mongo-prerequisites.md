# Phase 2E.1 MongoDB prerequisites

Source: `ensure_idempotency_indexes` in `services/goals/store.py`. The worker does not call it. Do not call it against production. Do not delete documents.

| Collection | Keys | Unique | Partial filter | Name |
| --- | --- | --- | --- | --- |
| `goal_research_sessions` | `goalId`, `operationId` | yes | `operationId` has BSON type string | `goal_research_operation` |
| `goal_research_evidence` | `goalId`, `operationId`, `evidenceId` | yes | `operationId` has BSON type string | `goal_research_evidence_operation` |
| `goal_plan_versions` | `goalId`, `version` | yes | none | `goal_plan_version` |
| `goal_evidence` | `goalId`, `sourceType`, `sourceId`, `sourceRevision`, `evidenceKey` | yes | none | `goal_evidence_identity` |

Rows with no `operationId` are outside the partial indexes. They can stay. They are not repaired by this release.

## Duplicate check

On the staging database, with a read-only user, count groups larger than one:

- `goal_research_sessions` where `operationId` is a string, grouped by `goalId` and `operationId`
- `goal_research_evidence` where `operationId` is a string, grouped by `goalId`, `operationId`, and `evidenceId`
- `goal_plan_versions` grouped by `goalId` and `version`
- `goal_evidence` grouped by `goalId`, `sourceType`, `sourceId`, `sourceRevision`, and `evidenceKey`

Also list `goal_jobs` with `status` `ready`, a non-empty `durableStep.plan.milestones`, and no `goal_plan_versions` row for that goal and version. The worker inserts that version only when the job is claimed again. It does not scan at startup. Leave those rows. Do not delete them.

If any duplicate count is greater than zero, stop. `create_index` will fail on the unique indexes. Repair is a separate approval. Dropping the collection is not the repair.

A name that already exists with different keys also fails `create_index`. Compare `getIndexes` and stop. Do not drop an index another query uses until the key spec is understood.

## Create and verify

After the counts are zero, and only on the staging URI:

```text
python -c "import asyncio; from motor.motor_asyncio import AsyncIOMotorClient; from services.goals.store import ensure_idempotency_indexes; async def main():
    client = AsyncIOMotorClient('<STAGING_MONGODB_URL>')
    await ensure_idempotency_indexes(client['buddy'])
    client.close()
asyncio.run(main())"
```

Then `getIndexes` shows those four names with `unique: true` and the partial filters on the two research indexes.

## Failure and rollback

If creation fails, leave the data and do not start the new worker against that database. Concurrent research or evidence writes can duplicate until the unique index exists.

Dropping these indexes is a rollback of the constraint, not of the data. Duplicates become possible again. Prefer to keep the indexes. Do not drop them to hide a duplicate.

Historical documents without `operationId` remain valid. New job-scoped research uses `goal-research:{goalId}:{jobId}` and is unique once the index exists.

Review and space-evidence `evidenceVersion` increments still have no operation id (`services/goals/jobs.py` `_review` and `_evidence`, and `services/goals/reviews.py`). A retry can move the counter twice. That does not delete the goal. It is an at-least-once limitation, not a reason to block the index step. It is not fixed in this phase.
