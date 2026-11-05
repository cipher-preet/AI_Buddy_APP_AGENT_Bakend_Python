# Phase 2E.2 Redis and MongoDB preparation

No index was created. No Redis command was sent to a production or hosted instance. The checklist is for the staging containers after the operator starts them.

## Redis roles that stay separate

| Role | Staging container | Who connects | What it holds |
| --- | --- | --- | --- |
| Worker and Queue API | `redis` | `queue-api`, `buddy-worker`. URL forced to database 0. | Speech and conversation streams, lists while transport is `list`, leases, dedupe keys. |
| Reminders and mind maps | `redis-reminder` | Node reminder client, Python mind-map client. | `buddy:reminder:schedule`, `buddy:reminder:processing`, `buddy:reminder:retry`, `buddy:reminder:dead-letter`. Mind-map jobs when `MINDMAP_REDIS_URL` points here. |
| Goals | `redis-goals` | Node `GOAL_REDIS_URL`, Python goal worker. | `buddy:goals:interactive` group `goal-interactive-workers`, `buddy:goals:jobs` group `goal-workers`. |

Collapsing these onto one Redis would change connection budgets and would put reminder keys next to queue payloads. Staging does not do that.

## Persistence

Each container runs Redis 7 with `--appendonly yes` and `--save 60 1`. `appendfsync` is not set, so Redis uses its default of `everysec`. A crash can drop about one second of writes. That is the same policy as `docker-compose.aws.yml`. Do not switch staging to `appendfsync always` without a measurement.

`maxmemory` is unset. Do not set `allkeys-lru`. Eviction can delete a stream entry that a consumer group still holds.

There is no `MAXLEN` on work, retry, or dead-letter streams. The old settings `REDIS_MINDMAP_STREAM_MAXLEN`, `REDIS_DOCUMENT_STREAM_MAXLEN`, `REDIS_GOAL_STREAM_MAXLEN`, and `REDIS_CLOUD_DEAD_LETTER_MAXLEN` still load and are not applied. Do not put them back on `XADD`.

Volumes: `staging_redis_data`, `staging_reminder_redis_data`, `staging_goal_redis_data`. They belong to compose project `kukunotes-staging`.

## Consumer groups

The worker creates groups on startup with `XGROUP CREATE ... MKSTREAM`. A second start returns `BUSYGROUP` and continues. Groups on the worker Redis use the code defaults:

| Stream | Group |
| --- | --- |
| `buddy:audio:ingestion` | `audio-workers` |
| `buddy:stt:jobs` | `stt-workers` and `meeting-video-workers` |
| `buddy:transcript:ready` | `transcript-workers` |
| `buddy:conversation:window-extraction` | `window-extraction-workers` |
| `buddy:conversation:finalization` | `finalization-workers` |
| `buddy:conversation:processing` | `conversation-processing-workers` |
| `buddy:conversation:schedule-extraction` | `schedule-extraction-workers` |
| `buddy:daily-briefing:jobs` | `daily-briefing-workers` |
| `buddy:meeting:video-merge` | `meeting-merge-workers` |
| `buddy:mindmap:jobs` | `mindmap-workers` |
| `buddy:document:jobs` | `document-workers` |
| `buddy:speech:jobs` | `speech-workers` |
| `buddy:vector:jobs` | `vector-workers` |

`buddy:conversation:retry` and `buddy:dead-letter` have no startup trim. `buddy:stt:jobs` is not `XDEL`'d after ack because two groups share it. Mind-map, document, and goal consumers delete one acknowledged id only.

Leases stay `buddy:job-lease:`, done keys `buddy:job-done:`, fence `buddy:job-fence:global` with no TTL. Heartbeat 20 seconds, TTL 90 seconds. Goal reclaim throttle stays the code value. Speech/vector transport stays `list` for the first deployment. The list-to-stream cutover is a separate approved operation and is not part of this runbook.

## Redis checks, after the staging containers exist

Run these inside the staging compose project, not against a hosted URL. Replace nothing with a production host.

```
docker compose -p kukunotes-staging -f docker-compose.staging.yml exec redis sh -c 'redis-cli --no-auth-warning -a "$REDIS_PASSWORD" INFO persistence'
docker compose -p kukunotes-staging -f docker-compose.staging.yml exec redis sh -c 'redis-cli --no-auth-warning -a "$REDIS_PASSWORD" CONFIG GET appendonly'
docker compose -p kukunotes-staging -f docker-compose.staging.yml exec redis sh -c 'redis-cli --no-auth-warning -a "$REDIS_PASSWORD" CONFIG GET maxmemory'
docker compose -p kukunotes-staging -f docker-compose.staging.yml exec redis sh -c 'redis-cli --no-auth-warning -a "$REDIS_PASSWORD" XPENDING buddy:speech:jobs speech-workers'
docker compose -p kukunotes-staging -f docker-compose.staging.yml exec redis-goals sh -c 'redis-cli --no-auth-warning -a "$GOAL_REDIS_PASSWORD" XINFO GROUPS buddy:goals:jobs'
```

The goal command is on `redis-goals`. The shell quotes keep the password inside the container. Expect `appendonly` yes, `maxmemory` 0, and `aof_enabled` 1. Expect no `maxlen` in a staging config rewrite. Disk: `docker system df -v` for the staging volumes. Memory: `INFO memory` on each of the three containers. These commands were not run in this phase.

Reset is deleting the staging volumes after the operator stops the staging project. That does not touch `redis_data` on a production project. Do not `FLUSHALL` a Redis whose host is not `redis`, `redis-reminder`, or `redis-goals` inside this project.

## MongoDB

Image `mongo:7`, service `staging-mongo`, database `buddy`, user from `MONGO_USER` / `MONGO_PASSWORD`, no published port. Backup before a destructive test is `mongodump` of that database to the staging disk. Reset is dropping the staging database, or removing volume `staging_mongo_data`, only after the operator confirms the container name is `staging-mongo`. Do not drop a database on any other host. Do not delete documents to make an index build succeed.

`ensure_idempotency_indexes` in `services/goals/store.py` is not called from worker startup. The four indexes are:

| Name | Collection | Keys | Unique | Partial filter |
| --- | --- | --- | --- | --- |
| `goal_research_operation` | `goal_research_sessions` | `goalId`, `operationId` | yes | `operationId` is a string |
| `goal_research_evidence_operation` | `goal_research_evidence` | `goalId`, `operationId`, `evidenceId` | yes | `operationId` is a string |
| `goal_plan_version` | `goal_plan_versions` | `goalId`, `version` | yes | none |
| `goal_evidence_identity` | `goal_evidence` | `goalId`, `sourceType`, `sourceId`, `sourceRevision`, `evidenceKey` | yes | none |

Rows without `operationId` are outside the two partial indexes. Duplicate groups inside a unique key stop the build. The procedure stops there. It does not delete rows.

## Index procedure, staging database only

1. Confirm the shell is inside compose project `kukunotes-staging` and the service is `staging-mongo`. Print `db.getName()` and stop unless it is `buddy` on that container.
2. `getIndexes()` on the four collections. Record names. Do not drop an index that is already the definition above.
3. Duplicate precheck, read only:

```
db.goal_research_sessions.aggregate([
  { $match: { operationId: { $type: "string" } } },
  { $group: { _id: { goalId: "$goalId", operationId: "$operationId" }, n: { $sum: 1 } } },
  { $match: { n: { $gt: 1 } } },
  { $limit: 20 }
])
```

Repeat for `goal_research_evidence` on `goalId`, `operationId`, `evidenceId`. Repeat for `goal_plan_versions` on `goalId`, `version`. Repeat for `goal_evidence` on `goalId`, `sourceType`, `sourceId`, `sourceRevision`, `evidenceKey`.

4. If any cursor returns a document, stop. Do not delete. Fix the staging fixtures and run the check again.
5. Create the four indexes with the keys, names, and partial filters above. `createIndex` is idempotent when the specification matches.
6. `getIndexes()` again and confirm `unique: true` on each name.
7. Concurrent insert test: two inserts of the same key, one succeeds, the other raises a duplicate-key error. A retry of a completed research operation still returns the original row. This test uses synthetic ids only.

`scripts/staging/phase2e_preflight.py` prints the index names and exits 2 on an execute flag. It does not connect. Use it as the reminder, then run the steps above by hand on staging.
