# Phase 2E.1 readiness report

Date: 2026-11-05. Nothing was deployed. No live AWS, Cloud Run, Redis, MongoDB, or S3 resource was changed. No commit was created and nothing was pushed. `SPEECH_VECTOR_QUEUE_TRANSPORT` remains `list`. Worker concurrency and replica count were not changed. Prompts and providers were not changed.

## Verdict

**BLOCKED FOR STAGING**

The two code defects in this phase are fixed and covered by local tests: Redis no longer trims unacknowledged stream entries, and a public `http://` Queue API URL is refused before the service bearer is sent.

Staging still cannot start, because:

1. The Phase 2 and Phase 2E.1 changes are uncommitted on `d606a98` and `6d77dfb`. A deploy of HEAD does not contain them. Commit creation was not authorized in this task.
2. No isolated staging Redis volume, MongoDB, or S3 bucket exists in the files that were inspected. The workstation environment points at the current systems. Copying it is not staging.
3. The Queue API hostname that would make Caddy serve HTTPS is not known. The code will not send credentials to the public `http://` URL that is configured today. An operator has to choose the name.
4. The four unique goal indexes are still not created automatically. They have to be created on the staging database after the duplicate check, and not before that database exists.

This is not a production approval, and it is not `STAGING VALIDATED`.

## What changed

- Producers and the maintenance loop no longer pass `MAXLEN` or call `XTRIM` on work, retry, or dead-letter streams. `XDEL` still removes a single acknowledged id on the mindmap, document, and goal consumers.
- The old maxlen settings remain in the environment schema and are ignored.
- Node goal publish no longer sends `MAXLEN ~ 200`.
- Python and Node reject a public HTTP Queue API base URL. HTTPS keeps certificate verification. Redirects are not followed. Private and loopback HTTP still works.
- Sanitized staging names, index steps, and a commit plan are in the other Phase 2E.1 documents.

## Durability, stated narrowly

A Redis Lua script that returns success means the commands ran in that Redis process. It does not mean the AOF has been fsynced. Compose uses `appendonly yes` and does not set `appendfsync`, so Redis uses `everysec`. About one second of acknowledged writes can be missing after a crash. `appendfsync always` would shrink that window and slow every write. This release does not change it. The operator can choose it later with a measurement. Worker fences and idempotent goal writes are what stop a replayed entry from doubling a plan version. That is at-least-once, not exactly-once.

## Same version versus mixed version

Two replicas of this same worker build are not, by themselves, incorrect: leases and fences exist so one owner writes. This release still keeps one worker and `WORKER_CONCURRENCY` 1 because that scale was not tested here. An old goal worker beside this one is unsafe. An old Queue API beside the atomic publisher is unsafe. A list publisher beside a stream consumer is unsafe. Those are mixed-version rules, not a claim that the current build cannot run one extra replica later.

## Evidence index

- `phase2e1-retention-audit.md`
- `phase2e1-transport-security.md`
- `phase2e1-release-manifest.md`
- `phase2e1-staging-env-template.md`
- `phase2e1-mongo-prerequisites.md`
- `phase2e1-test-report.md`
- `staging-deployment-runbook.md` (opening section updated)
