# Phase 2E.2 release artifacts

Checked 2026-11-05. No commit was created. No push was made. The working branch was not changed.

## Git state

| Repo | Branch | HEAD | Tree |
| --- | --- | --- | --- |
| AI_Orchestration | `main` | `d606a98ee8e08215b2df8a057b2b3f9229048947` | Dirty. Phase 2 through 2E.2 are uncommitted. Deploying this HEAD does not ship them. |
| Node_Backend | `main` | `6d77dfb63335cb341b6761305124faf02d4dd759` | Dirty. Auth, queue transport, and goal stream changes are uncommitted. |

Sibling client repos were not the subject of this phase. Do not fold them into the release commits unless a later review shows they must change for the new bearer.

## Commit plan

Proposed branch, created only after the operator says so: `release/phase-2e1-staging`. Do not amend `d606a98` or `6d77dfb`.

Node, in order:

1. Auth and home access: `utils/authToken.ts`, `MIddleware/Auth/Auth.middleware.ts`, `Buddy/dailyBriefingAuth.ts`, `Buddy/homeAccess.ts`, the home and daily-briefing files, and their tests.
2. Queue transport and retention: `MeetingExtension/queue.ts`, `queueTransport.ts`, `queueTransport.test.ts`, `Goals/queue.ts`, `.env.example`, `package.json`.

Python, in order, on the same branch name in AI_Orchestration:

1. Auth verifier and request dependency.
2. Queue HTTPS rejection and the atomic publish path.
3. Lease, fencing, and stream retention. No `MAXLEN`.
4. Goal recovery and the four index definitions. The worker still does not create the indexes.
5. File-read offload for Deepgram, Sarvam, and the meeting header. No provider or prompt change.
6. Phase 2E.2 staging templates, preflight, and these documents.

Unrelated dirty files stay out of those commits until someone reviews them: `DockerDeployee.md`, `apps/api_gateway/middleware/logging.py`, and any chat route edit that is not part of the list above. Do not discard them.

`SPEECH_VECTOR_QUEUE_TRANSPORT` stays `list` in every committed example. `WORKER_CONCURRENCY` stays 1 in the compose files. `legacy_list_migration.py` is not wired into startup.

## Secret exclusion

- Do not `git add -f` a file matching `.env`, `.env.*`, `cloud-run-env.yaml`, or `cloud-run-api.env`.
- `.gitignore` ignores `.env.*`, so `AI_Orchestration/.env.example` and `.env.aws.example` are not part of a normal commit. Node `.env.example` is tracked. It may contain empty placeholders. It must not contain a live token.
- `deploy/staging/staging.env.example` is placeholders (`replace-with`, `example.invalid`). It is allowed in the docs commit.
- `deploy/staging/staging.env` and `deploy/staging/*.local.env` are gitignored.
- The Cloud Run staging files in `deploy/staging/` contain the same placeholders and the words `NOT A DEPLOY FILE`.
- If a tracked diff shows a real password, key, or connection string, stop and report the path only. Do not copy the value into the commit message or this document.

This phase did not open production env files and did not find a new tracked secret. The earlier audit already recorded that a gitignored Node env file holds live credentials. That file stays gitignored.

## Build and test, from the release commits once they exist

Python, from AI_Orchestration, disposable Redis and Mongo only:

```
python -m pytest tests/test_stream_retention.py tests/test_queue_transport_security.py tests/test_p0_api_auth.py tests/test_goal_persistence_consistency.py tests/test_goal_crash_recovery.py tests/test_worker_perf.py -q
```

Node, from Node_Backend:

```
npm test
npx tsc --noEmit --pretty false
```

The last full Python selection in Phase 2E.1 was 248 passed. This phase did not rerun it. Do not rerun the Redis kill tests (`test_queue_api_atomic_redis7.py`, `test_speech_vector_streams_redis7.py`, `test_lease_redis7_integration.py`) against a shared Redis.

## Artifact references

- Images, when an operator builds them on the staging host: `kukunotes-staging-python:local`, `kukunotes-staging-node:local`. No ECR push is required for the first staging start.
- Compose project: `kukunotes-staging`.
- Rollback of the application is the previous image tag or the previous git revision on that staging host. Production rollback stays the existing `docs/release-readiness/phase2e/rollback-plan.md`. Staging rollback does not touch production revisions.
- Deploying current `main` is not a rollback path for this work, because `main` does not contain it.
