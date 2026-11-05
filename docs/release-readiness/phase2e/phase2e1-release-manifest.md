# Phase 2E.1 release manifest

Verified on 2026-11-05. No commit was created. No push was made.

| Repository | Branch | HEAD | Release state |
| --- | --- | --- | --- |
| AI_Orchestration | `main` | `d606a98ee8e08215b2df8a057b2b3f9229048947` | Phase 1, Phase 2, Phase 2E, and Phase 2E.1 changes are uncommitted on top of this commit. Deploying HEAD does not ship them. |
| Node_Backend | `main` | `6d77dfb63335cb341b6761305124faf02d4dd759` | Auth, home ownership, briefing proxy, goal `XADD`, and Queue API URL checks are uncommitted. |

`cloud-run-env.yaml` is gitignored and not in the index. `.env`, `.env.aws`, `.env.example`, and `.env.aws.example` in AI_Orchestration match `.gitignore` rule `.env.*`. Do not `git add -f` them. Node `.env.example` is tracked and contains empty placeholders only.

No merge conflicts. No submodules were listed by status.

## Do not commit

- `.env`, `.env.aws`, `cloud-run-env.yaml`
- `apps/secrets/`
- anything matching a live connection string, access key, or token

A name-only scan of the tracked diff did not show a newly added secret value. The Node `.env.example` diff adds empty `AUTH_TOKEN_SECRET` and `SESSION_SECRET` keys.

## Proposed commits

Create these only after a review of the full diff. Do not amend `d606a98` or `6d77dfb`. Suggested branch: `release/phase-2e1-staging`.

### Node, on `6d77dfb`

1. `node: require home auth and forward python bearers`
   - `utils/authToken.ts`
   - `MIddleware/Auth/Auth.middleware.ts`
   - `Buddy/homeAccess.ts`
   - `Buddy/homeSecurity.test.ts`
   - `Buddy/Controllers/Home.Controller.ts`
   - `Buddy/Repository/Home.repository.ts`
   - `Buddy/Routes.ts`
   - `Buddy/Services/Home.services.ts`
   - `Buddy/dailyBriefingAuth.ts`
   - `Buddy/Controllers/DailyBriefing.Controller.ts`
   - `Buddy/Repository/DailyBriefing.repository.ts`
   - `Buddy/Repository/DailyBriefing.repository.test.ts`
   - `.env.example`
   - `package.json` only if the test script change is split; otherwise leave the new test line with commit 2
2. `node: stop goal stream trimming and reject public queue http`
   - `Goals/queue.ts`
   - `MeetingExtension/queue.ts`
   - `MeetingExtension/queueTransport.ts`
   - `MeetingExtension/queueTransport.test.ts`
   - `package.json`

### Python, on `d606a98`

1. Security: `apps/api_gateway/auth.py`, `services/auth/`, route files that only add the dependency, `services/chat/node_client.py`, `tests/test_p0_api_auth.py`, `apps/api_gateway/main.py` if the change is auth wiring.
2. Leases and fences: `services/queue/job_lease.py`, `services/queue/mongo_fence.py`, worker files, `tests/test_lease_correctness.py`, `tests/test_job_lease_reliability.py`, `tests/test_lease_redis7_integration.py`.
3. Speech/vector transport, flag default remains `list`: `services/queue/legacy_list_migration.py`, `services/queue/redis_queue.py`, speech and vector workers, `tests/test_speech_vector_streams.py`, `tests/test_speech_vector_streams_redis7.py`, `tests/test_queue_transport.py`.
4. Goal recovery and atomic publish: `services/queue/atomic_publish.py`, `apps/api_gateway/queue_api.py`, `services/goals/jobs.py`, `services/goals/plan.py`, `services/goals/research.py`, `services/goals/store.py`, goal tests, `tests/test_queue_publication_consistency.py`, `tests/test_queue_api_atomic_redis7.py`.
5. Performance: `services/runtime/`, Deepgram and Sarvam providers, `services/meeting_extension/processor.py`, `services/meeting_extension/ffmpeg_audio.py`, `tests/test_worker_perf.py`, `scripts/benchmark_worker_perf.py`.
6. Phase 2E.1 safety: `services/queue/transport_security.py`, `services/queue/http_queue.py`, `services/queue/streams.py`, `services/queue/maintenance.py`, `services/goals/publisher.py`, `services/mindmap/enqueue.py`, `services/document_it/enqueue.py`, mindmap and document workers, `apps/api_gateway/config/setting.py`, `tests/test_stream_retention.py`, `tests/test_queue_transport_security.py`, `.env.aws.example` only if it is force-added, which it must not be.
7. Docs: `docs/release-readiness/phase2e/`, `README.md` if the README hunk is the retention note.

Files such as `DockerDeployee.md`, `services/chat/*` other than `node_client.py`, and `apps/api_gateway/middleware/logging.py` were already dirty before Phase 2E.1. Read those hunks in the commit review. Do not drop them to make the branch look smaller, and do not mix them into the retention commit if they are unrelated.

## Dependency order when the commits are deployed

Indexes on the staging database, then one Queue API, then one worker, then Node, then Python. Same secret on Node and Python. Queue credentials match and are not the example placeholders. Flag `list`. One replica. `WORKER_CONCURRENCY` stays at the compose default of 1.

## Rollback restriction

Do not deploy `d606a98` as the worker while the new publishers are live: that commit still trims streams and still has the old goal recovery. Do not run two goal workers. Details are in `rollback-plan.md`.

## Tests tied to this tree

Recorded in `phase2e1-test-report.md`. They ran against the dirty tree, not against HEAD.
