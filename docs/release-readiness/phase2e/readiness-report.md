# Phase 2E readiness report

Date: 2026-11-05. No staging or production resource was created, changed, or restarted. `SPEECH_VECTOR_QUEUE_TRANSPORT` was not switched. AI prompts, models, and worker concurrency were not changed.

Phase 2E.1 supersedes this verdict. See `phase2e1-readiness-report.md`. The stream-trimming and public-HTTP findings below were code defects and are addressed there. Isolation, commits, and a real HTTPS hostname remain operator work.

Phase 2E.2 prepares the staging design and does not lift this deploy block. See `phase2e2-readiness-report.md`. That file's verdict is `READY FOR AUTHORIZED STAGING PROVISIONING`. It is not a deploy approval.

## Verdict

**BLOCKED FOR STAGING**

The application contracts that were tested locally match the Phase 1 and Phase 2 design. The deploy itself is blocked because the release artifact is not a commit, and because an isolated staging Redis, MongoDB, bucket, and secret set were not evidenced. Deploying from the workstation Cloud Run env file or from git HEAD `d606a98` would miss the new worker or would touch the existing data stores.

This is not `STAGING VALIDATED`. No staging deploy ran. This is not a production approval.

## What was verified

- Node home routes require authentication, bind the user id from the session or bearer, and enforce space ownership. Transcript retrieval is bounded and the Node test asserts the transcript text is not logged.
- Python verifies the Node HMAC bearer, rejects a bad signature, an expiry, a non-string user id, and a missing secret. A missing secret raises `AuthNotConfigured` and the route returns 401.
- Node mints a 600-second bearer for a session-only daily-briefing call and refuses a bearer that belongs to someone else.
- Queue API service credentials are a different check from the user bearer.
- Goal plan versions are written before `ready`. Research sessions upsert by operation id. Baseline evidence increments once per job under the fence. Speech-job metadata is inside the publish script. Unique indexes exist as a function and are not applied at worker startup.
- Deepgram and Sarvam file reads, and the meeting header prefix read, go through `asyncio.to_thread`. Speech and vector consumers stay at concurrency 1. Compose defaults the worker concurrency to 1.
- Code default of `SPEECH_VECTOR_QUEUE_TRANSPORT` is `list`.
- Disposable local Redis 7.4.9 on `127.0.0.1:6399` had AOF enabled. Local MongoDB 7.0.12 answered on `127.0.0.1:27017`.

## What was not verified

- The live Cloud Run revision and its environment.
- EC2 Redis `INFO persistence`.
- Equality of the production Node and Python HMAC secrets.
- Whether the live Queue API uses the example placeholder credentials that are in the gitignored workstation env file.
- Installed Android and Electron binaries.
- Any staging end-to-end case in `staging-e2e-test-plan.md`.

## Local tests

Python: `C:\Users\PREET\AppData\Local\Programs\Python\Python312\python.exe`

Redis restart, Redis kill, and worker-kill integration files were not executed: `tests/test_queue_api_atomic_redis7.py`, `tests/test_speech_vector_streams_redis7.py`, `tests/test_lease_redis7_integration.py`.

```text
python -m pytest tests/test_p0_api_auth.py tests/test_goal_persistence_consistency.py tests/test_queue_publication_consistency.py tests/test_worker_perf.py tests/test_goal_crash_recovery.py tests/test_queue_transport.py tests/test_speech_vector_streams.py tests/test_lease_correctness.py tests/test_job_lease_reliability.py tests/test_daily_briefing.py tests/test_daily_briefing_force_guard.py tests/test_meeting_pipeline_invariants.py tests/test_reminder_scheduler.py tests/test_goal_access.py -q --tb=line
```

Result: 222 passed, 1 failed, 87.26s. The failure was `test_python_verifier_matches_node_contract`. The signature assertion reused the payload's last character, so when the signature already ended with that character the test presented a valid token and expected a rejection. A direct check of a payload with one character changed was rejected. The assertion was updated to flip each segment against its own last character. The single test then passed in 5.96s.

Node, from `Node_Backend`:

```text
npm test
```

Result: 106 passed, 0 failed, 0 skipped, about 27.6s. The script includes home security, daily briefing, meeting ownership, and reminder scheduling.

Not executed: live-provider tests, gold meeting tests, production-data tests, Redis process kill tests, and every staging case.

## End-to-end status

| Workflow | Local tests | Staging test required | Expected result | Status |
| --- | --- | --- | --- | --- |
| Login and dashboard | Node home security, 17 cases | E1 | Own data only, anonymous 401 | Local pass. Staging not run |
| Meeting transcription | Meeting invariants, queue publish tests | E2 | One transcript, duplicate publish is a no-op | Local pass. Staging not run |
| Meeting intelligence | Pipeline invariant tests with stubs | E3 | Owned notes and tasks | Local pass. Staging not run |
| Speech/vector list mode | `test_speech_vector_streams.py`, queue transport | E2 | List keys, flag stays `list` | Local pass. Staging not run |
| Speech/vector stream mode | Unit coverage only. Redis kill file not run | E4 after a second approval | One stream transcript, empty legacy list | Not run |
| Goal recovery and plan version | Goal crash and persistence files | E5 | Version before `ready`, fence holds | Local pass. Staging not run |
| Daily briefing | Python briefing tests and Node proxy tests | E6 | Owner from the token, short-lived session token | Local pass. Staging not run |
| Reminders | Node reminder tests and `test_reminder_scheduler.py` | E7 | Mock delivery on the reminder Redis | Local pass. Staging not run |
| Worker crash | Lease tests that do not kill Redis. Process-kill files not run | E8 | Pending work finishes once | Staging not run |
| Redis interruption | Not rerun. An earlier Phase 2C.1 restart on disposable 6399 was not repeated | E9 | No false 202, AOF keeps earlier entries | Staging not run |
| Mixed load | Not run | E10 | Five stub jobs finish at concurrency 1 | Staging not run |

## Proposed watch points

These are proposals for the staging window, not established SLOs.

- Unexpected authenticated 401 rate above the negative-test count: stop and compare secrets.
- Any Queue API 202 whose list or stream length did not increase: stop publishers.
- A goal in `running` with no heartbeat and age over 90 seconds after the worker is up: page the worker owner.
- Event-loop lag log above the existing 250 ms threshold on consecutive 8 second intervals during a single stub job: record it, do not raise concurrency.
- Pending count that grows for 10 minutes while the worker is healthy: stop the test and keep the entries.

## Staging acceptance checklist

Fill this in only after a real staging run. Every box is open today.

- [ ] Commit ids match the running images.
- [ ] Staging Redis, MongoDB, and bucket are not the production endpoints.
- [ ] HMAC secrets match and are not the production values reused by habit.
- [ ] Queue credentials are not the example placeholders.
- [ ] Speech/vector transport is `list` on the Queue API and the worker.
- [ ] Four unique indexes exist on the staging database.
- [ ] E1 through E3, E5, E6, E7, E8, and E9 passed on staging.
- [ ] E4 stream mode was not switched, or it was switched under the separate approval and then rolled back without deleting streams.
- [ ] A named person recorded the result.

## Artifacts

- `docs/release-readiness/phase2e/architecture-inventory.md`
- `docs/release-readiness/phase2e/environment-matrix.md`
- `docs/release-readiness/phase2e/security-integration-matrix.md`
- `docs/release-readiness/phase2e/data-migration-checklist.md`
- `docs/release-readiness/phase2e/staging-deployment-runbook.md`
- `docs/release-readiness/phase2e/staging-e2e-test-plan.md`
- `docs/release-readiness/phase2e/rollback-plan.md`
- `docs/release-readiness/phase2e/risk-register.md`
- `docs/release-readiness/phase2e/readiness-report.md`
- `scripts/staging/phase2e_preflight.py` (print only)
- Test correction in `tests/test_p0_api_auth.py` so a signature tamper always changes a character
