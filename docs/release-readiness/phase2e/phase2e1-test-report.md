# Phase 2E.1 test report

Date: 2026-11-05. No staging deploy. No paid provider. No production Redis, MongoDB, or S3.

Disposable Redis: `127.0.0.1:6399`, version 7.4.9, confirmed `tcp_port` 6399 before the retention tests. Keys were unique and deleted by the fixture. The user's worker process was not stopped.

Disposable MongoDB: goal tests use `mongodb://127.0.0.1:27017` and a database name created for the test. Version 7.0.12 was confirmed in the Phase 2E session on that port. This run did not print the version again.

## Python, one process, after the retention and transport edits

```text
python -m pytest tests/test_p0_api_auth.py tests/test_goal_persistence_consistency.py tests/test_queue_publication_consistency.py tests/test_worker_perf.py tests/test_goal_crash_recovery.py tests/test_queue_transport.py tests/test_speech_vector_streams.py tests/test_lease_correctness.py tests/test_job_lease_reliability.py tests/test_daily_briefing.py tests/test_daily_briefing_force_guard.py tests/test_meeting_pipeline_invariants.py tests/test_reminder_scheduler.py tests/test_goal_access.py tests/test_stream_retention.py tests/test_queue_transport_security.py -q --tb=line
```

248 passed, 0 failed, 0 skipped, 58.55s.

That command is the Phase 2E selection plus the new files. The earlier Phase 2E run was 222 passed and 1 failed, then the token assertion was corrected and rerun alone. This run is the full selection after that correction and after the retention and HTTPS changes. It is one execution, not a sum of earlier runs.

Included in those 248:

- Token contract, including a payload flip and a signature flip that each change a character of that segment
- Goal persistence and crash recovery on local MongoDB and Redis 6399
- Queue publication
- Speech/vector unit tests with the flag left at `list`
- Lease correctness
- Daily briefing, reminders, meeting invariants, worker performance
- 6 retention tests: pending entry after 500 extra publishes, two groups reading an unread entry, maintenance trim count 0, reaper leaving a healthy backlog, 60 dead-letter entries, ack deleting one id and leaving the neighbor
- 19 transport tests: public HTTP rejected, private HTTP allowed, TLS verify left on, redirects not followed, user bearer rejected, bad HMAC rejected

Not executed, on purpose:

- `tests/test_queue_api_atomic_redis7.py`
- `tests/test_speech_vector_streams_redis7.py`
- `tests/test_lease_redis7_integration.py`

Those files kill a Redis process. They were not pointed at 6399 in this phase.

Live-provider, gold, and staging end-to-end cases were not run.

## Node

```text
npm test
```

108 passed, 0 failed, 0 skipped, about 1.7s. Includes home security, the briefing proxy, and `MeetingExtension/queueTransport.test.ts`.

```text
npx tsc --noEmit --pretty false
```

Exit 0. No files emitted.

## What this does not prove

- A staging Cloud Run revision
- EC2 AOF on the live host
- That the operator's HTTPS name exists
- Exactly-once processing
