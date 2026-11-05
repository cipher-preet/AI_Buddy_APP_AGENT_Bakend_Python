# Phase 2E staging end-to-end test plan

Use synthetic users and the isolated staging stack from the runbook. Do not upload a real person's recording, call a paid provider, or send a real notification. Local tests already cover the contracts listed in the status column of `readiness-report.md`. Those local passes are not staging passes.

Cleanup for every case: delete the synthetic user's staging documents and the staging Redis keys created for that case. Do not use a pattern delete wider than the test id prefix.

## E1 — Login and dashboard

- Preconditions: Node and Python are on the staging revisions. User A and user B exist. User B owns one space that A must not see.
- Test data: email login or the staging auth fixture. No production account.
- Trigger: sign in as A, open spaces, tasks, and notes.
- Expected API: 200 for A's resources. A's space list does not include B's space id.
- Expected Redis: no new queue entries.
- Expected MongoDB: no writes except a normal session if the session store is staging.
- Expected client: dashboard stays on the signed-in page.
- Expected logs: no transcript body.
- Pass: A sees only A's rows, refresh does not 401, and a copied request without a cookie or bearer returns 401.

## E2 — Meeting transcription (list mode)

- Preconditions: `SPEECH_VECTOR_QUEUE_TRANSPORT=list`. Queue API credentials are the staging pair. Provider is a stub or a sandbox key.
- Test data: a short generated tone in a webm or wav file, not a customer recording. One new `eventId`.
- Trigger: the client upload path, or a direct Queue API publish if the client path is stubbed.
- Expected API: Queue API 202. A second publish of the same `eventId` is a duplicate.
- Expected Redis: `LLEN speech_transcribe_queue` increases by one, then returns to the baseline after the worker finishes. `speech_job:{jobId}` exists. One completion marker.
- Expected MongoDB: one transcript owned by user A.
- Expected client: the meeting shows one transcript.
- Expected logs: Queue API auth success, worker lease acquired and released, no second provider call for that job id.
- Pass: one transcript, duplicate publish does not create a second one.

## E3 — Meeting intelligence

- Preconditions: E2's transcript, or a synthetic transcript inserted as user A.
- Trigger: the existing consolidation and evidence path. Do not change prompts.
- Expected API: notes and tasks for A only.
- Expected Redis: conversation streams return to the baseline pending count.
- Expected MongoDB: notes and tasks have A's user id and a source reference to the synthetic meeting.
- Expected client: A sees the notes. B does not.
- Pass: one set of artifacts, source id matches, B's request is 403 or empty.

## E4 — Speech/vector transport

List mode is E2. Do not switch the flag during the consolidated deploy.

Stream mode, only after the separate approval in the runbook:

- Preconditions: legacy `LLEN` values recorded, worker stopped, no in-flight `BRPOP`, both processes set to `stream`.
- Trigger: one synthetic speech job.
- Expected Redis: one entry on `buddy:speech:jobs`, then one on `buddy:vector:jobs`, pending returns to 0, legacy lists stay at the recorded empty length.
- Expected MongoDB: one transcript and one vector record.
- Retry: kill the staging worker after the entry is pending and before ack. After the lease, one reclaim and one durable result.
- Pass: no job left only on the legacy list.

## E5 — Goal creation and recovery

- Preconditions: indexes from the data checklist exist. One worker.
- Test data: a synthetic goal for user A with a stub planner. No paid model.
- Trigger: create the goal and let the job run.
- Expected API: the job reaches `ready` only after a `goal_plan_versions` row exists for that version.
- Expected Redis: one goal stream entry, then pending 0. Fence does not decrease.
- Expected MongoDB: one research session for `goal-research:{goalId}:{jobId}` with `writeComplete` true, one plan version, baseline `evidenceVersion` advanced once.
- Recovery: stop the worker while a second job is `running`, wait past 90 seconds, start the worker. The job completes once. A write with the old fence does not change the plan.
- Known gap, not a failure of this case: a retried review or space-evidence update may increment `evidenceVersion` more than once. Record the count. Do not treat a double increment on those two paths as a surprise.
- Pass: one plan version, one completed research session, stale fence rejected, job not stuck in `running`.

## E6 — Daily briefing

Do not wait for midnight.

- Preconditions: user A's timezone is `Asia/Kolkata`. Insert one synthetic transcript dated yesterday in that zone. Scheduler interval on staging is shortened only in the staging env, or an operator invokes the internal schedule function once against staging Redis.
- Trigger: the scheduled path, then a force-generate from A's session with no bearer.
- Expected API: force-generate 200 for A and 403 for a body that names B.
- Expected Redis: one briefing job, then pending 0.
- Expected MongoDB: one briefing document owned by A for that date.
- Expected logs: outbound token is not printed. A 401 from Python fails the Node call and does not store a briefing for B.
- Pass: A can read the briefing. B cannot.

## E7 — Reminders

- Preconditions: reminder transport is a mock. `REMINDER_REDIS_URL` is the staging reminder instance.
- Trigger: create a reminder one minute ahead for user A.
- Expected Redis: a schedule entry on the reminder instance, not on the worker's compose Redis.
- Expected client: the mock records one delivery and sends nothing externally.
- Pass: one mock delivery. No change to the reminder key names.

## E8 — Worker crash

On staging only, stop the worker during one synthetic meeting, one speech job, one vector job, and one goal job. Each item should already be pending or leased.

- Expected Redis: the entry stays pending or the lease expires. It is not acknowledged by the stop.
- Expected MongoDB: no `ready` goal without a plan version. No second transcript after restart.
- Pass: after the worker starts, each item finishes once.

## E9 — Redis interruption

Stop the staging Redis container for a few seconds, then start the same volume.

- Expected: publishers receive an error and do not report 202 for a publish that did not land.
- Expected after restart: AOF returns the entries from before the stop, minus at most the `everysec` window. Pending entries are still pending. The fence is not lower than before the stop.
- Pass: no 202 for a failed publish, and previously accepted work is still readable.

## E10 — Mixed load

Keep concurrency at the staging values (worker 1, speech 1, vector 1).

- Trigger: two synthetic meetings, one goal, one briefing, one speech job, all with stubs.
- Watch: pending depth, heartbeat gap, CPU, memory, event-loop lag warnings, and a clean shutdown that leaves unfinished work pending rather than acknowledged.
- Pass: all five finish, pending returns to 0, lag warnings are recorded rather than ignored, and memory returns near the start value after the jobs complete.

## Local coverage already run

The commands and counts are in `readiness-report.md`. They used `127.0.0.1:6399` and `127.0.0.1:27017`. They did not use staging.
