# Phase 2E.1 Redis stream retention

Date: 2026-11-05. Disposable Redis only: `127.0.0.1:6399`, Redis 7.4.9. No production Redis was contacted. `SPEECH_VECTOR_QUEUE_TRANSPORT` stayed `list`.

## What Redis 7 does

`XADD MAXLEN` and `XTRIM`, exact or approximate (`~`), delete the oldest stream entries. They do not look at consumer groups. A pending entry list can still name an id after the payload is gone. `XCLAIM` then returns nothing. An entry that no group has read yet is deleted the same way. Approximate trim on this Redis left `XLEN` at 201 when the cap was 200.

Reproduction, before the code change, on a unique key that was deleted afterward:

- One entry delivered to group `g1` and not acknowledged. Group `g2` had not read it.
- 500 further `XADD`s with `MAXLEN ~ 200`.
- `XPENDING` still reported 1. `XRANGE` on the original id was empty. `XCLAIM` returned no row.
- A second stream that was never read lost its first entry the same way. `XLEN` was 201.

The lease and fence cannot recover a payload Redis has deleted.

## Inventory

| Stream | Producer | Consumer group | Previous trim | Retry / DLQ | Pending risk | Status |
| --- | --- | --- | --- | --- | --- | --- |
| `buddy:audio:ingestion` | conversation publishers, no MAXLEN on `XADD` | `audio-workers` | maintenance `XTRIM ~ 200` | retry stream, then `buddy:dead-letter` | Trim deleted unread and pending entries | Trim removed |
| `buddy:stt:jobs` | meeting publisher without MAXLEN; two groups share this stream | `stt-workers` and `meeting-video-workers` | maintenance `XTRIM ~ 200` | same | One trim dropped the entry for both groups | Trim removed. No `XDEL` on this stream |
| `buddy:transcript:ready` | conversation pipeline | `transcript-workers` | maintenance `XTRIM ~ 200` | same | Same | Trim removed |
| `buddy:conversation:window-extraction` | conversation pipeline | `window-extraction-workers` | maintenance `XTRIM ~ 200` | same | Same | Trim removed |
| `buddy:conversation:finalization` | conversation pipeline and Node meeting publisher | `finalization-workers` | maintenance `XTRIM ~ 200` | same | Same | Trim removed |
| `buddy:conversation:processing` | conversation pipeline | `conversation-processing-workers` | maintenance `XTRIM ~ 200` | same | Same | Trim removed |
| `buddy:conversation:schedule-extraction` | schedule publisher | `schedule-extraction-workers` | maintenance `XTRIM ~ 200` | same | Same | Trim removed |
| `buddy:daily-briefing:jobs` | briefing scheduler | `daily-briefing-workers` | maintenance `XTRIM ~ 200` | same | Same | Trim removed |
| `buddy:meeting:video-merge` | meeting publisher | `meeting-merge-workers` | maintenance `XTRIM ~ 200` | same | Same | Trim removed |
| `buddy:conversation:retry` | `RedisStreamConsumer` after a failed attempt, no MAXLEN on that `XADD` | none in maintenance | maintenance `XTRIM ~ 200` with no group | the stream is the retry queue | Delayed retries were deleted by age in the stream, not by due time | Trim removed |
| `buddy:dead-letter` | consumer, goal worker, reaper | none in maintenance | `XADD MAXLEN ~ 50` on some writers, maintenance `XTRIM ~ 50` | this is the dead letter | Older failures disappeared before inspection | Trim removed |
| `buddy:mindmap:jobs` | `services/mindmap/enqueue.py` | `mindmap-workers` | `XADD MAXLEN ~ 200`, startup exact trim, trim after ack | requeue on the same stream for the cloud client; DLQ otherwise | Pending and unread entries trimmed | `XADD`/`XTRIM` removed. `XDEL` still deletes the one acknowledged id |
| `buddy:document:jobs` | `services/document_it/enqueue.py` | `document-workers` | same as mindmap | same | Same | Same as mindmap |
| `buddy:goals:interactive` | Node `Goals/queue.ts` and `services/goals/publisher.py` | `goal-interactive-workers` | `XADD MAXLEN ~ 200`, exact trim when the group is created, approximate trim after every ack | goal DLQ write also had MAXLEN 50 | Same, and startup trim ran even when the consumer was behind | Trim removed. `XDEL` of the acknowledged id remains |
| `buddy:goals:jobs` | same publishers | `goal-workers` | same | same | Same | Same |
| `buddy:speech:jobs` | Queue API Lua `XADD` with no MAXLEN | `speech-workers` | none | retry / DLQ paths in the stream consumer | No automatic trim | Unchanged |
| `buddy:vector:jobs` | same | `vector-workers` | none | same | No automatic trim | Unchanged |
| `speech_transcribe_queue`, `completed_speech_queue` | list `LPUSH` while the flag is `list` | `BRPOP`, not a group | none | n/a | Lists are not trimmed by this change | Unchanged. Flag stays `list` |

`REDIS_MINDMAP_STREAM_MAXLEN`, `REDIS_DOCUMENT_STREAM_MAXLEN`, `REDIS_GOAL_STREAM_MAXLEN`, and `REDIS_CLOUD_DEAD_LETTER_MAXLEN` still load so old environment files do not fail validation. Nothing reads them to trim.

The reaper still dead-letters one pending id after 15 minutes idle or 8 deliveries, and only then acks and deletes that id. A failed dead-letter write leaves the pending entry in place. That path was not changed.

## Memory

A small goal envelope in a local fixture was 332 bytes. Production envelopes were not measured, and no production ingress rate is in the repository. Do not treat the following as a measured capacity plan.

Assumption, labeled as an assumption: 10,000 entries of 332 bytes are about 3.3 MB of payload, before Redis stream overhead and the AOF. A stuck consumer makes that grow without a cap. That is the trade for not deleting unacknowledged work.

Compose Redis has no `maxmemory` and therefore no eviction. Do not set `allkeys-lru`. An eviction policy would drop stream entries the same way `MAXLEN` did.

Watch, on the staging Redis only:

- `XLEN` per stream
- `XINFO GROUPS` lag and pending
- `XLEN` of `buddy:conversation:retry` and `buddy:dead-letter`
- `used_memory` and the AOF file size on the volume

Operator cleanup, later and not in this release: an entry may be deleted only when every required group on that stream reports the id as delivered and acknowledged, and no group still has lag that includes it. Pending count 0 is not enough while lag is greater than 0. `buddy:stt:jobs` has two groups, so both must be past the id. No such cleaner is implemented.

## Tests

`tests/test_stream_retention.py` on Redis 7.4.9 port 6399. 500 messages after the old cap of 200. Pending payload reclaimed, unread payload readable by two groups, maintenance trimmed 0, dead-letter length 60 against the old cap of 50, acknowledged id deleted without deleting its neighbor. Details are in `phase2e1-test-report.md`.
