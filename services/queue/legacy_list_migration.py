"""Move leftover speech and vector Redis List jobs onto the Phase 2A streams.

This module never runs from worker startup. Dry-run only reads list lengths.
Execute is a separate operator action and refuses to run unless the process
environment sets LEGACY_QUEUE_MIGRATION_CONFIRM=yes.

A crash after XADD and before LREM leaves the list item on the processing
list. The next run publishes it again. The logical job id is stable, so the
stream lease suppresses a second durable transcription or embedding.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any, Awaitable, Callable

from redis.exceptions import ResponseError

from services.observability.diagnostics import diag_log
from services.queue.job_lease import reclaim_min_idle_ms
from services.queue.redis_queue import (
    COMPLETED_MIGRATION_FAILED,
    COMPLETED_MIGRATION_PROCESSING,
    COMPLETED_SPEECH_QUEUE,
    EXPORT_LOCK_KEY,
    SPEECH_EVENT_TYPE,
    SPEECH_MIGRATION_FAILED,
    SPEECH_MIGRATION_PROCESSING,
    SPEECH_QUEUE,
    VECTOR_EVENT_TYPE,
    publish_completed_speech_job,
    publish_speech_job,
    speech_vector_transport,
)

EXPORT_LEDGER_KEY = "buddy:speech-vector:export-ledger"
_ENQUEUE_ONCE_LUA = """
if redis.call('SISMEMBER', KEYS[1], ARGV[1]) == 1 then
  return 0
end
redis.call('LPUSH', KEYS[2], ARGV[2])
redis.call('SADD', KEYS[1], ARGV[1])
return 1
"""

CONFIRM_ENV = "LEGACY_QUEUE_MIGRATION_CONFIRM"
Publish = Callable[[Any, str], Awaitable[None]]


class MigrationPoison(ValueError):
    """The list item cannot be parsed. It is parked, not deleted."""


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def parse_legacy_speech_item(raw: Any) -> dict:
    text = _text(raw).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise MigrationPoison("speech backlog item is not json") from error
    if not isinstance(data, dict):
        raise MigrationPoison("speech backlog item is not a job")
    if data.get("eventType") == SPEECH_EVENT_TYPE and isinstance(data.get("payload"), dict):
        data = data["payload"]
    if not str(data.get("job_id") or data.get("jobId") or "").strip():
        raise MigrationPoison("speech backlog item is missing job_id")
    return data


def parse_legacy_completed_item(raw: Any) -> str:
    text = _text(raw).strip()
    if not text:
        raise MigrationPoison("completed speech backlog item is empty")
    if text.startswith("{"):
        try:
            data = json.loads(text)
        except json.JSONDecodeError as error:
            raise MigrationPoison("completed speech backlog item is not a job id") from error
        if isinstance(data, dict):
            if data.get("eventType") == VECTOR_EVENT_TYPE and isinstance(data.get("payload"), dict):
                data = data["payload"]
            job_id = str(data.get("jobId") or data.get("job_id") or "").strip()
            if not job_id:
                raise MigrationPoison("completed speech backlog item is missing job_id")
            return job_id
    if len(text) > 200:
        raise MigrationPoison("completed speech backlog item is not a job id")
    return text


async def inventory(redis, *, speech_queue: str = SPEECH_QUEUE, completed_queue: str = COMPLETED_SPEECH_QUEUE) -> dict[str, int]:
    keys = (
        speech_queue,
        f"{speech_queue}:migrating" if speech_queue != SPEECH_QUEUE else SPEECH_MIGRATION_PROCESSING,
        f"{speech_queue}:migration_failed" if speech_queue != SPEECH_QUEUE else SPEECH_MIGRATION_FAILED,
        completed_queue,
        f"{completed_queue}:migrating" if completed_queue != COMPLETED_SPEECH_QUEUE else COMPLETED_MIGRATION_PROCESSING,
        f"{completed_queue}:migration_failed" if completed_queue != COMPLETED_SPEECH_QUEUE else COMPLETED_MIGRATION_FAILED,
    )
    counts: dict[str, int] = {}
    for key in keys:
        counts[key] = int(await redis.llen(key) or 0)
    return counts


async def _publish_speech(redis, raw: str) -> None:
    job = parse_legacy_speech_item(raw)
    await publish_speech_job(job, redis=redis)


async def _publish_completed(redis, raw: str) -> None:
    job_id = parse_legacy_completed_item(raw)
    await publish_completed_speech_job(job_id, redis=redis)


async def migrate_list(
    redis,
    *,
    source: str,
    processing: str,
    failed: str,
    publish: Publish,
    execute: bool,
    limit: int = 100,
    queue_name: str,
) -> dict[str, int]:
    """Move up to `limit` items. Dry-run (`execute` false) does not change Redis."""
    source_depth = int(await redis.llen(source) or 0)
    processing_depth = int(await redis.llen(processing) or 0)
    failed_depth = int(await redis.llen(failed) or 0)
    if not execute:
        return {
            "source_depth": source_depth,
            "processing_depth": processing_depth,
            "failed_depth": failed_depth,
            "migrated": 0,
            "failed": 0,
            "execute": 0,
        }

    diag_log(
        "legacy_queue_migration_started",
        queue_name=queue_name,
        legacy_backlog_count=source_depth + processing_depth,
        limit=limit,
    )
    migrated = 0
    failed_count = 0
    while migrated + failed_count < max(1, limit):
        held = await redis.lindex(processing, -1)
        if held is None:
            held = await redis.lmove(source, processing, "RIGHT", "LEFT")
            if held is None:
                break
        raw = _text(held)
        try:
            await publish(redis, raw)
        except MigrationPoison as error:
            await redis.lrem(processing, 1, held)
            await redis.lpush(failed, raw)
            failed_count += 1
            diag_log(
                "legacy_queue_migration_failed",
                queue_name=queue_name,
                reason=str(error)[:200],
            )
            continue
        except Exception as error:
            diag_log(
                "legacy_queue_migration_failed",
                queue_name=queue_name,
                reason=type(error).__name__,
            )
            raise
        removed = int(await redis.lrem(processing, 1, held) or 0)
        if removed < 1:
            diag_log(
                "legacy_queue_migration_failed",
                queue_name=queue_name,
                reason="processing_copy_missing_after_publish",
            )
            raise RuntimeError("legacy migration published a job but could not retire the processing copy")
        migrated += 1
        diag_log(
            "legacy_queue_message_migrated",
            queue_name=queue_name,
            migration_count=migrated,
        )
    diag_log(
        "legacy_queue_migration_completed",
        queue_name=queue_name,
        migration_count=migrated,
        failed_count=failed_count,
        legacy_backlog_count=int(await redis.llen(source) or 0) + int(await redis.llen(processing) or 0),
    )
    return {
        "source_depth": int(await redis.llen(source) or 0),
        "processing_depth": int(await redis.llen(processing) or 0),
        "failed_depth": int(await redis.llen(failed) or 0),
        "migrated": migrated,
        "failed": failed_count,
        "execute": 1,
    }


async def migrate_speech_and_vector_backlog(
    redis,
    *,
    execute: bool,
    limit: int = 100,
    speech_queue: str = SPEECH_QUEUE,
    completed_queue: str = COMPLETED_SPEECH_QUEUE,
) -> dict[str, Any]:
    if execute and speech_vector_transport() != "stream":
        raise RuntimeError(
            "Refusing to migrate while SPEECH_VECTOR_QUEUE_TRANSPORT is not 'stream'. "
            "List consumers would still be authoritative."
        )
    speech_processing = SPEECH_MIGRATION_PROCESSING if speech_queue == SPEECH_QUEUE else f"{speech_queue}:migrating"
    speech_failed = SPEECH_MIGRATION_FAILED if speech_queue == SPEECH_QUEUE else f"{speech_queue}:migration_failed"
    completed_processing = (
        COMPLETED_MIGRATION_PROCESSING if completed_queue == COMPLETED_SPEECH_QUEUE else f"{completed_queue}:migrating"
    )
    completed_failed = (
        COMPLETED_MIGRATION_FAILED if completed_queue == COMPLETED_SPEECH_QUEUE else f"{completed_queue}:migration_failed"
    )
    before = await inventory(redis, speech_queue=speech_queue, completed_queue=completed_queue)
    speech = await migrate_list(
        redis,
        source=speech_queue,
        processing=speech_processing,
        failed=speech_failed,
        publish=_publish_speech,
        execute=execute,
        limit=limit,
        queue_name=speech_queue,
    )
    completed = await migrate_list(
        redis,
        source=completed_queue,
        processing=completed_processing,
        failed=completed_failed,
        publish=_publish_completed,
        execute=execute,
        limit=limit,
        queue_name=completed_queue,
    )
    return {"before": before, "speech": speech, "completed": completed, "after": await inventory(redis, speech_queue=speech_queue, completed_queue=completed_queue)}


async def enqueue_exported_job(redis, destination: str, job_id: str, payload: str) -> int:
    """LPUSH a rolled-back job once. A crash after this and before XACK does not LPUSH again."""
    return int(await redis.eval(_ENQUEUE_ONCE_LUA, 2, EXPORT_LEDGER_KEY, destination, job_id, payload) or 0)


def _job_id_from_export_item(payload: str) -> str:
    text = payload.strip()
    if text.startswith("{"):
        data = json.loads(text)
        if isinstance(data, dict):
            return str(data.get("jobId") or data.get("job_id") or "").strip()
    return text


async def _pending_count(redis, stream: str, group: str) -> int:
    """Pending entries plus messages published before any consumer read them."""
    try:
        groups = await redis.xinfo_groups(stream)
    except ResponseError:
        return 0
    for row in groups:
        if _text(row.get("name")) == group:
            return int(row.get("pending") or 0) + int(row.get("lag") or 0)
    return 0


async def _matching_retry_count(redis, retry_stream: str, targets: set[str]) -> int:
    count = 0
    start = "-"
    while True:
        try:
            rows = await redis.xrange(retry_stream, min=start, max="+", count=200)
        except ResponseError:
            return count
        if not rows:
            return count
        for message_id, fields in rows:
            if _text(fields.get("targetStream") or "") in targets:
                count += 1
        if len(rows) < 200:
            return count
        start = "(" + _text(rows[-1][0])


async def restore_migrating_list(
    redis,
    *,
    processing: str,
    failed: str,
    destination: str,
    kind: str,
    limit: int = 100,
) -> dict[str, int]:
    """Return half-moved legacy items to the list a BRPOP worker actually reads."""
    restored = 0
    while restored < limit:
        held = await redis.lindex(processing, -1)
        if held is None:
            break
        raw = _text(held)
        try:
            if kind == "speech":
                job = parse_legacy_speech_item(raw)
                job_id = str(job.get("job_id") or job.get("jobId"))
                payload = json.dumps(job)
            else:
                job_id = parse_legacy_completed_item(raw)
                payload = job_id
        except MigrationPoison:
            await redis.lrem(processing, 1, held)
            await redis.lpush(failed, raw)
            continue
        await enqueue_exported_job(redis, destination, job_id, payload)
        removed = int(await redis.lrem(processing, 1, held) or 0)
        if removed < 1:
            raise RuntimeError("rollback restored a migrating job but could not remove the processing copy")
        restored += 1
    return {"restored": restored, "processing_depth": int(await redis.llen(processing) or 0)}


async def export_retry_stream(
    redis,
    *,
    retry_stream: str,
    speech_stream: str,
    vector_stream: str,
    speech_list: str,
    vector_list: str,
    limit: int = 100,
) -> dict[str, int]:
    """Copy speech and vector retries off the shared retry stream, then delete those entries."""
    exported = 0
    rows = await redis.xrange(retry_stream, min="-", max="+", count=max(1, limit))
    for message_id, fields in rows:
        if exported >= limit:
            break
        target = _text(fields.get("targetStream") or "")
        raw_event = fields.get("event")
        if target not in {speech_stream, vector_stream} or not raw_event:
            continue
        if target == speech_stream:
            payload = _speech_list_item(_text(raw_event))
            destination = speech_list
        else:
            payload = _completed_list_item(_text(raw_event))
            destination = vector_list
        job_id = _job_id_from_export_item(payload)
        await enqueue_exported_job(redis, destination, job_id, payload)
        await redis.xdel(retry_stream, message_id)
        exported += 1
    return {"exported": exported}


async def rollback_speech_vector_to_lists(
    redis,
    *,
    execute: bool,
    limit: int = 100,
    speech_stream: str | None = None,
    vector_stream: str | None = None,
    speech_group: str | None = None,
    vector_group: str | None = None,
    speech_list: str = SPEECH_QUEUE,
    vector_list: str = COMPLETED_SPEECH_QUEUE,
    retry_stream: str | None = None,
) -> dict[str, Any]:
    """Pause publishers, then copy every unfinished speech or vector job back to its list.

    The cutover lock stays up when pending entries are still too young to reclaim.
    Run the command again after the lease idle time. A successful pass deletes the lock.
    """
    from apps.api_gateway.config.setting import settings

    speech_stream = speech_stream or settings.REDIS_SPEECH_STREAM
    vector_stream = vector_stream or settings.REDIS_VECTOR_STREAM
    speech_group = speech_group or settings.REDIS_SPEECH_GROUP
    vector_group = vector_group or settings.REDIS_VECTOR_GROUP
    retry_stream = retry_stream or settings.REDIS_RETRY_STREAM
    if not execute:
        return await inventory(redis, speech_queue=speech_list, completed_queue=vector_list)

    await redis.set(EXPORT_LOCK_KEY, "rollback")
    speech_processing = SPEECH_MIGRATION_PROCESSING if speech_list == SPEECH_QUEUE else f"{speech_list}:migrating"
    vector_processing = (
        COMPLETED_MIGRATION_PROCESSING if vector_list == COMPLETED_SPEECH_QUEUE else f"{vector_list}:migrating"
    )
    speech_failed = SPEECH_MIGRATION_FAILED if speech_list == SPEECH_QUEUE else f"{speech_list}:migration_failed"
    vector_failed = COMPLETED_MIGRATION_FAILED if vector_list == COMPLETED_SPEECH_QUEUE else f"{vector_list}:migration_failed"
    try:
        speech_migrating = await restore_migrating_list(
            redis,
            processing=speech_processing,
            failed=speech_failed,
            destination=speech_list,
            kind="speech",
            limit=limit,
        )
        vector_migrating = await restore_migrating_list(
            redis,
            processing=vector_processing,
            failed=vector_failed,
            destination=vector_list,
            kind="vector",
            limit=limit,
        )
        speech = await export_unacked_stream_to_list(
            redis,
            stream=speech_stream,
            group=speech_group,
            destination=speech_list,
            to_list_item=_speech_list_item,
            execute=True,
            limit=limit,
        )
        vector = await export_unacked_stream_to_list(
            redis,
            stream=vector_stream,
            group=vector_group,
            destination=vector_list,
            to_list_item=_completed_list_item,
            execute=True,
            limit=limit,
        )
        retries = await export_retry_stream(
            redis,
            retry_stream=retry_stream,
            speech_stream=speech_stream,
            vector_stream=vector_stream,
            speech_list=speech_list,
            vector_list=vector_list,
            limit=limit,
        )
        still_pending = int(speech.get("still_pending") or 0) + int(vector.get("still_pending") or 0)
        migrating_left = int(speech_migrating["processing_depth"]) + int(vector_migrating["processing_depth"])
        retry_left = await _matching_retry_count(redis, retry_stream, {speech_stream, vector_stream})
        unfinished = still_pending + migrating_left + retry_left
        result = {
            "speech": speech,
            "vector": vector,
            "retries": retries,
            "speech_migrating": speech_migrating,
            "vector_migrating": vector_migrating,
            "still_pending": still_pending,
            "migrating_left": migrating_left,
            "retry_left": retry_left,
            "publishers_paused": 1 if unfinished else 0,
        }
        if unfinished == 0:
            await redis.delete(EXPORT_LEDGER_KEY)
            await redis.delete(EXPORT_LOCK_KEY)
        return result
    except Exception:
        diag_log("legacy_queue_migration_failed", queue_name="rollback", reason="export_interrupted")
        raise


async def export_unacked_stream_to_list(
    redis,
    *,
    stream: str,
    group: str,
    destination: str,
    to_list_item: Callable[[str], str],
    execute: bool,
    limit: int = 100,
) -> dict[str, int]:
    """Copy unacknowledged stream entries back to a legacy list, then acknowledge.

    Ack happens only after LPUSH. The stream entry is not deleted. A stopped
    BRPOP worker can then read the list. Do not run this while a stream
    consumer for the same group is still processing.
    """
    # id 0 is only used when this group has never existed, so every entry is
    # still unprocessed. Do not delete a live group before rollback: recreating
    # it would replay history that workers have already acknowledged.
    pending_before = 0
    try:
        await redis.xgroup_create(stream, group, id="0", mkstream=True)
    except ResponseError as error:
        if "BUSYGROUP" not in str(error):
            raise
    if not execute:
        try:
            info = await redis.xpending(stream, group)
            pending_before = int(info.get("pending") if isinstance(info, dict) else info[0] or 0)
        except ResponseError:
            pending_before = 0
        return {"exported": 0, "pending_before": pending_before, "execute": 0}

    exported = 0
    consumer = "legacy-export"
    # A previous pass may have LPUSH'd and crashed before XACK. Acknowledge those
    # immediately so a restarted export does not leave them for a stream worker.
    pending_rows = await redis.xpending_range(stream, group, min="-", max="+", count=max(1, limit))
    for row in pending_rows:
        message_id = row.get("message_id") if isinstance(row, dict) else row[0]
        stored = await redis.xrange(stream, min=message_id, max=message_id, count=1)
        if not stored:
            continue
        fields = stored[0][1]
        event_raw = fields.get("event") if isinstance(fields, dict) else None
        try:
            item = to_list_item(_text(event_raw or ""))
            job_id = _job_id_from_export_item(item)
        except (MigrationPoison, json.JSONDecodeError):
            continue
        if int(await redis.sismember(EXPORT_LEDGER_KEY, job_id) or 0):
            await redis.xack(stream, group, message_id)
            exported += 1
    # Do not steal a worker that still holds a live lease. A killed worker's
    # pending entry becomes idle after the lease TTL and is copied on the next pass.
    min_idle = reclaim_min_idle_ms()
    while exported < limit:
        claimed = await redis.xautoclaim(
            stream,
            group,
            consumer,
            min_idle_time=min_idle,
            start_id="0-0",
            count=min(50, limit - exported),
        )
        entries = list(claimed[1] if len(claimed) > 1 else [])
        if not entries:
            fresh = await redis.xreadgroup(group, consumer, {stream: ">"}, count=min(50, limit - exported))
            if fresh:
                entries = list(fresh[0][1])
        if not entries:
            break
        for message_id, fields in entries:
            if exported >= limit:
                break
            event_raw = fields.get("event") if isinstance(fields, dict) else None
            if event_raw is None and isinstance(fields, dict):
                event_raw = fields.get(b"event")
            item = to_list_item(_text(event_raw or ""))
            await enqueue_exported_job(redis, destination, _job_id_from_export_item(item), item)
            await redis.xack(stream, group, message_id)
            exported += 1
    return {
        "exported": exported,
        "pending_before": pending_before,
        "still_pending": await _pending_count(redis, stream, group),
        "execute": 1,
    }


def _speech_list_item(event_json: str) -> str:
    from services.queue.streams import EventEnvelope

    event = EventEnvelope.model_validate_json(event_json)
    if event.eventType != SPEECH_EVENT_TYPE:
        raise MigrationPoison("stream entry is not a speech job")
    return json.dumps(event.payload)


def _completed_list_item(event_json: str) -> str:
    from services.queue.streams import EventEnvelope

    event = EventEnvelope.model_validate_json(event_json)
    if event.eventType != VECTOR_EVENT_TYPE:
        raise MigrationPoison("stream entry is not a vector job")
    return str(event.payload.get("jobId") or event.payload.get("job_id"))


def _confirmed() -> bool:
    return os.environ.get(CONFIRM_ENV, "").strip().lower() == "yes"


async def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inventory or migrate legacy speech and vector lists")
    parser.add_argument("--execute", action="store_true", help="Move list items onto streams")
    parser.add_argument(
        "--export-to-lists",
        action="store_true",
        help="Copy unacked stream entries back to the legacy lists, then acknowledge them",
    )
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args(argv)
    if args.execute and args.export_to_lists:
        print("Choose either --execute or --export-to-lists.")
        return 2
    if (args.execute or args.export_to_lists) and not _confirmed():
        print(f"Refusing to change Redis. Set {CONFIRM_ENV}=yes for this process.")
        return 2
    from services.queue.redis_queue import get_conversation_redis_client

    redis = get_conversation_redis_client()
    if args.execute:
        result = await migrate_speech_and_vector_backlog(redis, execute=True, limit=args.limit)
    elif args.export_to_lists:
        result = await rollback_speech_vector_to_lists(redis, execute=True, limit=args.limit)
        if (
            int(result.get("still_pending") or 0)
            or int(result.get("migrating_left") or 0)
            or int(result.get("retry_left") or 0)
        ):
            print(json.dumps(result, sort_keys=True))
            print("Rollback paused publishers. Rerun after in-flight leases expire. The lock is buddy:speech-vector:export-lock.")
            return 3
    else:
        result = await inventory(redis)
    print(json.dumps(result, sort_keys=True))
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_cli()))


if __name__ == "__main__":
    main()
