"""Durable in-flight ownership for Redis Stream jobs.

Identities are not interchangeable:

- Redis message id: PEL entry. Retries and duplicate publishes get new ids.
- Event id: envelope id. The retry relay mints a new one.
- Logical job id: stable work identity derived from the event type and payload.
- Attempt / fence token: monotonic generation stored beside the lease.
- Owner id: the stream consumer name that holds the lease.

A lease is a Redis string with a TTL. Heartbeats extend it and refresh the
pending-entry idle time without increasing the delivery count. XAUTOCLAIM may
run only after the lease TTL, so a healthy long job is not treated as abandoned.
"""

from __future__ import annotations

import asyncio
import contextvars
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from redis.exceptions import RedisError

from apps.api_gateway.config.setting import settings
from services.observability.diagnostics import diag_log

_LEASE_PREFIX = "buddy:job-lease:"
_FENCE_KEY = "buddy:job-fence:global"
_DONE_PREFIX = "buddy:job-done:"

# One Redis script, so a pause longer than the old 2s lock cannot interleave two owners.
_ACQUIRE_LUA = """
-- job-lease-acquire
local function split(value)
  local parts = {}
  local start = 1
  while true do
    local found = string.find(value, "|", start, true)
    if not found then
      table.insert(parts, string.sub(value, start))
      break
    end
    table.insert(parts, string.sub(value, start, found - 1))
    start = found + 1
  end
  return parts
end
local done = redis.call("GET", KEYS[1])
if done and done ~= "" then
  return {"completed", "", ""}
end
local current = redis.call("GET", KEYS[2])
if current and current ~= "" then
  local parts = split(current)
  if parts[1] == "v1" and #parts >= 7 then
    if parts[2] == ARGV[1] and parts[4] == ARGV[2] then
      redis.call("PEXPIRE", KEYS[2], tonumber(ARGV[6]))
      return {"duplicate", parts[3], parts[2]}
    end
    return {"busy", parts[3], parts[2]}
  end
end
local token = tostring(redis.call("INCR", KEYS[3]))
-- The fence counter is shared by every job and is not expired. A 7-day TTL, or a
-- per-job counter, starts again at 1 and cannot overwrite processingFence.
local payload = table.concat({"v1", ARGV[1], token, ARGV[2], ARGV[3], ARGV[4], ARGV[5], ARGV[7]}, "|")
redis.call("SET", KEYS[2], payload, "PX", tonumber(ARGV[6]))
return {"run", token, ARGV[1]}
"""

_RENEW_LUA = """
-- job-lease-renew
local function split(value)
  local parts = {}
  local start = 1
  while true do
    local found = string.find(value, "|", start, true)
    if not found then
      table.insert(parts, string.sub(value, start))
      break
    end
    table.insert(parts, string.sub(value, start, found - 1))
    start = found + 1
  end
  return parts
end
local current = redis.call("GET", KEYS[1])
if not current or current == "" then
  return 0
end
local parts = split(current)
if parts[1] ~= "v1" or parts[2] ~= ARGV[1] or parts[3] ~= ARGV[2] then
  return 0
end
local payload = table.concat({"v1", ARGV[1], ARGV[2], ARGV[3], ARGV[4], ARGV[5], ARGV[6], ARGV[8]}, "|")
redis.call("SET", KEYS[1], payload, "PX", tonumber(ARGV[7]))
return 1
"""

_RELEASE_LUA = """
-- job-lease-release
local function split(value)
  local parts = {}
  local start = 1
  while true do
    local found = string.find(value, "|", start, true)
    if not found then
      table.insert(parts, string.sub(value, start))
      break
    end
    table.insert(parts, string.sub(value, start, found - 1))
    start = found + 1
  end
  return parts
end
local current = redis.call("GET", KEYS[1])
if not current or current == "" then
  return 0
end
local parts = split(current)
if parts[1] ~= "v1" or parts[2] ~= ARGV[1] or parts[3] ~= ARGV[2] then
  return 0
end
redis.call("DEL", KEYS[1])
return 1
"""

_COMPLETE_LUA = """
-- job-lease-complete
local function split(value)
  local parts = {}
  local start = 1
  while true do
    local found = string.find(value, "|", start, true)
    if not found then
      table.insert(parts, string.sub(value, start))
      break
    end
    table.insert(parts, string.sub(value, start, found - 1))
    start = found + 1
  end
  return parts
end
local current = redis.call("GET", KEYS[1])
if not current or current == "" then
  return 0
end
local parts = split(current)
if parts[1] ~= "v1" or parts[2] ~= ARGV[1] or parts[3] ~= ARGV[2] then
  return 0
end
redis.call("SET", KEYS[2], ARGV[2], "PX", tonumber(ARGV[3]))
return 1
"""

# Conversation-scoped stages. A second delivery of the same stage must not run
# beside the first, even when the payload also carries a sequence number.
_CONVERSATION_SERIAL_TYPES = frozenset(
    {
        "conversation.processing.requested",
        "conversation.finalization.requested",
        "conversation.transcript.ready",
        "conversation.schedule_extraction.requested",
        "meeting.video.merge.requested",
    }
)

_current_lease: contextvars.ContextVar["Lease | None"] = contextvars.ContextVar(
    "kukunotes_job_lease",
    default=None,
)


class StaleOwnershipError(RuntimeError):
    """The worker no longer holds the lease and must not commit or acknowledge."""

    def __init__(self, logical_key: str, *, reason: str = "lease_lost") -> None:
        super().__init__(reason)
        self.logical_key = logical_key
        self.reason = reason


class LeaseUnavailable(RuntimeError):
    """The ownership record could not be updated. Leave the Redis message pending."""

    def __init__(self, logical_key: str) -> None:
        super().__init__("lease_unavailable")
        self.logical_key = logical_key


class LeavePending(RuntimeError):
    """Another durable owner is active. Do not acknowledge the Redis message."""

    def __init__(self, logical_key: str) -> None:
        super().__init__("active_owner")
        self.logical_key = logical_key


@dataclass
class Lease:
    redis: Any
    logical_key: str
    owner: str
    token: str
    message_id: str
    event_id: str
    event_type: str
    retry_count: int
    lost: bool = False
    released: bool = False
    settled: bool = False
    reason: str = ""


@dataclass(frozen=True)
class LeaseDecision:
    action: str
    lease: Lease | None = None
    owner: str = ""
    token: str = ""


def reclaim_min_idle_ms() -> int:
    """Idle time before XAUTOCLAIM. Never shorter than the ownership lease."""
    lease_ms = int(settings.JOB_LEASE_TTL_SECONDS) * 1000
    configured = int(settings.REDIS_CLAIM_IDLE_MS or 0)
    if configured < 1000:
        configured = 1000
    return max(configured, lease_ms)


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _iso_stamp(event: Any) -> str:
    created = getattr(event, "createdAt", None)
    if isinstance(created, datetime):
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        return _field(created.astimezone(timezone.utc).isoformat())
    return _field(created)


def _with_generation(kind: str, base: str, payload: dict[str, Any], event: Any) -> str:
    """Separate a new processing generation from a retry of the same one."""
    version = _clean(payload.get("processingVersion"))
    if version:
        base = f"{base}|v:{_field(version)}"
    source = _clean(payload.get("source")).lower()
    if _truthy(payload.get("reprocess")) or "reprocess" in source:
        base = f"{base}|redo:{_iso_stamp(event) or _field(getattr(event, 'eventId', None))}"
    if kind == "conversation.schedule_extraction.requested":
        source_event = _clean(payload.get("sourceEventId"))
        if source_event:
            base = f"{base}|src:{_field(source_event)}"
    return base


def logical_job_key(event: Any) -> str:
    """Stable work identity. Event ids and Redis message ids are not job ids."""
    payload = getattr(event, "payload", None) or {}
    if not isinstance(payload, dict):
        payload = {}
    kind = _field(getattr(event, "eventType", None) or "event", limit=80) or "event"
    job_id = _clean(payload.get("jobId"))
    window_id = _clean(payload.get("windowId"))
    mindmap_id = _clean(payload.get("mindmapId"))
    document_id = _clean(payload.get("documentId"))
    date_key = _clean(payload.get("dateKey"))
    user_id = _clean(getattr(event, "userId", None))
    conversation = _clean(payload.get("conversationId") or getattr(event, "conversationId", None))
    sequence = payload.get("sequenceNumber")
    if sequence is None:
        sequence = payload.get("sequence")

    if job_id and not kind.startswith("conversation."):
        return f"{kind}|job:{_field(job_id)}"
    if window_id:
        return _with_generation(kind, f"{kind}|window:{_field(window_id)}", payload, event)
    if mindmap_id:
        return f"{kind}|mindmap:{_field(mindmap_id)}"
    if document_id:
        return f"{kind}|document:{_field(document_id)}"
    if date_key and user_id and "briefing" in kind:
        base = f"{kind}|user:{_field(user_id)}|date:{_field(date_key)}"
        if _truthy(payload.get("forced")):
            # createdAt survives the retry relay's new event id, and a later force click has a new one.
            base = f"{base}|force:{_iso_stamp(event) or _field(getattr(event, 'eventId', None))}"
        return base
    if kind.startswith("goal."):
        goal_id = _clean(payload.get("goalId")) or conversation
        goal_job = job_id or _clean(getattr(event, "eventId", None))
        return f"{kind}|goal:{_field(goal_id)}|job:{_field(goal_job)}"
    if kind in _CONVERSATION_SERIAL_TYPES and conversation:
        return _with_generation(kind, f"{kind}|conversation:{_field(conversation)}", payload, event)
    if sequence is not None and conversation and str(sequence).strip() != "":
        chunk = _clean(payload.get("chunkId"))
        suffix = f"|chunk:{_field(chunk)}" if chunk else ""
        return f"{kind}|conversation:{_field(conversation)}|seq:{_field(sequence)}{suffix}"
    if conversation and kind.startswith("conversation."):
        return f"{kind}|conversation:{_field(conversation)}"
    return f"{kind}|event:{_field(getattr(event, 'eventId', None) or 'unknown')}"


def current_lease() -> Lease | None:
    return _current_lease.get()


def mark_current_job_completed() -> None:
    """Ask the consumer to remember a terminal success for this logical job."""
    lease = _current_lease.get()
    if lease is not None:
        lease.settled = True


def push_lease(lease: Lease) -> contextvars.Token:
    return _current_lease.set(lease)


def reset_lease(token: contextvars.Token) -> None:
    _current_lease.reset(token)


async def require_active_lease() -> None:
    """No-op outside a leased worker. Raises when this attempt no longer owns the job."""
    lease = _current_lease.get()
    if lease is None:
        return
    if lease.lost or lease.released:
        raise StaleOwnershipError(lease.logical_key, reason=lease.reason or "lease_lost")
    try:
        owned = await renew_lease(lease.redis, lease)
    except (RedisError, OSError, LeaseUnavailable) as error:
        lease.lost = True
        lease.reason = "ownership_uncertain"
        _log(
            "job_lease_expired",
            job_type=lease.event_type,
            logical_job_id=lease.logical_key,
            attempt=lease.token,
            worker_id=lease.owner,
            retry_count=lease.retry_count,
            reason=type(error).__name__,
        )
        raise StaleOwnershipError(lease.logical_key, reason="ownership_uncertain") from error
    if not owned:
        lease.lost = True
        lease.reason = lease.reason or "lease_lost"
        _log(
            "job_lease_expired",
            job_type=lease.event_type,
            logical_job_id=lease.logical_key,
            attempt=lease.token,
            worker_id=lease.owner,
            retry_count=lease.retry_count,
            reason=lease.reason,
        )
        raise StaleOwnershipError(lease.logical_key, reason=lease.reason)


async def acquire_lease(redis: Any, event: Any, *, owner: str, message_id: str) -> LeaseDecision:
    logical_key = logical_job_key(event)
    owner_id = _field(owner, limit=80) or "worker"
    message = _field(message_id, limit=80)
    event_id = _field(getattr(event, "eventId", None), limit=80)
    event_type = _field(getattr(event, "eventType", None), limit=80) or "event"
    retry_count = int(getattr(event, "attempt", 0) or 0)
    ttl_ms = int(settings.JOB_LEASE_TTL_SECONDS) * 1000
    _log(
        "job_claim_attempt",
        job_type=event_type,
        logical_job_id=logical_key,
        worker_id=owner_id,
        retry_count=retry_count,
        message_id=message,
        queue_wait_ms=_queue_wait_ms(event),
    )
    try:
        raw = await redis.eval(
            _ACQUIRE_LUA,
            3,
            _done_key(logical_key),
            _lease_key(logical_key),
            _fence_key(),
            owner_id,
            message,
            event_id,
            event_type,
            str(retry_count),
            str(ttl_ms),
            str(int(time.time() * 1000)),
        )
        action, token, seen_owner = _eval_triple(raw)
    except (RedisError, OSError, LeaseUnavailable) as error:
        _log(
            "job_claim_attempt",
            job_type=event_type,
            logical_job_id=logical_key,
            worker_id=owner_id,
            retry_count=retry_count,
            outcome="unavailable",
            reason=type(error).__name__,
        )
        raise LeaseUnavailable(logical_key) from error

    if action == "completed":
        _log(
            "job_duplicate_suppressed",
            job_type=event_type,
            logical_job_id=logical_key,
            worker_id=owner_id,
            retry_count=retry_count,
            reason="completed",
        )
        return LeaseDecision(action="completed")
    if action == "duplicate":
        _log(
            "job_duplicate_suppressed",
            job_type=event_type,
            logical_job_id=logical_key,
            attempt=token,
            worker_id=owner_id,
            retry_count=retry_count,
            reason="same_delivery",
        )
        return LeaseDecision(action="duplicate", owner=owner_id, token=token)
    if action != "run":
        _log(
            "job_claim_rejected_active_owner",
            job_type=event_type,
            logical_job_id=logical_key,
            attempt=token,
            worker_id=owner_id,
            owner_id=seen_owner,
            retry_count=retry_count,
        )
        return LeaseDecision(action="busy", owner=seen_owner, token=token)

    lease = Lease(
        redis=redis,
        logical_key=logical_key,
        owner=owner_id,
        token=token,
        message_id=message,
        event_id=event_id,
        event_type=event_type,
        retry_count=retry_count,
    )
    _log(
        "job_claim_success",
        job_type=event_type,
        logical_job_id=logical_key,
        attempt=token,
        worker_id=owner_id,
        retry_count=retry_count,
        message_id=message,
        queue_wait_ms=_queue_wait_ms(event),
    )
    return LeaseDecision(action="run", lease=lease, owner=owner_id, token=token)


async def renew_lease(redis: Any, lease: Lease) -> bool:
    if lease.lost or lease.released:
        return False
    ttl_ms = int(settings.JOB_LEASE_TTL_SECONDS) * 1000
    try:
        owned = await redis.eval(
            _RENEW_LUA,
            1,
            _lease_key(lease.logical_key),
            lease.owner,
            lease.token,
            lease.message_id,
            lease.event_id,
            lease.event_type,
            str(lease.retry_count),
            str(ttl_ms),
            str(int(time.time() * 1000)),
        )
    except (RedisError, OSError) as error:
        raise LeaseUnavailable(lease.logical_key) from error
    if _eval_flag(owned):
        return True
    lease.reason = "token_mismatch"
    return False


async def release_lease(redis: Any, lease: Lease) -> None:
    if lease.released:
        return
    lease.released = True
    if lease.lost:
        return
    try:
        await redis.eval(
            _RELEASE_LUA,
            1,
            _lease_key(lease.logical_key),
            lease.owner,
            lease.token,
        )
    except (RedisError, OSError, LeaseUnavailable):
        lease.lost = True
        lease.reason = "ownership_uncertain"


async def remember_completion(redis: Any, lease: Lease) -> bool:
    if lease.lost or lease.released:
        return False
    ttl_ms = int(settings.JOB_COMPLETION_TTL_SECONDS) * 1000
    try:
        stored = await redis.eval(
            _COMPLETE_LUA,
            2,
            _lease_key(lease.logical_key),
            _done_key(lease.logical_key),
            lease.owner,
            lease.token,
            str(ttl_ms),
        )
    except (RedisError, OSError, LeaseUnavailable):
        lease.lost = True
        lease.reason = "ownership_uncertain"
        return False
    if _eval_flag(stored):
        return True
    lease.lost = True
    lease.reason = "token_mismatch"
    return False


async def preserve_pending_delivery(
    redis: Any,
    *,
    stream: str,
    group: str,
    consumer_name: str,
    message_id: str,
) -> None:
    """Reset idle on a message this attempt does not own, without taking it or resetting its delivery count."""
    del consumer_name
    try:
        rows = await redis.xpending_range(stream, group, min=message_id, max=message_id, count=1)
        if not rows:
            return
        row = rows[0]
        deliveries = _delivery_count(row)
        holder = row.get("consumer") if isinstance(row, dict) else None
        if deliveries is None or not holder:
            return
        await redis.xclaim(
            stream,
            group,
            holder,
            0,
            [message_id],
            retrycount=deliveries,
            justid=True,
        )
    except (RedisError, OSError, AttributeError, TypeError):
        return


async def refresh_pending_idle(
    redis: Any,
    *,
    stream: str,
    group: str,
    consumer_name: str,
    message_id: str,
) -> bool:
    """Reset PEL idle without counting the refresh as another delivery."""
    rows = await redis.xpending_range(stream, group, min=message_id, max=message_id, count=1)
    if not rows:
        return False
    deliveries = _delivery_count(rows[0])
    if deliveries is None:
        return False
    await redis.xclaim(
        stream,
        group,
        consumer_name,
        0,
        [message_id],
        retrycount=deliveries,
        justid=True,
    )
    return True


def spawn_heartbeat(
    redis: Any,
    lease: Lease,
    *,
    stream: str,
    group: str,
    consumer_name: str,
    message_id: str,
) -> asyncio.Task[None]:
    return asyncio.create_task(
        heartbeat_loop(
            redis,
            lease,
            stream=stream,
            group=group,
            consumer_name=consumer_name,
            message_id=message_id,
        ),
        name=f"job-lease:{lease.logical_key[:40]}",
    )


async def heartbeat_loop(
    redis: Any,
    lease: Lease,
    *,
    stream: str,
    group: str,
    consumer_name: str,
    message_id: str,
) -> None:
    interval = max(0.05, float(settings.JOB_LEASE_HEARTBEAT_SECONDS))
    failures = 0
    while not lease.released and not lease.lost:
        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise
        if lease.released or lease.lost:
            return
        try:
            owned = await renew_lease(redis, lease)
        except (RedisError, OSError, LeaseUnavailable) as error:
            failures += 1
            _log(
                "job_lease_renewal",
                job_type=lease.event_type,
                logical_job_id=lease.logical_key,
                attempt=lease.token,
                worker_id=lease.owner,
                retry_count=lease.retry_count,
                outcome="error",
                failures=failures,
                reason=type(error).__name__,
            )
            if failures >= 2:
                lease.lost = True
                lease.reason = "ownership_uncertain"
                _log(
                    "job_lease_expired",
                    job_type=lease.event_type,
                    logical_job_id=lease.logical_key,
                    attempt=lease.token,
                    worker_id=lease.owner,
                    retry_count=lease.retry_count,
                    reason="renew_failed",
                )
                return
            continue
        if not owned:
            lease.lost = True
            _log(
                "job_lease_expired",
                job_type=lease.event_type,
                logical_job_id=lease.logical_key,
                attempt=lease.token,
                worker_id=lease.owner,
                retry_count=lease.retry_count,
                reason=lease.reason or "expired",
            )
            return
        failures = 0
        pel_refreshed = False
        try:
            pel_refreshed = await refresh_pending_idle(
                redis,
                stream=stream,
                group=group,
                consumer_name=consumer_name,
                message_id=message_id,
            )
        except (RedisError, OSError, AttributeError, TypeError):
            pel_refreshed = False
        # A missed idle refresh must not drop ownership. The lease token still
        # rejects the worker that XAUTOCLAIMs the pending entry.
        _log(
            "job_lease_renewal",
            job_type=lease.event_type,
            logical_job_id=lease.logical_key,
            attempt=lease.token,
            worker_id=lease.owner,
            retry_count=lease.retry_count,
            outcome="ok",
            pel_refreshed=pel_refreshed,
        )


def apply_job_lease_eval(read, write, delete, incr, pexpire, script: str, numkeys: int, args: tuple | list):
    """In-process twin of the lease scripts. Real Redis runs the Lua; fakes call this."""
    del pexpire  # The fence counter must not receive a TTL.
    keys = [str(item) for item in list(args)[:numkeys]]
    argv = ["" if item is None else str(item) for item in list(args)[numkeys:]]
    if "job-lease-acquire" in script:
        done = read(keys[0])
        if done:
            return ["completed", "", ""]
        current = _parse(read(keys[1]))
        if current is not None:
            if current["owner"] == argv[0] and current["message_id"] == argv[1]:
                pexpire(keys[1], int(argv[5]))
                return ["duplicate", current["token"], current["owner"]]
            return ["busy", current["token"], current["owner"]]
        token = str(int(incr(keys[2])))
        payload = _encode(
            owner=argv[0],
            token=token,
            message_id=argv[1],
            event_id=argv[2],
            event_type=argv[3],
            attempt=int(argv[4] or 0),
        )
        write(keys[1], payload, int(argv[5]))
        return ["run", token, argv[0]]
    if "job-lease-renew" in script:
        current = _parse(read(keys[0]))
        if current is None or current["owner"] != argv[0] or current["token"] != argv[1]:
            return 0
        payload = _encode(
            owner=argv[0],
            token=argv[1],
            message_id=argv[2],
            event_id=argv[3],
            event_type=argv[4],
            attempt=int(argv[5] or 0),
        )
        write(keys[0], payload, int(argv[6]))
        return 1
    if "job-lease-release" in script:
        current = _parse(read(keys[0]))
        if current is None or current["owner"] != argv[0] or current["token"] != argv[1]:
            return 0
        delete(keys[0])
        return 1
    if "job-lease-complete" in script:
        current = _parse(read(keys[0]))
        if current is None or current["owner"] != argv[0] or current["token"] != argv[1]:
            return 0
        write(keys[1], argv[1], int(argv[2]))
        return 1
    raise ValueError("unsupported lease script")


def _lease_key(logical_key: str) -> str:
    return _LEASE_PREFIX + logical_key


def _fence_key(logical_key: str | None = None) -> str:
    """One counter for every job. Per-job counters restart below a stored fence."""
    del logical_key
    return _FENCE_KEY


def _done_key(logical_key: str) -> str:
    return _DONE_PREFIX + logical_key


def _encode(*, owner: str, token: str, message_id: str, event_id: str, event_type: str, attempt: int) -> str:
    heartbeat_ms = int(time.time() * 1000)
    return "|".join(
        (
            "v1",
            owner,
            str(token),
            message_id,
            event_id,
            event_type,
            str(int(attempt)),
            str(heartbeat_ms),
        )
    )


def _parse(raw: Any) -> dict[str, str] | None:
    text = _text(raw)
    if not text:
        return None
    parts = text.split("|")
    if len(parts) < 7 or parts[0] != "v1":
        return None
    return {
        "owner": parts[1],
        "token": parts[2],
        "message_id": parts[3],
        "event_id": parts[4],
        "event_type": parts[5],
        "attempt": parts[6],
    }


def _eval_triple(raw: Any) -> tuple[str, str, str]:
    if isinstance(raw, (list, tuple)):
        values = [_text(item) or "" for item in raw]
    else:
        values = [_text(raw) or ""]
    while len(values) < 3:
        values.append("")
    return values[0], values[1], values[2]


def _eval_flag(raw: Any) -> bool:
    text = _text(raw)
    return text not in {None, "", "0", "false", "False"}


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    text = str(value)
    return text or None


def _clean(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() in {"none", "null", "undefined"}:
        return ""
    return text


def _field(value: Any, limit: int = 120) -> str:
    return _clean(value).replace("|", "_").replace("\n", " ")[:limit]


def _queue_wait_ms(event: Any) -> int | None:
    created = getattr(event, "createdAt", None)
    if not isinstance(created, datetime):
        return None
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return max(0, int((datetime.now(timezone.utc) - created).total_seconds() * 1000))


def _delivery_count(row: Any) -> int | None:
    try:
        if isinstance(row, dict):
            value = row.get("times_delivered", row.get("times"))
            return int(value)
        if isinstance(row, (list, tuple)) and len(row) >= 4:
            return int(row[3])
    except (TypeError, ValueError):
        return None
    return None


def _log(event: str, **fields: Any) -> None:
    try:
        diag_log(event, **fields)
    except Exception:
        return
