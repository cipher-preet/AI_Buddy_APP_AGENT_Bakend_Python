"""Publish a Queue API event and record its dedupe key in one Redis script.

The script validates key types before it writes. Redis 7 does not roll a script
back after a command has succeeded, so a command after LPUSH or XADD must not be
able to fail for a reason the preflight can see. A duplicate retry repairs a
missing speech-job hash without pushing the queue entry again.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from redis.exceptions import ResponseError

from apps.api_gateway.config.setting import settings
from services.observability.diagnostics import diag_log
from services.queue.redis_queue import EXPORT_LOCK_KEY, SpeechVectorCutoverLocked

DEDUPE_PREFIX = "queue_api:event:"
UNUSED_SPEECH_META_KEY = "queue_api:speech-meta:unused"

_PUBLISH_LUA = """
local dedupe = KEYS[1]
local queue = KEYS[2]
local lock = KEYS[3]
local speech_key = KEYS[4]
local mode = ARGV[1]
local payload = ARGV[2]
local payload_hash = ARGV[3]
local ttl = tonumber(ARGV[4])
local correlation = ARGV[5]
local check_lock = ARGV[6]
local write_speech = ARGV[7]
local speech_ttl = tonumber(ARGV[8])
local speech_json = ARGV[9]

local function key_type(key)
  local kind = redis.call('TYPE', key)
  if type(kind) == 'table' then
    return kind['ok'] or ''
  end
  return tostring(kind)
end

local function write_hash(key, fields, hash_ttl)
  local flat = {}
  for field, value in pairs(fields) do
    if type(value) ~= 'table' then
      flat[#flat + 1] = tostring(field)
      flat[#flat + 1] = tostring(value)
    end
  end
  if #flat > 0 then
    redis.call('HSET', key, unpack(flat))
    redis.call('EXPIRE', key, hash_ttl)
  end
end

if mode ~= 'list' and mode ~= 'stream' then
  return redis.error_reply('bad-queue-mode')
end
if ttl == nil or ttl < 1 then
  return redis.error_reply('bad-dedupe-ttl')
end
if payload == nil or payload == '' then
  return redis.error_reply('empty-queue-payload')
end
if write_speech ~= '0' and write_speech ~= '1' then
  return redis.error_reply('bad-speech-flag')
end

local speech_fields = {}
if write_speech == '1' then
  if speech_ttl == nil or speech_ttl < 1 then
    return redis.error_reply('bad-speech-ttl')
  end
  local decoded = cjson.decode(speech_json)
  if type(decoded) ~= 'table' then
    return redis.error_reply('bad-speech-meta')
  end
  speech_fields = decoded
  local speech_type = key_type(speech_key)
  if speech_type ~= 'none' and speech_type ~= 'hash' then
    return redis.error_reply('bad-speech-meta-type')
  end
end

local dedupe_type = key_type(dedupe)
if dedupe_type ~= 'none' and dedupe_type ~= 'hash' then
  return redis.error_reply('bad-dedupe-type')
end
local queue_type = key_type(queue)
if mode == 'list' and queue_type ~= 'none' and queue_type ~= 'list' then
  return redis.error_reply('bad-queue-type')
end
if mode == 'stream' and queue_type ~= 'none' and queue_type ~= 'stream' then
  return redis.error_reply('bad-queue-type')
end

if dedupe_type == 'hash' then
  local existing_hash = redis.call('HGET', dedupe, 'payloadHash')
  local existing_id = redis.call('HGET', dedupe, 'redisMessageId') or ''
  local existing_queue = redis.call('HGET', dedupe, 'stream') or queue
  if existing_hash and existing_hash ~= '' and existing_hash ~= payload_hash then
    return {'conflict', existing_id, existing_queue}
  end
  if write_speech == '1' and redis.call('EXISTS', speech_key) == 0 then
    write_hash(speech_key, speech_fields, speech_ttl)
  end
  redis.call('EXPIRE', dedupe, ttl)
  return {'duplicate', existing_id, existing_queue}
end

if check_lock == '1' and redis.call('EXISTS', lock) == 1 then
  return redis.error_reply('speech-vector-export-locked')
end

local message_id
if mode == 'list' then
  redis.call('LPUSH', queue, payload)
  message_id = 'list'
else
  message_id = redis.call('XADD', queue, '*', 'event', payload)
end

redis.call('HSET', dedupe,
  'state', 'published',
  'stream', queue,
  'redisMessageId', message_id,
  'payloadHash', payload_hash,
  'correlationId', correlation)
redis.call('EXPIRE', dedupe, ttl)
if write_speech == '1' and redis.call('EXISTS', speech_key) == 0 then
  write_hash(speech_key, speech_fields, speech_ttl)
end
return {'published', message_id, queue}
"""


@dataclass(frozen=True)
class PublishResult:
    outcome: str
    message_id: str
    queue: str


class QueuePayloadConflict(RuntimeError):
    """The same event id arrived again with a different submission."""


def dedupe_key(event_id: str) -> str:
    return f"{DEDUPE_PREFIX}{event_id}"


def dedupe_ttl_seconds() -> int:
    """Cover ordinary HTTP retries. Processing idempotency does not depend on this TTL."""
    presign = int(settings.S3_PRESIGNED_URL_TTL_SECONDS or 0) * 24
    return max(86400, presign)


def submission_fingerprint(event: Any) -> str:
    """Identity of the submitted work. Timestamps and attempts are not part of it."""
    payload = getattr(event, "payload", None) or {}
    body = {
        "eventType": str(getattr(event, "eventType", "") or ""),
        "userId": str(getattr(event, "userId", "") or ""),
        "spaceId": str(getattr(event, "spaceId", "") or ""),
        "conversationId": str(getattr(event, "conversationId", "") or ""),
        "payload": payload if isinstance(payload, dict) else {},
    }
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value or "")


async def atomic_publish(
    redis,
    *,
    event_id: str,
    queue: str,
    mode: str,
    payload: str,
    fingerprint: str,
    correlation_id: str,
    check_export_lock: bool = False,
    event_type: str = "",
    speech_job_key: str = "",
    speech_job_fields: dict[str, Any] | None = None,
    speech_job_ttl: int = 0,
) -> PublishResult:
    """XADD or LPUSH and store the dedupe record, or return the existing publication.

    Speech metadata is written in the same script. A later duplicate retry fills
    the hash only when it is still missing, and does not push a second copy.
    """
    fields = speech_job_fields or {}
    write_speech = bool(speech_job_key and fields)
    diag_log("queue_publish_attempt", event_type=event_type, logical_job_id=event_id, queue_name=queue)
    try:
        raw = await redis.eval(
            _PUBLISH_LUA,
            4,
            dedupe_key(event_id),
            queue,
            EXPORT_LOCK_KEY,
            speech_job_key if write_speech else UNUSED_SPEECH_META_KEY,
            mode,
            payload,
            fingerprint,
            str(dedupe_ttl_seconds()),
            correlation_id or "",
            "1" if check_export_lock else "0",
            "1" if write_speech else "0",
            str(int(speech_job_ttl) if write_speech else 0),
            json.dumps(fields, separators=(",", ":"), default=str) if write_speech else "{}",
        )
    except ResponseError as error:
        message = str(error)
        if "speech-vector-export-locked" in message:
            diag_log("queue_atomic_publish_error", event_type=event_type, logical_job_id=event_id, reason="export_lock")
            raise SpeechVectorCutoverLocked("speech and vector publishing is paused for rollback") from error
        diag_log("queue_atomic_publish_error", event_type=event_type, logical_job_id=event_id, reason="redis")
        raise
    if not isinstance(raw, (list, tuple)) or len(raw) < 3:
        diag_log("queue_atomic_publish_error", event_type=event_type, logical_job_id=event_id, reason="bad_script_result")
        raise RuntimeError("queue publish script returned an unexpected result")
    result = PublishResult(outcome=_text(raw[0]), message_id=_text(raw[1]), queue=_text(raw[2]))
    if result.outcome == "published":
        diag_log("queue_publish_success", event_type=event_type, logical_job_id=event_id, queue_name=result.queue)
    elif result.outcome == "duplicate":
        diag_log("queue_publish_duplicate", event_type=event_type, logical_job_id=event_id, queue_name=result.queue)
    elif result.outcome == "conflict":
        diag_log("queue_publish_conflict", event_type=event_type, logical_job_id=event_id, queue_name=result.queue)
        raise QueuePayloadConflict(event_id)
    else:
        diag_log("queue_atomic_publish_error", event_type=event_type, logical_job_id=event_id, reason=result.outcome)
        raise RuntimeError("queue publish script returned an unknown outcome")
    return result
