from __future__ import annotations

import asyncio
from dataclasses import dataclass


CLAIM_LUA = """
local scheduleKey = KEYS[1]
local processingKey = KEYS[2]
local member = ARGV[1]
local now = tonumber(ARGV[2])
local claimUntil = tonumber(ARGV[3])
local score = redis.call('ZSCORE', scheduleKey, member)
if not score then
  return {0, 'missing'}
end
if tonumber(score) > now then
  return {0, 'not_due'}
end
redis.call('ZREM', scheduleKey, member)
redis.call('ZADD', processingKey, claimUntil, member)
return {1, tostring(score)}
"""


@dataclass
class ReminderRedisKeys:
    schedule: str = "buddy:reminder:schedule"
    processing: str = "buddy:reminder:processing"
    retry: str = "buddy:reminder:retry"
    dead_letter: str = "buddy:reminder:dead-letter"

    def payload(self, occurrence_id: str) -> str:
        return f"buddy:reminder:payload:{occurrence_id}"


class InMemoryScheduleStore:
    def __init__(self, keys: ReminderRedisKeys | None = None):
        self.keys = keys or ReminderRedisKeys()
        self.zsets: dict[str, dict[str, float]] = {
            self.keys.schedule: {},
            self.keys.processing: {},
            self.keys.retry: {},
        }
        self.payloads: dict[str, str] = {}
        self.dead_letter: list[str] = []
        self._lock = asyncio.Lock()

    async def due_members(self, zset_key: str, now_ts: int, limit: int) -> list[str]:
        items = [
            (member, score)
            for member, score in self.zsets.get(zset_key, {}).items()
            if score <= now_ts
        ]
        items.sort(key=lambda item: item[1])
        return [member for member, _ in items[:limit]]

    async def claim(self, zset_key: str, member: str, now_ts: int, claim_until: int) -> bool:
        async with self._lock:
            score = self.zsets.get(zset_key, {}).get(member)
            if score is None or score > now_ts:
                return False
            self.zsets[zset_key].pop(member, None)
            self.zsets[self.keys.processing][member] = float(claim_until)
            return True

    async def schedule(self, member: str, score: int, payload: str | None = None) -> None:
        self.zsets[self.keys.schedule][member] = float(score)
        if payload is not None:
            self.payloads[member] = payload

    async def retry_at(self, member: str, score: int) -> None:
        self.zsets[self.keys.retry][member] = float(score)
        self.zsets[self.keys.processing].pop(member, None)

    async def complete(self, member: str) -> None:
        self.zsets[self.keys.processing].pop(member, None)
        self.zsets[self.keys.schedule].pop(member, None)
        self.zsets[self.keys.retry].pop(member, None)

    async def cancel(self, member: str) -> None:
        await self.complete(member)
        self.payloads.pop(member, None)

    async def dead_letter_push(self, payload: str) -> None:
        self.dead_letter.append(payload)

    async def expired_processing(self, now_ts: int) -> list[str]:
        return [
            member
            for member, score in self.zsets[self.keys.processing].items()
            if score <= now_ts
        ]

    async def requeue(self, member: str, score: int) -> None:
        self.zsets[self.keys.processing].pop(member, None)
        self.zsets[self.keys.schedule][member] = float(score)

    async def has_scheduled(self, member: str) -> bool:
        return member in self.zsets[self.keys.schedule]


class RedisScheduleStore:
    def __init__(self, redis, keys: ReminderRedisKeys | None = None):
        self.redis = redis
        self.keys = keys or ReminderRedisKeys()

    async def due_members(self, zset_key: str, now_ts: int, limit: int) -> list[str]:
        return await self.redis.zrangebyscore(zset_key, min=0, max=now_ts, start=0, num=limit)

    async def claim(self, zset_key: str, member: str, now_ts: int, claim_until: int) -> bool:
        result = await self.redis.eval(
            CLAIM_LUA,
            2,
            zset_key,
            self.keys.processing,
            member,
            str(now_ts),
            str(claim_until),
        )
        return bool(result and int(result[0]) == 1)

    async def schedule(self, member: str, score: int, payload: str | None = None) -> None:
        pipe = self.redis.pipeline()
        pipe.zadd(self.keys.schedule, {member: score})
        if payload is not None:
            pipe.set(self.keys.payload(member), payload)
        await pipe.execute()

    async def retry_at(self, member: str, score: int) -> None:
        pipe = self.redis.pipeline()
        pipe.zrem(self.keys.processing, member)
        pipe.zadd(self.keys.retry, {member: score})
        await pipe.execute()

    async def complete(self, member: str) -> None:
        pipe = self.redis.pipeline()
        pipe.zrem(self.keys.schedule, member)
        pipe.zrem(self.keys.processing, member)
        pipe.zrem(self.keys.retry, member)
        await pipe.execute()

    async def cancel(self, member: str) -> None:
        await self.complete(member)
        await self.redis.delete(self.keys.payload(member))

    async def dead_letter_push(self, payload: str) -> None:
        await self.redis.lpush(self.keys.dead_letter, payload)

    async def expired_processing(self, now_ts: int) -> list[str]:
        return await self.redis.zrangebyscore(self.keys.processing, min=0, max=now_ts)

    async def requeue(self, member: str, score: int) -> None:
        pipe = self.redis.pipeline()
        pipe.zrem(self.keys.processing, member)
        pipe.zadd(self.keys.schedule, {member: score})
        await pipe.execute()

    async def has_scheduled(self, member: str) -> bool:
        score = await self.redis.zscore(self.keys.schedule, member)
        return score is not None
