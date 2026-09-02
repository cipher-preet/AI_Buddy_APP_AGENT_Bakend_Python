from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from bson import ObjectId

from services.reminders.occurrence import CLAIMABLE_STATUSES

UTC = timezone.utc


class InMemoryReminderStore:
    def __init__(self, reminders: list[dict] | None = None, tokens: dict[str, list[str]] | None = None):
        self.reminders = {str(item["_id"]): dict(item) for item in reminders or []}
        self.tokens = tokens or {}
        self._lock = asyncio.Lock()

    async def get_by_id(self, reminder_id: str) -> dict | None:
        item = self.reminders.get(str(reminder_id))
        return dict(item) if item else None

    async def claim(self, reminder_id: str, occurrence_id: str, now: datetime) -> dict | None:
        async with self._lock:
            item = self.reminders.get(str(reminder_id))
            if item is None:
                return None
            if item.get("lastDeliveredOccurrenceKey") == occurrence_id:
                return None
            if item.get("deliveryStatus") not in CLAIMABLE_STATUSES:
                return None
            item["deliveryStatus"] = "TRIGGERING"
            item["updatedAt"] = now
            return dict(item)

    async def mark_delivered(
        self,
        reminder_id: str,
        occurrence_id: str,
        now: datetime,
        next_trigger: datetime | None,
        next_occurrence_id: str | None,
        delivery_status: str,
    ) -> None:
        item = self.reminders.get(str(reminder_id))
        if item is None:
            return
        item["lastDeliveredOccurrenceKey"] = occurrence_id
        item["lastTriggerAtUtc"] = now
        item["deliveryStatus"] = delivery_status
        item["retryCount"] = 0
        item["retryAtUtc"] = None
        item["nextTriggerAtUtc"] = next_trigger
        item["scheduledOccurrenceId"] = next_occurrence_id
        item["updatedAt"] = now

    async def mark_retry(self, reminder_id: str, retry_count: int, retry_at: datetime, now: datetime) -> None:
        item = self.reminders.get(str(reminder_id))
        if item is None:
            return
        item["deliveryStatus"] = "RETRY_PENDING"
        item["retryCount"] = retry_count
        item["retryAtUtc"] = retry_at
        item["updatedAt"] = now

    async def mark_failed(self, reminder_id: str, now: datetime) -> None:
        item = self.reminders.get(str(reminder_id))
        if item is None:
            return
        item["deliveryStatus"] = "FAILED"
        item["updatedAt"] = now

    async def upcoming(self, until: datetime) -> list[dict]:
        results = []
        for item in self.reminders.values():
            nxt = item.get("nextTriggerAtUtc")
            if item.get("deliveryStatus") in ("SCHEDULED", "RETRY_PENDING") and nxt and nxt <= until:
                results.append(dict(item))
        return results

    async def tokens_for_user(self, user_id: str) -> list[str]:
        return list(self.tokens.get(str(user_id), []))

    async def delete_token(self, token: str) -> None:
        for user_id, tokens in self.tokens.items():
            self.tokens[user_id] = [item for item in tokens if item != token]


class MongoReminderStore:
    def __init__(self, database):
        self.database = database

    def _id_filter(self, reminder_id: str) -> dict:
        if ObjectId.is_valid(reminder_id):
            return {"_id": ObjectId(reminder_id)}
        return {"_id": reminder_id}

    async def get_by_id(self, reminder_id: str) -> dict | None:
        return await self.database.reminders.find_one(self._id_filter(reminder_id))

    async def claim(self, reminder_id: str, occurrence_id: str, now: datetime) -> dict | None:
        return await self.database.reminders.find_one_and_update(
            {
                **self._id_filter(reminder_id),
                "lastDeliveredOccurrenceKey": {"$ne": occurrence_id},
                "deliveryStatus": {"$in": list(CLAIMABLE_STATUSES)},
            },
            {"$set": {"deliveryStatus": "TRIGGERING", "updatedAt": now}},
            return_document=True,
        )

    async def mark_delivered(
        self,
        reminder_id: str,
        occurrence_id: str,
        now: datetime,
        next_trigger: datetime | None,
        next_occurrence_id: str | None,
        delivery_status: str,
    ) -> None:
        await self.database.reminders.update_one(
            self._id_filter(reminder_id),
            {
                "$set": {
                    "lastDeliveredOccurrenceKey": occurrence_id,
                    "lastTriggerAtUtc": now,
                    "deliveryStatus": delivery_status,
                    "retryCount": 0,
                    "retryAtUtc": None,
                    "nextTriggerAtUtc": next_trigger,
                    "scheduledOccurrenceId": next_occurrence_id,
                    "updatedAt": now,
                }
            },
        )

    async def mark_retry(self, reminder_id: str, retry_count: int, retry_at: datetime, now: datetime) -> None:
        await self.database.reminders.update_one(
            self._id_filter(reminder_id),
            {
                "$set": {
                    "deliveryStatus": "RETRY_PENDING",
                    "retryCount": retry_count,
                    "retryAtUtc": retry_at,
                    "updatedAt": now,
                }
            },
        )

    async def mark_failed(self, reminder_id: str, now: datetime) -> None:
        await self.database.reminders.update_one(
            self._id_filter(reminder_id),
            {"$set": {"deliveryStatus": "FAILED", "updatedAt": now}},
        )

    async def upcoming(self, until: datetime) -> list[dict]:
        cursor = self.database.reminders.find(
            {
                "deliveryStatus": {"$in": ["SCHEDULED", "RETRY_PENDING"]},
                "nextTriggerAtUtc": {"$lte": until},
            }
        )
        return await cursor.to_list(length=5000)

    async def tokens_for_user(self, user_id: str) -> list[str]:
        query: dict[str, Any]
        if ObjectId.is_valid(user_id):
            query = {"userId": {"$in": [user_id, ObjectId(user_id)]}}
        else:
            query = {"userId": user_id}
        cursor = self.database.device_tokens.find(query, {"token": 1})
        docs = await cursor.to_list(length=50)
        return [str(doc["token"]) for doc in docs if doc.get("token")]

    async def delete_token(self, token: str) -> None:
        if not token:
            return
        await self.database.device_tokens.delete_one({"token": token})

    async def upsert_token(self, user_id: str, token: str, platform: str | None = None) -> None:
        now = datetime.now(UTC)
        user_value: Any = ObjectId(user_id) if ObjectId.is_valid(user_id) else user_id
        payload: dict[str, Any] = {
            "userId": user_value,
            "token": token,
            "updatedAt": now,
        }
        if platform:
            payload["platform"] = platform
        await self.database.device_tokens.update_one(
            {"token": token},
            {"$set": payload, "$setOnInsert": {"createdAt": now}},
            upsert=True,
        )
