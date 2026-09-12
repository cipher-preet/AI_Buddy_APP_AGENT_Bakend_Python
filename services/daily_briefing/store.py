from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError

from services.conversation.models import utc_now
from services.daily_briefing.schemas import (
    PIPELINE_VERSION,
    BriefingStatus,
    DailyBriefingDocument,
    DailyBriefingSynthesis,
    SourceStats,
)

CLAIM_STALE = timedelta(minutes=15)
INTERNAL_OMIT = {"missedCandidates": 0}


def _as_utc(value: datetime | None) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def is_stale_for_requeue(doc: dict[str, Any] | None, now: datetime | None = None) -> bool:
    if not doc:
        return False
    status = doc.get("status")
    stamp = None
    if status == BriefingStatus.PENDING.value:
        stamp = doc.get("updatedAt") or doc.get("createdAt")
    elif status == BriefingStatus.PROCESSING.value:
        stamp = doc.get("claimedAt") or doc.get("updatedAt")
    else:
        return False
    parsed = _as_utc(stamp)
    if parsed is None:
        return True
    current = now or utc_now()
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return parsed <= current - CLAIM_STALE


class DailyBriefingStore:
    def __init__(self, database):
        self.db = database
        self.collection = database.daily_briefings

    async def reserve(
        self,
        user_id: str,
        date_key: str,
        timezone_name: str,
        period_start: datetime,
        period_end: datetime,
    ) -> bool:
        existing = await self.collection.find_one({"userId": user_id, "dateKey": date_key})
        if existing and existing.get("status") in {
            BriefingStatus.READY.value,
            BriefingStatus.PROCESSING.value,
            BriefingStatus.SKIPPED.value,
            BriefingStatus.PENDING.value,
        }:
            return False
        now = utc_now()
        seed = {
            "userId": user_id,
            "dateKey": date_key,
            "timezone": timezone_name,
            "periodStartUtc": period_start,
            "periodEndUtc": period_end,
            "status": BriefingStatus.PENDING.value,
            "pipelineVersion": PIPELINE_VERSION,
            "createdAt": now,
            "updatedAt": now,
        }
        if existing is None:
            try:
                await self.collection.insert_one(seed)
                return True
            except DuplicateKeyError:
                return False
        updated = await self.collection.find_one_and_update(
            {
                "userId": user_id,
                "dateKey": date_key,
                "status": BriefingStatus.FAILED.value,
            },
            {"$set": {**seed, "error": None}},
            return_document=ReturnDocument.AFTER,
        )
        return updated is not None

    async def get(self, user_id: str, date_key: str) -> dict[str, Any] | None:
        return await self.collection.find_one(
            {"userId": user_id, "dateKey": date_key},
            INTERNAL_OMIT,
        )

    async def get_latest(self, user_id: str) -> dict[str, Any] | None:
        return await self.collection.find_one(
            {"userId": user_id},
            INTERNAL_OMIT,
            sort=[("dateKey", -1)],
        )

    async def get_latest_ready(self, user_id: str) -> dict[str, Any] | None:
        return await self.collection.find_one(
            {"userId": user_id, "status": BriefingStatus.READY.value},
            INTERNAL_OMIT,
            sort=[("dateKey", -1)],
        )

    async def claim(
        self,
        user_id: str,
        date_key: str,
        timezone_name: str,
        period_start: datetime,
        period_end: datetime,
        job_id: str,
    ) -> tuple[str, dict[str, Any] | None]:
        existing = await self.collection.find_one({"userId": user_id, "dateKey": date_key})
        if existing and existing.get("status") == BriefingStatus.READY.value:
            return "exists", existing
        now = utc_now()
        stale_before = now - CLAIM_STALE
        claimed_at = existing.get("claimedAt") if existing else None
        if (
            existing
            and existing.get("status") == BriefingStatus.PROCESSING.value
            and isinstance(claimed_at, datetime)
            and claimed_at.replace(tzinfo=claimed_at.tzinfo or timezone.utc) > stale_before
            and existing.get("claimedBy") != job_id
        ):
            return "busy", existing

        seed = {
            "userId": user_id,
            "dateKey": date_key,
            "timezone": timezone_name,
            "periodStartUtc": period_start,
            "periodEndUtc": period_end,
            "status": BriefingStatus.PROCESSING.value,
            "pipelineVersion": PIPELINE_VERSION,
            "claimedBy": job_id,
            "claimedAt": now,
            "createdAt": now,
            "updatedAt": now,
        }
        if existing is None:
            try:
                await self.collection.insert_one(seed)
                return "claimed", seed
            except DuplicateKeyError:
                existing = await self.collection.find_one({"userId": user_id, "dateKey": date_key})
                if existing and existing.get("status") == BriefingStatus.READY.value:
                    return "exists", existing

        updated = await self.collection.find_one_and_update(
            {
                "userId": user_id,
                "dateKey": date_key,
                "$or": [
                    {"status": {"$in": [BriefingStatus.PENDING.value, BriefingStatus.FAILED.value, BriefingStatus.SKIPPED.value]}},
                    {"status": BriefingStatus.PROCESSING.value, "claimedAt": {"$lte": stale_before}},
                    {"status": BriefingStatus.PROCESSING.value, "claimedBy": job_id},
                ],
            },
            {
                "$set": {
                    "status": BriefingStatus.PROCESSING.value,
                    "timezone": timezone_name,
                    "periodStartUtc": period_start,
                    "periodEndUtc": period_end,
                    "claimedBy": job_id,
                    "claimedAt": now,
                    "updatedAt": now,
                    "error": None,
                }
            },
            return_document=ReturnDocument.AFTER,
        )
        if updated is None:
            latest = await self.collection.find_one({"userId": user_id, "dateKey": date_key})
            return "busy", latest
        return "claimed", updated

    async def mark_skipped(self, user_id: str, date_key: str, reason: str, stats: SourceStats) -> None:
        now = utc_now()
        await self.collection.update_one(
            {"userId": user_id, "dateKey": date_key},
            {
                "$set": {
                    "status": BriefingStatus.SKIPPED.value,
                    "skipReason": reason,
                    "sourceStats": stats.model_dump(),
                    "updatedAt": now,
                    "generatedAt": now,
                }
            },
        )

    async def mark_failed(self, user_id: str, date_key: str, error: str) -> None:
        await self.collection.update_one(
            {"userId": user_id, "dateKey": date_key},
            {
                "$set": {
                    "status": BriefingStatus.FAILED.value,
                    "error": error[:500],
                    "updatedAt": utc_now(),
                }
            },
        )

    async def save_ready(
        self,
        user_id: str,
        date_key: str,
        timezone_name: str,
        period_start: datetime,
        period_end: datetime,
        synthesis: DailyBriefingSynthesis,
        stats: SourceStats,
    ) -> DailyBriefingDocument:
        now = utc_now()
        document = DailyBriefingDocument(
            userId=user_id,
            dateKey=date_key,
            timezone=timezone_name,
            periodStartUtc=period_start,
            periodEndUtc=period_end,
            status=BriefingStatus.READY,
            headline=synthesis.headline,
            overview=synthesis.overview,
            highlights=synthesis.highlights,
            importantMoments=synthesis.importantMoments,
            completed=synthesis.completed,
            pendingTasks=synthesis.pendingTasks,
            decisions=synthesis.decisions,
            followUps=synthesis.followUps,
            tomorrowFocus=synthesis.tomorrowFocus,
            insights=synthesis.insights,
            people=synthesis.people,
            topics=synthesis.topics,
            tasks=synthesis.tasks,
            meetings=synthesis.meetings,
            missedCandidates=synthesis.missedCandidates,
            sourceStats=stats,
            generatedAt=now,
            updatedAt=now,
        )
        payload = document.model_dump()
        payload["status"] = BriefingStatus.READY.value
        await self.collection.update_one(
            {"userId": user_id, "dateKey": date_key},
            {"$set": payload},
            upsert=True,
        )
        return document
