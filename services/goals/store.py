from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from bson.errors import InvalidId


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def as_object_id(value: str) -> ObjectId | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return ObjectId(text)
    except (InvalidId, TypeError):
        return None


def id_candidates(value: str) -> list[Any]:
    text = str(value or "").strip()
    values: list[Any] = [text]
    parsed = as_object_id(text)
    if parsed is not None:
        values.append(parsed)
    return values


class GoalStore:
    def __init__(self, database):
        self.db = database

    async def claim(self, job_id: str) -> tuple[str, dict[str, Any] | None]:
        oid = as_object_id(job_id)
        if oid is None:
            return "missing", None
        claimed = await self.db.goal_jobs.find_one_and_update(
            {"_id": oid, "status": "queued"},
            {"$set": {"status": "running", "updatedAt": utc_now()}},
        )
        if claimed:
            return "claimed", claimed
        existing = await self.db.goal_jobs.find_one({"_id": oid})
        if not existing:
            return "missing", None
        return "done", existing

    async def load_goal(self, goal_id: str) -> dict[str, Any] | None:
        oid = as_object_id(goal_id)
        if oid is None:
            return None
        return await self.db.goals.find_one({"_id": oid})

    async def load_messages(self, goal_id: str, limit: int = 12) -> list[dict[str, Any]]:
        oid = as_object_id(goal_id)
        if oid is None:
            return []
        cursor = self.db.goal_messages.find({"goalId": oid}).sort("createdAt", -1).limit(limit)
        rows = await cursor.to_list(length=limit)
        rows.reverse()
        return rows

    async def save_step(
        self,
        goal_id: str,
        job_id: str,
        *,
        step: dict[str, Any],
        draft_updates: dict[str, Any],
        status: str,
        assistant_text: str,
        model: str,
        plan: dict[str, Any] | None = None,
    ) -> None:
        goal_oid = as_object_id(goal_id)
        job_oid = as_object_id(job_id)
        if goal_oid is None or job_oid is None:
            return
        goal = await self.load_goal(goal_id)
        if not goal:
            return
        now = utc_now()
        update: dict[str, Any] = {
            "latestStep": step,
            "status": status,
            "updatedAt": now,
            **draft_updates,
        }
        if plan is not None:
            update["proposedPlan"] = plan
        await self.db.goals.update_one({"_id": goal_oid}, {"$set": update})
        await self.db.goal_messages.insert_one(
            {
                "goalId": goal_oid,
                "ownerId": goal.get("ownerId"),
                "role": "assistant",
                "text": assistant_text[:4000],
                "stepId": str(step.get("stepId") or ""),
                "control": str(step.get("control") or ""),
                "createdAt": now,
                "updatedAt": now,
            }
        )
        await self.db.goal_jobs.update_one(
            {"_id": job_oid},
            {"$set": {"status": "ready", "error": "", "model": model, "updatedAt": now}},
        )

    async def fail(self, job_id: str, message: str) -> None:
        oid = as_object_id(job_id)
        if oid is None:
            return
        await self.db.goal_jobs.update_one(
            {"_id": oid},
            {"$set": {"status": "failed", "error": message[:500], "updatedAt": utc_now()}},
        )

    async def patch_goal(self, goal_id: str, updates: dict[str, Any]) -> None:
        oid = as_object_id(goal_id)
        if oid is None or not updates:
            return
        payload = {**updates, "updatedAt": utc_now()}
        await self.db.goals.update_one({"_id": oid}, {"$set": payload})

    async def load_research_evidence(self, goal_id: str, limit: int = 40) -> list[dict[str, Any]]:
        oid = as_object_id(goal_id)
        if oid is None:
            return []
        cursor = self.db.goal_research_evidence.find({"goalId": oid}).sort("retrievedAt", -1).limit(limit)
        return await cursor.to_list(length=limit)

    async def load_research_sources(self, goal_id: str, owner_id: Any | None = None) -> list[dict[str, Any]]:
        oid = as_object_id(goal_id)
        if oid is None:
            return []
        query: dict[str, Any] = {"goalId": oid}
        if owner_id is not None:
            query["ownerId"] = owner_id
        session = await self.db.goal_research_sessions.find_one(query, sort=[("createdAt", -1)])
        if not session:
            return []
        return list(session.get("sources") or [])

    async def save_research_session(
        self,
        goal: dict[str, Any],
        summary: dict[str, Any],
        sources: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
    ) -> None:
        now = utc_now()
        goal_oid = goal.get("_id")
        await self.db.goal_research_sessions.insert_one(
            {
                "goalId": goal_oid,
                "ownerId": goal.get("ownerId"),
                "sessionId": summary.get("sessionId"),
                "summary": summary,
                "sources": sources,
                "createdAt": now,
                "updatedAt": now,
            }
        )
        if evidence:
            await self.db.goal_research_evidence.insert_many(evidence)
        await self.db.goals.update_one(
            {"_id": goal_oid},
            {
                "$set": {
                    "researchSummary": summary,
                    "updatedAt": now,
                }
            },
        )

    async def linked_space_ids(self, goal_id: str) -> list[str]:
        oid = as_object_id(goal_id)
        if oid is None:
            return []
        cursor = self.db.goal_space_links.find({"goalId": oid})
        rows = await cursor.to_list(length=8)
        return [str(row.get("spaceId")) for row in rows if row.get("spaceId")]
