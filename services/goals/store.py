from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from bson.errors import InvalidId
from pymongo.errors import DuplicateKeyError


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


def _fence_value(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def research_operation_id(goal_id: Any, job_id: str) -> str:
    """Stable id for one goal job's research. Retries reuse it. A new job does not."""
    return f"goal-research:{goal_id}:{job_id}"


def plan_version_number(plan: dict[str, Any]) -> int:
    try:
        version = int(plan.get("version") or 1)
    except (TypeError, ValueError):
        version = 1
    return version if version >= 1 else 1


def plan_operation_id(goal_id: Any, job_id: str, version: int) -> str:
    return f"goal-plan:{goal_id}:{job_id}:{version}"


def baseline_evidence_operation_id(goal_id: Any, job_id: str) -> str:
    """One baseline job advances evidenceVersion once, including recovery of that job."""
    return f"baseline:{goal_id}:{job_id}"


def space_evidence_operation_id(goal_id: Any, job_id: str) -> str:
    """One space-evidence job advances evidenceVersion once, including recovery of that job."""
    return f"space-evidence:{goal_id}:{job_id}"


def review_evidence_operation_id(goal_id: Any, job_id: str) -> str:
    """One review job advances evidenceVersion once, including recovery of that job."""
    return f"goal-review:{goal_id}:{job_id}"


def space_ingest_operation_id(goal_id: Any, revision: str) -> str:
    """One space-ingest revision advances evidenceVersion once. A new revision is a new operation."""
    return f"space-ingest:{goal_id}:{revision}"


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
        """Claim one goal job.

        Outcomes are queued start (`claimed`), same-attempt resume (`owned`),
        abandoned `running` recovery (`recovered`), durable success (`done`),
        terminal failure (`terminal`), another live attempt (`busy`), or `missing`.
        `running` is not success.
        """
        from services.observability.diagnostics import diag_log
        from services.queue.job_lease import current_lease
        from services.queue.mongo_fence import FENCE_FIELD, lease_fence_token

        oid = as_object_id(job_id)
        if oid is None:
            return "missing", None
        token = lease_fence_token()
        lease = current_lease()
        owner = lease.owner if lease is not None else ""
        now = utc_now()
        diag_log(
            "goal_claim_attempt",
            logical_job_id=job_id,
            attempt=token if token is not None else 0,
            worker_id=owner,
        )
        started = await self.db.goal_jobs.find_one_and_update(
            {"_id": oid, "status": "queued"},
            {
                "$set": {
                    "status": "running",
                    "processingOwner": owner,
                    "processingFence": token if token is not None else 0,
                    "claimedAt": now,
                    "updatedAt": now,
                },
                "$inc": {"processingAttempt": 1},
            },
        )
        if started:
            return "claimed", started

        existing = await self.db.goal_jobs.find_one({"_id": oid})
        if not existing:
            return "missing", None
        status = str(existing.get("status") or "")
        if status == "ready":
            return "done", existing
        if status == "failed":
            return "terminal", existing
        if status != "running":
            return "terminal", existing

        fence = _fence_value(existing.get(FENCE_FIELD))
        if token is None or fence > token:
            diag_log(
                "goal_claim_rejected_active_owner",
                logical_job_id=job_id,
                attempt=token if token is not None else 0,
                worker_id=owner,
                processing_generation=fence,
            )
            return "busy", existing
        if fence == token:
            return "owned", existing

        # Compare-and-set the fence value this caller observed. A peer that already
        # wrote a newer fence does not match, so two reclaimers cannot both proceed.
        if FENCE_FIELD not in existing or existing.get(FENCE_FIELD) is None:
            fence_filter: dict[str, Any] = {
                "$or": [
                    {FENCE_FIELD: {"$exists": False}},
                    {FENCE_FIELD: None},
                    {FENCE_FIELD: 0},
                ]
            }
        else:
            fence_filter = {FENCE_FIELD: existing.get(FENCE_FIELD)}
        recovered = await self.db.goal_jobs.find_one_and_update(
            {"_id": oid, "status": "running", **fence_filter},
            {
                "$set": {
                    "status": "running",
                    "processingOwner": owner,
                    "processingFence": token,
                    "recoveredAt": now,
                    "updatedAt": now,
                },
                "$inc": {"processingAttempt": 1},
            },
        )
        if recovered:
            diag_log(
                "goal_claim_recovered",
                logical_job_id=job_id,
                attempt=token,
                worker_id=owner,
                processing_generation=token,
            )
            return "recovered", recovered
        again = await self.db.goal_jobs.find_one({"_id": oid})
        if not again:
            return "missing", None
        if str(again.get("status") or "") == "ready":
            return "done", again
        if str(again.get("status") or "") == "failed":
            return "terminal", again
        if _fence_value(again.get(FENCE_FIELD)) == token:
            return "owned", again
        diag_log(
            "goal_claim_rejected_active_owner",
            logical_job_id=job_id,
            attempt=token,
            worker_id=owner,
            processing_generation=_fence_value(again.get(FENCE_FIELD)),
        )
        return "busy", again

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
        from services.queue.job_lease import require_active_lease

        await require_active_lease()
        goal_oid = as_object_id(goal_id)
        job_oid = as_object_id(job_id)
        if goal_oid is None or job_oid is None:
            return
        goal = await self.load_goal(goal_id)
        if not goal:
            return
        now = utc_now()
        step_id = str(step.get("stepId") or "")
        durable = {
            "step": step,
            "draft": draft_updates,
            "status": status,
            "assistantText": assistant_text[:4000],
            "model": model,
            "plan": plan,
        }
        from services.queue.mongo_fence import fenced_update_one

        # Persist the step on the job before the goal write. A crash here leaves
        # the job running so recovery can finish this step instead of calling the model again.
        await fenced_update_one(
            self.db.goal_jobs,
            {"_id": job_oid},
            {"$set": {"durableStep": durable, "updatedAt": now}},
        )
        update: dict[str, Any] = {
            "latestStep": step,
            "status": status,
            "updatedAt": now,
            **draft_updates,
        }
        if plan is not None:
            update["proposedPlan"] = plan
        await fenced_update_one(self.db.goals, {"_id": goal_oid}, {"$set": update})
        await self.db.goal_messages.update_one(
            {"goalId": goal_oid, "jobId": job_oid, "stepId": step_id},
            {
                "$set": {
                    "ownerId": goal.get("ownerId"),
                    "role": "assistant",
                    "text": assistant_text[:4000],
                    "control": str(step.get("control") or ""),
                    "updatedAt": now,
                },
                "$setOnInsert": {"createdAt": now},
            },
            upsert=True,
        )
        # The third fenced write. It must stay ahead of the ready update so a crash
        # can still see durableStep and finish the version without another model call.
        if plan is not None:
            await self.save_plan_version(goal, job_id, plan)
        await fenced_update_one(
            self.db.goal_jobs,
            {"_id": job_oid},
            {"$set": {"status": "ready", "error": "", "model": model, "updatedAt": now}},
        )

    async def save_plan_version(self, goal: dict[str, Any], job_id: str, plan: dict[str, Any]) -> None:
        """Upsert one plan version for this goal. A newer fence can replace it. An older fence cannot."""
        from services.queue.mongo_fence import fenced_update_one

        version = plan_version_number(plan)
        now = utc_now()
        goal_id = goal.get("_id")
        try:
            await fenced_update_one(
                self.db.goal_plan_versions,
                {"goalId": goal_id, "version": version},
                {
                    "$set": {
                        "ownerId": goal.get("ownerId"),
                        "status": "proposed",
                        "plan": plan,
                        "operationId": plan_operation_id(goal_id, job_id, version),
                        "updatedAt": now,
                    },
                    "$setOnInsert": {"createdAt": now},
                },
                upsert=True,
            )
        except DuplicateKeyError:
            existing = await self.db.goal_plan_versions.find_one({"goalId": goal_id, "version": version})
            from services.queue.job_lease import StaleOwnershipError, current_lease
            from services.queue.mongo_fence import lease_fence_token

            token = lease_fence_token()
            if existing is not None and token is not None and _fence_value(existing.get("processingFence")) > token:
                lease = current_lease()
                raise StaleOwnershipError(lease.logical_key if lease is not None else job_id, reason="stale_fence")

    async def reconcile_missing_plan_version(self, job: dict[str, Any]) -> bool:
        """Insert a missing version from the job's own durable step. Never replace a stored version."""
        durable = job.get("durableStep") if isinstance(job, dict) else None
        plan = durable.get("plan") if isinstance(durable, dict) else None
        if not isinstance(plan, dict) or not plan.get("milestones"):
            return False
        goal_id = job.get("goalId")
        if goal_id is None:
            return False
        version = plan_version_number(plan)
        existing = await self.db.goal_plan_versions.find_one({"goalId": goal_id, "version": version}, {"_id": 1})
        if existing:
            return False
        now = utc_now()
        try:
            result = await self.db.goal_plan_versions.update_one(
                {"goalId": goal_id, "version": version},
                {
                    "$setOnInsert": {
                        "ownerId": job.get("ownerId"),
                        "status": "proposed",
                        "plan": plan,
                        "operationId": plan_operation_id(goal_id, str(job.get("_id") or ""), version),
                        "createdAt": now,
                        "updatedAt": now,
                        "reconciled": True,
                    }
                },
                upsert=True,
            )
        except DuplicateKeyError:
            return False
        return bool(getattr(result, "upserted_id", None))

    async def fail(self, job_id: str, message: str) -> None:
        from services.queue.job_lease import require_active_lease

        await require_active_lease()
        oid = as_object_id(job_id)
        if oid is None:
            return
        from services.queue.mongo_fence import fenced_update_one

        await fenced_update_one(
            self.db.goal_jobs,
            {"_id": oid},
            {"$set": {"status": "failed", "error": message[:500], "updatedAt": utc_now()}},
        )

    async def patch_goal(self, goal_id: str, updates: dict[str, Any]) -> None:
        oid = as_object_id(goal_id)
        if oid is None or not updates:
            return
        from services.queue.mongo_fence import fenced_update_one

        payload = {**updates, "updatedAt": utc_now()}
        await fenced_update_one(self.db.goals, {"_id": oid}, {"$set": payload})

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

    async def load_completed_research(self, goal_id: Any, operation_id: str) -> dict[str, Any] | None:
        session = await self.db.goal_research_sessions.find_one(
            {"goalId": goal_id, "operationId": operation_id, "writeComplete": True}
        )
        if not session:
            return None
        evidence = await self.db.goal_research_evidence.find({"goalId": goal_id, "operationId": operation_id}).to_list(length=40)
        return {
            "plan": {},
            "sources": list(session.get("sources") or []),
            "evidence": evidence,
            "summary": session.get("summary") or {},
            "cached": [],
            "reused": True,
        }

    async def save_research_session(
        self,
        goal: dict[str, Any],
        summary: dict[str, Any],
        sources: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
        operation_id: str | None = None,
    ) -> None:
        now = utc_now()
        goal_oid = goal.get("_id")
        logical_id = str(operation_id or "").strip()
        if not logical_id:
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
        else:
            from services.queue.mongo_fence import fenced_update_one

            try:
                await fenced_update_one(
                    self.db.goal_research_sessions,
                    {"goalId": goal_oid, "operationId": logical_id},
                    {
                        "$set": {"summary": summary, "sources": sources, "updatedAt": now},
                        "$setOnInsert": {
                            "ownerId": goal.get("ownerId"),
                            "sessionId": str(summary.get("sessionId") or logical_id),
                            "operationId": logical_id,
                            "createdAt": now,
                        },
                    },
                    upsert=True,
                )
            except DuplicateKeyError:
                pass
            for doc in evidence:
                row = dict(doc)
                row["goalId"] = goal_oid
                row["operationId"] = logical_id
                evidence_id = str(row.get("evidenceId") or "")
                if not evidence_id:
                    continue
                try:
                    await self.db.goal_research_evidence.update_one(
                        {"goalId": goal_oid, "operationId": logical_id, "evidenceId": evidence_id},
                        {"$setOnInsert": {**row, "createdAt": row.get("retrievedAt") or now}},
                        upsert=True,
                    )
                except DuplicateKeyError:
                    pass
            await fenced_update_one(
                self.db.goal_research_sessions,
                {"goalId": goal_oid, "operationId": logical_id},
                {"$set": {"writeComplete": True, "updatedAt": now}},
            )
            await fenced_update_one(
                self.db.goals,
                {"_id": goal_oid},
                {"$set": {"researchSummary": summary, "updatedAt": now}},
            )
            return
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


async def ensure_idempotency_indexes(database) -> None:
    """Create idempotency indexes on a disposable database.

    Not called from worker startup. Create these in maintenance before the new
    worker runs. Partial filters leave older session rows without operationId untouched.
    """
    await database.goal_research_sessions.create_index(
        [("goalId", 1), ("operationId", 1)],
        unique=True,
        partialFilterExpression={"operationId": {"$type": "string"}},
        name="goal_research_operation",
    )
    await database.goal_research_evidence.create_index(
        [("goalId", 1), ("operationId", 1), ("evidenceId", 1)],
        unique=True,
        partialFilterExpression={"operationId": {"$type": "string"}},
        name="goal_research_evidence_operation",
    )
    await database.goal_plan_versions.create_index(
        [("goalId", 1), ("version", 1)],
        unique=True,
        name="goal_plan_version",
    )
    await database.goal_evidence.create_index(
        [("goalId", 1), ("sourceType", 1), ("sourceId", 1), ("sourceRevision", 1), ("evidenceKey", 1)],
        unique=True,
        name="goal_evidence_identity",
    )
