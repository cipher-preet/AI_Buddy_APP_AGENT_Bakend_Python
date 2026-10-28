from __future__ import annotations

from typing import Any

from services.goals.baseline import build_baseline
from services.goals.clarify import build_clarify_step
from services.goals.plan import build_plan
from services.goals.store import GoalStore
from services.queue.streams import EventEnvelope, NonRetryableQueueError


class GoalJobHandler:
    def __init__(self, database):
        self.store = GoalStore(database)

    async def handle(self, event: EventEnvelope) -> None:
        payload = event.payload or {}
        goal_id = str(payload.get("goalId") or event.conversationId or "").strip()
        job_id = str(payload.get("jobId") or event.eventId or "").strip()
        job_type = str(payload.get("jobType") or event.eventType.removeprefix("goal.")).strip()
        if not goal_id or not job_id:
            raise NonRetryableQueueError("goal job missing goalId or jobId")

        claim, _job = await self.store.claim(job_id)
        if claim == "missing":
            raise NonRetryableQueueError(f"goal job not found: {job_id}")
        if claim == "done":
            return

        goal = await self.store.load_goal(goal_id)
        if not goal:
            await self.store.fail(job_id, "Goal not found.")
            raise NonRetryableQueueError(f"goal not found: {goal_id}")

        try:
            if job_type == "baseline":
                await self._baseline(goal, job_id, event.userId)
            elif job_type in {"plan", "research"}:
                await self._plan(goal, job_id, event.payload or {}, job_type=job_type)
            elif job_type == "evidence":
                await self._evidence(goal, job_id, event.payload or {})
            elif job_type == "review":
                await self._review(goal, job_id, event.payload or {})
            else:
                await self._clarify(goal, job_id)
        except NonRetryableQueueError:
            raise
        except Exception as error:
            await self.store.fail(job_id, "The goal step failed. Try again.")
            print("goal job failed", {"jobId": job_id, "error": type(error).__name__}, flush=True)

    async def _clarify(self, goal: dict[str, Any], job_id: str) -> None:
        messages = await self.store.load_messages(str(goal.get("_id")))
        built = await build_clarify_step(goal, messages)
        step = built["step"]
        await self.store.save_step(
            str(goal.get("_id")),
            job_id,
            step=step,
            draft_updates=built.get("draft") or {},
            status=built["status"],
            assistant_text=str(step.get("prompt") or ""),
            model=str(built.get("model") or ""),
        )

    async def _baseline(self, goal: dict[str, Any], job_id: str, user_id: str) -> None:
        built = await build_baseline(self.store, goal, user_id)
        step = {
            "stepId": "confirm-baseline",
            "prompt": built["summary"],
            "control": "confirm_baseline",
            "options": [],
            "allowFreeText": True,
            "findings": built.get("findings") or [],
            "milestones": [],
        }
        await self._store_findings(goal, built.get("findings") or [])
        await self.store.save_step(
            str(goal.get("_id")),
            job_id,
            step=step,
            draft_updates={"completeness": 80, "pipelineState": "READY_FOR_RESEARCH"},
            status="CONTEXT_ANALYSIS",
            assistant_text=built["summary"],
            model=str(built.get("model") or ""),
        )

    async def _plan(self, goal: dict[str, Any], job_id: str, payload: dict[str, Any] | None = None, job_type: str = "plan") -> None:
        messages = await self.store.load_messages(str(goal.get("_id")))
        correction = ""
        for row in reversed(messages):
            if row.get("role") == "user":
                correction = str(row.get("text") or "")
                break
        payload = payload or {}
        goal_id = str(goal.get("_id"))
        await self.store.patch_goal(goal_id, {"pipelineState": "RESEARCHING"})
        built = await build_plan(
            {**goal, "baselineCorrection": correction},
            self.store,
            force_refresh=bool(payload.get("forceRefresh") or job_type == "research"),
            previous_plan=(goal.get("approvedPlan") or goal.get("proposedPlan")) if payload.get("preserveProgress") else None,
        )
        milestones = list(built.get("milestones") or [])
        days = list(built.get("days") or [])
        if not days and milestones:
            from services.goals.plan import _days_from_milestones

            days = _days_from_milestones(milestones)
            built["days"] = days
        if not milestones or not days:
            await self.store.fail(job_id, "The plan did not include any steps. Try again.")
            raise NonRetryableQueueError("goal plan missing milestones or steps")
        summary = str(built.get("summary") or "").strip() or "Here is a practical plan for this goal."
        plan = {
            "version": int(built.get("version") or 1),
            "measurementType": built.get("measurementType") or "MILESTONE",
            "summary": summary,
            "mission": built.get("mission") or "",
            "recommendation": built.get("recommendation") or summary,
            "positioningHeadline": built.get("positioningHeadline") or "",
            "positioningBody": built.get("positioningBody") or "",
            "immediateNextStep": built.get("immediateNextStep") or "",
            "funnelTargets": built.get("funnelTargets") or [],
            "successDefinition": built.get("successDefinition") or {},
            "milestones": built["milestones"],
            "phases": built.get("phases") or [],
            "days": built.get("days") or [],
            "strategies": built.get("strategies") or [],
            "priorityRecommendations": built.get("priorityRecommendations") or [],
            "checklists": built.get("checklists") or [],
            "scorecard": built.get("scorecard") or [],
            "assumptions": built.get("assumptions") or [],
            "openQuestions": built.get("openQuestions") or [],
            "agentTrace": built.get("agentTrace") or [],
            "stepEvaluation": built.get("stepEvaluation") or {},
            "understanding": built.get("understanding") or goal.get("understanding") or {},
            "gaps": built.get("gaps") or {},
            "researchSummary": built.get("researchSummary") or {},
            "sources": built.get("sources") or [],
            "validationReport": built.get("validationReport") or {},
            "provisional": bool(built.get("provisional")),
        }
        if built.get("baselineValue") is not None:
            plan["baselineValue"] = built["baselineValue"]
        if built.get("targetValue") is not None:
            plan["targetValue"] = built["targetValue"]
        if built.get("measurementDirection"):
            plan["measurementDirection"] = built["measurementDirection"]
        prompt = plan["recommendation"]
        from services.goals.brief import review_extras

        step = {
            "stepId": "plan-review",
            "prompt": prompt,
            "control": "plan_review",
            "options": [],
            "allowFreeText": False,
            "findings": [],
            "milestones": milestones,
            "mission": plan["mission"],
            "recommendation": plan["recommendation"],
            "successDefinition": plan["successDefinition"],
            "phases": plan["phases"],
            "days": plan["days"][:12],
            "strategies": plan["strategies"],
            "priorityRecommendations": plan["priorityRecommendations"],
            "checklists": plan["checklists"],
            "scorecard": plan["scorecard"],
            **review_extras(plan),
        }
        await self.store.save_step(
            str(goal.get("_id")),
            job_id,
            step=step,
            draft_updates={
                "completeness": 90,
                "pipelineState": "READY",
                "gaps": built.get("gaps") or {},
                "researchSummary": built.get("researchSummary") or {},
                "validationReport": built.get("validationReport") or {},
                "measurementType": plan["measurementType"],
                **(
                    {"baselineValue": built["baselineValue"]}
                    if built.get("baselineValue") is not None and goal.get("baselineValue") is None
                    else {}
                ),
                **(
                    {"targetValue": built["targetValue"]}
                    if built.get("targetValue") is not None and goal.get("targetValue") is None
                    else {}
                ),
                **(
                    {"measurementDirection": built["measurementDirection"]}
                    if built.get("measurementDirection") and not goal.get("measurementDirection")
                    else {}
                ),
            },
            status="PLAN_REVIEW",
            assistant_text=summary,
            model=str(built.get("model") or ""),
            plan=plan,
        )
        await self.store.db.goal_plan_versions.update_one(
            {"goalId": goal.get("_id"), "version": 1},
            {
                "$set": {
                    "ownerId": goal.get("ownerId"),
                    "status": "proposed",
                    "plan": plan,
                    "updatedAt": __import__("services.goals.store", fromlist=["utc_now"]).utc_now(),
                }
            },
            upsert=True,
        )

    async def _store_findings(self, goal: dict[str, Any], findings: list[dict[str, Any]]) -> None:
        from services.goals.store import utc_now

        now = utc_now()
        wrote = False
        for finding in findings:
            text = str(finding.get("text") or "").strip()
            if not text:
                continue
            source_id = str(finding.get("sourceId") or "baseline")
            category = str(finding.get("category") or "OUTPUT")
            await self.store.db.goal_evidence.update_one(
                {
                    "goalId": goal.get("_id"),
                    "sourceType": str(finding.get("sourceType") or "baseline"),
                    "sourceId": source_id,
                    "sourceRevision": "baseline",
                    "evidenceKey": category,
                },
                {
                    "$setOnInsert": {
                        "ownerId": goal.get("ownerId"),
                        "category": category,
                        "verification": "SOURCE_SUPPORTED" if finding.get("sourceId") else "UNVERIFIED",
                        "text": text[:1000],
                        "milestoneTitle": "",
                        "createdAt": now,
                        "updatedAt": now,
                    }
                },
                upsert=True,
            )
            wrote = True
        if wrote:
            await self.store.db.goals.update_one({"_id": goal.get("_id")}, {"$inc": {"evidenceVersion": 1}})

    async def _evidence(self, goal: dict[str, Any], job_id: str, payload: dict[str, Any]) -> None:
        from services.goals.store import utc_now

        title = str(payload.get("title") or "Item").strip()
        status = str(payload.get("status") or "")
        source_type = str(payload.get("sourceType") or "task")
        if source_type == "note":
            category = "INTENT"
            text = f"Space note “{title}” was updated."
        elif source_type == "transcript":
            category = "OUTPUT"
            text = f"Space transcript: {title}"
        elif status == "reopened":
            category = "CORRECTION"
            text = f"Space task “{title}” was reopened."
        elif status in {"completed", "done"}:
            category = "OUTPUT"
            text = f"Space task “{title}” was completed."
        else:
            category = "COMMITMENT"
            text = f"Space task “{title}” was {status or 'added'}."
        milestone = ""
        plan = goal.get("approvedPlan") or {}
        for item in plan.get("milestones") or []:
            name = str(item.get("title") or "")
            if name and (name.lower() in title.lower() or title.lower() in name.lower()):
                milestone = name
                break
        now = utc_now()
        await self.store.db.goal_evidence.update_one(
            {
                "goalId": goal.get("_id"),
                "sourceType": source_type,
                "sourceId": str(payload.get("sourceId") or job_id),
                "sourceRevision": str(payload.get("sourceRevision") or "1"),
                "evidenceKey": category,
            },
            {
                "$set": {
                    "ownerId": goal.get("ownerId"),
                    "category": category,
                    "verification": "SOURCE_SUPPORTED",
                    "text": text[:1000],
                    "milestoneTitle": milestone,
                    "updatedAt": now,
                },
                "$setOnInsert": {"createdAt": now},
            },
            upsert=True,
        )
        await self.store.db.goals.update_one({"_id": goal.get("_id")}, {"$inc": {"evidenceVersion": 1}})
        await self.store.db.goal_jobs.update_one(
            {"_id": __import__("services.goals.store", fromlist=["as_object_id"]).as_object_id(job_id)},
            {"$set": {"status": "ready", "updatedAt": now}},
        )

    async def _review(self, goal: dict[str, Any], job_id: str, payload: dict[str, Any]) -> None:
        from services.goals.reviews import _local_now, build_review, period_key
        from services.goals.store import as_object_id, utc_now

        job = await self.store.db.goal_jobs.find_one({"_id": as_object_id(job_id)})
        merged = {**((job or {}).get("payload") or {}), **(payload or {})}
        period_type = str(merged.get("periodType") or "daily")
        key = str(merged.get("periodKey") or "").strip()
        if not key:
            key = period_key(period_type, _local_now(str(goal.get("timezone") or "Asia/Kolkata")))
        await build_review(self.store, goal, period_type, key)
        await self.store.db.goals.update_one(
            {"_id": goal.get("_id")},
            {"$inc": {"evidenceVersion": 1}},
        )
        await self.store.db.goal_jobs.update_one(
            {"_id": as_object_id(job_id)},
            {"$set": {"status": "ready", "updatedAt": utc_now()}},
        )
