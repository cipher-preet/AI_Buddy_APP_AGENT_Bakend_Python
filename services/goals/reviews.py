from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

from apps.api_gateway.config.setting import settings
from services.goals.progress import chart_actual, consultant_progress, is_done_status
from services.goals.schemas import PlanStrategy, PriorityRecommendation
from services.goals.store import GoalStore, utc_now
from services.llm.models import LLMMessage, StructuredLLMRequest
from services.llm.router import get_llm_router
from services.prompts.loader import load_prompt


class ReviewDraft(BaseModel):
    summary: str
    focus: str = ""
    recommendation: str = ""
    proposeRevision: bool = False
    revisionNote: str = ""
    strategies: list[PlanStrategy] = Field(default_factory=list)
    priorityRecommendations: list[PriorityRecommendation] = Field(default_factory=list)


def _local_now(timezone_name: str) -> datetime:
    try:
        return datetime.now(ZoneInfo(timezone_name or "Asia/Kolkata"))
    except Exception:
        return datetime.now(timezone.utc)


def period_key(period_type: str, moment: datetime) -> str:
    if period_type == "weekly":
        year, week, _day = moment.isocalendar()
        return f"{year}-W{week:02d}"
    return moment.date().isoformat()


def due_for_review(moment: datetime) -> bool:
    return moment.hour > 0 or (moment.hour == 0 and moment.minute >= 5)


def space_findings(context: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for item in context.get("tasks") or []:
        title = str(item.get("title") or "Untitled task")
        done = is_done_status(item.get("status"))
        detail = str(item.get("detail") or "").strip()
        text = f"Space task “{title}” is {'completed' if done else 'open'}."
        if detail:
            text = f"{text} {detail}"
        findings.append(
            {
                "sourceType": "task",
                "sourceId": str(item.get("sourceId") or ""),
                "category": "OUTPUT" if done else "COMMITMENT",
                "text": text[:1000],
            }
        )
    for item in context.get("notes") or []:
        title = str(item.get("title") or "Untitled note")
        detail = str(item.get("detail") or "").strip()
        text = f"Space note “{title}”."
        if detail:
            text = f"{text} {detail}"
        findings.append(
            {
                "sourceType": "note",
                "sourceId": str(item.get("sourceId") or ""),
                "category": "INTENT",
                "text": text[:1000],
            }
        )
    for item in context.get("transcripts") or []:
        title = str(item.get("title") or "Transcript")
        detail = str(item.get("detail") or title).strip()
        findings.append(
            {
                "sourceType": "transcript",
                "sourceId": str(item.get("sourceId") or ""),
                "category": "OUTPUT",
                "text": f"Space transcript: {detail or title}"[:1000],
            }
        )
    return findings[:80]


def _compact_context(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "spaces": context.get("spaces") or [],
        "tasks": [
            {"title": item.get("title"), "status": item.get("status"), "detail": item.get("detail")}
            for item in (context.get("tasks") or [])[:16]
        ],
        "notes": [
            {"title": item.get("title"), "detail": item.get("detail")}
            for item in (context.get("notes") or [])[:12]
        ],
        "transcripts": [
            {"title": item.get("title"), "detail": item.get("detail")}
            for item in (context.get("transcripts") or [])[:8]
        ],
    }


def fallback_coach(goal: dict[str, Any], context: dict[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    """Coach from the plan itself. Space tasks, notes, and transcripts are ignored."""
    del context
    plan = goal.get("approvedPlan") or {}
    title = str(goal.get("title") or "this goal")
    deadline = str(((plan.get("successDefinition") or {}) if isinstance(plan.get("successDefinition"), dict) else {}).get("deadlineLabel") or goal.get("timeframe") or "the deadline")
    open_days = [
        day
        for day in (plan.get("days") or [])
        if isinstance(day, dict) and str(day.get("status") or "open").lower() not in {"done", "skipped"}
    ]
    next_day = open_days[0] if open_days else None
    next_title = str((next_day or {}).get("title") or "")
    if next_title:
        focus = f"Do “{next_title}” before {deadline}."
    else:
        focus = f"Finish the next open step for “{title}” before {deadline}."

    blockers = [row.get("text") for row in evidence if row.get("category") == "BLOCKER"]
    summary_bits = [
        f"The plan for “{title}” is measured on its own steps.",
        f"Deadline: {deadline}.",
    ]
    if blockers:
        summary_bits.append(str(blockers[0]))
    summary_bits.append("The outside result is not done just because a step is checked.")

    recommendations = list(plan.get("priorityRecommendations") or [])[:5]
    if next_title and not recommendations:
        recommendations = [
            {
                "title": next_title,
                "reason": f"{next_title}: this is the next step inside the {deadline} deadline.",
                "priority": "HIGH",
                "howSteps": [f"Complete “{next_title}”.", "Check it only when the result is visible."],
                "measurableGoal": f"“{next_title}” is finished.",
            }
        ]

    strategies = list(plan.get("strategies") or [])
    if not strategies:
        strategies = [
            {
                "title": f"Follow the plan for {title}",
                "body": f"Work the open steps in order and keep {deadline} in view.",
                "measurableGoal": "The next open step is finished.",
                "why": f"Follow the plan for {title}: the deadline is {deadline}.",
                "proofPoints": ["Next step finished"],
                "effort": deadline,
                "tags": ["plan"],
            }
        ]

    return {
        "summary": " ".join(summary_bits)[:1500],
        "focus": focus[:300],
        "recommendation": focus[:800],
        "proposeRevision": False,
        "revisionNote": "",
        "strategies": strategies,
        "priorityRecommendations": recommendations[:5],
        "model": "fallback",
    }


async def _model_review(
    goal: dict[str, Any],
    evidence: list[dict[str, Any]],
    context: dict[str, Any],
    weekly: bool,
) -> dict[str, Any] | None:
    provider_name = settings.CONVERSATION_SEMANTIC_PROVIDER if weekly else settings.CONVERSATION_SEMANTIC_FALLBACK_PROVIDER
    model = settings.CONVERSATION_SEMANTIC_MODEL if weekly else settings.CONVERSATION_SEMANTIC_FALLBACK_MODEL
    provider = get_llm_router().providers.get(provider_name)
    if not provider or getattr(provider, "configured", True) is False or not model:
        return None
    plan = goal.get("approvedPlan") or {}
    payload = {
        "title": goal.get("title"),
        "successCriteria": goal.get("successCriteria"),
        "healthStatus": goal.get("healthStatus"),
        "progress": goal.get("progress") or {},
        "gaps": goal.get("gaps") or {},
        "period": "weekly" if weekly else "daily",
        "plan": {
            "recommendation": plan.get("recommendation") or plan.get("summary") or "",
            "strategies": (plan.get("strategies") or [])[:6],
            "priorityRecommendations": (plan.get("priorityRecommendations") or [])[:6],
            "milestones": [
                {"title": item.get("title"), "dueLabel": item.get("dueLabel")}
                for item in (plan.get("milestones") or [])[:8]
            ],
        },
        "space": _compact_context(context),
        "evidence": [
            {"category": row.get("category"), "text": row.get("text"), "sourceType": row.get("sourceType")}
            for row in evidence[:16]
        ],
    }
    request = StructuredLLMRequest(
        messages=[
            LLMMessage(role="system", content=load_prompt("goal-review-v1")),
            LLMMessage(role="user", content=json.dumps(payload, ensure_ascii=False, default=str)),
        ],
        model=model,
        temperature=0.2,
        max_tokens=1800,
        metadata={"stage": "goal_review"},
        schema_name=ReviewDraft.__name__,
    )
    try:
        result = await provider.generate_structured(request, ReviewDraft)
    except Exception as error:
        print("goal review model failed", {"error": type(error).__name__}, flush=True)
        return None
    if not isinstance(result, ReviewDraft) or not result.summary.strip():
        return None
    return {
        "summary": result.summary.strip()[:1500],
        "focus": result.focus.strip()[:300],
        "recommendation": (result.recommendation or result.focus).strip()[:800],
        "proposeRevision": bool(weekly and result.proposeRevision and result.revisionNote.strip()),
        "revisionNote": result.revisionNote.strip()[:800],
        "strategies": [item.model_dump() for item in result.strategies],
        "priorityRecommendations": [item.model_dump() for item in result.priorityRecommendations],
        "model": f"{provider_name}:{model}",
    }


def apply_coach_updates(plan: dict[str, Any], drafted: dict[str, Any]) -> dict[str, Any]:
    from services.goals.plan import _normalize_priority_recommendations, _normalize_strategies

    next_plan = dict(plan)
    strategies = _normalize_strategies(list(drafted.get("strategies") or []))
    recommendations = _normalize_priority_recommendations(list(drafted.get("priorityRecommendations") or []))
    if strategies:
        next_plan["strategies"] = strategies
    if recommendations:
        next_plan["priorityRecommendations"] = recommendations
    recommendation = str(drafted.get("recommendation") or "").strip()
    if recommendation:
        next_plan["recommendation"] = recommendation[:1200]
    focus = str(drafted.get("focus") or "").strip()
    if focus:
        next_plan["coachNote"] = focus[:320]
    return next_plan


async def _ingest_space_evidence(
    store: GoalStore, goal: dict[str, Any], findings: list[dict[str, Any]], revision: str
) -> list[dict[str, Any]]:
    now = utc_now()
    wrote = False
    for finding in findings:
        text = str(finding.get("text") or "").strip()
        source_id = str(finding.get("sourceId") or "").strip()
        if not text or not source_id:
            continue
        category = str(finding.get("category") or "OUTPUT")
        await store.db.goal_evidence.update_one(
            {
                "goalId": goal.get("_id"),
                "sourceType": str(finding.get("sourceType") or "space"),
                "sourceId": source_id,
                "sourceRevision": revision,
                "evidenceKey": category,
            },
            {
                "$set": {
                    "ownerId": goal.get("ownerId"),
                    "category": category,
                    "verification": "SOURCE_SUPPORTED",
                    "text": text[:1000],
                    "milestoneTitle": "",
                    "updatedAt": now,
                },
                "$setOnInsert": {"createdAt": now},
            },
            upsert=True,
        )
        wrote = True
    if wrote:
        await store.db.goals.update_one({"_id": goal.get("_id")}, {"$inc": {"evidenceVersion": 1}})
    cursor = store.db.goal_evidence.find({"goalId": goal.get("_id")}).sort("createdAt", -1).limit(80)
    return await cursor.to_list(length=80)


async def _write_snapshot(store: GoalStore, goal: dict[str, Any], period_type: str, period_key_value: str, actual: int) -> None:
    now = utc_now()
    await store.db.goal_snapshots.update_one(
        {"goalId": goal.get("_id"), "periodType": period_type, "periodKey": period_key_value},
        {
            "$set": {"target": 100, "actual": actual, "updatedAt": now},
            "$setOnInsert": {"createdAt": now},
        },
        upsert=True,
    )


async def build_review(store: GoalStore, goal: dict[str, Any], period_type: str, period_key_value: str) -> None:
    context: dict[str, Any] = {"spaces": [], "tasks": [], "notes": [], "transcripts": []}
    evidence: list[dict[str, Any]] = []
    progress = consultant_progress(goal, evidence)
    actual = chart_actual(goal, evidence, context)
    progress = {**progress, "chartActual": actual}
    goal = {**goal, "progress": progress}
    await store.patch_goal(str(goal.get("_id")), {"progress": progress})

    drafted = await _model_review(goal, evidence, context, period_type == "weekly")
    if drafted is None:
        drafted = fallback_coach(goal, context, evidence)

    approved = goal.get("approvedPlan")
    if isinstance(approved, dict) and (drafted.get("strategies") or drafted.get("priorityRecommendations")):
        next_plan = apply_coach_updates(approved, drafted)
        await store.patch_goal(str(goal.get("_id")), {"approvedPlan": next_plan})
        try:
            await store.db.goal_plan_versions.update_one(
                {"goalId": goal.get("_id"), "version": int(goal.get("activePlanVersion") or 1)},
                {"$set": {"plan": next_plan, "updatedAt": utc_now()}},
            )
        except Exception as error:
            print("goal plan coach update skipped", {"error": type(error).__name__}, flush=True)

    now = utc_now()
    await store.db.goal_reviews.update_one(
        {"goalId": goal.get("_id"), "periodType": period_type, "periodKey": period_key_value},
        {
            "$set": {
                "ownerId": goal.get("ownerId"),
                "summary": drafted["summary"],
                "focus": drafted["focus"],
                "healthStatus": goal.get("healthStatus") or "INSUFFICIENT_DATA",
                "spaceGrounded": True,
                "updatedAt": now,
            },
            "$setOnInsert": {"createdAt": now},
        },
        upsert=True,
    )
    await _write_snapshot(store, goal, period_type, period_key_value, actual)
    if period_type == "weekly":
        await _write_snapshot(store, goal, "daily", datetime.now(timezone.utc).date().isoformat(), actual)

    if period_type == "weekly" and drafted.get("proposeRevision") and drafted.get("revisionNote"):
        active = int(goal.get("activePlanVersion") or 1)
        approved = goal.get("approvedPlan") or {}
        if isinstance(approved, dict) and approved.get("milestones"):
            try:
                await store.db.goal_plan_versions.update_one(
                    {"goalId": goal.get("_id"), "version": active + 1, "status": "proposed"},
                    {
                        "$setOnInsert": {
                            "ownerId": goal.get("ownerId"),
                            "plan": {**approved, "version": active + 1, "summary": drafted["revisionNote"]},
                            "createdAt": now,
                            "updatedAt": now,
                        }
                    },
                    upsert=True,
                )
            except Exception as error:
                print("goal plan revision skipped", {"error": type(error).__name__}, flush=True)


async def enqueue_due_reviews(database) -> None:
    goals = await database.goals.find({"status": "ACTIVE"}).limit(200).to_list(length=200)
    for goal in goals:
        moment = _local_now(str(goal.get("timezone") or "Asia/Kolkata"))
        if not due_for_review(moment):
            continue
        for period_type in ("daily", "weekly"):
            if period_type == "weekly" and moment.isoweekday() != 1:
                continue
            key = period_key(period_type, moment)
            existing = await database.goal_reviews.find_one(
                {"goalId": goal.get("_id"), "periodType": period_type, "periodKey": key}
            )
            snapshot = await database.goal_snapshots.find_one(
                {"goalId": goal.get("_id"), "periodType": period_type, "periodKey": key}
            )
            review_ready = bool(existing and existing.get("spaceGrounded") and snapshot is not None and snapshot.get("actual") is not None)
            if review_ready:
                continue
            queued = await database.goal_jobs.find_one(
                {
                    "goalId": goal.get("_id"),
                    "jobType": "review",
                    "status": {"$in": ["queued", "running"]},
                    "payload.periodType": period_type,
                    "payload.periodKey": key,
                }
            )
            if queued:
                continue
            now = utc_now()
            job = await database.goal_jobs.insert_one(
                {
                    "goalId": goal.get("_id"),
                    "ownerId": goal.get("ownerId"),
                    "jobType": "review",
                    "status": "queued",
                    "error": "",
                    "model": "",
                    "payload": {"periodType": period_type, "periodKey": key},
                    "createdAt": now,
                    "updatedAt": now,
                }
            )
            from services.goals.publisher import publish_goal_event

            await publish_goal_event(
                job_id=str(job.inserted_id),
                job_type="review",
                goal_id=str(goal.get("_id")),
                user_id=str(goal.get("ownerId")),
                payload={"periodType": period_type, "periodKey": key},
            )


async def refresh_dirty_goals(_database) -> None:
    """Health is rebuilt in Node when evidenceVersion moves ahead. This process does not clear that flag."""
    return
