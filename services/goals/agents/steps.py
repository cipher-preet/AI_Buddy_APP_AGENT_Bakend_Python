"""Evaluate weak goal steps and rewrite them with Krutrim gpt-oss-120b.

Plans that already list real actions are left alone. Outcome labels and the
canned "Do the work for …" line are sent to the model, then checked locally.
If the model is unavailable or wanders off the goal, the local actions are used.
"""

from __future__ import annotations

from typing import Any

from services.goals.agents.llm import call_goal_agent, goal_context_payload, goal_step_pairs
from services.goals.agents.scheduler import assign_step_schedule
from services.goals.brief import stated_deadline
from services.goals.schemas import GoalStepWriterResult
from services.goals.step_actions import accept_model_days, steps_need_model


def _local(plan: dict[str, Any], goal: dict[str, Any], issues: list[str], model: str = "kept") -> dict[str, Any]:
    """Keep the plan the model already wrote. Do not replace it with a local template."""
    del goal
    kept = dict(plan)
    kept["stepEvaluation"] = {"accepted": False, "issues": issues[:8], "model": model}
    return kept


def _schedule_expanded(plan: dict[str, Any], goal: dict[str, Any], previous_count: int) -> dict[str, Any]:
    """A rewrite that adds blocks has to land the last one on the deadline."""
    if len(plan.get("days") or []) == previous_count:
        return plan
    stated = stated_deadline(goal)
    if stated is None:
        return plan
    from datetime import date

    span, label = stated
    assign_step_schedule(plan, goal, span, label, date.today())
    plan["horizonDays"] = span
    return plan


def _compact_steps(plan: dict[str, Any]) -> dict[str, Any]:
    days = []
    for day in plan.get("days") or []:
        if not isinstance(day, dict):
            continue
        days.append(
            {
                "title": day.get("title") or "",
                "focus": day.get("focus") or "",
                "definitionOfDone": day.get("definitionOfDone") or "",
                "tasks": [
                    {
                        "text": task.get("text") or "",
                        "why": task.get("why") or "",
                        "definitionOfDone": task.get("definitionOfDone") or "",
                    }
                    for task in (day.get("tasks") or [])
                    if isinstance(task, dict)
                ],
            }
        )
    milestones = []
    for item in (plan.get("milestones") or [])[:8]:
        if not isinstance(item, dict):
            continue
        milestones.append(
            {
                "title": item.get("title") or "",
                "successCriteria": item.get("successCriteria") or [],
            }
        )
    return {"days": days, "milestones": milestones}


async def evaluate_and_write_steps(goal: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    try:
        return await _evaluate_and_write_steps(goal, plan)
    except Exception as error:
        print(
            "goal step evaluation failed",
            {"error": type(error).__name__, "detail": str(error)[:240]},
            flush=True,
        )
        return _local(plan, goal, ["Step evaluation failed, so the actions were written locally."])


def _still_thin(plan: dict[str, Any], goal: dict[str, Any]) -> bool:
    """Ask the model again only when the plan has too little work for its deadline."""
    stated = stated_deadline(goal)
    days = [day for day in (plan.get("days") or []) if isinstance(day, dict)]
    open_days = [day for day in days if str(day.get("status") or "").lower() != "skipped" and not day.get("completedOn")]
    if stated is not None and stated[0] >= 14 and len(open_days) <= 1:
        return True
    return steps_need_model(plan)


async def _evaluate_and_write_steps(goal: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    if not any(isinstance(day, dict) for day in (plan.get("days") or [])):
        return plan
    if not _still_thin(plan, goal):
        plan["stepEvaluation"] = {"accepted": True, "issues": [], "model": "local-check"}
        return plan

    stated = stated_deadline(goal)
    payload = {**goal_context_payload(goal), "currentSteps": _compact_steps(plan)}
    if stated is not None and stated[0] >= 14:
        payload["coverFullDeadline"] = True
    called = await call_goal_agent(
        prompt_name="goal-steps-v1",
        schema=GoalStepWriterResult,
        payload=payload,
        stage="goal_steps",
        temperature=0.2,
        max_tokens=5200,
        pairs=goal_step_pairs(),
    )
    if called is not None:
        result, model_name = called
        previous_count = len(plan.get("days") or [])
        merged = accept_model_days(plan, result, goal)
        if merged is not None and not _still_thin(merged, goal):
            merged = _schedule_expanded(merged, goal, previous_count)
            issues = [str(item).strip() for item in (result.issues or []) if str(item).strip()][:8]
            merged["stepEvaluation"] = {
                "accepted": bool(result.accepted),
                "issues": issues,
                "model": model_name,
            }
            trace = [str(item) for item in (merged.get("agentTrace") or [])]
            trace.append(f"steps:{model_name}")
            merged["agentTrace"] = trace
            print(
                "goal steps rewritten",
                {"model": model_name, "accepted": bool(result.accepted), "days": len(merged.get("days") or [])},
                flush=True,
            )
            return merged
        print("goal step rewrite rejected", {"model": model_name}, flush=True)
    return _local(plan, goal, ["The step check could not use the model, so the actions were written locally."])
