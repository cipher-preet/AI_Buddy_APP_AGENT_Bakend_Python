from __future__ import annotations

from typing import Any

DONE_STATUSES = {"done", "completed", "complete", "closed"}


def _plan(goal: dict[str, Any]) -> dict[str, Any]:
    raw = goal.get("approvedPlan") or goal.get("proposedPlan") or {}
    return raw if isinstance(raw, dict) else {}


def is_done_status(value: Any) -> bool:
    return str(value or "").strip().lower() in DONE_STATUSES


def execution_progress(goal: dict[str, Any], space_context: dict[str, Any] | None = None) -> dict[str, Any]:
    plan = _plan(goal)
    days = plan.get("days") or []
    tasks = [task for day in days for task in (day.get("tasks") or [])]
    checklists = [item for row in (plan.get("checklists") or []) for item in (row.get("items") or [])]
    done_tasks = sum(1 for item in tasks if item.get("done"))
    done_checks = sum(1 for item in checklists if item.get("done"))
    plan_total = len(tasks) + len(checklists)
    plan_completed = done_tasks + done_checks
    space_tasks = list((space_context or {}).get("tasks") or [])
    space_done = sum(1 for item in space_tasks if is_done_status(item.get("status")))
    space_total = len(space_tasks)
    total = plan_total + space_total
    completed = plan_completed + space_done
    return {
        "completed": completed,
        "total": total,
        "ratio": round(completed / total, 3) if total else 0,
        "planCompleted": plan_completed,
        "planTotal": plan_total,
        "spaceCompleted": space_done,
        "spaceTotal": space_total,
    }


def chart_actual(
    goal: dict[str, Any],
    evidence: list[dict[str, Any]] | None = None,
    space_context: dict[str, Any] | None = None,
) -> int:
    """0–100 score for the dashboard graph: plan work + Space tasks + notes/transcripts."""
    plan = execution_progress(goal)
    plan_pct = (plan["planCompleted"] / plan["planTotal"] * 100) if plan["planTotal"] else None

    live = space_context if isinstance(space_context, dict) else None
    live_has_items = bool(live and (live.get("tasks") or live.get("notes") or live.get("transcripts")))
    if live_has_items:
        space_tasks = list(live.get("tasks") or [])
        space_total = len(space_tasks)
        space_done = sum(1 for item in space_tasks if is_done_status(item.get("status")))
        notes = len(live.get("notes") or [])
        transcripts = len(live.get("transcripts") or [])
    else:
        latest_tasks: dict[str, bool] = {}
        notes = 0
        transcripts = 0
        for row in evidence or []:
            source = str(row.get("sourceType") or "")
            source_id = str(row.get("sourceId") or "")
            if source == "task" and source_id and source_id not in latest_tasks:
                latest_tasks[source_id] = str(row.get("category") or "") == "OUTPUT"
            elif source == "note":
                notes += 1
            elif source == "transcript":
                transcripts += 1
        space_total = len(latest_tasks)
        space_done = sum(1 for done in latest_tasks.values() if done)

    space_pct = (space_done / space_total * 100) if space_total else None
    material = min(notes + transcripts, 8)
    material_pct = (material / 8) * 100 if material else 0.0

    weights: list[float] = []
    values: list[float] = []
    if plan_pct is not None:
        weights.append(0.5)
        values.append(plan_pct)
    if space_pct is not None:
        weights.append(0.35)
        values.append(space_pct)
    if material:
        weights.append(0.15)
        values.append(material_pct)
    if not values:
        return 0
    if plan_pct is None and space_pct is None:
        return min(15, int(round(material_pct * 0.2)))
    score = sum(value * weight for value, weight in zip(values, weights, strict=True)) / sum(weights)
    return int(round(min(100, max(0, score))))


def readiness_progress(goal: dict[str, Any]) -> dict[str, Any]:
    gaps = goal.get("gaps") if isinstance(goal.get("gaps"), dict) else {}
    satisfied = len(gaps.get("satisfied") or [])
    partial = len(gaps.get("partial") or [])
    missing = len(gaps.get("missing") or [])
    unknown = len(gaps.get("unknown") or [])
    known = satisfied + partial + missing
    return {
        "satisfied": satisfied,
        "partial": partial,
        "missing": missing,
        "unknown": unknown,
        "ratio": round(satisfied / known, 3) if known else 0,
        "provisional": bool(gaps.get("provisional") or unknown),
    }


def achievement_state(goal: dict[str, Any], evidence: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if str(goal.get("status") or "") == "COMPLETED":
        return {"achieved": True, "reason": "User confirmed the real-world success condition."}
    rows = evidence or []
    confirmed = [
        row
        for row in rows
        if str(row.get("category") or "") == "OUTCOME" and str(row.get("verification") or "") in {"USER_CONFIRMED", "SYSTEM_VERIFIED"}
    ]
    if confirmed:
        return {"achieved": True, "reason": confirmed[0].get("text") or "Outcome confirmed."}
    return {
        "achieved": False,
        "reason": "Preparation progress is not the same as achieving the external outcome.",
    }


def consultant_progress(goal: dict[str, Any], evidence: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    execution = execution_progress(goal)
    readiness = readiness_progress(goal)
    achievement = achievement_state(goal, evidence)
    return {
        "execution": execution,
        "readiness": readiness,
        "achievement": achievement,
    }
