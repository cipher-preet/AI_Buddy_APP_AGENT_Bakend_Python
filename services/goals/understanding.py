from __future__ import annotations

from typing import Any

from services.goals.domains import (
    context_profile,
    detect_goal_type,
    domain_to_goal_type,
    inferred_title,
    is_skip_answer,
    merge_context_profile,
    normalize_domain,
    understanding_as_dict,
    _text,
)
from services.goals.schemas import (
    GoalUnderstandingResult,
    InterviewQuestion,
    LabeledAssumption,
)

INTERVIEW_STEP_FIELDS = {
    "current-level": "startingLevel",
    "current-role": "currentRole",
    "starting-level": "startingLevel",
    "stage": "startingLevel",
    "location": "geography",
    "geography": "geography",
    "hours": "availableHoursPerWeek",
    "timeframe": "timeframe",
    "success": "successCriteria",
    "success-criteria": "successCriteria",
    "limits": "constraints",
    "constraints": "constraints",
    "career-detail": "description",
    "what-it-is": "description",
    "what-product": "description",
    "which-skill": "description",
    "subject": "description",
    "audience": "audience",
    "budget": "budget",
    "product-type": "productType",
    "product-kind": "productType",
}


def empty_understanding(statement: str) -> dict[str, Any]:
    goal_type = detect_goal_type(statement)
    domain = {
        "career": "CAREER",
        "skill": "EDUCATION",
        "product_launch": "BUSINESS",
        "app_build": "TECHNOLOGY",
        "sales": "BUSINESS",
        "habit": "PRODUCTIVITY",
        "freelance": "BUSINESS",
    }.get(goal_type, "OTHER")
    return GoalUnderstandingResult(
        primaryGoal=statement[:240],
        domain=domain,
        subdomain="",
        targetOutcome=statement[:240],
        startingLevel="UNKNOWN",
        researchNeeded=goal_type not in {"habit"},
        goalType=goal_type,
        assumptions=[
            LabeledAssumption(
                text="Starting level is unknown until the user confirms it.",
                reason="No current-state details were provided.",
                confirmed=False,
            )
        ],
        confidence=0.35,
    ).model_dump()


def _normalize_understanding(raw: GoalUnderstandingResult | dict[str, Any], statement: str) -> dict[str, Any]:
    data = understanding_as_dict(raw)
    data["domain"] = normalize_domain(str(data.get("domain") or "OTHER"))
    data["startingLevel"] = _text(data.get("startingLevel")).upper() or "UNKNOWN"
    if data["startingLevel"] not in {"UNKNOWN", "BEGINNER", "INTERMEDIATE", "ADVANCED", "EXPERT"}:
        if data["startingLevel"] not in {"STUDENT", "ENTRY", "MID", "SENIOR", "LEAD"}:
            data["startingLevel"] = "UNKNOWN"
    data["goalType"] = domain_to_goal_type(
        data["domain"],
        str(data.get("subdomain") or ""),
        statement,
    )
    data["primaryGoal"] = _text(data.get("primaryGoal")) or statement[:240]
    data["targetOutcome"] = _text(data.get("targetOutcome")) or data["primaryGoal"]
    data["researchNeeded"] = bool(data.get("researchNeeded", True))
    if data["domain"] == "PRODUCTIVITY" and "habit" in statement.lower():
        data["researchNeeded"] = bool(data.get("researchNeeded"))
    return data


async def understand_goal(goal: dict[str, Any], messages: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    existing = understanding_as_dict(goal.get("understanding"))
    statement = _text(goal.get("statement"))
    payload = {
        "statement": statement,
        "title": _text(goal.get("title")),
        "draft": {
            "description": _text(goal.get("description")),
            "successCriteria": _text(goal.get("successCriteria")),
            "timeframe": _text(goal.get("timeframe")),
            "constraints": goal.get("constraints") or [],
            "audience": _text(goal.get("audience")),
        },
        "contextProfile": context_profile(goal),
        "messages": [
            {"role": row.get("role"), "text": _text(row.get("text"))[:600], "stepId": _text(row.get("stepId"))}
            for row in (messages or [])
            if _text(row.get("text"))
        ][-8:],
        "existingUnderstanding": existing or None,
    }
    from services.goals.agents.llm import call_goal_agent

    called = await call_goal_agent(
        prompt_name="goal-understand-v1",
        schema=GoalUnderstandingResult,
        payload=payload,
        stage="goal_understand",
        temperature=0.1,
        max_tokens=1200,
    )
    if called is None:
        normalized = existing or empty_understanding(statement)
    else:
        result, _model = called
        normalized = _normalize_understanding(result, statement)
        if existing:
            for key in ("geography", "startingLevel", "subdomain"):
                if _text(existing.get(key)) and _text(existing.get(key)) not in {"UNKNOWN", "OTHER"}:
                    if not _text(normalized.get(key)) or normalized.get(key) in {"UNKNOWN", "OTHER"}:
                        normalized[key] = existing[key]
    profile = context_profile(goal)
    user_level = _text(profile.get("startingLevel") or profile.get("currentRole"))
    if user_level and user_level.upper() != "UNKNOWN" and not is_skip_answer(user_level):
        normalized["startingLevel"] = user_level[:80]
    else:
        normalized["startingLevel"] = "UNKNOWN"
    return normalized


def apply_message_answers(goal: dict[str, Any], messages: list[dict[str, Any]]) -> dict[str, Any]:
    profile = context_profile(goal)
    updates: dict[str, Any] = {}
    for row in messages:
        if row.get("role") != "user":
            continue
        step_id = _text(row.get("stepId")).lower()
        text = _text(row.get("text"))
        if not text:
            continue
        field = INTERVIEW_STEP_FIELDS.get(step_id)
        if not field:
            for key, mapped in INTERVIEW_STEP_FIELDS.items():
                if key in step_id:
                    field = mapped
                    break
        if not field:
            continue
        updates[field] = text
        if field == "availableHoursPerWeek" and not is_skip_answer(text):
            try:
                hours = float(text.split()[0])
                if 0 <= hours <= 168:
                    updates[field] = hours
            except (TypeError, ValueError):
                pass
    return merge_context_profile(profile, updates)


def _has_value(goal: dict[str, Any], profile: dict[str, Any], field: str) -> bool:
    if field == "availableHoursPerWeek":
        if goal.get("availableHoursPerWeek") is not None:
            return True
        return profile.get("availableHoursPerWeek") not in (None, "", "UNKNOWN")
    if field == "successCriteria":
        return bool(_text(goal.get("successCriteria")) or _text(profile.get("successCriteria")))
    if field == "timeframe":
        return bool(_text(goal.get("timeframe")) or _text(profile.get("timeframe")))
    if field == "constraints":
        return bool(goal.get("constraints") or profile.get("constraints"))
    value = _text(profile.get(field)) or _text(goal.get(field))
    if field in {"startingLevel", "currentRole", "geography"}:
        understanding = understanding_as_dict(goal.get("understanding"))
        value = value or _text(understanding.get(field))
    return bool(value) and value.upper() != "UNKNOWN"


def interview_questions_asked(messages: list[dict[str, Any]]) -> int:
    count = 0
    for row in messages:
        if row.get("role") != "assistant":
            continue
        step_id = _text(row.get("stepId")).lower()
        if any(token in step_id for token in INTERVIEW_STEP_FIELDS):
            count += 1
    return count


def next_interview_question(
    goal: dict[str, Any],
    understanding: dict[str, Any],
    profile: dict[str, Any],
    messages: list[dict[str, Any]],
) -> InterviewQuestion | None:
    """Pick the next goal-specific question, or the approach step, or nothing."""
    from services.goals.intake import next_intake_question

    return next_intake_question(goal, understanding, profile, messages)


def draft_from_understanding(
    goal: dict[str, Any],
    understanding: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    updates: dict[str, Any] = {
        "understanding": understanding,
        "contextProfile": profile,
        "pipelineState": "NEEDS_CONTEXT",
        "goalType": understanding.get("goalType") or detect_goal_type(_text(goal.get("statement"))),
    }
    title = inferred_title(goal, understanding)
    if title:
        updates["title"] = title[:160]
    description = _text(goal.get("description"))
    if description:
        updates["description"] = description[:2000]
    if _text(understanding.get("subdomain")):
        updates["category"] = _text(understanding.get("subdomain"))[:120]
    success = _text(profile.get("successCriteria")) or _text(goal.get("successCriteria"))
    if success and not is_skip_answer(success) and success.lower() != _text(goal.get("statement")).lower():
        updates["successCriteria"] = success[:1000]
    description = _text(profile.get("description"))
    if description and not is_skip_answer(description) and not _text(goal.get("description")):
        updates["description"] = description[:2000]
    audience = _text(profile.get("audience"))
    if audience and not is_skip_answer(audience) and audience.upper() != "UNKNOWN" and not _text(goal.get("audience")):
        updates["audience"] = audience[:240]
    budget = _text(profile.get("budget"))
    if budget and not is_skip_answer(budget) and budget.upper() != "UNKNOWN":
        existing_limits = [str(item) for item in (goal.get("constraints") or []) if str(item).strip()]
        line = f"Budget: {budget}"[:200]
        if line not in existing_limits:
            updates["constraints"] = [*existing_limits, line][:8]
    geo = _text(profile.get("geography")) or _text(understanding.get("geography"))
    if geo and geo.upper() != "UNKNOWN":
        updates.setdefault("constraints", list(goal.get("constraints") or []))
    hours = profile.get("availableHoursPerWeek")
    if hours is not None and goal.get("availableHoursPerWeek") is None:
        try:
            parsed = float(hours)
            if 0 <= parsed <= 168:
                updates["availableHoursPerWeek"] = parsed
        except (TypeError, ValueError):
            pass
    timeframe = _text(profile.get("timeframe")) or _text(goal.get("timeframe"))
    if timeframe and not is_skip_answer(timeframe):
        updates["timeframe"] = timeframe[:120]
    if understanding.get("constraints"):
        existing = [str(item) for item in (goal.get("constraints") or []) if str(item).strip()]
        incoming = [str(item)[:200] for item in understanding.get("constraints") or [] if str(item).strip()]
        updates["constraints"] = list(dict.fromkeys(existing + incoming))[:8]
    return updates


def ready_for_spaces(goal: dict[str, Any], understanding: dict[str, Any], profile: dict[str, Any], messages: list[dict[str, Any]]) -> bool:
    """Space selection starts after the approach step, not after a fixed count."""
    merged = {**goal, **draft_from_understanding(goal, understanding, profile)}
    return next_interview_question(merged, understanding, profile, messages) is None
