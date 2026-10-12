from __future__ import annotations

from typing import Any

from services.goals.domains import (
    context_profile,
    detect_goal_type,
    domain_to_goal_type,
    inferred_success,
    inferred_title,
    is_skip_answer,
    merge_context_profile,
    normalize_domain,
    understanding_as_dict,
    _text,
)
from services.goals.schemas import (
    GoalOption,
    GoalUnderstandingResult,
    InterviewQuestion,
    LabeledAssumption,
)

INTERVIEW_STEP_FIELDS = {
    "current-level": "startingLevel",
    "current-role": "currentRole",
    "starting-level": "startingLevel",
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
        return existing or empty_understanding(statement)
    result, _model = called
    normalized = _normalize_understanding(result, statement)
    if existing:
        for key in ("geography", "startingLevel", "subdomain"):
            if _text(existing.get(key)) and _text(existing.get(key)) not in {"UNKNOWN", "OTHER"}:
                if not _text(normalized.get(key)) or normalized.get(key) in {"UNKNOWN", "OTHER"}:
                    normalized[key] = existing[key]
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
    if interview_questions_asked(messages) >= 3:
        return None
    domain = normalize_domain(str(understanding.get("domain") or "OTHER"))
    title = inferred_title(goal, understanding) or "this goal"
    candidates: list[InterviewQuestion] = []

    if not _has_value(goal, profile, "startingLevel") and not _has_value(goal, profile, "currentRole"):
        if domain == "CAREER":
            candidates.append(
                InterviewQuestion(
                    stepId="current-level",
                    field="startingLevel",
                    prompt=(
                        f"What is your current role or stage for “{title}”? "
                        "If you are a student, say so. Skip if you want me to plan with labeled assumptions."
                    ),
                    control="text",
                    leverage="Changes which earlier stages belong on the roadmap.",
                )
            )
        elif domain in {"EDUCATION", "TECHNOLOGY", "HEALTH", "CREATIVE"}:
            candidates.append(
                InterviewQuestion(
                    stepId="current-level",
                    field="startingLevel",
                    prompt=(
                        f"What is your current level for “{title}”? "
                        "Beginner, some experience, or already practicing? You can skip."
                    ),
                    control="single_select",
                    options=[
                        GoalOption(id="beginner", label="Beginner / from zero"),
                        GoalOption(id="intermediate", label="Some experience"),
                        GoalOption(id="advanced", label="Already practicing at a high level"),
                        GoalOption(id="skip", label="Not sure / skip"),
                    ],
                    leverage="Changes whether fundamentals belong on the plan.",
                )
            )
        elif domain == "BUSINESS":
            candidates.append(
                InterviewQuestion(
                    stepId="current-level",
                    field="startingLevel",
                    prompt=(
                        f"Where are you with “{title}” today — idea, building, already operating, or growing? "
                        "Skip if you want a provisional outline."
                    ),
                    control="single_select",
                    options=[
                        GoalOption(id="idea", label="Idea / not started"),
                        GoalOption(id="building", label="Building the first version"),
                        GoalOption(id="operating", label="Already operating"),
                        GoalOption(id="skip", label="Not sure / skip"),
                    ],
                    leverage="Changes whether the plan starts at discovery or scale.",
                )
            )

    if not _has_value(goal, profile, "successCriteria") and not inferred_success(goal, understanding):
        candidates.append(
            InterviewQuestion(
                stepId="success",
                field="successCriteria",
                prompt=f"What will ‘done’ look like for “{title}”? One concrete result is enough.",
                control="text",
                leverage="Defines the real-world success condition.",
            )
        )

    if not _has_value(goal, profile, "timeframe"):
        candidates.append(
            InterviewQuestion(
                stepId="timeframe",
                field="timeframe",
                prompt=f"When do you want a meaningful result for “{title}”? Skip if the timeline is open.",
                control="single_select",
                options=[
                    GoalOption(id="30-days", label="Within 30 days"),
                    GoalOption(id="90-days", label="Within 90 days"),
                    GoalOption(id="6-months", label="Within 6 months"),
                    GoalOption(id="1-year", label="Within a year"),
                    GoalOption(id="multi-year", label="This will take years"),
                    GoalOption(id="skip", label="Not sure / skip"),
                ],
                leverage="Changes estimate vs target dates.",
            )
        )

    if domain in {"CAREER", "EDUCATION", "FINANCIAL"} and not _has_value(goal, profile, "geography"):
        candidates.append(
            InterviewQuestion(
                stepId="location",
                field="geography",
                prompt=(
                    "Which country or region should the plan assume? "
                    "Requirements often differ by place. Skip to keep it general."
                ),
                control="text",
                leverage="Changes researched requirements and regulations.",
            )
        )

    if goal.get("availableHoursPerWeek") is None and not _has_value(goal, profile, "availableHoursPerWeek"):
        candidates.append(
            InterviewQuestion(
                stepId="hours",
                field="availableHoursPerWeek",
                prompt="About how many hours each week can you spend on this? Skip if it varies.",
                control="number",
                leverage="Changes capacity and timeline realism.",
            )
        )

    return candidates[0] if candidates else None


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
    description = _text(goal.get("description")) or _text(understanding.get("primaryGoal"))
    if description:
        updates["description"] = description[:2000]
    if _text(understanding.get("subdomain")):
        updates["category"] = _text(understanding.get("subdomain"))[:120]
    success = inferred_success(goal, understanding)
    if success:
        updates["successCriteria"] = success[:1000]
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
    merged = {**goal, **draft_from_understanding(goal, understanding, profile)}
    has_what = bool(
        _text(merged.get("description"))
        or _text(merged.get("statement"))
        or _text(understanding.get("primaryGoal"))
    )
    has_success = bool(_text(merged.get("successCriteria")))
    asked = interview_questions_asked(messages)
    if asked >= 3:
        return has_what
    if has_what and has_success and next_interview_question(merged, understanding, profile, messages) is None:
        return True
    if has_what and has_success and _text(merged.get("timeframe")):
        return next_interview_question(merged, understanding, profile, messages) is None
    return False
