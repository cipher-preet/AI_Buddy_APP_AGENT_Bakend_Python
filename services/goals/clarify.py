from __future__ import annotations

import json
import re
from typing import Any

from apps.api_gateway.config.setting import settings
from services.goals.domains import detect_goal_type as keyword_detect_goal_type, is_skip_answer
from services.goals.schemas import GoalClarifyResult, GoalNextStep, GoalOption
from services.goals.understanding import (
    apply_message_answers,
    draft_from_understanding,
    understand_goal,
)
from services.llm.models import LLMMessage, StructuredLLMRequest
from services.llm.router import get_llm_router
from services.prompts.loader import load_prompt

CONTROLS = {"text", "single_select", "multi_select", "number", "space_picker"}
MEASUREMENTS = {"", "NUMERIC", "MILESTONE", "HABIT", "BINARY", "QUALITATIVE"}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _step(step_id: str, prompt: str, control: str, options: list[dict[str, str]] | None = None) -> dict[str, Any]:
    return {
        "stepId": step_id,
        "prompt": prompt,
        "control": control,
        "options": options or [],
        "allowFreeText": control != "space_picker",
        "findings": [],
        "milestones": [],
    }


def detect_goal_type(statement: str, explicit: str = "", domain: str = "") -> str:
    """Prefer an explicit type or understanding domain. Keywords are last-resort only."""
    return keyword_detect_goal_type(statement, explicit, domain)


def _understood(goal: dict[str, Any]) -> bool:
    """Basic understanding only: what, success, and when."""
    return bool(
        (_text(goal.get("description")) or _text(goal.get("category")) or _text(goal.get("statement")))
        and _text(goal.get("successCriteria"))
        and _text(goal.get("timeframe"))
    )


def _named_thing(goal: dict[str, Any]) -> str:
    statement = _text(goal.get("statement"))
    title = _text(goal.get("title"))
    named = re.search(r"(?:called|named)\s+([A-Za-z0-9][\w\s-]{1,40})", statement, re.I)
    if named:
        return named.group(1).strip()
    cleaned = re.sub(
        r"^(i\s+want\s+to|i'd\s+like\s+to|i\s+need\s+to|help\s+me)\s+",
        "",
        (title or statement).strip(),
        flags=re.I,
    )
    cleaned = cleaned.strip(" .")
    if cleaned and cleaned.lower() not in {"new goal", "goal"} and len(cleaned) <= 60:
        return cleaned
    if cleaned:
        return cleaned[:57].rstrip() + "…"
    return "this goal"


def fallback_clarify(goal: dict[str, Any], assistant_turns: int) -> dict[str, Any]:
    title = _named_thing(goal)
    understanding = goal.get("understanding") if isinstance(goal.get("understanding"), dict) else {}
    goal_type = detect_goal_type(
        _text(goal.get("statement")),
        _text(goal.get("goalType")),
        str((understanding or {}).get("domain") or ""),
    )
    draft: dict[str, Any] = {"goalType": goal_type} if not _text(goal.get("goalType")) else {}
    if understanding:
        draft["understanding"] = understanding
    if isinstance(goal.get("contextProfile"), dict):
        draft["contextProfile"] = goal["contextProfile"]

    if assistant_turns >= settings.GOAL_MAX_CLARIFY_TURNS or _understood(goal):
        return {
            "draft": draft,
            "status": "READY_FOR_PLAN",
            "completeness": 75,
            "step": _step(
                "build-plan",
                "I'll build the plan from your answers and the deadline. No Space is used.",
                "brief",
            ),
            "model": "fallback",
        }

    if not _text(goal.get("description")) and not _text(goal.get("category")):
        prompts = {
            "product_launch": (
                "what-product",
                f"What is “{title}”, and what problem does it solve?",
                "text",
                None,
            ),
            "skill": (
                "which-skill",
                f"What skill do you want to learn or improve with “{title}”?",
                "text",
                None,
            ),
            "freelance": (
                "freelance-scope",
                f"What freelance work is this, and what should you deliver?",
                "text",
                None,
            ),
            "app_build": (
                "what-app",
                f"What is the app, and who is it for?",
                "text",
                None,
            ),
            "sales": (
                "sales-offer",
                f"What are you selling, and what result do you want?",
                "text",
                None,
            ),
            "habit": (
                "habit-detail",
                f"What habit do you want, and how often?",
                "text",
                None,
            ),
            "career": (
                "current-level",
                (
                    f"What is your current role or stage toward “{title}”? "
                    "Student, early-career, already in the field, or skip if you want labeled assumptions."
                ),
                "text",
                None,
            ),
            "other": (
                "what-it-is",
                f"In simple words, what do you want from “{title}”?",
                "text",
                None,
            ),
        }
        step_id, prompt, control, options = prompts.get(goal_type, prompts["other"])
        return {
            "draft": draft,
            "status": "CLARIFYING",
            "completeness": 18,
            "step": _step(step_id, prompt, control, options),
            "model": "fallback",
        }

    if not _text(goal.get("successCriteria")):
        prompts = {
            "product_launch": (
                "launch-definition",
                f"What does a successful launch look like for “{title}”?",
                "single_select",
                [
                    {"id": "mvp", "label": "A usable MVP is live"},
                    {"id": "public", "label": "Public release people can use"},
                    {"id": "paid", "label": "First paying customers"},
                    {"id": "store", "label": "Listed on a store or marketplace"},
                ],
            ),
            "skill": (
                "skill-proof",
                f"How will you know you achieved the skill for “{title}”?",
                "single_select",
                [
                    {"id": "project", "label": "I shipped a project that proves it"},
                    {"id": "job", "label": "I can do the job or interview for it"},
                    {"id": "exam", "label": "I passed a course or exam"},
                    {"id": "confidence", "label": "I can use it confidently on my own"},
                ],
            ),
            "freelance": (
                "freelance-done",
                "What counts as the project being complete?",
                "text",
                None,
            ),
            "app_build": (
                "app-done",
                "What does “complete app” mean here: working MVP, published build, or polished release?",
                "text",
                None,
            ),
            "sales": (
                "sales-target",
                "What sales number or customer count would make this a win?",
                "text",
                None,
            ),
        }
        step_id, prompt, control, options = prompts.get(
            goal_type,
            (
                "success",
                f"What will “done” look like for “{title}”?",
                "text",
                None,
            ),
        )
        return {
            "draft": draft,
            "status": "CLARIFYING",
            "completeness": 38,
            "step": _step(step_id, prompt, control, options),
            "model": "fallback",
        }

    if goal_type == "sales" and not _text(goal.get("targetMetric")):
        return {
            "draft": draft,
            "status": "CLARIFYING",
            "completeness": 48,
            "step": _step(
                "expected-sales",
                "What sales or customer target are you aiming for?",
                "text",
            ),
            "model": "fallback",
        }

    if goal_type == "product_launch" and not _text(goal.get("audience")):
        return {
            "draft": draft,
            "status": "CLARIFYING",
            "completeness": 48,
            "step": _step(
                "audience",
                f"Who is “{title}” for? Describe the main user or customer.",
                "text",
            ),
            "model": "fallback",
        }

    if not _text(goal.get("timeframe")):
        return {
            "draft": draft,
            "status": "CLARIFYING",
            "completeness": 55,
            "step": _step(
                "timeframe",
                f"When do you want to reach that result for “{title}”?",
                "single_select",
                [
                    {"id": "30-days", "label": "Within 30 days"},
                    {"id": "90-days", "label": "Within 90 days"},
                    {"id": "6-months", "label": "Within 6 months"},
                    {"id": "1-year", "label": "Within a year"},
                ],
            ),
            "model": "fallback",
        }

    if goal.get("availableHoursPerWeek") is None:
        return {
            "draft": draft,
            "status": "CLARIFYING",
            "completeness": 68,
            "step": _step(
                "hours",
                "About how many hours each week can you spend on this?",
                "number",
            ),
            "model": "fallback",
        }

    # Optional soft limit question once — never block Space on constraints.
    if not any(_text(item) for item in (goal.get("constraints") or [])) and assistant_turns < 4:
        prompts = {
            "product_launch": f"Any must-haves for “{title}”, or can we keep the first version simple?",
            "skill": f"Any tools or limits for learning “{title}”?",
            "freelance": f"Any client limits or must-deliver items for this work?",
            "app_build": f"Any must-have platform or feature for “{title}”?",
            "sales": f"Any important limit on how you can sell?",
        }
        return {
            "draft": draft,
            "status": "CLARIFYING",
            "completeness": 72,
            "step": _step(
                "limits",
                prompts.get(goal_type, f"Any important limit for “{title}”? If none, say “none”."),
                "text",
            ),
            "model": "fallback",
        }

    return {
        "draft": draft,
        "status": "READY_FOR_PLAN",
        "completeness": 78,
        "step": _step(
            "build-plan",
            "I'll build the plan from your answers and the deadline. No Space is used.",
            "brief",
        ),
        "model": "fallback",
    }


def _draft_updates(goal: dict[str, Any], result: GoalClarifyResult) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    for field in ("title", "description", "category", "audience", "targetMetric", "timeframe", "successCriteria"):
        incoming = _text(getattr(result, field, ""))
        if incoming and not _text(goal.get(field)):
            updates[field] = incoming[:2000 if field == "description" else 1000]
        elif incoming and field in {"title", "successCriteria", "description"}:
            # Prefer a richer rewrite once the model is confident.
            if len(incoming) >= len(_text(goal.get(field))):
                updates[field] = incoming[:2000 if field == "description" else 1000]
    goal_type = _text(result.goalType) or _text(goal.get("goalType"))
    if goal_type:
        updates["goalType"] = goal_type[:40]
    profile_updates: dict[str, Any] = {}
    if _text(result.currentRole) and not is_skip_answer(result.currentRole):
        profile_updates["currentRole"] = result.currentRole[:240]
    if _text(result.startingLevel) and not is_skip_answer(result.startingLevel):
        profile_updates["startingLevel"] = result.startingLevel[:80]
    if _text(result.geography) and not is_skip_answer(result.geography):
        profile_updates["geography"] = result.geography[:120]
    if profile_updates:
        from services.goals.domains import merge_context_profile, context_profile

        updates["contextProfile"] = merge_context_profile(context_profile({**goal, **updates}), profile_updates)
    measurement = _text(result.measurementType).upper()
    if measurement in MEASUREMENTS and measurement:
        updates["measurementType"] = measurement
    if result.availableHoursPerWeek is not None:
        hours = float(result.availableHoursPerWeek)
        if 0 <= hours <= 168:
            updates["availableHoursPerWeek"] = hours
    if result.constraints:
        updates["constraints"] = [item.strip()[:200] for item in result.constraints if item.strip()][:8]
    if result.missingMaterial:
        updates["missingMaterial"] = [item.strip()[:160] for item in result.missingMaterial if item.strip()][:8]
    completeness = max(0, min(100, int(result.completeness or 0)))
    merged = {**goal, **updates}
    if _understood(merged):
        completeness = max(completeness, 75)
    elif _text(merged.get("description")) and _text(merged.get("successCriteria")):
        completeness = max(completeness, 45)
    updates["completeness"] = completeness
    return updates


def _ready_step(updates: dict[str, Any]) -> tuple[dict[str, Any], str, dict[str, Any]]:
    step = _step("build-plan", "I'll build the plan from what you told me.", "brief")
    step["allowFreeText"] = False
    updates["completeness"] = max(int(updates.get("completeness") or 0), 75)
    updates["pipelineState"] = "READY_FOR_RESEARCH"
    return updates, "READY_FOR_PLAN", step


def _public_step(result: GoalClarifyResult, goal: dict[str, Any], assistant_turns: int) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Use the model's question. The model chooses the fact. A fixed field list does not."""
    updates = _draft_updates(goal, result)
    at_cap = assistant_turns + 1 >= settings.GOAL_MAX_CLARIFY_TURNS
    control = result.nextStep.control if result.nextStep.control in CONTROLS - {"space_picker"} else "text"
    step_id = result.nextStep.stepId.strip()[:40]
    ready = at_cap or result.readyForBaseline or control == "brief" or step_id in {"approach", "build-plan"}
    if ready:
        return _ready_step(updates)

    options = [
        {"id": option.id.strip()[:40] or f"option-{index}", "label": option.label.strip()[:160]}
        for index, option in enumerate(result.nextStep.options)
        if option.label.strip()
    ][:6]
    if control in {"single_select", "multi_select"} and not options:
        control = "text"
    prompt = result.nextStep.prompt.strip()[:500]
    if not prompt:
        prompt = "What else should the plan know so it matches what you want to achieve? Skip anything that would not change the plan."
        control = "text"
        options = []
    step = _step(step_id or "detail", prompt, control, options)
    return updates, "CLARIFYING", step


async def _call_model(goal: dict[str, Any], messages: list[dict[str, Any]]) -> tuple[GoalClarifyResult, str] | None:
    router = get_llm_router()
    pairs = [
        (settings.CONVERSATION_SEMANTIC_PROVIDER, settings.CONVERSATION_SEMANTIC_MODEL),
        (settings.CONVERSATION_SEMANTIC_FALLBACK_PROVIDER, settings.CONVERSATION_SEMANTIC_FALLBACK_MODEL),
    ]
    payload = {
        "statement": _text(goal.get("statement")),
        "detectedGoalType": detect_goal_type(
            _text(goal.get("statement")),
            _text(goal.get("goalType")),
            str((goal.get("understanding") or {}).get("domain") or "") if isinstance(goal.get("understanding"), dict) else "",
        ),
        "understanding": goal.get("understanding") if isinstance(goal.get("understanding"), dict) else {},
        "contextProfile": goal.get("contextProfile") if isinstance(goal.get("contextProfile"), dict) else {},
        "draft": {
            "title": _text(goal.get("title")),
            "goalType": _text(goal.get("goalType")),
            "description": _text(goal.get("description")),
            "category": _text(goal.get("category")),
            "audience": _text(goal.get("audience")),
            "targetMetric": _text(goal.get("targetMetric")),
            "timeframe": _text(goal.get("timeframe")),
            "successCriteria": _text(goal.get("successCriteria")),
            "availableHoursPerWeek": goal.get("availableHoursPerWeek"),
            "constraints": goal.get("constraints") or [],
            "missingMaterial": goal.get("missingMaterial") or [],
            "measurementType": _text(goal.get("measurementType")),
            "completeness": goal.get("completeness") or 0,
        },
        "messages": [
            {"role": row.get("role"), "text": _text(row.get("text"))[:1000]}
            for row in messages
            if _text(row.get("text"))
        ],
        "instruction": (
            "This can be any kind of goal. Decide the one missing fact that would change the plan, or stop. "
            "Do not use a fixed questionnaire. Do not ask for a Space. Allow skip."
        ),
    }
    llm_messages = [
        LLMMessage(role="system", content=load_prompt("goal-clarify-v1")),
        LLMMessage(role="user", content=json.dumps(payload, ensure_ascii=False)),
    ]
    seen: set[tuple[str, str]] = set()
    for provider_name, model in pairs:
        key = (provider_name, model)
        if not provider_name or not model or key in seen:
            continue
        seen.add(key)
        provider = router.providers.get(provider_name)
        if not provider or getattr(provider, "configured", True) is False:
            continue
        request = StructuredLLMRequest(
            messages=llm_messages,
            model=model,
            temperature=0.3,
            max_tokens=settings.GOAL_CLARIFY_MAX_TOKENS,
            metadata={"stage": "goal_clarify"},
            schema_name=GoalClarifyResult.__name__,
        )
        try:
            result = await provider.generate_structured(request, GoalClarifyResult)
        except Exception as error:
            print(
                "goal clarify model failed",
                {"provider": provider_name, "model": model, "error": type(error).__name__},
                flush=True,
            )
            continue
        if isinstance(result, GoalClarifyResult) and _text(result.nextStep.prompt):
            return result, f"{provider_name}:{model}"
    return None


async def build_clarify_step(goal: dict[str, Any], messages: list[dict[str, Any]]) -> dict[str, Any]:
    assistant_turns = sum(1 for row in messages if row.get("role") == "assistant")
    profile = apply_message_answers(goal, messages)
    understanding = await understand_goal({**goal, "contextProfile": profile}, messages)
    seeded = draft_from_understanding(goal, understanding, profile)
    working = {**goal, **seeded, "contextProfile": profile, "understanding": understanding}

    if assistant_turns >= settings.GOAL_MAX_CLARIFY_TURNS:
        updates, status, step = _ready_step(seeded)
        return {"draft": updates, "status": status, "step": step, "model": "understand"}

    called = await _call_model(working, messages)
    if called is not None:
        result, model_name = called
        updates, status, step = _public_step(result, working, assistant_turns)
        return {"draft": {**seeded, **updates}, "status": status, "step": step, "model": model_name}

    seeded["completeness"] = min(70, 20 + assistant_turns * 12)
    seeded["pipelineState"] = "NEEDS_CONTEXT"
    return {
        "draft": seeded,
        "status": "CLARIFYING",
        "step": _step(
            "detail",
            "What else should the plan know so it matches what you want to achieve? Skip anything that would not change the plan.",
            "text",
        ),
        "model": "fallback",
    }


def normalize_step(step: GoalNextStep | dict[str, Any]) -> dict[str, Any]:
    if isinstance(step, GoalNextStep):
        payload = step.model_dump()
    else:
        payload = step
    options = payload.get("options") or []
    payload["options"] = [
        GoalOption.model_validate(option).model_dump()
        for option in options
        if isinstance(option, dict) and _text(option.get("label"))
    ]
    return payload
