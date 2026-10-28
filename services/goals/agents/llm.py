from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel

from apps.api_gateway.config.setting import settings
from services.llm.models import LLMMessage, StructuredLLMRequest
from services.llm.router import get_llm_router
from services.prompts.loader import load_prompt

T = TypeVar("T", bound=BaseModel)


def goal_context_payload(goal: dict[str, Any]) -> dict[str, Any]:
    from services.goals.brief import stated_deadline
    from services.goals.domains import context_profile, understanding_as_dict

    statement = str(goal.get("statement") or "").strip()
    understanding = understanding_as_dict(goal.get("understanding"))
    profile = context_profile(goal)
    stated = stated_deadline(goal)
    level = str(profile.get("startingLevel") or profile.get("currentRole") or understanding.get("startingLevel") or "").strip()
    payload = {
        "title": goal.get("title"),
        "statement": statement,
        "goalType": str(goal.get("goalType") or understanding.get("goalType") or "").strip(),
        "description": goal.get("description") or goal.get("category") or "",
        "audience": goal.get("audience") or "",
        "targetMetric": goal.get("targetMetric") or "",
        "timeframe": goal.get("timeframe"),
        "successCriteria": goal.get("successCriteria"),
        "availableHoursPerWeek": goal.get("availableHoursPerWeek"),
        "constraints": goal.get("constraints") or [],
        "measurementType": goal.get("measurementType") or "",
        "baseline": (goal.get("latestStep") or {}).get("prompt") or "",
        "baselineCorrection": goal.get("baselineCorrection") or "",
        "understanding": understanding,
        "contextProfile": context_profile(goal),
        "researchEvidence": goal.get("researchEvidencePack") or [],
        "gaps": goal.get("gaps") or {},
        "researchSummary": goal.get("researchSummary") or {},
        "planIsProvisional": bool((goal.get("gaps") or {}).get("provisional")),
        "startingPoint": level,
    }
    if stated is not None:
        payload["deadlineDays"] = stated[0]
        payload["deadlineLabel"] = stated[1]
    return payload


def provider_pairs() -> list[tuple[str, str]]:
    return [
        (settings.CONVERSATION_SYNTHESIS_PROVIDER, settings.CONVERSATION_SYNTHESIS_MODEL),
        (settings.CONVERSATION_SYNTHESIS_FALLBACK_PROVIDER, settings.CONVERSATION_SYNTHESIS_FALLBACK_MODEL),
        (settings.CONVERSATION_SEMANTIC_PROVIDER, settings.CONVERSATION_SEMANTIC_MODEL),
        (settings.CONVERSATION_SEMANTIC_FALLBACK_PROVIDER, settings.CONVERSATION_SEMANTIC_FALLBACK_MODEL),
    ]


def goal_step_pairs() -> list[tuple[str, str]]:
    """Step evaluation and rewriting use Krutrim gpt-oss-120b, then Gemma if it fails."""
    primary = (settings.GOAL_STEP_PROVIDER.strip(), settings.GOAL_STEP_MODEL.strip())
    fallback = (settings.GOAL_STEP_FALLBACK_PROVIDER.strip(), settings.GOAL_STEP_FALLBACK_MODEL.strip())
    return [primary, fallback]


async def call_goal_agent(
    *,
    prompt_name: str,
    schema: type[T],
    payload: dict[str, Any],
    stage: str,
    temperature: float = 0.3,
    max_tokens: int = 4500,
    pairs: list[tuple[str, str]] | None = None,
) -> tuple[T, str] | None:
    router = get_llm_router()
    llm_messages = [
        LLMMessage(role="system", content=load_prompt(prompt_name)),
        LLMMessage(role="user", content=json.dumps(payload, ensure_ascii=False, default=str)),
    ]
    seen: set[tuple[str, str]] = set()
    for provider_name, model in pairs if pairs is not None else provider_pairs():
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
            temperature=temperature,
            max_tokens=max_tokens,
            metadata={"stage": stage},
            schema_name=schema.__name__,
        )
        try:
            result = await provider.generate_structured(request, schema)
        except Exception as error:
            print(
                "goal agent model failed",
                {
                    "stage": stage,
                    "provider": provider_name,
                    "model": model,
                    "error": type(error).__name__,
                    "detail": str(error)[:240],
                },
                flush=True,
            )
            continue
        if isinstance(result, schema):
            return result, f"{provider_name}:{model}"
    return None
