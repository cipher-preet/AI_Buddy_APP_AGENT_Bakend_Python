"""Dedicated structured LLM caller for the v2 daily briefing (strong Krutrim route)."""

from __future__ import annotations

import json
import time
from typing import Any, Protocol

from pydantic import BaseModel

from apps.api_gateway.config.setting import settings
from services.daily_briefing.log import briefing_log
from services.llm.models import LLMMessage, StructuredLLMRequest
from services.llm.router import get_llm_router
from services.prompts.loader import load_prompt

STAGE_WINDOW = "window"
STAGE_SYNTHESIS = "synthesis"
REASONING_MODEL_MARKERS = ("gpt-oss",)


class BriefingCaller(Protocol):
    async def generate(
        self,
        stage: str,
        prompt_name: str,
        schema: type[BaseModel],
        payload: dict[str, Any],
    ) -> Any:
        ...


class BriefingLLMError(RuntimeError):
    pass


def parse_model_route(value: str) -> list[tuple[str, str]]:
    route: list[tuple[str, str]] = []
    for part in (value or "").split(","):
        provider, _, model = part.strip().partition(":")
        if provider and model and (provider.strip(), model.strip()) not in route:
            route.append((provider.strip(), model.strip()))
    return route


def _extra_body(model: str) -> dict[str, Any]:
    effort = settings.DAILY_BRIEFING_REASONING_EFFORT
    if effort and any(marker in model for marker in REASONING_MODEL_MARKERS):
        return {"reasoning_effort": effort}
    return {}


def _truncated(diagnostics: dict[str, Any], max_tokens: int) -> bool:
    finish = str(diagnostics.get("finishReason") or "").casefold()
    completion = int(diagnostics.get("completionTokens") or 0)
    return finish in {"length", "max_tokens"} or completion >= max_tokens


class KrutrimBriefingCaller:
    """Walks the configured model route; a truncated answer counts as a failure."""

    def __init__(self, router=None):
        self._router = router

    def _route(self, stage: str) -> list[tuple[Any, str]]:
        router = self._router or get_llm_router()
        spec = settings.DAILY_BRIEFING_WINDOW_MODELS if stage == STAGE_WINDOW else settings.DAILY_BRIEFING_SYNTHESIS_MODELS
        route = [(router.providers[name], model) for name, model in parse_model_route(spec) if name in router.providers]
        if not route:
            raise BriefingLLMError(f"daily_briefing_no_llm_route stage={stage}")
        return route

    async def generate(self, stage, prompt_name, schema, payload):
        max_tokens = (
            settings.DAILY_BRIEFING_WINDOW_MAX_OUTPUT_TOKENS
            if stage == STAGE_WINDOW
            else settings.DAILY_BRIEFING_SYNTHESIS_MAX_OUTPUT_TOKENS
        )
        messages = [
            LLMMessage(role="system", content=load_prompt(prompt_name)),
            LLMMessage(role="user", content=json.dumps(payload, ensure_ascii=False, default=str)),
        ]
        errors: list[str] = []
        for provider, model in self._route(stage):
            request = StructuredLLMRequest(
                messages=messages,
                model=model,
                temperature=0.2,
                max_tokens=max_tokens,
                metadata={"stage": f"daily_briefing_{stage}", "extra_body": _extra_body(model)},
                schema_name=schema.__name__,
            )
            started = time.perf_counter()
            try:
                result = await provider.generate_structured(request, schema)
            except Exception as error:
                errors.append(f"{model}: {type(error).__name__}: {str(error)[:160]}")
                continue
            diagnostics = getattr(provider, "last_structured_diagnostics", None) or {}
            latency = int((time.perf_counter() - started) * 1000)
            if _truncated(diagnostics, max_tokens):
                errors.append(f"{model}: truncated")
                briefing_log("daily_briefing_llm_truncated", stage=stage, model=model, latencyMs=latency)
                continue
            briefing_log(
                "daily_briefing_llm_call",
                stage=stage,
                prompt=prompt_name,
                provider=provider.name,
                model=model,
                latencyMs=latency,
                outputTokens=diagnostics.get("completionTokens"),
            )
            return result
        raise BriefingLLMError(f"daily_briefing_llm_failed stage={stage} errors={errors}")
