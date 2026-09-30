"""Structured LLM caller that walks a provider:model route until one answers cleanly."""

from __future__ import annotations

import json
import time
from typing import Any, Protocol

from pydantic import BaseModel

from apps.api_gateway.config.setting import settings
from services.llm.models import LLMMessage, StructuredLLMRequest
from services.llm.router import get_llm_router
from services.prompts.loader import load_prompt
from services.schedule_extraction.log import extraction_log

REASONING_MODEL_MARKERS = ("gpt-oss",)


class ExtractionCaller(Protocol):
    async def generate(self, prompt_name: str, schema: type[BaseModel], payload: dict[str, Any]) -> BaseModel:
        ...


class ExtractionLLMError(RuntimeError):
    pass


def parse_model_route(value: str) -> list[tuple[str, str]]:
    route: list[tuple[str, str]] = []
    for part in (value or "").split(","):
        provider, _, model = part.strip().partition(":")
        pair = (provider.strip(), model.strip())
        if pair[0] and pair[1] and pair not in route:
            route.append(pair)
    return route


def _extra_body(model: str) -> dict[str, Any]:
    effort = settings.SCHEDULE_EXTRACTION_REASONING_EFFORT
    if effort and any(marker in model for marker in REASONING_MODEL_MARKERS):
        return {"reasoning_effort": effort}
    return {}


def _truncated(diagnostics: dict[str, Any], max_tokens: int) -> bool:
    finish = str(diagnostics.get("finishReason") or "").casefold()
    completion = int(diagnostics.get("completionTokens") or 0)
    return finish in {"length", "max_tokens"} or completion >= max_tokens


class RoutedExtractionCaller:
    def __init__(self, router=None, route: str | None = None, max_tokens: int | None = None):
        self._router = router
        self._route_spec = route
        self._max_tokens = max_tokens

    def _route(self) -> list[tuple[Any, str]]:
        router = self._router or get_llm_router()
        spec = self._route_spec or settings.SCHEDULE_EXTRACTION_MODELS
        route = [
            (router.providers[name], model)
            for name, model in parse_model_route(spec)
            if name in router.providers and getattr(router.providers[name], "configured", True) is not False
        ]
        if not route:
            raise ExtractionLLMError("schedule_extraction_no_llm_route")
        return route

    async def generate(self, prompt_name, schema, payload):
        max_tokens = self._max_tokens or settings.SCHEDULE_EXTRACTION_MAX_OUTPUT_TOKENS
        messages = [
            LLMMessage(role="system", content=load_prompt(prompt_name)),
            LLMMessage(role="user", content=json.dumps(payload, ensure_ascii=False, default=str)),
        ]
        errors: list[str] = []
        for provider, model in self._route():
            request = StructuredLLMRequest(
                messages=messages,
                model=model,
                temperature=0.0,
                max_tokens=max_tokens,
                metadata={"stage": f"schedule_extraction:{prompt_name}", "extra_body": _extra_body(model)},
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
                extraction_log("schedule_extraction_llm_truncated", prompt=prompt_name, model=model, latencyMs=latency)
                continue
            extraction_log(
                "schedule_extraction_llm_call",
                prompt=prompt_name,
                model=model,
                latencyMs=latency,
                outputTokens=diagnostics.get("completionTokens"),
            )
            return result if isinstance(result, schema) else schema.model_validate(result)
        raise ExtractionLLMError(f"schedule_extraction_llm_failed prompt={prompt_name} errors={errors}")
