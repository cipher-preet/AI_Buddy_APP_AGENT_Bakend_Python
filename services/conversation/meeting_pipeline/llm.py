"""Structured LLM calls for the meeting pipeline. No sequence-marker clipping."""

from __future__ import annotations

import json
import time
from contextvars import ContextVar
from typing import Any

from pydantic import BaseModel

from services.conversation.agents import _structured_with_recovery
from services.conversation.transcript import estimate_tokens
from services.llm.router import LLMCapability, LLMRouter

_USAGE: ContextVar[list[dict[str, Any]] | None] = ContextVar("meeting_pipeline_usage", default=None)


def bind_usage(records: list[dict[str, Any]]):
    return _USAGE.set(records)


def reset_usage(token) -> None:
    _USAGE.reset(token)


def current_usage() -> list[dict[str, Any]]:
    return list(_USAGE.get() or [])


async def generate_structured(
    router: LLMRouter,
    capability: LLMCapability,
    prompt_name: str,
    schema: type[BaseModel],
    payload: dict[str, Any] | str,
    *,
    background: str = "",
    stage: str = "",
    meta: dict[str, Any] | None = None,
) -> tuple[Any, Any, str]:
    current = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, default=str)
    estimated = estimate_tokens(background) + estimate_tokens(current)
    started = time.perf_counter()
    provider, model = router.route(capability)
    response, used_provider, used_model = await _structured_with_recovery(
        router,
        provider,
        model,
        prompt_name,
        schema,
        background,
        current,
    )
    actual_provider = (
        getattr(used_provider, "last_successful_provider", None)
        or getattr(used_provider, "name", None)
        or provider
    )
    actual_model = getattr(used_provider, "last_successful_model", None) or used_model or model
    diagnostics = getattr(used_provider, "last_structured_diagnostics", None) or {}
    input_tokens = int(diagnostics.get("promptTokens") or diagnostics.get("inputTokens") or estimated or 0)
    output_tokens = int(diagnostics.get("completionTokens") or diagnostics.get("outputTokens") or 0)
    record = {
        "stage": stage or prompt_name,
        "capability": capability.value if hasattr(capability, "value") else str(capability),
        "provider": str(actual_provider),
        "model": str(actual_model),
        "prompt": prompt_name,
        "inputTokens": input_tokens,
        "outputTokens": output_tokens,
        "latencyMs": int((time.perf_counter() - started) * 1000),
        "fallback": bool(getattr(used_provider, "last_successful_model", None) and str(actual_model) != str(model)),
        "finishReason": diagnostics.get("finishReason"),
        "parsingOutcome": diagnostics.get("parsingOutcome"),
        "structuredOutputSuccess": diagnostics.get("structuredOutputSuccess"),
        "schemaError": (
            diagnostics.get("retryReason") or diagnostics.get("parsingOutcome")
            if diagnostics.get("structuredOutputSuccess") is False
            else None
        ),
        "rawContent": diagnostics.get("rawContent"),
        **(meta or {}),
    }
    sink = _USAGE.get()
    if sink is not None:
        sink.append(record)
    return response, actual_provider, actual_model
