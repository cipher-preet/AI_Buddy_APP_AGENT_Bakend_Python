"""Canonical structured-output calls for event-pipeline stages.

Always goes through LLMRouter + json_schema. Provider failure uses the existing
fallback chain. Schemas are never weakened to accept an invalid response.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from pydantic import BaseModel

from services.conversation.budget import expected_request_tokens
from services.conversation.event_pipeline.budget import charge_model_call, charge_retry, stage_timeout_seconds
from services.conversation.event_pipeline.observability import log_model_route, record_failure, record_llm_usage
from services.conversation.event_pipeline.routing import (
    PipelineStage,
    cap_payload,
    capability_for_stage,
    capability_log_name,
    route_for_stage,
    stage_log_name,
)
from services.llm.async_runtime import reraise_if_hard_runtime
from services.llm.errors import LLMProviderError
from services.llm.router import LLMRouter
from services.llm.schema_adapter import PROVIDER_TIMEOUT


async def generate_structured_for_stage(
    router: LLMRouter,
    stage: PipelineStage | str,
    prompt_name: str,
    schema: type[BaseModel],
    payload: dict[str, Any] | str,
    *,
    background: str = "",
) -> tuple[Any, Any, str]:
    from services.conversation.agents import _structured_with_recovery

    current = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=True, default=str)
    current = cap_payload(current, stage)
    background = cap_payload(background, stage) if background else ""
    estimated = expected_request_tokens(background, current)
    provider, model, capability = route_for_stage(router, stage, estimated)
    charge_model_call()
    timeout = stage_timeout_seconds()
    try:
        response, provider, model = await asyncio.wait_for(
            _structured_with_recovery(
                router,
                provider,
                model,
                prompt_name,
                schema,
                background,
                current,
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError as error:
        record_failure(error)
        raise LLMProviderError(
            f"{stage_log_name(stage)} timed out after {timeout}s",
            retryable=True,
            failure_reason=PROVIDER_TIMEOUT,
        ) from error
    except Exception as error:
        record_failure(error)
        reraise_if_hard_runtime(error)
        raise
    actual_provider = (
        getattr(provider, "last_successful_provider", None)
        or getattr(provider, "name", None)
        or "unknown"
    )
    actual_model = getattr(provider, "last_successful_model", None) or model
    route = getattr(provider, "last_structured_route", None) or {}
    fallback = bool(route.get("fallbackDepth")) or (
        bool(getattr(provider, "last_successful_model", None))
        and str(actual_model or "").casefold() != str(model or "").casefold()
    )
    diagnostics = getattr(provider, "last_structured_diagnostics", None) or {}
    extra_attempts = max(0, int(route.get("attemptCount") or diagnostics.get("attemptCount") or 1) - 1)
    if extra_attempts:
        charge_retry(extra_attempts)
    log_model_route(
        stage=stage_log_name(stage),
        capability=capability_log_name(capability),
        provider=str(actual_provider),
        model=str(actual_model),
        fallback=fallback,
        requested=capability.value,
    )
    record_llm_usage(
        model=str(actual_model),
        input_tokens=int(diagnostics.get("promptTokens") or diagnostics.get("inputTokens") or estimated or 0),
        output_tokens=int(diagnostics.get("completionTokens") or diagnostics.get("outputTokens") or 0),
        fallback=fallback,
        attempts=int(route.get("attemptCount") or diagnostics.get("attemptCount") or 1),
    )
    return response, provider, model


def compact_thread(thread: Any) -> dict[str, Any]:
    if thread is None:
        return {}
    return {
        "threadId": getattr(thread, "threadId", None),
        "label": getattr(thread, "label", None),
        "entities": list(getattr(thread, "entities", None) or [])[:12],
        "latestState": getattr(thread, "latestState", None),
        "eventIds": list(getattr(thread, "eventIds", None) or [])[-8:],
        "sequenceStart": getattr(thread, "sequenceStart", None),
        "sequenceEnd": getattr(thread, "sequenceEnd", None),
    }


def compact_event(event: Any) -> dict[str, Any]:
    if event is None:
        return {}
    dump = event.model_dump(exclude={"embedding", "conversationId", "userId", "spaceId"})
    dump["evidence"] = dump.get("evidence") or []
    return dump


def requested_capability(stage: PipelineStage | str):
    return capability_for_stage(stage)
