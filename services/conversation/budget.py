from __future__ import annotations

from apps.api_gateway.config.setting import settings
from services.conversation.transcript import estimate_tokens


def provider_context_limit(provider_name: str) -> int:
    name = str(provider_name or "").strip().lower()
    return settings.provider_context_token_limits.get(name, settings.FINAL_MODEL_INPUT_TOKEN_LIMIT)


def safe_input_budget(provider_name: str, output_reserve: int | None = None) -> int:
    context = provider_context_limit(provider_name)
    reserved_output = output_reserve if output_reserve is not None else settings.LLM_STRUCTURED_MAX_TOKENS
    overhead = 1500
    usable = int(context * settings.SEMANTIC_WINDOW_SAFE_CONTEXT_RATIO) - reserved_output - overhead
    return max(1000, usable)


def largest_routable_input_budget() -> int:
    limits = [safe_input_budget(name) for name in settings.provider_context_token_limits]
    return max(limits) if limits else settings.FINAL_MODEL_INPUT_TOKEN_LIMIT


def semantic_window_token_target() -> int:
    configured = settings.SEMANTIC_WINDOW_MAX_TRANSCRIPT_TOKENS
    incremental = settings.INCREMENTAL_WINDOW_TARGET_TOKENS
    # Tests historically patch INCREMENTAL_WINDOW_TARGET_TOKENS to tiny values.
    # Honor the tighter of the two so existing monkeypatches keep working.
    return min(configured, incremental)


def semantic_window_token_max() -> int:
    target = semantic_window_token_target()
    configured_max = max(settings.INCREMENTAL_WINDOW_MAX_TOKENS, target)
    model_budget = largest_routable_input_budget()
    return max(target, min(configured_max, model_budget))


def semantic_window_useful_duration_ms() -> int:
    configured = settings.semantic_window_useful_duration_ms
    incremental = settings.INCREMENTAL_WINDOW_MAX_DURATION_MS
    return min(configured, incremental)


def payload_fits(payload: str, provider_name: str) -> bool:
    return estimate_tokens(payload) <= safe_input_budget(provider_name)


def expected_request_tokens(*parts: str) -> int:
    return estimate_tokens("\n".join(part for part in parts if part))
