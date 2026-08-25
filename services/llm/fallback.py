from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from apps.api_gateway.config.setting import settings
from services.llm.errors import LLMProviderError
from services.llm.models import LLMMessage, LLMProvider, LLMRequest, LLMResponse, ProviderHealth, StructuredLLMRequest
from services.llm.quota import ProviderQuota, quota_guard
from services.llm.schema_adapter import classify_llm_failure


@dataclass(frozen=True)
class LLMRouteCandidate:
    provider: LLMProvider
    model: str
    quota: ProviderQuota | None = None


@dataclass
class StructuredRouteResult:
    schema_name: str
    provider_used: str | None = None
    model_used: str | None = None
    structured_mode_used: str | None = None
    attempt_count: int = 0
    fallback_depth: int = 0
    failure_history: list[dict[str, str]] = field(default_factory=list)
    attempted_providers: list[str] = field(default_factory=list)

    def as_log(self) -> dict[str, Any]:
        return {
            "schema": self.schema_name,
            "attemptedProviders": list(self.attempted_providers),
            "failureHistory": list(self.failure_history),
            "providerUsed": self.provider_used,
            "modelUsed": self.model_used,
            "structuredModeUsed": self.structured_mode_used,
            "attemptCount": self.attempt_count,
            "fallbackDepth": self.fallback_depth,
        }


class FallbackLLMProvider:
    def __init__(self, name: str, candidates: list[LLMRouteCandidate]):
        self.name = name
        self.candidates = candidates
        self.last_structured_diagnostics: dict[str, Any] = {}
        self.last_successful_provider: str | None = None
        self.last_successful_model: str | None = None
        self.last_structured_route: dict[str, Any] = {}

    async def generate(self, request: LLMRequest) -> LLMResponse:
        last_error: Exception | None = None
        estimated_tokens = _estimate_request_tokens(request.messages, request.max_tokens)
        for candidate in self.candidates:
            if not _candidate_fits(candidate, estimated_tokens):
                continue
            routed_request = request.model_copy(deep=True)
            routed_request.model = candidate.model
            quota_key = f"{candidate.provider.name}:{candidate.model}"
            try:
                print(
                    "LLM provider attempt started:",
                    {
                        "provider": candidate.provider.name,
                        "model": candidate.model,
                        "mode": "generate",
                    },
                )
                quota_guard.reserve(quota_key, candidate.quota, estimated_tokens)
                response = await candidate.provider.generate(routed_request)
                quota_guard.record_actual_tokens(
                    quota_key,
                    candidate.quota,
                    estimated_tokens,
                    response.usage.totalTokens,
                )
                print(
                    "LLM provider attempt succeeded:",
                    {
                        "provider": candidate.provider.name,
                        "model": candidate.model,
                        "mode": "generate",
                        "latencyMs": response.latencyMs,
                        "totalTokens": response.usage.totalTokens,
                    },
                )
                self.last_successful_provider = candidate.provider.name
                self.last_successful_model = candidate.model
                return response
            except Exception as error:
                last_error = error
                reason = classify_llm_failure(error)
                print(
                    "LLM provider attempt failed:",
                    {
                        "provider": candidate.provider.name,
                        "model": candidate.model,
                        "mode": "generate",
                        "failureReason": reason,
                        "error": str(error)[:300],
                    },
                )
                if not _should_try_next(error):
                    raise
        raise LLMProviderError(f"all LLM fallbacks failed for {self.name}: {last_error}", retryable=True)

    async def generate_structured(
        self,
        request: StructuredLLMRequest,
        response_schema: type[BaseModel],
    ) -> BaseModel:
        last_error: Exception | None = None
        schema_name = request.schema_name or getattr(response_schema, "__name__", "schema")
        route = StructuredRouteResult(schema_name=schema_name)
        estimated_tokens = _estimate_request_tokens(request.messages, request.max_tokens)
        eligible = [candidate for candidate in self.candidates if _candidate_fits(candidate, estimated_tokens)]
        if not eligible:
            eligible = list(self.candidates)
        for candidate in eligible:
            routed_request = request.model_copy(deep=True)
            routed_request.model = candidate.model
            routed_request.max_tokens = _structured_max_tokens(candidate.provider.name, routed_request.max_tokens)
            quota_key = f"{candidate.provider.name}:{candidate.model}"
            route.attempted_providers.append(candidate.provider.name)
            route.attempt_count += 1
            try:
                print(
                    "LLM provider attempt started:",
                    {
                        "provider": candidate.provider.name,
                        "model": candidate.model,
                        "mode": "structured",
                        "schema": schema_name,
                    },
                )
                quota_guard.reserve(quota_key, candidate.quota, estimated_tokens)
                result = await candidate.provider.generate_structured(routed_request, response_schema)
                diagnostics = getattr(candidate.provider, "last_structured_diagnostics", {}) or {}
                self.last_structured_diagnostics = {
                    **diagnostics,
                    **route.as_log(),
                    "providerUsed": candidate.provider.name,
                    "modelUsed": candidate.model,
                }
                self.last_successful_provider = candidate.provider.name
                self.last_successful_model = candidate.model
                route.provider_used = candidate.provider.name
                route.model_used = candidate.model
                route.structured_mode_used = str(
                    diagnostics.get("structuredModeUsed")
                    or diagnostics.get("requestedStructuredMode")
                    or diagnostics.get("actualResponseFormatMode")
                    or ""
                ) or None
                route.fallback_depth = max(0, len(route.failure_history))
                self.last_structured_route = route.as_log()
                print(
                    "LLM provider attempt succeeded:",
                    {
                        "provider": candidate.provider.name,
                        "model": candidate.model,
                        "mode": "structured",
                        "schema": schema_name,
                        "structuredModeUsed": route.structured_mode_used,
                    },
                )
                _log_structured_route(self.last_structured_route)
                return result
            except Exception as error:
                self.last_structured_diagnostics = getattr(candidate.provider, "last_structured_diagnostics", {}) or {}
                last_error = error
                reason = classify_llm_failure(error)
                route.failure_history.append({"provider": candidate.provider.name, "reason": reason})
                print(
                    "LLM provider attempt failed:",
                    {
                        "provider": candidate.provider.name,
                        "model": candidate.model,
                        "mode": "structured",
                        "schema": schema_name,
                        "failureReason": reason,
                        "error": str(error)[:300],
                    },
                )
                if not _should_try_next(error):
                    route.fallback_depth = max(0, len(route.failure_history) - 1)
                    self.last_structured_route = route.as_log()
                    _log_structured_route(self.last_structured_route)
                    raise
        route.fallback_depth = max(0, len(route.failure_history) - 1)
        self.last_structured_route = route.as_log()
        _log_structured_route(self.last_structured_route)
        raise LLMProviderError(
            f"all structured LLM fallbacks failed for {self.name}: {last_error}",
            retryable=True,
            failure_reason=classify_llm_failure(last_error) if last_error else None,
        )

    async def health_check(self) -> ProviderHealth:
        healthy = []
        errors = []
        for candidate in self.candidates:
            result = await candidate.provider.health_check()
            if result.healthy:
                healthy.append(candidate.provider.name)
            elif result.error:
                errors.append(f"{candidate.provider.name}: {result.error}")
        return ProviderHealth(
            provider=self.name,
            healthy=bool(healthy),
            error=None if healthy else "; ".join(errors),
        )


def resolved_provider_name(provider, fallback: str = "unknown") -> str:
    used = getattr(provider, "last_successful_provider", None)
    if used:
        return used
    route = getattr(provider, "last_structured_route", None) or {}
    attempted = route.get("attemptedProviders") or []
    if attempted:
        return str(attempted[-1])
    return getattr(provider, "name", None) or fallback


def resolved_provider_model(provider, fallback: str | None = None) -> str | None:
    used = getattr(provider, "last_successful_model", None)
    if used:
        return used
    return fallback


def _should_try_next(error: Exception) -> bool:
    if isinstance(error, LLMProviderError):
        return error.retryable or error.status_code in {400, 413, 422, 429}
    return True


def _candidate_fits(candidate: LLMRouteCandidate, estimated_tokens: int) -> bool:
    limits = settings.provider_context_token_limits
    context = limits.get(candidate.provider.name, settings.FINAL_MODEL_INPUT_TOKEN_LIMIT)
    usable = int(context * settings.SEMANTIC_WINDOW_SAFE_CONTEXT_RATIO) - settings.LLM_STRUCTURED_MAX_TOKENS - 1500
    return estimated_tokens <= max(1000, usable)


def _structured_max_tokens(provider_name: str, configured: int | None) -> int | None:
    limit = configured or settings.LLM_STRUCTURED_MAX_TOKENS
    if provider_name == "groq":
        return max(512, min(limit, settings.GROQ_MAX_TPM // 2))
    return limit


def _estimate_request_tokens(messages: list[LLMMessage], max_tokens: int | None) -> int:
    input_chars = sum(len(message.content or "") for message in messages)
    input_tokens = max(1, input_chars // 4)
    return input_tokens + (max_tokens or 0)


def _log_structured_route(route: dict[str, Any]) -> None:
    print("LLM structured route completed", route)
