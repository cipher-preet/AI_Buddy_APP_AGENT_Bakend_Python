from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

from services.llm.errors import LLMProviderError
from services.llm.models import LLMMessage, LLMProvider, LLMRequest, LLMResponse, ProviderHealth, StructuredLLMRequest
from services.llm.quota import ProviderQuota, quota_guard


@dataclass(frozen=True)
class LLMRouteCandidate:
    provider: LLMProvider
    model: str
    quota: ProviderQuota | None = None


class FallbackLLMProvider:
    def __init__(self, name: str, candidates: list[LLMRouteCandidate]):
        self.name = name
        self.candidates = candidates

    async def generate(self, request: LLMRequest) -> LLMResponse:
        last_error: Exception | None = None
        for candidate in self.candidates:
            routed_request = request.model_copy(deep=True)
            routed_request.model = candidate.model
            estimated_tokens = _estimate_request_tokens(routed_request.messages, routed_request.max_tokens)
            quota_key = f"{candidate.provider.name}:{candidate.model}"
            try:
                quota_guard.reserve(quota_key, candidate.quota, estimated_tokens)
                response = await candidate.provider.generate(routed_request)
                quota_guard.record_actual_tokens(
                    quota_key,
                    candidate.quota,
                    estimated_tokens,
                    response.usage.totalTokens,
                )
                return response
            except Exception as error:
                last_error = error
                if not _should_try_next(error):
                    raise
        raise LLMProviderError(f"all LLM fallbacks failed for {self.name}: {last_error}", retryable=True)

    async def generate_structured(
        self,
        request: StructuredLLMRequest,
        response_schema: type[BaseModel],
    ) -> BaseModel:
        last_error: Exception | None = None
        for candidate in self.candidates:
            routed_request = request.model_copy(deep=True)
            routed_request.model = candidate.model
            estimated_tokens = _estimate_request_tokens(routed_request.messages, routed_request.max_tokens)
            quota_key = f"{candidate.provider.name}:{candidate.model}"
            try:
                quota_guard.reserve(quota_key, candidate.quota, estimated_tokens)
                return await candidate.provider.generate_structured(routed_request, response_schema)
            except Exception as error:
                last_error = error
                if not _should_try_next(error):
                    raise
        raise LLMProviderError(f"all structured LLM fallbacks failed for {self.name}: {last_error}", retryable=True)

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


def _should_try_next(error: Exception) -> bool:
    if isinstance(error, LLMProviderError):
        return error.retryable or error.status_code in {400, 413, 422, 429}
    return True


def _estimate_request_tokens(messages: list[LLMMessage], max_tokens: int | None) -> int:
    input_chars = sum(len(message.content or "") for message in messages)
    input_tokens = max(1, input_chars // 4)
    return input_tokens + (max_tokens or 0)
