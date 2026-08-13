from __future__ import annotations

from apps.api_gateway.config.setting import settings
from services.llm.errors import LLMProviderError
from services.llm.models import LLMRequest, LLMResponse, ProviderHealth, StructuredLLMRequest
from services.llm.openai_compatible import OpenAICompatibleProvider
from pydantic import BaseModel


def build_sarvam_provider() -> OpenAICompatibleProvider:
    base_url = settings.SARVAM_BASE_URL.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"

    return OpenAICompatibleProvider(
        name="sarvam",
        api_key=settings.secret_value(settings.SARVAM_API_KEY),
        base_url=base_url,
        default_model=settings.SARVAM_DEFAULT_MODEL,
        timeout_seconds=settings.SARVAM_TIMEOUT_SECONDS,
        max_retries=settings.SARVAM_MAX_RETRIES,
        max_concurrency=settings.SARVAM_MAX_CONCURRENCY,
        auth_header="api-subscription-key",
        auth_prefix="",
        max_tokens_limit=settings.SARVAM_MAX_TOKENS,
    )


def build_openai_provider() -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        name="openai",
        api_key=settings.secret_value(settings.OPENAI_API_KEY),
        base_url=settings.OPENAI_BASE_URL,
        default_model=settings.LLM_DEFAULT_MODEL,
        timeout_seconds=settings.LLM_TIMEOUT_SECONDS,
        max_retries=settings.SARVAM_MAX_RETRIES,
        max_concurrency=settings.LLM_MAX_CONCURRENCY,
    )


def build_gemini_provider() -> OpenAICompatibleProvider | NotConfiguredProvider:
    if not settings.secret_value(settings.GEMINI_API_KEY):
        return NotConfiguredProvider("gemini")
    return OpenAICompatibleProvider(
        name="gemini",
        api_key=settings.secret_value(settings.GEMINI_API_KEY),
        base_url=settings.GEMINI_BASE_URL,
        default_model=settings.GEMINI_FREE_MODEL,
        timeout_seconds=settings.LLM_TIMEOUT_SECONDS,
        max_retries=settings.SARVAM_MAX_RETRIES,
        max_concurrency=settings.LLM_MAX_CONCURRENCY,
    )


def build_groq_provider() -> OpenAICompatibleProvider | NotConfiguredProvider:
    if not settings.secret_value(settings.GROQ_API_KEY):
        return NotConfiguredProvider("groq")
    return OpenAICompatibleProvider(
        name="groq",
        api_key=settings.secret_value(settings.GROQ_API_KEY),
        base_url=settings.GROQ_BASE_URL,
        default_model=settings.GROQ_FREE_MODEL,
        timeout_seconds=settings.LLM_TIMEOUT_SECONDS,
        max_retries=settings.SARVAM_MAX_RETRIES,
        max_concurrency=settings.LLM_MAX_CONCURRENCY,
        max_tokens_limit=max(256, min(settings.LLM_STRUCTURED_MAX_TOKENS, settings.GROQ_MAX_TPM // 2)),
    )


def build_mistral_provider() -> OpenAICompatibleProvider | NotConfiguredProvider:
    if not settings.secret_value(settings.MISTRAL_API_KEY):
        return NotConfiguredProvider("mistral")
    return OpenAICompatibleProvider(
        name="mistral",
        api_key=settings.secret_value(settings.MISTRAL_API_KEY),
        base_url=settings.MISTRAL_BASE_URL,
        default_model=settings.MISTRAL_CHEAP_MODEL,
        timeout_seconds=settings.LLM_TIMEOUT_SECONDS,
        max_retries=settings.SARVAM_MAX_RETRIES,
        max_concurrency=settings.LLM_MAX_CONCURRENCY,
    )


class NotConfiguredProvider:
    def __init__(self, name: str):
        self.name = name
        self.configured = False

    async def generate(self, request: LLMRequest) -> LLMResponse:
        raise LLMProviderError(f"{self.name} provider is not configured", retryable=True, status_code=503)

    async def generate_structured(
        self,
        request: StructuredLLMRequest,
        response_schema: type[BaseModel],
    ) -> BaseModel:
        raise LLMProviderError(f"{self.name} provider is not configured", retryable=True, status_code=503)

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(provider=self.name, healthy=False, error="provider is not configured")


def build_anthropic_provider() -> NotConfiguredProvider:
    return NotConfiguredProvider("anthropic")
