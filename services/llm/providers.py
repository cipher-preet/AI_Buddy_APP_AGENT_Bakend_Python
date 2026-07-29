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
        api_key=settings.SARVAM_API_KEY,
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
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
        default_model=settings.LLM_DEFAULT_MODEL,
        timeout_seconds=settings.LLM_TIMEOUT_SECONDS,
        max_retries=settings.SARVAM_MAX_RETRIES,
        max_concurrency=settings.LLM_MAX_CONCURRENCY,
    )


class NotConfiguredProvider:
    def __init__(self, name: str):
        self.name = name

    async def generate(self, request: LLMRequest) -> LLMResponse:
        raise LLMProviderError(f"{self.name} provider is not configured", retryable=False)

    async def generate_structured(
        self,
        request: StructuredLLMRequest,
        response_schema: type[BaseModel],
    ) -> BaseModel:
        raise LLMProviderError(f"{self.name} provider is not configured", retryable=False)

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(provider=self.name, healthy=False, error="provider is not configured")


def build_anthropic_provider() -> NotConfiguredProvider:
    return NotConfiguredProvider("anthropic")


def build_gemini_provider() -> NotConfiguredProvider:
    return NotConfiguredProvider("gemini")
