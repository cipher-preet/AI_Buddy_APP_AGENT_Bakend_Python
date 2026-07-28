from __future__ import annotations

from enum import Enum

from apps.api_gateway.config.setting import settings
from services.llm.models import LLMProvider
from services.llm.providers import (
    build_anthropic_provider,
    build_gemini_provider,
    build_openai_provider,
    build_sarvam_provider,
)


class LLMCapability(str, Enum):
    HIGH_ACCURACY_REASONING = "high_accuracy_reasoning"
    VALIDATION = "validation"
    COMPLEX_TASK_MATCHING = "complex_task_matching"
    SIMPLE_SUMMARY = "simple_summary"
    NORMALIZATION = "normalization"
    FALLBACK = "fallback"


class LLMRouter:
    def __init__(self, providers: dict[str, LLMProvider]):
        self.providers = providers

    def route(self, capability: LLMCapability) -> tuple[LLMProvider, str]:
        provider_name = settings.LLM_DEFAULT_PROVIDER
        model = settings.LLM_DEFAULT_MODEL
        if capability in {
            LLMCapability.VALIDATION,
            LLMCapability.HIGH_ACCURACY_REASONING,
            LLMCapability.COMPLEX_TASK_MATCHING,
        }:
            model = settings.LLM_VALIDATION_MODEL
        elif capability in {LLMCapability.SIMPLE_SUMMARY, LLMCapability.NORMALIZATION}:
            model = settings.LLM_FAST_MODEL
        elif capability == LLMCapability.FALLBACK:
            provider_name = settings.LLM_SECONDARY_PROVIDER

        provider = self.providers.get(provider_name)
        if not provider:
            provider = self.providers[settings.LLM_DEFAULT_PROVIDER]
        return provider, model


_router: LLMRouter | None = None


def get_llm_router() -> LLMRouter:
    global _router
    if _router is None:
        _router = LLMRouter(
            {
                "sarvam": build_sarvam_provider(),
                "openai": build_openai_provider(),
                "anthropic": build_anthropic_provider(),
                "gemini": build_gemini_provider(),
            }
        )
    return _router
