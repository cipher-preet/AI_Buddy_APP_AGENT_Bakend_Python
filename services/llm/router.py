from __future__ import annotations

from enum import Enum

from apps.api_gateway.config.setting import settings
from services.llm.fallback import FallbackLLMProvider, LLMRouteCandidate
from services.llm.models import LLMProvider
from services.llm.providers import (
    build_anthropic_provider,
    build_gemini_provider,
    build_groq_provider,
    build_mistral_provider,
    build_openai_provider,
    build_sarvam_provider,
)
from services.llm.quota import ProviderQuota


class LLMCapability(str, Enum):
    HIGH_ACCURACY_REASONING = "high_accuracy_reasoning"
    CHAT_ANSWER = "chat_answer"
    VALIDATION = "validation"
    COMPLEX_TASK_MATCHING = "complex_task_matching"
    SIMPLE_SUMMARY = "simple_summary"
    NORMALIZATION = "normalization"
    FALLBACK = "fallback"


class LLMRouter:
    def __init__(self, providers: dict[str, LLMProvider]):
        self.providers = providers

    def route(self, capability: LLMCapability) -> tuple[LLMProvider, str]:
        if settings.LLM_ENABLE_COST_OPTIMIZED_ROUTING:
            candidates = self._cost_optimized_candidates(capability)
            if candidates:
                primary = candidates[0]
                return FallbackLLMProvider(primary.provider.name, candidates), primary.model

        provider_name = settings.LLM_DEFAULT_PROVIDER
        model = settings.LLM_DEFAULT_MODEL
        if capability in {
            LLMCapability.VALIDATION,
            LLMCapability.HIGH_ACCURACY_REASONING,
            LLMCapability.COMPLEX_TASK_MATCHING,
        }:
            model = settings.LLM_VALIDATION_MODEL
        elif capability == LLMCapability.SIMPLE_SUMMARY:
            model = settings.LLM_SUMMARY_MODEL
        elif capability == LLMCapability.NORMALIZATION:
            model = settings.LLM_FAST_MODEL
        elif capability == LLMCapability.FALLBACK:
            provider_name = settings.LLM_SECONDARY_PROVIDER

        provider = self.providers.get(provider_name)
        if not provider:
            provider = self.providers[settings.LLM_DEFAULT_PROVIDER]
        return provider, model

    def _cost_optimized_candidates(self, capability: LLMCapability) -> list[LLMRouteCandidate]:
        gemini = self._candidate(
            "gemini",
            settings.GEMINI_FREE_MODEL,
            ProviderQuota(rpm=settings.GEMINI_MAX_RPM, rpd=settings.GEMINI_MAX_RPD),
        )
        groq = self._candidate(
            "groq",
            settings.GROQ_FREE_MODEL,
            ProviderQuota(
                rpm=settings.GROQ_MAX_RPM,
                rpd=settings.GROQ_MAX_RPD,
                tpm=settings.GROQ_MAX_TPM,
                tpd=settings.GROQ_MAX_TPD,
            ),
        )
        mistral = self._candidate("mistral", settings.MISTRAL_CHEAP_MODEL)
        sarvam = self._candidate("sarvam", settings.SARVAM_DEFAULT_MODEL)

        if capability == LLMCapability.NORMALIZATION:
            return [item for item in [gemini, groq, mistral, sarvam] if item]
        if capability == LLMCapability.SIMPLE_SUMMARY:
            return [item for item in [groq, mistral, sarvam] if item]
        if capability == LLMCapability.CHAT_ANSWER:
            return [item for item in [groq, gemini, mistral, sarvam] if item]
        if capability == LLMCapability.VALIDATION:
            return [item for item in [groq, mistral, sarvam] if item]
        if capability in {LLMCapability.HIGH_ACCURACY_REASONING, LLMCapability.COMPLEX_TASK_MATCHING}:
            return [item for item in [groq, mistral, sarvam] if item]
        if capability == LLMCapability.FALLBACK:
            return [item for item in [mistral, sarvam] if item]
        return [item for item in [groq, mistral, sarvam] if item]

    def _candidate(
        self,
        provider_name: str,
        model: str,
        quota: ProviderQuota | None = None,
    ) -> LLMRouteCandidate | None:
        provider = self.providers.get(provider_name)
        if not provider or getattr(provider, "configured", True) is False:
            return None
        return LLMRouteCandidate(provider=provider, model=model, quota=quota)


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
                "groq": build_groq_provider(),
                "mistral": build_mistral_provider(),
            }
        )
    return _router
