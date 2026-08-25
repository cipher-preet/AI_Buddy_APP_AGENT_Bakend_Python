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
from services.llm.routing_policy import (
    conversation_intelligence_provider_order,
    provider_model_for,
    provider_quota_for,
    uses_conversation_intelligence_policy,
)


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
        if uses_conversation_intelligence_policy(capability) or capability == LLMCapability.CHAT_ANSWER:
            return self._candidates_in_order(conversation_intelligence_provider_order())
        if capability == LLMCapability.NORMALIZATION:
            return self._candidates_in_order(("gemini", "groq", "mistral", "sarvam"))
        return self._candidates_in_order(conversation_intelligence_provider_order())

    def _candidates_in_order(self, names: tuple[str, ...] | list[str]) -> list[LLMRouteCandidate]:
        candidates: list[LLMRouteCandidate] = []
        for name in names:
            item = self._candidate(name, provider_model_for(name), provider_quota_for(name))
            if item:
                candidates.append(item)
        return candidates

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


def llm_provider_status() -> list[dict]:
    router = get_llm_router()
    status = []
    for name in ("gemini", "groq", "mistral", "sarvam", "openai"):
        provider = router.providers.get(name)
        configured = bool(provider) and getattr(provider, "configured", True) is not False
        status.append(
            {
                "provider": name,
                "configured": configured,
                "model": getattr(provider, "default_model", None),
            }
        )
    return status


def log_llm_provider_status(source: str = "worker") -> None:
    status = llm_provider_status()
    print("LLM provider status:", {"source": source, "providers": status})
    for item in status:
        if item["provider"] in {"gemini", "groq"}:
            state = "READY" if item["configured"] else "NOT CONFIGURED (missing API key)"
            print(
                "LLM provider check:",
                {"provider": item["provider"], "model": item["model"], "state": state},
            )
