from __future__ import annotations

import asyncio

from enum import Enum

from apps.api_gateway.config.setting import settings
from services.llm.fallback import FallbackLLMProvider, LLMRouteCandidate
from services.llm.models import LLMProvider
from services.llm.providers import (
    build_anthropic_provider,
    build_gemini_provider,
    build_groq_provider,
    build_krutrim_provider,
    build_mistral_provider,
    build_openai_provider,
    build_sarvam_provider,
    NotConfiguredProvider,
)
from services.llm.quota import ProviderQuota
from services.llm.routing_policy import (
    CHAT_PROVIDER_ORDER,
    NORMALIZATION_PROVIDER_ORDER,
    conversation_route_spec,
    provider_model_for,
    provider_quota_for,
    uses_conversation_intelligence_policy,
)


class LLMCapability(str, Enum):
    HIGH_ACCURACY_REASONING = "high_accuracy_reasoning"
    SEMANTIC_EXTRACTION = "semantic_extraction"
    CHAT_ANSWER = "chat_answer"
    VALIDATION = "validation"
    COMPLEX_TASK_MATCHING = "complex_task_matching"
    SIMPLE_SUMMARY = "simple_summary"
    NORMALIZATION = "normalization"
    FALLBACK = "fallback"
    FINAL_SYNTHESIS = "final_synthesis"


class LLMRouter:
    def __init__(self, providers: dict[str, LLMProvider]):
        self.providers = providers

    def route(self, capability: LLMCapability) -> tuple[LLMProvider, str]:
        if settings.LLM_ENABLE_COST_OPTIMIZED_ROUTING:
            candidates = self._cost_optimized_candidates(capability)
            if candidates:
                primary = candidates[0]
                return FallbackLLMProvider(primary.provider.name, candidates), primary.model
            if uses_conversation_intelligence_policy(capability):
                spec = conversation_route_spec(capability)
                provider_name, model = spec[0]
                provider = self.providers.get(provider_name) or NotConfiguredProvider(provider_name)
                return provider, model

        provider_name = settings.LLM_DEFAULT_PROVIDER
        model = settings.LLM_DEFAULT_MODEL
        if capability in {
            LLMCapability.VALIDATION,
            LLMCapability.HIGH_ACCURACY_REASONING,
            LLMCapability.SEMANTIC_EXTRACTION,
            LLMCapability.COMPLEX_TASK_MATCHING,
            LLMCapability.FINAL_SYNTHESIS,
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
        if uses_conversation_intelligence_policy(capability):
            return self._conversation_role_candidates(capability)
        if capability == LLMCapability.CHAT_ANSWER:
            return self._candidates_in_order(CHAT_PROVIDER_ORDER)
        if capability == LLMCapability.NORMALIZATION:
            return self._candidates_in_order(NORMALIZATION_PROVIDER_ORDER)
        return self._conversation_role_candidates(capability)

    def _conversation_role_candidates(self, capability: LLMCapability) -> list[LLMRouteCandidate]:
        candidates: list[LLMRouteCandidate] = []
        for provider_name, model in conversation_route_spec(capability):
            item = self._candidate(provider_name, model, provider_quota_for(provider_name))
            if item:
                candidates.append(item)
        return candidates

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

    async def aclose_transports(self) -> None:
        for provider in self.providers.values():
            closer = getattr(provider, "aclose", None)
            if closer is None:
                continue
            result = closer()
            if asyncio.iscoroutine(result):
                await result


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
                "krutrim": build_krutrim_provider(),
            }
        )
    return _router


async def close_llm_runtime() -> None:
    """Dispose loop-bound HTTP transports. Router configuration is kept."""
    if _router is not None:
        await _router.aclose_transports()
    try:
        from services.vector.embedding_service import close_embedding_client

        await close_embedding_client()
    except Exception:
        pass
    try:
        from services.speech.providers.sarvam_provider import close_sarvam_stt_client

        await close_sarvam_stt_client()
    except Exception:
        pass


def llm_provider_status() -> list[dict]:
    router = get_llm_router()
    status = []
    for name in ("krutrim", "mistral", "gemini", "groq", "sarvam", "openai"):
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
        if item["provider"] in {"krutrim", "mistral", "gemini", "groq"}:
            state = "READY" if item["configured"] else "NOT CONFIGURED (missing API key)"
            print(
                "LLM provider check:",
                {"provider": item["provider"], "model": item["model"], "state": state},
            )
