from __future__ import annotations

from apps.api_gateway.config.setting import settings
from services.llm.quota import ProviderQuota


CONVERSATION_INTELLIGENCE_PROVIDER_ORDER = ("groq", "gemini", "mistral", "sarvam")

CONVERSATION_INTELLIGENCE_CAPABILITY_VALUES = {
    "high_accuracy_reasoning",
    "validation",
    "simple_summary",
    "complex_task_matching",
    "fallback",
}


def conversation_intelligence_provider_order() -> tuple[str, ...]:
    return CONVERSATION_INTELLIGENCE_PROVIDER_ORDER


def uses_conversation_intelligence_policy(capability) -> bool:
    value = capability.value if hasattr(capability, "value") else str(capability)
    return value in CONVERSATION_INTELLIGENCE_CAPABILITY_VALUES


def provider_model_for(provider_name: str) -> str:
    if provider_name == "groq":
        return settings.GROQ_FREE_MODEL
    if provider_name == "gemini":
        return settings.GEMINI_FREE_MODEL
    if provider_name == "mistral":
        return settings.MISTRAL_CHEAP_MODEL
    if provider_name == "sarvam":
        return settings.SARVAM_DEFAULT_MODEL
    return settings.LLM_DEFAULT_MODEL


def provider_quota_for(provider_name: str) -> ProviderQuota | None:
    if provider_name == "groq":
        return ProviderQuota(
            rpm=settings.GROQ_MAX_RPM,
            rpd=settings.GROQ_MAX_RPD,
            tpm=settings.GROQ_MAX_TPM,
            tpd=settings.GROQ_MAX_TPD,
        )
    if provider_name == "gemini":
        return ProviderQuota(rpm=settings.GEMINI_MAX_RPM, rpd=settings.GEMINI_MAX_RPD)
    return None
