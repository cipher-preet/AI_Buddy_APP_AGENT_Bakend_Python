from __future__ import annotations

from apps.api_gateway.config.setting import settings
from services.llm.quota import ProviderQuota


# Unrelated chat/normalization still use the free-tier chain. Conversation
# intelligence no longer shares this order.
CHAT_PROVIDER_ORDER = ("groq", "gemini", "mistral", "sarvam")
NORMALIZATION_PROVIDER_ORDER = ("gemini", "groq", "mistral", "sarvam")
CONVERSATION_INTELLIGENCE_FREE_PROVIDERS = ("groq", "gemini", "sarvam")

CONVERSATION_INTELLIGENCE_CAPABILITY_VALUES = {
    "high_accuracy_reasoning",
    "validation",
    "simple_summary",
    "complex_task_matching",
    "fallback",
    "final_synthesis",
}

_SEMANTIC_CAPABILITIES = {
    "high_accuracy_reasoning",
    "simple_summary",
    "complex_task_matching",
}
_SYNTHESIS_CAPABILITIES = {"final_synthesis"}
_VALIDATION_CAPABILITIES = {"validation"}
_VALIDATION_FALLBACK_CAPABILITIES = {"fallback"}


def conversation_intelligence_provider_order() -> tuple[str, ...]:
    return CHAT_PROVIDER_ORDER


def uses_conversation_intelligence_policy(capability) -> bool:
    value = capability.value if hasattr(capability, "value") else str(capability)
    return value in CONVERSATION_INTELLIGENCE_CAPABILITY_VALUES


def conversation_role_for(capability) -> str:
    value = capability.value if hasattr(capability, "value") else str(capability)
    if value in _SYNTHESIS_CAPABILITIES:
        return "synthesis"
    if value in _VALIDATION_CAPABILITIES:
        return "validation"
    if value in _VALIDATION_FALLBACK_CAPABILITIES:
        return "validation_fallback"
    return "semantic"


def conversation_route_spec(capability) -> list[tuple[str, str]]:
    """Role-specific (provider, model) pairs. Not a generic free-model chain."""
    role = conversation_role_for(capability)
    if role == "semantic":
        return [(settings.CONVERSATION_SEMANTIC_PROVIDER, settings.CONVERSATION_SEMANTIC_MODEL)]
    if role == "synthesis":
        pairs = [(settings.CONVERSATION_SYNTHESIS_PROVIDER, settings.CONVERSATION_SYNTHESIS_MODEL)]
        fallback = (
            settings.CONVERSATION_SYNTHESIS_FALLBACK_PROVIDER,
            settings.CONVERSATION_SYNTHESIS_FALLBACK_MODEL,
        )
        if fallback[0] and fallback[1] and fallback not in pairs:
            pairs.append(fallback)
        return pairs
    if role == "validation":
        return [
            (settings.CONVERSATION_VALIDATION_PROVIDER, settings.CONVERSATION_VALIDATION_MODEL),
            (
                settings.CONVERSATION_VALIDATION_FALLBACK_PROVIDER,
                settings.CONVERSATION_VALIDATION_FALLBACK_MODEL,
            ),
        ]
    if role == "validation_fallback":
        return [
            (
                settings.CONVERSATION_VALIDATION_FALLBACK_PROVIDER,
                settings.CONVERSATION_VALIDATION_FALLBACK_MODEL,
            )
        ]
    return [(settings.CONVERSATION_SEMANTIC_PROVIDER, settings.CONVERSATION_SEMANTIC_MODEL)]


def provider_model_for(provider_name: str) -> str:
    if provider_name == "groq":
        return settings.GROQ_FREE_MODEL
    if provider_name == "gemini":
        return settings.GEMINI_FREE_MODEL
    if provider_name == "mistral":
        return settings.MISTRAL_CHEAP_MODEL
    if provider_name == "sarvam":
        return settings.SARVAM_DEFAULT_MODEL
    if provider_name == "krutrim":
        return settings.CONVERSATION_SEMANTIC_MODEL
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
