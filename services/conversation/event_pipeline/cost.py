"""Estimated provider cost from observed token usage. Missing rates stay unmeasured."""

from __future__ import annotations

from typing import Any

from apps.api_gateway.config.setting import settings


def classify_model(model: str | None) -> str:
    value = (model or "").casefold()
    if "gemma" in value:
        return "gemma"
    if "gpt-oss-120b" in value or "gpt_oss_120b" in value:
        return "gpt_oss_120b"
    if "gpt-oss-20b" in value or "gpt_oss_20b" in value:
        return "gpt_oss_20b"
    if "ministral" in value:
        return "validation_primary"
    return "other"


def estimate_cost_usd(obs) -> float | None:
    rates = _rates()
    if not any(value is not None for value in rates.values()):
        return None
    total = 0.0
    measured = False
    gemma_in, gemma_out = rates["gemma_in"], rates["gemma_out"]
    oss120_in, oss120_out = rates["oss120_in"], rates["oss120_out"]
    oss20_in, oss20_out = rates["oss20_in"], rates["oss20_out"]
    embed = rates["embed"]
    calls = max(obs.gemmaCalls + obs.gptOss120bCalls + obs.gptOss20bCalls + obs.otherLlmCalls, 1)
    input_share = obs.inputTokens / calls
    output_share = obs.outputTokens / calls
    if gemma_in is not None or gemma_out is not None:
        measured = True
        total += _tokens_cost(input_share * obs.gemmaCalls, gemma_in) + _tokens_cost(output_share * obs.gemmaCalls, gemma_out)
    if oss120_in is not None or oss120_out is not None:
        measured = True
        total += _tokens_cost(input_share * obs.gptOss120bCalls, oss120_in) + _tokens_cost(
            output_share * obs.gptOss120bCalls, oss120_out
        )
    if oss20_in is not None or oss20_out is not None:
        measured = True
        total += _tokens_cost(input_share * obs.gptOss20bCalls, oss20_in) + _tokens_cost(
            output_share * obs.gptOss20bCalls, oss20_out
        )
    if embed is not None:
        measured = True
        total += _tokens_cost(obs.embeddingItems * 32, embed)
    return round(total, 6) if measured else None


def cost_report(obs) -> dict[str, Any]:
    estimated = estimate_cost_usd(obs)
    stage_tokens = {
        stage.name: {
            "durationMs": stage.durationMs,
            "inputTokens": stage.inputTokens,
            "outputTokens": stage.outputTokens,
            "llmCalls": stage.llmCalls,
            "embeddingCalls": stage.embeddingCalls,
        }
        for stage in obs.stages
    }
    return {
        "embeddingRequests": obs.embeddingRequests,
        "embeddingItems": obs.embeddingItems,
        "embeddingCostUsd": _stage_cost(obs, "embedding") if _rates()["embed"] is not None else "not measured",
        "semanticExtractionCostUsd": _named_stage_cost(obs, ("event_extraction", "topics", "micro_blocks")),
        "threadVerificationCostUsd": _named_stage_cost(obs, ("thread_linking",)),
        "taskSynthesisCostUsd": _named_stage_cost(obs, ("task_synthesis",)),
        "noteSynthesisCostUsd": _named_stage_cost(obs, ("note_synthesis",)),
        "validationCostUsd": _named_stage_cost(obs, ("evidence_validation",)),
        "gemmaCalls": obs.gemmaCalls,
        "gptOss120bCalls": obs.gptOss120bCalls,
        "gptOss20bCalls": obs.gptOss20bCalls,
        "otherLlmCalls": obs.otherLlmCalls,
        "inputTokens": obs.inputTokens,
        "outputTokens": obs.outputTokens,
        "latencyByStageMs": {stage.name: stage.durationMs for stage in obs.stages},
        "totalLatencyMs": sum(stage.durationMs for stage in obs.stages),
        "fallbackCount": obs.fallbackCount,
        "retryCount": obs.retryCount,
        "providerErrors": getattr(obs, "providerErrors", 0),
        "asyncLifecycleErrors": getattr(obs, "asyncLifecycleErrors", 0),
        "estimatedCostUsd": estimated if estimated is not None else "not measured",
        "stageTokens": stage_tokens,
        "modelRoutes": [item.model_dump() for item in obs.modelRoutes],
    }


def _rates() -> dict[str, float | None]:
    return {
        "gemma_in": _maybe_float(getattr(settings, "LLM_COST_GEMMA_INPUT_PER_MILLION", None)),
        "gemma_out": _maybe_float(getattr(settings, "LLM_COST_GEMMA_OUTPUT_PER_MILLION", None)),
        "oss120_in": _maybe_float(getattr(settings, "LLM_COST_GPT_OSS_120B_INPUT_PER_MILLION", None)),
        "oss120_out": _maybe_float(getattr(settings, "LLM_COST_GPT_OSS_120B_OUTPUT_PER_MILLION", None)),
        "oss20_in": _maybe_float(getattr(settings, "LLM_COST_GPT_OSS_20B_INPUT_PER_MILLION", None)),
        "oss20_out": _maybe_float(getattr(settings, "LLM_COST_GPT_OSS_20B_OUTPUT_PER_MILLION", None)),
        "embed": _maybe_float(getattr(settings, "LLM_COST_EMBEDDING_PER_MILLION", None)),
    }


def _maybe_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _tokens_cost(tokens: float, rate_per_million: float | None) -> float:
    if not tokens or rate_per_million is None:
        return 0.0
    return (float(tokens) / 1_000_000.0) * float(rate_per_million)


def _named_stage_cost(obs, names: tuple[str, ...]):
    stages = [stage for stage in obs.stages if stage.name in names]
    return {
        "inputTokens": sum(stage.inputTokens for stage in stages),
        "outputTokens": sum(stage.outputTokens for stage in stages),
        "llmCalls": sum(stage.llmCalls for stage in stages),
        "estimatedCostUsd": "not measured",
    }


def _stage_cost(obs, kind: str):
    rates = _rates()
    if kind == "embedding":
        if rates["embed"] is None:
            return {"items": obs.embeddingItems, "requests": obs.embeddingRequests, "estimatedCostUsd": "not measured"}
        return {
            "items": obs.embeddingItems,
            "requests": obs.embeddingRequests,
            "estimatedCostUsd": round(_tokens_cost(obs.embeddingItems * 32, rates["embed"]), 6),
        }
    return "not measured"
