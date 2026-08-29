"""Capability routing for the hierarchical event pipeline.

This package remains available for rollback when ENABLE_MEETING_PIPELINE=false.
The production default semantic path is services.conversation.meeting_pipeline.

Stages request semantic capabilities. Buddy's existing LLMRouter resolves the
concrete provider/model and fallback chain. This module does not hardcode
Gemma, gpt-oss-120b, or gpt-oss-20b.

Role mapping (current conversation-intelligence policy):

    SEMANTIC_EXTRACTION     → semantic role   (currently Gemma-4-31B-it)
    FINAL_SYNTHESIS         → synthesis role  (currently gpt-oss-120b)
                              Event-pipeline logs this as HIGH_ACCURACY_REASONING.
                              The HIGH_ACCURACY_REASONING enum still routes window
                              extraction to the semantic (Gemma) role.
    VALIDATION              → validation role (gpt-oss-20b preferred, then existing fallback)
    EMBEDDINGS              → existing embedding provider
"""

from __future__ import annotations

from enum import Enum

from apps.api_gateway.config.setting import settings
from services.conversation.budget import expected_request_tokens, safe_input_budget
from services.llm.fallback import FallbackLLMProvider, LLMRouteCandidate
from services.llm.router import LLMCapability, LLMRouter


class PipelineStage(str, Enum):
    TOPIC_LABEL = "topic_label"
    ATOMIC_EVENTS = "atomic_events"
    THREAD_VERIFY = "thread_verify"
    THREAD_HARD = "thread_hard"
    TASK_SYNTHESIS = "task_synthesis"
    NOTE_SYNTHESIS = "note_synthesis"
    VALIDATION = "validation"
    SEMANTIC_COMPLETENESS = "semantic_completeness"
    COVERAGE_REPAIR_EVENTS = "coverage_repair_events"
    COVERAGE_REPAIR_SYNTHESIS = "coverage_repair_synthesis"


# High-accuracy reasoning/synthesis in this repository is FINAL_SYNTHESIS.
HIGH_ACCURACY_SYNTHESIS = LLMCapability.FINAL_SYNTHESIS

STAGE_CAPABILITIES: dict[PipelineStage, LLMCapability] = {
    PipelineStage.TOPIC_LABEL: LLMCapability.SEMANTIC_EXTRACTION,
    PipelineStage.ATOMIC_EVENTS: LLMCapability.SEMANTIC_EXTRACTION,
    PipelineStage.THREAD_VERIFY: LLMCapability.SEMANTIC_EXTRACTION,
    PipelineStage.THREAD_HARD: HIGH_ACCURACY_SYNTHESIS,
    PipelineStage.TASK_SYNTHESIS: HIGH_ACCURACY_SYNTHESIS,
    PipelineStage.NOTE_SYNTHESIS: HIGH_ACCURACY_SYNTHESIS,
    PipelineStage.VALIDATION: LLMCapability.VALIDATION,
    PipelineStage.SEMANTIC_COMPLETENESS: LLMCapability.SEMANTIC_EXTRACTION,
    PipelineStage.COVERAGE_REPAIR_EVENTS: LLMCapability.SEMANTIC_EXTRACTION,
    PipelineStage.COVERAGE_REPAIR_SYNTHESIS: HIGH_ACCURACY_SYNTHESIS,
}

_STAGE_CAP_SETTINGS = {
    PipelineStage.TOPIC_LABEL: "EVENT_PIPELINE_TOPIC_LABEL_MAX_INPUT_TOKENS",
    PipelineStage.ATOMIC_EVENTS: "EVENT_PIPELINE_ATOMIC_EVENT_MAX_INPUT_TOKENS",
    PipelineStage.THREAD_VERIFY: "EVENT_PIPELINE_THREAD_VERIFY_MAX_INPUT_TOKENS",
    PipelineStage.THREAD_HARD: "EVENT_PIPELINE_THREAD_HARD_MAX_INPUT_TOKENS",
    PipelineStage.TASK_SYNTHESIS: "EVENT_PIPELINE_SYNTHESIS_MAX_INPUT_TOKENS",
    PipelineStage.NOTE_SYNTHESIS: "EVENT_PIPELINE_SYNTHESIS_MAX_INPUT_TOKENS",
    PipelineStage.VALIDATION: "EVENT_PIPELINE_VALIDATION_MAX_INPUT_TOKENS",
    PipelineStage.SEMANTIC_COMPLETENESS: "EVENT_PIPELINE_ATOMIC_EVENT_MAX_INPUT_TOKENS",
    PipelineStage.COVERAGE_REPAIR_EVENTS: "EVENT_PIPELINE_ATOMIC_EVENT_MAX_INPUT_TOKENS",
    PipelineStage.COVERAGE_REPAIR_SYNTHESIS: "EVENT_PIPELINE_SYNTHESIS_MAX_INPUT_TOKENS",
}

_STAGE_CAP_DEFAULTS = {
    PipelineStage.TOPIC_LABEL: 800,
    PipelineStage.ATOMIC_EVENTS: 2500,
    PipelineStage.THREAD_VERIFY: 1200,
    PipelineStage.THREAD_HARD: 1800,
    PipelineStage.TASK_SYNTHESIS: 1800,
    PipelineStage.NOTE_SYNTHESIS: 1800,
    PipelineStage.VALIDATION: 1800,
    PipelineStage.SEMANTIC_COMPLETENESS: 2500,
    PipelineStage.COVERAGE_REPAIR_EVENTS: 2500,
    PipelineStage.COVERAGE_REPAIR_SYNTHESIS: 1800,
}

def _topic_bounded_sequence_markers() -> int:
    """Cap extractor payloads to one local topic, not a whole meeting."""
    blocks = int(getattr(settings, "TOPIC_SAFETY_MAX_MICRO_BLOCKS", 6))
    turns = int(getattr(settings, "EVENT_PIPELINE_MICROBLOCK_MAX_TURNS", 5))
    return max(12, blocks * turns)


_TOPIC_MARKERS = _topic_bounded_sequence_markers()

_MAX_SEQUENCE_MARKERS = {
    PipelineStage.TOPIC_LABEL: 8,
    PipelineStage.ATOMIC_EVENTS: _TOPIC_MARKERS,
    PipelineStage.THREAD_VERIFY: 6,
    PipelineStage.THREAD_HARD: 6,
    PipelineStage.TASK_SYNTHESIS: 8,
    PipelineStage.NOTE_SYNTHESIS: 8,
    PipelineStage.VALIDATION: 10,
    PipelineStage.SEMANTIC_COMPLETENESS: _TOPIC_MARKERS,
    PipelineStage.COVERAGE_REPAIR_EVENTS: _TOPIC_MARKERS,
    PipelineStage.COVERAGE_REPAIR_SYNTHESIS: 8,
}


def capability_for_stage(stage: PipelineStage | str) -> LLMCapability:
    key = PipelineStage(stage) if not isinstance(stage, PipelineStage) else stage
    return STAGE_CAPABILITIES[key]


def capability_log_name(stage: PipelineStage | str | LLMCapability) -> str:
    """User-facing capability name for [MODEL_ROUTE] logs.

    Task/note synthesis and hard thread escalation request FINAL_SYNTHESIS from
    the router (gpt-oss-120b). Logs report that role as HIGH_ACCURACY_REASONING.
    """
    if isinstance(stage, LLMCapability):
        capability = stage
    else:
        capability = capability_for_stage(stage)
    if capability == LLMCapability.FINAL_SYNTHESIS:
        return "HIGH_ACCURACY_REASONING"
    if capability == LLMCapability.SEMANTIC_EXTRACTION:
        return "SEMANTIC_EXTRACTION"
    if capability == LLMCapability.VALIDATION:
        return "VALIDATION"
    return str(capability.value).upper()


def stage_log_name(stage: PipelineStage | str) -> str:
    key = PipelineStage(stage) if not isinstance(stage, PipelineStage) else stage
    return {
        PipelineStage.TOPIC_LABEL: "topic_label",
        PipelineStage.ATOMIC_EVENTS: "atomic_event_extraction",
        PipelineStage.THREAD_VERIFY: "thread_semantic_verification",
        PipelineStage.THREAD_HARD: "thread_hard_ambiguity",
        PipelineStage.TASK_SYNTHESIS: "task_synthesis",
        PipelineStage.NOTE_SYNTHESIS: "note_synthesis",
        PipelineStage.VALIDATION: "evidence_validation",
        PipelineStage.SEMANTIC_COMPLETENESS: "semantic_completeness",
        PipelineStage.COVERAGE_REPAIR_EVENTS: "coverage_repair_events",
        PipelineStage.COVERAGE_REPAIR_SYNTHESIS: "coverage_repair_synthesis",
    }.get(key, key.value)


def stage_input_cap(stage: PipelineStage | str) -> int:
    key = PipelineStage(stage) if not isinstance(stage, PipelineStage) else stage
    name = _STAGE_CAP_SETTINGS[key]
    return int(getattr(settings, name, _STAGE_CAP_DEFAULTS[key]))


def cap_payload(text: str, stage: PipelineStage | str) -> str:
    """Conservative stage budget. Never send 200–500 raw chunks even if they fit."""
    key = PipelineStage(stage) if not isinstance(stage, PipelineStage) else stage
    limited = _limit_sequence_markers(text, _MAX_SEQUENCE_MARKERS[key])
    cap = stage_input_cap(key)
    estimated = expected_request_tokens(limited)
    if estimated <= cap:
        return limited
    char_budget = max(200, cap * 4)
    return limited[:char_budget]


def route_for_stage(
    router: LLMRouter,
    stage: PipelineStage | str,
    estimated_tokens: int = 0,
) -> tuple[object, str, LLMCapability]:
    """Resolve provider/model through the existing router, then keep fallbacks that fit.

    Context overflow on the primary must not drop a later candidate that can
    serve the request.
    """
    capability = capability_for_stage(stage)
    provider, model = router.route(capability)
    candidates = list(router._cost_optimized_candidates(capability) or [])
    if not candidates:
        return provider, model, capability
    budgeted = max(1, int(estimated_tokens or 0))
    fitting = [item for item in candidates if _candidate_fits(item, budgeted)]
    if not fitting:
        return provider, model, capability
    if len(fitting) == len(candidates):
        return provider, model, capability
    wrapped = FallbackLLMProvider(fitting[0].provider.name, fitting)
    return wrapped, fitting[0].model, capability


def _candidate_fits(candidate: LLMRouteCandidate, estimated_tokens: int) -> bool:
    usable = safe_input_budget(candidate.provider.name, model=candidate.model)
    return estimated_tokens <= usable


def _limit_sequence_markers(text: str, max_markers: int) -> str:
    count = text.count("\n[") + (1 if text.startswith("[") else 0)
    if count <= max_markers:
        return text
    lines = text.splitlines()
    kept: list[str] = []
    seen = 0
    for line in lines:
        if line.startswith("[") and "]" in line[:12]:
            seen += 1
            if seen > max_markers:
                continue
        kept.append(line)
    return "\n".join(kept)
