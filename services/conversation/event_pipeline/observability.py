from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

from services.conversation.event_pipeline.cost import classify_model, estimate_cost_usd
from services.conversation.event_pipeline.schemas import (
    CoverageLedger,
    ModelRouteRecord,
    PipelineObservability,
    StageMetrics,
)

_OBS: ContextVar[PipelineObservability | None] = ContextVar("event_pipeline_obs", default=None)
_STAGE: ContextVar[StageMetrics | None] = ContextVar("event_pipeline_stage", default=None)


def bind_observability(obs: PipelineObservability):
    return _OBS.set(obs)


def reset_observability(token) -> None:
    _OBS.reset(token)


def current_observability() -> PipelineObservability | None:
    return _OBS.get()


@contextmanager
def timed_stage(obs: PipelineObservability, name: str) -> Iterator[StageMetrics]:
    import time

    started = time.perf_counter()
    metrics = StageMetrics(name=name)
    obs.stages.append(metrics)
    token = _STAGE.set(metrics)
    try:
        yield metrics
    finally:
        metrics.durationMs = int((time.perf_counter() - started) * 1000)
        _STAGE.reset(token)


def log_model_route(
    *,
    stage: str,
    capability: str,
    provider: str,
    model: str,
    fallback: bool = False,
    requested: str = "",
) -> None:
    line = (
        f"[MODEL_ROUTE] stage={stage} capability={capability} "
        f"provider={provider} model={model} fallback={str(fallback).lower()}"
    )
    if requested:
        line += f" requested={requested}"
    print(line)
    obs = _OBS.get()
    if obs is None:
        return
    obs.logs.append(line)
    obs.modelRoutes.append(
        ModelRouteRecord(
            stage=stage,
            capability=capability,
            provider=provider,
            model=model,
            fallback=fallback,
            requested=requested,
        )
    )


def record_llm_usage(
    *,
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    fallback: bool = False,
    attempts: int = 1,
) -> None:
    obs = _OBS.get()
    if obs is None:
        return
    family = classify_model(model)
    if family == "gemma":
        obs.gemmaCalls += 1
    elif family == "gpt_oss_120b":
        obs.gptOss120bCalls += 1
    elif family == "gpt_oss_20b":
        obs.gptOss20bCalls += 1
    else:
        obs.otherLlmCalls += 1
    obs.inputTokens += max(0, int(input_tokens or 0))
    obs.outputTokens += max(0, int(output_tokens or 0))
    if fallback:
        obs.fallbackCount += 1
    extra_attempts = max(0, int(attempts or 1) - 1)
    if extra_attempts:
        obs.retryCount += extra_attempts
    stage = _STAGE.get()
    if stage is not None:
        stage.llmCalls += 1
        stage.inputTokens += max(0, int(input_tokens or 0))
        stage.outputTokens += max(0, int(output_tokens or 0))
    obs.estimatedCostUsd = estimate_cost_usd(obs)


def record_embedding_usage(*, requests: int = 1, items: int = 0) -> None:
    obs = _OBS.get()
    if obs is None:
        return
    obs.embeddingRequests += max(0, int(requests or 0))
    obs.embeddingItems += max(0, int(items or 0))
    stage = _STAGE.get()
    if stage is not None:
        stage.embeddingCalls += max(0, int(requests or 0))
        stage.embeddingItems += max(0, int(items or 0))


def record_provider_error(reason: str | None = None) -> None:
    from services.llm.schema_adapter import ASYNC_LIFECYCLE_ERROR

    obs = _OBS.get()
    if obs is None:
        return
    if str(reason or "") == ASYNC_LIFECYCLE_ERROR:
        obs.asyncLifecycleErrors += 1
        return
    obs.providerErrors += 1


def record_async_lifecycle_error() -> None:
    obs = _OBS.get()
    if obs is not None:
        obs.asyncLifecycleErrors += 1


def record_failure(error: Exception) -> None:
    from services.llm.async_runtime import is_async_lifecycle_error
    from services.llm.schema_adapter import classify_failure_class

    obs = _OBS.get()
    if obs is None:
        return
    if is_async_lifecycle_error(error) or classify_failure_class(error) == "ASYNC_LIFECYCLE_ERROR":
        obs.asyncLifecycleErrors += 1
        return
    obs.providerErrors += 1


def job_record(obs: PipelineObservability) -> dict:
    return {
        "sessionId": obs.sessionId,
        "workerId": obs.workerId,
        "mode": obs.mode,
        "pipelineMode": obs.mode,
        "rawSequences": obs.rawSequences,
        "usefulSequences": obs.usefulSequences,
        "microBlocks": obs.microBlockCount,
        "topics": obs.topicCount,
        "events": obs.eventCount,
        "threads": obs.threadCount,
        "actionEvents": obs.actionEvents,
        "memoryEvents": obs.memoryEvents,
        "tasksPublished": obs.tasksPublished,
        "notesPublished": obs.notesPublished,
        "genericTasks": obs.genericTasks,
        "mixedThreads": obs.mixedThreads,
        "duplicates": obs.duplicates,
        "unaccountedBlocks": obs.unaccountedBlocks,
        "unaccountedSemanticUnits": obs.unaccountedSemanticUnits,
        "semanticCoverageFailures": obs.semanticCoverageFailures,
        "semanticUnitsDetected": obs.semanticUnitsDetected,
        "semanticUnitsCreated": obs.semanticUnitsCreated,
        "memoryCoverageFailures": obs.memoryCoverageFailures,
        "actionCoverageFailures": obs.actionCoverageFailures,
        "atomicEvents": obs.atomicEvents,
        "actionableEvents": obs.actionableEvents,
        "explicitActionEvents": obs.explicitActionEvents,
        "groundedActionObjects": obs.groundedActionObjects,
        "actionChannelEvents": obs.actionChannelEvents,
        "taskSynthesisInputEvents": obs.taskSynthesisInputEvents,
        "taskCandidates": obs.taskCandidates,
        "taskValidationAccepted": obs.taskValidationAccepted,
        "taskValidationRejected": obs.taskValidationRejected,
        "tasksPersisted": obs.tasksPersisted,
        "tasksReturnedByApi": obs.tasksReturnedByApi,
        "embeddingCalls": obs.embedding_calls(),
        "GemmaCalls": obs.gemmaCalls,
        "gptOss120bCalls": obs.gptOss120bCalls,
        "gptOss20bCalls": obs.gptOss20bCalls,
        "inputTokens": obs.inputTokens,
        "outputTokens": obs.outputTokens,
        "fallbackCount": obs.fallbackCount,
        "retryCount": obs.retryCount,
        "providerErrors": obs.providerErrors,
        "asyncLifecycleErrors": obs.asyncLifecycleErrors,
        "stageLatency": {stage.name: stage.durationMs for stage in obs.stages},
        "totalLatency": sum(stage.durationMs for stage in obs.stages),
        "pipelineVersion": obs.pipelineVersion,
        "eventSchemaVersion": obs.eventSchemaVersion,
        "promptVersion": obs.promptVersion,
        "artifactPipelineVersion": obs.artifactPipelineVersion,
        "alerts": list(obs.alerts),
    }


def log_pipeline(obs: PipelineObservability, cleaning, blocks, topics, events, actions, memory, other, threads, links, tasks, notes, coverage: CoverageLedger | None) -> None:
    from services.conversation.event_pipeline.alerts import evaluate_alerts, log_priority_alerts
    from services.conversation.event_pipeline.channels import is_generic_task_text
    from services.conversation.event_pipeline.flags import event_pipeline_mode
    from services.conversation.event_pipeline.validation import mixed_thread_rate
    from services.conversation.event_pipeline.versions import version_metadata
    from services.llm.async_runtime import worker_id

    versions = version_metadata()
    obs.workerId = obs.workerId or worker_id()
    obs.mode = obs.mode or event_pipeline_mode()
    obs.rawSequences = getattr(cleaning, "totalSequences", 0)
    obs.usefulSequences = getattr(cleaning, "usefulSequences", 0)
    obs.microBlockCount = len(blocks)
    obs.topicCount = len(topics)
    obs.eventCount = len(events)
    obs.threadCount = len(threads)
    obs.actionEvents = len(actions)
    obs.memoryEvents = len(memory)
    obs.tasksPublished = len(tasks)
    obs.notesPublished = len(notes)
    obs.genericTasks = sum(1 for task in tasks if is_generic_task_text(task.title, task.body, getattr(task, "object", None)))
    obs.mixedThreads = 1 if mixed_thread_rate([*tasks, *notes], events) > 0 else 0
    obs.duplicates = sum(1 for event in events if getattr(event, "disposition", None) and event.disposition.value == "DUPLICATE")
    obs.unaccountedBlocks = coverage.unaccounted_blocks if coverage else 0
    obs.memoryCoverageFailures = 1 if coverage and coverage.memoryCoverageFailure else 0
    obs.actionCoverageFailures = 1 if coverage and coverage.actionCoverageFailure else 0
    obs.semanticCoverageFailures = 1 if coverage and getattr(coverage, "semanticCoverageFailure", False) else 0
    obs.unaccountedSemanticUnits = int(getattr(coverage, "unaccountedSemanticUnits", 0) or 0) if coverage else 0
    obs.semanticUnitsDetected = int(getattr(coverage, "semanticUnitsDetected", 0) or 0) if coverage else 0
    obs.semanticUnitsCreated = int(getattr(coverage, "semanticUnitsCreated", 0) or 0) if coverage else 0
    fill_task_pipeline_trace(obs, events, actions, tasks, coverage)
    obs.pipelineVersion = versions["pipelineVersion"]
    obs.eventSchemaVersion = versions["eventSchemaVersion"]
    obs.promptVersion = versions["promptVersion"]
    obs.artifactPipelineVersion = versions["artifactPipelineVersion"]
    obs.alerts = evaluate_alerts(
        observability=obs,
        coverage=coverage,
        events=events,
        tasks=tasks,
        notes=notes,
        fallback_rate=(obs.fallbackCount / max(obs.llm_calls(), 1)),
        latency_ms=sum(stage.durationMs for stage in obs.stages),
    )
    log_priority_alerts(obs.alerts)
    record = job_record(obs)
    lines = [
        f"[CLEANING] raw={getattr(cleaning, 'totalSequences', 0)} useful={getattr(cleaning, 'usefulSequences', 0)} excluded={getattr(cleaning, 'excludedStructuralSequences', 0)}",
        f"[MICRO_BLOCKS] count={len(blocks)}",
        f"[TOPICS] count={len(topics)}",
        f"[EVENT_EXTRACTION] events={len(events)} actions={len(actions)} memory={len(memory)} other={len(other)}",
        f"[THREAD_LINKING] threads={len(threads)} cross_window_links={sum(1 for link in links if getattr(link, 'crossWindow', False))}",
        f"[TASK_SYNTHESIS] candidates={len(actions)} accepted={len(tasks)} rejected={max(0, len(actions) - len(tasks))}",
        f"[NOTE_SYNTHESIS] candidates={len(memory)} accepted={len(notes)} rejected={max(0, len(memory) - len(notes))}",
        f"[COVERAGE] usefulBlocks={len(blocks)} accountedBlocks={len(blocks) - (coverage.unaccounted_blocks if coverage else 0)} unaccounted={coverage.unaccounted_blocks if coverage else 0} unaccountedSemanticUnits={coverage.unaccountedSemanticUnits if coverage else 0} semanticCoverage={getattr(coverage, 'semanticCoverage', 1.0) if coverage else 1.0}",
        f"[PERSISTENCE] tasks={len(tasks)} notes={len(notes)}",
        _task_pipeline_trace_line(obs),
        (
            f"[COST] embedding_requests={obs.embeddingRequests} embedding_items={obs.embeddingItems} "
            f"gemma={obs.gemmaCalls} gpt_oss_120b={obs.gptOss120bCalls} gpt_oss_20b={obs.gptOss20bCalls} "
            f"input_tokens={obs.inputTokens} output_tokens={obs.outputTokens} "
            f"fallbacks={obs.fallbackCount} retries={obs.retryCount} "
            f"estimated_usd={obs.estimatedCostUsd if obs.estimatedCostUsd is not None else 'not measured'}"
        ),
    ]
    obs.logs.extend(lines)
    for line in lines:
        print(line)
    print("[JOB_RECORD]", {key: value for key, value in record.items() if key != "alerts"})
    if obs.alerts:
        print("[ALERTS]", obs.alerts)


def fill_task_pipeline_trace(obs: PipelineObservability, events, actions, tasks, coverage) -> None:
    from services.conversation.event_pipeline.channels import (
        action_object_grounded,
        action_strength,
        event_is_actionable,
        object_grounding_type,
    )

    actionable = [event for event in events if event_is_actionable(event)]
    explicit = [event for event in actionable if action_strength(event.actionSignal) == "EXPLICIT"]
    grounded = [
        event
        for event in explicit
        if action_object_grounded(event) and object_grounding_type(event) not in {"INFERRED", "UNRESOLVED"}
    ]
    obs.atomicEvents = len(events)
    obs.actionableEvents = len(actionable)
    obs.explicitActionEvents = len(explicit)
    obs.groundedActionObjects = len(grounded)
    obs.actionChannelEvents = len(actions)
    obs.taskSynthesisInputEvents = len(actions)
    if obs.taskCandidates is None or obs.taskCandidates == 0:
        obs.taskCandidates = obs.taskCandidates or len(tasks)
    obs.tasksPersisted = len(tasks)
    if not obs.tasksReturnedByApi:
        obs.tasksReturnedByApi = len(tasks)
    obs.tasksPublished = len(tasks)


def _task_pipeline_trace_line(obs: PipelineObservability) -> str:
    return (
        f"[TASK_PIPELINE_TRACE] atomicEvents={obs.atomicEvents} "
        f"actionableEvents={obs.actionableEvents} "
        f"explicitActionEvents={obs.explicitActionEvents} "
        f"groundedActionObjects={obs.groundedActionObjects} "
        f"actionChannelEvents={obs.actionChannelEvents} "
        f"taskSynthesisInputEvents={obs.taskSynthesisInputEvents} "
        f"taskCandidates={obs.taskCandidates} "
        f"accepted={obs.taskValidationAccepted} "
        f"rejected={obs.taskValidationRejected} "
        f"persisted={obs.tasksPersisted} "
        f"returned={obs.tasksReturnedByApi}"
    )


def stage_lookup(obs: PipelineObservability, name: str) -> StageMetrics | None:
    for stage in obs.stages:
        if stage.name == name:
            return stage
    return None
