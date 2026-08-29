"""Production diagnostic alerts. These are not semantic extraction rules."""

from __future__ import annotations

from typing import Any

from services.conversation.event_pipeline.channels import is_generic_task_text
from services.conversation.event_pipeline.schemas import EventDisposition, EventKind
from services.conversation.event_pipeline.validation import mixed_thread_rate


def evaluate_alerts(
    *,
    observability,
    coverage,
    events: list | None = None,
    tasks: list | None = None,
    notes: list | None = None,
    fallback_rate: float | None = None,
    error_rate: float | None = None,
    latency_ms: int | None = None,
    latency_baseline_ms: int | None = None,
) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    events = events or []
    tasks = tasks or []
    notes = notes or []

    async_errors = int(getattr(observability, "asyncLifecycleErrors", 0) or 0)
    if async_errors > 0:
        alerts.append(_alert("ASYNC_LIFECYCLE_ERROR", "critical", f"asyncLifecycleErrors={async_errors}"))

    unaccounted = int(getattr(coverage, "unaccounted_blocks", 0) or 0) if coverage is not None else 0
    if unaccounted > 0:
        alerts.append(_alert("UNACCOUNTED_BLOCKS", "critical", f"unaccountedBlocks={unaccounted}"))

    memory_fail = bool(getattr(coverage, "memoryCoverageFailure", False)) if coverage is not None else False
    if memory_fail:
        alerts.append(_alert("MEMORY_COVERAGE_FAILURE", "critical", "memoryCoverageFailure=true"))

    if coverage is not None and bool(getattr(coverage, "hardFailure", False)):
        alerts.append(_alert("PARTIAL_PUBLICATION", "critical", "coverage.hardFailure=true"))

    mixed = mixed_thread_rate([*tasks, *notes], events) if events else 0.0
    if mixed > 0.05:
        alerts.append(_alert("MIXED_THREAD_RATE", "warning", f"mixedThreadRate={mixed:.3f}"))

    generic = sum(1 for task in tasks if is_generic_task_text(task.title, task.body, getattr(task, "object", None)))
    if generic > 0:
        alerts.append(_alert("GENERIC_TASK_RATE", "warning", f"genericTasks={generic}"))

    if fallback_rate is not None and fallback_rate > 0.25:
        alerts.append(_alert("FALLBACK_RATE_SPIKE", "warning", f"fallbackRate={fallback_rate:.3f}"))

    if error_rate is not None and error_rate > 0.1:
        alerts.append(_alert("PIPELINE_ERROR_RATE_SPIKE", "critical", f"errorRate={error_rate:.3f}"))

    if latency_ms is not None and latency_baseline_ms and latency_ms > max(latency_baseline_ms * 2, 60_000):
        alerts.append(_alert("LATENCY_SPIKE", "warning", f"totalLatencyMs={latency_ms}"))

    provider_errors = int(getattr(observability, "providerErrors", 0) or 0)
    model_calls = max(int(getattr(observability, "gemmaCalls", 0) or 0) + int(getattr(observability, "gptOss120bCalls", 0) or 0) + int(getattr(observability, "gptOss20bCalls", 0) or 0) + int(getattr(observability, "otherLlmCalls", 0) or 0), 1)
    if provider_errors / model_calls > 0.2:
        alerts.append(_alert("PROVIDER_FAILURE_SPIKE", "warning", f"providerErrors={provider_errors}"))

    memory_events = [event for event in events if getattr(event, "channel", "") == "memory" or (getattr(event, "memorySignal", None) and event.memorySignal.isMemoryWorthy)]
    if memory_events and not notes and not _explicit_memory_suppression(memory_events):
        alerts.append(_alert("MEMORY_WITHOUT_NOTES", "warning", f"memoryEvents={len(memory_events)} notesPublished=0"))

    action_events = [
        event
        for event in events
        if getattr(event, "channel", "") == "action"
        or getattr(event, "kind", None) in {EventKind.REQUEST, EventKind.COMMITMENT, EventKind.ASSIGNMENT}
        or bool(getattr(getattr(event, "actionSignal", None), "isActionable", False))
    ]
    if action_events and not tasks and not _explicit_action_abstention(action_events):
        alerts.append(_alert("ACTION_WITHOUT_TASKS", "warning", f"actionEvents={len(action_events)} tasksPublished=0"))

    if coverage is not None and bool(getattr(coverage, "actionCoverageFailure", False)):
        alerts.append(_alert("ACTION_COVERAGE_FAILURE", "critical", "actionCoverageFailure=true"))
    if coverage is not None and bool(getattr(coverage, "semanticCoverageFailure", False)):
        alerts.append(_alert("SEMANTIC_COVERAGE_FAILURE", "critical", f"unaccountedSemanticUnits={getattr(coverage, 'unaccountedSemanticUnits', 0)}"))
    if coverage is not None and "SUSPICIOUS_ZERO_TASK_OUTPUT" in (getattr(coverage, "suspicious", None) or []):
        alerts.append(_alert("SUSPICIOUS_ZERO_TASK_OUTPUT", "critical", "grounded explicit actions produced zero tasks"))
    if coverage is not None and "ACTION_COVERAGE_FAILURE" in (getattr(coverage, "suspicious", None) or []):
        if not any(alert.get("name") == "ACTION_COVERAGE_FAILURE" for alert in alerts):
            alerts.append(_alert("ACTION_COVERAGE_FAILURE", "critical", "ACTION_COVERAGE_FAILURE"))

    return alerts


_PRIORITY_ALERTS = frozenset(
    {
        "ASYNC_LIFECYCLE_ERROR",
        "UNACCOUNTED_BLOCKS",
        "MEMORY_COVERAGE_FAILURE",
        "GENERIC_TASK_RATE",
        "PARTIAL_PUBLICATION",
        "EVENT_PIPELINE_LEGACY_FALLBACK",
        "ACTION_WITHOUT_TASKS",
        "MEMORY_WITHOUT_NOTES",
        "ACTION_COVERAGE_FAILURE",
        "SEMANTIC_COVERAGE_FAILURE",
        "SUSPICIOUS_ZERO_TASK_OUTPUT",
    }
)


def log_priority_alerts(alerts: list[dict[str, Any]] | None) -> None:
    """Immediately surface production-critical pipeline outcomes. No transcript text."""
    for alert in alerts or []:
        name = str(alert.get("name") or "")
        if alert.get("severity") == "critical" or name in _PRIORITY_ALERTS:
            print("[PRIORITY_ALERT]", alert)


def log_pipeline_fallback(*, reason: str, path: str, error: str = "") -> dict[str, Any]:
    payload = {
        "name": "EVENT_PIPELINE_LEGACY_FALLBACK",
        "severity": "critical",
        "detail": reason,
        "pipelineFallback": True,
        "fallbackReason": reason,
        "path": path,
        "error": error[:500],
    }
    print("[PRIORITY_ALERT]", payload)
    return payload


def _explicit_memory_suppression(events: list) -> bool:
    reasons = {str(getattr(event, "dispositionReason", "") or getattr(event, "memoryDispositionReason", "") or "") for event in events}
    return any(
        token in reason
        for reason in reasons
        for token in ("low_value", "unsupported", "duplicate", "rejected", "intentionally", "no_publishable")
    )


def _explicit_action_abstention(events: list) -> bool:
    reasons = {str(getattr(event, "dispositionReason", "") or "") for event in events}
    dispositions = {getattr(event, "disposition", None) for event in events}
    if EventDisposition.INTENTIONALLY_NON_PUBLISHABLE in dispositions or EventDisposition.REJECTED in dispositions:
        return True
    return any(
        token in reason
        for reason in reasons
        for token in ("abstain", "unresolved", "generic", "rejected", "intentionally", "duplicate")
    )


def _alert(name: str, severity: str, detail: str) -> dict[str, Any]:
    return {"name": name, "severity": severity, "detail": detail}
