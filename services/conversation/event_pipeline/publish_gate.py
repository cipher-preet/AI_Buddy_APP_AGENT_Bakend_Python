"""Publication gate: never publish partial unvalidated event-pipeline artifacts."""

from __future__ import annotations

from services.conversation.event_pipeline.flags import coverage_ledger_enabled
from services.conversation.event_pipeline.schemas import EventPipelineResult

REQUIRED_STAGES = (
    "cleaning",
    "micro_blocks",
    "topics",
    "event_extraction",
    "thread_linking",
    "task_synthesis",
    "note_synthesis",
    "evidence_validation",
)


class EventPipelineHardFailure(RuntimeError):
    def __init__(self, reason: str, message: str | None = None):
        super().__init__(message or reason)
        self.reason = reason
        self.retryable = False
        self.failure_reason = "EVENT_PIPELINE_HARD_FAILURE"


def publication_ready(result: EventPipelineResult) -> tuple[bool, str]:
    obs = result.observability
    if int(getattr(obs, "asyncLifecycleErrors", 0) or 0) > 0:
        return False, "async_lifecycle_error"
    stage_names = {stage.name for stage in obs.stages}
    missing = [name for name in REQUIRED_STAGES if name not in stage_names]
    if missing:
        return False, f"incomplete_stages:{','.join(missing)}"
    coverage = result.coverage
    if coverage_ledger_enabled():
        if coverage is None:
            return False, "coverage_ledger_missing"
        if int(coverage.unaccounted_blocks or 0) > 0:
            return False, "unaccounted_blocks"
        if bool(coverage.memoryCoverageFailure):
            return False, "memory_coverage_failure"
        if bool(getattr(coverage, "actionCoverageFailure", False)):
            return False, "action_coverage_failure"
        if bool(getattr(coverage, "semanticCoverageFailure", False)):
            return False, "semantic_coverage_failure"
        if bool(coverage.hardFailure):
            return False, "coverage_hard_failure"
    return True, ""


def require_publication_ready(result: EventPipelineResult) -> None:
    ok, reason = publication_ready(result)
    if not ok:
        raise EventPipelineHardFailure(reason, f"event pipeline publication blocked: {reason}")
