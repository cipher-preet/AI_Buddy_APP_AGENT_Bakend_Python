"""Job-level execution budget for the event pipeline.

Limits are intentionally high so normal long meetings complete. Exceeding a
budget fails the job without publishing partial unvalidated artifacts.
"""

from __future__ import annotations

import time
from contextvars import ContextVar

from apps.api_gateway.config.setting import settings

_BUDGET: ContextVar["PipelineBudget | None"] = ContextVar("event_pipeline_budget", default=None)


class PipelineBudgetExceeded(RuntimeError):
    def __init__(self, limit_name: str, message: str | None = None):
        super().__init__(message or f"event pipeline budget exceeded: {limit_name}")
        self.limit_name = limit_name
        self.retryable = False
        self.failure_reason = "PIPELINE_BUDGET_EXCEEDED"


class PipelineBudget:
    def __init__(
        self,
        *,
        max_runtime: float | None = None,
        max_model_calls: int | None = None,
        max_retries: int | None = None,
    ):
        self.started = time.monotonic()
        self.model_calls = 0
        self.retries = 0
        self.max_runtime = float(
            max_runtime if max_runtime is not None else getattr(settings, "EVENT_PIPELINE_MAX_TOTAL_RUNTIME", 1800)
        )
        self.max_model_calls = int(
            max_model_calls if max_model_calls is not None else getattr(settings, "EVENT_PIPELINE_MAX_MODEL_CALLS", 800)
        )
        self.max_retries = int(
            max_retries if max_retries is not None else getattr(settings, "EVENT_PIPELINE_MAX_RETRIES", 12)
        )

    def elapsed_seconds(self) -> float:
        return max(0.0, time.monotonic() - self.started)

    def remaining_seconds(self) -> float:
        return max(0.1, self.max_runtime - self.elapsed_seconds())

    def snapshot(self) -> dict:
        return {
            "elapsedSeconds": round(self.elapsed_seconds(), 3),
            "modelCalls": self.model_calls,
            "retries": self.retries,
            "maxRuntime": self.max_runtime,
            "maxModelCalls": self.max_model_calls,
            "maxRetries": self.max_retries,
        }

    def check(self) -> None:
        if self.elapsed_seconds() > self.max_runtime:
            raise PipelineBudgetExceeded("EVENT_PIPELINE_MAX_TOTAL_RUNTIME")
        if self.model_calls > self.max_model_calls:
            raise PipelineBudgetExceeded("EVENT_PIPELINE_MAX_MODEL_CALLS")
        if self.retries > self.max_retries:
            raise PipelineBudgetExceeded("EVENT_PIPELINE_MAX_RETRIES")

    def charge_model_call(self) -> None:
        self.model_calls += 1
        self.check()

    def charge_retry(self, extra_attempts: int = 1) -> None:
        self.retries += max(0, int(extra_attempts or 0))
        self.check()


def bind_budget(budget: PipelineBudget):
    return _BUDGET.set(budget)


def reset_budget(token) -> None:
    _BUDGET.reset(token)


def current_budget() -> PipelineBudget | None:
    return _BUDGET.get()


def charge_model_call() -> None:
    budget = _BUDGET.get()
    if budget is not None:
        budget.charge_model_call()


def charge_retry(extra_attempts: int = 1) -> None:
    budget = _BUDGET.get()
    if budget is not None:
        budget.charge_retry(extra_attempts)


def stage_timeout_seconds(default: float | None = None) -> float:
    configured = float(getattr(settings, "EVENT_PIPELINE_STAGE_TIMEOUT_SECONDS", 240) or 240)
    budget = _BUDGET.get()
    if budget is None:
        return default or configured
    return min(default or configured, budget.remaining_seconds())
