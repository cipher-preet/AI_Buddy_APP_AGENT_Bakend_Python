from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

from services.llm.errors import LLMProviderError
from services.llm.schema_adapter import QUOTA_UNAVAILABLE


@dataclass
class ProviderQuota:
    rpm: int | None = None
    rpd: int | None = None
    tpm: int | None = None
    tpd: int | None = None


@dataclass
class _QuotaWindow:
    minute_requests: deque[datetime] = field(default_factory=deque)
    minute_tokens: deque[tuple[datetime, int]] = field(default_factory=deque)
    day: date = field(default_factory=lambda: datetime.now(timezone.utc).date())
    day_requests: int = 0
    day_tokens: int = 0


class InMemoryQuotaGuard:
    def __init__(self):
        self._windows: dict[str, _QuotaWindow] = {}

    def reserve(self, key: str, quota: ProviderQuota | None, estimated_tokens: int) -> None:
        if quota is None:
            return
        now = datetime.now(timezone.utc)
        window = self._windows.setdefault(key, _QuotaWindow())
        self._reset_if_new_day(window, now)
        self._trim_minute(window, now)

        if quota.rpm is not None and len(window.minute_requests) >= quota.rpm:
            raise LLMProviderError(
                f"{key} quota guard blocked request: RPM limit reached",
                retryable=True,
                status_code=429,
                failure_reason=QUOTA_UNAVAILABLE,
            )
        if quota.rpd is not None and window.day_requests >= quota.rpd:
            raise LLMProviderError(
                f"{key} quota guard blocked request: RPD limit reached",
                retryable=True,
                status_code=429,
                failure_reason=QUOTA_UNAVAILABLE,
            )
        if quota.tpm is not None and self._minute_token_total(window) + estimated_tokens > quota.tpm:
            raise LLMProviderError(
                f"{key} quota guard blocked request: TPM limit reached",
                retryable=True,
                status_code=429,
                failure_reason=QUOTA_UNAVAILABLE,
            )
        if quota.tpd is not None and window.day_tokens + estimated_tokens > quota.tpd:
            raise LLMProviderError(
                f"{key} quota guard blocked request: TPD limit reached",
                retryable=True,
                status_code=429,
                failure_reason=QUOTA_UNAVAILABLE,
            )

        window.minute_requests.append(now)
        window.minute_tokens.append((now, estimated_tokens))
        window.day_requests += 1
        window.day_tokens += estimated_tokens

    def record_actual_tokens(self, key: str, quota: ProviderQuota | None, estimated_tokens: int, actual_tokens: int) -> None:
        if quota is None or quota.tpd is None:
            return
        delta = max(0, actual_tokens - estimated_tokens)
        if not delta:
            return
        window = self._windows.setdefault(key, _QuotaWindow())
        window.day_tokens += delta

    def _reset_if_new_day(self, window: _QuotaWindow, now: datetime) -> None:
        today = now.date()
        if window.day == today:
            return
        window.day = today
        window.day_requests = 0
        window.day_tokens = 0
        window.minute_requests.clear()
        window.minute_tokens.clear()

    def _trim_minute(self, window: _QuotaWindow, now: datetime) -> None:
        cutoff = now.timestamp() - 60
        while window.minute_requests and window.minute_requests[0].timestamp() < cutoff:
            window.minute_requests.popleft()
        while window.minute_tokens and window.minute_tokens[0][0].timestamp() < cutoff:
            window.minute_tokens.popleft()

    def _minute_token_total(self, window: _QuotaWindow) -> int:
        return sum(tokens for _, tokens in window.minute_tokens)

    def reset(self) -> None:
        self._windows.clear()


quota_guard = InMemoryQuotaGuard()
