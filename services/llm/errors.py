class LLMProviderError(Exception):
    def __init__(
        self,
        message: str,
        retryable: bool = False,
        status_code: int | None = None,
        failure_reason: str | None = None,
        provider_outage: bool = False,
    ):
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code
        self.failure_reason = failure_reason
        self.provider_outage = provider_outage


class AsyncLifecycleError(LLMProviderError):
    def __init__(self, message: str = "Event loop is closed"):
        super().__init__(message, retryable=False, status_code=None, failure_reason="ASYNC_LIFECYCLE_ERROR")


class StructuredOutputError(LLMProviderError):
    def __init__(self, outcome: str, message: str | None = None):
        super().__init__(message or outcome, retryable=True, status_code=422, failure_reason=outcome)
        self.outcome = outcome


def is_retryable_status(status_code: int | None) -> bool:
    return status_code in {408, 409, 425, 429, 500, 502, 503, 504}


def provider_pressure(error: BaseException | None) -> bool:
    """The provider refused this call. Another attempt on the same model will not finish the job."""
    if error is None:
        return False
    if getattr(error, "provider_outage", False):
        return True
    status = getattr(error, "status_code", None)
    if status in {429, 500, 502, 503, 504}:
        return True
    text = str(error).casefold()
    return any(
        token in text
        for token in (
            "not enough capacity",
            "backend_out_of_capacity",
            "db_unavailable",
            "verifying your key",
            "too many requests",
            "error 429",
            "error 500",
            "error 502",
            "error 503",
            "error 504",
            "all structured llm fallbacks failed",
            "all llm fallbacks failed",
        )
    )
