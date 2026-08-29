class LLMProviderError(Exception):
    def __init__(
        self,
        message: str,
        retryable: bool = False,
        status_code: int | None = None,
        failure_reason: str | None = None,
    ):
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code
        self.failure_reason = failure_reason


class AsyncLifecycleError(LLMProviderError):
    def __init__(self, message: str = "Event loop is closed"):
        super().__init__(message, retryable=False, status_code=None, failure_reason="ASYNC_LIFECYCLE_ERROR")


class StructuredOutputError(LLMProviderError):
    def __init__(self, outcome: str, message: str | None = None):
        super().__init__(message or outcome, retryable=True, status_code=422, failure_reason=outcome)
        self.outcome = outcome


def is_retryable_status(status_code: int | None) -> bool:
    return status_code in {408, 409, 425, 429, 500, 502, 503, 504}
