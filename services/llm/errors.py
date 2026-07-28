class LLMProviderError(Exception):
    def __init__(self, message: str, retryable: bool = False, status_code: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


def is_retryable_status(status_code: int | None) -> bool:
    return status_code in {408, 409, 425, 429, 500, 502, 503, 504}
