from __future__ import annotations


class STTProviderError(Exception):
    def __init__(
        self,
        message: str,
        *,
        provider: str,
        status_code: int | None = None,
        retryable: bool = True,
    ):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.retryable = retryable


class STTPermanentAudioError(STTProviderError):
    def __init__(self, message: str, *, provider: str, status_code: int | None = None):
        super().__init__(message, provider=provider, status_code=status_code, retryable=False)


class STTProviderAuthError(STTProviderError):
    pass


class STTProviderBillingError(STTProviderError):
    pass


class STTProviderRateLimitError(STTProviderError):
    pass


class STTProviderTemporaryError(STTProviderError):
    pass


def is_permanent_audio_message(message: str) -> bool:
    normalized = str(message or "").lower()
    permanent_markers = (
        "audio format",
        "audio duration exceeds",
        "corrupt",
        "corrupted",
        "empty audio",
        "failed to read the file",
        "file is empty",
        "file too large",
        "invalid audio",
        "missing or empty",
        "no audio",
        "over 30 seconds",
        "unsupported file",
        "unsupported media",
        "unsupported audio",
        "unsupported data",
        "unprocessable",
    )
    return any(marker in normalized for marker in permanent_markers)
