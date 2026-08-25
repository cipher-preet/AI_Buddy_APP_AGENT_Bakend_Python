from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from apps.api_gateway.config.setting import settings
from services.conversation.models import STTStatus
from services.speech.errors import (
    STTPermanentAudioError,
    STTProviderRateLimitError,
    STTProviderTemporaryError,
    is_permanent_audio_message,
)
from services.storage.s3_audio_storage import PermanentS3StorageError, TemporaryS3StorageError

TERMINAL_COMPLETED_NON_EMPTY = "COMPLETED_NON_EMPTY"
TERMINAL_COMPLETED_EMPTY = "COMPLETED_EMPTY"
TERMINAL_FAILED_PERMANENTLY = "FAILED_PERMANENTLY"

FAILURE_S3_OBJECT_MISSING = "S3_OBJECT_MISSING"
FAILURE_S3_PERMANENT = "S3_PERMANENT"
FAILURE_CORRUPT_AUDIO = "CORRUPT_AUDIO"
FAILURE_UNSUPPORTED_AUDIO = "UNSUPPORTED_AUDIO"
FAILURE_PERMANENT_AUDIO = "PERMANENT_AUDIO"
FAILURE_TIMEOUT = "TIMEOUT"
FAILURE_NETWORK = "NETWORK"
FAILURE_RATE_LIMIT = "RATE_LIMIT"
FAILURE_PROVIDER_5XX = "PROVIDER_5XX"
FAILURE_TRANSIENT_S3 = "TRANSIENT_S3"
FAILURE_RETRY_EXHAUSTED = "RETRY_EXHAUSTED"
FAILURE_UNKNOWN = "UNKNOWN"

STAGE_S3_DOWNLOAD = "s3_download"
STAGE_STT_PROVIDER = "stt_provider"
STAGE_LOCAL_AUDIO = "local_audio"
STAGE_QUEUE_DLQ = "queue_dlq"

_PERMANENT_TYPE_MARKERS = (
    "s3_object_missing",
    "s3_permanent",
    "corrupt_audio",
    "unsupported_audio",
    "permanent_audio",
    "retry_exhausted",
)
_S3_MISSING_MARKERS = (
    "nosuchkey",
    "no such key",
    "nosuchbucket",
    "no such bucket",
    "object does not exist",
    "object missing",
    "s3 permanent error: 404",
    "s3 permanent error: nosuchkey",
    "s3 permanent error: nosuchbucket",
)
_CORRUPT_MARKERS = ("corrupt", "corrupted")
_UNSUPPORTED_MARKERS = (
    "unsupported audio",
    "unsupported data",
    "unsupported file",
    "unsupported media",
    "unsupported format",
    "invalid audio",
    "audio format",
)
_TIMEOUT_MARKERS = ("timeout", "timed out", "deadline exceeded")
_NETWORK_MARKERS = (
    "connection reset",
    "connection error",
    "connection aborted",
    "network",
    "temporarily unavailable",
    "temporary failure",
)
_RATE_LIMIT_MARKERS = ("429", "too many requests", "rate limit")
_5XX_MARKERS = (" 500", " 502", " 503", " 504", "status 500", "status 502", "status 503", "status 504")


@dataclass(frozen=True)
class SttFailureClassification:
    permanent: bool
    failure_type: str
    failure_stage: str
    provider: str | None
    sanitized_message: str


def classify_stt_failure(
    error: Exception | str | None,
    *,
    retry_exhausted: bool = False,
    stage: str | None = None,
) -> SttFailureClassification:
    message = str(error or "")
    normalized = message.lower()
    provider = getattr(error, "provider", None) if isinstance(error, Exception) else None
    status_code = getattr(error, "status_code", None) if isinstance(error, Exception) else None
    resolved_stage = stage or _infer_stage(error, normalized)

    if _contains_any(normalized, _PERMANENT_TYPE_MARKERS):
        failure_type = _permanent_type_from_message(normalized)
        stage_name = STAGE_S3_DOWNLOAD if failure_type in {FAILURE_S3_OBJECT_MISSING, FAILURE_S3_PERMANENT} else resolved_stage or STAGE_STT_PROVIDER
        return _classification(True, failure_type, stage_name, provider, failure_type)

    if isinstance(error, PermanentS3StorageError) or _is_s3_missing_message(normalized):
        failure_type = FAILURE_S3_OBJECT_MISSING if _is_s3_missing_message(normalized) or _is_missing_status(status_code) else FAILURE_S3_PERMANENT
        if isinstance(error, PermanentS3StorageError) and not _is_s3_missing_message(normalized) and not _is_missing_status(status_code):
            failure_type = FAILURE_S3_PERMANENT
        return _classification(True, failure_type, resolved_stage or STAGE_S3_DOWNLOAD, provider, failure_type)

    if isinstance(error, STTPermanentAudioError) or is_permanent_audio_message(message) or _is_corrupt_or_unsupported(normalized):
        failure_type = _permanent_audio_type(normalized)
        return _classification(True, failure_type, resolved_stage or STAGE_STT_PROVIDER, provider, failure_type)

    if isinstance(error, ValueError) and not _looks_retryable(normalized):
        return _classification(True, FAILURE_PERMANENT_AUDIO, resolved_stage or STAGE_LOCAL_AUDIO, provider, FAILURE_PERMANENT_AUDIO)

    if retry_exhausted:
        return _classification(
            True,
            FAILURE_RETRY_EXHAUSTED,
            resolved_stage or STAGE_QUEUE_DLQ,
            provider,
            FAILURE_RETRY_EXHAUSTED,
        )

    if isinstance(error, TemporaryS3StorageError):
        return _classification(False, FAILURE_TRANSIENT_S3, resolved_stage or STAGE_S3_DOWNLOAD, provider, FAILURE_TRANSIENT_S3)
    if isinstance(error, STTProviderRateLimitError) or _contains_any(normalized, _RATE_LIMIT_MARKERS) or status_code == 429:
        return _classification(False, FAILURE_RATE_LIMIT, resolved_stage or STAGE_STT_PROVIDER, provider, FAILURE_RATE_LIMIT)
    if (status_code is not None and 500 <= int(status_code) <= 599) or _contains_any(normalized, _5XX_MARKERS):
        return _classification(False, FAILURE_PROVIDER_5XX, resolved_stage or STAGE_STT_PROVIDER, provider, FAILURE_PROVIDER_5XX)
    if _contains_any(normalized, _TIMEOUT_MARKERS):
        return _classification(False, FAILURE_TIMEOUT, resolved_stage or STAGE_STT_PROVIDER, provider, FAILURE_TIMEOUT)
    if isinstance(error, STTProviderTemporaryError) or _contains_any(normalized, _NETWORK_MARKERS):
        failure_type = FAILURE_NETWORK if _contains_any(normalized, _NETWORK_MARKERS) else FAILURE_PROVIDER_5XX
        return _classification(False, failure_type, resolved_stage or STAGE_STT_PROVIDER, provider, failure_type)

    return _classification(False, FAILURE_UNKNOWN, resolved_stage or STAGE_STT_PROVIDER, provider, FAILURE_UNKNOWN)


def is_permanent_stt_failure(error: Exception | str | None) -> bool:
    return classify_stt_failure(error).permanent


def is_terminal_failed_chunk(chunk: Any) -> bool:
    if getattr(chunk, "sttStatus", None) != STTStatus.FAILED:
        return False
    if bool(getattr(chunk, "terminal", False)):
        return True
    failure_type = str(getattr(chunk, "failureType", "") or "")
    if failure_type in {
        FAILURE_S3_OBJECT_MISSING,
        FAILURE_S3_PERMANENT,
        FAILURE_CORRUPT_AUDIO,
        FAILURE_UNSUPPORTED_AUDIO,
        FAILURE_PERMANENT_AUDIO,
        FAILURE_RETRY_EXHAUSTED,
    }:
        return True
    if is_permanent_stt_failure(getattr(chunk, "lastError", None)):
        return True
    return int(getattr(chunk, "sttAttempts", 0) or getattr(chunk, "retryCount", 0) or 0) >= settings.WORKER_MAX_RETRIES


def is_retrying_failed_chunk(chunk: Any) -> bool:
    return getattr(chunk, "sttStatus", None) == STTStatus.FAILED and not is_terminal_failed_chunk(chunk)


def log_stt_terminal_state(
    *,
    conversation_id: str,
    sequence_number: int,
    job_id: str | None,
    failure_type: str | None,
    retry_count: int,
    terminal_state: str = TERMINAL_FAILED_PERMANENTLY,
) -> None:
    print(
        "STT terminal state recorded:",
        {
            "conversationId": conversation_id,
            "sequenceNumber": sequence_number,
            "jobId": job_id,
            "terminalState": terminal_state,
            "failureType": failure_type,
            "retryCount": retry_count,
        },
    )


def _classification(
    permanent: bool,
    failure_type: str,
    failure_stage: str,
    provider: str | None,
    sanitized_message: str,
) -> SttFailureClassification:
    return SttFailureClassification(
        permanent=permanent,
        failure_type=failure_type,
        failure_stage=failure_stage,
        provider=str(provider) if provider else None,
        sanitized_message=sanitized_message,
    )


def _infer_stage(error: Exception | str | None, normalized: str) -> str | None:
    if isinstance(error, (PermanentS3StorageError, TemporaryS3StorageError)) or "s3" in normalized:
        return STAGE_S3_DOWNLOAD
    if isinstance(error, Exception) and "missing audio file reference" in normalized:
        return STAGE_LOCAL_AUDIO
    return None


def _is_s3_missing_message(normalized: str) -> bool:
    if _contains_any(normalized, _S3_MISSING_MARKERS):
        return True
    return "s3" in normalized and "404" in normalized


def _is_missing_status(status_code: int | None) -> bool:
    return status_code == 404


def _is_corrupt_or_unsupported(normalized: str) -> bool:
    return _contains_any(normalized, _CORRUPT_MARKERS) or _contains_any(normalized, _UNSUPPORTED_MARKERS)


def _permanent_type_from_message(normalized: str) -> str:
    if "s3_object_missing" in normalized:
        return FAILURE_S3_OBJECT_MISSING
    if "s3_permanent" in normalized:
        return FAILURE_S3_PERMANENT
    if "corrupt_audio" in normalized:
        return FAILURE_CORRUPT_AUDIO
    if "unsupported_audio" in normalized:
        return FAILURE_UNSUPPORTED_AUDIO
    if "retry_exhausted" in normalized:
        return FAILURE_RETRY_EXHAUSTED
    return FAILURE_PERMANENT_AUDIO


def _permanent_audio_type(normalized: str) -> str:
    if _contains_any(normalized, _CORRUPT_MARKERS):
        return FAILURE_CORRUPT_AUDIO
    if _contains_any(normalized, _UNSUPPORTED_MARKERS):
        return FAILURE_UNSUPPORTED_AUDIO
    return FAILURE_PERMANENT_AUDIO


def _looks_retryable(normalized: str) -> bool:
    return (
        _contains_any(normalized, _TIMEOUT_MARKERS)
        or _contains_any(normalized, _NETWORK_MARKERS)
        or _contains_any(normalized, _RATE_LIMIT_MARKERS)
        or _contains_any(normalized, _5XX_MARKERS)
    )


def _contains_any(normalized: str, markers: tuple[str, ...]) -> bool:
    return any(marker in normalized for marker in markers)
