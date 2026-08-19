from __future__ import annotations

import time
from typing import Awaitable, Callable

from apps.api_gateway.config.setting import settings
from services.speech.errors import STTPermanentAudioError, STTProviderError, STTProviderTemporaryError
from services.speech.providers.deepgram_provider import deepgram_transcribe_from_path
from services.speech.providers.sarvam_provider import sarvam_transcribe_from_path


Transcriber = Callable[[str, str, str], Awaitable[dict]]

_PROVIDERS: dict[str, Transcriber] = {
    "deepgram": deepgram_transcribe_from_path,
    "sarvam": sarvam_transcribe_from_path,
}


async def transcribe_from_path_with_fallback(
    file_path: str,
    filename: str,
    content_type: str,
    keyterms: list[str] | None = None,
    context: dict | None = None,
) -> dict:
    attempts: list[dict] = []
    last_error: Exception | None = None
    provider_order = settings.stt_provider_order_list

    print(
        "STT provider routing started:",
        {
            "provider_order": provider_order,
            "filename": filename,
            "content_type": content_type,
        },
    )

    for provider_name in provider_order:
        transcriber = _PROVIDERS.get(provider_name)
        if transcriber is None:
            print(
                "STT provider skipped:",
                {
                    "provider": provider_name,
                    "reason": "unknown_provider",
                },
            )
            attempts.append(
                {
                    "provider": provider_name,
                    "status": "skipped",
                    "error": "Unknown STT provider",
                }
            )
            continue

        for retry_index in range(_max_attempts()):
            started = time.perf_counter()
            try:
                print(
                    "STT provider attempt started:",
                    {
                        "provider": provider_name,
                        "attempt": retry_index + 1,
                        "max_attempts": _max_attempts(),
                    },
                )
                transcribe_kwargs: dict = {
                    "file_path": file_path,
                    "filename": filename,
                    "content_type": content_type,
                }
                if provider_name == "deepgram":
                    if keyterms:
                        transcribe_kwargs["keyterms"] = keyterms
                    if context:
                        transcribe_kwargs["context"] = context
                result = await transcriber(**transcribe_kwargs)
                result = _clean_result(result)
                result["provider"] = str(result.get("provider") or provider_name)
                result["fallback_attempts"] = attempts
                attempts.append(
                    {
                        "provider": provider_name,
                        "status": "completed",
                        "duration_ms": _elapsed_ms(started),
                    }
                )
                _log_attempt(provider_name, "completed", attempts[-1])
                return result
            except STTPermanentAudioError as error:
                attempts.append(_attempt_error(provider_name, "permanent_audio_error", error, started))
                _log_attempt(provider_name, "permanent_audio_error", attempts[-1])
                raise ValueError(str(error)) from error
            except STTProviderTemporaryError as error:
                last_error = error
                status = "retry" if _should_retry(error, retry_index) else "fallback"
                attempts.append(_attempt_error(provider_name, status, error, started))
                _log_attempt(provider_name, status, attempts[-1])
                if status == "retry":
                    continue
                break
            except STTProviderError as error:
                last_error = error
                attempts.append(_attempt_error(provider_name, "fallback", error, started))
                _log_attempt(provider_name, "fallback", attempts[-1])
                break
            except ValueError:
                raise
            except Exception as error:
                last_error = error
                wrapped = STTProviderTemporaryError(
                    f"{provider_name} speech-to-text failed: {error}",
                    provider=provider_name,
                )
                status = "retry" if _should_retry(wrapped, retry_index) else "fallback"
                attempts.append(_attempt_error(provider_name, status, wrapped, started))
                _log_attempt(provider_name, status, attempts[-1])
                if status == "retry":
                    continue
                break

    message = "All speech-to-text providers failed"
    if last_error:
        message = f"{message}: {last_error}"
    error = RuntimeError(message)
    setattr(error, "fallback_attempts", attempts)
    raise error


def _clean_result(result: dict) -> dict:
    cleaned = dict(result or {})
    transcript = str(cleaned.get("transcript") or "").strip()
    cleaned["transcript"] = transcript
    cleaned["is_empty_transcript"] = not bool(transcript)
    if cleaned.get("raw_provider_response") is None:
        cleaned.pop("raw_provider_response", None)
    return cleaned


def _max_attempts() -> int:
    return max(1, settings.STT_MAX_RETRIES + 1)


def _should_retry(error: Exception, retry_index: int) -> bool:
    if retry_index >= _max_attempts() - 1:
        return False
    message = str(error).lower()
    non_retryable_markers = (
        "api key is not configured",
        "sdk is not installed",
    )
    return not any(marker in message for marker in non_retryable_markers)


def _attempt_error(provider_name: str, status: str, error: Exception, started: float) -> dict:
    return {
        "provider": provider_name,
        "status": status,
        "error_type": type(error).__name__,
        "error": str(error),
        "duration_ms": _elapsed_ms(started),
    }


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _log_attempt(provider_name: str, status: str, data: dict) -> None:
    print(
        "STT provider attempt:",
        {
            "provider": provider_name,
            "status": status,
            "duration_ms": data.get("duration_ms"),
            "error_type": data.get("error_type"),
            "error": data.get("error"),
        },
    )
