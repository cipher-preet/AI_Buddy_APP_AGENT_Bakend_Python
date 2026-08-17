from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

import httpx

from apps.api_gateway.config.setting import settings
from services.speech.errors import (
    STTPermanentAudioError,
    STTProviderAuthError,
    STTProviderBillingError,
    STTProviderRateLimitError,
    STTProviderTemporaryError,
    is_permanent_audio_message,
)
from services.speech.providers.sarvam_provider import _normalize_audio_content_type


async def deepgram_transcribe_from_path(
    file_path: str,
    filename: str,
    content_type: str,
) -> dict[str, Any]:
    provider = "deepgram"
    path = Path(file_path)
    if not path.exists() or path.stat().st_size <= 0:
        raise STTPermanentAudioError("Speech audio file is missing or empty", provider=provider)

    api_key = settings.secret_value(settings.DEEPGRAM_API_KEY)
    if not api_key:
        raise STTProviderAuthError("Deepgram API key is not configured", provider=provider)

    mimetype = _normalize_audio_content_type(content_type, filename)
    language = settings.DEEPGRAM_LANGUAGE.strip()

    try:
        from deepgram import DeepgramClient, FileSource, PrerecordedOptions
    except ImportError:
        print("Deepgram SDK unavailable; using HTTP transcription fallback.")
        return await _deepgram_transcribe_with_http(path, mimetype, language, api_key)

    source: FileSource = {
        "buffer": path.read_bytes(),
        "mimetype": mimetype,
    }
    options_kwargs: dict[str, Any] = {
        "model": settings.DEEPGRAM_MODEL,
        "smart_format": settings.DEEPGRAM_SMART_FORMAT,
    }
    if language:
        options_kwargs["language"] = language
    if settings.DEEPGRAM_DETECT_LANGUAGE:
        options_kwargs["detect_language"] = True

    options = PrerecordedOptions(**options_kwargs)

    try:
        client = DeepgramClient(api_key)
        response = await asyncio.wait_for(
            asyncio.to_thread(
                client.listen.rest.v("1").transcribe_file,
                source,
                options,
                timeout=settings.STT_TIMEOUT_SECONDS,
            ),
            timeout=settings.STT_TIMEOUT_SECONDS + 5,
        )
    except Exception as error:
        raise _classify_deepgram_error(error) from error

    result = _to_mapping(response)
    transcript = _extract_transcript(result)
    return {
        "transcript": transcript,
        "provider": provider,
        "model": settings.DEEPGRAM_MODEL,
        "language_code": _extract_language(result) or language or None,
        "request_id": _extract_request_id(result),
        "is_empty_transcript": not bool(transcript),
        "raw_provider_response": result if settings.ENABLE_TRANSCRIPT_DEBUG_LOGS else None,
    }


async def _deepgram_transcribe_with_http(
    path: Path,
    mimetype: str,
    language: str,
    api_key: str,
) -> dict[str, Any]:
    params: dict[str, str | bool] = {
        "model": settings.DEEPGRAM_MODEL,
        "smart_format": settings.DEEPGRAM_SMART_FORMAT,
    }
    if language:
        params["language"] = language
    if settings.DEEPGRAM_DETECT_LANGUAGE:
        params["detect_language"] = True

    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": mimetype,
    }
    try:
        async with httpx.AsyncClient(timeout=settings.STT_TIMEOUT_SECONDS) as client:
            response = await client.post(
                "https://api.deepgram.com/v1/listen",
                params=params,
                headers=headers,
                content=path.read_bytes(),
            )
    except Exception as error:
        raise _classify_deepgram_error(error) from error

    if response.status_code >= 400:
        raise _classify_deepgram_error(
            RuntimeError(f"HTTP {response.status_code}: {response.text}")
        )

    result = response.json()
    transcript = _extract_transcript(result)
    return {
        "transcript": transcript,
        "provider": "deepgram",
        "model": settings.DEEPGRAM_MODEL,
        "language_code": _extract_language(result) or language or None,
        "request_id": _extract_request_id(result),
        "is_empty_transcript": not bool(transcript),
        "raw_provider_response": result if settings.ENABLE_TRANSCRIPT_DEBUG_LOGS else None,
    }


def _to_mapping(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    if hasattr(response, "to_dict"):
        return response.to_dict()
    if hasattr(response, "model_dump"):
        return response.model_dump()
    return {}


def _extract_transcript(result: dict[str, Any]) -> str:
    channels = ((result.get("results") or {}).get("channels") or [])
    if not channels:
        return ""
    alternatives = (channels[0] or {}).get("alternatives") or []
    if not alternatives:
        return ""
    return str((alternatives[0] or {}).get("transcript") or "").strip()


def _extract_language(result: dict[str, Any]) -> str | None:
    channels = ((result.get("results") or {}).get("channels") or [])
    if channels:
        language = (channels[0] or {}).get("detected_language")
        if language:
            return str(language)
    metadata = result.get("metadata") or {}
    language = metadata.get("detected_language") or metadata.get("language")
    return str(language) if language else None


def _extract_request_id(result: dict[str, Any]) -> str | None:
    metadata = result.get("metadata") or {}
    request_id = metadata.get("request_id") or metadata.get("transaction_key")
    return str(request_id) if request_id else None


def _classify_deepgram_error(error: Exception) -> STTProviderTemporaryError:
    status_code = _status_code(error)
    message = f"Deepgram speech-to-text failed: {error}"
    lowered = message.lower()
    if is_permanent_audio_message(message) or status_code in {413, 415, 422}:
        return STTPermanentAudioError(message, provider="deepgram", status_code=status_code)
    if status_code in {401, 403} or "unauthorized" in lowered or "forbidden" in lowered:
        return STTProviderAuthError(message, provider="deepgram", status_code=status_code)
    if status_code == 402 or "billing" in lowered or "payment" in lowered or "credit" in lowered:
        return STTProviderBillingError(message, provider="deepgram", status_code=status_code)
    if status_code == 429 or "rate limit" in lowered or "too many requests" in lowered:
        return STTProviderRateLimitError(message, provider="deepgram", status_code=status_code)
    return STTProviderTemporaryError(message, provider="deepgram", status_code=status_code)


def _status_code(error: Exception) -> int | None:
    for attr in ("status_code", "status"):
        value = getattr(error, attr, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    response = getattr(error, "response", None)
    value = getattr(response, "status_code", None) if response is not None else None
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    match = re.search(r"\bHTTP\s+(\d{3})\b", str(error))
    if match:
        return int(match.group(1))
    return None
