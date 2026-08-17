import asyncio
import mimetypes
import random
from pathlib import Path

import httpx
from apps.api_gateway.config.setting import settings
from services.speech.errors import STTProviderAuthError


_client: httpx.AsyncClient | None = None
_semaphore = asyncio.Semaphore(settings.SARVAM_MAX_CONCURRENCY)
_AUDIO_EXTENSIONS_BY_TYPE = {
    "audio/aac": ".aac",
    "audio/aiff": ".aiff",
    "audio/amr": ".amr",
    "audio/flac": ".flac",
    "audio/m4a": ".m4a",
    "audio/mp4": ".m4a",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/ogg": ".ogg",
    "audio/opus": ".opus",
    "audio/wav": ".wav",
    "audio/wave": ".wav",
    "audio/webm": ".webm",
    "audio/x-m4a": ".m4a",
    "audio/x-wav": ".wav",
    "video/mp4": ".m4a",
    "video/webm": ".webm",
}


def get_sarvam_stt_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        timeout = httpx.Timeout(
            connect=10,
            read=settings.SARVAM_TIMEOUT_SECONDS,
            write=settings.SARVAM_TIMEOUT_SECONDS,
            pool=settings.SARVAM_TIMEOUT_SECONDS,
        )
        _client = httpx.AsyncClient(timeout=timeout)
    return _client


async def sarvam_transcribe_from_path(
    file_path: str,
    filename: str,
    content_type: str,
):
    if not settings.STT_ALLOW_SARVAM_FALLBACK:
        raise STTProviderAuthError(
            "Sarvam speech-to-text fallback is disabled by STT_ALLOW_SARVAM_FALLBACK=false",
            provider="sarvam",
        )

    url = f"{_sarvam_speech_base_url()}/speech-to-text"
    path = Path(file_path)
    if not path.exists() or path.stat().st_size <= 0:
        raise ValueError("Speech audio file is missing or empty")
    normalized_content_type = _normalize_audio_content_type(content_type, filename)
    upload_filename = _audio_filename_for_upload(filename, normalized_content_type)

    headers = {
        "api-subscription-key": settings.secret_value(settings.SARVAM_API_KEY),
    }

    data = {
        "model": "saaras:v3",
        "mode": "transcribe",
        "language_code": "unknown",
    }

    with open(file_path, "rb") as audio_file:
        files = {
            "file": (
                upload_filename,
                audio_file,
                normalized_content_type,
            )
        }

        async with _semaphore:
            response = await _post_with_retries(url, headers, data, files)

    if response.status_code >= 400:
        if _is_permanent_audio_error(response):
            message = _sarvam_error_message(response)
            print("Sarvam permanent rejection:", message)
            raise ValueError(message)
        print("Sarvam status:", response.status_code)
        print("Sarvam error:", response.text)

    response.raise_for_status()
    result = response.json()
    transcript = str(result.get("transcript") or "").strip()
    result["transcript"] = transcript
    result["provider"] = "sarvam"
    result["model"] = "saaras:v3"
    result["is_empty_transcript"] = not bool(transcript)
    return result


async def _post_with_retries(url: str, headers: dict, data: dict, files: dict) -> httpx.Response:
    last_error: Exception | None = None
    for attempt in range(settings.SARVAM_MAX_RETRIES + 1):
        try:
            _rewind_files(files)
            response = await get_sarvam_stt_client().post(
                url,
                headers=headers,
                data=data,
                files=files,
            )
            if response.status_code < 400:
                return response
            if response.status_code not in {408, 409, 425, 429, 500, 502, 503, 504}:
                return response
            retry_after = response.headers.get("retry-after")
            await asyncio.sleep(_retry_delay(attempt, retry_after))
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.PoolTimeout) as error:
            last_error = error
            await asyncio.sleep(_retry_delay(attempt, None))
    if last_error:
        raise last_error
    return response


def _retry_delay(attempt: int, retry_after: str | None) -> float:
    if retry_after:
        try:
            return min(float(retry_after), 60)
        except ValueError:
            pass
    return min(60, (2**attempt) + random.uniform(0, 0.5))


def _normalize_audio_content_type(content_type: str | None, filename: str | None) -> str:
    value = str(content_type or "").split(";", 1)[0].strip().lower()
    if value in _AUDIO_EXTENSIONS_BY_TYPE:
        return value
    guessed, _ = mimetypes.guess_type(filename or "")
    guessed = str(guessed or "").split(";", 1)[0].strip().lower()
    if guessed in _AUDIO_EXTENSIONS_BY_TYPE:
        return guessed
    return "audio/wav"


def _audio_filename_for_upload(filename: str | None, content_type: str) -> str:
    name = Path(filename or "audio").name.strip() or "audio"
    suffix = Path(name).suffix.lower()
    if suffix:
        return name
    return f"{name}{_AUDIO_EXTENSIONS_BY_TYPE.get(content_type, '.wav')}"


def _rewind_files(files: dict) -> None:
    for file_value in files.values():
        if not isinstance(file_value, tuple) or len(file_value) < 2:
            continue
        stream = file_value[1]
        try:
            stream.seek(0)
        except Exception:
            pass


def _is_permanent_audio_error(response: httpx.Response) -> bool:
    if response.status_code not in {400, 413, 422}:
        return False
    message = _sarvam_error_message(response).lower()
    permanent_markers = (
        "failed to read the file",
        "audio format",
        "audio duration exceeds",
        "exceeds the maximum limit",
        "invalid audio",
        "file too large",
        "over 30 seconds",
        "30 seconds",
        "batch api",
        "unprocessable",
    )
    return any(marker in message for marker in permanent_markers)


def _sarvam_error_message(response: httpx.Response) -> str:
    try:
        error = response.json().get("error") or {}
        message = error.get("message") or response.text
    except Exception:
        message = response.text
    return f"Sarvam speech-to-text rejected audio: {message}"


def _sarvam_speech_base_url() -> str:
    base_url = settings.SARVAM_SPEECH_BASE_URL.rstrip("/")
    if base_url.endswith("/v1"):
        return base_url[:-3]
    return base_url
