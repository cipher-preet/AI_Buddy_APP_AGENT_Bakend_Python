import asyncio
import random

import httpx
from apps.api_gateway.config.setting import settings


_client: httpx.AsyncClient | None = None
_semaphore = asyncio.Semaphore(settings.SARVAM_MAX_CONCURRENCY)


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
    url = f"{_sarvam_speech_base_url()}/speech-to-text"

    headers = {
        "api-subscription-key": settings.SARVAM_API_KEY,
    }

    data = {
        "model": "saaras:v3",
        "mode": "transcribe",
        "language_code": "unknown",
    }

    with open(file_path, "rb") as audio_file:
        files = {
            "file": (
                filename,
                audio_file,
                content_type or "audio/wav",
            )
        }

        async with _semaphore:
            response = await _post_with_retries(url, headers, data, files)

    if response.status_code >= 400:
        print("Sarvam status:", response.status_code)
        print("Sarvam error:", response.text)

    response.raise_for_status()
    result = response.json()
    transcript = str(result.get("transcript") or "").strip()
    result["transcript"] = transcript
    result["provider"] = "sarvam"
    result["is_empty_transcript"] = not bool(transcript)
    return result


async def _post_with_retries(url: str, headers: dict, data: dict, files: dict) -> httpx.Response:
    last_error: Exception | None = None
    for attempt in range(settings.SARVAM_MAX_RETRIES + 1):
        try:
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


def _sarvam_speech_base_url() -> str:
    base_url = settings.SARVAM_SPEECH_BASE_URL.rstrip("/")
    if base_url.endswith("/v1"):
        return base_url[:-3]
    return base_url
