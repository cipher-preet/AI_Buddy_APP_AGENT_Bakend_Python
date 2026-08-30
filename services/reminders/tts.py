"""TTS for reminder voice: Deepgram for English, Sarvam for Hindi."""

from __future__ import annotations

import asyncio
import base64
from typing import Any

import httpx

from apps.api_gateway.config.setting import settings
from services.reminders.language import is_hindi_text

_CACHE: dict[str, dict[str, str]] = {}
_LOCK = asyncio.Lock()
_DEEPGRAM_SPEAK_URL = "https://api.deepgram.com/v1/speak"


async def synthesize_speech(text: str) -> dict[str, str] | None:
    cleaned = " ".join((text or "").strip().split())
    if not cleaned:
        return None

    cached = _CACHE.get(cleaned)
    if cached:
        return cached

    async with _LOCK:
        cached = _CACHE.get(cleaned)
        if cached:
            return cached
        audio = await _speak(cleaned)
        if audio:
            _CACHE[cleaned] = audio
        return audio


async def _speak(text: str) -> dict[str, str] | None:
    if is_hindi_text(text):
        spoken = await _sarvam_speak(text)
        if spoken:
            return spoken
    return await _deepgram_speak(text)


async def _deepgram_speak(text: str) -> dict[str, str] | None:
    api_key = settings.secret_value(settings.DEEPGRAM_API_KEY)
    if not api_key:
        return None

    model = (settings.DEEPGRAM_TTS_MODEL or "aura-asteria-en").strip()
    timeout = httpx.Timeout(
        connect=5,
        read=max(8, settings.DEEPGRAM_TTS_TIMEOUT_SECONDS),
        write=5,
        pool=5,
    )
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                _DEEPGRAM_SPEAK_URL,
                params={"model": model},
                headers={
                    "Authorization": f"Token {api_key}",
                    "Content-Type": "application/json",
                },
                json={"text": text},
            )
    except Exception as error:
        print("Deepgram TTS failed:", {"error": str(error)[:300]})
        return None

    if response.status_code >= 400:
        print(
            "Deepgram TTS rejected:",
            {
                "status": response.status_code,
                "body": response.text[:300],
                "model": model,
            },
        )
        return None

    content_type = response.headers.get("content-type") or "audio/mpeg"
    return {
        "audioBase64": base64.b64encode(response.content).decode("ascii"),
        "contentType": content_type.split(";")[0].strip() or "audio/mpeg",
    }


async def _sarvam_speak(text: str) -> dict[str, str] | None:
    api_key = settings.secret_value(settings.SARVAM_API_KEY)
    if not api_key:
        return None

    base_url = settings.SARVAM_SPEECH_BASE_URL.rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    timeout = httpx.Timeout(connect=5, read=20, write=5, pool=5)
    payload: dict[str, Any] = {
        "inputs": [text[:500]],
        "target_language_code": "hi-IN",
        "speaker": getattr(settings, "SARVAM_TTS_SPEAKER", "meera") or "meera",
        "model": getattr(settings, "SARVAM_TTS_MODEL", "bulbul:v2") or "bulbul:v2",
        "enable_preprocessing": True,
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{base_url}/text-to-speech",
                headers={"api-subscription-key": api_key},
                json=payload,
            )
    except Exception as error:
        print("Sarvam TTS failed:", {"error": str(error)[:300]})
        return None

    if response.status_code >= 400:
        print(
            "Sarvam TTS rejected:",
            {"status": response.status_code, "body": response.text[:300]},
        )
        return None

    try:
        data = response.json()
    except Exception:
        return None

    audios = data.get("audios") if isinstance(data, dict) else None
    if not isinstance(audios, list) or not audios:
        return None
    audio_b64 = audios[0]
    if not isinstance(audio_b64, str) or not audio_b64:
        return None
    return {
        "audioBase64": audio_b64,
        "contentType": "audio/wav",
    }
