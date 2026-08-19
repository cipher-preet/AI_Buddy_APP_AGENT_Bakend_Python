from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

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


_DIARIZE_MODEL = "latest"
_UTTERANCES = True
_UTT_SPLIT = 0.8
_MAX_KEYTERMS = 100
_MAX_KEYTERM_TOKENS = 500
_MAX_WORDS_PER_KEYTERM = 8
_MAX_ERROR_BODY_CHARS = 500
_UNCERTAIN_MEAN_CONFIDENCE = 0.6
_UNCERTAIN_WORD_CONFIDENCE = 0.5
_UNCERTAIN_LOW_WORD_RATIO = 0.3
_SDK_V3: tuple[Any, Any, Any] | None | bool = False
_LOGGED_HTTP_FALLBACK = False


async def deepgram_transcribe_from_path(
    file_path: str,
    filename: str,
    content_type: str,
    keyterms: Sequence[str] | None = None,
    context: Mapping[str, Any] | None = None,
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
    resolved_keyterms = resolve_deepgram_keyterms(keyterms, context)
    sdk = _sdk_v3()
    if sdk is None:
        return await _deepgram_transcribe_with_http(
            path, mimetype, language, api_key, resolved_keyterms
        )

    DeepgramClient, FileSource, PrerecordedOptions = sdk
    source: FileSource = {
        "buffer": path.read_bytes(),
        "mimetype": mimetype,
    }
    options = _prerecorded_options(PrerecordedOptions, language, resolved_keyterms)

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

    return _finalize_transcription_result(_to_mapping(response), language)


async def _deepgram_transcribe_with_http(
    path: Path,
    mimetype: str,
    language: str,
    api_key: str,
    keyterms: Sequence[str] | None = None,
) -> dict[str, Any]:
    params = build_deepgram_listen_params(language, keyterms)
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
            RuntimeError(f"HTTP {response.status_code}: {_trim_error_body(response.text)}")
        )

    try:
        payload = response.json()
    except ValueError as error:
        raise _classify_deepgram_error(error) from error
    if not isinstance(payload, dict):
        raise _classify_deepgram_error(
            RuntimeError("Deepgram returned a non-object transcription payload")
        )
    return _finalize_transcription_result(payload, language)


def build_deepgram_listen_params(
    language: str,
    keyterms: Sequence[str] | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model": settings.DEEPGRAM_MODEL,
        "smart_format": settings.DEEPGRAM_SMART_FORMAT,
        "detect_language": bool(settings.DEEPGRAM_DETECT_LANGUAGE),
        "diarize_model": _DIARIZE_MODEL,
        "utterances": _UTTERANCES,
        "utt_split": _UTT_SPLIT,
    }
    if language:
        params["language"] = language
    normalized_keyterms = normalize_deepgram_keyterms(keyterms)
    if normalized_keyterms:
        params["keyterm"] = normalized_keyterms
    return params


def resolve_deepgram_keyterms(
    keyterms: Sequence[str] | None = None,
    context: Mapping[str, Any] | None = None,
) -> list[str]:
    values: list[Any] = []
    _extend_keyterm_values(values, keyterms)
    if isinstance(context, Mapping):
        for field in ("keyterms", "keyterm", "space_keyterms", "terminology", "terms"):
            _extend_keyterm_values(values, context.get(field))
    return normalize_deepgram_keyterms(values)


def normalize_deepgram_keyterms(keyterms: Sequence[Any] | None) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    token_count = 0
    for raw in keyterms or []:
        if isinstance(raw, (dict, bool, bytes, bytearray)) or raw is None:
            continue
        if isinstance(raw, (list, tuple, set)):
            continue
        term = " ".join(str(raw).split())
        if not term:
            continue
        key = term.casefold()
        if key in seen:
            continue
        tokens = max(1, len(term.split()))
        if tokens > _MAX_WORDS_PER_KEYTERM:
            continue
        if len(normalized) >= _MAX_KEYTERMS or token_count + tokens > _MAX_KEYTERM_TOKENS:
            break
        seen.add(key)
        normalized.append(term)
        token_count += tokens
    return normalized


def _extend_keyterm_values(values: list[Any], extra: Any) -> None:
    if extra is None or extra is False:
        return
    if isinstance(extra, (list, tuple, set)):
        values.extend(extra)
        return
    if isinstance(extra, dict):
        return
    values.append(extra)


def _sdk_v3() -> tuple[Any, Any, Any] | None:
    global _SDK_V3, _LOGGED_HTTP_FALLBACK
    if _SDK_V3 is False:
        try:
            from deepgram import DeepgramClient, FileSource, PrerecordedOptions

            _SDK_V3 = (DeepgramClient, FileSource, PrerecordedOptions)
        except ImportError:
            _SDK_V3 = None
            if not _LOGGED_HTTP_FALLBACK:
                print("Deepgram SDK v3 options unavailable; using HTTP transcription.")
                _LOGGED_HTTP_FALLBACK = True
    return None if _SDK_V3 is False else _SDK_V3


def _prerecorded_options(options_cls: Any, language: str, keyterms: Sequence[str] | None) -> Any:
    options_kwargs = build_deepgram_listen_params(language, keyterms)
    try:
        return options_cls(**options_kwargs)
    except TypeError:
        fields = getattr(options_cls, "model_fields", None) or getattr(options_cls, "__fields__", {})
        supported = {key: value for key, value in options_kwargs.items() if key in fields}
        if "diarize_model" not in supported and "diarize" in fields:
            supported["diarize"] = True
        return options_cls(**supported)


def _finalize_transcription_result(result: Mapping[str, Any] | None, language: str) -> dict[str, Any]:
    payload = result if isinstance(result, Mapping) else {}
    transcript = _extract_transcript(payload)
    quality = assess_transcript_quality(payload)
    if quality.get("uncertain"):
        print(
            "Deepgram transcript quality uncertain:",
            {
                "mean_confidence": quality.get("mean_confidence"),
                "min_confidence": quality.get("min_confidence"),
                "word_count": quality.get("word_count"),
                "low_confidence_word_count": quality.get("low_confidence_word_count"),
                "utterance_count": quality.get("utterance_count"),
            },
        )
    elif settings.ENABLE_TRANSCRIPT_DEBUG_LOGS:
        print("Deepgram transcript quality:", quality)

    return {
        "transcript": transcript,
        "provider": "deepgram",
        "model": settings.DEEPGRAM_MODEL,
        "language_code": _extract_language(payload) or language or None,
        "request_id": _extract_request_id(payload),
        "is_empty_transcript": not bool(transcript),
        "is_uncertain_transcript": bool(quality.get("uncertain")),
        "transcript_quality": quality,
        "raw_provider_response": dict(payload) if settings.ENABLE_TRANSCRIPT_DEBUG_LOGS else None,
    }


def _to_mapping(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    if hasattr(response, "to_dict"):
        mapped = response.to_dict()
        return mapped if isinstance(mapped, dict) else {}
    if hasattr(response, "model_dump"):
        mapped = response.model_dump()
        return mapped if isinstance(mapped, dict) else {}
    return {}


def _extract_transcript(result: Mapping[str, Any]) -> str:
    reconstructed = reconstruct_speaker_transcript(result)
    if reconstructed:
        return reconstructed
    return _alternative_transcript(result)


def reconstruct_speaker_transcript(result: Mapping[str, Any] | None) -> str:
    utterances = _utterances(result)
    lines: list[str] = []
    for utterance in utterances:
        text = str(utterance.get("transcript") or "").strip()
        if not text:
            continue
        lines.append(_format_speaker_line(utterance.get("speaker"), text))
    if lines:
        return "\n".join(lines)

    words = _words(result)
    if any(word.get("speaker") is not None for word in words):
        return _reconstruct_from_words(words)
    return ""


def assess_transcript_quality(result: Mapping[str, Any] | None) -> dict[str, Any]:
    utterances = _utterances(result)
    words = _all_words(result)
    word_confidences = _numeric_values(words, "confidence")
    speaker_confidences = _numeric_values(words, "speaker_confidence")
    utterance_confidences = _numeric_values(utterances, "confidence")
    confidences = word_confidences or utterance_confidences
    speakers: list[int] = []
    seen_speakers: set[int] = set()
    for item in (*utterances, *words):
        speaker = _speaker_number(item.get("speaker"))
        if speaker is None or speaker in seen_speakers:
            continue
        seen_speakers.add(speaker)
        speakers.append(speaker)
    speakers.sort()
    mean_confidence = _mean(confidences)
    min_confidence = min(confidences) if confidences else None
    low_confidence_word_count = sum(
        1 for value in word_confidences if value < _UNCERTAIN_WORD_CONFIDENCE
    )
    uncertain = False
    if confidences:
        low_ratio = (
            low_confidence_word_count / len(word_confidences) if word_confidences else 0.0
        )
        uncertain = bool(
            (mean_confidence is not None and mean_confidence < _UNCERTAIN_MEAN_CONFIDENCE)
            or low_ratio >= _UNCERTAIN_LOW_WORD_RATIO
        )
    return {
        "uncertain": uncertain,
        "word_count": len(words),
        "mean_confidence": mean_confidence,
        "min_confidence": min_confidence,
        "low_confidence_word_count": low_confidence_word_count,
        "mean_speaker_confidence": _mean(speaker_confidences),
        "utterance_count": len(utterances),
        "speakers": speakers,
    }


def _reconstruct_from_words(words: Sequence[Mapping[str, Any]]) -> str:
    lines: list[str] = []
    current_speaker: Any = object()
    current_parts: list[str] = []
    for word in words:
        text = str(word.get("punctuated_word") or word.get("word") or "").strip()
        if not text:
            continue
        speaker = _speaker_number(word.get("speaker"))
        if current_parts and speaker != current_speaker:
            lines.append(_format_speaker_line(current_speaker, " ".join(current_parts)))
            current_parts = []
        current_speaker = speaker
        current_parts.append(text)
    if current_parts:
        lines.append(_format_speaker_line(current_speaker, " ".join(current_parts)))
    return "\n".join(lines)


def _format_speaker_line(speaker: Any, text: str) -> str:
    label = _speaker_number(speaker)
    if label is None:
        return text
    return f"[Speaker {label}] {text}"


def _speaker_number(speaker: Any) -> int | None:
    if speaker is None or isinstance(speaker, bool) or speaker == "":
        return None
    try:
        return int(speaker)
    except (TypeError, ValueError):
        return None


def _alternative_transcript(result: Mapping[str, Any] | None) -> str:
    alternative = _first_alternative(result)
    return str((alternative or {}).get("transcript") or "").strip()


def _results(result: Mapping[str, Any] | None) -> Mapping[str, Any]:
    results = (result or {}).get("results") if isinstance(result, Mapping) else None
    return results if isinstance(results, Mapping) else {}


def _utterances(result: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    raw = _results(result).get("utterances") or []
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, Mapping)]


def _first_alternative(result: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    channels = _results(result).get("channels") or []
    if not isinstance(channels, list) or not channels:
        return None
    channel = channels[0]
    if not isinstance(channel, Mapping):
        return None
    alternatives = channel.get("alternatives") or []
    if not isinstance(alternatives, list) or not alternatives:
        return None
    alternative = alternatives[0]
    return alternative if isinstance(alternative, Mapping) else None


def _words(result: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    alternative = _first_alternative(result) or {}
    raw = alternative.get("words") or []
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, Mapping)]


def _all_words(result: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    words: list[Mapping[str, Any]] = []
    for utterance in _utterances(result):
        raw = utterance.get("words") or []
        if not isinstance(raw, list):
            continue
        words.extend(item for item in raw if isinstance(item, Mapping))
    return words or _words(result)


def _numeric_values(items: Sequence[Mapping[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for item in items:
        value = _as_float(item.get(field))
        if value is not None:
            values.append(value)
    return values


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool) or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return number


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _extract_language(result: Mapping[str, Any]) -> str | None:
    channels = _results(result).get("channels") or []
    if isinstance(channels, list) and channels and isinstance(channels[0], Mapping):
        language = channels[0].get("detected_language")
        if language:
            return str(language)
    metadata = result.get("metadata") if isinstance(result.get("metadata"), Mapping) else {}
    language = metadata.get("detected_language") or metadata.get("language")
    return str(language) if language else None


def _extract_request_id(result: Mapping[str, Any]) -> str | None:
    metadata = result.get("metadata") if isinstance(result.get("metadata"), Mapping) else {}
    request_id = metadata.get("request_id") or metadata.get("transaction_key")
    return str(request_id) if request_id else None


def _trim_error_body(body: str | None) -> str:
    text = " ".join(str(body or "").split())
    if len(text) <= _MAX_ERROR_BODY_CHARS:
        return text
    return text[:_MAX_ERROR_BODY_CHARS] + "..."


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
