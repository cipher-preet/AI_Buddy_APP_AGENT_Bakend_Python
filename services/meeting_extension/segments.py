from __future__ import annotations

from typing import Any, Mapping

from services.meeting_extension.timestamps import meeting_relative_ms, spoken_at_utc_iso


def extract_stt_segments(
    result: Mapping[str, Any] | None,
    *,
    chunk_start_offset_ms: int,
    chunk_end_offset_ms: int,
    started_at: str | None,
    sequence: int,
    chunk_id: str,
) -> list[dict[str, Any]]:
    utterances = _utterances(result)
    segments: list[dict[str, Any]] = []
    for index, utterance in enumerate(utterances):
        text = str(utterance.get("transcript") or utterance.get("text") or "").strip()
        if not text:
            continue
        start_s = _seconds(utterance, "start")
        end_s = _seconds(utterance, "end")
        if start_s is None:
            words = utterance.get("words") or []
            if words:
                start_s = _seconds(words[0], "start")
                end_s = _seconds(words[-1], "end")
        start_ms = meeting_relative_ms(chunk_start_offset_ms, start_s or 0)
        end_ms = meeting_relative_ms(chunk_start_offset_ms, end_s if end_s is not None else start_s or 0)
        if end_ms < start_ms:
            end_ms = start_ms
        speaker = utterance.get("speaker")
        segments.append(
            {
                "id": f"{chunk_id}:{index}",
                "index": index,
                "text": text,
                "startOffsetMs": start_ms,
                "endOffsetMs": end_ms,
                "spokenAtUtc": spoken_at_utc_iso(started_at, start_ms),
                "speakerId": speaker if speaker is None or isinstance(speaker, int) else str(speaker),
                "speakerLabel": None,
                "chunkSequence": sequence,
            }
        )
    if segments:
        return segments

    transcript = str((result or {}).get("transcript") or "").strip()
    if not transcript:
        return []
    return [
        {
            "id": f"{chunk_id}:0",
            "index": 0,
            "text": transcript,
            "startOffsetMs": int(chunk_start_offset_ms),
            "endOffsetMs": int(chunk_end_offset_ms),
            "spokenAtUtc": spoken_at_utc_iso(started_at, int(chunk_start_offset_ms)),
            "speakerId": None,
            "speakerLabel": None,
            "chunkSequence": sequence,
        }
    ]


def _utterances(result: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    if not isinstance(result, Mapping):
        return []
    results = result.get("results")
    if isinstance(results, Mapping) and isinstance(results.get("utterances"), list):
        return [item for item in results["utterances"] if isinstance(item, Mapping)]
    raw = result.get("utterances") or result.get("timestamps") or []
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, Mapping)]
    return []


def _seconds(item: Mapping[str, Any], field: str) -> float | None:
    value = item.get(field)
    if value is None and field == "start":
        value = item.get("start_time")
    if value is None and field == "end":
        value = item.get("end_time")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
