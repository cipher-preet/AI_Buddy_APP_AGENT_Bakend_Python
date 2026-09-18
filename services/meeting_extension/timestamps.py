from __future__ import annotations


def meeting_relative_ms(chunk_start_offset_ms: int, stt_seconds: float | int | None) -> int:
    """Convert a chunk-relative STT timestamp to meeting-relative milliseconds."""
    start = int(chunk_start_offset_ms or 0)
    seconds = 0.0 if stt_seconds is None else float(stt_seconds)
    return int(round(start + seconds * 1000.0))


def spoken_at_utc_iso(started_at_iso: str | None, start_offset_ms: int) -> str | None:
    if not started_at_iso:
        return None
    from datetime import datetime, timedelta, timezone

    raw = str(started_at_iso).replace("Z", "+00:00")
    try:
        started = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    return (started + timedelta(milliseconds=int(start_offset_ms))).isoformat()
