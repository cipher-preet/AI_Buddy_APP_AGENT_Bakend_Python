"""Deterministic mechanical transcript windows. No semantic splitting."""

from __future__ import annotations

import re
from typing import Iterable

from services.conversation.event_pipeline.textutil import stable_id
from services.conversation.meeting_pipeline.flags import (
    extraction_window_max_tokens,
    extraction_window_overlap_ratio,
    extraction_window_target_tokens,
)
from services.conversation.meeting_pipeline.schemas import ExtractionWindow, TranscriptTurn
from services.conversation.transcript import estimate_tokens
from services.conversation.windowing import useful_transcript_text


_SPEAKER_PREFIX = re.compile(
    r"^(?:\[)?(?:speaker[\s_:-]*)(\d+)(?:\])?\s*[:\-]\s*(.*)$",
    re.IGNORECASE | re.DOTALL,
)


def turns_from_chunks(chunks: Iterable) -> list[TranscriptTurn]:
    turns: list[TranscriptTurn] = []
    seen: set[int] = set()
    for chunk in sorted(chunks, key=lambda item: int(getattr(item, "sequenceNumber", 0) or 0)):
        sequence = int(getattr(chunk, "sequenceNumber", 0) or 0)
        if sequence in seen:
            continue
        seen.add(sequence)
        raw = useful_transcript_text(chunk) if hasattr(chunk, "rawText") or hasattr(chunk, "normalizedText") else str(getattr(chunk, "raw_text", "") or "")
        speaker = str(getattr(chunk, "speaker", None) or "") or None
        text = raw
        parsed_speaker, parsed_text = split_speaker_label(raw)
        if parsed_speaker and not speaker:
            speaker = parsed_speaker
        if parsed_text:
            text = parsed_text
        turns.append(TranscriptTurn(sequence_id=sequence, speaker=speaker, raw_text=text))
    return turns


def split_speaker_label(text: str) -> tuple[str | None, str]:
    value = (text or "").strip()
    if not value:
        return None, ""
    match = _SPEAKER_PREFIX.match(value)
    if not match:
        return None, value
    return f"Speaker {match.group(1)}", (match.group(2) or "").strip()


def format_window_line(turn: TranscriptTurn) -> str:
    speaker = turn.speaker or "Speaker"
    body = (turn.raw_text or "").strip()
    return f"[{turn.sequence_id}][{speaker}] {body}".rstrip()


def useful_token_count(turn: TranscriptTurn) -> int:
    text = (turn.raw_text or "").strip()
    if not text:
        return 0
    return estimate_tokens(format_window_line(turn))


def build_extraction_windows(
    turns: list[TranscriptTurn],
    *,
    conversation_id: str = "",
    target_tokens: int | None = None,
    max_tokens: int | None = None,
    overlap_ratio: float | None = None,
) -> list[ExtractionWindow]:
    ordered = sorted(turns, key=lambda turn: turn.sequence_id)
    useful = [turn for turn in ordered if (turn.raw_text or "").strip()]
    if not useful:
        return []

    target = max(1, int(target_tokens if target_tokens is not None else extraction_window_target_tokens()))
    maximum = max(target, int(max_tokens if max_tokens is not None else extraction_window_max_tokens()))
    ratio = float(overlap_ratio if overlap_ratio is not None else extraction_window_overlap_ratio())
    overlap_budget = max(0, int(target * ratio))

    windows: list[ExtractionWindow] = []
    overlap: list[TranscriptTurn] = []
    current: list[TranscriptTurn] = []
    current_tokens = 0
    overlap_ids = {turn.sequence_id for turn in overlap}

    def window_text(items: list[TranscriptTurn]) -> str:
        return "\n".join(format_window_line(turn) for turn in items if (turn.raw_text or "").strip())

    def flush() -> None:
        nonlocal current, current_tokens, overlap, overlap_ids
        useful_current = [turn for turn in current if (turn.raw_text or "").strip()]
        if not useful_current:
            current = []
            current_tokens = 0
            return
        owned = [turn.sequence_id for turn in useful_current if turn.sequence_id not in overlap_ids]
        if not owned:
            current = list(overlap)
            current_tokens = _token_total(current)
            return
        sequence_ids = [turn.sequence_id for turn in useful_current]
        overlap_sequence_ids = [turn.sequence_id for turn in useful_current if turn.sequence_id in overlap_ids]
        index = len(windows)
        window_id = stable_id("w", conversation_id, index, sequence_ids[0], sequence_ids[-1])
        windows.append(
            ExtractionWindow(
                window_id=window_id,
                window_index=index,
                sequence_start=sequence_ids[0],
                sequence_end=sequence_ids[-1],
                sequence_ids=sequence_ids,
                owned_sequence_ids=owned,
                overlap_sequence_ids=overlap_sequence_ids,
                text=window_text(useful_current),
                token_count=_token_total(useful_current),
            )
        )
        overlap = _tail_for_overlap(useful_current, overlap_budget)
        overlap_ids = {turn.sequence_id for turn in overlap}
        current = list(overlap)
        current_tokens = _token_total(current)

    for turn in useful:
        tokens = useful_token_count(turn)
        if tokens <= 0:
            continue
        owned_exists = any(item.sequence_id not in overlap_ids for item in current)
        if owned_exists and current_tokens + tokens > maximum:
            flush()
        elif owned_exists and current_tokens >= target and current_tokens + tokens > target:
            flush()
        if tokens > maximum and not current:
            fragments = _split_oversize_turn(turn, maximum)
            for fragment in fragments:
                current = [fragment]
                current_tokens = useful_token_count(fragment)
                flush()
            continue
        current.append(turn)
        current_tokens += tokens

    owned_exists = any(item.sequence_id not in overlap_ids for item in current)
    if owned_exists and any((turn.raw_text or "").strip() for turn in current):
        flush()
    error = window_coverage_error(windows, [turn.sequence_id for turn in useful])
    if error:
        raise RuntimeError(error)
    return windows


def window_coverage_error(windows: list[ExtractionWindow], useful_ids: list[int]) -> str | None:
    """Return an error if any useful sequence is missing from extraction windows."""
    useful = list(useful_ids)
    if not useful:
        return None if not windows else "windows_built_for_empty_transcript"
    union = {sequence for window in windows for sequence in window.sequence_ids}
    missing = [sequence for sequence in useful if sequence not in union]
    extra = sorted(union - set(useful))
    if missing or extra:
        return f"window_coverage_mismatch missing={missing} extra={extra}"
    if useful[-1] not in (windows[-1].sequence_ids if windows else []):
        return "last_useful_sequence_missing_from_final_window"
    return None


def covered_sequence_ids(windows: list[ExtractionWindow]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for window in windows:
        for sequence in window.owned_sequence_ids or window.sequence_ids:
            if sequence in seen:
                continue
            seen.add(sequence)
            ordered.append(sequence)
    return ordered


def _token_total(turns: list[TranscriptTurn]) -> int:
    return sum(useful_token_count(turn) for turn in turns)


def _tail_for_overlap(turns: list[TranscriptTurn], token_budget: int) -> list[TranscriptTurn]:
    if token_budget <= 0 or not turns:
        return []
    tail: list[TranscriptTurn] = []
    total = 0
    for turn in reversed(turns):
        total += useful_token_count(turn)
        tail.insert(0, turn)
        if total >= token_budget:
            break
    return tail


def _split_oversize_turn(turn: TranscriptTurn, max_tokens: int) -> list[TranscriptTurn]:
    """Last-resort split of one extreme sequence. Evidence IDs stay the same."""
    words = (turn.raw_text or "").split()
    if len(words) <= 1:
        return [turn]
    fragments: list[TranscriptTurn] = []
    current: list[str] = []
    for word in words:
        probe = TranscriptTurn(sequence_id=turn.sequence_id, speaker=turn.speaker, raw_text=" ".join([*current, word]))
        if current and useful_token_count(probe) > max_tokens:
            fragments.append(
                TranscriptTurn(sequence_id=turn.sequence_id, speaker=turn.speaker, raw_text=" ".join(current))
            )
            current = [word]
        else:
            current.append(word)
    if current:
        fragments.append(TranscriptTurn(sequence_id=turn.sequence_id, speaker=turn.speaker, raw_text=" ".join(current)))
    return fragments or [turn]
