from __future__ import annotations

from dataclasses import dataclass

from apps.api_gateway.config.setting import settings
from services.conversation.models import ConversationDocument, ConversationWindowDocument, TranscriptChunkDocument
from services.conversation.transcript import estimate_tokens, normalize_chunk_text


@dataclass(frozen=True)
class BuiltWindow:
    window: ConversationWindowDocument
    sequence_numbers: list[int]
    owned_sequence_numbers: list[int]


def overlap_token_budget() -> int:
    ratio_tokens = int(settings.INCREMENTAL_WINDOW_TARGET_TOKENS * settings.INCREMENTAL_WINDOW_OVERLAP_RATIO)
    return max(settings.INCREMENTAL_WINDOW_OVERLAP_TOKENS, ratio_tokens, 0)


def build_ready_windows(
    conversation: ConversationDocument,
    chunks: list[TranscriptChunkDocument],
    start_index: int,
    force_final: bool = False,
    overlap_prefix: list[TranscriptChunkDocument] | None = None,
) -> list[BuiltWindow]:
    ordered = sorted(chunks, key=lambda chunk: chunk.sequenceNumber)
    contiguous = _contiguous_prefix(ordered)
    if not contiguous:
        return []

    windows: list[BuiltWindow] = []
    overlap_chunks = _ordered_unique(overlap_prefix or [])
    overlap_ids = {chunk.sequenceNumber for chunk in overlap_chunks}
    current: list[TranscriptChunkDocument] = list(overlap_chunks)
    current_tokens = _token_count(current)
    current_duration = 0
    previous_text = _last_text(current)
    index = start_index
    budget = overlap_token_budget()
    target = settings.INCREMENTAL_WINDOW_TARGET_TOKENS
    maximum = max(settings.INCREMENTAL_WINDOW_MAX_TOKENS, target)

    def flush(final_partial: bool = False) -> None:
        nonlocal current, current_tokens, current_duration, previous_text, index, overlap_ids
        if not current:
            return
        owned = [chunk.sequenceNumber for chunk in current if chunk.sequenceNumber not in overlap_ids]
        if not owned and not final_partial:
            return
        text = _window_text(current)
        sequence_numbers = [chunk.sequenceNumber for chunk in current]
        tail = [] if final_partial else _tail_chunks(current, budget)
        window = ConversationWindowDocument(
            conversationId=conversation.id,
            userId=conversation.userId,
            spaceId=conversation.spaceId,
            processingVersion=conversation.processingVersion,
            windowIndex=index,
            sequenceStart=sequence_numbers[0],
            sequenceEnd=sequence_numbers[-1],
            text=text,
            tokenCount=estimate_tokens(text),
            durationMs=current_duration or None,
            overlapSequenceStart=tail[0].sequenceNumber if tail else None,
            isFinalPartial=final_partial,
        )
        windows.append(
            BuiltWindow(
                window=window,
                sequence_numbers=sequence_numbers,
                owned_sequence_numbers=owned or sequence_numbers,
            )
        )
        index += 1
        current = list(tail)
        overlap_ids = {chunk.sequenceNumber for chunk in current}
        current_tokens = _token_count(current)
        current_duration = 0
        previous_text = _last_text(current)

    for chunk in contiguous:
        text = normalize_chunk_text(chunk.normalizedText or chunk.rawText or "", previous_text)
        line = f"[{chunk.sequenceNumber}] {text}"
        tokens = estimate_tokens(line)
        duration = int(chunk.endTimeMs or 0)
        would_exceed_max = current_tokens + tokens > maximum
        would_exceed_target = current_tokens + tokens > target
        duration_exceeded = bool(
            current
            and settings.INCREMENTAL_WINDOW_MAX_DURATION_MS > 0
            and current_duration + duration > settings.INCREMENTAL_WINDOW_MAX_DURATION_MS
        )
        should_close = bool(current) and any(
            chunk.sequenceNumber not in overlap_ids for chunk in current
        ) and (would_exceed_max or would_exceed_target or duration_exceeded)
        if should_close:
            flush(False)
        current.append(chunk)
        current_tokens += tokens
        current_duration += duration
        previous_text = text

    if current and (force_final or current_tokens >= target):
        flush(force_final)
    return windows


def _contiguous_prefix(chunks: list[TranscriptChunkDocument]) -> list[TranscriptChunkDocument]:
    if not chunks:
        return []
    ordered = sorted(chunks, key=lambda chunk: chunk.sequenceNumber)
    expected = ordered[0].sequenceNumber
    result: list[TranscriptChunkDocument] = []
    for chunk in ordered:
        if chunk.sequenceNumber != expected:
            break
        result.append(chunk)
        expected += 1
    return result


def _window_text(chunks: list[TranscriptChunkDocument]) -> str:
    lines: list[str] = []
    previous = ""
    for chunk in chunks:
        text = normalize_chunk_text(chunk.normalizedText or chunk.rawText or "", previous)
        lines.append(f"[{chunk.sequenceNumber}] {text}")
        previous = text
    return "\n".join(lines).strip()


def _tail_chunks(chunks: list[TranscriptChunkDocument], token_budget: int) -> list[TranscriptChunkDocument]:
    if token_budget <= 0 or not chunks:
        return []
    token_total = 0
    tail: list[TranscriptChunkDocument] = []
    for chunk in reversed(chunks):
        text = chunk.normalizedText or chunk.rawText or ""
        token_total += estimate_tokens(text)
        tail.insert(0, chunk)
        if token_total >= token_budget:
            break
    return tail


def _token_count(chunks: list[TranscriptChunkDocument]) -> int:
    if not chunks:
        return 0
    return estimate_tokens(_window_text(chunks))


def _last_text(chunks: list[TranscriptChunkDocument]) -> str:
    if not chunks:
        return ""
    chunk = chunks[-1]
    return normalize_chunk_text(chunk.normalizedText or chunk.rawText or "", "")


def _ordered_unique(chunks: list[TranscriptChunkDocument]) -> list[TranscriptChunkDocument]:
    seen: set[int] = set()
    unique: list[TranscriptChunkDocument] = []
    for chunk in sorted(chunks, key=lambda item: item.sequenceNumber):
        if chunk.sequenceNumber in seen:
            continue
        seen.add(chunk.sequenceNumber)
        unique.append(chunk)
    return unique
