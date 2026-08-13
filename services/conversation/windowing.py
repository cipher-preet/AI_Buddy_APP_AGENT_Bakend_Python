from __future__ import annotations

from dataclasses import dataclass

from apps.api_gateway.config.setting import settings
from services.conversation.models import ConversationDocument, ConversationWindowDocument, TranscriptChunkDocument
from services.conversation.transcript import estimate_tokens, normalize_chunk_text


@dataclass(frozen=True)
class BuiltWindow:
    window: ConversationWindowDocument
    sequence_numbers: list[int]


def build_ready_windows(
    conversation: ConversationDocument,
    chunks: list[TranscriptChunkDocument],
    start_index: int,
    force_final: bool = False,
) -> list[BuiltWindow]:
    ordered = sorted(chunks, key=lambda chunk: chunk.sequenceNumber)
    contiguous = _contiguous_prefix(ordered)
    if not contiguous:
        return []

    windows: list[BuiltWindow] = []
    current: list[TranscriptChunkDocument] = []
    current_tokens = 0
    current_duration = 0
    previous_text = ""
    index = start_index

    def flush(final_partial: bool = False) -> None:
        nonlocal current, current_tokens, current_duration, previous_text, index
        if not current:
            return
        text = _window_text(current)
        sequence_numbers = [chunk.sequenceNumber for chunk in current]
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
            overlapSequenceStart=_overlap_sequence_start(current),
            isFinalPartial=final_partial,
        )
        windows.append(BuiltWindow(window=window, sequence_numbers=sequence_numbers))
        index += 1
        current = []
        current_tokens = 0
        current_duration = 0
        previous_text = ""

    for chunk in contiguous:
        text = normalize_chunk_text(chunk.normalizedText or chunk.rawText or "", previous_text)
        line = f"[{chunk.sequenceNumber}] {text}"
        tokens = estimate_tokens(line)
        duration = int(chunk.endTimeMs or 0)
        should_close = bool(current) and (
            current_tokens + tokens > settings.INCREMENTAL_WINDOW_TARGET_TOKENS
            or (
                settings.INCREMENTAL_WINDOW_MAX_DURATION_MS > 0
                and current_duration + duration > settings.INCREMENTAL_WINDOW_MAX_DURATION_MS
            )
        )
        if should_close:
            flush(False)
        current.append(chunk)
        current_tokens += tokens
        current_duration += duration
        previous_text = text

    if current and (force_final or current_tokens >= settings.INCREMENTAL_WINDOW_TARGET_TOKENS):
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


def _overlap_sequence_start(chunks: list[TranscriptChunkDocument]) -> int | None:
    if settings.INCREMENTAL_WINDOW_OVERLAP_TOKENS <= 0:
        return None
    token_total = 0
    for chunk in reversed(chunks):
        text = chunk.normalizedText or chunk.rawText or ""
        token_total += estimate_tokens(text)
        if token_total >= settings.INCREMENTAL_WINDOW_OVERLAP_TOKENS:
            return chunk.sequenceNumber
    return chunks[0].sequenceNumber if chunks else None
