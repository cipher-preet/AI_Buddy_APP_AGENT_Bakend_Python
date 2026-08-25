from __future__ import annotations

from dataclasses import dataclass, field

from apps.api_gateway.config.setting import settings
from services.conversation.budget import (
    semantic_window_token_max,
    semantic_window_token_target,
    semantic_window_useful_duration_ms,
)
from services.conversation.models import ConversationDocument, ConversationWindowDocument, TranscriptChunkDocument
from services.conversation.transcript import estimate_tokens, normalize_chunk_text


CLOSE_REASON_TOKEN_TARGET = "token_target"
CLOSE_REASON_TOKEN_MAX = "token_max"
CLOSE_REASON_DURATION_MAX = "duration_max"
CLOSE_REASON_SPARSE_TIMEOUT = "sparse_timeout"
CLOSE_REASON_FORCED_FINAL = "forced_final"


@dataclass(frozen=True)
class BuiltWindow:
    window: ConversationWindowDocument
    sequence_numbers: list[int]
    owned_sequence_numbers: list[int]
    skipped_sequence_numbers: list[int] = field(default_factory=list)
    close_reason: str = CLOSE_REASON_TOKEN_TARGET


def overlap_token_budget() -> int:
    ratio_tokens = int(semantic_window_token_target() * settings.INCREMENTAL_WINDOW_OVERLAP_RATIO)
    return max(settings.INCREMENTAL_WINDOW_OVERLAP_TOKENS, ratio_tokens, 0)


def useful_transcript_text(chunk: TranscriptChunkDocument) -> str:
    return (chunk.normalizedText or chunk.rawText or "").strip()


def is_useful_chunk(chunk: TranscriptChunkDocument) -> bool:
    """Non-empty transcript text only (technical). Not semantic usefulness."""
    return bool(useful_transcript_text(chunk))


# Alias clarifying that this is transcript integrity, not LLM usefulness.
is_transcript_usable = is_useful_chunk


def semantic_window_text(chunks: list[TranscriptChunkDocument]) -> str:
    return _window_text(chunks)


def chunk_duration_ms(chunk: TranscriptChunkDocument) -> int:
    return max(0, int(chunk.endTimeMs or 0))


def build_ready_windows(
    conversation: ConversationDocument,
    chunks: list[TranscriptChunkDocument],
    start_index: int,
    force_final: bool = False,
    overlap_prefix: list[TranscriptChunkDocument] | None = None,
    skippable_sequences: set[int] | None = None,
) -> list[BuiltWindow]:
    ordered = sorted(chunks, key=lambda chunk: chunk.sequenceNumber)
    reachable = _reachable_semantic_prefix(ordered, skippable_sequences or set())
    if not reachable:
        return []

    windows: list[BuiltWindow] = []
    overlap_chunks = [chunk for chunk in _ordered_unique(overlap_prefix or []) if is_useful_chunk(chunk)]
    overlap_ids = {chunk.sequenceNumber for chunk in overlap_chunks}
    current: list[TranscriptChunkDocument] = list(overlap_chunks)
    skipped_in_current: list[int] = []
    current_tokens = _useful_token_count(current)
    current_useful_duration = sum(chunk_duration_ms(chunk) for chunk in current)
    current_wall_clock = current_useful_duration
    previous_text = _last_useful_text(current)
    index = start_index
    budget = overlap_token_budget()
    target = semantic_window_token_target()
    maximum = max(semantic_window_token_max(), target)
    useful_duration_limit = semantic_window_useful_duration_ms()
    pending_empties: list[TranscriptChunkDocument] = []

    def flush(reason: str, final_partial: bool = False) -> None:
        nonlocal current, skipped_in_current, current_tokens, current_useful_duration, current_wall_clock, previous_text, index
        useful = [chunk for chunk in current if is_useful_chunk(chunk)]
        if not useful:
            current = []
            skipped_in_current = []
            current_tokens = 0
            current_useful_duration = 0
            current_wall_clock = 0
            previous_text = ""
            return
        owned = [chunk.sequenceNumber for chunk in useful if chunk.sequenceNumber not in overlap_ids]
        if not owned:
            return
        text = _window_text(useful)
        sequence_numbers = [chunk.sequenceNumber for chunk in useful]
        tail = [] if final_partial else _tail_chunks(useful, budget)
        useful_chars = len(text)
        useful_words = len(text.split())
        useful_tokens = estimate_tokens(text) if text else 0
        window = ConversationWindowDocument(
            conversationId=conversation.id,
            userId=conversation.userId,
            spaceId=conversation.spaceId,
            processingVersion=conversation.processingVersion,
            windowIndex=index,
            sequenceStart=sequence_numbers[0],
            sequenceEnd=sequence_numbers[-1],
            text=text,
            tokenCount=useful_tokens,
            durationMs=current_useful_duration or None,
            overlapSequenceStart=tail[0].sequenceNumber if tail else None,
            isFinalPartial=final_partial,
            closeReason=reason,
            sequenceCount=len(sequence_numbers) + len(skipped_in_current),
            emptyChunkCount=len(skipped_in_current),
            nonEmptyChunkCount=len(useful),
            usefulCharCount=useful_chars,
            usefulWordCount=useful_words,
            usefulTokenCount=useful_tokens,
            wallClockSpanMs=current_wall_clock or None,
            meaningfulSpeechMs=current_useful_duration or None,
        )
        windows.append(
            BuiltWindow(
                window=window,
                sequence_numbers=sequence_numbers,
                owned_sequence_numbers=owned or sequence_numbers,
                skipped_sequence_numbers=list(skipped_in_current),
                close_reason=reason,
            )
        )
        index += 1
        current = list(tail)
        overlap_ids.clear()
        overlap_ids.update(chunk.sequenceNumber for chunk in current)
        skipped_in_current = []
        current_tokens = _useful_token_count(current)
        current_useful_duration = sum(chunk_duration_ms(chunk) for chunk in current)
        current_wall_clock = current_useful_duration
        previous_text = _last_useful_text(current)

    for chunk in reachable:
        if not is_useful_chunk(chunk):
            pending_empties.append(chunk)
            continue

        text = normalize_chunk_text(useful_transcript_text(chunk), previous_text)
        line = f"[{chunk.sequenceNumber}] {text}"
        tokens = estimate_tokens(line)
        useful_duration = chunk_duration_ms(chunk)
        empty_span = sum(chunk_duration_ms(item) for item in pending_empties)
        wall_add = empty_span + useful_duration
        owned_exists = any(item.sequenceNumber not in overlap_ids for item in current if is_useful_chunk(item))
        would_exceed_max = current_tokens + tokens > maximum
        would_exceed_target = current_tokens + tokens > target
        useful_duration_exceeded = bool(
            current
            and owned_exists
            and useful_duration_limit > 0
            and current_useful_duration + useful_duration > useful_duration_limit
        )
        sparse_timeout = bool(
            current
            and owned_exists
            and settings.SPARSE_WINDOW_MAX_WALL_CLOCK_MS > 0
            and current_tokens >= settings.SPARSE_WINDOW_MIN_USEFUL_TOKENS
            and current_wall_clock + wall_add > settings.SPARSE_WINDOW_MAX_WALL_CLOCK_MS
        )
        reason = None
        if owned_exists and would_exceed_max:
            reason = CLOSE_REASON_TOKEN_MAX
        elif owned_exists and would_exceed_target:
            reason = CLOSE_REASON_TOKEN_TARGET
        elif useful_duration_exceeded:
            reason = CLOSE_REASON_DURATION_MAX
        elif sparse_timeout:
            reason = CLOSE_REASON_SPARSE_TIMEOUT
        if reason:
            flush(reason, False)

        skipped_in_current.extend(item.sequenceNumber for item in pending_empties)
        current_wall_clock += empty_span
        pending_empties = []
        current.append(chunk)
        current_tokens += tokens
        current_useful_duration += useful_duration
        current_wall_clock += useful_duration
        previous_text = text

    leftover_useful = any(is_useful_chunk(chunk) for chunk in current)
    owned_exists = any(chunk.sequenceNumber not in overlap_ids for chunk in current if is_useful_chunk(chunk))
    trailing_empty_ms = sum(chunk_duration_ms(item) for item in pending_empties)
    if leftover_useful and owned_exists:
        if pending_empties:
            skipped_in_current.extend(item.sequenceNumber for item in pending_empties)
            current_wall_clock += trailing_empty_ms
            pending_empties = []
        if force_final or current_tokens >= target:
            flush(CLOSE_REASON_FORCED_FINAL if force_final else CLOSE_REASON_TOKEN_TARGET, force_final)
        elif (
            current_tokens >= settings.SPARSE_WINDOW_MIN_USEFUL_TOKENS
            and settings.SPARSE_WINDOW_MAX_WALL_CLOCK_MS > 0
            and current_wall_clock >= settings.SPARSE_WINDOW_MAX_WALL_CLOCK_MS
        ):
            flush(CLOSE_REASON_SPARSE_TIMEOUT, False)
    return windows


def leading_skippable_sequences(
    chunks: list[TranscriptChunkDocument],
    skippable_sequences: set[int] | None = None,
) -> list[int]:
    reachable = _reachable_semantic_prefix(
        sorted(chunks, key=lambda chunk: chunk.sequenceNumber),
        skippable_sequences or set(),
    )
    skipped: list[int] = []
    for chunk in reachable:
        if is_useful_chunk(chunk):
            break
        skipped.append(chunk.sequenceNumber)
    return skipped


def _reachable_semantic_prefix(
    chunks: list[TranscriptChunkDocument],
    skippable_sequences: set[int],
) -> list[TranscriptChunkDocument]:
    if not chunks:
        return []
    ordered = sorted(chunks, key=lambda chunk: chunk.sequenceNumber)
    by_sequence = {chunk.sequenceNumber: chunk for chunk in ordered}
    start = ordered[0].sequenceNumber
    end = ordered[-1].sequenceNumber
    result: list[TranscriptChunkDocument] = []
    expected = start
    while expected <= end:
        chunk = by_sequence.get(expected)
        if chunk is not None:
            result.append(chunk)
            expected += 1
            continue
        if expected in skippable_sequences:
            expected += 1
            continue
        break
    return result


def _contiguous_prefix(chunks: list[TranscriptChunkDocument]) -> list[TranscriptChunkDocument]:
    return _reachable_semantic_prefix(chunks, set())


def _window_text(chunks: list[TranscriptChunkDocument]) -> str:
    lines: list[str] = []
    previous = ""
    for chunk in chunks:
        raw = useful_transcript_text(chunk)
        if not raw:
            continue
        text = normalize_chunk_text(raw, previous)
        if not text:
            continue
        lines.append(f"[{chunk.sequenceNumber}] {text}")
        previous = text
    return "\n".join(lines).strip()


def _tail_chunks(chunks: list[TranscriptChunkDocument], token_budget: int) -> list[TranscriptChunkDocument]:
    useful = [chunk for chunk in chunks if is_useful_chunk(chunk)]
    if token_budget <= 0 or not useful:
        return []
    token_total = 0
    tail: list[TranscriptChunkDocument] = []
    for chunk in reversed(useful):
        token_total += estimate_tokens(useful_transcript_text(chunk))
        tail.insert(0, chunk)
        if token_total >= token_budget:
            break
    return tail


def _useful_token_count(chunks: list[TranscriptChunkDocument]) -> int:
    text = _window_text(chunks)
    if not text:
        return 0
    return estimate_tokens(text)


def _last_useful_text(chunks: list[TranscriptChunkDocument]) -> str:
    for chunk in reversed(chunks):
        raw = useful_transcript_text(chunk)
        if raw:
            return normalize_chunk_text(raw, "")
    return ""


def _ordered_unique(chunks: list[TranscriptChunkDocument]) -> list[TranscriptChunkDocument]:
    seen: set[int] = set()
    unique: list[TranscriptChunkDocument] = []
    for chunk in sorted(chunks, key=lambda item: item.sequenceNumber):
        if chunk.sequenceNumber in seen:
            continue
        seen.add(chunk.sequenceNumber)
        unique.append(chunk)
    return unique
