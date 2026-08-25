from __future__ import annotations

import re
from dataclasses import dataclass

from services.conversation.models import Segment, TranscriptChunkDocument


_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?।])\s+")
_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class AssembledTranscript:
    raw_transcript: str
    normalized_transcript: str
    chunks: list[TranscriptChunkDocument]


def assemble_transcript(chunks: list[TranscriptChunkDocument]) -> AssembledTranscript:
    ordered = sorted(chunks, key=lambda chunk: chunk.sequenceNumber)
    raw_parts: list[str] = []
    normalized_parts: list[str] = []
    previous = ""

    for chunk in ordered:
        text = (chunk.normalizedText or chunk.rawText or "").strip()
        normalized = normalize_chunk_text(text, previous)
        raw_parts.append(f"[{chunk.sequenceNumber}] {text}")
        normalized_parts.append(f"[{chunk.sequenceNumber}] {normalized}")
        previous = normalized

    return AssembledTranscript(
        raw_transcript="\n".join(raw_parts),
        normalized_transcript="\n".join(normalized_parts),
        chunks=ordered,
    )


def normalize_chunk_text(text: str, previous_text: str = "") -> str:
    text = _SPACE_RE.sub(" ", text).strip()
    if not text:
        return ""

    previous_tail = previous_text[-120:].lower()
    lowered = text.lower()
    for size in range(min(80, len(text)), 7, -1):
        prefix = lowered[:size]
        if prefix and prefix in previous_tail:
            text = text[size:].lstrip(" ,.-")
            break

    return text


def estimate_tokens(text: str) -> int:
    """Estimate tokens using a tokenizer when available, else a safe char/word mix."""
    value = text or ""
    if not value.strip():
        return 0
    tokenizer = _tiktoken_encoder()
    if tokenizer is not None:
        try:
            return max(1, len(tokenizer.encode(value)))
        except Exception:
            pass
    words = len(value.split())
    char_estimate = max(1, (len(value) + 3) // 4)
    return max(words, char_estimate)


_TIKTOKEN_ENCODER = None
_TIKTOKEN_LOADED = False


def _tiktoken_encoder():
    global _TIKTOKEN_ENCODER, _TIKTOKEN_LOADED
    if _TIKTOKEN_LOADED:
        return _TIKTOKEN_ENCODER
    _TIKTOKEN_LOADED = True
    try:
        import tiktoken

        _TIKTOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
    except Exception:
        _TIKTOKEN_ENCODER = None
    return _TIKTOKEN_ENCODER


def segment_transcript(
    conversation_id: str,
    chunks: list[TranscriptChunkDocument],
    target_tokens: int,
    overlap_ratio: float,
    max_segments: int,
) -> list[Segment]:
    if not chunks:
        return []

    ordered = sorted(chunks, key=lambda chunk: chunk.sequenceNumber)
    overlap_tokens = max(0, int(target_tokens * overlap_ratio))
    segments: list[Segment] = []
    current_lines: list[str] = []
    current_start = ordered[0].sequenceNumber
    current_end = current_start
    current_tokens = 0

    def flush() -> None:
        nonlocal current_lines, current_start, current_end, current_tokens
        if not current_lines:
            return
        text = "\n".join(current_lines).strip()
        segments.append(
            Segment(
                segmentId=f"segment_{len(segments) + 1}",
                conversationId=conversation_id,
                sequenceStart=current_start,
                sequenceEnd=current_end,
                text=text,
                tokenCount=estimate_tokens(text),
            )
        )
        if overlap_tokens and current_lines:
            overlap: list[str] = []
            token_total = 0
            for line in reversed(current_lines):
                token_total += estimate_tokens(line)
                overlap.insert(0, line)
                if token_total >= overlap_tokens:
                    break
            current_lines = overlap
            current_tokens = sum(estimate_tokens(line) for line in current_lines)
            current_start = _sequence_from_line(current_lines[0]) if current_lines else current_end
        else:
            current_lines = []
            current_tokens = 0

    for chunk in ordered:
        text = chunk.normalizedText or chunk.rawText or ""
        sentences = _SENTENCE_BOUNDARY_RE.split(text.strip()) if text.strip() else [""]
        for sentence in sentences:
            line = f"[{chunk.sequenceNumber}] {sentence.strip()}"
            token_count = estimate_tokens(line)
            if current_lines and current_tokens + token_count > target_tokens:
                flush()
                if len(segments) >= max_segments:
                    return segments
            if not current_lines:
                current_start = chunk.sequenceNumber
            current_lines.append(line)
            current_end = chunk.sequenceNumber
            current_tokens += token_count

    flush()
    return segments[:max_segments]


def _sequence_from_line(line: str) -> int:
    match = re.match(r"\[(\d+)\]", line)
    return int(match.group(1)) if match else 0


def detect_missing_sequences(sequence_numbers: list[int], expected_last_sequence: int) -> list[int]:
    present = set(sequence_numbers)
    return [number for number in range(0, expected_last_sequence + 1) if number not in present]
