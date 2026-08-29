"""Deterministic structural cleaning. Accounting, not semantic deletion."""

from __future__ import annotations

from services.conversation.event_pipeline.schemas import CleanedTranscriptRecord, CleaningLedger
from services.conversation.event_pipeline.textutil import casefold_text, normalize_text
from services.conversation.models import STTStatus, TranscriptChunkDocument
from services.conversation.semantic_input import as_sequence_number
from services.conversation.windowing import useful_transcript_text


_PLACEHOLDERS = frozenset(
    {
        "",
        ".",
        "..",
        "...",
        "null",
        "none",
        "n/a",
        "na",
        "[inaudible]",
        "[silence]",
        "(blank)",
        "blank",
        "<unk>",
        "[unk]",
        "silence",
        "(silence)",
        "[empty]",
        "empty",
        "undefined",
        "(inaudible)",
    }
)


def clean_transcripts(
    chunks: list[TranscriptChunkDocument],
    *,
    conversation_id: str = "",
    user_id: str = "",
    space_id: str = "",
) -> CleaningLedger:
    ordered = sorted(chunks, key=lambda chunk: as_sequence_number(chunk.sequenceNumber, 0))
    seen_sequences: set[int] = set()
    seen_exact: set[tuple[int, str]] = set()
    previous_text = ""
    records: list[CleanedTranscriptRecord] = []
    useful: list[CleanedTranscriptRecord] = []
    excluded: list[CleanedTranscriptRecord] = []

    for chunk in ordered:
        sequence = as_sequence_number(chunk.sequenceNumber, 0)
        raw = useful_transcript_text(chunk)
        normalized = normalize_text(raw)
        source_id = str(getattr(chunk, "chunkId", None) or getattr(chunk, "id", "") or f"seq-{sequence}")
        record = CleanedTranscriptRecord(
            sequenceId=sequence,
            chunkId=str(getattr(chunk, "chunkId", "") or source_id),
            sourceId=source_id,
            speaker=str(getattr(chunk, "speaker", None) or "") or None,
            rawText=raw or normalized,
            timestampMs=_timestamp_ms(chunk),
            sessionId=str(getattr(chunk, "conversationId", None) or conversation_id),
            spaceId=str(getattr(chunk, "spaceId", None) or space_id),
            userId=str(getattr(chunk, "userId", None) or user_id),
            languageCode=getattr(chunk, "languageCode", None),
        )
        reason = _structural_exclusion_reason(chunk, sequence, normalized, seen_sequences, seen_exact, previous_text)
        if reason:
            record.excluded = True
            record.exclusionReason = reason
            record.rawText = raw or normalized
            excluded.append(record)
        else:
            useful.append(record)
            seen_sequences.add(sequence)
            seen_exact.add((sequence, casefold_text(normalized)))
            previous_text = casefold_text(normalized)
        records.append(record)

    accounted = [record.sequenceId for record in records]
    return CleaningLedger(
        totalSequences=len(records),
        usefulSequences=len(useful),
        excludedStructuralSequences=len(excluded),
        records=records,
        useful=useful,
        excluded=excluded,
        accountedSequenceIds=accounted,
    )


def _structural_exclusion_reason(
    chunk: TranscriptChunkDocument,
    sequence: int,
    normalized: str,
    seen_sequences: set[int],
    seen_exact: set[tuple[int, str]],
    previous_text: str,
) -> str | None:
    if getattr(chunk, "sttStatus", None) == STTStatus.FAILED:
        return "stt_failed"
    if sequence in seen_sequences:
        return "duplicate_sequence"
    folded = casefold_text(normalized)
    if (sequence, folded) in seen_exact:
        return "duplicate_record"
    if not folded:
        return "empty_or_whitespace"
    if folded in _PLACEHOLDERS:
        return "transport_placeholder"
    if folded == previous_text:
        return "duplicate_exact_transcript"
    return None


def _timestamp_ms(chunk: TranscriptChunkDocument) -> int | None:
    for field in ("startTimeMs", "endTimeMs"):
        value = getattr(chunk, field, None)
        if value is not None:
            return int(value)
    captured = getattr(chunk, "capturedAt", None)
    if captured is not None and hasattr(captured, "timestamp"):
        return int(captured.timestamp() * 1000)
    return None
