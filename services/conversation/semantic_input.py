from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from services.conversation.models import STTStatus, TranscriptChunkDocument, WindowProcessingStatus
from services.conversation.transcript import estimate_tokens
from services.conversation.windowing import semantic_window_text, useful_transcript_text


SEMANTIC_INPUT_ASSEMBLY_FAILED = "SEMANTIC_INPUT_ASSEMBLY_FAILED"

_LINE_RE = re.compile(r"^\s*\[(?P<seq>\d+)\]\s*(?P<text>.*)$")

# Technical rejection reasons only. Never semantic (actionable/important/topic/etc.).
_ALLOWED_REJECTION_REASONS = (
    "wrong_conversation",
    "outside_window_range",
    "empty_text",
    "stt_failed",
    "damaged",
    "duplicate_sequence",
    "invalid_sequence",
)


def as_sequence_number(value: Any, default: int | None = None) -> int:
    """Canonical int sequence. String '5' and int 5 are identical."""
    try:
        return int(value)
    except (TypeError, ValueError):
        if default is not None:
            return default
        raise


def is_transcript_usable(chunk: TranscriptChunkDocument) -> bool:
    """Technical usability only: non-empty normalized/raw STT text.

    Does NOT judge tasks, notes, actions, importance, or semantic value.
    """
    return bool(useful_transcript_text(chunk))


def parsed_semantic_sequences(window_text: str) -> dict[int, str]:
    transcript: dict[int, str] = {}
    for line in (window_text or "").splitlines():
        match = _LINE_RE.match(line)
        if not match:
            continue
        text = " ".join(match.group("text").split())
        if not text:
            continue
        transcript[as_sequence_number(match.group("seq"))] = text
    return transcript


@dataclass
class SemanticWindowAssembly:
    text: str
    sequence_start: int = 0
    sequence_end: int = 0
    window_id: str | None = None
    window_index: int = 0
    failed: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def useful_sequence_numbers(self) -> list[int]:
        return list(self.diagnostics.get("usefulSequenceNumbers") or [])


def assemble_semantic_window_input(
    *,
    conversation_id: str,
    chunks: Iterable[TranscriptChunkDocument],
    windows: Iterable[Any] | None = None,
    sequence_start: int | None = None,
    sequence_end: int | None = None,
    mode: str = "final_raw",
) -> SemanticWindowAssembly:
    """Assemble usefulChunks as a pure transcript-integrity layer.

    Flow:
      persisted chunks → conversation match → window range → successful STT →
      non-empty text → not damaged → dedupe → sort ASC → usefulChunks → LLM

    No semantic judgment (tasks/notes/importance/intent) happens here.
    Never queries only unwindowed rows; uses durable membership.
    """
    window_list = list(windows or [])
    range_start = as_sequence_number(sequence_start, 0) if sequence_start is not None else None
    range_end = as_sequence_number(sequence_end, 0) if sequence_end is not None else None
    ordered = sorted(chunks, key=lambda chunk: as_sequence_number(chunk.sequenceNumber, 0))
    rejection_counts = {reason: 0 for reason in _ALLOWED_REJECTION_REASONS}
    persisted_sequences = [as_sequence_number(chunk.sequenceNumber, 0) for chunk in ordered]
    checkpoint_owned = _checkpoint_owned_sequences(window_list) if mode == "leftover" else set()

    conversation_matched = 0
    window_matched = 0
    seen: set[int] = set()
    useful: list[TranscriptChunkDocument] = []

    for chunk in ordered:
        try:
            sequence = as_sequence_number(chunk.sequenceNumber)
        except (TypeError, ValueError):
            rejection_counts["invalid_sequence"] += 1
            continue

        if conversation_id and not _same_conversation(chunk.conversationId, conversation_id):
            rejection_counts["wrong_conversation"] += 1
            continue
        conversation_matched += 1

        if not _in_range(sequence, range_start, range_end) or sequence in checkpoint_owned:
            rejection_counts["outside_window_range"] += 1
            continue
        window_matched += 1

        if sequence in seen:
            rejection_counts["duplicate_sequence"] += 1
            continue

        reason = _technical_exclusion_reason(chunk)
        if reason:
            rejection_counts[reason] += 1
            continue

        seen.add(sequence)
        useful.append(chunk)

    # Deterministic ASC order; never rely on Mongo/Redis return order.
    useful.sort(key=lambda chunk: as_sequence_number(chunk.sequenceNumber))
    useful_sequences = [as_sequence_number(chunk.sequenceNumber) for chunk in useful]

    # usefulChunks text is built from integrity-filtered rows, not semantic filters.
    text = semantic_window_text(useful)
    if not text:
        durable_text, _ = _durable_window_text(window_list, mode, useful_sequences)
        if parsed_semantic_sequences(durable_text):
            text = durable_text

    parsed = parsed_semantic_sequences(text)
    start = min(useful_sequences) if useful_sequences else (range_start if range_start is not None else 0)
    end = max(useful_sequences) if useful_sequences else (range_end if range_end is not None else start)

    eligible_non_empty = [
        chunk
        for chunk in ordered
        if _same_conversation(chunk.conversationId, conversation_id)
        and is_transcript_usable(chunk)
        and chunk.sttStatus == STTStatus.COMPLETED
        and as_sequence_number(chunk.sequenceNumber, -1) not in checkpoint_owned
        and _in_range(as_sequence_number(chunk.sequenceNumber, -1), range_start, range_end)
    ]
    failed = bool(eligible_non_empty) and not useful_sequences
    source_window = _covering_window(window_list, start, end)

    completed_non_empty = [
        chunk
        for chunk in ordered
        if is_transcript_usable(chunk) and chunk.sttStatus == STTStatus.COMPLETED
    ]
    diagnostics = {
        "persistedTranscriptCount": len(ordered),
        "persistedNonEmptyTranscriptCount": len(completed_non_empty),
        "persistedSequenceNumbers": persisted_sequences,
        "conversationMatchedCount": conversation_matched,
        "windowMatchedCount": window_matched,
        "queriedTranscriptCount": len(ordered),
        "queriedSequenceNumbers": persisted_sequences,
        "windowId": str(source_window.id) if source_window is not None else None,
        "windowIndex": int(getattr(source_window, "windowIndex", 0) or 0),
        "sequenceStart": start,
        "sequenceEnd": end,
        "expectedSequenceCount": (end - start + 1) if useful_sequences or range_start is not None else 0,
        "windowTranscriptCountBeforeFiltering": window_matched,
        "emptyTranscriptCount": rejection_counts["empty_text"],
        "failedTranscriptCount": rejection_counts["stt_failed"],
        "damagedTranscriptCount": rejection_counts["damaged"],
        "duplicateTranscriptCount": rejection_counts["duplicate_sequence"],
        "emptyFilteredCount": rejection_counts["empty_text"],
        "unusableFilteredCount": (
            rejection_counts["stt_failed"]
            + rejection_counts["damaged"]
            + rejection_counts["wrong_conversation"]
            + rejection_counts["outside_window_range"]
            + rejection_counts["invalid_sequence"]
        ),
        "usefulTranscriptCount": len(useful),
        "usefulTranscriptCountAfterFiltering": len(useful),
        "usefulSequenceNumbers": useful_sequences,
        "usefulChunks": useful_sequences,
        "semanticInputTranscriptCount": len(parsed),
        "semanticInputCharacterCount": len(text or ""),
        "semanticInputEstimatedTokens": estimate_tokens(text) if text else 0,
        "rejectionCounts": dict(rejection_counts),
        "semanticInputAssemblyFailed": failed,
        "semanticInputSource": "persisted_chunks",
        "unpublishedFilterApplied": False,
        "unwindowedOnlyQuery": False,
    }
    print(
        "Useful chunk assembly completed:",
        {
            "conversationId": conversation_id,
            "windowId": diagnostics["windowId"],
            "windowIndex": diagnostics["windowIndex"],
            "persistedTranscriptCount": diagnostics["persistedTranscriptCount"],
            "conversationMatchedCount": conversation_matched,
            "windowMatchedCount": window_matched,
            "emptyTranscriptCount": diagnostics["emptyTranscriptCount"],
            "failedTranscriptCount": diagnostics["failedTranscriptCount"],
            "damagedTranscriptCount": diagnostics["damagedTranscriptCount"],
            "duplicateTranscriptCount": diagnostics["duplicateTranscriptCount"],
            "usefulTranscriptCount": diagnostics["usefulTranscriptCount"],
            "usefulSequenceNumbers": useful_sequences,
            "sequenceStart": start,
            "sequenceEnd": end,
            "rejectionCounts": rejection_counts,
        },
    )
    return SemanticWindowAssembly(
        text=text,
        sequence_start=start,
        sequence_end=end,
        window_id=diagnostics["windowId"],
        window_index=diagnostics["windowIndex"],
        failed=failed,
        diagnostics=diagnostics,
    )


def semantic_input_assembly_failed(window, window_text: str | None = None) -> bool:
    diagnostics = getattr(window, "semanticInputDiagnostics", None) or {}
    if diagnostics.get("semanticInputAssemblyFailed"):
        return True
    persisted = int(
        diagnostics.get("persistedNonEmptyTranscriptCount")
        or getattr(window, "nonEmptyChunkCount", 0)
        or 0
    )
    text = window_text if window_text is not None else getattr(window, "text", "")
    return persisted > 0 and not parsed_semantic_sequences(text or "")


def empty_semantic_input_diagnostics() -> dict[str, Any]:
    return {
        "persistedTranscriptCount": 0,
        "persistedNonEmptyTranscriptCount": 0,
        "persistedSequenceNumbers": [],
        "conversationMatchedCount": 0,
        "windowMatchedCount": 0,
        "queriedTranscriptCount": 0,
        "queriedSequenceNumbers": [],
        "windowId": None,
        "windowIndex": 0,
        "sequenceStart": 0,
        "sequenceEnd": 0,
        "expectedSequenceCount": 0,
        "windowTranscriptCountBeforeFiltering": 0,
        "emptyTranscriptCount": 0,
        "failedTranscriptCount": 0,
        "damagedTranscriptCount": 0,
        "duplicateTranscriptCount": 0,
        "emptyFilteredCount": 0,
        "unusableFilteredCount": 0,
        "usefulTranscriptCount": 0,
        "usefulTranscriptCountAfterFiltering": 0,
        "usefulSequenceNumbers": [],
        "usefulChunks": [],
        "semanticInputTranscriptCount": 0,
        "semanticInputCharacterCount": 0,
        "semanticInputEstimatedTokens": 0,
        "rejectionCounts": {reason: 0 for reason in _ALLOWED_REJECTION_REASONS},
        "semanticInputAssemblyFailed": False,
        "unpublishedFilterApplied": False,
        "unwindowedOnlyQuery": False,
    }


def _technical_exclusion_reason(chunk: TranscriptChunkDocument) -> str | None:
    status = chunk.sttStatus
    status_value = status.value if hasattr(status, "value") else str(status or "")
    if status_value == STTStatus.FAILED.value:
        return "stt_failed"
    exclusion = str(getattr(chunk, "exclusionReason", "") or "")
    if exclusion and "empty" not in exclusion.casefold() and not is_transcript_usable(chunk):
        return "damaged"
    if status_value != STTStatus.COMPLETED.value:
        # Pending/processing without completed successful STT cannot enter usefulChunks.
        if is_transcript_usable(chunk):
            return "stt_failed"
        return "empty_text"
    if not is_transcript_usable(chunk):
        return "empty_text"
    return None


def _in_range(sequence: int, start: int | None, end: int | None) -> bool:
    if start is not None and sequence < start:
        return False
    if end is not None and sequence > end:
        return False
    return True


def _checkpoint_owned_sequences(windows: list[Any]) -> set[int]:
    """Sequences already covered by completed semantic checkpoints (leftover mode only)."""
    owned: set[int] = set()
    for window in windows:
        if getattr(window, "isFinalPartial", False) or getattr(window, "extractionSkipped", False):
            continue
        if str(getattr(window, "checkpointKind", "") or "") == "raw_final":
            continue
        status = getattr(window, "status", None)
        status_value = status.value if hasattr(status, "value") else str(status or "")
        if status_value and status_value != WindowProcessingStatus.COMPLETED.value:
            continue
        result = getattr(window, "result", None)
        has_checkpoint = bool(
            result is not None
            and (
                getattr(result, "isCheckpoint", False)
                or getattr(result, "semanticUnits", None)
                or getattr(result, "tasks", None)
                or getattr(result, "notes", None)
            )
        )
        if not has_checkpoint:
            continue
        start = as_sequence_number(getattr(window, "sequenceStart", 0), 0)
        end = as_sequence_number(getattr(window, "sequenceEnd", 0), 0)
        if end >= start:
            owned.update(range(start, end + 1))
    return owned


def _durable_window_text(windows: list[Any], mode: str, useful_sequences: list[int]) -> tuple[str, Any]:
    candidates = []
    for window in windows:
        raw_passthrough = bool(getattr(window, "isFinalPartial", False) or getattr(window, "extractionSkipped", False))
        if mode == "window_range":
            candidates.append(window)
        elif raw_passthrough or str(getattr(window, "checkpointKind", "") or "") == "raw_final":
            candidates.append(window)
        elif mode == "final_raw" and not _checkpoint_owned_sequences([window]):
            candidates.append(window)
    candidates.sort(
        key=lambda window: (
            as_sequence_number(getattr(window, "sequenceStart", 0), 0),
            int(getattr(window, "windowIndex", 0) or 0),
        )
    )
    texts: list[str] = []
    source = None
    useful_set = set(useful_sequences)
    for window in candidates:
        text = str(getattr(window, "text", "") or "").strip()
        parsed = parsed_semantic_sequences(text)
        if not parsed:
            continue
        if useful_set and not (set(parsed) & useful_set):
            continue
        texts.append(text)
        source = source or window
    return "\n".join(texts).strip(), source


def _covering_window(windows: list[Any], start: int, end: int) -> Any:
    for window in windows:
        window_start = as_sequence_number(getattr(window, "sequenceStart", 0), 0)
        window_end = as_sequence_number(getattr(window, "sequenceEnd", 0), 0)
        if window_start <= start and window_end >= end:
            return window
    return windows[0] if windows else None


def _same_conversation(chunk_conversation_id: Any, conversation_id: str) -> bool:
    if not conversation_id:
        return True
    return str(chunk_conversation_id) == str(conversation_id)
