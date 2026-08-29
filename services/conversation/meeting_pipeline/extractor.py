"""Recall-first meeting candidate extraction. LLM semantics; Python provenance."""

from __future__ import annotations

from typing import Any

from services.conversation.event_pipeline.textutil import stable_id
from services.conversation.meeting_pipeline.llm import generate_structured
from services.conversation.meeting_pipeline.schemas import (
    CandidateKind,
    ExtractionWindow,
    MeetingCandidate,
    MeetingCandidateExtractorResponse,
)
from services.llm.router import LLMCapability, LLMRouter

_PROMPT_VERSION = "meeting-candidate-extractor-v1"


class MeetingCandidateExtractor:
    def __init__(self, router: LLMRouter):
        self.router = router
        self.calls = 0
        self.last_provider = "none"
        self.last_model = "none"
        self.window_records_by_id: dict[str, dict[str, Any]] = {}

    async def extract(
        self,
        window: ExtractionWindow,
        *,
        conversation_id: str,
    ) -> list[MeetingCandidate]:
        self.calls += 1
        allowed = set(window.sequence_ids)
        payload = {
            "windowId": window.window_id,
            "windowIndex": window.window_index,
            "sequenceStart": window.sequence_start,
            "sequenceEnd": window.sequence_end,
            "transcript": window.text,
        }
        response, provider, model = await generate_structured(
            self.router,
            LLMCapability.SEMANTIC_EXTRACTION,
            _PROMPT_VERSION,
            MeetingCandidateExtractorResponse,
            payload,
            stage="extractor",
            meta={"windowId": window.window_id},
        )
        self.last_provider = str(getattr(provider, "name", None) or provider or "unknown")
        self.last_model = str(model or "unknown")
        diagnostics = getattr(provider, "last_structured_diagnostics", None) or {}
        raw_items = list(response.candidates or [])
        candidates: list[MeetingCandidate] = []
        dropped_empty_meaning = 0
        dropped_no_evidence = 0
        seen: set[str] = set()
        parsed: list[dict[str, Any]] = []
        for index, item in enumerate(raw_items):
            meaning = " ".join(str(item.meaning or "").split())
            evidence = _exact_sequences(item.evidenceSequences, allowed)
            parsed.append(
                {
                    "kind": item.kind.value if hasattr(item.kind, "value") else str(item.kind),
                    "meaning": meaning,
                    "evidenceSequences": list(item.evidenceSequences or []),
                    "keptEvidence": evidence,
                }
            )
            if not meaning:
                dropped_empty_meaning += 1
                continue
            if not evidence:
                dropped_no_evidence += 1
                continue
            kind = item.kind if isinstance(item.kind, CandidateKind) else CandidateKind.FACT
            owner = _optional_text(item.owner)
            due = _optional_text(item.dueDate)
            candidate_id = stable_id(
                "c",
                conversation_id,
                window.window_id,
                kind.value,
                meaning,
                ",".join(str(sequence) for sequence in evidence),
                index,
            )
            if candidate_id in seen:
                continue
            seen.add(candidate_id)
            candidates.append(
                MeetingCandidate(
                    candidateId=candidate_id,
                    kind=kind,
                    meaning=meaning,
                    evidenceSequences=evidence,
                    owner=owner,
                    dueDate=due,
                    sourceWindowId=window.window_id,
                    sourceWindowIndex=window.window_index,
                )
            )
        record = {
            "windowId": window.window_id,
            "sequenceStart": window.sequence_start,
            "sequenceEnd": window.sequence_end,
            "sequenceIds": list(window.sequence_ids),
            "ownedSequenceIds": list(window.owned_sequence_ids),
            "usefulTokenCount": window.token_count,
            "inputCharacterCount": len(window.text or ""),
            "provider": self.last_provider,
            "model": self.last_model,
            "promptVersion": _PROMPT_VERSION,
            "inputTranscript": window.text,
            "rawModelResponse": diagnostics.get("rawContent"),
            "parsedResponse": parsed,
            "candidateCount": len(candidates),
            "rawCandidateCount": len(raw_items),
            "droppedEmptyMeaning": dropped_empty_meaning,
            "droppedNoEvidence": dropped_no_evidence,
            "failureClass": _window_failure_class(
                raw_count=len(raw_items),
                kept=len(candidates),
                dropped_no_evidence=dropped_no_evidence,
                diagnostics=diagnostics,
            ),
            "finishReason": diagnostics.get("finishReason"),
            "inputTokens": diagnostics.get("promptTokens") or diagnostics.get("inputTokens"),
            "outputTokens": diagnostics.get("completionTokens") or diagnostics.get("outputTokens"),
            "latencyMs": diagnostics.get("latencyMs"),
            "retryCount": 0,
            "fallbackProvider": diagnostics.get("provider") if diagnostics.get("fallback") else None,
            "fallbackModel": diagnostics.get("model") if diagnostics.get("fallback") else None,
            "schemaError": diagnostics.get("retryReason") if diagnostics.get("structuredOutputSuccess") is False else None,
            "parsingOutcome": diagnostics.get("parsingOutcome"),
        }
        self.window_records_by_id[window.window_id] = record
        return candidates


def _window_failure_class(
    *,
    raw_count: int,
    kept: int,
    dropped_no_evidence: int,
    diagnostics: dict[str, Any],
) -> str | None:
    finish = str(diagnostics.get("finishReason") or "").lower()
    if diagnostics.get("structuredOutputSuccess") is False:
        return "STRUCTURED_OUTPUT_PARSE_LOSS"
    if finish in {"length", "max_tokens", "truncated"}:
        return "OUTPUT_TRUNCATION"
    if diagnostics.get("fallback"):
        return "PROVIDER_RETRY_LOSS"
    if raw_count > 0 and kept == 0 and dropped_no_evidence:
        return "STRUCTURED_OUTPUT_PARSE_LOSS"
    if kept == 0:
        return "MODEL_EXTRACTION_MISS"
    return None


def _exact_sequences(values, allowed: set[int]) -> list[int]:
    sequences: list[int] = []
    seen: set[int] = set()
    for value in values or []:
        try:
            sequence = int(value)
        except (TypeError, ValueError):
            continue
        if sequence not in allowed or sequence in seen:
            continue
        seen.add(sequence)
        sequences.append(sequence)
    return sequences


def _optional_text(value: str | None) -> str | None:
    text = " ".join(str(value or "").split())
    return text or None
