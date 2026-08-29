"""Single semantic path for short and long meetings."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any

from services.conversation.fingerprints import note_fingerprint, task_fingerprint
from services.conversation.meeting_pipeline.consolidator import GlobalArtifactConsolidator, recover_unpublished_notes
from services.conversation.meeting_pipeline.dates import normalize_supported_due_date
from services.conversation.meeting_pipeline.extractor import MeetingCandidateExtractor
from services.conversation.meeting_pipeline.flags import max_extraction_concurrency
from services.conversation.meeting_pipeline.llm import bind_usage, reset_usage
from services.conversation.meeting_pipeline.invariants import apply_invariant_gate
from services.conversation.meeting_pipeline.ledger import CandidateLedger
from services.conversation.meeting_pipeline.observability import log_artifact, log_pipeline
from services.conversation.meeting_pipeline.schemas import (
    MeetingCandidate,
    MeetingPipelineResult,
    VerifiedArtifact,
    VerifierVerdict,
)
from services.conversation.meeting_pipeline.verifier import ArtifactEvidenceVerifier
from services.conversation.meeting_pipeline.windows import (
    build_extraction_windows,
    covered_sequence_ids,
    turns_from_chunks,
)
from services.conversation.models import EvidenceSpan, ExtractedNote, ExtractedTask, WindowExtractionResult
from services.llm.async_runtime import reraise_if_hard_runtime
from services.llm.router import LLMRouter


PIPELINE_VERSION = "meeting-extract-consolidate-verify-v1"
_MAX_WINDOW_ATTEMPTS = 3
# API/schema compatibility only. Persistence uses verificationVerdict == SUPPORTED.
COMPAT_CONFIDENCE = 0.5


async def run_meeting_pipeline(
    chunks,
    conversation_id: str,
    user_id: str,
    space_id: str,
    *,
    router: LLMRouter,
    extractor: MeetingCandidateExtractor | None = None,
    consolidator: GlobalArtifactConsolidator | None = None,
    verifier: ArtifactEvidenceVerifier | None = None,
    meeting_at: datetime | None = None,
) -> MeetingPipelineResult:
    started = time.perf_counter()
    usage: list[dict[str, Any]] = []
    usage_token = bind_usage(usage)
    turns = turns_from_chunks(chunks)
    sequence_text = {turn.sequence_id: turn.raw_text for turn in turns if (turn.raw_text or "").strip()}
    windows = build_extraction_windows(turns, conversation_id=conversation_id)
    extractor = extractor or MeetingCandidateExtractor(router)
    consolidator = consolidator or GlobalArtifactConsolidator(router)
    verifier = verifier or ArtifactEvidenceVerifier(router)
    ledger = CandidateLedger()
    retries = 0
    window_counts: list[int] = []
    window_records: list[dict[str, Any]] = []
    useful_zero = 0
    max_inflight = 0
    try:
        if windows:
            results = await _extract_windows(
                extractor,
                windows,
                conversation_id,
            )
            retries = results["retries"]
            max_inflight = int(results.get("maxInflight") or 0)
            for window, extracted in zip(windows, results["candidates"]):
                ledger.replace_window(window, extracted)
                window_counts.append(len(extracted))
                record = dict((getattr(extractor, "window_records_by_id", {}) or {}).get(window.window_id) or {})
                if not record:
                    record = {
                        "windowId": window.window_id,
                        "sequenceStart": window.sequence_start,
                        "sequenceEnd": window.sequence_end,
                        "sequenceIds": list(window.sequence_ids),
                        "ownedSequenceIds": list(window.owned_sequence_ids),
                        "usefulTokenCount": window.token_count,
                        "inputCharacterCount": len(window.text or ""),
                        "candidateCount": len(extracted),
                        "promptVersion": "meeting-candidate-extractor-v1",
                    }
                record["retryCount"] = retries
                window_records.append(record)
                if window.owned_sequence_ids and not extracted:
                    useful_zero += 1
                    log_pipeline(
                        {
                            "event": "useful_window_zero_candidates",
                            "windowId": window.window_id,
                            "sequenceStart": window.sequence_start,
                            "sequenceEnd": window.sequence_end,
                            "ownedSequenceIds": list(window.owned_sequence_ids),
                            "candidateCount": 0,
                            "rawCandidateCount": record.get("rawCandidateCount"),
                            "provider": record.get("provider"),
                            "model": record.get("model"),
                            "finishReason": record.get("finishReason"),
                            "schemaError": record.get("schemaError"),
                        }
                    )

        claims, summary, topics = await consolidator.consolidate(ledger, sequence_text)
        claims, recovered_notes = recover_unpublished_notes(claims, ledger, set(sequence_text))
        verified = await verifier.verify(claims, sequence_text, meeting_at=meeting_at) if claims else []
        accepted, rejected = apply_invariant_gate(
            verified,
            ledger=ledger,
            sequence_text=sequence_text,
            conversation_id=conversation_id,
        )
        tasks = [
            _to_task(item, conversation_id, space_id, sequence_text, meeting_at)
            for item in accepted
            if item.kind == "task"
        ]
        notes = [
            _to_note(item, conversation_id, space_id, sequence_text)
            for item in accepted
            if item.kind == "note"
        ]
        duration_ms = int((time.perf_counter() - started) * 1000)
        provider = consolidator.last_provider or extractor.last_provider
        model = consolidator.last_model or extractor.last_model
        usage_records = list(usage)
        input_tokens = sum(int(item.get("inputTokens") or 0) for item in usage_records)
        output_tokens = sum(int(item.get("outputTokens") or 0) for item in usage_records)
        observability = {
            "session_id": conversation_id,
            "window_count": len(windows),
            "window_sequence_ranges": [
                {"windowId": window.window_id, "start": window.sequence_start, "end": window.sequence_end, "owned": window.owned_sequence_ids}
                for window in windows
            ],
            "extractor_candidate_count_per_window": window_counts,
            "extraction_window_records": window_records,
            "emptyCandidateWindowRate": (useful_zero / len(windows)) if windows else 0.0,
            "usefulWindowsWithZeroCandidates": useful_zero,
            "total_candidate_count": len(ledger.candidates),
            "consolidated_task_count": sum(1 for item in claims if item.kind == "task"),
            "consolidated_note_count": sum(1 for item in claims if item.kind == "note"),
            "recovered_note_count": recovered_notes,
            "verified_supported_count": sum(1 for item in verified if item.verdict == VerifierVerdict.SUPPORTED),
            "verified_partial_count": verifier.partial,
            "verified_unsupported_count": sum(1 for item in verified if item.verdict == VerifierVerdict.UNSUPPORTED),
            "rejected_artifact_count": len(rejected),
            "provider": provider,
            "model": model,
            "extractor_provider": extractor.last_provider,
            "extractor_model": extractor.last_model,
            "consolidator_provider": consolidator.last_provider,
            "consolidator_model": consolidator.last_model,
            "verifier_provider": verifier.last_provider,
            "verifier_model": verifier.last_model,
            "retry_count": retries,
            "max_extraction_inflight": max_inflight,
            "processing_duration_ms": duration_ms,
            "extractor_calls": extractor.calls,
            "consolidator_calls": consolidator.calls,
            "verifier_calls": verifier.calls,
            "repair_calls": getattr(verifier, "repair_calls", 0),
            "model_calls": extractor.calls + consolidator.calls + verifier.calls,
            "transcript_sequence_count": len(sequence_text),
            "consolidator_cited_sequence_count": len(
                {sequence for item in ledger.candidates for sequence in item.evidenceSequences}
            ),
            "ledger_size": len(ledger.candidates),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "covered_sequences": covered_sequence_ids(windows),
            "pipelineVersion": PIPELINE_VERSION,
        }
        log_pipeline(
            {
                **{key: value for key, value in observability.items() if key != "extraction_window_records"},
                "extraction_window_summaries": [
                    {
                        "windowId": record.get("windowId"),
                        "sequenceStart": record.get("sequenceStart"),
                        "sequenceEnd": record.get("sequenceEnd"),
                        "candidateCount": record.get("candidateCount"),
                        "rawCandidateCount": record.get("rawCandidateCount"),
                        "finishReason": record.get("finishReason"),
                        "schemaError": record.get("schemaError"),
                        "usefulTokenCount": record.get("usefulTokenCount"),
                    }
                    for record in window_records
                ],
            }
        )
        for item in [*accepted, *rejected]:
            log_artifact(
                {
                    "artifact_id": item.artifactKey,
                    "kind": item.kind,
                    "sourceCandidateIds": item.sourceCandidateIds,
                    "evidenceSequences": item.evidenceSequences,
                    "verifier_verdict": item.verdict.value,
                    "reason": item.reason,
                    "owner": item.owner,
                    "dueDate": item.dueDate,
                }
            )
        diagnostics = {
            "pipeline": PIPELINE_VERSION,
            "pipelineVersion": PIPELINE_VERSION,
            "artifactPipelineVersion": PIPELINE_VERSION,
            "pipelineMode": "meeting_pipeline",
            "eventSchemaVersion": "meeting-candidate-v1",
            "promptVersion": "meeting-candidate-extractor-v1,meeting-artifact-consolidator-v1,meeting-evidence-verifier-v1",
            "finalSynthesisInvoked": bool(ledger.candidates),
            "finalSynthesisVerdict": "PUBLISH" if tasks or notes else "NO_PUBLISHABLE_ARTIFACTS",
            "qualityAcceptedTaskCount": len(tasks),
            "qualityAcceptedNoteCount": len(notes),
            "validatedSemanticUnitCount": len(ledger.candidates),
            **observability,
        }
        return MeetingPipelineResult(
            tasks=tasks,
            notes=notes,
            candidates=ledger.candidates,
            windows=windows,
            claims=claims,
            verified=accepted,
            rejected=rejected,
            summary=summary,
            topics=topics,
            provider=str(provider),
            model=str(model),
            diagnostics=diagnostics,
            observability=observability,
            usage=usage_records,
            ledgerPayload=ledger.compact_payload(),
        )
    finally:
        reset_usage(usage_token)


def to_window_result(result: MeetingPipelineResult) -> WindowExtractionResult:
    return WindowExtractionResult(
        summary=result.summary,
        topics=result.topics,
        importantFacts=[item.meaning for item in result.candidates[:20]],
        tasks=result.tasks,
        notes=result.notes,
        extractionDiagnostics=result.diagnostics,
    )


async def _extract_windows(
    extractor: MeetingCandidateExtractor,
    windows,
    conversation_id: str,
) -> dict[str, Any]:
    limit = max_extraction_concurrency()
    semaphore = asyncio.Semaphore(limit)
    inflight = 0
    max_inflight = 0
    lock = asyncio.Lock()
    candidates: list[list[MeetingCandidate] | None] = [None] * len(windows)
    errors: list[Exception | None] = [None] * len(windows)
    retries = 0

    async def run_one(index: int) -> None:
        nonlocal inflight, max_inflight
        async with semaphore:
            async with lock:
                inflight += 1
                max_inflight = max(max_inflight, inflight)
            try:
                candidates[index] = await extractor.extract(windows[index], conversation_id=conversation_id)
                errors[index] = None
            except Exception as error:
                reraise_if_hard_runtime(error)
                errors[index] = error
            finally:
                async with lock:
                    inflight -= 1

    pending = list(range(len(windows)))
    attempts = 0
    while pending:
        await asyncio.gather(*(run_one(index) for index in pending))
        pending = [index for index in pending if candidates[index] is None]
        if not pending:
            break
        attempts += 1
        retries += len(pending)
        if attempts >= _MAX_WINDOW_ATTEMPTS:
            raise pending_error(errors, pending)
    return {"candidates": candidates, "retries": retries, "maxInflight": max_inflight}


def pending_error(errors: list[Exception | None], pending: list[int]) -> Exception:
    for index in pending:
        if errors[index] is not None:
            return errors[index]  # type: ignore[return-value]
    return RuntimeError("meeting extractor failed after retries")


def _to_task(
    item: VerifiedArtifact,
    conversation_id: str,
    space_id: str,
    sequence_text: dict[int, str],
    meeting_at: datetime | None = None,
) -> ExtractedTask:
    evidence = _evidence_spans(item.evidenceSequences, sequence_text)
    resolved = normalize_supported_due_date(item.dueDate, meeting_at)
    task = ExtractedTask(
        title=item.title,
        body=item.body,
        operation="CREATE",
        ownerText=item.owner,
        dueDateText=item.dueDate,
        dueDateResolved=resolved,
        dueDateStatus="resolved" if resolved else ("ambiguous" if item.dueDate else "none"),
        confidence=COMPAT_CONFIDENCE,
        sourceConversationId=conversation_id,
        evidence=evidence,
        origin="explicit",
        changes={
            "sourceCandidateIds": list(item.sourceCandidateIds),
            "evidenceVerified": item.verdict == VerifierVerdict.SUPPORTED,
            "verificationVerdict": item.verdict.value,
            "verifierReason": item.reason,
            "fieldSupport": item.fieldSupport.model_dump() if item.fieldSupport else {},
            "pipelineMode": "meeting_pipeline",
            "artifactPipelineVersion": PIPELINE_VERSION,
            "synthesisSource": "llm",
        },
    )
    task.fingerprint = task_fingerprint(space_id, task)
    return task


def _to_note(item: VerifiedArtifact, conversation_id: str, space_id: str, sequence_text: dict[int, str]) -> ExtractedNote:
    evidence = _evidence_spans(item.evidenceSequences, sequence_text)
    note = ExtractedNote(
        title=item.title,
        body=item.body,
        confidence=COMPAT_CONFIDENCE,
        sourceConversationId=conversation_id,
        evidence=evidence,
        debug={
            "sourceCandidateIds": list(item.sourceCandidateIds),
            "evidenceVerified": item.verdict == VerifierVerdict.SUPPORTED,
            "verificationVerdict": item.verdict.value,
            "verifierReason": item.reason,
            "fieldSupport": item.fieldSupport.model_dump() if item.fieldSupport else {},
            "pipelineMode": "meeting_pipeline",
            "artifactPipelineVersion": PIPELINE_VERSION,
            "synthesisSource": "llm",
        },
    )
    note.fingerprint = note_fingerprint(space_id, note)
    return note


def _evidence_spans(sequences: list[int], sequence_text: dict[int, str]) -> list[EvidenceSpan]:
    spans: list[EvidenceSpan] = []
    for sequence in sequences:
        text = (sequence_text.get(sequence) or "").strip() or f"sequence {sequence}"
        spans.append(EvidenceSpan(sequenceStart=sequence, sequenceEnd=sequence, text=text))
    return spans
