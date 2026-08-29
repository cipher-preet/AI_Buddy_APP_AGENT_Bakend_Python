from __future__ import annotations

import json
import re
from contextvars import ContextVar
from types import SimpleNamespace
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from apps.api_gateway.config.setting import settings
from services.conversation.fingerprints import note_fingerprint, task_fingerprint
from services.conversation.extraction_contract import (
    LAST_EXTRACTION_PARSE_TRACE,
    classify_extraction_outcome,
    drop_stage_for,
    empty_parse_trace,
    hydrate_and_validate_unit_evidence,
    hydrate_synthesized_artifacts,
    alias_extraction_payload,
    alias_synthesis_payload,
    alias_quality_review_payload,
    coerce_extraction_lists,
    suspicious_empty_retry_instruction,
    upstream_has_grounded_evidence,
)
from services.conversation.task_coverage import (
    TASK_COVERAGE_CONFLICT,
    annotate_semantic_units,
    coverage_repair_payload,
    evaluate_task_coverage,
    merge_uncovered_action_units,
)
from services.llm.schema_adapter import INCOMPLETE_STRUCTURED_OUTPUT, STRING_LIST_FIELDS
from services.llm.structured_output import (
    drop_stage_for_structured_outcome,
    structured_outcome_from_error,
)
from services.llm.fallback import FallbackLLMProvider, resolved_provider_model, resolved_provider_name
from services.conversation.intelligence import score_and_filter_result
from services.conversation.models import (
    ArtifactReconcileResponse,
    ConversationSummaryDocument,
    CoverageReport,
    EvidenceSpan,
    ExtractionOutcome,
    SemanticUnit,
    WindowExtractionResult,
    ExtractedDecision,
    ExtractedIssue,
    ExtractedNote,
    ExtractedTask,
    Operation,
    SectionExtractionResult,
    Segment,
    SpaceMemoryDocument,
)
from services.conversation.semantic_reconstruction import (
    reconstruct_window_intelligence,
)
from services.conversation.semantic_input import (
    SEMANTIC_INPUT_ASSEMBLY_FAILED,
    empty_semantic_input_diagnostics,
    parsed_semantic_sequences,
    semantic_input_assembly_failed,
)
from services.llm.models import LLMMessage, StructuredLLMRequest
from services.llm.router import LLMCapability, LLMRouter
from services.llm.async_runtime import is_async_lifecycle_error
from services.prompts.loader import load_prompt


_SEMANTIC_CLASSIFICATION_CACHE: dict[str, list[dict[str, Any]]] = {}


class TaskExtractionResponse(BaseModel):
    tasks: list[ExtractedTask] = Field(default_factory=list)


class NoteExtractionResponse(BaseModel):
    notes: list[ExtractedNote] = Field(default_factory=list)


class RepairedTask(BaseModel):
    title: str
    body: str = ""
    operation: Operation
    existingTaskId: str | None = None
    ownerText: str | None = None
    ownerUserId: str | None = None
    dueDateText: str | None = None
    dueDateResolved: str | None = None
    dueDateStatus: Literal["resolved", "ambiguous", "none"] = "none"
    confidence: float = Field(ge=0, le=1)
    needsConfirmation: bool = False
    fingerprint: str | None = None
    changes: dict[str, Any] = Field(default_factory=dict)
    semanticArtifactKey: str = ""
    quality: dict[str, Any] = Field(default_factory=dict)
    semanticConflict: bool = False
    semanticSpeculation: bool = False
    evidence: list[EvidenceSpan]
    origin: Literal["explicit", "strongly_inferred", "unknown"] = "unknown"
    sourceSemanticUnitIds: list[str] = Field(default_factory=list)


class RepairedNote(BaseModel):
    title: str
    body: str
    confidence: float = Field(ge=0, le=1)
    fingerprint: str | None = None
    semanticArtifactKey: str = ""
    quality: dict[str, Any] = Field(default_factory=dict)
    semanticConflict: bool = False
    evidence: list[EvidenceSpan]
    sourceSemanticUnitIds: list[str] = Field(default_factory=list)


class RepairedDecision(BaseModel):
    title: str = Field(description="Short decision title. Do not use description.")
    status: Literal["confirmed_decision", "proposal", "idea", "unresolved_discussion"] = Field(
        description="confirmed_decision, proposal, idea, or unresolved_discussion"
    )
    confidence: float = Field(ge=0, le=1, description="Confidence between 0 and 1")
    evidence: list[EvidenceSpan]


class RepairedIssue(BaseModel):
    title: str = Field(description="Short issue title. Do not use description.")
    kind: Literal["blocker", "risk", "open_question", "missing_information"] = Field(
        description="blocker, risk, open_question, or missing_information"
    )
    confidence: float = Field(ge=0, le=1, description="Confidence between 0 and 1")
    evidence: list[EvidenceSpan]


class MissingItemRepairLLMResponse(BaseModel):
    tasks: list[RepairedTask] = Field(default_factory=list)
    notes: list[RepairedNote] = Field(default_factory=list)


class MissingItemRepairResponse(BaseModel):
    tasks: list[ExtractedTask] = Field(default_factory=list)
    notes: list[ExtractedNote] = Field(default_factory=list)


class WindowExtractionLLMResponse(BaseModel):
    summary: str = ""
    narrative: str = ""
    topics: list[str] = Field(default_factory=list)
    importantFacts: list[str] = Field(default_factory=list)
    semanticUnits: list[SemanticUnit] = Field(default_factory=list)
    tasks: list[RepairedTask] = Field(default_factory=list)
    notes: list[RepairedNote] = Field(default_factory=list)
    decisions: list[RepairedDecision] = Field(default_factory=list)
    issues: list[RepairedIssue] = Field(default_factory=list)
    openQuestions: list[str] = Field(default_factory=list)
    understanding: dict[str, Any] = Field(default_factory=dict)
    rejectedCandidates: list[dict[str, Any]] = Field(default_factory=list)
    supportedUnitVerdict: Literal["has_supported_units", "no_supported_units"] | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_payload(cls, value):
        payload = alias_extraction_payload(value)
        if not isinstance(payload, dict):
            return payload
        payload = coerce_extraction_lists(
            payload, SemanticUnit, RepairedTask, RepairedNote, RepairedDecision, RepairedIssue
        )
        for field_name in ("semanticUnits", "tasks", "notes", "decisions", "issues"):
            payload[field_name] = [
                item.model_dump() if hasattr(item, "model_dump") else item
                for item in payload.get(field_name) or []
            ]
        return payload


LAST_FINAL_SYNTHESIS_PARSE_TRACE: ContextVar[dict[str, Any] | None] = ContextVar(
    "last_final_synthesis_parse_trace",
    default=None,
)


class FinalSynthesisLLMResponse(BaseModel):
    summary: str = ""
    narrative: str = ""
    topics: list[str] = Field(default_factory=list)
    importantFacts: list[str] = Field(default_factory=list)
    tasks: list[RepairedTask] = Field(default_factory=list)
    notes: list[RepairedNote] = Field(default_factory=list)
    decisions: list[RepairedDecision] = Field(default_factory=list)
    issues: list[RepairedIssue] = Field(default_factory=list)
    openQuestions: list[str] = Field(default_factory=list)
    publishVerdict: Literal["PUBLISH", "NO_PUBLISHABLE_ARTIFACTS"] | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_payload(cls, value):
        if not isinstance(value, dict):
            return value
        raw_tasks = value.get("tasks")
        raw_notes = value.get("notes")
        LAST_FINAL_SYNTHESIS_PARSE_TRACE.set(
            {
                "rawResponseKeys": sorted(str(key) for key in value.keys()),
                "rawTaskCount": len(raw_tasks) if isinstance(raw_tasks, list) else 0,
                "rawNoteCount": len(raw_notes) if isinstance(raw_notes, list) else 0,
                "publishVerdict": value.get("publishVerdict"),
            }
        )
        if value and ("tasks" not in value or "notes" not in value or raw_tasks is None or raw_notes is None):
            raise ValueError(INCOMPLETE_STRUCTURED_OUTPUT)
        payload = alias_synthesis_payload(value)
        if isinstance(payload.get("tasks"), list):
            payload["tasks"] = [
                item.model_dump() if hasattr(item, "model_dump") else item
                for item in payload.get("tasks") or []
            ]
        if isinstance(payload.get("notes"), list):
            payload["notes"] = [
                item.model_dump() if hasattr(item, "model_dump") else item
                for item in payload.get("notes") or []
            ]
        return payload


class FinalSynthesisError(RuntimeError):
    def __init__(self, verdict: str, message: str):
        self.verdict = verdict
        super().__init__(message)


class PersistenceFailedError(RuntimeError):
    def __init__(self, message: str = "PERSISTENCE_FAILED"):
        self.verdict = "PERSISTENCE_FAILED"
        super().__init__(message)


class ConversationUnderstandingResponse(BaseModel):
    topics: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    problems: list[str] = Field(default_factory=list)
    solutions: list[str] = Field(default_factory=list)
    commitments: list[str] = Field(default_factory=list)
    requests: list[str] = Field(default_factory=list)
    followUps: list[str] = Field(default_factory=list)
    deadlines: list[str] = Field(default_factory=list)
    owners: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    importantFacts: list[str] = Field(default_factory=list)
    ideas: list[str] = Field(default_factory=list)
    unresolvedQuestions: list[str] = Field(default_factory=list)
    nextSteps: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _reject_object_shaped_strings(cls, value):
        if not isinstance(value, dict):
            return value
        for field_name in STRING_LIST_FIELDS:
            items = value.get(field_name)
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict):
                    raise ValueError("SCHEMA_VALIDATION_FAILED")
        return value


class ExtractionQualityDecision(BaseModel):
    kind: Literal["task", "note"]
    index: int = Field(ge=0)
    keep: bool
    reason: str
    revisedBody: str | None = None
    quality: dict[str, Any] = Field(default_factory=dict)


class ExtractionQualityReviewResponse(BaseModel):
    decisions: list[ExtractionQualityDecision] = Field(default_factory=list)
    missingActionable: list[str] = Field(default_factory=list)
    missingNotes: list[str] = Field(default_factory=list)
    failed: bool = False

    @model_validator(mode="before")
    @classmethod
    def _normalize_payload(cls, value):
        payload = alias_quality_review_payload(value)
        if not isinstance(payload, dict):
            return payload
        payload = coerce_extraction_lists(
            payload,
            unit_cls=None,
            decision_cls=ExtractionQualityDecision,
            update_trace=False,
        )
        payload["decisions"] = [
            item.model_dump() if hasattr(item, "model_dump") else item
            for item in payload.get("decisions") or []
        ]
        return payload


class DecisionExtractionResponse(BaseModel):
    decisions: list[ExtractedDecision] = Field(default_factory=list)


class IssueExtractionResponse(BaseModel):
    issues: list[ExtractedIssue] = Field(default_factory=list)


class SummaryResponse(BaseModel):
    summary: str
    topics: list[str] = Field(default_factory=list)
    importantFacts: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    openQuestions: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)


class MemoryUpdateResponse(BaseModel):
    currentSummary: str
    importantFacts: list[str] = Field(default_factory=list)
    importantDecisions: list[str] = Field(default_factory=list)


class SemanticRoleUnit(BaseModel):
    roles: list[Literal["fact", "claim", "explanation", "decision", "action", "commitment", "request", "question", "answer", "problem", "solution", "requirement", "instruction", "definition", "example", "important_point", "disagreement", "conclusion", "follow_up", "deadline", "assignment", "reference", "unresolved"]] = Field(default_factory=list)
    topic: str = ""
    # A model-created, opaque relationship key. It lets orchestration group
    # paraphrases and code-switched turns without inventing lexical rules.
    threadKey: str = ""
    normalizedMeaning: str = ""
    evidenceIds: list[int] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    uncertain: bool = False


class SemanticRoleClassificationResponse(BaseModel):
    units: list[SemanticRoleUnit] = Field(default_factory=list)


async def extract_segment(
    router: LLMRouter,
    segment: Segment,
    context: dict[str, Any],
    user_id: str,
    space_id: str,
) -> SectionExtractionResult:
    # The legacy batch path now uses the same evidence-packet → synthesis
    # pipeline as incremental windows. Separate note/task prompts could not
    # aggregate meaning before writing and made their own lexical assumptions.
    window = SimpleNamespace(
        conversationId=segment.conversationId,
        userId=user_id,
        spaceId=space_id,
        id=segment.segmentId,
        windowIndex=0,
        sequenceStart=segment.sequenceStart,
        sequenceEnd=segment.sequenceEnd,
        text=segment.text,
    )
    result, _, _ = await extract_window(router, window, context, meeting_context=None, mode="final")
    return SectionExtractionResult(
        segmentId=segment.segmentId,
        tasks=result.tasks,
        notes=result.notes,
        decisions=result.decisions,
        issues=result.issues,
    )


async def validate_coverage(
    router: LLMRouter,
    transcript: str,
    outputs: dict[str, Any],
    context: dict[str, Any],
) -> CoverageReport:
    return await _structured(
        router,
        "coverage-validator-v1",
        CoverageReport,
        json.dumps(context, default=str, ensure_ascii=True),
        f"CURRENT CONVERSATION:\n{transcript}\n\nPROPOSED OUTPUTS:\n{json.dumps(outputs, default=str, ensure_ascii=True)}",
        LLMCapability.VALIDATION,
    )


async def extract_window(
    router: LLMRouter,
    window,
    context: dict[str, Any],
    meeting_context: dict[str, Any] | None = None,
    recovery: bool = False,
    mode: str = "checkpoint",
) -> tuple[WindowExtractionResult, str, str]:
    from services.conversation.budget import expected_request_tokens

    estimated_input = expected_request_tokens(str(context), str(meeting_context or {}), window.text)
    provider, model = _route_for_input(router, LLMCapability.HIGH_ACCURACY_REASONING, estimated_input)
    assembly_diagnostics = dict(getattr(window, "semanticInputDiagnostics", None) or empty_semantic_input_diagnostics())
    parsed_input = parsed_semantic_sequences(getattr(window, "text", "") or "")
    if not assembly_diagnostics.get("semanticInputTranscriptCount") and parsed_input:
        assembly_diagnostics["semanticInputTranscriptCount"] = len(parsed_input)
        assembly_diagnostics["usefulSequenceNumbers"] = sorted(parsed_input)
        assembly_diagnostics["semanticInputCharacterCount"] = len(getattr(window, "text", "") or "")
        assembly_diagnostics["semanticInputEstimatedTokens"] = expected_request_tokens(getattr(window, "text", "") or "")
    if semantic_input_assembly_failed(window, window.text):
        failed = WindowExtractionResult(
            extractionOutcome=ExtractionOutcome.SEMANTIC_INPUT_ASSEMBLY_FAILED,
            extractionError=SEMANTIC_INPUT_ASSEMBLY_FAILED,
            isCheckpoint=mode == "checkpoint",
            extractionDiagnostics={
                **assembly_diagnostics,
                "dropStage": "semantic_input_assembly_failed",
                "finalSynthesisInvoked": False,
                "persistenceAttempted": False,
            },
        )
        print(
            "Semantic input assembly failed before understanding:",
            {
                "conversationId": str(window.conversationId),
                "windowIndex": getattr(window, "windowIndex", 0),
                **{key: assembly_diagnostics.get(key) for key in (
                    "persistedTranscriptCount",
                    "persistedNonEmptyTranscriptCount",
                    "usefulSequenceNumbers",
                    "semanticInputTranscriptCount",
                    "semanticInputCharacterCount",
                    "semanticInputEstimatedTokens",
                )},
            },
        )
        return failed, "none", "none"
    prompt_name = "memory-recovery-v1" if recovery else ("semantic-checkpoint-v1" if mode == "checkpoint" else "window-extractor-v1")
    semantic_units = await _classify_window_semantics(provider, model, window.text)
    reconstruction = reconstruct_window_intelligence(window.text, str(window.conversationId), str(window.spaceId), semantic_units)
    understanding, understanding_meta = await understand_window(
        router, window, context, meeting_context, reconstruction.prompt_block()
    )
    extraction_context = _context_with_understanding(
        meeting_context,
        understanding,
        reconstruction.diagnostics,
        reconstruction.prompt_block(),
    )
    current = _window_extraction_prompt_input(window, extraction_context)
    try:
        response, provider, model = await _structured_with_recovery(
            router,
            provider,
            model,
            prompt_name,
            WindowExtractionLLMResponse,
            json.dumps(context, default=str, ensure_ascii=True),
            current,
        )
    except Exception as error:
        outcome = structured_outcome_from_error(error)
        drop_stage = drop_stage_for_structured_outcome(outcome)
        print(
            "Window extraction failed after structured-output recovery and provider fallback:",
            {
                "conversationId": str(window.conversationId),
                "windowIndex": window.windowIndex,
                "provider": resolved_provider_name(provider),
                "model": resolved_provider_model(provider, model) or model,
                "recovery": recovery,
                "structuredOutputOutcome": outcome,
                "schemaEchoDetected": outcome == "STRUCTURED_SCHEMA_ECHO",
                "error": str(error)[:500],
            },
        )
        failed = WindowExtractionResult(
            extractionOutcome=ExtractionOutcome.EXTRACTION_FAILED,
            extractionError=str(error)[:500],
            isCheckpoint=mode == "checkpoint",
            extractionDiagnostics=_extraction_diagnostics(
                reconstruction.diagnostics,
                {
                    "schemaEchoDetected": outcome == "STRUCTURED_SCHEMA_ECHO",
                    "structuredOutputOutcome": outcome,
                    "parsingOutcome": outcome,
                    **(getattr(provider, "last_structured_diagnostics", None) or {}),
                },
                technical_failure=True,
                drop_stage=drop_stage,
            ),
        )
        return failed, resolved_provider_name(provider), resolved_provider_model(provider, model) or model
    if not response.understanding:
        response.understanding = understanding.model_dump()
    result, parse_trace, llm_empty = _materialize_extraction_result(
        response, window, str(window.conversationId), str(window.spaceId), mode
    )
    upstream_evidence = upstream_has_grounded_evidence(reconstruction.diagnostics)
    parse_trace["zeroOutputRecoveryEligible"] = bool(llm_empty and upstream_evidence)
    explicit_empty = _explicit_empty_verdict(response)
    if llm_empty and upstream_evidence and not explicit_empty:
        parse_trace["zeroOutputRecoveryAttempted"] = True
        parse_trace["zeroOutputRecoverySource"] = "suspicious_empty_retry"
        reconstruction.diagnostics["zeroOutputRecoveryEligible"] = True
        try:
            retry_response, provider, model = await _structured_with_recovery(
                router,
                provider,
                model,
                prompt_name,
                WindowExtractionLLMResponse,
                json.dumps(context, default=str, ensure_ascii=True),
                f"{current}\n\n{suspicious_empty_retry_instruction(reconstruction.diagnostics)}",
            )
            if not retry_response.understanding:
                retry_response.understanding = understanding.model_dump()
            retried, retry_trace, _ = _materialize_extraction_result(
                retry_response, window, str(window.conversationId), str(window.spaceId), mode
            )
            parse_trace = {**parse_trace, **{key: retry_trace.get(key) for key in retry_trace if retry_trace.get(key) not in (None, [], 0, "")}}
            parse_trace["zeroOutputRecoveryAttempted"] = True
            parse_trace["zeroOutputRecoverySource"] = "suspicious_empty_retry"
            result = retried
            response = retry_response
            explicit_empty = _explicit_empty_verdict(retry_response)
        except Exception as error:
            print(
                "Suspicious empty extraction recovery failed after retry and fallback:",
                {
                    "conversationId": str(window.conversationId),
                    "windowIndex": window.windowIndex,
                    "error": str(error)[:500],
                },
            )
            result.extractionOutcome = ExtractionOutcome.EXTRACTION_FAILED
            result.extractionError = str(error)[:500]
            parse_trace["dropStage"] = "provider_or_parser_failure"
            result.extractionDiagnostics = _extraction_diagnostics(
                reconstruction.diagnostics, parse_trace, technical_failure=True
            )
            _log_window_intelligence(
                window, understanding_meta, provider, model, reconstruction, result, mode, estimated_input
            )
            return result, resolved_provider_name(provider), resolved_provider_model(provider, model) or model
        if not _has_extracted_units(result) and not explicit_empty:
            fallback_provider, fallback_model = router.route(LLMCapability.FALLBACK)
            if not _same_structured_route(provider, model, fallback_provider, fallback_model):
                try:
                    fallback_response = await _structured_with_provider(
                        fallback_provider,
                        fallback_model,
                        prompt_name,
                        WindowExtractionLLMResponse,
                        json.dumps(context, default=str, ensure_ascii=True),
                        f"{current}\n\n{suspicious_empty_retry_instruction(reconstruction.diagnostics)}",
                    )
                    if not fallback_response.understanding:
                        fallback_response.understanding = understanding.model_dump()
                    result, fallback_trace, _ = _materialize_extraction_result(
                        fallback_response, window, str(window.conversationId), str(window.spaceId), mode
                    )
                    parse_trace = {**parse_trace, **fallback_trace}
                    parse_trace["zeroOutputRecoveryAttempted"] = True
                    parse_trace["zeroOutputRecoverySource"] = "eligible_model_fallback"
                    provider, model = fallback_provider, fallback_model
                    response = fallback_response
                    explicit_empty = _explicit_empty_verdict(fallback_response)
                except Exception as error:
                    print(
                        "Suspicious empty extraction fallback provider failed:",
                        {"conversationId": str(window.conversationId), "error": str(error)[:500]},
                    )
    evidence_rejected = int(parse_trace.get("evidenceRejectedUnitCount") or 0)
    extractor_abstained = explicit_empty and not result.semanticUnits and evidence_rejected == 0
    if not extractor_abstained:
        before = len(result.semanticUnits)
        result.semanticUnits = merge_uncovered_action_units(
            result.semanticUnits, getattr(reconstruction, "threads", None), window.text
        )
        added = len(result.semanticUnits) - before
        if added:
            print(
                "Merged uncovered action units:",
                {"added": added, "validated": len(result.semanticUnits)},
            )
    if mode != "checkpoint" and not recovery and not result.semanticUnits and _needs_window_recovery(result, window.text):
        reconstruction.diagnostics["fallbackTriggered"] = True
        try:
            recovery_response, provider, model = await _structured_with_recovery(
                router,
                provider,
                model,
                "memory-recovery-v1",
                WindowExtractionLLMResponse,
                json.dumps(context, default=str, ensure_ascii=True),
                (
                    f"{current}\n\n"
                    "PREVIOUS EXTRACTION THAT MAY HAVE MISSED MEMORY ITEMS:\n"
                    f"{json.dumps(result.model_dump(), default=str, ensure_ascii=True)}"
                ),
            )
            recovered, _, _ = _materialize_extraction_result(
                recovery_response, window, str(window.conversationId), str(window.spaceId), mode
            )
            result = _merge_window_extraction_results(result, recovered)
        except Exception as error:
            print(
                "Window memory recovery skipped after failure:",
                {
                    "conversationId": str(window.conversationId),
                    "windowIndex": window.windowIndex,
                    "error": str(error)[:500],
                },
            )
    result.extractionOutcome = classify_extraction_outcome(
        has_units=_has_extracted_units(result),
        technical_failure=False,
        upstream_evidence=upstream_evidence,
        explicit_empty_verdict=explicit_empty and not _has_extracted_units(result),
        recovery_attempted=bool(parse_trace.get("zeroOutputRecoveryAttempted")),
        semantic_input_assembly_failed=semantic_input_assembly_failed(window, window.text),
    )
    parse_trace["dropStage"] = drop_stage_for(parse_trace, _has_extracted_units(result))
    structured_diag = getattr(provider, "last_structured_diagnostics", {}) or {}
    parse_trace["finishReason"] = structured_diag.get("finishReason")
    parse_trace["requestedStructuredMode"] = structured_diag.get("requestedStructuredMode")
    parse_trace["actualResponseFormatMode"] = structured_diag.get("actualResponseFormatMode")
    parse_trace["topLevelResponseKeys"] = structured_diag.get("topLevelResponseKeys") or parse_trace.get("rawResponseKeys") or []
    parse_trace["schemaEchoDetected"] = bool(structured_diag.get("schemaEchoDetected") or parse_trace.get("schemaEchoDetected"))
    parse_trace["parsingOutcome"] = structured_diag.get("parsingOutcome") or parse_trace.get("parsingOutcome")
    parse_trace["structuredOutputOutcome"] = parse_trace.get("parsingOutcome")
    result.extractionDiagnostics = _extraction_diagnostics(reconstruction.diagnostics, parse_trace)
    result.extractionDiagnostics.update(assembly_diagnostics)
    result.extractionDiagnostics["validatedSemanticUnitCount"] = len(result.semanticUnits)
    annotate_semantic_units(result.semanticUnits, getattr(reconstruction, "threads", None))
    result.extractionDiagnostics["unitEvidenceOutcomes"] = [
        {
            "semanticKey": unit.semanticKey,
            "kind": unit.kind,
            "evidenceIds": list(unit.evidenceIds or []),
            "evidenceOutcome": (unit.quality or {}).get("evidenceOutcome"),
            "actionable": (unit.quality or {}).get("actionable"),
        }
        for unit in [*result.semanticUnits]
    ]
    if result.extractionDiagnostics["unitEvidenceOutcomes"] or parse_trace.get("evidenceRejectedUnitCount"):
        print("Semantic unit evidence outcomes:", result.extractionDiagnostics["unitEvidenceOutcomes"])
        if parse_trace.get("evidenceRejectedUnits"):
            print("Semantic unit evidence rejected:", parse_trace.get("evidenceRejectedUnits"))
    if mode == "checkpoint":
        result.extractionDiagnostics.update(
            empty_final_synthesis_diagnostics(
                validated_unit_count=len(result.semanticUnits),
                invoked=False,
                verdict="SKIPPED_CHECKPOINT",
            )
        )
        _log_window_intelligence(
            window, understanding_meta, provider, model, reconstruction, result, mode, estimated_input
        )
        return result, resolved_provider_name(provider), resolved_provider_model(provider, model) or model
    if result.extractionOutcome == ExtractionOutcome.SUCCESS and result.semanticUnits:
        synthesized, synth_provider, synth_model, synthesis_diag = await synthesize_final_from_semantic_units(
            router,
            str(window.conversationId),
            str(window.userId),
            str(window.spaceId),
            result.semanticUnits,
            context,
            transcript=window.text,
        )
        result.extractionDiagnostics.update(synthesis_diag)
        if synthesis_diag.get("finalSynthesisVerdict") in {"PROVIDER_FAILED", "MALFORMED_SCHEMA"}:
            _log_window_intelligence(
                window, understanding_meta, synth_provider, synth_model, reconstruction, result, mode, estimated_input
            )
            _log_final_synthesis(result.extractionDiagnostics)
            raise FinalSynthesisError(
                synthesis_diag["finalSynthesisVerdict"],
                f"final synthesis {synthesis_diag['finalSynthesisVerdict']}",
            )
        result = _apply_synthesized_artifacts(result, synthesized)
        result = await apply_final_artifact_quality_gate(
            router,
            result,
            result.semanticUnits,
            window.text,
            context,
            str(window.conversationId),
            str(window.spaceId),
            result.extractionDiagnostics,
        )
        _log_window_intelligence(
            window, understanding_meta, synth_provider, synth_model, reconstruction, result, mode, estimated_input
        )
        _log_final_synthesis(result.extractionDiagnostics)
        return result, resolved_provider_name(synth_provider), resolved_provider_model(synth_provider, synth_model) or synth_model
    result.extractionDiagnostics.update(
        empty_final_synthesis_diagnostics(
            validated_unit_count=len(result.semanticUnits),
            invoked=False,
            verdict=None,
            task_count=len(result.tasks),
            note_count=len(result.notes),
        )
    )
    _log_window_intelligence(
        window, understanding_meta, provider, model, reconstruction, result, mode, estimated_input
    )
    return result, resolved_provider_name(provider), resolved_provider_model(provider, model) or model


async def extract_from_raw_transcript(
    router: LLMRouter,
    conversation_id: str,
    user_id: str,
    space_id: str,
    transcript: str,
    context: dict[str, Any],
    *,
    sequence_start: int | None = None,
    sequence_end: int | None = None,
    window_index: int = 0,
    window_id: str | None = None,
    semantic_input_diagnostics: dict[str, Any] | None = None,
) -> tuple[WindowExtractionResult, str, str]:
    parsed = parsed_semantic_sequences(transcript)
    start = sequence_start if sequence_start is not None else (min(parsed) if parsed else 0)
    end = sequence_end if sequence_end is not None else (max(parsed) if parsed else start)
    window = SimpleNamespace(
        conversationId=conversation_id,
        userId=user_id,
        spaceId=space_id,
        id=window_id or f"{conversation_id}:raw-final",
        windowIndex=window_index,
        sequenceStart=start,
        sequenceEnd=end,
        text=transcript,
        nonEmptyChunkCount=len(parsed),
        semanticInputDiagnostics=semantic_input_diagnostics or {},
    )
    return await extract_window(router, window, context, meeting_context=None, mode="final")


def empty_final_synthesis_diagnostics(
    *,
    validated_unit_count: int = 0,
    invoked: bool = False,
    verdict: str | None = None,
    task_count: int = 0,
    note_count: int = 0,
) -> dict[str, Any]:
    return {
        "validatedSemanticUnitCount": validated_unit_count,
        "finalSynthesisInvoked": invoked,
        "finalSynthesisInputUnitCount": 0,
        "finalSynthesisProvider": None,
        "finalSynthesisModel": None,
        "finalSynthesisRawTaskCount": 0,
        "finalSynthesisRawNoteCount": 0,
        "finalSynthesisParsedTaskCount": 0,
        "finalSynthesisParsedNoteCount": 0,
        "finalSynthesisVerdict": verdict,
        "qualityAcceptedTaskCount": 0,
        "qualityAcceptedNoteCount": 0,
        "qualityArtifactDiagnostics": [],
        "qualityRepairAttempted": False,
        "qualityRepairRound": 0,
        "requiredConfidence": None,
        "persistenceAttempted": False,
        "persistenceOutcome": None,
        "tasksPersistedCount": 0,
        "notesPersistedCount": 0,
        "persistedTaskIds": [],
        "persistedNoteIds": [],
        "taskCountAfterConfidence": task_count,
        "noteCountAfterConfidence": note_count,
    }


def _apply_synthesized_artifacts(
    extraction: WindowExtractionResult,
    synthesized: WindowExtractionResult,
) -> WindowExtractionResult:
    extraction.tasks = list(synthesized.tasks)
    extraction.notes = list(synthesized.notes)
    extraction.decisions = list(synthesized.decisions)
    extraction.issues = list(synthesized.issues)
    extraction.summary = synthesized.summary or extraction.summary
    extraction.narrative = synthesized.narrative or extraction.narrative
    extraction.topics = synthesized.topics or extraction.topics
    extraction.importantFacts = synthesized.importantFacts or extraction.importantFacts
    extraction.openQuestions = synthesized.openQuestions or extraction.openQuestions
    return extraction


def _log_final_synthesis(diagnostics: dict[str, Any]) -> None:
    print(
        "Final synthesis stages completed:",
        {
            "validatedSemanticUnitCount": diagnostics.get("validatedSemanticUnitCount"),
            "finalSynthesisInvoked": diagnostics.get("finalSynthesisInvoked"),
            "finalSynthesisInputUnitCount": diagnostics.get("finalSynthesisInputUnitCount"),
            "finalSynthesisProvider": diagnostics.get("finalSynthesisProvider"),
            "finalSynthesisModel": diagnostics.get("finalSynthesisModel"),
            "finalSynthesisRawTaskCount": diagnostics.get("finalSynthesisRawTaskCount"),
            "finalSynthesisRawNoteCount": diagnostics.get("finalSynthesisRawNoteCount"),
            "finalSynthesisParsedTaskCount": diagnostics.get("finalSynthesisParsedTaskCount"),
            "finalSynthesisParsedNoteCount": diagnostics.get("finalSynthesisParsedNoteCount"),
            "finalSynthesisVerdict": diagnostics.get("finalSynthesisVerdict"),
            "taskCountAfterConfidence": diagnostics.get("taskCountAfterConfidence"),
            "noteCountAfterConfidence": diagnostics.get("noteCountAfterConfidence"),
            "qualityAcceptedTaskCount": diagnostics.get("qualityAcceptedTaskCount"),
            "qualityAcceptedNoteCount": diagnostics.get("qualityAcceptedNoteCount"),
            "qualityRejectedTaskCount": diagnostics.get("qualityRejectedTaskCount"),
            "qualityRejectedNoteCount": diagnostics.get("qualityRejectedNoteCount"),
            "requiredConfidence": diagnostics.get("requiredConfidence"),
            "qualityArtifactDiagnostics": diagnostics.get("qualityArtifactDiagnostics") or [],
            "qualityRepairAttempted": diagnostics.get("qualityRepairAttempted"),
            "qualityRepairRound": diagnostics.get("qualityRepairRound"),
            "persistenceAttempted": diagnostics.get("persistenceAttempted"),
            "persistenceOutcome": diagnostics.get("persistenceOutcome"),
            "tasksPersistedCount": diagnostics.get("tasksPersistedCount"),
            "notesPersistedCount": diagnostics.get("notesPersistedCount"),
            "persistedTaskIds": diagnostics.get("persistedTaskIds"),
            "persistedNoteIds": diagnostics.get("persistedNoteIds"),
        },
    )


async def synthesize_final_from_semantic_units(
    router: LLMRouter,
    conversation_id: str,
    user_id: str,
    space_id: str,
    units: list[SemanticUnit],
    context: dict[str, Any],
    *,
    transcript: str = "",
    checkpoints: list[dict[str, Any]] | None = None,
    artifacts: list[dict[str, Any]] | None = None,
    leftover_raw: str = "",
    meeting_memory: dict[str, Any] | None = None,
) -> tuple[WindowExtractionResult, Any, str, dict[str, Any]]:
    unit_payload = [unit.model_dump() if hasattr(unit, "model_dump") else unit for unit in units]
    diagnostics = empty_final_synthesis_diagnostics(
        validated_unit_count=len(unit_payload),
        invoked=True,
        verdict=None,
    )
    diagnostics["finalSynthesisInputUnitCount"] = len(unit_payload)
    estimated = _rough_token_count(json.dumps(unit_payload, default=str, ensure_ascii=True) + (transcript or leftover_raw))
    provider, model = _route_for_input(router, LLMCapability.FINAL_SYNTHESIS, estimated)
    diagnostics["finalSynthesisProvider"] = resolved_provider_name(provider)
    diagnostics["finalSynthesisModel"] = resolved_provider_model(provider, model) or model
    payload = {
        "path": "long_checkpoint_synthesis" if checkpoints else "short_raw_transcript",
        "conversationId": conversation_id,
        "semanticUnits": unit_payload,
        "semanticCheckpoints": checkpoints or [],
        "artifacts": artifacts or [],
        "meetingMemory": meeting_memory or {},
        "leftoverRawTranscript": leftover_raw,
        "rawTranscript": transcript,
    }
    LAST_FINAL_SYNTHESIS_PARSE_TRACE.set(None)
    try:
        response, provider, model = await _structured_with_recovery(
            router,
            provider,
            model,
            "final-synthesis-v1",
            FinalSynthesisLLMResponse,
            json.dumps(context, default=str, ensure_ascii=True),
            json.dumps(payload, default=str, ensure_ascii=True),
        )
    except ValidationError as error:
        diagnostics["finalSynthesisProvider"] = resolved_provider_name(provider)
        diagnostics["finalSynthesisModel"] = resolved_provider_model(provider, model) or model
        diagnostics["finalSynthesisVerdict"] = "MALFORMED_SCHEMA"
        diagnostics["finalSynthesisError"] = str(error)[:300]
        return WindowExtractionResult(), provider, model, diagnostics
    except Exception as error:
        diagnostics["finalSynthesisProvider"] = resolved_provider_name(provider)
        diagnostics["finalSynthesisModel"] = resolved_provider_model(provider, model) or model
        message = str(error).casefold()
        malformed = isinstance(error, ValidationError) or "validation" in message or "schema" in message
        diagnostics["finalSynthesisVerdict"] = "MALFORMED_SCHEMA" if malformed else "PROVIDER_FAILED"
        diagnostics["finalSynthesisError"] = str(error)[:300]
        return WindowExtractionResult(), provider, model, diagnostics
    diagnostics["finalSynthesisProvider"] = resolved_provider_name(provider)
    diagnostics["finalSynthesisModel"] = resolved_provider_model(provider, model) or model
    route = getattr(provider, "last_structured_route", None) or {}
    if route:
        diagnostics["providerUsed"] = route.get("providerUsed") or diagnostics["finalSynthesisProvider"]
        diagnostics["modelUsed"] = route.get("modelUsed") or diagnostics["finalSynthesisModel"]
        diagnostics["structuredModeUsed"] = route.get("structuredModeUsed")
        diagnostics["attemptCount"] = route.get("attemptCount")
        diagnostics["fallbackDepth"] = route.get("fallbackDepth")
        diagnostics["failureHistory"] = route.get("failureHistory") or []
    parse_trace = LAST_FINAL_SYNTHESIS_PARSE_TRACE.get() or {}
    diagnostics["finalSynthesisRawTaskCount"] = int(parse_trace.get("rawTaskCount") or len(response.tasks or []))
    diagnostics["finalSynthesisRawNoteCount"] = int(parse_trace.get("rawNoteCount") or len(response.notes or []))
    synthesized = _window_result_from_llm(response, conversation_id, space_id)
    diagnostics["finalSynthesisParsedTaskCount"] = len(synthesized.tasks)
    diagnostics["finalSynthesisParsedNoteCount"] = len(synthesized.notes)
    if synthesized.tasks or synthesized.notes:
        diagnostics["finalSynthesisVerdict"] = "PUBLISH"
        if response.publishVerdict == "NO_PUBLISHABLE_ARTIFACTS":
            diagnostics["publishVerdictOverridden"] = True
    elif response.publishVerdict == "NO_PUBLISHABLE_ARTIFACTS" or (
        not synthesized.tasks and not synthesized.notes
    ):
        diagnostics["finalSynthesisVerdict"] = "NO_PUBLISHABLE_ARTIFACTS"
    else:
        diagnostics["finalSynthesisVerdict"] = "PUBLISH"
    return synthesized, provider, model, diagnostics


def _has_extracted_units(result: WindowExtractionResult) -> bool:
    return bool(result.semanticUnits or result.tasks or result.notes or result.decisions or result.issues)


def _explicit_empty_verdict(response: WindowExtractionLLMResponse) -> bool:
    if getattr(response, "supportedUnitVerdict", None) == "no_supported_units":
        return True
    return bool(getattr(response, "rejectedCandidates", None)) and not (
        response.semanticUnits or response.tasks or response.notes or response.decisions or response.issues
    )


def _materialize_extraction_result(
    response: WindowExtractionLLMResponse,
    window,
    conversation_id: str,
    space_id: str,
    mode: str,
) -> tuple[WindowExtractionResult, dict[str, Any], bool]:
    parse_trace = dict(LAST_EXTRACTION_PARSE_TRACE.get() or empty_parse_trace())
    result = _window_result_from_llm(response, conversation_id, space_id)
    result.isCheckpoint = mode == "checkpoint"
    result.narrative = result.narrative or result.summary
    result.semanticUnits, evidence_rejected = hydrate_and_validate_unit_evidence(result.semanticUnits, window.text)
    post_trace = LAST_EXTRACTION_PARSE_TRACE.get() or {}
    parse_trace["evidenceRejectedUnitCount"] = evidence_rejected
    parse_trace["evidenceRejectedUnits"] = list(post_trace.get("evidenceRejectedUnits") or [])
    parse_trace["validatedSemanticUnitCount"] = len(result.semanticUnits)
    parse_trace["unitEvidenceOutcomes"] = [
        {
            "semanticKey": unit.semanticKey,
            "kind": unit.kind,
            "evidenceIds": list(unit.evidenceIds or []),
            "evidenceOutcome": (unit.quality or {}).get("evidenceOutcome"),
            "actionable": (unit.quality or {}).get("actionable"),
        }
        for unit in result.semanticUnits
    ]
    llm_empty = not _has_extracted_units(result)
    if mode != "checkpoint" and not result.semanticUnits:
        quality: dict[str, int] = {}
        result = score_and_filter_result(result, window.text, diagnostics=quality)
        parse_trace["qualityRejectedTaskCount"] = quality.get("qualityRejectedTaskCount", 0)
        parse_trace["qualityRejectedNoteCount"] = quality.get("qualityRejectedNoteCount", 0)
    parse_trace["parsedSemanticUnitCount"] = len(result.semanticUnits)
    parse_trace["supportedUnitVerdict"] = getattr(response, "supportedUnitVerdict", None)
    return result, parse_trace, llm_empty


def _extraction_diagnostics(
    reconstruction_diagnostics: dict[str, Any],
    parse_trace: dict[str, Any] | None = None,
    technical_failure: bool = False,
    drop_stage: str | None = None,
) -> dict[str, Any]:
    trace = {**empty_parse_trace(), **(parse_trace or {})}
    if drop_stage:
        trace["dropStage"] = drop_stage
    if technical_failure:
        trace["dropStage"] = trace.get("dropStage") or "provider_or_parser_failure"
    trace["upstreamTaskCandidates"] = int(reconstruction_diagnostics.get("taskCandidatesGenerated") or 0)
    trace["upstreamNoteCandidates"] = int(reconstruction_diagnostics.get("noteCandidatesGenerated") or 0)
    trace["upstreamDiscussionThreads"] = int(reconstruction_diagnostics.get("discussionThreadCount") or 0)
    return {**reconstruction_diagnostics, **trace}


def _log_window_intelligence(
    window,
    understanding_meta: dict[str, Any],
    provider,
    model: str,
    reconstruction,
    result: WindowExtractionResult,
    mode: str,
    estimated_input: int,
) -> None:
    from services.conversation.budget import safe_input_budget

    provider_name = resolved_provider_name(provider)
    diagnostics = result.extractionDiagnostics or {}
    print(
        "Window intelligence stages completed:",
        {
            "conversationId": str(window.conversationId),
            "windowIndex": window.windowIndex,
            "understandingProvider": understanding_meta.get("provider"),
            "understandingModel": understanding_meta.get("model"),
            "extractionProvider": provider_name,
            "extractionModel": model,
            "usefulChunks": diagnostics.get("usefulChunks")
            or diagnostics.get("usefulSequenceNumbers")
            or reconstruction.diagnostics["usefulChunks"],
            "persistedTranscriptCount": diagnostics.get("persistedTranscriptCount"),
            "persistedNonEmptyTranscriptCount": diagnostics.get("persistedNonEmptyTranscriptCount"),
            "persistedSequenceNumbers": diagnostics.get("persistedSequenceNumbers"),
            "queriedTranscriptCount": diagnostics.get("queriedTranscriptCount"),
            "queriedSequenceNumbers": diagnostics.get("queriedSequenceNumbers"),
            "windowId": diagnostics.get("windowId") or str(getattr(window, "id", "")),
            "sequenceStart": getattr(window, "sequenceStart", diagnostics.get("sequenceStart")),
            "sequenceEnd": getattr(window, "sequenceEnd", diagnostics.get("sequenceEnd")),
            "expectedSequenceCount": diagnostics.get("expectedSequenceCount"),
            "windowTranscriptCountBeforeFiltering": diagnostics.get("windowTranscriptCountBeforeFiltering"),
            "emptyFilteredCount": diagnostics.get("emptyFilteredCount"),
            "unusableFilteredCount": diagnostics.get("unusableFilteredCount"),
            "usefulTranscriptCountAfterFiltering": diagnostics.get("usefulTranscriptCountAfterFiltering"),
            "usefulSequenceNumbers": diagnostics.get("usefulSequenceNumbers"),
            "semanticInputTranscriptCount": diagnostics.get("semanticInputTranscriptCount"),
            "semanticInputCharacterCount": diagnostics.get("semanticInputCharacterCount"),
            "semanticInputEstimatedTokens": diagnostics.get("semanticInputEstimatedTokens"),
            "discussionThreadsCreated": reconstruction.diagnostics["discussionThreadCount"],
            "factsExtracted": len(reconstruction.facts),
            "deterministicCandidatesGenerated": reconstruction.diagnostics["candidatesGenerated"],
            "semanticDiagnostics": reconstruction.diagnostics,
            "semanticUnitCount": len(result.semanticUnits),
            "validatedSemanticUnitCount": diagnostics.get("validatedSemanticUnitCount", len(result.semanticUnits)),
            "isCheckpoint": result.isCheckpoint,
            "extractionOutcome": result.extractionOutcome.value,
            "mode": mode,
            "estimatedInputTokens": estimated_input,
            "providerBudget": safe_input_budget(provider_name, model=model),
            "providerBudgetMeaning": "safe_input_token_budget",
            "structuredOutputMaxTokens": _provider_structured_max_tokens(provider_name, "WindowExtractionLLMResponse"),
            "finishReason": diagnostics.get("finishReason"),
            "rawResponseKeys": diagnostics.get("rawResponseKeys"),
            "topLevelResponseKeys": diagnostics.get("topLevelResponseKeys") or diagnostics.get("rawResponseKeys"),
            "schemaEchoDetected": diagnostics.get("schemaEchoDetected"),
            "requestedStructuredMode": diagnostics.get("requestedStructuredMode"),
            "actualResponseFormatMode": diagnostics.get("actualResponseFormatMode"),
            "parsingOutcome": diagnostics.get("parsingOutcome"),
            "structuredOutputOutcome": diagnostics.get("structuredOutputOutcome"),
            "rawSemanticUnitCount": diagnostics.get("rawSemanticUnitCount"),
            "parsedSemanticUnitCount": diagnostics.get("parsedSemanticUnitCount"),
            "schemaRejectedUnitCount": diagnostics.get("schemaRejectedUnitCount"),
            "evidenceRejectedUnitCount": diagnostics.get("evidenceRejectedUnitCount"),
            "qualityRejectedTaskCount": diagnostics.get("qualityRejectedTaskCount"),
            "qualityRejectedNoteCount": diagnostics.get("qualityRejectedNoteCount"),
            "qualityArtifactDiagnostics": diagnostics.get("qualityArtifactDiagnostics") or [],
            "qualityRepairAttempted": diagnostics.get("qualityRepairAttempted"),
            "requiredConfidence": diagnostics.get("requiredConfidence"),
            "dropStage": diagnostics.get("dropStage"),
            "zeroOutputRecoveryEligible": diagnostics.get("zeroOutputRecoveryEligible"),
            "zeroOutputRecoveryAttempted": diagnostics.get("zeroOutputRecoveryAttempted"),
            "finalSynthesisInvoked": diagnostics.get("finalSynthesisInvoked"),
            "finalSynthesisInputUnitCount": diagnostics.get("finalSynthesisInputUnitCount"),
            "finalSynthesisProvider": diagnostics.get("finalSynthesisProvider"),
            "finalSynthesisModel": diagnostics.get("finalSynthesisModel"),
            "finalSynthesisRawTaskCount": diagnostics.get("finalSynthesisRawTaskCount"),
            "finalSynthesisRawNoteCount": diagnostics.get("finalSynthesisRawNoteCount"),
            "finalSynthesisParsedTaskCount": diagnostics.get("finalSynthesisParsedTaskCount"),
            "finalSynthesisParsedNoteCount": diagnostics.get("finalSynthesisParsedNoteCount"),
            "finalSynthesisVerdict": diagnostics.get("finalSynthesisVerdict"),
            "taskCountAfterConfidence": diagnostics.get("taskCountAfterConfidence", len(result.tasks)),
            "noteCountAfterConfidence": diagnostics.get("noteCountAfterConfidence", len(result.notes)),
            "qualityAcceptedTaskCount": diagnostics.get("qualityAcceptedTaskCount"),
            "qualityAcceptedNoteCount": diagnostics.get("qualityAcceptedNoteCount"),
            "persistenceAttempted": diagnostics.get("persistenceAttempted"),
            "persistenceOutcome": diagnostics.get("persistenceOutcome"),
            "tasksPersistedCount": diagnostics.get("tasksPersistedCount"),
            "notesPersistedCount": diagnostics.get("notesPersistedCount"),
            "persistedTaskIds": diagnostics.get("persistedTaskIds"),
            "persistedNoteIds": diagnostics.get("persistedNoteIds"),
        },
    )


async def reconcile_artifacts(router: LLMRouter, candidates, incoming, window_text: str, repair: dict | None = None):
    from services.conversation.artifact_resolver import candidate_payload, incoming_payload

    provider, model = _route_for_input(
        router,
        LLMCapability.FINAL_SYNTHESIS,
        _rough_token_count(window_text) + 800,
    )
    payload = {
        "existingArtifacts": [candidate_payload(item) for item in candidates],
        "incomingUnits": [incoming_payload(index, item) for index, item in enumerate(incoming)],
        "windowText": window_text,
        "validTargetArtifactIds": [str(item.id) for item in candidates],
    }
    if repair:
        payload["repair"] = repair
    response, _, _ = await _structured_with_recovery(
        router,
        provider,
        model,
        "artifact-reconciler-v1",
        ArtifactReconcileResponse,
        "{}",
        json.dumps(payload, default=str, ensure_ascii=True),
    )
    return response


async def _classify_window_semantics(provider, model: str, window_text: str) -> list[dict[str, Any]]:
    """One bounded structured call; Python validates evidence IDs but infers no language meaning."""
    cache_key = re.sub(r"\s+", " ", window_text or "").strip()
    if cache_key in _SEMANTIC_CLASSIFICATION_CACHE:
        return _SEMANTIC_CLASSIFICATION_CACHE[cache_key]
    try:
        response = await _structured_with_provider(
            provider, model, "semantic-role-classifier-v1", SemanticRoleClassificationResponse, "{}",
            f"EVIDENCE UNITS:\n{window_text}",
        )
    except Exception:
        return []
    valid_ids = {int(value) for value in re.findall(r"\[(\d+)\]", window_text or "")}
    units = [
        unit.model_dump()
        for unit in response.units
        if unit.roles and unit.normalizedMeaning and unit.evidenceIds and set(unit.evidenceIds).issubset(valid_ids)
    ]
    if len(_SEMANTIC_CLASSIFICATION_CACHE) >= 256:
        _SEMANTIC_CLASSIFICATION_CACHE.pop(next(iter(_SEMANTIC_CLASSIFICATION_CACHE)))
    _SEMANTIC_CLASSIFICATION_CACHE[cache_key] = units
    return units


async def understand_window(
    router: LLMRouter,
    window,
    context: dict[str, Any],
    meeting_context: dict[str, Any] | None = None,
    semantic_evidence_packets: str | None = None,
) -> tuple[ConversationUnderstandingResponse, dict[str, str]]:
    provider, model = router.route(LLMCapability.SIMPLE_SUMMARY)
    try:
        response = await _structured_with_provider(
            provider,
            model,
            "conversation-understanding-v1",
            ConversationUnderstandingResponse,
            json.dumps(context, default=str, ensure_ascii=True),
            _window_extraction_prompt_input(
                window,
                {
                    **(meeting_context or {}),
                    "semanticReconstruction": semantic_evidence_packets or "SEMANTIC EVIDENCE PACKETS: []",
                },
            ),
        )
        return response, {
            "provider": resolved_provider_name(provider),
            "model": resolved_provider_model(provider, model) or model,
        }
    except Exception as error:
        print(
            "Window understanding skipped after structured LLM failure:",
            {
                "conversationId": str(window.conversationId),
                "windowIndex": window.windowIndex,
                "provider": resolved_provider_name(provider),
                "model": resolved_provider_model(provider, model) or model,
                "error": str(error)[:500],
            },
        )
        return ConversationUnderstandingResponse(), {
            "provider": resolved_provider_name(provider),
            "model": resolved_provider_model(provider, model) or model,
        }


def _context_with_understanding(
    meeting_context: dict[str, Any] | None,
    understanding: ConversationUnderstandingResponse,
    reconstruction_diagnostics: dict[str, Any] | None = None,
    reconstruction_prompt: str | None = None,
) -> dict[str, Any]:
    context = dict(meeting_context or {})
    context["conversationUnderstanding"] = understanding.model_dump()
    if reconstruction_diagnostics:
        context["semanticReconstructionDiagnostics"] = reconstruction_diagnostics
    if reconstruction_prompt:
        context["semanticReconstruction"] = reconstruction_prompt
    context["pipeline"] = [
        "Transcript/Window Summaries",
        "Context Reconstruction",
        "Conversation Understanding",
        "Candidate Extraction",
        "Evidence Retrieval",
        "Task/Note Enrichment",
        "Semantic Deduplication + Merge",
        "Critic/Validator",
        "Deterministic Confidence Scoring",
        "Final Publish",
        "Memory Update",
    ]
    return context


def _window_extraction_prompt_input(window, meeting_context: dict[str, Any] | None) -> str:
    parts = [
        f"WINDOW {window.windowIndex} [{window.sequenceStart}-{window.sequenceEnd}]:",
        window.text,
    ]
    if meeting_context:
        context_payload = dict(meeting_context)
        semantic_block = context_payload.pop("semanticReconstruction", None)
        parts = [
            "CURRENT MEETING STATE (bounded, do not rediscover; use for NEW/UPDATE/CONFIRM/CONTRADICT/COMPLETE):",
            json.dumps(context_payload, default=str, ensure_ascii=True),
            "",
            *(["", semantic_block, ""] if semantic_block else []),
            *parts,
        ]
    return "\n".join(parts)


async def repair_missing_items(
    router: LLMRouter,
    transcript: str,
    missing_items: list[dict[str, Any]],
    outputs: dict[str, Any],
    context: dict[str, Any],
    conversation_id: str,
    space_id: str,
) -> MissingItemRepairResponse:
    response = await _structured(
        router,
        "missing-item-repair-v1",
        MissingItemRepairLLMResponse,
        json.dumps(context, default=str, ensure_ascii=True),
        (
            f"CURRENT CONVERSATION:\n{transcript}\n\n"
            f"MISSING COVERAGE ITEMS:\n{json.dumps(missing_items, default=str, ensure_ascii=True)}\n\n"
            f"ALREADY EXTRACTED OUTPUTS:\n{json.dumps(outputs, default=str, ensure_ascii=True)}"
        ),
        LLMCapability.FINAL_SYNTHESIS,
    )
    repaired_tasks: list[ExtractedTask] = []
    for task in response.tasks:
        if task.operation == "NO_ACTION":
            continue
        task_data = task.model_dump(exclude={"fingerprint", "semanticArtifactKey", "quality", "semanticConflict", "semanticSpeculation", "sourceSemanticUnitIds"})
        task_data["changes"] = {
            **task_data.get("changes", {}),
            "semanticArtifactKey": task.semanticArtifactKey or None,
            "quality": task.quality,
            "synthesisSource": "llm",
            "semanticConflict": task.semanticConflict,
            "semanticSpeculation": task.semanticSpeculation,
            "sourceSemanticUnitIds": list(getattr(task, "sourceSemanticUnitIds", None) or []),
        }
        repaired_task = ExtractedTask(
            **task_data,
            sourceConversationId=conversation_id,
            fingerprint=task.fingerprint,
        )
        repaired_task.fingerprint = repaired_task.fingerprint or task_fingerprint(space_id, repaired_task)
        repaired_tasks.append(repaired_task)

    repaired_notes: list[ExtractedNote] = []
    for note in response.notes:
        note_data = note.model_dump(exclude={"fingerprint", "semanticArtifactKey", "quality", "semanticConflict", "sourceSemanticUnitIds"})
        note_data["debug"] = {
            "semanticArtifactKey": note.semanticArtifactKey or None,
            "quality": note.quality,
            "synthesisSource": "llm",
            "semanticConflict": note.semanticConflict,
            "sourceSemanticUnitIds": list(getattr(note, "sourceSemanticUnitIds", None) or []),
        }
        repaired_note = ExtractedNote(
            **note_data,
            sourceConversationId=conversation_id,
            fingerprint=note.fingerprint,
        )
        repaired_note.fingerprint = repaired_note.fingerprint or note_fingerprint(space_id, repaired_note)
        repaired_notes.append(repaired_note)
    return MissingItemRepairResponse(tasks=repaired_tasks, notes=repaired_notes)


async def review_extraction_quality(
    router: LLMRouter,
    transcript: str,
    outputs: dict[str, Any],
    context: dict[str, Any],
) -> ExtractionQualityReviewResponse:
    return await _structured(
        router,
        "extraction-quality-review-v1",
        ExtractionQualityReviewResponse,
        json.dumps(context, default=str, ensure_ascii=True),
        (
            f"CURRENT CONVERSATION:\n{transcript}\n\n"
            f"EXTRACTED TASKS AND NOTES TO REVIEW:\n{json.dumps(outputs, default=str, ensure_ascii=True)}"
        ),
        LLMCapability.VALIDATION,
    )


async def summarize_conversation(
    router: LLMRouter,
    conversation_id: str,
    user_id: str,
    space_id: str,
    transcript: str,
    outputs: dict[str, Any],
    processing_version: int,
) -> ConversationSummaryDocument:
    response = await _structured(
        router,
        "conversation-summary-v1",
        SummaryResponse,
        "{}",
        f"CURRENT CONVERSATION:\n{transcript}\n\nVALIDATED OUTPUTS:\n{json.dumps(outputs, default=str, ensure_ascii=True)}",
        LLMCapability.SIMPLE_SUMMARY,
    )
    provider, model = router.route(LLMCapability.SIMPLE_SUMMARY)
    return ConversationSummaryDocument(
        conversationId=conversation_id,
        userId=user_id,
        spaceId=space_id,
        summary=response.summary,
        topics=response.topics,
        importantFacts=response.importantFacts,
        decisions=response.decisions,
        openQuestions=response.openQuestions,
        blockers=response.blockers,
        processingVersion=processing_version,
        modelProvider=provider.name,
        modelName=model,
        promptVersion="conversation-summary-v1",
    )


async def reconcile_meeting(
    router: LLMRouter,
    conversation_id: str,
    user_id: str,
    space_id: str,
    artifacts_payload: list[dict[str, Any]],
    window_summaries: list[dict[str, Any]],
    meeting_memory: dict[str, Any] | None,
    context: dict[str, Any],
    processing_version: int,
    leftover_raw: str = "",
    checkpoints: list[dict[str, Any]] | None = None,
) -> tuple[WindowExtractionResult, str, str]:
    from services.conversation.budget import expected_request_tokens

    estimated = expected_request_tokens(
        json.dumps(artifacts_payload, default=str),
        json.dumps(checkpoints or window_summaries, default=str),
        leftover_raw,
    )
    provider, model = _route_for_input(router, LLMCapability.FINAL_SYNTHESIS, estimated)
    payload = _trim_reconcile_payload(
        {
            "conversationId": conversation_id,
            "processingVersion": processing_version,
            "meetingMemory": meeting_memory or {},
            "artifacts": artifacts_payload,
            "semanticCheckpoints": checkpoints or window_summaries,
            "windowSummaries": window_summaries,
            "leftoverRawTranscript": leftover_raw,
        },
        provider.name,
        model,
    )
    unit_count = sum(
        len(item.get("semanticUnits") or [])
        for item in (payload.get("semanticCheckpoints") or checkpoints or [])
    )
    try:
        result = await _structured_with_provider(
            provider,
            model,
            "final-synthesis-v1",
            FinalSynthesisLLMResponse,
            json.dumps(context, default=str, ensure_ascii=True),
            json.dumps(payload, default=str, ensure_ascii=True),
        )
    except Exception as error:
        print(
            "Reconciliation using artifact fallback after structured LLM failure:",
            {
                "conversationId": conversation_id,
                "provider": resolved_provider_name(provider),
                "model": resolved_provider_model(provider, model) or model,
                "artifactCount": len(artifacts_payload),
                "error": str(error)[:500],
            },
        )
        if unit_count > 0:
            failed = WindowExtractionResult(summary="Final synthesis failed.")
            failed.extractionDiagnostics = empty_final_synthesis_diagnostics(
                validated_unit_count=unit_count,
                invoked=True,
                verdict="PROVIDER_FAILED",
            )
            failed.extractionDiagnostics["finalSynthesisProvider"] = resolved_provider_name(provider)
            failed.extractionDiagnostics["finalSynthesisModel"] = resolved_provider_model(provider, model) or model
            raise FinalSynthesisError("PROVIDER_FAILED", str(error)[:500]) from error
        return WindowExtractionResult(summary="Meeting organized from stored artifacts after LLM failure."), resolved_provider_name(provider), resolved_provider_model(provider, model) or model
    parse_trace = LAST_FINAL_SYNTHESIS_PARSE_TRACE.get() or {}
    finalized = _window_result_from_llm(result, conversation_id, space_id)
    parsed_task_count = len(finalized.tasks)
    parsed_note_count = len(finalized.notes)
    diagnostics = empty_final_synthesis_diagnostics(
        validated_unit_count=unit_count,
        invoked=True,
        verdict="PUBLISH" if parsed_task_count or parsed_note_count else "NO_PUBLISHABLE_ARTIFACTS",
    )
    diagnostics["finalSynthesisInputUnitCount"] = unit_count
    diagnostics["finalSynthesisProvider"] = resolved_provider_name(provider)
    diagnostics["finalSynthesisModel"] = resolved_provider_model(provider, model) or model
    diagnostics["finalSynthesisRawTaskCount"] = int(parse_trace.get("rawTaskCount") or parsed_task_count)
    diagnostics["finalSynthesisRawNoteCount"] = int(parse_trace.get("rawNoteCount") or parsed_note_count)
    diagnostics["finalSynthesisParsedTaskCount"] = parsed_task_count
    diagnostics["finalSynthesisParsedNoteCount"] = parsed_note_count
    if getattr(result, "publishVerdict", None) == "NO_PUBLISHABLE_ARTIFACTS" or not (parsed_task_count or parsed_note_count):
        diagnostics["finalSynthesisVerdict"] = "NO_PUBLISHABLE_ARTIFACTS"
    else:
        diagnostics["finalSynthesisVerdict"] = "PUBLISH"
    units = [
        unit
        for item in (checkpoints or payload.get("semanticCheckpoints") or [])
        for unit in (item.get("semanticUnits") or [])
    ]
    finalized.extractionDiagnostics.update(diagnostics)
    finalized = await apply_final_artifact_quality_gate(
        router,
        finalized,
        units,
        leftover_raw or _evidence_corpus(finalized),
        context,
        conversation_id,
        space_id,
        finalized.extractionDiagnostics,
    )
    _log_final_synthesis(finalized.extractionDiagnostics)
    return finalized, provider.name, model


async def finalize_from_window_results(
    router: LLMRouter,
    conversation_id: str,
    user_id: str,
    space_id: str,
    window_payload: list[dict[str, Any]],
    context: dict[str, Any],
    processing_version: int,
) -> tuple[WindowExtractionResult, str, str]:
    provider, model = router.route(LLMCapability.FINAL_SYNTHESIS)
    window_payload = _trim_payload_for_provider(window_payload, provider.name, model)
    try:
        result, provider, model = await _structured_with_recovery(
            router,
            provider,
            model,
            "meeting-finalizer-v1",
            WindowExtractionLLMResponse,
            json.dumps(context, default=str, ensure_ascii=True),
            json.dumps({"conversationId": conversation_id, "windows": window_payload}, default=str, ensure_ascii=True),
        )
    except Exception as error:
        print(
            "Finalization using minimal fallback after structured LLM failure:",
            {
                "conversationId": conversation_id,
                "provider": resolved_provider_name(provider),
                "model": resolved_provider_model(provider, model) or model,
                "error": str(error)[:500],
            },
        )
        result = _minimal_final_response(window_payload)
    finalized = _window_result_from_llm(result, conversation_id, space_id)
    finalized = score_and_filter_result(finalized, _evidence_corpus(finalized))
    if _needs_final_memory_recovery(finalized, window_payload):
        try:
            recovery_response = await _structured_with_provider(
                provider,
                model,
                "final-memory-recovery-v1",
                WindowExtractionLLMResponse,
                json.dumps(context, default=str, ensure_ascii=True),
                (
                    "FINALIZATION INPUT WINDOWS:\n"
                    f"{json.dumps({'conversationId': conversation_id, 'windows': window_payload}, default=str, ensure_ascii=True)}\n\n"
                    "PREVIOUS FINALIZATION THAT MAY HAVE MISSED STORED MEMORY OBJECTS:\n"
                    f"{json.dumps(finalized.model_dump(), default=str, ensure_ascii=True)}"
                ),
            )
            recovered = _window_result_from_llm(recovery_response, conversation_id, space_id)
            recovered = score_and_filter_result(recovered, _evidence_corpus(recovered))
            finalized = _merge_window_extraction_results(finalized, recovered)
        except Exception as error:
            print(
                "Final memory recovery skipped after failure:",
                {
                    "conversationId": conversation_id,
                    "error": str(error)[:500],
                },
            )
    finalized = _preserve_window_candidates_when_final_empty(finalized, window_payload, conversation_id, space_id)
    finalized = score_and_filter_result(finalized, _evidence_corpus(finalized))
    return finalized, provider.name, model


async def update_space_memory(
    router: LLMRouter,
    previous: SpaceMemoryDocument,
    summary: ConversationSummaryDocument,
) -> SpaceMemoryDocument:
    response = await _structured(
        router,
        "space-memory-update-v1",
        MemoryUpdateResponse,
        json.dumps(previous.model_dump(by_alias=True), default=str, ensure_ascii=True),
        json.dumps(summary.model_dump(by_alias=True), default=str, ensure_ascii=True),
        LLMCapability.SIMPLE_SUMMARY,
    )
    return previous.model_copy(
        update={
            "currentSummary": response.currentSummary,
            "importantFacts": response.importantFacts,
            "importantDecisions": response.importantDecisions,
            "recentConversationSummaryIds": [summary.id, *previous.recentConversationSummaryIds[:9]],
            "lastUpdatedConversationId": summary.conversationId,
            "version": previous.version + 1,
        }
    )


async def apply_final_artifact_quality_gate(
    router: LLMRouter,
    result: WindowExtractionResult,
    units: list,
    transcript: str,
    context: dict[str, Any],
    conversation_id: str,
    space_id: str,
    diagnostics: dict[str, Any],
) -> WindowExtractionResult:
    result = hydrate_synthesized_artifacts(result, units, transcript)
    quality: dict[str, Any] = {}
    result = score_and_filter_result(result, transcript, diagnostics=quality)
    first_records = list(quality.get("qualityArtifactDiagnostics") or [])
    _merge_quality_diagnostics(diagnostics, quality, result)
    coverage = evaluate_task_coverage(units, result)
    missed_units = list(coverage.get("missedActionableUnits") or [])
    diagnostics["validatedActionableUnitCount"] = coverage.get("validatedActionableUnitCount", 0)
    diagnostics["taskCoverageConflict"] = bool(coverage.get("taskCoverageConflict"))
    diagnostics["unitDispositions"] = coverage.get("unitDispositions") or []
    diagnostics["undisposedActionableUnitCount"] = coverage.get("undisposedActionableUnitCount", 0)
    if (result.tasks or result.notes) and diagnostics.get("finalSynthesisVerdict") == "NO_PUBLISHABLE_ARTIFACTS":
        diagnostics["finalSynthesisVerdict"] = "PUBLISH"
        diagnostics["publishVerdictOverridden"] = True
    if coverage.get("taskCoverageConflict"):
        diagnostics["finalSynthesisVerdict"] = TASK_COVERAGE_CONFLICT
    parsed_tasks = int(diagnostics.get("finalSynthesisParsedTaskCount") or 0)
    parsed_notes = int(diagnostics.get("finalSynthesisParsedNoteCount") or 0)
    quality_empty_after_publish = (
        diagnostics.get("finalSynthesisVerdict") in {"PUBLISH", TASK_COVERAGE_CONFLICT}
        and (parsed_tasks or parsed_notes)
        and not result.tasks
        and not result.notes
    )
    should_repair = (
        (quality_empty_after_publish or coverage.get("taskCoverageConflict"))
        and settings.MAX_QUALITY_REPAIR_ROUNDS >= 1
        and not diagnostics.get("qualityRepairAttempted")
    )
    if not should_repair:
        return result
    diagnostics["qualityRepairAttempted"] = True
    diagnostics["qualityRepairRound"] = 1
    try:
        repaired = await repair_final_quality_failures(
            router,
            transcript,
            quality.get("qualityRejectedTaskItems") or [],
            quality.get("qualityRejectedNoteItems") or [],
            quality.get("qualityArtifactDiagnostics") or [],
            units,
            context,
            conversation_id,
            space_id,
            missed_actionable_units=missed_units,
            current_tasks=list(result.tasks),
            coverage_conflict=bool(coverage.get("taskCoverageConflict")),
        )
        result.tasks = list(result.tasks) + list(repaired.tasks)
        result.notes = list(result.notes) + list(repaired.notes)
        result = hydrate_synthesized_artifacts(result, units, transcript)
        quality_after: dict[str, Any] = {}
        result = score_and_filter_result(result, transcript, diagnostics=quality_after)
        diagnostics["qualityRepairAcceptedTaskCount"] = len(result.tasks)
        diagnostics["qualityRepairAcceptedNoteCount"] = len(result.notes)
        _merge_quality_diagnostics(diagnostics, quality_after, result)
        coverage_after = evaluate_task_coverage(units, result)
        diagnostics["validatedActionableUnitCount"] = coverage_after.get("validatedActionableUnitCount", 0)
        diagnostics["taskCoverageConflict"] = bool(coverage_after.get("taskCoverageConflict"))
        diagnostics["unitDispositions"] = coverage_after.get("unitDispositions") or []
        if coverage_after.get("taskCoverageConflict"):
            diagnostics["finalSynthesisVerdict"] = TASK_COVERAGE_CONFLICT
        elif result.tasks or result.notes:
            diagnostics["finalSynthesisVerdict"] = "PUBLISH"
        if not result.tasks and not result.notes and first_records:
            diagnostics["qualityArtifactDiagnostics"] = first_records
    except Exception as error:
        diagnostics["qualityRepairError"] = str(error)[:300]
        print("Final quality repair skipped:", {"conversationId": conversation_id, "error": str(error)[:300]})
    return result


def _merge_quality_diagnostics(diagnostics: dict[str, Any], quality: dict[str, Any], result: WindowExtractionResult) -> None:
    diagnostics["taskCountAfterConfidence"] = len(result.tasks)
    diagnostics["noteCountAfterConfidence"] = len(result.notes)
    diagnostics["qualityRejectedTaskCount"] = quality.get("qualityRejectedTaskCount", 0)
    diagnostics["qualityRejectedNoteCount"] = quality.get("qualityRejectedNoteCount", 0)
    diagnostics["qualityAcceptedTaskCount"] = len(result.tasks)
    diagnostics["qualityAcceptedNoteCount"] = len(result.notes)
    diagnostics["requiredConfidence"] = quality.get("requiredConfidence")
    diagnostics["qualityArtifactDiagnostics"] = quality.get("qualityArtifactDiagnostics") or []


def _repair_artifact_payload(item: ExtractedTask | ExtractedNote) -> dict[str, Any]:
    metadata = getattr(item, "changes", None) or getattr(item, "debug", None) or {}
    return {
        "title": item.title,
        "body": item.body,
        "operation": getattr(item, "operation", None),
        "ownerText": getattr(item, "ownerText", None),
        "dueDateText": getattr(item, "dueDateText", None),
        "origin": getattr(item, "origin", None),
        "semanticArtifactKey": metadata.get("semanticArtifactKey"),
        "sourceSemanticUnitIds": list(metadata.get("sourceSemanticUnitIds") or []),
        "qualityRejectionReasons": list(metadata.get("qualityRejectionReasons") or []),
        "evidence": [
            {"sequenceStart": span.sequenceStart, "sequenceEnd": span.sequenceEnd}
            for span in (item.evidence or [])
        ],
    }


def _repair_unit_payload(unit) -> dict[str, Any]:
    if hasattr(unit, "model_dump"):
        payload = unit.model_dump()
    elif isinstance(unit, dict):
        payload = dict(unit)
    else:
        return {}
    return {
        "semanticKey": payload.get("semanticKey"),
        "kind": payload.get("kind"),
        "meaning": payload.get("meaning"),
        "ownerText": payload.get("ownerText"),
        "dueDateText": payload.get("dueDateText"),
        "evidence": [
            {"sequenceStart": span.get("sequenceStart") if isinstance(span, dict) else span.sequenceStart,
             "sequenceEnd": span.get("sequenceEnd") if isinstance(span, dict) else span.sequenceEnd}
            for span in (payload.get("evidence") or [])
        ],
        "evidenceIds": payload.get("evidenceIds") or [],
    }


async def repair_final_quality_failures(
    router: LLMRouter,
    transcript: str,
    rejected_tasks: list[ExtractedTask],
    rejected_notes: list[ExtractedNote],
    artifact_diagnostics: list[dict[str, Any]],
    units: list,
    context: dict[str, Any],
    conversation_id: str,
    space_id: str,
    missed_actionable_units: list | None = None,
    current_tasks: list[ExtractedTask] | None = None,
    coverage_conflict: bool = False,
) -> MissingItemRepairResponse:
    payload = {
        "rejectedTasks": [_repair_artifact_payload(item) for item in rejected_tasks],
        "rejectedNotes": [_repair_artifact_payload(item) for item in rejected_notes],
        "qualityArtifactDiagnostics": artifact_diagnostics,
        "validatedSemanticUnits": [_repair_unit_payload(unit) for unit in units],
    }
    if coverage_conflict:
        payload["taskCoverage"] = coverage_repair_payload(list(missed_actionable_units or []), list(current_tasks or []))
    response = await _structured(
        router,
        "final-quality-repair-v1",
        MissingItemRepairLLMResponse,
        json.dumps(context, default=str, ensure_ascii=True),
        (
            f"CURRENT CONVERSATION:\n{transcript}\n\n"
            f"QUALITY FAILURES TO REPAIR:\n{json.dumps(payload, default=str, ensure_ascii=True)}"
        ),
        LLMCapability.VALIDATION,
    )
    return await _materialize_repair_response(response, conversation_id, space_id)


async def _materialize_repair_response(response, conversation_id: str, space_id: str) -> MissingItemRepairResponse:
    placeholder = SimpleNamespace(tasks=response.tasks, notes=response.notes, summary="", narrative="", topics=[], importantFacts=[], decisions=[], issues=[], openQuestions=[], semanticUnits=[])
    materialized = _window_result_from_llm(placeholder, conversation_id, space_id)
    return MissingItemRepairResponse(tasks=materialized.tasks, notes=materialized.notes)


def _window_result_from_llm(response, conversation_id: str, space_id: str) -> WindowExtractionResult:
    tasks: list[ExtractedTask] = []
    for task in response.tasks:
        if task.operation == "NO_ACTION":
            continue
        task_data = task.model_dump(exclude={"fingerprint", "semanticArtifactKey", "quality", "semanticConflict", "semanticSpeculation", "sourceSemanticUnitIds"})
        task_data["changes"] = {
            **task_data.get("changes", {}),
            "semanticArtifactKey": task.semanticArtifactKey or None,
            "quality": task.quality,
            "synthesisSource": "llm",
            "semanticConflict": task.semanticConflict,
            "semanticSpeculation": task.semanticSpeculation,
            "sourceSemanticUnitIds": list(getattr(task, "sourceSemanticUnitIds", None) or []),
        }
        extracted = ExtractedTask(
            **task_data,
            sourceConversationId=conversation_id,
            fingerprint=task.fingerprint,
        )
        extracted.fingerprint = extracted.fingerprint or task_fingerprint(space_id, extracted)
        tasks.append(extracted)

    notes: list[ExtractedNote] = []
    for note in response.notes:
        note_data = note.model_dump(exclude={"fingerprint", "semanticArtifactKey", "quality", "semanticConflict", "sourceSemanticUnitIds"})
        note_data["debug"] = {
            "semanticArtifactKey": note.semanticArtifactKey or None,
            "quality": note.quality,
            "synthesisSource": "llm",
            "semanticConflict": note.semanticConflict,
            "sourceSemanticUnitIds": list(getattr(note, "sourceSemanticUnitIds", None) or []),
        }
        extracted = ExtractedNote(
            **note_data,
            sourceConversationId=conversation_id,
            fingerprint=note.fingerprint,
        )
        extracted.fingerprint = extracted.fingerprint or note_fingerprint(space_id, extracted)
        notes.append(extracted)

    return WindowExtractionResult(
        summary=response.summary or response.narrative,
        narrative=response.narrative or response.summary,
        topics=response.topics,
        importantFacts=response.importantFacts,
        semanticUnits=list(getattr(response, "semanticUnits", None) or []),
        tasks=tasks,
        notes=notes,
        decisions=[
            ExtractedDecision(**decision.model_dump(), sourceConversationId=conversation_id)
            for decision in response.decisions
        ],
        issues=[ExtractedIssue(**issue.model_dump(), sourceConversationId=conversation_id) for issue in response.issues],
        openQuestions=response.openQuestions,
    )


def _evidence_corpus(result: WindowExtractionResult) -> str:
    spans: list[str] = []
    for collection in (result.tasks, result.notes, result.decisions, result.issues):
        for item in collection:
            spans.extend(f"[{span.sequenceStart}] {span.text}" for span in item.evidence)
    spans.extend(result.importantFacts)
    spans.extend(result.openQuestions)
    return "\n".join(spans)


def _needs_window_recovery(result: WindowExtractionResult, window_text: str) -> bool:
    text = (window_text or "").strip()
    if not text:
        return False
    extracted = bool(result.tasks or result.notes or result.decisions or result.issues)
    if not extracted:
        return _result_has_note_source(result)
    if not result.notes and _result_has_note_source(result):
        return True
    return False


def _needs_final_memory_recovery(finalized: WindowExtractionResult, window_payload: list[dict[str, Any]]) -> bool:
    if not finalized.notes and _window_payload_has_note_source(window_payload):
        return True
    if finalized.tasks or finalized.notes or finalized.decisions or finalized.issues:
        return False
    return _window_payload_has_memory_source(window_payload)


def _result_has_note_source(result: WindowExtractionResult) -> bool:
    return bool(
        result.importantFacts
        or result.topics
        or result.openQuestions
        or _rough_token_count(result.summary) >= 20
    )


def _window_payload_has_note_source(window_payload: list[dict[str, Any]]) -> bool:
    for window in window_payload:
        if window.get("notes") or window.get("importantFacts"):
            return True
        if _rough_token_count(str(window.get("summary") or "")) >= 30:
            return True
    return False


def _window_payload_has_memory_source(window_payload: list[dict[str, Any]]) -> bool:
    for window in window_payload:
        if window.get("tasks") or window.get("notes") or window.get("decisions") or window.get("issues"):
            return True
        if window.get("importantFacts") or window.get("openQuestions"):
            return True
        if _rough_token_count(str(window.get("summary") or "")) >= 30:
            return True
    return False


def _merge_window_extraction_results(primary: WindowExtractionResult, recovery: WindowExtractionResult) -> WindowExtractionResult:
    primary.summary = primary.summary or recovery.summary
    primary.topics = _dedupe_values([*primary.topics, *recovery.topics])
    primary.importantFacts = _dedupe_values([*primary.importantFacts, *recovery.importantFacts])
    primary.openQuestions = _dedupe_values([*primary.openQuestions, *recovery.openQuestions])
    primary.tasks = _dedupe_items([*primary.tasks, *recovery.tasks])
    primary.notes = _dedupe_items([*primary.notes, *recovery.notes])
    primary.decisions = _dedupe_items([*primary.decisions, *recovery.decisions])
    primary.issues = _dedupe_items([*primary.issues, *recovery.issues])
    seen_units = {unit.semanticKey or unit.meaning for unit in primary.semanticUnits}
    for unit in recovery.semanticUnits:
        key = unit.semanticKey or unit.meaning
        if key in seen_units:
            continue
        seen_units.add(key)
        primary.semanticUnits.append(unit)
    return primary


def _preserve_window_candidates_when_final_empty(
    finalized: WindowExtractionResult,
    window_payload: list[dict[str, Any]],
    conversation_id: str,
    space_id: str,
) -> WindowExtractionResult:
    carried = _extract_window_candidates(window_payload, conversation_id, space_id)
    if not finalized.tasks:
        finalized.tasks = _dedupe_items(carried.tasks)
    else:
        finalized.tasks = _dedupe_items([*finalized.tasks, *carried.tasks])
    if not finalized.notes:
        finalized.notes = _dedupe_items(carried.notes)
    else:
        finalized.notes = _dedupe_items([*finalized.notes, *carried.notes])
    if not finalized.decisions:
        finalized.decisions = _dedupe_items(carried.decisions)
    else:
        finalized.decisions = _dedupe_items([*finalized.decisions, *carried.decisions])
    if not finalized.issues:
        finalized.issues = _dedupe_items(carried.issues)
    else:
        finalized.issues = _dedupe_items([*finalized.issues, *carried.issues])
    return finalized


def _extract_window_candidates(
    window_payload: list[dict[str, Any]],
    conversation_id: str,
    space_id: str,
) -> WindowExtractionResult:
    carried = WindowExtractionResult()
    for window in window_payload:
        for raw_task in window.get("tasks", []) or []:
            task = _safe_task_from_payload(raw_task, conversation_id, space_id)
            if task:
                carried.tasks.append(task)
        for raw_note in window.get("notes", []) or []:
            note = _safe_note_from_payload(raw_note, conversation_id, space_id)
            if note:
                carried.notes.append(note)
        for raw_decision in window.get("decisions", []) or []:
            decision = _safe_decision_from_payload(raw_decision, conversation_id)
            if decision:
                carried.decisions.append(decision)
        for raw_issue in window.get("issues", []) or []:
            issue = _safe_issue_from_payload(raw_issue, conversation_id)
            if issue:
                carried.issues.append(issue)
    return carried


def _safe_task_from_payload(raw: dict[str, Any], conversation_id: str, space_id: str) -> ExtractedTask | None:
    if not isinstance(raw, dict) or not raw.get("title") or not raw.get("evidence"):
        return None
    data = dict(raw)
    if data.get("operation") == "NO_ACTION":
        return None
    data.setdefault("body", "")
    data.setdefault("operation", "NEEDS_CONFIRMATION")
    data.setdefault("confidence", 0.5)
    data.setdefault("needsConfirmation", data.get("operation") == "NEEDS_CONFIRMATION")
    data.setdefault("sourceConversationId", conversation_id)
    try:
        task = ExtractedTask.model_validate(data)
    except Exception:
        return None
    task.fingerprint = task.fingerprint or task_fingerprint(space_id, task)
    return task


def _safe_note_from_payload(raw: dict[str, Any], conversation_id: str, space_id: str) -> ExtractedNote | None:
    if not isinstance(raw, dict) or not raw.get("title") or not raw.get("body") or not raw.get("evidence"):
        return None
    data = dict(raw)
    data.setdefault("confidence", 0.5)
    data.setdefault("sourceConversationId", conversation_id)
    try:
        note = ExtractedNote.model_validate(data)
    except Exception:
        return None
    note.fingerprint = note.fingerprint or note_fingerprint(space_id, note)
    return note


def _safe_decision_from_payload(raw: dict[str, Any], conversation_id: str) -> ExtractedDecision | None:
    if not isinstance(raw, dict) or not raw.get("title") or not raw.get("evidence"):
        return None
    data = dict(raw)
    data.setdefault("status", "unresolved_discussion")
    data.setdefault("confidence", 0.5)
    data.setdefault("sourceConversationId", conversation_id)
    try:
        return ExtractedDecision.model_validate(data)
    except Exception:
        return None


def _safe_issue_from_payload(raw: dict[str, Any], conversation_id: str) -> ExtractedIssue | None:
    if not isinstance(raw, dict) or not raw.get("title") or not raw.get("evidence"):
        return None
    data = dict(raw)
    data.setdefault("kind", "open_question")
    data.setdefault("confidence", 0.5)
    data.setdefault("sourceConversationId", conversation_id)
    try:
        return ExtractedIssue.model_validate(data)
    except Exception:
        return None


def _dedupe_values(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        normalized = " ".join(str(value or "").split()).casefold()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(value)
    return unique


def _dedupe_items(items: list[Any]) -> list[Any]:
    seen: set[str] = set()
    unique: list[Any] = []
    for item in items:
        fingerprint = getattr(item, "fingerprint", None)
        evidence = getattr(item, "evidence", [])
        evidence_key = "|".join(f"{span.sequenceStart}:{span.sequenceEnd}" for span in evidence)
        identity = fingerprint or "|".join(
            [
                str(getattr(item, "title", "")),
                str(getattr(item, "body", ""))[:300],
                str(getattr(item, "operation", "")),
                str(getattr(item, "status", "")),
                str(getattr(item, "kind", "")),
                evidence_key,
            ]
        ).casefold()
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(item)
    return unique


def _trim_reconcile_payload(payload: dict[str, Any], provider_name: str, model: str | None = None) -> dict[str, Any]:
    limit = _provider_input_token_limit(provider_name, model)
    artifacts = list(payload.get("artifacts") or [])
    window_summaries = list(payload.get("windowSummaries") or [])
    checkpoints = list(payload.get("semanticCheckpoints") or window_summaries)
    leftover_raw = str(payload.get("leftoverRawTranscript") or "")
    meeting_memory = payload.get("meetingMemory") or {}
    content_limit = 400
    while True:
        candidate = {
            "conversationId": payload.get("conversationId"),
            "processingVersion": payload.get("processingVersion"),
            "meetingMemory": meeting_memory,
            "semanticCheckpoints": checkpoints,
            "windowSummaries": window_summaries,
            "leftoverRawTranscript": leftover_raw,
            "artifacts": [
                {
                    **item,
                    "content": str(item.get("content") or "")[:content_limit],
                    "evidence": (item.get("evidence") or [])[:3],
                }
                for item in artifacts
            ],
        }
        if _rough_token_count(json.dumps(candidate, default=str, ensure_ascii=True)) <= limit or content_limit <= 80:
            if leftover_raw and _rough_token_count(json.dumps(candidate, default=str, ensure_ascii=True)) > limit:
                candidate["leftoverRawTranscript"] = leftover_raw
            return candidate
        content_limit = max(80, content_limit // 2)
        if content_limit <= 120:
            window_summaries = [
                {**item, "summary": str(item.get("summary") or item.get("narrative") or "")[:240]}
                for item in window_summaries
            ]
            checkpoints = [
                {**item, "narrative": str(item.get("narrative") or item.get("summary") or "")[:400]}
                for item in checkpoints
            ]


def _trim_payload_for_provider(window_payload: list[dict[str, Any]], provider_name: str, model: str | None = None) -> list[dict[str, Any]]:
    limit = _provider_input_token_limit(provider_name, model)
    if _rough_token_count(json.dumps(window_payload, default=str, ensure_ascii=True)) <= limit:
        return window_payload

    compacted: list[dict[str, Any]] = []
    for window in window_payload:
        compacted.append(
            {
                "windowIndex": window.get("windowIndex"),
                "sequenceStart": window.get("sequenceStart"),
                "sequenceEnd": window.get("sequenceEnd"),
                "summary": str(window.get("summary") or "")[:1200],
                "topics": window.get("topics", [])[:12],
                "importantFacts": window.get("importantFacts", [])[:20],
                "tasks": [_compact_extracted_item(item) for item in window.get("tasks", [])[:30]],
                "notes": [_compact_extracted_item(item) for item in window.get("notes", [])[:20]],
                "decisions": [_compact_extracted_item(item) for item in window.get("decisions", [])[:20]],
                "issues": [_compact_extracted_item(item) for item in window.get("issues", [])[:20]],
                "openQuestions": window.get("openQuestions", [])[:20],
            }
        )
        if _rough_token_count(json.dumps(compacted, default=str, ensure_ascii=True)) > limit:
            compacted[-1]["summary"] = str(compacted[-1].get("summary") or "")[:400]
            compacted[-1]["notes"] = compacted[-1].get("notes", [])[:8]
            compacted[-1]["importantFacts"] = compacted[-1].get("importantFacts", [])[:8]
    while compacted and _rough_token_count(json.dumps(compacted, default=str, ensure_ascii=True)) > limit:
        for window in compacted:
            window["notes"] = window.get("notes", [])[: max(0, len(window.get("notes", [])) - 1)]
            window["importantFacts"] = window.get("importantFacts", [])[: max(0, len(window.get("importantFacts", [])) - 1)]
        if all(not window.get("notes") and not window.get("importantFacts") for window in compacted):
            break
    return compacted


def _compact_extracted_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": item.get("title"),
        "body": str(item.get("body") or "")[:500],
        "confidence": item.get("confidence"),
        "sourceConversationId": item.get("sourceConversationId"),
        "fingerprint": item.get("fingerprint"),
        "operation": item.get("operation"),
        "status": item.get("status"),
        "kind": item.get("kind"),
        "existingTaskId": item.get("existingTaskId"),
        "ownerText": item.get("ownerText"),
        "ownerUserId": item.get("ownerUserId"),
        "dueDateText": item.get("dueDateText"),
        "dueDateResolved": item.get("dueDateResolved"),
        "dueDateStatus": item.get("dueDateStatus"),
        "needsConfirmation": item.get("needsConfirmation"),
        "evidence": item.get("evidence", []),
    }


def _provider_input_token_limit(provider_name: str, model: str | None = None) -> int:
    from services.conversation.budget import safe_input_budget

    return min(settings.FINAL_MODEL_INPUT_TOKEN_LIMIT, safe_input_budget(provider_name, model=model))


def _route_for_input(router: LLMRouter, capability: LLMCapability, estimated_input_tokens: int):
    # Role-based routing only. Long meetings keep existing window/checkpoint budgets
    # instead of switching to a different conversation-intelligence role.
    del estimated_input_tokens
    return router.route(capability)


def _rough_token_count(text: str) -> int:
    return max(1, len(text) // 4)


def _minimal_window_response(window_text: str) -> WindowExtractionLLMResponse:
    summary = " ".join(str(window_text or "").split())
    if len(summary) > 500:
        summary = f"{summary[:497].rstrip()}..."
    return WindowExtractionLLMResponse(summary=summary)


def _minimal_final_response(window_payload: list[dict[str, Any]]) -> WindowExtractionLLMResponse:
    summaries = [
        str(window.get("summary") or "").strip()
        for window in window_payload
        if str(window.get("summary") or "").strip()
    ]
    summary = " ".join(summaries) or "Conversation processed from transcript windows."
    if len(summary) > 800:
        summary = f"{summary[:797].rstrip()}..."
    return WindowExtractionLLMResponse(
        summary=summary,
        topics=_dedupe_values(
            [
                str(topic)
                for window in window_payload
                for topic in (window.get("topics") or [])
                if str(topic).strip()
            ]
        )[:20],
        importantFacts=_dedupe_values(
            [
                str(fact)
                for window in window_payload
                for fact in (window.get("importantFacts") or [])
                if str(fact).strip()
            ]
        )[:30],
        openQuestions=_dedupe_values(
            [
                str(question)
                for window in window_payload
                for question in (window.get("openQuestions") or [])
                if str(question).strip()
            ]
        )[:20],
    )


async def _structured(
    router: LLMRouter,
    prompt_name: str,
    schema: type[BaseModel],
    background: str,
    current: str,
    capability: LLMCapability,
) -> Any:
    provider, model = router.route(capability)
    request = StructuredLLMRequest(
        model=model,
        temperature=0,
        schema_name=schema.__name__,
        messages=[
            LLMMessage(role="system", content=load_prompt(prompt_name)),
            LLMMessage(
                role="user",
                content=(
                    "BACKGROUND SPACE CONTEXT:\n"
                    f"{background}\n\n"
                    "CURRENT CONVERSATION - AUTHORITATIVE SOURCE:\n"
                    f"{current}"
                ),
            ),
        ],
    )
    return await provider.generate_structured(request, schema)


async def _structured_with_provider(
    provider,
    model: str,
    prompt_name: str,
    schema: type[BaseModel],
    background: str,
    current: str,
) -> Any:
    request = StructuredLLMRequest(
        model=model,
        temperature=0,
        max_tokens=_provider_structured_max_tokens(getattr(provider, "name", ""), schema.__name__),
        schema_name=schema.__name__,
        messages=[
            LLMMessage(role="system", content=load_prompt(prompt_name)),
            LLMMessage(
                role="user",
                content=(
                    "BACKGROUND SPACE CONTEXT:\n"
                    f"{background}\n\n"
                    "CURRENT CONVERSATION - AUTHORITATIVE SOURCE:\n"
                    f"{current}"
                ),
            ),
        ],
    )
    return await provider.generate_structured(request, schema)


async def _structured_with_recovery(
    router: LLMRouter,
    provider,
    model: str,
    prompt_name: str,
    schema: type[BaseModel],
    background: str,
    current: str,
) -> tuple[Any, Any, str]:
    attempts = [(provider, model)]
    if not isinstance(provider, FallbackLLMProvider):
        fallback_provider, fallback_model = router.route(LLMCapability.FALLBACK)
        if not _same_structured_route(provider, model, fallback_provider, fallback_model):
            attempts.append((fallback_provider, fallback_model))
    last_error: Exception | None = None
    for candidate_provider, candidate_model in attempts:
        try:
            response = await _structured_with_provider(
                candidate_provider,
                candidate_model,
                prompt_name,
                schema,
                background,
                current,
            )
            used_model = resolved_provider_model(candidate_provider, candidate_model) or candidate_model
            return response, candidate_provider, used_model
        except Exception as error:
            last_error = error
            if is_async_lifecycle_error(error):
                raise
            print(
                "Structured LLM attempt failed; trying eligible-model fallback if available:",
                {
                    "prompt": prompt_name,
                    "provider": getattr(candidate_provider, "name", "unknown"),
                    "model": candidate_model,
                    "error": str(error)[:500],
                },
            )
    raise last_error or RuntimeError(f"structured recovery exhausted for {prompt_name}")


def _same_structured_route(left_provider, left_model: str, right_provider, right_model: str) -> bool:
    left_name = getattr(left_provider, "name", "")
    right_name = getattr(right_provider, "name", "")
    return left_name == right_name and left_model == right_model and type(left_provider) is type(right_provider)


def _provider_structured_max_tokens(provider_name: str, schema_name: str = "") -> int | None:
    if schema_name in {"WindowExtractionLLMResponse", "MeetingCandidateExtractorResponse"}:
        configured = settings.LLM_EXTRACTION_OUTPUT_MAX_TOKENS
    elif schema_name in {"FinalSynthesisLLMResponse", "MeetingConsolidatorResponse"}:
        configured = min(settings.LLM_SYNTHESIS_OUTPUT_START_TOKENS, settings.LLM_SYNTHESIS_OUTPUT_MAX_TOKENS)
    else:
        configured = settings.LLM_STRUCTURED_MAX_TOKENS
    if provider_name == "groq":
        return max(512, min(configured, settings.GROQ_MAX_TPM // 2))
    return configured


async def _structured_or_empty(
    router: LLMRouter,
    prompt_name: str,
    schema: type[BaseModel],
    background: str,
    current: str,
    capability: LLMCapability,
    warnings: list[str],
) -> Any:
    try:
        return await _structured(router, prompt_name, schema, background, current, capability)
    except Exception as error:
        warnings.append(f"{prompt_name} failed: {str(error)[:500]}")
        return schema()
