from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from apps.api_gateway.config.setting import settings
from services.conversation.fingerprints import note_fingerprint, task_fingerprint
from services.conversation.models import (
    ConversationSummaryDocument,
    CoverageReport,
    EvidenceSpan,
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
from services.llm.models import LLMMessage, StructuredLLMRequest
from services.llm.router import LLMCapability, LLMRouter
from services.prompts.loader import load_prompt


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
    evidence: list[EvidenceSpan]


class RepairedNote(BaseModel):
    title: str
    body: str
    confidence: float = Field(ge=0, le=1)
    fingerprint: str | None = None
    evidence: list[EvidenceSpan]


class RepairedDecision(BaseModel):
    title: str
    status: Literal["confirmed_decision", "proposal", "idea", "unresolved_discussion"]
    confidence: float = Field(ge=0, le=1)
    evidence: list[EvidenceSpan]


class RepairedIssue(BaseModel):
    title: str
    kind: Literal["blocker", "risk", "open_question", "missing_information"]
    confidence: float = Field(ge=0, le=1)
    evidence: list[EvidenceSpan]


class MissingItemRepairLLMResponse(BaseModel):
    tasks: list[RepairedTask] = Field(default_factory=list)
    notes: list[RepairedNote] = Field(default_factory=list)


class MissingItemRepairResponse(BaseModel):
    tasks: list[ExtractedTask] = Field(default_factory=list)
    notes: list[ExtractedNote] = Field(default_factory=list)


class WindowExtractionLLMResponse(BaseModel):
    summary: str = ""
    topics: list[str] = Field(default_factory=list)
    importantFacts: list[str] = Field(default_factory=list)
    tasks: list[RepairedTask] = Field(default_factory=list)
    notes: list[RepairedNote] = Field(default_factory=list)
    decisions: list[RepairedDecision] = Field(default_factory=list)
    issues: list[RepairedIssue] = Field(default_factory=list)
    openQuestions: list[str] = Field(default_factory=list)


class ExtractionQualityDecision(BaseModel):
    kind: Literal["task", "note"]
    index: int = Field(ge=0)
    keep: bool
    reason: str
    revisedBody: str | None = None


class ExtractionQualityReviewResponse(BaseModel):
    decisions: list[ExtractionQualityDecision] = Field(default_factory=list)


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


async def extract_segment(
    router: LLMRouter,
    segment: Segment,
    context: dict[str, Any],
    user_id: str,
    space_id: str,
) -> SectionExtractionResult:
    background = json.dumps(context, default=str, ensure_ascii=True)
    current = segment.text
    warnings: list[str] = []
    tasks = await _structured_or_empty(
        router,
        "task-extractor-v1",
        TaskExtractionResponse,
        background,
        current,
        LLMCapability.HIGH_ACCURACY_REASONING,
        warnings,
    )
    notes = await _structured_or_empty(
        router,
        "note-extractor-v1",
        NoteExtractionResponse,
        background,
        current,
        LLMCapability.HIGH_ACCURACY_REASONING,
        warnings,
    )
    decisions = await _structured_or_empty(
        router,
        "decision-extractor-v1",
        DecisionExtractionResponse,
        background,
        current,
        LLMCapability.HIGH_ACCURACY_REASONING,
        warnings,
    )
    issues = await _structured_or_empty(
        router,
        "risk-question-extractor-v1",
        IssueExtractionResponse,
        background,
        current,
        LLMCapability.HIGH_ACCURACY_REASONING,
        warnings,
    )
    result = SectionExtractionResult(
        segmentId=segment.segmentId,
        tasks=tasks.tasks,
        notes=notes.notes,
        decisions=decisions.decisions,
        issues=issues.issues,
        warnings=warnings,
    )
    for task in result.tasks:
        task.sourceConversationId = segment.conversationId
        task.fingerprint = task.fingerprint or task_fingerprint(space_id, task)
    for note in result.notes:
        note.sourceConversationId = segment.conversationId
        note.fingerprint = note.fingerprint or note_fingerprint(space_id, note)
    for decision in result.decisions:
        decision.sourceConversationId = segment.conversationId
    for issue in result.issues:
        issue.sourceConversationId = segment.conversationId
    return result


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
) -> tuple[WindowExtractionResult, str, str]:
    provider, model = router.route(LLMCapability.HIGH_ACCURACY_REASONING)
    response = await _structured_with_provider(
        provider,
        model,
        "window-extractor-v1",
        WindowExtractionLLMResponse,
        json.dumps(context, default=str, ensure_ascii=True),
        f"WINDOW {window.windowIndex} [{window.sequenceStart}-{window.sequenceEnd}]:\n{window.text}",
    )
    result = _window_result_from_llm(response, str(window.conversationId), str(window.spaceId))
    return result, provider.name, model


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
        LLMCapability.HIGH_ACCURACY_REASONING,
    )
    repaired_tasks: list[ExtractedTask] = []
    for task in response.tasks:
        repaired_task = ExtractedTask(
            **task.model_dump(exclude={"fingerprint"}),
            sourceConversationId=conversation_id,
            fingerprint=task.fingerprint,
        )
        repaired_task.fingerprint = repaired_task.fingerprint or task_fingerprint(space_id, repaired_task)
        repaired_tasks.append(repaired_task)

    repaired_notes: list[ExtractedNote] = []
    for note in response.notes:
        repaired_note = ExtractedNote(
            **note.model_dump(exclude={"fingerprint"}),
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


async def finalize_from_window_results(
    router: LLMRouter,
    conversation_id: str,
    user_id: str,
    space_id: str,
    window_payload: list[dict[str, Any]],
    context: dict[str, Any],
    processing_version: int,
) -> tuple[WindowExtractionResult, str, str]:
    provider, model = router.route(LLMCapability.HIGH_ACCURACY_REASONING)
    window_payload = _trim_payload_for_provider(window_payload, provider.name)
    result = await _structured_with_provider(
        provider,
        model,
        "meeting-finalizer-v1",
        WindowExtractionLLMResponse,
        json.dumps(context, default=str, ensure_ascii=True),
        json.dumps({"conversationId": conversation_id, "windows": window_payload}, default=str, ensure_ascii=True),
    )
    return _window_result_from_llm(result, conversation_id, space_id), provider.name, model


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


def _window_result_from_llm(response: WindowExtractionLLMResponse, conversation_id: str, space_id: str) -> WindowExtractionResult:
    tasks: list[ExtractedTask] = []
    for task in response.tasks:
        extracted = ExtractedTask(
            **task.model_dump(exclude={"fingerprint"}),
            sourceConversationId=conversation_id,
            fingerprint=task.fingerprint,
        )
        extracted.fingerprint = extracted.fingerprint or task_fingerprint(space_id, extracted)
        tasks.append(extracted)

    notes: list[ExtractedNote] = []
    for note in response.notes:
        extracted = ExtractedNote(
            **note.model_dump(exclude={"fingerprint"}),
            sourceConversationId=conversation_id,
            fingerprint=note.fingerprint,
        )
        extracted.fingerprint = extracted.fingerprint or note_fingerprint(space_id, extracted)
        notes.append(extracted)

    return WindowExtractionResult(
        summary=response.summary,
        topics=response.topics,
        importantFacts=response.importantFacts,
        tasks=tasks,
        notes=notes,
        decisions=[
            ExtractedDecision(**decision.model_dump(), sourceConversationId=conversation_id)
            for decision in response.decisions
        ],
        issues=[ExtractedIssue(**issue.model_dump(), sourceConversationId=conversation_id) for issue in response.issues],
        openQuestions=response.openQuestions,
    )


def _trim_payload_for_provider(window_payload: list[dict[str, Any]], provider_name: str) -> list[dict[str, Any]]:
    limit = _provider_input_token_limit(provider_name)
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
        "operation": item.get("operation"),
        "status": item.get("status"),
        "kind": item.get("kind"),
        "ownerText": item.get("ownerText"),
        "dueDateText": item.get("dueDateText"),
        "dueDateResolved": item.get("dueDateResolved"),
        "needsConfirmation": item.get("needsConfirmation"),
        "evidence": item.get("evidence", []),
    }


def _provider_input_token_limit(provider_name: str) -> int:
    if provider_name == "groq":
        return max(1000, settings.GROQ_MAX_TPM - max(512, min(settings.LLM_STRUCTURED_MAX_TOKENS, settings.GROQ_MAX_TPM // 2)) - 500)
    return settings.FINAL_MODEL_INPUT_TOKEN_LIMIT


def _rough_token_count(text: str) -> int:
    return max(1, len(text) // 4)


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
        temperature=settings.LLM_TEMPERATURE,
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
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=_provider_structured_max_tokens(getattr(provider, "name", "")),
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


def _provider_structured_max_tokens(provider_name: str) -> int | None:
    if provider_name == "groq":
        return max(256, min(settings.LLM_STRUCTURED_MAX_TOKENS, settings.GROQ_MAX_TPM // 2))
    return settings.LLM_STRUCTURED_MAX_TOKENS


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
