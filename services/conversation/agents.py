from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from apps.api_gateway.config.setting import settings
from services.conversation.fingerprints import note_fingerprint, task_fingerprint
from services.conversation.models import (
    ConversationSummaryDocument,
    CoverageReport,
    ExtractedDecision,
    ExtractedIssue,
    ExtractedNote,
    ExtractedTask,
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
    tasks = await _structured(
        router,
        "task-extractor-v1",
        TaskExtractionResponse,
        background,
        current,
        LLMCapability.HIGH_ACCURACY_REASONING,
    )
    notes = await _structured(
        router,
        "note-extractor-v1",
        NoteExtractionResponse,
        background,
        current,
        LLMCapability.HIGH_ACCURACY_REASONING,
    )
    decisions = await _structured(
        router,
        "decision-extractor-v1",
        DecisionExtractionResponse,
        background,
        current,
        LLMCapability.HIGH_ACCURACY_REASONING,
    )
    issues = await _structured(
        router,
        "risk-question-extractor-v1",
        IssueExtractionResponse,
        background,
        current,
        LLMCapability.HIGH_ACCURACY_REASONING,
    )
    result = SectionExtractionResult(
        segmentId=segment.segmentId,
        tasks=tasks.tasks,
        notes=notes.notes,
        decisions=decisions.decisions,
        issues=issues.issues,
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
