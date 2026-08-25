from types import SimpleNamespace

from services.conversation.models import EvidenceSpan, ExtractedNote, ExtractedTask, WindowExtractionResult


def evidence(text: str, sequence: int) -> EvidenceSpan:
    return EvidenceSpan(sequenceStart=sequence, sequenceEnd=sequence, text=text)


def scripted_task(title: str, body: str, sequence: int, text: str, **kwargs) -> ExtractedTask:
    spans = [evidence(text, sequence)]
    for extra in kwargs.get("extraEvidence", []):
        spans.append(evidence(extra[1], extra[0]))
    return ExtractedTask(
        title=title,
        body=body,
        operation=kwargs.get("operation", "CREATE"),
        ownerText=kwargs.get("ownerText"),
        dueDateText=kwargs.get("dueDateText"),
        dueDateStatus=kwargs.get("dueDateStatus", "none"),
        confidence=kwargs.get("confidence", 0.9),
        sourceConversationId=kwargs.get("sourceConversationId", "conv"),
        evidence=spans,
        origin=kwargs.get("origin", "explicit"),
        changes={
            "semanticArtifactKey": kwargs.get("semanticArtifactKey", title),
            "quality": {"grounded": True, "independentlyUseful": True},
            "synthesisSource": "llm",
        },
    )


def scripted_note(title: str, body: str, sequence: int, text: str, **kwargs) -> ExtractedNote:
    return ExtractedNote(
        title=title,
        body=body,
        confidence=kwargs.get("confidence", 0.9),
        sourceConversationId=kwargs.get("sourceConversationId", "conv"),
        evidence=[evidence(text, sequence)],
        debug={
            "semanticArtifactKey": kwargs.get("semanticArtifactKey", title),
            "quality": {"grounded": True, "independentlyUseful": True},
            "synthesisSource": "llm",
        },
    )


COMPLEX_MEETING_TRANSCRIPT = "\n".join(
    [
        "[0] We reviewed why pending STT chunks can be skipped before STOP drain completes.",
        "[1] The queue currently finalizes while sequence 41 is still processing.",
        "[2] That is informational; it explains the race, not a request yet.",
        "[3] I will write the drain-safe finalizer notes before Friday.",
        "[4] Please also open a ticket to wait for every expected sequence ID before READY_FOR_PROCESSING.",
        "[5] Mira should own the ticket instead of Rahul.",
        "[6] Make the deadline Thursday evening.",
        "[7] We decided to keep raw transcript as the source of truth.",
        "[8] Rahul already sent the drain-safe finalizer notes to the channel.",
        "[9] The login banner copy still needs a separate review by design.",
        "[10] What happens if a transcript job fails after retries?",
        "[11] Earlier I said sequence 41, I meant sequence 14.",
    ]
)


def complex_meeting_result() -> WindowExtractionResult:
    return WindowExtractionResult(
        summary="Drain-safety and sequence accounting were discussed.",
        tasks=[
            scripted_task(
                "Wait for expected STT sequences before finalization",
                "Open and complete a ticket so STOP waits until every expected sequence ID has been processed before READY_FOR_PROCESSING, because pending STT chunks can currently be skipped.",
                4,
                "Please also open a ticket to wait for every expected sequence ID before READY_FOR_PROCESSING.",
                ownerText="Mira",
                dueDateText="Thursday evening",
                dueDateStatus="ambiguous",
                extraEvidence=[
                    (5, "Mira should own the ticket instead of Rahul."),
                    (6, "Make the deadline Thursday evening."),
                ],
            ),
        ],
        notes=[
            scripted_note(
                "Raw transcript remains authoritative",
                "The group decided to keep the raw transcript as the source of truth rather than treating a checkpoint summary as authoritative.",
                7,
                "We decided to keep raw transcript as the source of truth.",
                semanticArtifactKey="raw-transcript-authority",
            ),
            scripted_note(
                "Sequence ID correction",
                "The earlier reference to sequence 41 was corrected to sequence 14.",
                11,
                "Earlier I said sequence 41, I meant sequence 14.",
                semanticArtifactKey="sequence-correction",
            ),
            scripted_note(
                "Unresolved failure behavior",
                "It remains an open question what should happen if a transcript job fails after retries.",
                10,
                "What happens if a transcript job fails after retries?",
                semanticArtifactKey="stt-retry-question",
            ),
        ],
        decisions=[],
        issues=[],
        importantFacts=["Rahul already sent the drain-safe finalizer notes."],
        openQuestions=["What happens if a transcript job fails after retries?"],
    )


def failing_router():
    from services.llm.router import LLMCapability

    class _FailingProvider:
        name = "failing-test-provider"

        async def generate_structured(self, request, schema):
            raise TimeoutError("simulated timeout")

    class _FailingRouter:
        def route(self, capability: LLMCapability):
            return _FailingProvider(), "failing-model"

    return _FailingRouter()


def scripted_router(result: WindowExtractionResult):
    from services.llm.router import LLMCapability

    class _Provider:
        name = "scripted-test-provider"

        async def generate_structured(self, request, schema):
            name = getattr(schema, "__name__", "")
            if name in {"SemanticRoleClassificationResponse", "ConversationUnderstandingResponse", "ExtractionQualityReviewResponse"}:
                return schema()
            if name == "MemoryUpdateResponse":
                return schema(currentSummary=result.summary or "updated")
            if name == "ArtifactReconcileResponse":
                return schema(decisions=[])
            if name == "FinalSynthesisLLMResponse":
                payload = {
                    "summary": result.summary,
                    "narrative": result.narrative or result.summary,
                    "topics": result.topics,
                    "importantFacts": result.importantFacts,
                    "tasks": [
                        {
                            **task.model_dump(),
                            "semanticArtifactKey": (task.changes or {}).get("semanticArtifactKey") or "",
                            "quality": (task.changes or {}).get("quality") or {"grounded": True, "independentlyUseful": True},
                        }
                        for task in result.tasks
                    ],
                    "notes": [
                        {
                            **note.model_dump(),
                            "semanticArtifactKey": (note.debug or {}).get("semanticArtifactKey") or "",
                            "quality": (note.debug or {}).get("quality") or {"grounded": True, "independentlyUseful": True},
                        }
                        for note in result.notes
                    ],
                    "decisions": [item.model_dump() for item in result.decisions],
                    "issues": [item.model_dump() for item in result.issues],
                    "openQuestions": result.openQuestions,
                    "publishVerdict": "PUBLISH" if (result.tasks or result.notes) else "NO_PUBLISHABLE_ARTIFACTS",
                }
                return schema.model_validate(payload)
            payload = {
                "summary": result.summary,
                "narrative": result.narrative or result.summary,
                "topics": result.topics,
                "importantFacts": result.importantFacts,
                "semanticUnits": [unit.model_dump() for unit in result.semanticUnits],
                "tasks": [
                    {
                        **task.model_dump(),
                        "semanticArtifactKey": (task.changes or {}).get("semanticArtifactKey") or "",
                        "quality": (task.changes or {}).get("quality") or {"grounded": True, "independentlyUseful": True},
                    }
                    for task in result.tasks
                ],
                "notes": [
                    {
                        **note.model_dump(),
                        "semanticArtifactKey": (note.debug or {}).get("semanticArtifactKey") or "",
                        "quality": (note.debug or {}).get("quality") or {"grounded": True, "independentlyUseful": True},
                    }
                    for note in result.notes
                ],
                "decisions": [item.model_dump() for item in result.decisions],
                "issues": [item.model_dump() for item in result.issues],
                "openQuestions": result.openQuestions,
            }
            return schema.model_validate(payload)

    class _Router:
        def route(self, capability: LLMCapability):
            return _Provider(), "scripted-model"

    return _Router()
