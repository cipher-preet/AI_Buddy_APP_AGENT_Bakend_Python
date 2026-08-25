import asyncio
from types import SimpleNamespace

from services.conversation import agents
from services.conversation.agents import WindowExtractionLLMResponse
from services.conversation.artifacts import artifacts_from_window
from services.conversation.extraction_contract import (
    alias_extraction_payload,
    classify_extraction_outcome,
    upstream_has_grounded_evidence,
)
from services.conversation.models import ExtractionOutcome
from services.llm.errors import LLMProviderError
from services.llm.router import LLMCapability


GROUNDING_WINDOW = "\n".join(
    [
        "[1] Mira will write the ulari drain notes before Thursday.",
        "[2] Rahul will open the sequence-wait ticket today.",
        "[3] Please assign the banner review to design.",
        "[4] I will confirm the retry budget with ops.",
        "[5] Raw transcript remains the source of truth.",
        "[6] Sequence 41 was corrected to sequence 14.",
        "[7] The login banner still needs a separate review.",
        "[8] Pending STT chunks can currently be skipped.",
        "[9] STOP currently finalizes while a sequence is in flight.",
        "[10] Tide records belong beside each crossing report.",
        "[11] The ulari bridge expands when the tide rises.",
    ]
)


def _window(text: str = GROUNDING_WINDOW):
    return SimpleNamespace(
        conversationId="conv_zero",
        userId="user_1",
        spaceId="space_1",
        id="window_zero",
        windowIndex=0,
        sequenceStart=1,
        sequenceEnd=11,
        text=text,
    )


def _classifier_units():
    actions = [
        (["action", "commitment"], "Mira will write the ulari drain notes before Thursday.", 1, "drain-notes"),
        (["action", "request"], "Rahul will open the sequence-wait ticket today.", 2, "sequence-ticket"),
        (["instruction", "assignment"], "Assign the banner review to design.", 3, "banner-review"),
        (["commitment"], "Confirm the retry budget with ops.", 4, "retry-budget"),
    ]
    notes = [
        (["decision"], "Raw transcript remains the source of truth.", 5, "transcript-authority"),
        (["fact"], "Sequence 41 was corrected to sequence 14.", 6, "sequence-correction"),
        (["requirement"], "The login banner still needs a separate review.", 7, "banner-note"),
        (["problem"], "Pending STT chunks can currently be skipped.", 8, "stt-skip"),
        (["fact"], "STOP currently finalizes while a sequence is in flight.", 9, "stop-race"),
        (["requirement"], "Tide records belong beside each crossing report.", 10, "tide-record"),
        (["fact", "explanation"], "The ulari bridge expands when the tide rises.", 11, "ulari-bridge"),
    ]
    units = []
    for roles, meaning, sequence, key in [*actions, *notes]:
        units.append(
            {
                "roles": roles,
                "topic": key,
                "threadKey": key,
                "normalizedMeaning": meaning,
                "evidenceIds": [sequence],
                "confidence": 0.92,
                "uncertain": False,
            }
        )
    return units


def _span(sequence: int, text: str) -> dict:
    return {"sequenceStart": sequence, "sequenceEnd": sequence, "text": text}


def _grounded_task(sequence: int, text: str, title: str, key: str) -> dict:
    return {
        "title": title,
        "body": f"{text} Complete this supported action with the cited sequence evidence.",
        "operation": "CREATE",
        "confidence": 0.86,
        "origin": "explicit",
        "semanticArtifactKey": key,
        "quality": {"grounded": True, "independentlyUseful": True},
        "evidence": [_span(sequence, text)],
    }


def _grounded_note(sequence: int, text: str, title: str, key: str) -> dict:
    return {
        "title": title,
        "body": f"{text} Keep this as durable meeting context because the cited evidence supports it independently of any task.",
        "confidence": 0.84,
        "semanticArtifactKey": key,
        "quality": {"grounded": True, "independentlyUseful": True},
        "evidence": [_span(sequence, text)],
    }


def _grounded_payload() -> dict:
    task = _grounded_task(1, "Mira will write the ulari drain notes before Thursday.", "Write ulari drain notes", "drain-notes")
    note = _grounded_note(5, "Raw transcript remains the source of truth.", "Raw transcript is authoritative", "transcript-authority")
    return {
        "summary": "Grounded extraction recovered supported units.",
        "semanticUnits": [
            {
                "semanticKey": "drain-notes",
                "kind": "action_candidate",
                "meaning": "Mira will write the ulari drain notes before Thursday.",
                "evidence": [_span(1, "Mira will write the ulari drain notes before Thursday.")],
                "quality": {"grounded": True, "independentlyUseful": True},
            },
            {
                "semanticKey": "transcript-authority",
                "kind": "note_candidate",
                "meaning": "Raw transcript remains the source of truth.",
                "evidence": [_span(5, "Raw transcript remains the source of truth.")],
                "quality": {"grounded": True, "independentlyUseful": True},
            },
        ],
        "tasks": [task],
        "notes": [note],
        "supportedUnitVerdict": "has_supported_units",
    }


def _synthesis_from_grounded(schema):
    name = getattr(schema, "__name__", "")
    if name != "FinalSynthesisLLMResponse":
        return None
    payload = _grounded_payload()
    return schema(
        summary=payload["summary"],
        tasks=payload["tasks"],
        notes=payload["notes"],
        publishVerdict="PUBLISH",
    )


def _router(handler):
    def _dispatch(route, schema):
        result = handler(route, schema)
        if getattr(schema, "__name__", "") != "FinalSynthesisLLMResponse":
            return result
        if getattr(result, "publishVerdict", None) or getattr(result, "tasks", None) or getattr(result, "notes", None):
            return result
        return _synthesis_from_grounded(schema) or result

    class _Provider:
        name = "primary-test-provider"

        async def generate_structured(self, request, schema):
            return _dispatch("primary", schema)

    class _Fallback:
        name = "fallback-test-provider"

        async def generate_structured(self, request, schema):
            return _dispatch("fallback", schema)

    class _Router:
        def route(self, capability: LLMCapability):
            if capability == LLMCapability.FALLBACK:
                return _Fallback(), "fallback-model"
            return _Provider(), "primary-model"

    return _Router()


def _run(router, text: str = GROUNDING_WINDOW, mode: str = "final"):
    agents._SEMANTIC_CLASSIFICATION_CACHE.clear()
    return asyncio.run(agents.extract_window(router, _window(text), context={}, meeting_context={}, mode=mode))


def test_units_alias_is_mapped_onto_semantic_units():
    payload = alias_extraction_payload(
        {
            "units": [
                {
                    "normalizedMeaning": "Mira will write the ulari drain notes before Thursday.",
                    "semanticArtifactKey": "drain-notes",
                    "evidence": [_span(1, "Mira will write the ulari drain notes before Thursday.")],
                }
            ]
        }
    )
    parsed = WindowExtractionLLMResponse.model_validate(payload)
    assert len(parsed.semanticUnits) == 1
    assert parsed.semanticUnits[0].meaning.startswith("Mira will write")


def test_upstream_candidates_make_empty_extraction_suspicious():
    diagnostics = {
        "taskCandidatesGenerated": 4,
        "noteCandidatesGenerated": 7,
        "discussionThreadCount": 8,
        "discussionThreads": [{"roles": ["action"], "semanticEvidenceConfidence": 0.8}],
        "factsExtracted": [{"kind": "fact"}],
    }
    assert upstream_has_grounded_evidence(diagnostics) is True
    assert classify_extraction_outcome(
        has_units=False,
        technical_failure=False,
        upstream_evidence=True,
        explicit_empty_verdict=False,
        recovery_attempted=False,
    ) == ExtractionOutcome.EXTRACTION_FAILED


def test_empty_extraction_with_upstream_candidates_retries_and_keeps_units():
    calls = {"extraction": 0}

    def handler(route, schema):
        name = getattr(schema, "__name__", "")
        if name == "SemanticRoleClassificationResponse":
            return schema(units=_classifier_units())
        if name == "WindowExtractionLLMResponse":
            calls["extraction"] += 1
            if calls["extraction"] == 1:
                return schema()
            return schema(**_grounded_payload())
        return schema()

    result, _, _ = _run(_router(handler))
    assert calls["extraction"] >= 2
    assert result.extractionDiagnostics.get("zeroOutputRecoveryAttempted") is True
    assert result.extractionOutcome == ExtractionOutcome.SUCCESS
    assert result.semanticUnits
    assert result.tasks
    assert result.notes
    artifacts = artifacts_from_window(_window(), result)
    assert artifacts


def test_retry_may_legitimately_conclude_nothing_should_publish():
    calls = {"extraction": 0}

    def handler(route, schema):
        name = getattr(schema, "__name__", "")
        if name == "SemanticRoleClassificationResponse":
            return schema(units=_classifier_units())
        if name == "WindowExtractionLLMResponse":
            calls["extraction"] += 1
            if calls["extraction"] == 1:
                return schema()
            return schema(
                supportedUnitVerdict="no_supported_units",
                rejectedCandidates=[{"reason": "informational_only", "candidateKind": "note"}],
            )
        return schema()

    result, _, _ = _run(_router(handler))
    assert calls["extraction"] >= 2
    assert result.extractionOutcome == ExtractionOutcome.VALID_EMPTY_EXTRACTION
    assert not result.tasks and not result.notes and not result.semanticUnits


def test_truly_empty_conversation_is_valid_empty_without_retry():
    text = "[1] okay\n[2] hmm"
    calls = {"extraction": 0}

    def handler(route, schema):
        name = getattr(schema, "__name__", "")
        if name == "WindowExtractionLLMResponse":
            calls["extraction"] += 1
            return schema()
        return schema()

    result, _, _ = _run(_router(handler), text=text)
    assert calls["extraction"] == 1
    assert result.extractionDiagnostics.get("zeroOutputRecoveryAttempted") is not True
    assert result.extractionOutcome == ExtractionOutcome.VALID_EMPTY_EXTRACTION
    assert not result.tasks and not result.notes


def test_malformed_structured_output_uses_fallback_not_valid_empty():
    def handler(route, schema):
        name = getattr(schema, "__name__", "")
        if route == "primary":
            raise LLMProviderError("malformed structured json", retryable=True, status_code=422)
        if name == "SemanticRoleClassificationResponse":
            return schema(units=_classifier_units())
        if name == "WindowExtractionLLMResponse":
            return schema(**_grounded_payload())
        return schema()

    result, provider, _ = _run(_router(handler))
    assert provider == "fallback-test-provider"
    assert result.extractionOutcome == ExtractionOutcome.SUCCESS
    assert result.semanticUnits


def test_evidence_invalid_units_are_rejected():
    def handler(route, schema):
        name = getattr(schema, "__name__", "")
        if name == "WindowExtractionLLMResponse":
            return schema(
                summary="invalid evidence",
                semanticUnits=[
                    {
                        "semanticKey": "invented",
                        "kind": "fact",
                        "meaning": "Invented meaning",
                        "evidence": [_span(99, "this line is not in the transcript")],
                    }
                ],
                tasks=[],
                notes=[],
            )
        return schema()

    result, _, _ = _run(_router(handler), text="[1] A harmless observation was mentioned.")
    assert not result.semanticUnits
    assert result.extractionDiagnostics.get("evidenceRejectedUnitCount", 0) >= 1


def test_valid_first_pass_units_do_not_retry():
    calls = {"extraction": 0}

    def handler(route, schema):
        name = getattr(schema, "__name__", "")
        if name == "WindowExtractionLLMResponse":
            calls["extraction"] += 1
            return schema(**_grounded_payload())
        return schema()

    result, _, _ = _run(_router(handler))
    assert calls["extraction"] == 1
    assert result.extractionDiagnostics.get("zeroOutputRecoveryAttempted") is not True
    assert result.extractionOutcome == ExtractionOutcome.SUCCESS
    assert result.semanticUnits


def test_zero_units_from_first_provider_recovered_by_fallback_provider():
    def handler(route, schema):
        name = getattr(schema, "__name__", "")
        if name == "SemanticRoleClassificationResponse":
            return schema(units=_classifier_units())
        if name == "WindowExtractionLLMResponse" and route == "primary":
            return schema()
        if name == "WindowExtractionLLMResponse":
            return schema(**_grounded_payload())
        return schema()

    result, provider, _ = _run(_router(handler))
    assert result.extractionDiagnostics.get("zeroOutputRecoverySource") == "eligible_model_fallback"
    assert result.extractionOutcome == ExtractionOutcome.SUCCESS
    assert result.semanticUnits
    assert result.tasks and result.notes
    assert result.extractionDiagnostics["finalSynthesisInvoked"] is True


def test_technical_failure_is_not_converted_to_valid_empty_by_confidence_filter():
    def handler(route, schema):
        name = getattr(schema, "__name__", "")
        if name == "SemanticRoleClassificationResponse":
            return schema(units=_classifier_units())
        raise LLMProviderError("provider exploded", retryable=True, status_code=500)

    result, _, _ = _run(_router(handler))
    assert result.extractionOutcome == ExtractionOutcome.EXTRACTION_FAILED
    assert result.extractionOutcome != ExtractionOutcome.VALID_EMPTY_EXTRACTION
    assert not result.tasks and not result.notes
