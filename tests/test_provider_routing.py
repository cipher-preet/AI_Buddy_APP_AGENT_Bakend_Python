import asyncio
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from services.conversation.agents import ConversationUnderstandingResponse, FinalSynthesisLLMResponse
from services.llm.errors import LLMProviderError, StructuredOutputError
from services.llm.fallback import FallbackLLMProvider, LLMRouteCandidate
from services.llm.models import LLMMessage, StructuredLLMRequest
from services.llm.openai_compatible import parse_structured_content
from services.llm.quota import ProviderQuota, quota_guard
from services.llm.router import LLMCapability, LLMRouter
from services.llm.routing_policy import CONVERSATION_INTELLIGENCE_FREE_PROVIDERS, conversation_route_spec
from apps.api_gateway.config.setting import settings
from services.llm.schema_adapter import (
    INCOMPLETE_STRUCTURED_OUTPUT,
    QUOTA_UNAVAILABLE,
    SCHEMA_VALIDATION_FAILED,
    STRING_LIST_FIELDS,
    WIRE_REQUIRED_COLLECTIONS,
    canonical_json_schema,
    build_structured_plan,
)
from services.llm.structured_output import structured_modes_for
from tests.test_final_synthesis_persistence import _synthesized_note, _synthesized_task
from tests.test_structured_output_fallback import _CannedProvider


def _understanding(**overrides) -> dict:
    payload = {name: [] for name in ConversationUnderstandingResponse.model_fields}
    payload["problems"] = ["Duplicate outlet appeared twice"]
    payload.update(overrides)
    return payload


def _understanding_request(model: str = "mistral-small-latest") -> StructuredLLMRequest:
    return StructuredLLMRequest(
        model=model,
        schema_name="ConversationUnderstandingResponse",
        messages=[LLMMessage(role="user", content="understand this meeting window")],
    )


def _synthesis_request(model: str = "mistral-small-latest") -> StructuredLLMRequest:
    return StructuredLLMRequest(
        model=model,
        schema_name="FinalSynthesisLLMResponse",
        messages=[LLMMessage(role="user", content="synthesize tasks and notes")],
    )


def _synthesis_payload(**overrides) -> dict:
    payload = {
        "summary": "Grounded synthesis.",
        "publishVerdict": "PUBLISH",
        "tasks": [_synthesized_task(), {**_synthesized_task(), "title": "Open the sequence-wait ticket", "semanticArtifactKey": "sequence-ticket"}],
        "notes": [_synthesized_note()],
    }
    payload.update(overrides)
    return payload


class _RecordingProvider:
    def __init__(self, name: str, result=None, error: Exception | None = None):
        self.name = name
        self.calls = 0
        self.result = result
        self.error = error
        self.last_structured_diagnostics = {}
        self.configured = True

    async def generate_structured(self, request, schema):
        self.calls += 1
        if self.error is not None:
            raise self.error
        if self.result is not None:
            return self.result
        if schema is ConversationUnderstandingResponse:
            return schema.model_validate(_understanding())
        if schema is FinalSynthesisLLMResponse:
            return schema.model_validate(_synthesis_payload())
        return schema()


def _route_wrapper(providers: list[_RecordingProvider], quotas: dict[str, ProviderQuota] | None = None):
    quotas = quotas or {}
    candidates = [
        LLMRouteCandidate(provider=item, model=f"{item.name}-model", quota=quotas.get(item.name))
        for item in providers
    ]
    return FallbackLLMProvider(candidates[0].provider.name, candidates)


def _ci_router(**overrides):
    providers = {
        "krutrim": SimpleNamespace(name="krutrim", configured=True),
        "mistral": SimpleNamespace(name="mistral", configured=True),
        "groq": SimpleNamespace(name="groq", configured=True),
        "gemini": SimpleNamespace(name="gemini", configured=True),
        "sarvam": SimpleNamespace(name="sarvam", configured=True),
        "openai": SimpleNamespace(name="openai", configured=True),
    }
    providers.update(overrides)
    return LLMRouter(providers)


def _candidate_names(router: LLMRouter, capability: LLMCapability) -> list[str]:
    return [item.provider.name for item in router._cost_optimized_candidates(capability)]


def _candidate_models(router: LLMRouter, capability: LLMCapability) -> list[str]:
    return [item.model for item in router._cost_optimized_candidates(capability)]


def test_semantic_extraction_selects_krutrim_gemma():
    router = _ci_router()
    for capability in (LLMCapability.HIGH_ACCURACY_REASONING, LLMCapability.SEMANTIC_EXTRACTION):
        names = _candidate_names(router, capability)
        models = _candidate_models(router, capability)
        provider, model = router.route(capability)
        assert names == ["krutrim"]
        assert models == ["gemma-4-31b-it"]
        assert getattr(provider, "name", None) == "krutrim"
        assert model == "gemma-4-31b-it"
    assert conversation_route_spec(LLMCapability.SIMPLE_SUMMARY) == [("krutrim", "gemma-4-31b-it")]
    assert conversation_route_spec(LLMCapability.SEMANTIC_EXTRACTION) == [("krutrim", "gemma-4-31b-it")]


def test_final_synthesis_selects_krutrim_gpt_oss_120b():
    router = _ci_router()
    names = _candidate_names(router, LLMCapability.FINAL_SYNTHESIS)
    models = _candidate_models(router, LLMCapability.FINAL_SYNTHESIS)
    provider, model = router.route(LLMCapability.FINAL_SYNTHESIS)
    assert names == ["krutrim", "krutrim"]
    assert models == ["gpt-oss-120b", "gemma-4-31b-it"]
    assert getattr(provider, "name", None) == "krutrim"
    assert model == "gpt-oss-120b"
    assert conversation_route_spec(LLMCapability.FINAL_SYNTHESIS) == [
        ("krutrim", "gpt-oss-120b"),
        ("krutrim", "gemma-4-31b-it"),
    ]


def test_default_validation_selects_krutrim_gpt_oss_20b():
    router = _ci_router()
    names = _candidate_names(router, LLMCapability.VALIDATION)
    models = _candidate_models(router, LLMCapability.VALIDATION)
    provider, model = router.route(LLMCapability.VALIDATION)
    assert names[0] == "krutrim"
    assert models[0] == "gpt-oss-20b"
    assert getattr(provider, "name", None) == "krutrim"
    assert model == "gpt-oss-20b"
    assert conversation_route_spec(LLMCapability.VALIDATION)[0] == (
        settings.CONVERSATION_VALIDATION_FALLBACK_PROVIDER,
        settings.CONVERSATION_VALIDATION_FALLBACK_MODEL,
    )
    assert conversation_route_spec(LLMCapability.VALIDATION)[-1] == (
        settings.CONVERSATION_VALIDATION_PROVIDER,
        settings.CONVERSATION_VALIDATION_MODEL,
    )


def test_hard_validation_fallback_selects_krutrim_gpt_oss_20b():
    router = _ci_router()
    validation = router._cost_optimized_candidates(LLMCapability.VALIDATION)
    fallback = router._cost_optimized_candidates(LLMCapability.FALLBACK)
    assert [(item.provider.name, item.model) for item in validation] == [
        ("krutrim", "gpt-oss-20b"),
        ("mistral", "ministral-14b-latest"),
    ]
    assert [(item.provider.name, item.model) for item in fallback] == [("krutrim", "gpt-oss-20b")]
    provider, model = router.route(LLMCapability.FALLBACK)
    assert getattr(provider, "name", None) == "krutrim"
    assert model == "gpt-oss-20b"


def test_conversation_intelligence_never_selects_free_providers():
    router = _ci_router()
    for capability in (
        LLMCapability.HIGH_ACCURACY_REASONING,
        LLMCapability.SEMANTIC_EXTRACTION,
        LLMCapability.SIMPLE_SUMMARY,
        LLMCapability.FINAL_SYNTHESIS,
        LLMCapability.VALIDATION,
        LLMCapability.FALLBACK,
        LLMCapability.COMPLEX_TASK_MATCHING,
    ):
        names = _candidate_names(router, capability)
        assert names
        for name in names:
            assert name not in CONVERSATION_INTELLIGENCE_FREE_PROVIDERS
            assert name != "xai"
            assert "grok" not in name.lower()


def test_short_and_long_meeting_keep_role_based_routing():
    from services.conversation.agents import _route_for_input

    router = _ci_router()
    for estimated in (200, 80_000):
        semantic_provider, semantic_model = _route_for_input(router, LLMCapability.SEMANTIC_EXTRACTION, estimated)
        reasoning_provider, reasoning_model = _route_for_input(router, LLMCapability.HIGH_ACCURACY_REASONING, estimated)
        synthesis_provider, synthesis_model = _route_for_input(router, LLMCapability.FINAL_SYNTHESIS, estimated)
        assert getattr(semantic_provider, "name", None) == "krutrim"
        assert semantic_model == "gemma-4-31b-it"
        assert getattr(reasoning_provider, "name", None) == "krutrim"
        assert reasoning_model == "gemma-4-31b-it"
        assert getattr(synthesis_provider, "name", None) == "krutrim"
        assert synthesis_model == "gpt-oss-120b"
        assert semantic_model != synthesis_model


def test_canonical_task_note_schemas_unchanged():
    from services.conversation.agents import (
        ExtractionQualityReviewResponse,
        FinalSynthesisLLMResponse,
        WindowExtractionLLMResponse,
    )

    synthesis = canonical_json_schema(FinalSynthesisLLMResponse, "FinalSynthesisLLMResponse")
    extraction = canonical_json_schema(WindowExtractionLLMResponse, "WindowExtractionLLMResponse")
    review = canonical_json_schema(ExtractionQualityReviewResponse, "ExtractionQualityReviewResponse")
    understanding = canonical_json_schema(ConversationUnderstandingResponse, "ConversationUnderstandingResponse")
    assert set(WIRE_REQUIRED_COLLECTIONS["FinalSynthesisLLMResponse"]) <= set(synthesis["properties"])
    assert "tasks" in synthesis["properties"]
    assert "notes" in synthesis["properties"]
    assert "semanticUnits" in extraction["properties"]
    assert "tasks" in extraction["properties"]
    assert "notes" in extraction["properties"]
    assert extraction["properties"]["decisions"]["items"].get("properties")
    assert review["properties"]["decisions"]["items"]["properties"]["kind"]
    assert review["properties"]["missingActionable"]["items"]["type"] == "string"
    assert understanding["properties"]["decisions"]["items"]["type"] == "string"
    plan = build_structured_plan("mistral", "ministral-14b-latest", FinalSynthesisLLMResponse, "FinalSynthesisLLMResponse")
    assert plan.attempts[0].mode == "json_schema"
    krutrim_plan = build_structured_plan("krutrim", "gpt-oss-120b", FinalSynthesisLLMResponse, "FinalSynthesisLLMResponse")
    assert krutrim_plan.attempts[0].mode == "json_schema"


def test_validator_failure_uses_gpt_oss_20b_not_obsolete_provider():
    mistral = _RecordingProvider("mistral", error=StructuredOutputError(SCHEMA_VALIDATION_FAILED))
    krutrim = _RecordingProvider("krutrim")
    groq = _RecordingProvider("groq")
    gemini = _RecordingProvider("gemini")
    sarvam = _RecordingProvider("sarvam")
    wrapper = FallbackLLMProvider(
        "mistral",
        [
            LLMRouteCandidate(provider=mistral, model="ministral-14b-latest"),
            LLMRouteCandidate(provider=krutrim, model="gpt-oss-20b"),
            LLMRouteCandidate(provider=groq, model="openai/gpt-oss-20b"),
            LLMRouteCandidate(provider=gemini, model="gemini-3.5-flash-lite"),
            LLMRouteCandidate(provider=sarvam, model="sarvam-105b"),
        ],
    )
    router = LLMRouter(
        {
            "mistral": mistral,
            "krutrim": krutrim,
            "groq": groq,
            "gemini": gemini,
            "sarvam": sarvam,
        }
    )
    routed, model = router.route(LLMCapability.VALIDATION)
    parsed = asyncio.run(routed.generate_structured(_understanding_request(model), ConversationUnderstandingResponse))
    assert parsed.problems == ["Duplicate outlet appeared twice"]
    assert routed.last_successful_provider == "krutrim"
    assert routed.last_successful_model == "gpt-oss-20b"
    assert groq.calls == 0
    assert gemini.calls == 0
    assert sarvam.calls == 0
    assert krutrim.calls == 1
    assert mistral.calls == 0


def test_provider_error_is_not_valid_empty_extraction():
    from services.conversation import agents
    from services.conversation.extraction_contract import classify_extraction_outcome
    from services.conversation.models import ExtractionOutcome
    from services.llm.errors import LLMProviderError

    outcome = classify_extraction_outcome(
        has_units=False,
        technical_failure=True,
        upstream_evidence=True,
        explicit_empty_verdict=False,
        recovery_attempted=False,
    )
    assert outcome == ExtractionOutcome.EXTRACTION_FAILED
    assert outcome != ExtractionOutcome.VALID_EMPTY_EXTRACTION

    wrapper = _route_wrapper(
        [_RecordingProvider("krutrim", error=LLMProviderError("krutrim down", retryable=True, status_code=500, failure_reason="HTTP_ERROR"))]
    )
    with pytest.raises(LLMProviderError, match="all structured LLM fallbacks failed"):
        asyncio.run(wrapper.generate_structured(_understanding_request("gemma-4-31b-it"), ConversationUnderstandingResponse))
    assert wrapper.last_successful_provider is None

    class _FailingProvider:
        name = "krutrim"

        async def generate_structured(self, request, schema):
            raise LLMProviderError("krutrim down", retryable=False, status_code=500, failure_reason="HTTP_ERROR")

    class _Router:
        def route(self, capability: LLMCapability):
            return _FailingProvider(), "gemma-4-31b-it"

    window = SimpleNamespace(
        conversationId="conv_provider_error",
        userId="user_1",
        spaceId="space_1",
        id="window_provider_error",
        windowIndex=0,
        sequenceStart=1,
        sequenceEnd=1,
        text="[1] Mira will write the ulari drain notes before Thursday.",
        semanticInputDiagnostics=None,
    )
    result, _, _ = asyncio.run(agents.extract_window(_Router(), window, context={}, meeting_context={}, mode="checkpoint"))
    assert result.extractionOutcome == ExtractionOutcome.EXTRACTION_FAILED
    assert result.extractionOutcome != ExtractionOutcome.VALID_EMPTY_EXTRACTION
    assert result.extractionError


def test_unconfigured_krutrim_does_not_fall_back_to_free_models():
    router = LLMRouter(
        {
            "krutrim": SimpleNamespace(name="krutrim", configured=False),
            "mistral": SimpleNamespace(name="mistral", configured=True),
            "groq": SimpleNamespace(name="groq", configured=True),
            "gemini": SimpleNamespace(name="gemini", configured=True),
            "sarvam": SimpleNamespace(name="sarvam", configured=True),
        }
    )
    provider, model = router.route(LLMCapability.HIGH_ACCURACY_REASONING)
    assert getattr(provider, "name", None) == "krutrim"
    assert model == "gemma-4-31b-it"
    assert _candidate_names(router, LLMCapability.HIGH_ACCURACY_REASONING) == []
    assert _candidate_names(router, LLMCapability.SEMANTIC_EXTRACTION) == []
    assert _candidate_names(router, LLMCapability.FINAL_SYNTHESIS) == []
    assert "groq" not in _candidate_names(router, LLMCapability.VALIDATION)
    assert "gemini" not in _candidate_names(router, LLMCapability.VALIDATION)
    assert "sarvam" not in _candidate_names(router, LLMCapability.VALIDATION)
    router = _ci_router()
    names = _candidate_names(router, LLMCapability.CHAT_ANSWER)
    assert names == ["groq", "gemini", "mistral", "sarvam"]


def test_conversation_intelligence_order_includes_gemini():
    router = _ci_router()
    expected = {
        LLMCapability.HIGH_ACCURACY_REASONING: ["krutrim"],
        LLMCapability.SEMANTIC_EXTRACTION: ["krutrim"],
        LLMCapability.SIMPLE_SUMMARY: ["krutrim"],
        LLMCapability.FINAL_SYNTHESIS: ["krutrim", "krutrim"],
        LLMCapability.VALIDATION: ["krutrim", "mistral"],
        LLMCapability.FALLBACK: ["krutrim"],
    }
    for capability, names in expected.items():
        assert _candidate_names(router, capability) == names
    assert _candidate_names(router, LLMCapability.CHAT_ANSWER) == ["groq", "gemini", "mistral", "sarvam"]


def test_conversation_understanding_schema_requires_string_arrays():
    schema = canonical_json_schema(ConversationUnderstandingResponse, "ConversationUnderstandingResponse")
    for field_name in ("problems", "requirements", "decisions", "commitments", "deadlines", "importantFacts", "nextSteps", "owners", "unresolvedQuestions"):
        spec = schema["properties"][field_name]
        assert spec["type"] == "array"
        assert spec["items"]["type"] == "string"
        assert field_name in schema["required"]
    plan = build_structured_plan("mistral", "mistral-small-latest", ConversationUnderstandingResponse, "ConversationUnderstandingResponse")
    assert plan.attempts[0].mode == "json_schema"
    assert plan.attempts[1].mode == "json_object"
    assert plan.attempts[0].response_format["json_schema"]["schema"]["properties"]["problems"]["items"]["type"] == "string"


def test_mistral_object_shaped_problems_is_schema_validation_failed_without_coercion():
    payload = _understanding(problems=[{"description": "Duplicate outlet appeared twice"}])
    with pytest.raises(ValidationError):
        ConversationUnderstandingResponse.model_validate(payload)
    with pytest.raises(StructuredOutputError) as caught:
        parse_structured_content(ConversationUnderstandingResponse, json.dumps(payload))
    assert caught.value.outcome == SCHEMA_VALIDATION_FAILED
    assert "Duplicate outlet appeared twice" not in json.dumps(caught.value.outcome)


def test_mistral_valid_string_problems_succeeds_without_fallback():
    provider = _CannedProvider("mistral", "mistral-small-latest", [json.dumps(_understanding())])
    parsed = asyncio.run(provider.generate_structured(_understanding_request(), ConversationUnderstandingResponse))
    assert parsed.problems == ["Duplicate outlet appeared twice"]
    assert provider.seen_formats[0]["type"] == "json_schema"
    assert len(provider.seen_formats) == 1
    assert provider.last_structured_diagnostics["requestedStructuredMode"] == "json_schema"


def test_mistral_object_problems_bounded_recovery_then_sarvam():
    bad = json.dumps(_understanding(problems=[{"description": "Duplicate outlet appeared twice"}]))
    good = json.dumps(_understanding())
    mistral = _CannedProvider("mistral", "mistral-small-latest", [bad, bad, bad, bad])
    sarvam = _CannedProvider("sarvam", "sarvam-105b", [good])
    wrapper = FallbackLLMProvider(
        "groq",
        [
            LLMRouteCandidate(provider=mistral, model="mistral-small-latest"),
            LLMRouteCandidate(provider=sarvam, model="sarvam-105b"),
        ],
    )
    parsed = asyncio.run(wrapper.generate_structured(_understanding_request(), ConversationUnderstandingResponse))
    assert parsed.problems == ["Duplicate outlet appeared twice"]
    assert wrapper.last_successful_provider == "sarvam"
    assert len(mistral.seen_formats) == 2
    assert mistral.seen_formats[0]["type"] == "json_schema"
    assert mistral.seen_formats[1]["type"] == "json_object"
    assert mistral.last_structured_diagnostics["parsingOutcome"] == SCHEMA_VALIDATION_FAILED
    route = wrapper.last_structured_route
    assert route["providerUsed"] == "sarvam"
    assert any(item["provider"] == "mistral" and item["reason"] == SCHEMA_VALIDATION_FAILED for item in route["failureHistory"])


def test_mistral_recovery_can_accept_corrected_schema():
    bad = json.dumps(_understanding(problems=[{"description": "Duplicate outlet appeared twice"}]))
    good = json.dumps(_understanding())
    provider = _CannedProvider("mistral", "mistral-small-latest", [bad, good])
    parsed = asyncio.run(provider.generate_structured(_understanding_request(), ConversationUnderstandingResponse))
    assert parsed.problems == ["Duplicate outlet appeared twice"]
    assert len(provider.seen_formats) == 2
    assert provider.seen_formats[1]["type"] == "json_object"


def test_extraction_quality_review_accepts_mistral_object_aliases():
    from services.conversation.agents import ExtractionQualityReviewResponse

    payload = {
        "decisions": [
            {
                "type": "task",
                "itemIndex": 0,
                "accept": True,
                "explanation": "Grounded and independently useful.",
                "quality": {"grounded": True, "independentlyUseful": True},
            }
        ],
        "missingActionable": [{"meaning": "Run another long-meeting transcription test."}],
        "missingNotes": [{"description": "Internal testing will use 1-hour and 2-4 hour recordings."}],
        "failed": "true",
    }
    parsed, diagnostics = parse_structured_content(ExtractionQualityReviewResponse, json.dumps(payload))
    assert diagnostics["parsingOutcome"] == "PARSED_INSTANCE"
    assert parsed.decisions[0].kind == "task"
    assert parsed.decisions[0].index == 0
    assert parsed.decisions[0].keep is True
    assert parsed.missingActionable == ["Run another long-meeting transcription test."]
    assert "1-hour" in parsed.missingNotes[0]
    assert parsed.failed is True


def test_mistral_quality_review_schema_keeps_object_decisions():
    from services.conversation.agents import ExtractionQualityReviewResponse

    plan = build_structured_plan(
        "mistral",
        "ministral-14b-latest",
        ExtractionQualityReviewResponse,
        "ExtractionQualityReviewResponse",
    )
    schema = plan.attempts[0].response_format["json_schema"]["schema"]
    assert schema["properties"]["decisions"]["items"]["type"] == "object"
    assert "kind" in schema["properties"]["decisions"]["items"]["properties"]
    assert schema["properties"]["missingActionable"]["items"]["type"] == "string"


def test_final_synthesis_accepts_two_tasks_and_one_note():
    provider = _CannedProvider("mistral", "mistral-small-latest", [json.dumps(_synthesis_payload())])
    parsed = asyncio.run(provider.generate_structured(_synthesis_request(), FinalSynthesisLLMResponse))
    assert len(parsed.tasks) == 2
    assert len(parsed.notes) == 1
    assert parsed.publishVerdict == "PUBLISH"
    assert provider.seen_formats[0]["type"] == "json_schema"


def test_final_synthesis_malformed_json_one_recovery_then_next_provider():
    valid = json.dumps(_synthesis_payload())
    mistral = _CannedProvider("mistral", "mistral-small-latest", ["not-json", "still-not-json"])
    sarvam = _CannedProvider("sarvam", "sarvam-105b", [valid])
    wrapper = FallbackLLMProvider(
        "mistral",
        [
            LLMRouteCandidate(provider=mistral, model="mistral-small-latest"),
            LLMRouteCandidate(provider=sarvam, model="sarvam-105b"),
        ],
    )
    parsed = asyncio.run(wrapper.generate_structured(_synthesis_request(), FinalSynthesisLLMResponse))
    assert len(parsed.tasks) == 2
    assert wrapper.last_successful_provider == "sarvam"
    assert len(mistral.seen_formats) == 2


def test_final_synthesis_incorrect_field_types_are_schema_validation_failed():
    payload = _synthesis_payload(tasks=["not-an-object"])
    with pytest.raises(StructuredOutputError) as caught:
        parse_structured_content(FinalSynthesisLLMResponse, json.dumps(payload))
    assert caught.value.outcome == SCHEMA_VALIDATION_FAILED


def test_final_synthesis_missing_collections_are_incomplete():
    payload = {"publishVerdict": "NO_PUBLISHABLE_ARTIFACTS"}
    with pytest.raises(StructuredOutputError) as caught:
        parse_structured_content(FinalSynthesisLLMResponse, json.dumps(payload))
    assert caught.value.outcome == INCOMPLETE_STRUCTURED_OUTPUT
    with pytest.raises(ValidationError):
        FinalSynthesisLLMResponse.model_validate(payload)


def test_final_synthesis_explicit_empty_is_valid():
    payload = {"publishVerdict": "NO_PUBLISHABLE_ARTIFACTS", "tasks": [], "notes": []}
    parsed, _ = parse_structured_content(FinalSynthesisLLMResponse, json.dumps(payload))
    assert parsed.tasks == []
    assert parsed.notes == []
    assert parsed.publishVerdict == "NO_PUBLISHABLE_ARTIFACTS"


def test_final_synthesis_accepts_content_instead_of_title_body():
    payload = {
        "publishVerdict": "PUBLISH",
        "tasks": [
            {
                "content": "Fix the duplicate task issue today and then run long-meeting tests.",
                "quality": {"independentlyUseful": True},
            }
        ],
        "notes": [
            {
                "content": "The team plans internal testing with 1-hour and 2-4 hour recordings before release.",
                "quality": {"independentlyUseful": True},
            }
        ],
    }
    parsed, diagnostics = parse_structured_content(FinalSynthesisLLMResponse, json.dumps(payload))
    assert diagnostics["parsingOutcome"] == "PARSED_INSTANCE"
    assert parsed.tasks[0].title
    assert "duplicate task" in parsed.tasks[0].body
    assert parsed.notes[0].title
    assert "internal testing" in parsed.notes[0].body


def test_truncated_synthesis_json_is_repaired():
    from services.llm.openai_compatible import _close_truncated_json

    truncated = (
        '{"publishVerdict":"PUBLISH","tasks":[{"title":"Fix duplicate tasks","body":"Fix the duplicate task issue.",'
        '"operation":"CREATE","confidence":0.8,"evidence":[]}],"notes":[{"title":"Release plan",'
        '"body":"Test 1-hour and 2-4 hour meetings first.","confidence":0.8,"evidence":[]}'
    )
    repaired = _close_truncated_json(truncated)
    assert repaired is not None
    parsed, _ = parse_structured_content(FinalSynthesisLLMResponse, truncated)
    assert parsed.tasks[0].title == "Fix duplicate tasks"
    assert parsed.notes[0].title == "Release plan"


def test_truncated_output_retries_higher_budget_then_falls_back_to_gemma():
    truncated = '{"tasks":[{"title":"Fix duplicate tasks"'
    valid = json.dumps(_synthesis_payload())
    oss = _CannedProvider(
        "krutrim",
        "gpt-oss-120b",
        [truncated, truncated],
        finish_reasons=["length", "length"],
        completion_tokens=[8192, 16000],
    )
    gemma = _CannedProvider("krutrim", "gemma-4-31b-it", [valid])
    wrapper = FallbackLLMProvider(
        "krutrim",
        [
            LLMRouteCandidate(provider=oss, model="gpt-oss-120b"),
            LLMRouteCandidate(provider=gemma, model="gemma-4-31b-it"),
        ],
    )
    parsed = asyncio.run(wrapper.generate_structured(_synthesis_request("gpt-oss-120b"), FinalSynthesisLLMResponse))
    assert len(parsed.tasks) == 2
    assert wrapper.last_successful_model == "gemma-4-31b-it"
    assert oss.seen_max_tokens == [8192, 16000]
    assert len(oss.seen_formats) == 2
    assert gemma.seen_formats


def test_truncated_synthesis_succeeds_after_raising_max_tokens():
    truncated = '{"tasks":[{"title":"Fix duplicate tasks"'
    valid = json.dumps(_synthesis_payload())
    oss = _CannedProvider(
        "krutrim",
        "gpt-oss-120b",
        [truncated, valid],
        finish_reasons=["length", "stop"],
        completion_tokens=[8192, 1200],
    )
    gemma = _CannedProvider("krutrim", "gemma-4-31b-it", [valid])
    wrapper = FallbackLLMProvider(
        "krutrim",
        [
            LLMRouteCandidate(provider=oss, model="gpt-oss-120b"),
            LLMRouteCandidate(provider=gemma, model="gemma-4-31b-it"),
        ],
    )
    parsed = asyncio.run(wrapper.generate_structured(_synthesis_request("gpt-oss-120b"), FinalSynthesisLLMResponse))
    assert len(parsed.tasks) == 2
    assert wrapper.last_successful_model == "gpt-oss-120b"
    assert oss.seen_max_tokens == [8192, 16000]
    assert gemma.seen_formats == []


def test_synthesis_uses_reasoning_content_when_message_content_empty():
    from services.llm.openai_compatible import _assistant_message_text

    payload = json.dumps(_synthesis_payload())
    text = _assistant_message_text({"content": "", "reasoning_content": payload})
    parsed, _ = parse_structured_content(FinalSynthesisLLMResponse, text)
    assert len(parsed.tasks) == 2
    assert len(parsed.notes) == 1


def test_synthesis_output_budget_is_higher_than_generic_structured():
    from apps.api_gateway.config.setting import settings
    from services.conversation.agents import _provider_structured_max_tokens
    from services.llm.openai_compatible import _output_budgets_for

    assert settings.LLM_SYNTHESIS_OUTPUT_START_TOKENS == 8192
    assert settings.LLM_SYNTHESIS_OUTPUT_MAX_TOKENS == 16000
    start = _provider_structured_max_tokens("krutrim", "FinalSynthesisLLMResponse")
    assert start == 8192
    assert start < settings.LLM_SYNTHESIS_OUTPUT_MAX_TOKENS
    assert start > _provider_structured_max_tokens("krutrim", "ConversationUnderstandingResponse")
    assert _output_budgets_for("FinalSynthesisLLMResponse", start) == [8192, 16000]
    assert _output_budgets_for("ConversationUnderstandingResponse", 4096) == [4096]


def test_groq_success_does_not_call_gemini():
    groq = _RecordingProvider("groq")
    gemini = _RecordingProvider("gemini")
    wrapper = _route_wrapper([groq, gemini])
    asyncio.run(wrapper.generate_structured(_understanding_request("openai/gpt-oss-20b"), ConversationUnderstandingResponse))
    assert groq.calls == 1
    assert gemini.calls == 0
    assert wrapper.last_successful_provider == "groq"


def test_groq_quota_exhausted_calls_gemini_immediately():
    quota_guard.reset()
    groq = _RecordingProvider("groq", error=LLMProviderError("groq should not be called", retryable=True, status_code=429))
    gemini = _RecordingProvider("gemini")
    mistral = _RecordingProvider("mistral")
    wrapper = _route_wrapper(
        [groq, gemini, mistral],
        {"groq": ProviderQuota(tpm=1)},
    )
    request = StructuredLLMRequest(
        model="openai/gpt-oss-20b",
        schema_name="ConversationUnderstandingResponse",
        messages=[LLMMessage(role="user", content="token estimate padding " * 40)],
        max_tokens=8,
    )
    asyncio.run(wrapper.generate_structured(request, ConversationUnderstandingResponse))
    assert groq.calls == 0
    assert gemini.calls == 1
    assert mistral.calls == 0
    assert wrapper.last_successful_provider == "gemini"
    history = wrapper.last_structured_route["failureHistory"]
    assert history[0]["provider"] == "groq"
    assert history[0]["reason"] == QUOTA_UNAVAILABLE
    assert wrapper.last_structured_route["providerUsed"] == "gemini"


def test_groq_http_failure_calls_gemini():
    groq = _RecordingProvider("groq", error=LLMProviderError("groq down", retryable=True, status_code=500, failure_reason="HTTP_ERROR"))
    gemini = _RecordingProvider("gemini")
    wrapper = _route_wrapper([groq, gemini])
    asyncio.run(wrapper.generate_structured(_understanding_request(), ConversationUnderstandingResponse))
    assert groq.calls == 1
    assert gemini.calls == 1
    assert wrapper.last_successful_provider == "gemini"


def test_groq_schema_failure_calls_gemini():
    groq = _RecordingProvider("groq", error=StructuredOutputError(SCHEMA_VALIDATION_FAILED))
    gemini = _RecordingProvider("gemini")
    wrapper = _route_wrapper([groq, gemini])
    asyncio.run(wrapper.generate_structured(_understanding_request(), ConversationUnderstandingResponse))
    assert groq.calls == 1
    assert gemini.calls == 1
    assert wrapper.last_successful_provider == "gemini"


def test_gemini_success_does_not_call_mistral():
    groq = _RecordingProvider("groq", error=StructuredOutputError(SCHEMA_VALIDATION_FAILED))
    gemini = _RecordingProvider("gemini")
    mistral = _RecordingProvider("mistral")
    wrapper = _route_wrapper([groq, gemini, mistral])
    asyncio.run(wrapper.generate_structured(_understanding_request(), ConversationUnderstandingResponse))
    assert gemini.calls == 1
    assert mistral.calls == 0
    assert wrapper.last_successful_provider == "gemini"


def test_gemini_failure_calls_mistral():
    groq = _RecordingProvider("groq", error=StructuredOutputError(SCHEMA_VALIDATION_FAILED))
    gemini = _RecordingProvider("gemini", error=StructuredOutputError(SCHEMA_VALIDATION_FAILED))
    mistral = _RecordingProvider("mistral")
    wrapper = _route_wrapper([groq, gemini, mistral])
    asyncio.run(wrapper.generate_structured(_understanding_request(), ConversationUnderstandingResponse))
    assert mistral.calls == 1
    assert wrapper.last_successful_provider == "mistral"


def test_mistral_failure_calls_sarvam():
    groq = _RecordingProvider("groq", error=StructuredOutputError(SCHEMA_VALIDATION_FAILED))
    gemini = _RecordingProvider("gemini", error=StructuredOutputError(SCHEMA_VALIDATION_FAILED))
    mistral = _RecordingProvider("mistral", error=StructuredOutputError(SCHEMA_VALIDATION_FAILED))
    sarvam = _RecordingProvider("sarvam")
    wrapper = _route_wrapper([groq, gemini, mistral, sarvam])
    asyncio.run(wrapper.generate_structured(_understanding_request(), ConversationUnderstandingResponse))
    assert sarvam.calls == 1
    assert wrapper.last_successful_provider == "sarvam"
    assert wrapper.last_structured_route["providerUsed"] == "sarvam"


def test_all_providers_fail_returns_provider_failure():
    providers = [
        _RecordingProvider(name, error=StructuredOutputError(SCHEMA_VALIDATION_FAILED))
        for name in ("groq", "gemini", "mistral", "sarvam")
    ]
    wrapper = _route_wrapper(providers)
    with pytest.raises(LLMProviderError, match="all structured LLM fallbacks failed"):
        asyncio.run(wrapper.generate_structured(_understanding_request(), ConversationUnderstandingResponse))
    assert [item.calls for item in providers] == [1, 1, 1, 1]
    assert wrapper.last_successful_provider is None
    reasons = [item["reason"] for item in wrapper.last_structured_route["failureHistory"]]
    assert reasons == [SCHEMA_VALIDATION_FAILED] * 4


def test_gemini_adapter_uses_openai_compatible_json_schema():
    plan = build_structured_plan("gemini", "gemini-3.5-flash-lite", ConversationUnderstandingResponse, "ConversationUnderstandingResponse")
    assert plan.attempts[0].mode == "json_schema"
    assert plan.attempts[0].extra_body == {}
    assert plan.attempts[0].response_format["type"] == "json_schema"
    assert "json_schema" in plan.attempts[0].response_format
    provider = _CannedProvider("gemini", "gemini-3.5-flash-lite", [json.dumps(_understanding())])
    parsed = asyncio.run(provider.generate_structured(_understanding_request("gemini-3.5-flash-lite"), ConversationUnderstandingResponse))
    assert parsed.problems == ["Duplicate outlet appeared twice"]
    extra = provider.seen_formats[0]
    assert extra["type"] == "json_schema"


def test_groq_strict_schema_marks_required_and_closed_objects():
    schema = canonical_json_schema(ConversationUnderstandingResponse, "ConversationUnderstandingResponse")
    plan = build_structured_plan("groq", "openai/gpt-oss-20b", ConversationUnderstandingResponse, "ConversationUnderstandingResponse")
    assert plan.attempts[0].mode == "json_schema"
    groq_schema = plan.attempts[0].response_format["json_schema"]["schema"]
    assert plan.attempts[0].response_format["json_schema"]["strict"] is True
    assert groq_schema["additionalProperties"] is False
    assert set(groq_schema["required"]) == set(groq_schema["properties"].keys())
    assert "problems" in schema["required"]


def test_structured_temperature_is_deterministic():
    plan = build_structured_plan("mistral", "mistral-small-latest", ConversationUnderstandingResponse, "ConversationUnderstandingResponse")
    assert all(attempt.temperature == 0 for attempt in plan.attempts)


def test_string_list_prompt_fields_match_schema():
    for name in ("problems", "requirements", "decisions", "commitments", "deadlines", "importantFacts", "nextSteps", "owners", "unresolvedQuestions"):
        assert name in ConversationUnderstandingResponse.model_fields
        assert name in STRING_LIST_FIELDS


def test_json_object_cannot_bypass_canonical_schema():
    invalid = json.dumps(_synthesis_payload(tasks=["not-an-object"]))
    with pytest.raises(StructuredOutputError) as caught:
        parse_structured_content(FinalSynthesisLLMResponse, invalid)
    assert caught.value.outcome == SCHEMA_VALIDATION_FAILED

    plan = build_structured_plan("krutrim", "gpt-oss-120b", FinalSynthesisLLMResponse, "FinalSynthesisLLMResponse")
    json_object = next(item for item in plan.attempts if item.mode == "json_object")
    assert json_object.response_format == {"type": "json_object"}
    assert json_object.parsing_strategy == "canonical_pydantic"
    assert "tasks" in (json_object.schema.get("properties") or {})
    assert "tasks" in json_object.instruction

    provider = _CannedProvider("krutrim", "gpt-oss-120b", [invalid, invalid, invalid])
    with pytest.raises(StructuredOutputError) as structured:
        asyncio.run(provider.generate_structured(_synthesis_request("gpt-oss-120b"), FinalSynthesisLLMResponse))
    assert structured.value.outcome == SCHEMA_VALIDATION_FAILED
    modes = [("plain_json_prompt" if item is None else item.get("type")) for item in provider.seen_formats]
    assert "json_object" in modes
    assert modes == ["json_schema", "json_object", "plain_json_prompt"]


def test_model_specific_context_limits_override_provider_wide_krutrim_budget(monkeypatch):
    from apps.api_gateway.config.setting import settings
    from services.conversation.budget import provider_context_limit, safe_input_budget

    monkeypatch.setattr(
        settings,
        "LLM_MODEL_CONTEXT_TOKENS",
        "gemma-4-31b-it:131072,gpt-oss-120b:65536,gpt-oss-20b:131072,ministral-14b-latest:262144,ministral-14b-2512:262144",
    )
    monkeypatch.setattr(
        settings,
        "LLM_PROVIDER_CONTEXT_TOKENS",
        "groq:8192,gemini:1048576,mistral:262144,sarvam:32768,openai:128000,anthropic:200000,krutrim:65536",
    )
    assert provider_context_limit("krutrim", "gemma-4-31b-it") == 131072
    assert provider_context_limit("krutrim", "gpt-oss-120b") == 65536
    assert provider_context_limit("krutrim", "gpt-oss-20b") == 131072
    assert provider_context_limit("mistral", "ministral-14b-latest") == 262144
    assert provider_context_limit("krutrim", "gpt-oss-120b") != provider_context_limit("krutrim", "gemma-4-31b-it")
    assert safe_input_budget("krutrim", model="gpt-oss-120b") < safe_input_budget("krutrim", model="gemma-4-31b-it")
    assert provider_context_limit("krutrim") == 65536


def test_mistral_modes_are_bounded():
    assert structured_modes_for("mistral", "mistral-small-latest") == ["json_schema", "json_object"]
    assert structured_modes_for("gemini", "gemini-3.5-flash-lite")[0] == "json_schema"
    assert structured_modes_for("krutrim", "gemma-4-31b-it") == ["json_schema", "json_object", "plain_json_prompt"]
    assert structured_modes_for("krutrim", "gpt-oss-120b") == ["json_schema", "json_object", "plain_json_prompt"]
