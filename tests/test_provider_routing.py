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
from services.llm.schema_adapter import (
    INCOMPLETE_STRUCTURED_OUTPUT,
    QUOTA_UNAVAILABLE,
    SCHEMA_VALIDATION_FAILED,
    STRING_LIST_FIELDS,
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


def test_conversation_intelligence_order_includes_gemini():
    router = LLMRouter(
        {
            "groq": SimpleNamespace(name="groq", configured=True),
            "gemini": SimpleNamespace(name="gemini", configured=True),
            "mistral": SimpleNamespace(name="mistral", configured=True),
            "sarvam": SimpleNamespace(name="sarvam", configured=True),
        }
    )
    expected = ["groq", "gemini", "mistral", "sarvam"]
    for capability in (
        LLMCapability.HIGH_ACCURACY_REASONING,
        LLMCapability.SIMPLE_SUMMARY,
        LLMCapability.VALIDATION,
        LLMCapability.FALLBACK,
        LLMCapability.CHAT_ANSWER,
    ):
        names = [item.provider.name for item in router._cost_optimized_candidates(capability)]
        assert names == expected


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


def test_mistral_modes_are_bounded():
    assert structured_modes_for("mistral", "mistral-small-latest") == ["json_schema", "json_object"]
    assert structured_modes_for("gemini", "gemini-3.5-flash-lite")[0] == "json_schema"
