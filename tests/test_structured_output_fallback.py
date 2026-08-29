import asyncio
import json

import pytest
from pydantic import ValidationError

from services.conversation.agents import WindowExtractionLLMResponse
from services.conversation.extraction_contract import alias_extraction_payload, classify_extraction_outcome
from services.llm.schema_adapter import canonical_json_schema, gemini_response_schema, mistral_json_schema, build_structured_plan
from services.conversation.models import ConversationStatus, ExtractionOutcome, ExtractionRunDocument, ExtractionRunStatus
from services.conversation.workflow import ConversationProcessingWorkflow
from services.llm.errors import StructuredOutputError
from services.llm.fallback import FallbackLLMProvider, LLMRouteCandidate
from services.llm.models import LLMMessage, LLMResponse, LLMUsage, StructuredLLMRequest
from services.llm.openai_compatible import OpenAICompatibleProvider, parse_structured_content
from services.llm.router import LLMCapability
from services.llm.structured_output import (
    MALFORMED_STRUCTURED_OUTPUT,
    STRUCTURED_SCHEMA_ECHO,
    is_schema_echo,
    structured_capabilities,
    structured_modes_for,
)
from tests.test_final_synthesis_persistence import FakeRepository, _chunks
from tests.test_zero_output_extraction import (
    _classifier_units,
    _grounded_payload,
    _run,
    _router,
    _synthesis_from_grounded,
)
from apps.api_gateway.config.setting import settings


@pytest.fixture(autouse=True)
def _keep_legacy_short_session_path(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_EVENT_PIPELINE", False)
    monkeypatch.setattr(settings, "ENABLE_MEETING_PIPELINE", False)



DEFS_ECHO = {"$defs": {"EvidenceSpan": {"type": "object", "properties": {"text": {"type": "string"}}}}}
SCHEMA_ROOT_ECHO = {
    "type": "object",
    "properties": {"semanticUnits": {"type": "array"}, "tasks": {"type": "array"}},
    "required": ["semanticUnits"],
}


class _CannedProvider(OpenAICompatibleProvider):
    def __init__(
        self,
        name: str,
        model: str,
        contents: list[str],
        finish_reasons: list[str | None] | None = None,
        completion_tokens: list[int] | None = None,
        reasoning: list[str] | None = None,
    ):
        super().__init__(
            name=name,
            api_key="test",
            base_url="http://localhost",
            default_model=model,
            timeout_seconds=1,
            max_retries=0,
            max_concurrency=1,
        )
        self.contents = list(contents)
        self.finish_reasons = list(finish_reasons or [])
        self.completion_tokens = list(completion_tokens or [])
        self.reasoning = list(reasoning or [])
        self.seen_formats: list[dict | None] = []
        self.seen_max_tokens: list[int | None] = []

    async def generate(self, request):
        extra = (request.metadata or {}).get("extra_body") or {}
        self.seen_formats.append(extra.get("response_format"))
        self.seen_max_tokens.append(request.max_tokens)
        content = self.contents.pop(0) if self.contents else "{}"
        finish = self.finish_reasons.pop(0) if self.finish_reasons else None
        tokens = self.completion_tokens.pop(0) if self.completion_tokens else 0
        reasoning = self.reasoning.pop(0) if self.reasoning else ""
        if reasoning and not content:
            from services.llm.openai_compatible import _assistant_message_text

            content = _assistant_message_text({"content": content, "reasoning_content": reasoning})
        return LLMResponse(
            content=content,
            provider=self.name,
            model=request.model or self.default_model,
            usage=LLMUsage(completionTokens=tokens, totalTokens=tokens),
            finishReason=finish,
        )


def _structured_request(model: str) -> StructuredLLMRequest:
    return StructuredLLMRequest(
        model=model,
        schema_name="WindowExtractionLLMResponse",
        messages=[LLMMessage(role="user", content="extract")],
    )


def _echo_then_success_router(payload: dict | None = None):
    echo = payload if payload is not None else DEFS_ECHO
    calls = {"primary": 0, "fallback": 0}

    class _Primary:
        name = "mistral"

        async def generate_structured(self, request, schema):
            calls["primary"] += 1
            name = getattr(schema, "__name__", "")
            if name == "SemanticRoleClassificationResponse":
                return schema(units=_classifier_units())
            if name in {"WindowExtractionLLMResponse", "FinalSynthesisLLMResponse"}:
                raise StructuredOutputError(
                    STRUCTURED_SCHEMA_ECHO if is_schema_echo(echo) else MALFORMED_STRUCTURED_OUTPUT
                )
            if name == "MemoryUpdateResponse":
                return schema(currentSummary="updated")
            return schema()

    class _Fallback:
        name = "sarvam"

        async def generate_structured(self, request, schema):
            calls["fallback"] += 1
            name = getattr(schema, "__name__", "")
            if name == "SemanticRoleClassificationResponse":
                return schema(units=_classifier_units())
            if name == "WindowExtractionLLMResponse":
                return schema(**_grounded_payload())
            if name == "MemoryUpdateResponse":
                return schema(currentSummary="updated")
            synthesized = _synthesis_from_grounded(schema)
            if synthesized is not None:
                return synthesized
            return schema()

    class _Router:
        def route(self, capability: LLMCapability):
            if capability == LLMCapability.FALLBACK:
                return _Fallback(), "sarvam-105b"
            return _Primary(), "ministral-3b-2512"

    return _Router(), calls


def test_defs_payload_is_schema_echo_not_extraction_instance():
    assert is_schema_echo(DEFS_ECHO) is True
    assert is_schema_echo(SCHEMA_ROOT_ECHO) is True
    assert is_schema_echo({"semanticUnits": [], "supportedUnitVerdict": "no_supported_units"}) is False
    with pytest.raises((ValueError, ValidationError)):
        alias_extraction_payload(DEFS_ECHO)
    with pytest.raises((ValueError, ValidationError)):
        WindowExtractionLLMResponse.model_validate(DEFS_ECHO)
    with pytest.raises(StructuredOutputError) as caught:
        parse_structured_content(WindowExtractionLLMResponse, json.dumps(DEFS_ECHO))
    assert caught.value.outcome == STRUCTURED_SCHEMA_ECHO


def test_schema_root_echo_is_malformed_structured_output():
    with pytest.raises(StructuredOutputError) as caught:
        parse_structured_content(WindowExtractionLLMResponse, json.dumps(SCHEMA_ROOT_ECHO))
    assert caught.value.outcome == STRUCTURED_SCHEMA_ECHO
    assert classify_extraction_outcome(
        has_units=False,
        technical_failure=True,
        upstream_evidence=True,
        explicit_empty_verdict=False,
        recovery_attempted=False,
    ) == ExtractionOutcome.EXTRACTION_FAILED


def test_gemini_schema_keeps_issue_title_property():
    canonical = canonical_json_schema(WindowExtractionLLMResponse, "WindowExtractionLLMResponse")
    gemini_schema = gemini_response_schema(canonical)
    issue_props = gemini_schema["properties"]["issues"]["items"]["properties"]
    assert "title" in issue_props
    assert "kind" in issue_props
    assert "confidence" in issue_props
    assert "evidence" in issue_props
    plan = build_structured_plan("gemini", "gemini-3.5-flash-lite", WindowExtractionLLMResponse, "WindowExtractionLLMResponse")
    assert plan.attempts[0].extra_body == {}
    assert "title" in plan.attempts[0].response_format["json_schema"]["schema"]["properties"]["issues"]["items"]["properties"]


def test_mistral_schema_keeps_issue_title_and_closes_objects():
    canonical = canonical_json_schema(WindowExtractionLLMResponse, "WindowExtractionLLMResponse")
    mistral_schema = mistral_json_schema(canonical)
    issue_props = mistral_schema["properties"]["issues"]["items"]["properties"]
    assert "title" in issue_props
    assert mistral_schema["additionalProperties"] is False
    assert mistral_schema["properties"]["issues"]["items"]["additionalProperties"] is False


def test_window_extraction_aliases_issue_description_and_keeps_valid_siblings():
    payload = {
        **_grounded_payload(),
        "issues": [
            {
                "description": "Confusion regarding ownership for multiple agencies",
                "evidence": [{"sequenceStart": 1, "sequenceEnd": 1, "text": "who owns this"}],
            },
            {"title": "Missing kind should not fail the window"},
        ],
    }
    parsed, diagnostics = parse_structured_content(WindowExtractionLLMResponse, json.dumps(payload))
    assert diagnostics["parsingOutcome"] == "PARSED_INSTANCE"
    assert parsed.tasks
    assert parsed.notes
    assert len(parsed.issues) == 1
    assert parsed.issues[0].title.startswith("Confusion regarding")
    assert parsed.issues[0].kind == "open_question"


def test_mistral_prefers_json_schema_and_bounds_json_object_recovery():
    capability = structured_capabilities("mistral", "ministral-3b-2512")
    assert capability.supports_json_schema is True
    assert structured_modes_for("mistral", "ministral-3b-2512")[0] == "json_schema"
    assert structured_modes_for("mistral", "ministral-3b-2512")[1] == "json_object"
    assert structured_modes_for("mistral", "mistral-small-latest")[0] == "json_schema"

    payload = json.dumps({"semanticUnits": [], "supportedUnitVerdict": "no_supported_units"})
    provider = _CannedProvider("mistral", "ministral-3b-2512", [payload])
    parsed = asyncio.run(provider.generate_structured(_structured_request("ministral-3b-2512"), WindowExtractionLLMResponse))
    assert parsed.supportedUnitVerdict == "no_supported_units"
    assert provider.seen_formats[0]["type"] == "json_schema"
    assert len(provider.seen_formats) == 1
    assert provider.last_structured_diagnostics["requestedStructuredMode"] == "json_schema"
    assert provider.last_structured_diagnostics["schemaEchoDetected"] is False


def test_json_schema_echo_recovers_to_json_object_on_same_provider():
    echo = json.dumps(DEFS_ECHO)
    instance = json.dumps({"semanticUnits": _grounded_payload()["semanticUnits"], "supportedUnitVerdict": "has_supported_units"})
    provider = _CannedProvider("mistral", "mistral-small-latest", [echo, instance])
    parsed = asyncio.run(provider.generate_structured(_structured_request("mistral-small-latest"), WindowExtractionLLMResponse))
    assert parsed.semanticUnits
    assert provider.seen_formats[0]["type"] == "json_schema"
    assert provider.seen_formats[1]["type"] == "json_object"
    assert provider.last_structured_diagnostics["requestedStructuredMode"] == "json_object"


def test_schema_echo_is_not_valid_empty_and_invokes_next_provider():
    router, calls = _echo_then_success_router()
    result, provider, _ = _run(router)
    assert calls["primary"] >= 1
    assert calls["fallback"] >= 1
    assert provider == "sarvam"
    assert result.extractionOutcome == ExtractionOutcome.SUCCESS
    assert result.semanticUnits
    assert result.extractionDiagnostics["finalSynthesisInvoked"] is True
    assert result.extractionDiagnostics.get("schemaEchoDetected") is not True
    assert result.tasks and result.notes


def test_schema_root_echo_uses_eligible_provider_fallback():
    router, calls = _echo_then_success_router(SCHEMA_ROOT_ECHO)
    result, provider, _ = _run(router)
    assert calls["primary"] >= 1
    assert calls["fallback"] >= 1
    assert provider == "sarvam"
    assert result.extractionOutcome == ExtractionOutcome.SUCCESS
    assert result.semanticUnits
    assert result.extractionDiagnostics["finalSynthesisInvoked"] is True
    assert result.tasks and result.notes


def test_genuine_valid_empty_is_preserved():
    def handler(route, schema):
        name = getattr(schema, "__name__", "")
        if name == "WindowExtractionLLMResponse":
            return schema(semanticUnits=[], supportedUnitVerdict="no_supported_units", rejectedCandidates=[{"reason": "none"}])
        return schema()

    result, _, _ = _run(_router(handler), text="[1] okay\n[2] hmm")
    assert result.extractionOutcome == ExtractionOutcome.VALID_EMPTY_EXTRACTION
    assert not result.semanticUnits


def test_suspicious_empty_still_retries():
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
    assert result.extractionOutcome == ExtractionOutcome.SUCCESS
    assert result.semanticUnits


def test_all_providers_schema_echo_is_extraction_failed_not_valid_empty():
    class _Echo:
        name = "mistral"

        async def generate_structured(self, request, schema):
            raise StructuredOutputError(STRUCTURED_SCHEMA_ECHO)

    class _Router:
        def route(self, capability: LLMCapability):
            return _Echo(), "ministral-3b-2512"

    result, _, _ = _run(_Router())
    assert result.extractionOutcome == ExtractionOutcome.EXTRACTION_FAILED
    assert result.extractionOutcome != ExtractionOutcome.VALID_EMPTY_EXTRACTION
    assert result.extractionDiagnostics["dropStage"] == "structured_schema_echo"
    assert result.extractionDiagnostics["schemaEchoDetected"] is True
    assert result.extractionDiagnostics.get("finalSynthesisInvoked") is not True
    assert not result.semanticUnits


def test_fallback_chain_tries_next_provider_before_giving_up():
    echo = json.dumps(DEFS_ECHO)
    instance = json.dumps({"semanticUnits": _grounded_payload()["semanticUnits"], "supportedUnitVerdict": "has_supported_units"})
    mistral = _CannedProvider("mistral", "ministral-3b-2512", [echo, echo])
    sarvam = _CannedProvider("sarvam", "sarvam-105b", [instance])
    wrapper = FallbackLLMProvider(
        "mistral",
        [
            LLMRouteCandidate(provider=mistral, model="ministral-3b-2512"),
            LLMRouteCandidate(provider=sarvam, model="sarvam-105b"),
        ],
    )
    parsed = asyncio.run(wrapper.generate_structured(_structured_request("ministral-3b-2512"), WindowExtractionLLMResponse))
    assert parsed.semanticUnits
    assert wrapper.last_successful_provider == "sarvam"


def test_schema_echo_recovery_persists_tasks_and_notes():
    router, _calls = _echo_then_success_router()
    repo = FakeRepository(_chunks())
    workflow = ConversationProcessingWorkflow(repo, router)
    run = ExtractionRunDocument(
        conversationId=repo.conversation.id,
        userId=repo.conversation.userId,
        spaceId=repo.conversation.spaceId,
        processingVersion=1,
        provider="mistral",
        model="ministral-3b-2512",
    )
    asyncio.run(workflow._run_short_session_finalization(repo.conversation, run, []))
    diagnostics = run.checkpoints["short_raw_transcript"]
    assert diagnostics["finalSynthesisInvoked"] is True
    assert diagnostics["persistenceOutcome"] == "PERSISTED"
    assert diagnostics["persistedTaskIds"]
    assert diagnostics["persistedNoteIds"]
    assert repo.conversation.status == ConversationStatus.COMPLETED
    assert run.status == ExtractionRunStatus.PUBLISHED


def test_all_providers_malformed_raises_for_queue_retry():
    class _Echo:
        name = "mistral"

        async def generate_structured(self, request, schema):
            raise StructuredOutputError(MALFORMED_STRUCTURED_OUTPUT)

    class _Router:
        def route(self, capability: LLMCapability):
            return _Echo(), "ministral-3b-2512"

    repo = FakeRepository(_chunks())
    workflow = ConversationProcessingWorkflow(repo, _Router())
    run = ExtractionRunDocument(
        conversationId=repo.conversation.id,
        userId=repo.conversation.userId,
        spaceId=repo.conversation.spaceId,
        processingVersion=1,
        provider="mistral",
        model="ministral-3b-2512",
    )
    with pytest.raises(RuntimeError):
        asyncio.run(workflow._run_short_session_finalization(repo.conversation, run, []))
    assert run.status != ExtractionRunStatus.PUBLISHED
    assert repo.conversation.status != ConversationStatus.COMPLETED
