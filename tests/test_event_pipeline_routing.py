import ast
import asyncio
import inspect
from pathlib import Path

from pydantic import BaseModel

from apps.api_gateway.config.setting import settings
from services.conversation.budget import safe_input_budget
from services.conversation.event_pipeline.cleaning import clean_transcripts
from services.conversation.event_pipeline.embeddings import CachedEmbedder, LexicalEmbedder, ProviderEmbedder
from services.conversation.event_pipeline.events import LLMEventExtractor, ScriptedEventExtractor
from services.conversation.event_pipeline.llm import generate_structured_for_stage
from services.conversation.event_pipeline.microblocks import build_micro_blocks
from services.conversation.event_pipeline.pipeline import run_event_pipeline
from services.conversation.event_pipeline.routing import (
    HIGH_ACCURACY_SYNTHESIS,
    PipelineStage,
    cap_payload,
    capability_for_stage,
    route_for_stage,
    stage_input_cap,
)
from services.conversation.event_pipeline.schemas import AtomicEvent, EventKind, LocalTopic, MicroBlock
from services.conversation.event_pipeline.synthesis import LLMTaskSynthesizer
from services.conversation.event_pipeline.threads import ThreadMembershipVerifier, link_global_threads
from services.conversation.event_pipeline.validation import ArtifactValidationResponse, LLMArtifactValidator
from services.conversation.models import EvidenceSpan, ExtractedTask, STTStatus, TranscriptChunkDocument
from services.llm.errors import LLMProviderError, StructuredOutputError
from services.llm.fallback import FallbackLLMProvider, LLMRouteCandidate
from services.llm.router import LLMCapability, LLMRouter
from services.llm.schema_adapter import MALFORMED_JSON
from services.llm.structured_output import structured_modes_for as public_structured_modes
from tests.test_provider_routing import _ci_router, _candidate_models, _candidate_names


class TrackingRouter(LLMRouter):
    def __init__(self, providers):
        super().__init__(providers)
        self.requested: list[LLMCapability] = []
        self.structured_calls = 0

    def route(self, capability: LLMCapability):
        self.requested.append(capability)
        provider, model = super().route(capability)
        return provider, model


class SchemaProvider:
    def __init__(self, name: str, error: Exception | None = None, result=None):
        self.name = name
        self.configured = True
        self.error = error
        self.result = result
        self.calls = 0
        self.schemas: list[str] = []
        self.models: list[str] = []
        self.last_structured_diagnostics = {}

    async def generate_structured(self, request, schema):
        self.calls += 1
        self.schemas.append(getattr(schema, "__name__", str(schema)))
        self.models.append(getattr(request, "model", None) or "")
        if self.error is not None:
            raise self.error
        if self.result is not None:
            return self.result
        return _default_schema_instance(schema)


def _default_schema_instance(schema: type[BaseModel]):
    name = getattr(schema, "__name__", "")
    if name == "SynthesizedTaskItem":
        return schema(title="Create server ID", body="Please create the server ID.")
    if name == "SynthesizedNoteItem":
        return schema(title="S3 frontend", body="S3 is not reaching the frontend.")
    if name == "AtomicEventLLMResponse":
        return schema(events=[])
    if name == "ArtifactValidationResponse":
        return schema(items=[{"key": "item", "action": "ACCEPT"}])
    if name == "TopicLabelResponse":
        return schema(label="S3 frontend", entities=["S3"])
    if name == "ThreadMembershipVerdict":
        return schema(sameThread=False, ambiguous=False, confidence=0.9)
    try:
        return schema()
    except Exception:
        return schema.model_construct()


def _llm_router(**overrides) -> TrackingRouter:
    providers = {
        "krutrim": SchemaProvider("krutrim"),
        "mistral": SchemaProvider("mistral"),
        "groq": SchemaProvider("groq"),
        "gemini": SchemaProvider("gemini"),
        "sarvam": SchemaProvider("sarvam"),
    }
    providers.update(overrides)
    return TrackingRouter(providers)


def _chunk(sequence: int, text: str) -> TranscriptChunkDocument:
    return TranscriptChunkDocument(
        conversationId="conv",
        userId="u",
        spaceId="s",
        chunkId=f"chunk_{sequence}",
        sequenceNumber=sequence,
        rawText=text,
        sttStatus=STTStatus.COMPLETED,
    )


def _action_event() -> AtomicEvent:
    return AtomicEvent(
        eventId="e-server",
        topicId="T1",
        kind=EventKind.REQUEST,
        meaning="Please create the server ID.",
        object="server ID",
        entities=["Server", "ID"],
        evidence=[EvidenceSpan(sequenceStart=0, sequenceEnd=0, text="Please create the server ID.")],
        sequenceIds=[0],
        conversationId="conv",
        userId="u",
        spaceId="s",
    )


def test_microblocking_does_not_invoke_generative_llm():
    class ForbiddenRouter(LLMRouter):
        def __init__(self):
            super().__init__({"krutrim": SchemaProvider("krutrim")})

        def route(self, capability: LLMCapability):
            raise AssertionError(f"generative LLM routed during micro-blocking: {capability}")

    chunks = [_chunk(0, "S3 frontend nahi reach kar raha"), _chunk(1, "Please create server ID")]
    ledger = clean_transcripts(chunks)
    embedder = CachedEmbedder(LexicalEmbedder())
    blocks = asyncio.run(build_micro_blocks(ledger.useful, embedder))
    assert blocks
    assert "router" not in inspect.signature(build_micro_blocks).parameters
    result = asyncio.run(
        run_event_pipeline(
            chunks,
            "conv",
            "u",
            "s",
            router=ForbiddenRouter(),
            event_extractor=ScriptedEventExtractor(events=[]),
            embedder=embedder,
            polish_with_llm=False,
        )
    )
    assert result.microBlocks
    micro = next(stage for stage in result.observability.stages if stage.name == "micro_blocks")
    assert micro.llmCalls == 0


def test_atomic_event_extraction_routes_through_semantic_extraction():
    router = _llm_router()
    extractor = LLMEventExtractor(router)
    topic = LocalTopic(
        topicId="T1",
        label="server ID",
        sequenceStart=0,
        sequenceEnd=0,
        sequenceIds=[0],
        entities=["Server"],
        text="[0] Please create the server ID",
    )
    asyncio.run(extractor.extract(topic, [], {0: "Please create the server ID"}))
    assert extractor.capability == LLMCapability.SEMANTIC_EXTRACTION
    assert capability_for_stage(PipelineStage.ATOMIC_EVENTS) == LLMCapability.SEMANTIC_EXTRACTION
    assert capability_for_stage(PipelineStage.SEMANTIC_COMPLETENESS) == LLMCapability.SEMANTIC_EXTRACTION
    assert LLMCapability.SEMANTIC_EXTRACTION in router.requested
    assert LLMCapability.FINAL_SYNTHESIS not in router.requested
    assert _candidate_models(_ci_router(), LLMCapability.SEMANTIC_EXTRACTION) == ["gemma-4-31b-it"]


def test_final_synthesis_routes_through_high_accuracy_reasoning():
    router = _llm_router()
    synthesizer = LLMTaskSynthesizer(router)
    asyncio.run(synthesizer.synthesize(_action_event(), None))
    assert capability_for_stage(PipelineStage.TASK_SYNTHESIS) == HIGH_ACCURACY_SYNTHESIS
    assert HIGH_ACCURACY_SYNTHESIS == LLMCapability.FINAL_SYNTHESIS
    assert LLMCapability.FINAL_SYNTHESIS in router.requested
    assert LLMCapability.SEMANTIC_EXTRACTION not in router.requested
    assert _candidate_models(_ci_router(), LLMCapability.FINAL_SYNTHESIS)[0] == settings.CONVERSATION_SYNTHESIS_MODEL
    assert synthesizer.calls == 1


def test_validation_routes_through_validation_capability():
    router = _llm_router()
    validator = LLMArtifactValidator(router)
    task = ExtractedTask(
        title="Create server ID",
        body="Please create the server ID.",
        operation="CREATE",
        confidence=0.7,
        sourceConversationId="conv",
        evidence=[EvidenceSpan(sequenceStart=0, sequenceEnd=0, text="Please create the server ID.")],
        changes={"sourceSemanticUnitIds": ["e-server"]},
    )
    asyncio.run(validator.review(task, [_action_event()], "task"))
    assert capability_for_stage(PipelineStage.VALIDATION) == LLMCapability.VALIDATION
    assert LLMCapability.VALIDATION in router.requested
    assert validator.requested_capabilities == [LLMCapability.VALIDATION]
    names = _candidate_names(_ci_router(), LLMCapability.VALIDATION)
    models = _candidate_models(_ci_router(), LLMCapability.VALIDATION)
    assert names[0] == settings.CONVERSATION_VALIDATION_FALLBACK_PROVIDER
    assert models[0] == settings.CONVERSATION_VALIDATION_FALLBACK_MODEL
    assert names[-1] == settings.CONVERSATION_VALIDATION_PROVIDER
    assert models[-1] == settings.CONVERSATION_VALIDATION_MODEL


def test_event_pipeline_evidence_validation_prefers_gpt_oss_20b():
    assert capability_for_stage(PipelineStage.VALIDATION) == LLMCapability.VALIDATION
    models = _candidate_models(_ci_router(), LLMCapability.VALIDATION)
    names = _candidate_names(_ci_router(), LLMCapability.VALIDATION)
    provider, model = _ci_router().route(LLMCapability.VALIDATION)
    assert names[0] == "krutrim"
    assert models[0] == "gpt-oss-20b"
    assert model == "gpt-oss-20b"
    assert getattr(provider, "name", None) == "krutrim"
    assert names[-1] == settings.CONVERSATION_VALIDATION_PROVIDER
    assert models[-1] == settings.CONVERSATION_VALIDATION_MODEL
    assert models[-1] != "gpt-oss-20b" or len(models) == 1


def test_provider_failure_uses_existing_fallback_chain():
    failing = SchemaProvider(
        "krutrim",
        error=LLMProviderError("krutrim down", retryable=True, status_code=500, failure_reason="HTTP_ERROR"),
    )
    backup = SchemaProvider("mistral")
    router = _llm_router(krutrim=failing, mistral=backup)
    response, provider, model = asyncio.run(
        generate_structured_for_stage(
            router,
            PipelineStage.VALIDATION,
            "event-artifact-validator-v1",
            ArtifactValidationResponse,
            {"title": "Create server ID"},
        )
    )
    assert isinstance(response, ArtifactValidationResponse)
    assert failing.calls == 1
    assert backup.calls == 1
    assert LLMCapability.VALIDATION in router.requested
    assert getattr(provider, "last_successful_provider", None) in {"mistral", None} or backup.calls == 1


def test_context_overflow_does_not_discard_fitting_fallback(monkeypatch):
    monkeypatch.setattr(
        settings,
        "LLM_MODEL_CONTEXT_TOKENS",
        "gpt-oss-120b:2000,gemma-4-31b-it:131072,gpt-oss-20b:131072,ministral-14b-latest:262144",
    )
    router = _ci_router()
    primary_budget = safe_input_budget("krutrim", model="gpt-oss-120b")
    fallback_budget = safe_input_budget("krutrim", model="gemma-4-31b-it")
    assert primary_budget < fallback_budget
    estimated = primary_budget + 500
    assert estimated > primary_budget
    assert estimated <= fallback_budget or fallback_budget >= 1000
    provider, model, capability = route_for_stage(router, PipelineStage.TASK_SYNTHESIS, estimated)
    assert capability == LLMCapability.FINAL_SYNTHESIS
    assert isinstance(provider, FallbackLLMProvider)
    models = [item.model for item in provider.candidates]
    assert "gemma-4-31b-it" in models
    assert models[0] != "gpt-oss-120b" or primary_budget >= estimated
    if primary_budget < estimated <= fallback_budget:
        assert models[0] == "gemma-4-31b-it"
        assert "gpt-oss-120b" not in models


def test_malformed_structured_output_retries_fallback():
    failing = SchemaProvider("krutrim", error=StructuredOutputError(MALFORMED_JSON, "bad json"))
    backup = SchemaProvider("krutrim")
    router = _llm_router(krutrim=failing)
    wrapped = FallbackLLMProvider(
        "krutrim",
        [
            LLMRouteCandidate(provider=failing, model="gpt-oss-120b"),
            LLMRouteCandidate(provider=backup, model="gemma-4-31b-it"),
        ],
    )
    router.route = lambda capability: (wrapped, "gpt-oss-120b")  # type: ignore[method-assign]
    router._cost_optimized_candidates = lambda capability: []  # type: ignore[method-assign]
    response, provider, model = asyncio.run(
        generate_structured_for_stage(
            router,
            PipelineStage.TASK_SYNTHESIS,
            "task-synthesizer-v1",
            __import__("services.conversation.event_pipeline.synthesis", fromlist=["SynthesizedTaskItem"]).SynthesizedTaskItem,
            {"event": {"meaning": "Please create the server ID."}},
        )
    )
    assert failing.calls == 1
    assert backup.calls == 1
    assert response.title


def test_gpt_oss_120b_is_not_called_for_every_event_pair():
    router = _llm_router()
    embedder = CachedEmbedder(LexicalEmbedder())
    events = []
    for index in range(12):
        events.append(
            AtomicEvent(
                eventId=f"e-{index}",
                topicId=f"T{index}",
                kind=EventKind.FACT,
                meaning=f"Unique subject {index} about widget-{index}",
                object=f"widget-{index}",
                entities=[f"Widget{index}"],
                evidence=[EvidenceSpan(sequenceStart=index, sequenceEnd=index, text=f"Unique subject {index} about widget-{index}")],
                sequenceIds=[index],
                conversationId="conv",
                userId="u",
                spaceId="s",
            )
        )
    verifier = ThreadMembershipVerifier(router)
    threads, _links, comparisons = asyncio.run(link_global_threads(events, embedder, verifier=verifier))
    pair_count = 12 * 11 // 2
    assert comparisons < pair_count
    assert verifier.hard_escalations == 0
    assert LLMCapability.FINAL_SYNTHESIS not in router.requested
    assert verifier.verify_calls < pair_count
    assert len(threads) >= 1


def test_event_pipeline_does_not_hardcode_provider_api_calls():
    root = Path("services/conversation/event_pipeline")
    forbidden = (
        "AsyncOpenAI",
        "OpenAI(",
        "httpx.AsyncClient",
        "https://api.openai.com",
        "chat/completions",
        "mistralai",
        "build_krutrim_provider",
        "build_mistral_provider",
    )
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path} hardcodes provider API via {token}"
    routing = (root / "routing.py").read_text(encoding="utf-8")
    assert "router.route(" in routing
    assert "capability_for_stage" in routing
    tree = ast.parse((root / "events.py").read_text(encoding="utf-8"))
    hardcoded_models = {"gemma-4-31b-it", "gpt-oss-120b", "gpt-oss-20b"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value in hardcoded_models:
            raise AssertionError("events.py hardcodes a concrete model")


def test_embedding_calls_are_batched_and_cached():
    class RecordingInner:
        def __init__(self):
            self.calls: list[list[str]] = []

        async def embed_many(self, texts):
            self.calls.append(list(texts))
            return [[float(index), 1.0] for index, _ in enumerate(texts)]

    inner = RecordingInner()
    cached = CachedEmbedder(inner)
    first = asyncio.run(cached.embed_many(["alpha", "beta", "alpha"]))
    second = asyncio.run(cached.embed_many(["alpha", "beta"]))
    assert inner.calls == [["alpha", "beta"]]
    assert cached.batch_calls == 1
    assert cached.cache_hits >= 3
    assert first[0] == second[0]
    assert cached.calls == 2

    source = Path("services/conversation/event_pipeline/embeddings.py").read_text(encoding="utf-8")
    provider_source = source.split("class ProviderEmbedder")[1].split("def default_embedder")[0]
    assert "generate_embeddings" in provider_source
    assert "generate_embedding(" not in provider_source
    assert asyncio.iscoroutinefunction(ProviderEmbedder().embed_many)


def test_structured_stages_prefer_json_schema():
    for model in (
        settings.CONVERSATION_SEMANTIC_MODEL,
        settings.CONVERSATION_SYNTHESIS_MODEL,
        settings.CONVERSATION_VALIDATION_FALLBACK_MODEL,
    ):
        modes = public_structured_modes("krutrim", model)
        assert modes[0] == "json_schema"
    assert public_structured_modes("mistral", settings.CONVERSATION_VALIDATION_MODEL)[0] == "json_schema"
    assert public_structured_modes("krutrim", settings.CONVERSATION_VALIDATION_FALLBACK_MODEL)[0] == "json_schema"


def test_stage_context_budget_never_sends_full_transcript():
    raw = "\n".join(f"[{index}] filler chunk {index} about hallway status" for index in range(300))
    limited = cap_payload(raw, PipelineStage.ATOMIC_EVENTS)
    assert limited.count("\n[") + 1 <= 30
    assert limited.count("\n[") + 1 < 80
    assert stage_input_cap(PipelineStage.ATOMIC_EVENTS) < 20_000
    assert len(limited) < len(raw)
    synthesis = cap_payload(raw, PipelineStage.TASK_SYNTHESIS)
    assert synthesis.count("\n[") + 1 <= 8


def test_model_route_logs_use_intended_capability_names():
    from services.conversation.event_pipeline.observability import bind_observability, reset_observability
    from services.conversation.event_pipeline.routing import capability_log_name, stage_log_name
    from services.conversation.event_pipeline.schemas import PipelineObservability
    from services.conversation.event_pipeline.synthesis import LLMNoteSynthesizer, SynthesizedNoteItem

    assert capability_log_name(PipelineStage.ATOMIC_EVENTS) == "SEMANTIC_EXTRACTION"
    assert capability_log_name(PipelineStage.TASK_SYNTHESIS) == "HIGH_ACCURACY_REASONING"
    assert capability_log_name(PipelineStage.NOTE_SYNTHESIS) == "HIGH_ACCURACY_REASONING"
    assert capability_log_name(PipelineStage.THREAD_HARD) == "HIGH_ACCURACY_REASONING"
    assert capability_log_name(PipelineStage.VALIDATION) == "VALIDATION"
    assert stage_log_name(PipelineStage.ATOMIC_EVENTS) == "atomic_event_extraction"
    assert _candidate_models(_ci_router(), LLMCapability.SEMANTIC_EXTRACTION) == ["gemma-4-31b-it"]
    assert _candidate_models(_ci_router(), LLMCapability.FINAL_SYNTHESIS)[0] == "gpt-oss-120b"
    assert _candidate_models(_ci_router(), LLMCapability.VALIDATION)[0] == settings.CONVERSATION_VALIDATION_FALLBACK_MODEL
    assert _candidate_models(_ci_router(), LLMCapability.VALIDATION)[-1] == settings.CONVERSATION_VALIDATION_MODEL

    obs = PipelineObservability()
    token = bind_observability(obs)
    try:
        router = _llm_router()
        synthesizer = LLMNoteSynthesizer(router)
        asyncio.run(synthesizer.synthesize(_action_event(), None))
    finally:
        reset_observability(token)
    route_logs = [line for line in obs.logs if line.startswith("[MODEL_ROUTE]")]
    assert route_logs
    assert "stage=note_synthesis" in route_logs[0]
    assert "capability=HIGH_ACCURACY_REASONING" in route_logs[0]
    assert "fallback=false" in route_logs[0]
    _ = SynthesizedNoteItem


def test_schema_is_not_weakened_for_invalid_output():
    from services.llm.schema_adapter import MALFORMED_JSON
    from services.conversation.event_pipeline.synthesis import SynthesizedTaskItem

    failing = SchemaProvider("krutrim", error=StructuredOutputError(MALFORMED_JSON, "not json"))
    router = _llm_router(krutrim=failing)
    wrapped = FallbackLLMProvider(
        "krutrim",
        [LLMRouteCandidate(provider=failing, model="gpt-oss-120b")],
    )
    router.route = lambda capability: (wrapped, "gpt-oss-120b")  # type: ignore[method-assign]
    router._cost_optimized_candidates = lambda capability: []  # type: ignore[method-assign]
    try:
        asyncio.run(
            generate_structured_for_stage(
                router,
                PipelineStage.TASK_SYNTHESIS,
                "task-synthesizer-v1",
                SynthesizedTaskItem,
                {"event": {"meaning": "Please create the server ID."}},
            )
        )
        raise AssertionError("invalid structured output must not be accepted")
    except (StructuredOutputError, LLMProviderError):
        pass
    assert failing.calls >= 1

