import asyncio
from datetime import timedelta

from pydantic import BaseModel

from services.conversation.fingerprints import task_fingerprint
from services.conversation import agents
from services.conversation.models import (
    ConversationDocument,
    ConversationSummaryDocument,
    ConversationStatus,
    CoverageReport,
    EvidenceSpan,
    ExtractedDecision,
    ExtractedIssue,
    ExtractedNote,
    ExtractedTask,
    ExtractionRunDocument,
    ExtractionRunStatus,
    SectionExtractionResult,
    Segment,
    SpaceMemoryDocument,
    TranscriptChunkDocument,
    assert_valid_transition,
    utc_now,
)
from services.conversation.repository import ConversationRepository
from services.conversation.service import ConversationService
from services.conversation.transcript import assemble_transcript, detect_missing_sequences, segment_transcript
from services.conversation.workflow import ConversationProcessingWorkflow
from services.conversation.workflow_state import ConversationGraphState
from services.llm.models import LLMMessage, StructuredLLMRequest
from services.llm.openai_compatible import OpenAICompatibleProvider
from services.queue.streams import EventEnvelope
from apps.api_gateway.main import app
from apps.api_gateway.workers import conversation_workers


def test_invalid_state_transition_rejected():
    try:
        assert_valid_transition(ConversationStatus.COMPLETED, ConversationStatus.PROCESSING)
    except ValueError:
        return
    raise AssertionError("completed conversations must not be reprocessed without explicit retry state")


def test_sequence_gap_detection():
    assert detect_missing_sequences([0, 2, 4], 4) == [1, 3]


def test_transcript_ordering_and_overlap_normalization():
    chunks = [
        TranscriptChunkDocument(
            conversationId="conv_1",
            userId="user_1",
            spaceId="space_1",
            chunkId="b",
            sequenceNumber=1,
            rawText="callback tomorrow. Also update notes.",
        ),
        TranscriptChunkDocument(
            conversationId="conv_1",
            userId="user_1",
            spaceId="space_1",
            chunkId="a",
            sequenceNumber=0,
            rawText="Rahul will test the payment callback tomorrow.",
        ),
    ]
    assembled = assemble_transcript(chunks)
    assert assembled.raw_transcript.startswith("[0]")
    assert "[1] Also update notes." in assembled.normalized_transcript


def test_segmenter_preserves_sequence_ranges():
    chunks = [
        TranscriptChunkDocument(
            conversationId="conv_1",
            userId="user_1",
            spaceId="space_1",
            chunkId=str(index),
            sequenceNumber=index,
            rawText=f"Sentence {index}. Another sentence {index}.",
        )
        for index in range(5)
    ]
    segments = segment_transcript("conv_1", chunks, target_tokens=8, overlap_ratio=0.1, max_segments=10)
    assert len(segments) > 1
    assert segments[0].sequenceStart == 0
    assert segments[-1].sequenceEnd == 4


def test_task_fingerprint_is_stable():
    task = ExtractedTask(
        title="Complete payment callback testing",
        operation="CREATE",
        ownerText="Rahul",
        dueDateResolved="2026-07-29",
        confidence=0.94,
        sourceConversationId="conv_1",
        evidence=[EvidenceSpan(sequenceStart=18, sequenceEnd=20, text="Rahul will test it tomorrow.")],
    )
    assert task_fingerprint("space_1", task) == task_fingerprint("space_1", task)


def test_legacy_speech_listening_end_route_is_registered():
    paths = {route.path for route in app.routes}
    assert "/api/v1/speech/listening/start" in paths
    assert "/api/v1/speech/listening/end" in paths


class TinyStructuredResponse(BaseModel):
    ok: bool


class FakeLLMHttpResponse:
    def json(self):
        return {
            "choices": [{"message": {"content": '{"ok": true}'}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 3, "total_tokens": 4},
        }


def test_sarvam_structured_requests_are_capped_to_subscription_max():
    asyncio.run(_sarvam_structured_requests_are_capped_to_subscription_max())


async def _sarvam_structured_requests_are_capped_to_subscription_max():
    captured_payloads = []
    provider = OpenAICompatibleProvider(
        name="sarvam",
        api_key="test-key",
        base_url="https://example.test/v1",
        default_model="sarvam-105b",
        timeout_seconds=1,
        max_retries=0,
        max_concurrency=1,
        auth_header="api-subscription-key",
        auth_prefix="",
        max_tokens_limit=4096,
    )

    async def fake_post(path, payload):
        captured_payloads.append(payload)
        return FakeLLMHttpResponse()

    provider._post_with_retries = fake_post
    try:
        result = await provider.generate_structured(
            StructuredLLMRequest(
                messages=[LLMMessage(role="user", content="Return ok true.")],
                model="sarvam-105b",
                max_tokens=8192,
            ),
            TinyStructuredResponse,
        )
    finally:
        await provider._client.aclose()

    assert result.ok is True
    assert captured_payloads[0]["max_tokens"] == 4096


class FakeAsyncCollection:
    def __init__(self):
        self.updated = []
        self.deleted = []
        self.inserted = []

    async def update_one(self, *args, **kwargs):
        self.updated.append((args, kwargs))

    async def delete_many(self, query):
        self.deleted.append(query)

    async def insert_many(self, docs, ordered=False):
        self.inserted.extend(docs)


class FakeDb(dict):
    def __init__(self):
        super().__init__()
        self.extraction_runs = FakeAsyncCollection()

    def __getitem__(self, name):
        if name not in self:
            self[name] = FakeAsyncCollection()
        return super().__getitem__(name)


def test_save_extraction_run_writes_all_split_staged_collections():
    asyncio.run(_save_extraction_run_writes_all_split_staged_collections())


async def _save_extraction_run_writes_all_split_staged_collections():
    db = FakeDb()
    run = ExtractionRunDocument(
        conversationId="507f1f77bcf86cd799439011",
        userId="507f1f77bcf86cd799439012",
        spaceId="507f1f77bcf86cd799439013",
        processingVersion=1,
        provider="test",
        model="test-model",
        stagedTasks=[
            ExtractedTask(
                title="Ship the fix",
                operation="CREATE",
                confidence=0.9,
                sourceConversationId="507f1f77bcf86cd799439011",
                evidence=[EvidenceSpan(sequenceStart=0, sequenceEnd=0, text="Ship the fix.")],
            )
        ],
        stagedNotes=[
            ExtractedNote(
                title="Root cause",
                body="Split staged tables skipped some output types.",
                confidence=0.9,
                sourceConversationId="507f1f77bcf86cd799439011",
                evidence=[EvidenceSpan(sequenceStart=0, sequenceEnd=0, text="Split staged tables.")],
            )
        ],
        stagedDecisions=[
            ExtractedDecision(
                title="Use split collections",
                status="confirmed_decision",
                confidence=0.9,
                sourceConversationId="507f1f77bcf86cd799439011",
                evidence=[EvidenceSpan(sequenceStart=0, sequenceEnd=0, text="Use split collections.")],
            )
        ],
        stagedIssues=[
            ExtractedIssue(
                title="Status can hang",
                kind="risk",
                confidence=0.9,
                sourceConversationId="507f1f77bcf86cd799439011",
                evidence=[EvidenceSpan(sequenceStart=0, sequenceEnd=0, text="Status can hang.")],
            )
        ],
    )

    await ConversationRepository(db).save_extraction_run(run)

    for collection_name in ("stagedTasks", "stagedNotes", "stagedDecisions", "stagedIssues"):
        assert db[collection_name].deleted == [{"extractionRunId": run.id}]
        assert len(db[collection_name].inserted) == 1


def test_save_extraction_run_clears_all_split_staged_collections_on_publish():
    asyncio.run(_save_extraction_run_clears_all_split_staged_collections_on_publish())


async def _save_extraction_run_clears_all_split_staged_collections_on_publish():
    db = FakeDb()
    run = ExtractionRunDocument(
        conversationId="507f1f77bcf86cd799439011",
        userId="507f1f77bcf86cd799439012",
        spaceId="507f1f77bcf86cd799439013",
        processingVersion=1,
        provider="test",
        model="test-model",
        status=ExtractionRunStatus.PUBLISHED,
    )

    await ConversationRepository(db).save_extraction_run(run)

    for collection_name in ("stagedTasks", "stagedNotes", "stagedDecisions", "stagedIssues"):
        assert db[collection_name].deleted == [{"extractionRunId": run.id}]
        assert db[collection_name].inserted == []


class HangingWorkflow:
    def __init__(self, repository):
        self.repository = repository

    async def run(self, conversation_id):
        await asyncio.Event().wait()


class TimeoutRepository:
    def __init__(self, db):
        self.active_failed = []
        self.conversation_failed = []

    async def mark_active_extraction_run_failed(self, conversation_id, error):
        self.active_failed.append((conversation_id, str(error)))

    async def mark_conversation_failed(self, conversation_id, error):
        self.conversation_failed.append((conversation_id, str(error)))


def test_processing_worker_times_out_stuck_workflow(monkeypatch):
    repository = TimeoutRepository(None)
    monkeypatch.setattr(conversation_workers, "ConversationRepository", lambda db: repository)
    monkeypatch.setattr(conversation_workers, "ConversationProcessingWorkflow", HangingWorkflow)
    monkeypatch.setattr(conversation_workers, "get_database", lambda: object())
    monkeypatch.setattr(conversation_workers.settings, "CONVERSATION_PROCESSING_TIMEOUT_SECONDS", 0.01)

    event = EventEnvelope(
        eventType="conversation.processing.requested",
        correlationId="conv_1",
        userId="user_1",
        spaceId="space_1",
        conversationId="conv_1",
    )

    asyncio.run(conversation_workers.handle_processing_event(event))

    assert repository.active_failed[0][0] == "conv_1"
    assert repository.conversation_failed[0][0] == "conv_1"
    assert "timed out" in repository.conversation_failed[0][1]


class StaleStatusRepository:
    def __init__(self):
        self.conversation = ConversationDocument(
            _id="conv_1",
            userId="user_1",
            spaceId="space_1",
            status=ConversationStatus.PROCESSING,
            updatedAt=utc_now() - timedelta(seconds=60),
            activeExtractionRunId="run_1",
        )
        self.active_failed = []
        self.conversation_failed = []

    async def get_conversation(self, conversation_id):
        return self.conversation

    async def get_extraction_run(self, run_id):
        return ExtractionRunDocument(
            _id=run_id,
            conversationId="conv_1",
            userId="user_1",
            spaceId="space_1",
            processingVersion=1,
            provider="test",
            model="test-model",
            status=ExtractionRunStatus.FAILED,
            validationErrors=[{"code": "WORKFLOW_EXCEPTION", "message": "timed out"}],
        )

    async def mark_active_extraction_run_failed(self, conversation_id, error):
        self.active_failed.append((conversation_id, str(error)))

    async def mark_conversation_failed(self, conversation_id, error):
        self.conversation_failed.append((conversation_id, str(error)))
        self.conversation.status = ConversationStatus.FAILED


def test_status_marks_stale_processing_conversation_failed(monkeypatch):
    repository = StaleStatusRepository()
    monkeypatch.setattr("services.conversation.service.settings.CONVERSATION_PROCESSING_TIMEOUT_SECONDS", 1)

    status = asyncio.run(ConversationService(repository=repository).status("conv_1", "user_1", "space_1"))

    assert status["status"] == ConversationStatus.FAILED.value
    assert status["activeExtractionRun"]["status"] == ExtractionRunStatus.FAILED.value
    assert status["activeExtractionRun"]["validationErrors"][0]["code"] == "WORKFLOW_EXCEPTION"
    assert repository.active_failed[0][0] == "conv_1"
    assert repository.conversation_failed[0][0] == "conv_1"


class WorkflowRepository:
    def __init__(self):
        self.conversation = ConversationDocument(
            _id="507f1f77bcf86cd799439011",
            userId="507f1f77bcf86cd799439012",
            spaceId="507f1f77bcf86cd799439013",
            status=ConversationStatus.READY_FOR_PROCESSING,
        )
        self.run = None
        self.transitions = []

    async def get_conversation(self, conversation_id):
        return self.conversation

    async def transition(self, conversation_id, target, updates=None):
        self.transitions.append(target)
        self.conversation.status = target
        if updates and updates.get("activeExtractionRunId") is not None:
            self.conversation.activeExtractionRunId = updates["activeExtractionRunId"]
        return self.conversation

    async def create_extraction_run(self, conversation, provider, model):
        self.run = ExtractionRunDocument(
            conversationId=conversation.id,
            userId=conversation.userId,
            spaceId=conversation.spaceId,
            processingVersion=conversation.processingVersion,
            provider=provider,
            model=model,
        )
        return self.run

    async def save_extraction_run(self, run):
        self.run = run

    async def list_transcript_chunks(self, conversation_id):
        return [
            TranscriptChunkDocument(
                conversationId=conversation_id,
                userId=self.conversation.userId,
                spaceId=self.conversation.spaceId,
                chunkId="chunk_1",
                sequenceNumber=0,
                rawText="Rahul will test the payment callback tomorrow.",
            )
        ]

    async def list_active_tasks(self, user_id, space_id):
        return []

    async def list_recent_summaries(self, user_id, space_id, limit=5):
        return []

    async def get_space_memory(self, user_id, space_id):
        return SpaceMemoryDocument(userId=user_id, spaceId=space_id)

    async def publish_outputs(self, run, summary, memory):
        return {"taskIds": []}

    async def schedule_transcript_expiry(self, conversation_id):
        return None

    async def mark_extraction_run_failed(self, run_id, error):
        raise AssertionError("validation gaps must not mark extraction run failed")


class WorkflowProvider:
    name = "test"


class WorkflowRouter:
    def route(self, capability):
        return WorkflowProvider(), "test-model"


def test_workflow_publishes_partial_when_coverage_has_critical_gap(monkeypatch):
    repository = WorkflowRepository()

    async def extract_segment(router, segment, context, user_id, space_id):
        return SectionExtractionResult(
            segmentId=segment.segmentId,
            tasks=[
                ExtractedTask(
                    title="Test payment callback",
                    operation="CREATE",
                    confidence=0.9,
                    sourceConversationId=segment.conversationId,
                    evidence=[EvidenceSpan(sequenceStart=0, sequenceEnd=0, text="Rahul will test the payment callback tomorrow.")],
                )
            ],
        )

    async def validate_coverage(router, transcript, outputs, context):
        return CoverageReport(score=0.5, criticalMissingCount=1)

    async def review_extraction_quality(router, transcript, outputs, context):
        return agents.ExtractionQualityReviewResponse(
            decisions=[
                agents.ExtractionQualityDecision(kind="task", index=0, keep=True, reason="Relevant task.")
            ]
        )

    async def summarize_conversation(router, conversation_id, user_id, space_id, transcript, outputs, processing_version):
        return ConversationSummaryDocument(
            conversationId=conversation_id,
            userId=user_id,
            spaceId=space_id,
            summary="Payment callback testing was discussed.",
            processingVersion=processing_version,
            modelProvider="test",
            modelName="test-model",
            promptVersion="test",
        )

    async def update_space_memory(router, previous, summary):
        return previous

    monkeypatch.setattr("services.conversation.agents.extract_segment", extract_segment)
    monkeypatch.setattr("services.conversation.agents.review_extraction_quality", review_extraction_quality)
    monkeypatch.setattr("services.conversation.agents.validate_coverage", validate_coverage)
    monkeypatch.setattr("services.conversation.agents.summarize_conversation", summarize_conversation)
    monkeypatch.setattr("services.conversation.agents.update_space_memory", update_space_memory)

    asyncio.run(ConversationProcessingWorkflow(repository, WorkflowRouter()).run(str(repository.conversation.id)))

    assert repository.conversation.status == ConversationStatus.PARTIAL
    assert ConversationStatus.FAILED not in repository.transitions
    assert repository.run.status == ExtractionRunStatus.PUBLISHED
    assert repository.run.validationErrors[0]["code"] == "CRITICAL_COVERAGE_GAP"


def test_orchestration_applies_llm_quality_review(monkeypatch):
    workflow = ConversationProcessingWorkflow(object(), WorkflowRouter())
    state = ConversationGraphState(
        conversation_id="conv_1",
        user_id="user_1",
        space_id="space_1",
        processing_version=1,
        extraction_run_id="run_1",
        conversation_status=ConversationStatus.PROCESSING,
        raw_transcript=(
            "[0] Rahul will test the payment callback tomorrow. "
            "[1] By the way I need to renew my car insurance tomorrow. "
            "[2] Payment callback retries need observability."
        ),
        normalized_transcript=(
            "[0] Rahul will test the payment callback tomorrow. "
            "[1] By the way I need to renew my car insurance tomorrow. "
            "[2] Payment callback retries need observability."
        ),
        space_memory={"currentSummary": "Payment callback project and checkout retries are the active work."},
        active_tasks=[],
        relevant_previous_summaries=[],
        segments=[],
        section_results=[
            SectionExtractionResult(
                segmentId="segment_1",
                tasks=[
                    ExtractedTask(
                        title="Test payment callback",
                        body="",
                        operation="CREATE",
                        confidence=0.9,
                        sourceConversationId="conv_1",
                        evidence=[EvidenceSpan(sequenceStart=0, sequenceEnd=0, text="Rahul will test the payment callback tomorrow.")],
                    ),
                    ExtractedTask(
                        title="Renew car insurance",
                        body="Renew the car insurance.",
                        operation="CREATE",
                        confidence=0.9,
                        sourceConversationId="conv_1",
                        evidence=[EvidenceSpan(sequenceStart=1, sequenceEnd=1, text="By the way I need to renew my car insurance tomorrow.")],
                    ),
                ],
                notes=[
                    ExtractedNote(
                        title="Payment retry observability",
                        body="Payment callback retries need observability so the project team can inspect retry behavior.",
                        confidence=0.9,
                        sourceConversationId="conv_1",
                        evidence=[EvidenceSpan(sequenceStart=2, sequenceEnd=2, text="Payment callback retries need observability.")],
                    ),
                    ExtractedNote(
                        title="Car insurance",
                        body="The speaker mentioned a car insurance renewal as an unrelated side topic.",
                        confidence=0.9,
                        sourceConversationId="conv_1",
                        evidence=[EvidenceSpan(sequenceStart=1, sequenceEnd=1, text="By the way I need to renew my car insurance tomorrow.")],
                    ),
                ],
            )
        ],
    )

    async def review_extraction_quality(router, transcript, outputs, context):
        return agents.ExtractionQualityReviewResponse(
            decisions=[
                agents.ExtractionQualityDecision(
                    kind="task",
                    index=0,
                    keep=True,
                    reason="Relevant to the active payment callback project.",
                    revisedBody="Rahul needs to test the payment callback as part of the active payment project. The conversation states this should happen tomorrow.",
                ),
                agents.ExtractionQualityDecision(
                    kind="task",
                    index=1,
                    keep=False,
                    reason="Unrelated tangent, not part of the active project context.",
                ),
                agents.ExtractionQualityDecision(
                    kind="note",
                    index=0,
                    keep=True,
                    reason="Relevant project context.",
                ),
                agents.ExtractionQualityDecision(
                    kind="note",
                    index=1,
                    keep=False,
                    reason="Unrelated tangent.",
                ),
            ]
        )

    monkeypatch.setattr(agents, "review_extraction_quality", review_extraction_quality)

    workflow._merge(state)
    asyncio.run(workflow._review_extracted_outputs(state, {"spaceMemory": state.space_memory}))

    assert [task.title for task in state.merged_tasks] == ["Test payment callback"]
    assert "active payment project" in state.merged_tasks[0].body
    assert [note.title for note in state.merged_notes] == ["Payment retry observability"]
    assert any("DROPPED_TASK_BY_LLM_REVIEW" in warning for warning in state.warnings)
    assert any("DROPPED_NOTE_BY_LLM_REVIEW" in warning for warning in state.warnings)


def test_extract_segment_tolerates_single_structured_section_failure(monkeypatch):
    async def fake_structured(router, prompt_name, schema, background, current, capability):
        if prompt_name == "risk-question-extractor-v1":
            raise ValueError("Structured response validation failed")
        return schema()

    monkeypatch.setattr(agents, "_structured", fake_structured)
    segment = Segment(
        segmentId="segment_1",
        conversationId="conv_1",
        sequenceStart=0,
        sequenceEnd=0,
        text="[0] Please test the payment callback tomorrow.",
        tokenCount=8,
    )

    result = asyncio.run(agents.extract_segment(WorkflowRouter(), segment, {}, "user_1", "space_1"))

    assert result.issues == []
    assert result.warnings
    assert "risk-question-extractor-v1 failed" in result.warnings[0]
