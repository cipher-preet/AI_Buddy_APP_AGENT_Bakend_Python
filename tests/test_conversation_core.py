import asyncio
from datetime import datetime, timedelta, timezone

from pydantic import BaseModel

from services.conversation.fingerprints import task_fingerprint
from services.conversation import agents
from services.conversation.finalization import ConversationFinalizationCoordinator
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
    STTStatus,
    TranscriptChunkDocument,
    assert_valid_transition,
    utc_now,
)
from services.conversation.repository import ConversationRepository
from services.conversation.service import ConversationService
from services.conversation.transcript import assemble_transcript, detect_missing_sequences, segment_transcript
from services.conversation.workflow import ConversationProcessingWorkflow
from services.conversation.workflow_state import ConversationGraphState
from services.chat.planner import ChatQueryPlan
from services.chat.service import ChatService, _merge_pending_space_action
from services.chat import tools as chat_tools
from services.chat.tools import ChatToolRunner
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


def test_finalization_requeues_pending_s3_transcripts():
    class Repository:
        def __init__(self):
            self.transitions = []

        async def get_conversation(self, conversation_id):
            return ConversationDocument(
                _id=conversation_id,
                userId="user-1",
                spaceId="space-1",
                status=ConversationStatus.STOP_REQUESTED,
                expectedLastSequence=1,
                receivedAudioChunkCount=2,
            )

        async def list_transcript_chunks(self, conversation_id):
            return [
                TranscriptChunkDocument(
                    conversationId=conversation_id,
                    userId="user-1",
                    spaceId="space-1",
                    chunkId="chunk-0",
                    sequenceNumber=0,
                    audioFilePath="s3://bucket/buddy/audio/chunk-0.webm",
                    sttStatus=STTStatus.PENDING,
                ),
                TranscriptChunkDocument(
                    conversationId=conversation_id,
                    userId="user-1",
                    spaceId="space-1",
                    chunkId="chunk-1",
                    sequenceNumber=1,
                    audioFilePath="s3://bucket/buddy/audio/chunk-1.webm",
                    sttStatus=STTStatus.PENDING,
                ),
            ]

        async def get_audio_chunk(self, conversation_id, sequence_number):
            return {
                "filename": f"chunk-{sequence_number}.webm",
                "contentType": "audio/webm",
                "storageProvider": "s3",
                "s3Bucket": "bucket",
                "s3ObjectKey": f"buddy/audio/chunk-{sequence_number}.webm",
            }

        async def transition(self, conversation_id, target, updates=None):
            self.transitions.append((conversation_id, target, updates or {}))

    class Producer:
        def __init__(self):
            self.events = []

        async def publish(self, stream, event):
            self.events.append((stream, event))

    async def run():
        producer = Producer()
        repository = Repository()
        await ConversationFinalizationCoordinator(repository, producer).finalize("conv-1")
        return repository, producer

    repository, producer = asyncio.run(run())

    assert len(producer.events) == 2
    assert all(event.eventType == "stt.requested" for _, event in producer.events)
    assert producer.events[0][1].payload["storageProvider"] == "s3"
    assert producer.events[0][1].payload["contentType"] == "audio/webm"
    assert repository.transitions[0][1] == ConversationStatus.WAITING_FOR_TRANSCRIPTS


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


class PublishCollection:
    def __init__(self):
        self.docs = []

    async def find_one_and_update(self, query, update, upsert=False, return_document=None):
        for doc in self.docs:
            if all(doc.get(key) == value for key, value in query.items()):
                if "$set" in update:
                    doc.update(update["$set"])
                return doc
        if not upsert:
            return None
        doc = dict(update.get("$setOnInsert") or {})
        self.docs.append(doc)
        return doc

    async def update_one(self, query, update, upsert=False):
        self.docs.append({"query": query, "update": update, "upsert": upsert})


class PublishDb(dict):
    def __init__(self):
        super().__init__()
        self.tasks = PublishCollection()
        self.notes = PublishCollection()
        self.conversation_summaries = PublishCollection()
        self.space_memory = PublishCollection()


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


def test_publish_outputs_writes_final_tasks_and_notes():
    asyncio.run(_publish_outputs_writes_final_tasks_and_notes())


async def _publish_outputs_writes_final_tasks_and_notes():
    db = PublishDb()
    run = ExtractionRunDocument(
        conversationId="507f1f77bcf86cd799439011",
        userId="507f1f77bcf86cd799439012",
        spaceId="507f1f77bcf86cd799439013",
        processingVersion=1,
        provider="test",
        model="test-model",
        stagedTasks=[
            ExtractedTask(
                title="Test payment callback",
                body="Test the callback retry flow.",
                operation="CREATE",
                dueDateResolved="2026-08-05",
                confidence=0.9,
                sourceConversationId="507f1f77bcf86cd799439011",
                fingerprint="task-fingerprint",
                evidence=[EvidenceSpan(sequenceStart=0, sequenceEnd=0, text="Test callback today.")],
            )
        ],
        stagedNotes=[
            ExtractedNote(
                title="Callback retry note",
                body="The callback retry flow needs observability.",
                confidence=0.9,
                sourceConversationId="507f1f77bcf86cd799439011",
                fingerprint="note-fingerprint",
                evidence=[EvidenceSpan(sequenceStart=1, sequenceEnd=1, text="Retry observability.")],
            )
        ],
    )
    summary = ConversationSummaryDocument(
        conversationId=run.conversationId,
        userId=run.userId,
        spaceId=run.spaceId,
        summary="Payment callback testing was discussed.",
        processingVersion=1,
        modelProvider="test",
        modelName="test-model",
        promptVersion="test",
    )
    memory = SpaceMemoryDocument(userId=run.userId, spaceId=run.spaceId)

    result = await ConversationRepository(db).publish_outputs(run, summary, memory)

    assert result["taskIds"]
    assert result["noteIds"]
    assert db.tasks.docs[0]["title"] == "Test payment callback"
    assert db.tasks.docs[0]["status"] == "pending"
    assert db.notes.docs[0]["title"] == "Callback retry note"


CONTROLLED_CHAT_NOW = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)


class FrozenChatDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        if tz is not None:
            return CONTROLLED_CHAT_NOW.astimezone(tz)
        return CONTROLLED_CHAT_NOW.replace(tzinfo=None)


def controlled_chat_date(offset_days: int = 0) -> str:
    return (CONTROLLED_CHAT_NOW + timedelta(days=offset_days)).date().isoformat()


def freeze_chat_tool_clock(monkeypatch) -> None:
    monkeypatch.setattr(chat_tools, "datetime", FrozenChatDateTime)


class ToolRepository:
    async def get_space_memory(self, user_id, space_id):
        return SpaceMemoryDocument(
            userId=user_id,
            spaceId=space_id,
            currentSummary="Payment callback project is active.",
            importantFacts=["Retries need observability."],
            importantDecisions=["Use structured chat tools."],
        )

    async def list_recent_summaries(self, user_id, space_id, limit=8):
        return [{"summary": "Discussed callback testing.", "topics": ["payments"], "createdAt": utc_now()}]

    async def list_tasks(self, user_id, space_id, limit=100):
        return [
            {
                "title": "Test payment callback",
                "body": "Run the retry test.",
                "status": "pending",
                "dueDateResolved": controlled_chat_date(),
                "ownerText": "Rahul",
            }
        ]

    async def list_recent_notes(self, user_id, space_id, limit=50):
        return [{"title": "Retry observability", "body": "Retries need logs.", "updatedAt": utc_now()}]

    async def list_user_spaces(self, user_id, limit=50):
        return [{"spaceId": "space_1", "label": "Payments", "sources": ["tasks"]}]


def test_chat_tools_include_structured_task_note_and_summary_context(monkeypatch):
    freeze_chat_tool_clock(monkeypatch)

    result = asyncio.run(
        ChatToolRunner(ToolRepository()).run(
            "summary my space and tell today unfinished tasks and notes",
            "user_1",
            "space_1",
            ChatQueryPlan(
                understoodRequest="Summarize space with today's tasks and notes.",
                useStructuredTools=True,
                useVectorSearch=False,
                directToolAnswerAllowed=True,
                toolFocus=["summaries", "space_memory", "tasks", "notes"],
                temporalScope="today",
            ),
        )
    )
    context = str(result["context"])

    assert "Tool: space_memory" in context
    assert "Tool: tasks" in context
    assert "due_today=1" in context
    assert "Test payment callback" in context
    assert "Tool: notes" in context
    assert "Retry observability" in context
    assert "You have 1 task(s) due today" in str(result["answer"])
    assert result["direct"] is True


def test_chat_tools_directly_answer_today_tasks(monkeypatch):
    freeze_chat_tool_clock(monkeypatch)

    result = asyncio.run(
        ChatToolRunner(ToolRepository()).run(
            "What are my today's tasks",
            "user_1",
            "space_1",
            ChatQueryPlan(
                understoodRequest="List today's tasks.",
                useStructuredTools=True,
                useVectorSearch=False,
                directToolAnswerAllowed=True,
                toolFocus=["tasks"],
                temporalScope="today",
            ),
        )
    )

    assert "Structured tool context" not in str(result["answer"])
    assert "You have 1 task(s) due today" in str(result["answer"])
    assert "Test payment callback" in str(result["answer"])
    assert result["direct"] is True


class MixedDueDateToolRepository(ToolRepository):
    async def list_tasks(self, user_id, space_id, limit=100):
        return [
            {
                "title": "Due today payment callback",
                "body": "Run the retry test today.",
                "status": "pending",
                "dueDateResolved": controlled_chat_date(),
                "ownerText": "Rahul",
            },
            {
                "title": "Overdue payment callback",
                "body": "This is unfinished from yesterday.",
                "status": "pending",
                "dueDateResolved": controlled_chat_date(-1),
                "ownerText": "Rahul",
            },
            {
                "title": "Future payment callback",
                "body": "This is not due yet.",
                "status": "pending",
                "dueDateResolved": controlled_chat_date(1),
                "ownerText": "Rahul",
            },
        ]


def test_chat_tools_keep_overdue_unfinished_out_of_due_today(monkeypatch):
    freeze_chat_tool_clock(monkeypatch)

    result = asyncio.run(
        ChatToolRunner(MixedDueDateToolRepository()).run(
            "What are my today's tasks",
            "user_1",
            "space_1",
            ChatQueryPlan(
                understoodRequest="List today's tasks.",
                useStructuredTools=True,
                useVectorSearch=False,
                directToolAnswerAllowed=True,
                toolFocus=["tasks"],
                temporalScope="today",
            ),
        )
    )
    context = str(result["context"])
    answer = str(result["answer"])

    assert "due_today=1" in context
    assert "Due today payment callback" in answer
    assert "Overdue payment callback" in context
    assert f"due={controlled_chat_date(-1)}" in context
    assert "Overdue payment callback" not in answer
    assert "Future payment callback" in context
    assert "Future payment callback" not in answer


def test_chat_tools_do_not_direct_answer_topic_specific_task_requests():
    result = asyncio.run(
        ChatToolRunner(ToolRepository()).run(
            "Please give me all task details for payment callback",
            "user_1",
            "space_1",
            ChatQueryPlan(
                understoodRequest="Give task details for the payment callback topic.",
                useStructuredTools=True,
                useVectorSearch=True,
                directToolAnswerAllowed=False,
                toolFocus=["tasks"],
                temporalScope="all",
                searchQueries=["payment callback task details"],
            ),
        )
    )

    assert result["direct"] is False
    assert "Test payment callback" in str(result["context"])
    assert "I found 1 task(s)" in str(result["answer"])


def test_chat_tools_list_spaces_when_plan_requests_options():
    result = asyncio.run(
        ChatToolRunner(ToolRepository()).run(
            "Give me list of spaces",
            "user_1",
            None,
            ChatQueryPlan(
                understoodRequest="List available spaces.",
                responseMode="list_options",
                requiresSpace=False,
                useStructuredTools=True,
                useVectorSearch=False,
                directToolAnswerAllowed=True,
                optionKind="spaces",
            ),
        )
    )

    assert result["direct"] is True
    assert "Available spaces" in str(result["answer"])
    assert "Payments" in str(result["answer"])


def test_chat_tools_ignore_non_english_planner_question_for_space_options():
    result = asyncio.run(
        ChatToolRunner(ToolRepository()).run(
            "कल के लिए मेरे tasks plan करो",
            "user_1",
            None,
            ChatQueryPlan(
                understoodRequest="Plan tasks for tomorrow.",
                responseMode="ask_clarifying_question",
                requiresSpace=True,
                optionKind="spaces",
                missingInfoQuestion="कौन सा space उपयोग करना है?",
            ),
        )
    )

    assert "Which space should I use?" in str(result["answer"])
    assert "कौन" not in str(result["answer"])


def test_chat_tools_missing_info_answer_is_english_only():
    result = asyncio.run(
        ChatToolRunner(ToolRepository()).run(
            "कल के लिए मेरे tasks plan करो",
            "user_1",
            None,
            ChatQueryPlan(
                understoodRequest="Plan tasks for tomorrow.",
                responseMode="ask_clarifying_question",
                requiresSpace=False,
                optionKind="none",
                missingInfoQuestion="कृपया और जानकारी दें",
            ),
        )
    )

    assert result["answer"] == "I need one more detail before I can answer. Please clarify what you mean."


def test_chat_tools_allow_general_note_request_without_selected_space():
    result = asyncio.run(
        ChatToolRunner(ToolRepository()).run(
            "Give me some notes for the space company",
            "user_1",
            None,
            ChatQueryPlan(
                understoodRequest="Give general notes about a space company.",
                useStructuredTools=False,
                useVectorSearch=False,
                directToolAnswerAllowed=False,
            ),
        )
    )

    assert result["direct"] is False
    assert result["answer"] is None
    assert "Structured tools skipped by query plan." in str(result["context"])


def test_chat_tools_do_not_require_space_for_general_answer_plan():
    result = asyncio.run(
        ChatToolRunner(ToolRepository()).run(
            "Hi",
            "user_1",
            None,
            ChatQueryPlan(
                understoodRequest="Greet the user.",
                responseMode="answer",
                requiresSpace=False,
                useStructuredTools=True,
                useVectorSearch=False,
                directToolAnswerAllowed=False,
                optionKind="none",
                toolFocus=[],
            ),
        )
    )

    assert result["direct"] is False
    assert result["answer"] is None
    assert "does not require workspace tools" in str(result["context"])


def test_chat_preserves_pending_original_request_when_user_asks_to_list_spaces():
    existing = {
        "type": "select_option",
        "optionKind": "spaces",
        "originalQuestion": "Give me all notes",
        "plan": {"responseMode": "ask_clarifying_question", "toolFocus": ["notes"]},
        "options": [{"index": 1, "label": "Old", "value": "old_space"}],
    }
    current = {
        "type": "select_option",
        "optionKind": "spaces",
        "originalQuestion": "Give me list of all spaces",
        "plan": {"responseMode": "list_options", "optionKind": "spaces"},
        "options": [{"index": 1, "label": "Payments", "value": "space_1"}],
    }

    merged = _merge_pending_space_action(existing, current)

    assert merged["originalQuestion"] == "Give me all notes"
    assert merged["plan"]["toolFocus"] == ["notes"]
    assert merged["options"][0]["value"] == "space_1"


def test_chat_pending_action_resolves_numbered_option_without_reasking():
    pending = {
        "type": "select_option",
        "optionKind": "spaces",
        "originalQuestion": "Plan my tasks for tomorrow",
        "options": [
            {"index": 1, "label": "Payments", "value": "space_1"},
            {"index": 2, "label": "Sales", "value": "space_2"},
        ],
    }

    selected_space, resumed_question = ChatService()._resolve_pending_action("1", None, pending)

    assert selected_space == "space_1"
    assert resumed_question == "Plan my tasks for tomorrow"


def test_chat_pending_action_resolves_label_option_without_reasking():
    pending = {
        "type": "select_option",
        "optionKind": "spaces",
        "originalQuestion": "Summarize my day",
        "options": [
            {"index": 1, "label": "Payments", "value": "space_1"},
        ],
    }

    selected_space, resumed_question = ChatService()._resolve_pending_action("payments", None, pending)

    assert selected_space == "space_1"
    assert resumed_question == "Summarize my day"


def test_chat_resolves_inline_space_number_before_planning():
    async def run():
        service = ChatService(tool_runner=ChatToolRunner(ToolRepository()))
        return await service._resolve_space_from_message_or_history("Give notes for space 1", "user_1", [])

    selected_space, resumed_question = asyncio.run(run())

    assert selected_space == "space_1"
    assert resumed_question == "Give notes"


def test_chat_resolves_number_reply_from_history_when_pending_action_missing():
    class Human:
        type = "human"
        content = "Give notes for space 1"

    async def run():
        service = ChatService(tool_runner=ChatToolRunner(ToolRepository()))
        return await service._resolve_space_from_message_or_history("1", "user_1", [Human()])

    selected_space, resumed_question = asyncio.run(run())

    assert selected_space == "space_1"
    assert resumed_question == "Give notes"


class StagedToolRepository(ToolRepository):
    async def list_recent_notes(self, user_id, space_id, limit=50):
        return []

    async def list_staged_notes(self, user_id, space_id, limit=50):
        return [{"title": "Demo note", "body": "This comes from stagedNotes.", "updatedAt": utc_now()}]


def test_chat_tools_directly_answer_staged_notes_when_published_notes_are_empty():
    result = asyncio.run(
        ChatToolRunner(StagedToolRepository()).run(
            "Give notes",
            "user_1",
            "space_1",
            ChatQueryPlan(
                understoodRequest="List notes.",
                useStructuredTools=True,
                useVectorSearch=False,
                directToolAnswerAllowed=True,
                toolFocus=["notes"],
            ),
        )
    )

    assert result["direct"] is True
    assert "Demo note" in str(result["answer"])
    assert "staged note" in str(result["answer"]).lower()


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
