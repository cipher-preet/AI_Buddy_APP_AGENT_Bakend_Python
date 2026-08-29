import asyncio
from types import SimpleNamespace

import pytest

from services.conversation import agents
from services.conversation.agents import (
    FinalSynthesisError,
    PersistenceFailedError,
    empty_final_synthesis_diagnostics,
)
from services.conversation.models import (
    ConversationDocument,
    ConversationStatus,
    ConversationSummaryDocument,
    ExtractionRunDocument,
    ExtractionRunStatus,
    SpaceMemoryDocument,
    STTStatus,
    TranscriptChunkDocument,
    WindowProcessingStatus,
)
from services.conversation.workflow import ConversationProcessingWorkflow
from services.llm.router import LLMCapability
from apps.api_gateway.config.setting import settings


@pytest.fixture(autouse=True)
def _keep_legacy_short_session_path(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_EVENT_PIPELINE", False)
    monkeypatch.setattr(settings, "ENABLE_MEETING_PIPELINE", False)


TRANSCRIPT_LINES = {
    1: "Mira will write the ulari drain notes before Thursday.",
    2: "Rahul will open the sequence-wait ticket today.",
    3: "Please assign the banner review to design.",
    4: "I will confirm the retry budget with ops.",
    5: "Raw transcript remains the source of truth.",
    6: "Sequence 41 was corrected to sequence 14.",
    7: "The login banner still needs a separate review.",
    8: "Pending STT chunks can currently be skipped.",
    9: "STOP currently finalizes while a sequence is in flight.",
    10: "Tide records belong beside each crossing report.",
    11: "The ulari bridge expands when the tide rises.",
}
TRANSCRIPT = "\n".join(f"[{seq}] {text}" for seq, text in TRANSCRIPT_LINES.items())


def _span(sequence: int) -> dict:
    return {
        "sequenceStart": sequence,
        "sequenceEnd": sequence,
        "text": TRANSCRIPT_LINES[sequence],
    }


def _unit(key: str, kind: str, sequence: int, **kwargs) -> dict:
    payload = {
        "semanticKey": key,
        "kind": kind,
        "meaning": TRANSCRIPT_LINES.get(sequence, kwargs.get("meaning", "Invented meaning")),
        "evidence": kwargs.get("evidence") or [_span(sequence)],
        "quality": {"grounded": True, "independentlyUseful": True},
    }
    payload.update({name: value for name, value in kwargs.items() if name != "evidence"})
    return payload


def _ten_extracted_units() -> list[dict]:
    units = [
        _unit("drain-notes", "action_candidate", 1),
        _unit("sequence-ticket", "commitment", 2),
        _unit("banner-review", "request", 3),
        _unit("retry-budget", "commitment", 4),
        _unit("transcript-authority", "fact", 5),
        _unit("sequence-correction", "fact", 6),
        _unit("banner-note", "requirement", 7),
        _unit("stt-skip", "fact", 8),
        _unit("tide-record", "requirement", 10),
        _unit(
            "invented",
            "fact",
            99,
            meaning="Invented meaning that is not in the transcript.",
            evidence=[{"sequenceStart": 99, "sequenceEnd": 99, "text": "this line is not in the transcript"}],
        ),
    ]
    return units


def _seventeen_classifier_units() -> list[dict]:
    units = []
    for index in range(17):
        sequence = (index % 11) + 1
        roles = ["action", "commitment"] if index < 8 else ["fact", "explanation"]
        units.append(
            {
                "roles": roles,
                "topic": f"topic-{index}",
                "threadKey": f"thread-{index}",
                "normalizedMeaning": TRANSCRIPT_LINES[sequence],
                "evidenceIds": [sequence],
                "confidence": 0.91,
                "uncertain": False,
            }
        )
    return units


def _synthesized_task() -> dict:
    text = TRANSCRIPT_LINES[1]
    return {
        "title": "Write ulari drain notes",
        "body": f"{text} Complete this supported action using the cited sequence evidence before Thursday.",
        "operation": "CREATE",
        "confidence": 0.9,
        "origin": "explicit",
        "ownerText": "Mira",
        "dueDateText": "Thursday",
        "semanticArtifactKey": "drain-notes",
        "quality": {"grounded": True, "independentlyUseful": True},
        "evidence": [_span(1)],
    }


def _synthesized_note() -> dict:
    text = TRANSCRIPT_LINES[5]
    return {
        "title": "Raw transcript is authoritative",
        "body": f"{text} Keep this as durable meeting context because the cited evidence supports it independently of any task.",
        "confidence": 0.88,
        "semanticArtifactKey": "transcript-authority",
        "quality": {"grounded": True, "independentlyUseful": True},
        "evidence": [_span(5)],
    }


def _window(text: str = TRANSCRIPT, mode_window_id: str = "window_final"):
    return SimpleNamespace(
        conversationId="conv_final",
        userId="user_1",
        spaceId="space_1",
        id=mode_window_id,
        windowIndex=0,
        sequenceStart=1,
        sequenceEnd=11,
        text=text,
        isFinalPartial=False,
        extractionSkipped=False,
        status=WindowProcessingStatus.COMPLETED,
        result=None,
    )


class PipelineRouter:
    def __init__(self, synthesis_mode: str = "publish"):
        self.synthesis_mode = synthesis_mode
        self.calls: list[str] = []

    def route(self, capability: LLMCapability):
        return _PipelineProvider(self, "scripted-final-provider"), "scripted-final-model"


class _PipelineProvider:
    def __init__(self, router: PipelineRouter, name: str):
        self.router = router
        self.name = name

    async def generate_structured(self, request, schema):
        name = getattr(schema, "__name__", "")
        self.router.calls.append(name)
        if name == "SemanticRoleClassificationResponse":
            return schema(units=_seventeen_classifier_units())
        if name == "ConversationUnderstandingResponse":
            return schema(
                commitments=["Mira will write the notes.", "Rahul will open the ticket."],
                requests=["Assign the banner review."],
                importantFacts=["Raw transcript remains the source of truth."] * 7,
                followUps=["Confirm the retry budget."],
                nextSteps=["Keep tide records."],
                unresolvedQuestions=["Banner still needs review."],
                decisions=["Keep raw transcript as source of truth."],
                problems=["STOP finalizes while a sequence is in flight."],
                solutions=["Wait for expected sequences."],
            )
        if name == "WindowExtractionLLMResponse":
            return schema(
                summary="Grounded semantic units were extracted.",
                semanticUnits=_ten_extracted_units(),
                supportedUnitVerdict="has_supported_units",
            )
        if name == "FinalSynthesisLLMResponse":
            if self.router.synthesis_mode == "fail":
                raise TimeoutError("final synthesis provider failed")
            if self.router.synthesis_mode == "malformed":
                raise ValueError("schema validation failed for FinalSynthesisLLMResponse")
            if self.router.synthesis_mode == "none":
                return schema(
                    summary="Nothing to publish.",
                    publishVerdict="NO_PUBLISHABLE_ARTIFACTS",
                    tasks=[],
                    notes=[],
                )
            return schema(
                summary="Final synthesis published grounded artifacts.",
                tasks=[_synthesized_task()],
                notes=[_synthesized_note()],
                publishVerdict="PUBLISH",
            )
        if name == "ExtractionQualityReviewResponse":
            return schema(
                decisions=[
                    {"kind": "task", "index": 0, "keep": True, "reason": "grounded", "quality": {"grounded": True, "independentlyUseful": True}},
                    {"kind": "note", "index": 0, "keep": True, "reason": "grounded", "quality": {"grounded": True, "independentlyUseful": True}},
                ]
            )
        if name == "MemoryUpdateResponse":
            return schema(currentSummary="updated")
        return schema()


class FakeRepository:
    def __init__(self, chunks, fail_persist: bool = False):
        self.chunks = chunks
        self.fail_persist = fail_persist
        self.conversation = ConversationDocument(
            _id="conv_final",
            userId="user_1",
            spaceId="space_1",
            status=ConversationStatus.PROCESSING,
            processingVersion=1,
        )
        self.tasks: dict[str, dict] = {}
        self.notes: dict[str, dict] = {}
        self.runs: list[ExtractionRunDocument] = []
        self.summaries: dict[str, ConversationSummaryDocument] = {}
        self.publish_calls = 0

    async def get_space_memory(self, user_id, space_id):
        return SpaceMemoryDocument(userId=user_id, spaceId=space_id)

    async def list_active_tasks(self, user_id, space_id):
        return []

    async def list_recent_notes(self, user_id, space_id, limit=25):
        return []

    async def list_recent_summaries(self, user_id, space_id, limit=5):
        return []

    async def list_transcript_chunks(self, conversation_id):
        return self.chunks

    async def save_extraction_run(self, run):
        self.runs.append(run)

    async def transition(self, conversation_id, target, updates=None):
        self.conversation.status = target
        return self.conversation

    async def publish_outputs(self, run, summary, memory):
        self.publish_calls += 1
        if self.fail_persist:
            raise RuntimeError("db upsert failed")
        task_ids = []
        for task in run.stagedTasks:
            if task.operation == "NO_ACTION":
                continue
            key = task.fingerprint or task.title
            if key not in self.tasks:
                self.tasks[key] = {"_id": f"task-{len(self.tasks)+1}", **task.model_dump()}
            task_ids.append(self.tasks[key]["_id"])
        note_ids = []
        for note in run.stagedNotes:
            key = note.fingerprint or note.title
            if key not in self.notes:
                self.notes[key] = {"_id": f"note-{len(self.notes)+1}", **note.model_dump()}
            note_ids.append(self.notes[key]["_id"])
        summary.taskIds = task_ids
        self.summaries[str(summary.conversationId)] = summary
        return {"taskIds": task_ids, "noteIds": note_ids}

    async def mark_transcripts_published(self, conversation_id):
        return None

    async def schedule_transcript_expiry(self, conversation_id):
        return None


def _chunks():
    return [
        TranscriptChunkDocument(
            conversationId="conv_final",
            userId="user_1",
            spaceId="space_1",
            chunkId=f"chunk_{sequence}",
            sequenceNumber=sequence,
            rawText=text,
            sttStatus=STTStatus.COMPLETED,
        )
        for sequence, text in TRANSCRIPT_LINES.items()
    ]


def _run_extraction(router, mode: str = "final"):
    agents._SEMANTIC_CLASSIFICATION_CACHE.clear()
    return asyncio.run(agents.extract_window(router, _window(), context={}, meeting_context={}, mode=mode))


def test_production_units_reach_final_synthesis_quality_and_persistence():
    router = PipelineRouter("publish")
    repo = FakeRepository(_chunks())
    workflow = ConversationProcessingWorkflow(repo, router)
    run = ExtractionRunDocument(
        conversationId=repo.conversation.id,
        userId=repo.conversation.userId,
        spaceId=repo.conversation.spaceId,
        processingVersion=1,
        provider="scripted-final-provider",
        model="scripted-final-model",
    )
    asyncio.run(workflow._run_short_session_finalization(repo.conversation, run, []))
    diagnostics = run.checkpoints["short_raw_transcript"]
    assert diagnostics["validatedSemanticUnitCount"] == 9
    assert diagnostics["finalSynthesisInvoked"] is True
    assert diagnostics["finalSynthesisInputUnitCount"] == 9
    assert diagnostics["finalSynthesisProvider"] == "scripted-final-provider"
    assert diagnostics["finalSynthesisModel"] == "scripted-final-model"
    assert diagnostics["finalSynthesisRawTaskCount"] == 1
    assert diagnostics["finalSynthesisRawNoteCount"] == 1
    assert diagnostics["finalSynthesisParsedTaskCount"] == 1
    assert diagnostics["finalSynthesisParsedNoteCount"] == 1
    assert diagnostics["finalSynthesisVerdict"] == "PUBLISH"
    assert diagnostics["taskCountAfterConfidence"] == 1
    assert diagnostics["noteCountAfterConfidence"] == 1
    assert diagnostics["qualityAcceptedTaskCount"] == 1
    assert diagnostics["qualityAcceptedNoteCount"] == 1
    assert diagnostics["persistenceAttempted"] is True
    assert diagnostics["tasksPersistedCount"] == 1
    assert diagnostics["notesPersistedCount"] == 1
    assert diagnostics["persistedTaskIds"]
    assert diagnostics["persistedNoteIds"]
    assert len(repo.tasks) == 1
    assert len(repo.notes) == 1
    summary = next(iter(repo.summaries.values()))
    assert summary.taskIds
    assert repo.conversation.status == ConversationStatus.COMPLETED
    assert run.status == ExtractionRunStatus.PUBLISHED
    assert "FinalSynthesisLLMResponse" in router.calls
    assert router.calls.count("WindowExtractionLLMResponse") >= 1
    assert router.calls.count("FinalSynthesisLLMResponse") >= 1


def test_final_model_may_return_no_publishable_artifacts():
    router = PipelineRouter("none")
    result, _, _ = _run_extraction(router)
    assert result.extractionOutcome.value == "SUCCESS"
    assert result.semanticUnits
    assert result.extractionDiagnostics["finalSynthesisInvoked"] is True
    assert result.extractionDiagnostics["finalSynthesisVerdict"] == "TASK_COVERAGE_CONFLICT"
    assert result.extractionDiagnostics["taskCoverageConflict"] is True
    assert not result.tasks
    assert not result.notes


def test_final_synthesis_provider_failure_does_not_publish_empty_success():
    router = PipelineRouter("fail")
    with pytest.raises(FinalSynthesisError) as caught:
        _run_extraction(router)
    assert caught.value.verdict == "PROVIDER_FAILED"


def test_final_synthesis_malformed_schema_does_not_publish_empty_success():
    router = PipelineRouter("malformed")
    with pytest.raises(FinalSynthesisError) as caught:
        _run_extraction(router)
    assert caught.value.verdict == "MALFORMED_SCHEMA"


def test_final_synthesis_success_with_db_persistence_failure():
    router = PipelineRouter("publish")
    repo = FakeRepository(_chunks(), fail_persist=True)
    workflow = ConversationProcessingWorkflow(repo, router)
    run = ExtractionRunDocument(
        conversationId=repo.conversation.id,
        userId=repo.conversation.userId,
        spaceId=repo.conversation.spaceId,
        processingVersion=1,
        provider="scripted-final-provider",
        model="scripted-final-model",
    )
    with pytest.raises(PersistenceFailedError):
        asyncio.run(workflow._run_short_session_finalization(repo.conversation, run, []))
    assert run.status != ExtractionRunStatus.PUBLISHED
    assert repo.conversation.status != ConversationStatus.COMPLETED
    assert any(error.get("code") == "PERSISTENCE_FAILED" for error in run.validationErrors)


def test_final_synthesis_success_is_idempotent_on_duplicate_finalization():
    router = PipelineRouter("publish")
    repo = FakeRepository(_chunks())
    workflow = ConversationProcessingWorkflow(repo, router)
    run = ExtractionRunDocument(
        conversationId=repo.conversation.id,
        userId=repo.conversation.userId,
        spaceId=repo.conversation.spaceId,
        processingVersion=1,
        provider="scripted-final-provider",
        model="scripted-final-model",
    )
    asyncio.run(workflow._run_short_session_finalization(repo.conversation, run, []))
    first_task_ids = list(run.checkpoints["short_raw_transcript"]["persistedTaskIds"])
    first_note_ids = list(run.checkpoints["short_raw_transcript"]["persistedNoteIds"])
    repo.conversation.status = ConversationStatus.PROCESSING
    asyncio.run(workflow._run_short_session_finalization(repo.conversation, run, []))
    assert len(repo.tasks) == 1
    assert len(repo.notes) == 1
    assert run.checkpoints["short_raw_transcript"]["persistedTaskIds"] == first_task_ids
    assert run.checkpoints["short_raw_transcript"]["persistedNoteIds"] == first_note_ids


def test_intermediate_checkpoint_does_not_publish_tasks_or_notes():
    router = PipelineRouter("publish")
    result, _, _ = _run_extraction(router, mode="checkpoint")
    assert result.isCheckpoint is True
    assert result.semanticUnits
    assert result.extractionDiagnostics["finalSynthesisInvoked"] is False
    assert result.extractionDiagnostics["finalSynthesisVerdict"] == "SKIPPED_CHECKPOINT"
    assert not result.tasks
    assert not result.notes
    assert "FinalSynthesisLLMResponse" not in router.calls


def test_short_raw_session_does_perform_final_synthesis():
    router = PipelineRouter("publish")
    result, _, _ = asyncio.run(
        agents.extract_from_raw_transcript(router, "conv_final", "user_1", "space_1", TRANSCRIPT, {})
    )
    assert result.extractionDiagnostics["finalSynthesisInvoked"] is True
    assert result.extractionDiagnostics["finalSynthesisInputUnitCount"] == 9
    assert result.tasks
    assert result.notes
    assert result.extractionDiagnostics["taskCountAfterConfidence"] == len(result.tasks)
    assert result.extractionDiagnostics["noteCountAfterConfidence"] == len(result.notes)
    assert "FinalSynthesisLLMResponse" in router.calls


def test_empty_diagnostics_shape_includes_required_counters():
    diagnostics = empty_final_synthesis_diagnostics()
    for key in (
        "validatedSemanticUnitCount",
        "finalSynthesisInvoked",
        "finalSynthesisInputUnitCount",
        "finalSynthesisProvider",
        "finalSynthesisModel",
        "finalSynthesisRawTaskCount",
        "finalSynthesisRawNoteCount",
        "finalSynthesisParsedTaskCount",
        "finalSynthesisParsedNoteCount",
        "qualityAcceptedTaskCount",
        "qualityAcceptedNoteCount",
        "qualityArtifactDiagnostics",
        "qualityRepairAttempted",
        "persistenceAttempted",
        "tasksPersistedCount",
        "notesPersistedCount",
        "persistedTaskIds",
        "persistedNoteIds",
    ):
        assert key in diagnostics
