import asyncio
from types import SimpleNamespace

from services.conversation import agents
from services.conversation.extraction_contract import hydrate_synthesized_artifacts
from services.conversation.intelligence import (
    ConfidencePolicy,
    default_confidence_policy,
    score_and_filter_result,
    validation_decision_for_task,
)
from services.conversation.models import (
    ConversationStatus,
    EvidenceSpan,
    ExtractedNote,
    ExtractedTask,
    ExtractionRunDocument,
    ExtractionRunStatus,
    SemanticUnit,
    STTStatus,
    TranscriptChunkDocument,
    WindowExtractionResult,
)
from services.conversation.workflow import ConversationProcessingWorkflow
from services.llm.router import LLMCapability
from tests.test_final_synthesis_persistence import FakeRepository, PipelineRouter, _window

TRANSCRIPT_LINES = {
    1: "Mira will write the ulari drain notes before Thursday.",
    2: "Rahul will open the sequence-wait ticket today.",
    3: "Tide records belong beside each crossing report.",
}
TRANSCRIPT = "\n".join(f"[{seq}] {text}" for seq, text in TRANSCRIPT_LINES.items())


def _span(sequence: int, text: str | None = None) -> EvidenceSpan:
    return EvidenceSpan(sequenceStart=sequence, sequenceEnd=sequence, text=text or TRANSCRIPT_LINES[sequence])


def _unit(key: str, kind: str, sequence: int) -> SemanticUnit:
    return SemanticUnit(
        semanticKey=key,
        kind=kind,
        meaning=TRANSCRIPT_LINES[sequence],
        evidence=[_span(sequence)],
        evidenceIds=[sequence],
        quality={"grounded": True, "independentlyUseful": True},
    )


def _units() -> list[SemanticUnit]:
    return [
        _unit("drain-notes", "action_candidate", 1),
        _unit("sequence-ticket", "commitment", 2),
    ]


def _task(sequence: int, title: str, **kwargs) -> ExtractedTask:
    text = TRANSCRIPT_LINES[sequence]
    return ExtractedTask(
        title=title,
        body=kwargs.get("body", f"{text} Complete this supported action using the cited sequence evidence."),
        operation="CREATE",
        confidence=kwargs.get("confidence", 0.5),
        sourceConversationId="conv_final",
        evidence=kwargs.get("evidence", [_span(sequence, kwargs.get("evidence_text", text))]),
        origin=kwargs.get("origin", "unknown"),
        ownerText=kwargs.get("ownerText"),
        dueDateText=kwargs.get("dueDateText"),
        changes={
            "synthesisSource": "llm",
            "semanticArtifactKey": kwargs.get("semanticArtifactKey"),
            "sourceSemanticUnitIds": kwargs.get("sourceSemanticUnitIds", []),
            "quality": kwargs.get("quality", {}),
        },
    )


def _note(sequence: int, **kwargs) -> ExtractedNote:
    text = TRANSCRIPT_LINES[sequence]
    return ExtractedNote(
        title=kwargs.get("title", "Tide records are required"),
        body=kwargs.get("body", f"{text} Keep this as durable meeting context independent of either task."),
        confidence=kwargs.get("confidence", 0.5),
        sourceConversationId="conv_final",
        evidence=kwargs.get("evidence", [_span(sequence, kwargs.get("evidence_text", text))]),
        debug={
            "synthesisSource": "llm",
            "semanticArtifactKey": kwargs.get("semanticArtifactKey"),
            "sourceSemanticUnitIds": kwargs.get("sourceSemanticUnitIds", []),
            "quality": kwargs.get("quality", {}),
        },
    )


def _synthesized_payloads():
    return [
        {
            "title": "Write ulari drain notes",
            "body": "Mira will write the ulari drain notes before Thursday using the cited evidence.",
            "operation": "CREATE",
            "confidence": 0.9,
            "semanticArtifactKey": "drain-notes",
            "sourceSemanticUnitIds": ["drain-notes"],
            "evidence": [{"sequenceStart": 1, "sequenceEnd": 1, "text": "paraphrased drain-note evidence"}],
        },
        {
            "title": "Open sequence-wait ticket",
            "body": "Rahul will open the sequence-wait ticket today as committed in the meeting.",
            "operation": "CREATE",
            "confidence": 0.9,
            "semanticArtifactKey": "sequence-ticket",
            "sourceSemanticUnitIds": ["sequence-ticket"],
            "evidence": [{"sequenceStart": 2, "sequenceEnd": 2, "text": "paraphrased ticket evidence"}],
        },
    ], {
        "title": "Tide records stay with reports",
        "body": "Tide records belong beside each crossing report and remain durable meeting context.",
        "confidence": 0.88,
        "semanticArtifactKey": "tide-record",
        "sourceSemanticUnitIds": ["sequence-ticket"],
        "evidence": [{"sequenceStart": 2, "sequenceEnd": 2, "text": "paraphrased ticket evidence"}],
    }


class QualityGateRouter(PipelineRouter):
    def __init__(self, synthesis_mode: str = "production_missing_metadata"):
        super().__init__(synthesis_mode)
        self.repair_calls = 0

    def route(self, capability: LLMCapability):
        return _QualityGateProvider(self, "scripted-final-provider"), "scripted-final-model"


class _QualityGateProvider:
    def __init__(self, router: QualityGateRouter, name: str):
        self.router = router
        self.name = name

    async def generate_structured(self, request, schema):
        name = getattr(schema, "__name__", "")
        self.router.calls.append(name)
        tasks, note = _synthesized_payloads()
        if name == "SemanticRoleClassificationResponse":
            return schema(
                units=[
                    {
                        "roles": ["action", "commitment"],
                        "topic": "drain notes",
                        "threadKey": "drain",
                        "normalizedMeaning": TRANSCRIPT_LINES[1],
                        "evidenceIds": [1],
                        "confidence": 0.91,
                        "uncertain": False,
                    },
                    {
                        "roles": ["action", "commitment"],
                        "topic": "sequence ticket",
                        "threadKey": "ticket",
                        "normalizedMeaning": TRANSCRIPT_LINES[2],
                        "evidenceIds": [2],
                        "confidence": 0.9,
                        "uncertain": False,
                    },
                ]
            )
        if name == "ConversationUnderstandingResponse":
            return schema(
                commitments=["Mira will write the notes.", "Rahul will open the ticket."],
                requests=[],
                importantFacts=["Tide records belong beside each crossing report."] * 7,
                followUps=[],
                nextSteps=[],
                unresolvedQuestions=[],
                decisions=[],
                problems=[],
                solutions=[],
            )
        if name == "WindowExtractionLLMResponse":
            return schema(
                summary="Two grounded semantic units were extracted.",
                semanticUnits=[
                    {
                        "semanticKey": "drain-notes",
                        "kind": "action_candidate",
                        "meaning": TRANSCRIPT_LINES[1],
                        "evidence": [{"sequenceStart": 1, "sequenceEnd": 1, "text": TRANSCRIPT_LINES[1]}],
                        "quality": {"grounded": True, "independentlyUseful": True},
                    },
                    {
                        "semanticKey": "sequence-ticket",
                        "kind": "commitment",
                        "meaning": TRANSCRIPT_LINES[2],
                        "evidence": [{"sequenceStart": 2, "sequenceEnd": 2, "text": TRANSCRIPT_LINES[2]}],
                        "quality": {"grounded": True, "independentlyUseful": True},
                    },
                ],
                supportedUnitVerdict="has_supported_units",
            )
        if name == "FinalSynthesisLLMResponse":
            if self.router.synthesis_mode == "hallucinated":
                tasks[0]["title"] = "Launch the orbital beacon"
                tasks[0]["body"] = "Launch the orbital beacon before dawn even though nobody mentioned it."
                tasks[0]["sourceSemanticUnitIds"] = []
                tasks[0]["semanticArtifactKey"] = ""
                tasks[0]["evidence"] = [{"sequenceStart": 99, "sequenceEnd": 99, "text": "invented evidence"}]
            elif self.router.synthesis_mode == "partial":
                tasks[1]["title"] = "Launch the orbital beacon"
                tasks[1]["body"] = "Launch the orbital beacon before dawn even though nobody mentioned it."
                tasks[1]["sourceSemanticUnitIds"] = []
                tasks[1]["semanticArtifactKey"] = ""
                tasks[1]["evidence"] = [{"sequenceStart": 99, "sequenceEnd": 99, "text": "invented evidence"}]
            elif self.router.synthesis_mode == "missing_provenance":
                for item in (*tasks, note):
                    item.pop("sourceSemanticUnitIds", None)
                    item["evidence"] = []
            elif self.router.synthesis_mode == "all_quality_failed":
                for item in tasks:
                    item["evidence"] = [{"sequenceStart": 99, "sequenceEnd": 99, "text": "invented evidence"}]
                    item["sourceSemanticUnitIds"] = []
                    item["semanticArtifactKey"] = ""
                note["evidence"] = [{"sequenceStart": 99, "sequenceEnd": 99, "text": "invented evidence"}]
                note["sourceSemanticUnitIds"] = []
                note["semanticArtifactKey"] = ""
            return schema(
                summary="Final synthesis published grounded artifacts.",
                tasks=tasks,
                notes=[note],
                publishVerdict="PUBLISH",
            )
        if name == "MissingItemRepairLLMResponse":
            self.router.repair_calls += 1
            if self.router.synthesis_mode == "all_quality_failed":
                return schema(
                    tasks=[
                        {
                            "title": "Still unsupported",
                            "body": "This repair still has no validated evidence linkage.",
                            "operation": "CREATE",
                            "confidence": 0.9,
                            "evidence": [{"sequenceStart": 99, "sequenceEnd": 99, "text": "invented evidence"}],
                        }
                    ]
                )
            if self.router.synthesis_mode == "missing_provenance":
                return schema(tasks=tasks, notes=[note])
            return schema()
        if name == "ExtractionQualityReviewResponse":
            return schema(decisions=[])
        if name == "MemoryUpdateResponse":
            return schema(currentSummary="updated")
        return schema()


def _run_workflow(mode: str):
    agents._SEMANTIC_CLASSIFICATION_CACHE.clear()
    router = QualityGateRouter(mode)
    repo = FakeRepository(
        [
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
    )
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
    return router, repo, run


def test_production_pattern_does_not_drop_supported_synthesis_for_missing_metadata():
    result = WindowExtractionResult(
        tasks=[
            _task(1, "Write ulari drain notes", evidence_text="paraphrased drain-note evidence", semanticArtifactKey="drain-notes"),
            _task(2, "Open sequence-wait ticket", evidence_text="paraphrased ticket evidence", semanticArtifactKey="sequence-ticket"),
        ],
        notes=[_note(2, evidence_text="paraphrased ticket evidence", semanticArtifactKey="sequence-ticket")],
        semanticUnits=_units(),
    )
    hydrated = hydrate_synthesized_artifacts(result, _units(), TRANSCRIPT)
    diagnostics = {}
    filtered = score_and_filter_result(hydrated, TRANSCRIPT, diagnostics=diagnostics)
    assert len(filtered.tasks) == 2
    assert len(filtered.notes) == 1
    assert diagnostics["qualityRejectedTaskCount"] == 0
    assert diagnostics["qualityRejectedNoteCount"] == 0
    assert all(item["qualityVerdict"] == "accepted" for item in diagnostics["qualityArtifactDiagnostics"])
    assert all(item["sourceSemanticUnitIds"] for item in diagnostics["qualityArtifactDiagnostics"])
    assert all(record["computedConfidence"] >= record["requiredConfidence"] for record in diagnostics["qualityArtifactDiagnostics"])
    assert default_confidence_policy().publish_threshold == 0.55


def test_hallucinated_task_is_rejected_while_supported_siblings_persist():
    result = WindowExtractionResult(
        tasks=[
            _task(1, "Write ulari drain notes", semanticArtifactKey="drain-notes", sourceSemanticUnitIds=["drain-notes"]),
            ExtractedTask(
                title="Launch orbital beacon",
                body="Launch the orbital beacon before dawn even though nobody mentioned it in the meeting.",
                operation="CREATE",
                confidence=0.99,
                sourceConversationId="conv_final",
                evidence=[EvidenceSpan(sequenceStart=99, sequenceEnd=99, text="invented evidence")],
                origin="unknown",
                changes={"synthesisSource": "llm"},
            ),
        ],
        notes=[_note(2, semanticArtifactKey="sequence-ticket", sourceSemanticUnitIds=["sequence-ticket"])],
        semanticUnits=_units(),
    )
    hydrated = hydrate_synthesized_artifacts(result, _units(), TRANSCRIPT)
    diagnostics = {}
    filtered = score_and_filter_result(hydrated, TRANSCRIPT, diagnostics=diagnostics)
    assert len(filtered.tasks) == 1
    assert filtered.tasks[0].title == "Write ulari drain notes"
    assert len(filtered.notes) == 1
    rejected = [item for item in diagnostics["qualityArtifactDiagnostics"] if item["qualityVerdict"] == "rejected"]
    assert len(rejected) == 1
    assert "evidence_sequence_mismatch" in rejected[0]["qualityRejectionReasons"] or "disconnected_from_validated_units" in rejected[0]["qualityRejectionReasons"]


def test_missing_evidence_propagation_is_restored_from_validated_units():
    result = WindowExtractionResult(
        tasks=[
            _task(1, "Write ulari drain notes", evidence=[], semanticArtifactKey="drain-notes", sourceSemanticUnitIds=["drain-notes"]),
            _task(2, "Open sequence-wait ticket", evidence=[], semanticArtifactKey="sequence-ticket", sourceSemanticUnitIds=["sequence-ticket"]),
        ],
        notes=[_note(2, evidence=[], semanticArtifactKey="sequence-ticket", sourceSemanticUnitIds=["sequence-ticket"])],
        semanticUnits=_units(),
    )
    hydrated = hydrate_synthesized_artifacts(result, _units(), TRANSCRIPT)
    assert all(item.evidence for item in [*hydrated.tasks, *hydrated.notes])
    diagnostics = {}
    filtered = score_and_filter_result(hydrated, TRANSCRIPT, diagnostics=diagnostics)
    assert len(filtered.tasks) == 2
    assert len(filtered.notes) == 1
    assert diagnostics["qualityRejectedTaskCount"] == 0


def test_confidence_boundary_uses_configured_threshold_not_hidden_floor():
    policy = ConfidencePolicy(publish_threshold=0.55)
    supported = hydrate_synthesized_artifacts(
        WindowExtractionResult(tasks=[_task(1, "Write ulari drain notes", semanticArtifactKey="drain-notes")], notes=[], semanticUnits=_units()),
        _units(),
        TRANSCRIPT,
    ).tasks[0]
    keep, reason = validation_decision_for_task(supported, TRANSCRIPT, policy)
    assert keep and reason == "accepted"
    below = ConfidencePolicy(publish_threshold=0.99)
    keep_below, reason_below = validation_decision_for_task(supported, TRANSCRIPT, below)
    assert keep_below is False and reason_below == "low_confidence"


def test_production_pattern_persists_accepted_artifacts(monkeypatch):
    monkeypatch.setattr(
        "services.conversation.transcript.assemble_transcript",
        lambda chunks: SimpleNamespace(raw_transcript=TRANSCRIPT, normalized_transcript=TRANSCRIPT),
    )
    router, repo, run = _run_workflow("production_missing_metadata")
    diagnostics = run.checkpoints["short_raw_transcript"]
    assert diagnostics["validatedSemanticUnitCount"] == 2
    assert diagnostics["finalSynthesisParsedTaskCount"] == 2
    assert diagnostics["finalSynthesisParsedNoteCount"] == 1
    assert diagnostics["finalSynthesisVerdict"] == "PUBLISH"
    assert diagnostics["qualityAcceptedTaskCount"] == 2
    assert diagnostics["qualityAcceptedNoteCount"] == 1
    assert diagnostics["taskCountAfterConfidence"] == 2
    assert diagnostics["noteCountAfterConfidence"] == 1
    assert diagnostics["persistenceOutcome"] == "PERSISTED"
    assert diagnostics["persistedTaskIds"]
    assert diagnostics["persistedNoteIds"]
    assert len(repo.tasks) == 2
    assert len(repo.notes) == 1
    assert router.repair_calls == 0
    records = diagnostics["qualityArtifactDiagnostics"]
    assert len(records) == 3
    assert all("qualityRejectionReasons" in record for record in records)


def test_all_quality_failed_runs_one_repair_and_reports_no_publishable_artifacts(monkeypatch):
    monkeypatch.setattr(
        "services.conversation.transcript.assemble_transcript",
        lambda chunks: SimpleNamespace(raw_transcript=TRANSCRIPT, normalized_transcript=TRANSCRIPT),
    )
    router, repo, run = _run_workflow("all_quality_failed")
    diagnostics = run.checkpoints["short_raw_transcript"]
    assert diagnostics["finalSynthesisVerdict"] == "PUBLISH"
    assert diagnostics["qualityRepairAttempted"] is True
    assert diagnostics["qualityRepairRound"] == 1
    assert router.repair_calls == 1
    assert diagnostics["qualityAcceptedTaskCount"] == 0
    assert diagnostics["qualityAcceptedNoteCount"] == 0
    assert diagnostics["persistenceOutcome"] == "NO_PUBLISHABLE_ARTIFACTS"
    assert diagnostics["tasksPersistedCount"] == 0
    assert diagnostics["notesPersistedCount"] == 0
    assert run.status == ExtractionRunStatus.PUBLISHED
    assert repo.conversation.status in {ConversationStatus.COMPLETED, ConversationStatus.PARTIAL}
    assert not repo.tasks
    assert not repo.notes


def test_partial_rejection_persists_only_supported_artifacts(monkeypatch):
    monkeypatch.setattr(
        "services.conversation.transcript.assemble_transcript",
        lambda chunks: SimpleNamespace(raw_transcript=TRANSCRIPT, normalized_transcript=TRANSCRIPT),
    )
    _router, repo, run = _run_workflow("partial")
    diagnostics = run.checkpoints["short_raw_transcript"]
    assert diagnostics["qualityAcceptedTaskCount"] == 1
    assert diagnostics["qualityAcceptedNoteCount"] == 1
    assert diagnostics["qualityRejectedTaskCount"] == 1
    assert diagnostics["persistenceOutcome"] == "PERSISTED"
    assert len(repo.tasks) == 1
    assert len(repo.notes) == 1
    rejected = [item for item in diagnostics["qualityArtifactDiagnostics"] if item["qualityVerdict"] == "rejected"]
    assert rejected and rejected[0]["artifactType"] == "task"


def test_missing_provenance_is_restored_or_repaired_then_persisted(monkeypatch):
    monkeypatch.setattr(
        "services.conversation.transcript.assemble_transcript",
        lambda chunks: SimpleNamespace(raw_transcript=TRANSCRIPT, normalized_transcript=TRANSCRIPT),
    )
    _router, repo, run = _run_workflow("missing_provenance")
    diagnostics = run.checkpoints["short_raw_transcript"]
    assert diagnostics["qualityAcceptedTaskCount"] >= 1
    assert diagnostics["persistenceOutcome"] == "PERSISTED"
    assert repo.tasks or diagnostics["qualityRepairAttempted"] is True
    assert diagnostics["persistedTaskIds"] or diagnostics["persistedNoteIds"]


def test_extract_window_records_per_artifact_quality_reasons():
    agents._SEMANTIC_CLASSIFICATION_CACHE.clear()
    router = QualityGateRouter("production_missing_metadata")
    result, _, _ = asyncio.run(agents.extract_window(router, _window(TRANSCRIPT), context={}, meeting_context={}, mode="final"))
    diagnostics = result.extractionDiagnostics
    assert diagnostics["finalSynthesisParsedTaskCount"] == 2
    assert diagnostics["finalSynthesisParsedNoteCount"] == 1
    assert diagnostics["qualityAcceptedTaskCount"] == 2
    assert diagnostics["qualityAcceptedNoteCount"] == 1
    for record in diagnostics["qualityArtifactDiagnostics"]:
        assert "artifactId" in record
        assert "artifactType" in record
        assert "evidenceSpanCount" in record
        assert "evidenceIntegrityScore" in record
        assert "groundingScore" in record
        assert "contextCompletenessScore" in record
        assert "corroborationScore" in record
        assert "ambiguityScore" in record
        assert "consistencyScore" in record
        assert "computedConfidence" in record
        assert "requiredConfidence" in record
        assert "qualityVerdict" in record
        assert "qualityRejectionReasons" in record
        assert record["requiredConfidence"] == 0.55
