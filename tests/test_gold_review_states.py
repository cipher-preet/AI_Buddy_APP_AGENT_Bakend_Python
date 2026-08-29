"""Benchmark review states and semantic scoring. Does not change runtime validation."""

from __future__ import annotations

import asyncio

from services.conversation.event_pipeline.embeddings import CachedEmbedder, LexicalEmbedder
from services.conversation.event_pipeline.events import ScriptedEventExtractor
from services.conversation.event_pipeline.gold_scoring import (
    GoldFailureClass,
    artifacts_semantically_equivalent,
    pipeline_benchmark,
)
from services.conversation.event_pipeline.pipeline import run_event_pipeline
from services.conversation.event_pipeline.schemas import EventPipelineResult
from services.conversation.models import EvidenceSpan, ExtractedNote, ExtractedTask
from tests.fixtures.long_meeting_gold import build_gold_transcript
from tests.fixtures.reviewed_meetings import build_meeting_b


def _task(title: str, body: str, sequences: list[int], text: str = "") -> ExtractedTask:
    evidence = [
        EvidenceSpan(sequenceStart=sequence, sequenceEnd=sequence, text=text or title)
        for sequence in sequences
    ]
    return ExtractedTask(
        title=title,
        body=body,
        operation="CREATE",
        confidence=0.8,
        sourceConversationId="conv",
        evidence=evidence,
    )


def _note(title: str, body: str, sequences: list[int], text: str = "") -> ExtractedNote:
    evidence = [
        EvidenceSpan(sequenceStart=sequence, sequenceEnd=sequence, text=text or title)
        for sequence in sequences
    ]
    return ExtractedNote(
        title=title,
        body=body,
        confidence=0.8,
        sourceConversationId="conv",
        evidence=evidence,
    )


def test_keep_and_preserve_are_semantically_equivalent_actions():
    assert artifacts_semantically_equivalent(
        "Preserve the retry limit for billing attempts",
        "Keep billing retry limit at 3",
        pred_seqs=[110],
        gold_seqs=[10],
        pred_verb="preserve",
        gold_verb="keep",
        pred_object="billing retry limit",
        gold_object="billing retry limit",
    )
    assert not artifacts_semantically_equivalent(
        "Update the docs",
        "Keep billing retry limit at 3",
        pred_seqs=[70],
        gold_seqs=[10],
    )


def test_wording_variance_does_not_count_as_extraction_miss():
    gold_tasks = [
        {
            "id": "t-billing",
            "kind": "task",
            "meaning": "Keep billing retry limit at 3",
            "evidenceSequences": [10],
            "reviewStatus": "REQUIRED",
        }
    ]
    result = EventPipelineResult(
        tasks=[
            _task(
                "Preserve the retry limit for billing attempts",
                "Preserve the retry limit for billing attempts.",
                [110],
                "billing retry wapas discuss kiya, limit same rahegi",
            )
        ]
    )
    transcript = "[10] billing retry limit 3 pe rakhna hai\n[110] billing retry wapas discuss kiya, limit same rahegi"
    report = pipeline_benchmark(
        result,
        gold_tasks,
        [],
        case_id="wording-variance",
        transcript=transcript,
        gold_complete=True,
    )
    assert report["requiredTaskRecall"] == 1.0
    assert report["taskRecall"] == 1.0
    assert report["matchedTasks"] == 1
    assert not report["goldFailures"]


def test_required_recall_ignores_unpublished_optional_note():
    gold_notes = [
        {
            "id": "n-keys",
            "kind": "note",
            "meaning": "Old keys are currently in use",
            "evidenceSequences": [90],
            "reviewStatus": "REQUIRED",
        },
        {
            "id": "n-monday",
            "kind": "note",
            "meaning": "Monday meeting mentioned",
            "evidenceSequences": [42],
            "reviewStatus": "OPTIONAL_VALID",
        },
        {
            "id": "n-low",
            "kind": "note",
            "meaning": "Someone coughed",
            "evidenceSequences": [3],
            "reviewStatus": "LOW_VALUE",
        },
        {
            "id": "n-bad",
            "kind": "note",
            "meaning": "Invented gold",
            "evidenceSequences": [4],
            "reviewStatus": "INVALID_GOLD",
        },
    ]
    result = EventPipelineResult(
        notes=[_note("Old keys in use", "Old keys are currently in use.", [90], "old keys are currently in use")]
    )
    report = pipeline_benchmark(
        result,
        [],
        gold_notes,
        case_id="optional-monday",
        transcript="[90] old keys are currently in use\n[42] Monday meeting mein pricing finalize karenge",
        gold_complete=True,
    )
    assert report["requiredNoteRecall"] == 1.0
    assert report["noteRecall"] == 1.0
    assert report["optionalValidFound"] == 0
    assert report["lowValueSuppressed"] == 1
    assert report["invalidGoldCount"] == 1
    assert report["missingNotes"] == 0


def test_meeting_b_scripted_gold_tasks_have_complete_traces():
    meeting = build_meeting_b()
    transcript = "\n".join(f"[{seq}] {text}" for seq, text in meeting["lines"].items() if str(text).strip())
    result = asyncio.run(
        run_event_pipeline(
            meeting["chunks"],
            meeting["id"],
            "user_1",
            "space_1",
            event_extractor=ScriptedEventExtractor(events=meeting["events"]),
            embedder=CachedEmbedder(LexicalEmbedder()),
        )
    )
    report = pipeline_benchmark(
        result,
        meeting["goldTasks"],
        meeting["goldNotes"],
        case_id=meeting["id"],
        transcript=transcript,
        gold_events=meeting["events"],
        gold_threads=meeting.get("goldThreads"),
        gold_complete=True,
        original_actionable_ids=meeting.get("originalActionableEventIds"),
        reviewed_actionable_ids=meeting.get("reviewedActionableEventIds"),
    )
    task_traces = [row for row in report["goldTraces"] if row["kind"] == "task"]
    assert len(task_traces) == 3
    for trace in task_traces:
        assert trace["sourceTranscript"]
        assert trace["microBlocks"]
        assert trace["topics"]
        assert trace["events"]
        assert trace["actionSignal"]["isActionable"] is True
        assert trace["actionSignal"]["rawVerb"]
        assert trace["actionSignal"]["canonicalObject"]
        assert trace["threads"]
        assert trace["finalArtifact"]
        assert trace["benchmarkMatch"] is True
        assert trace["failureClass"] is None
        assert trace["reviewStatus"] == "REQUIRED"
    assert report["requiredTaskRecall"] == 1.0
    assert report["groundedPrecisionTasks"] == 1.0
    assert not report["goldFailures"]


def test_monday_optional_note_does_not_force_required_recall_on_scripted_221():
    gold = build_gold_transcript()
    monday = next(item for item in gold["goldNotes"] if item["id"] == "n-monday")
    assert monday["reviewStatus"] == "OPTIONAL_VALID"
    transcript = "\n".join(f"[{seq}] {text}" for seq, text in gold["lines"].items() if str(text).strip())
    result = asyncio.run(
        run_event_pipeline(
            gold["chunks"],
            "gold-long-meeting",
            "user_1",
            "space_1",
            event_extractor=ScriptedEventExtractor(events=gold["events"]),
            embedder=CachedEmbedder(LexicalEmbedder()),
        )
    )
    report = pipeline_benchmark(
        result,
        gold["goldTasks"],
        gold["goldNotes"],
        case_id="gold-long-meeting",
        transcript=transcript,
        valid_additional_notes=gold.get("validAdditionalNotes"),
        valid_additional_tasks=gold.get("validAdditionalTasks"),
        gold_events=gold["events"],
        gold_complete=True,
    )
    monday_trace = next(row for row in report["goldTraces"] if row["goldId"] == "n-monday")
    assert monday_trace["reviewStatus"] == "OPTIONAL_VALID"
    assert report["requiredNoteRecall"] == 1.0
    assert GoldFailureClass.EXTRACTION_MISS.value not in {
        row.get("failureClass") for row in report["goldTraces"] if row["goldId"] == "n-monday"
    }
