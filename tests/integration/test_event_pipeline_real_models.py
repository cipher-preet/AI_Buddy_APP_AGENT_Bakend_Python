"""Full gold transcript through the real production event-pipeline path.

Does not mock embeddings, Gemma extraction, thread verification, gpt-oss
synthesis, or validation.

    pytest tests/integration/test_event_pipeline_real_models.py -v
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest

from services.conversation.event_pipeline.cost import cost_report
from services.conversation.event_pipeline.embeddings import default_embedder
from services.conversation.event_pipeline.gold_scoring import NOT_MEASURED, pipeline_benchmark
from services.conversation.event_pipeline.pipeline import run_event_pipeline
from services.llm.router import get_llm_router
from tests.fixtures.long_meeting_gold import build_gold_transcript
from tests.fixtures.scale_meetings import build_scale_meeting
from tests.integration.conftest import requires_real_models


pytestmark = [pytest.mark.integration, pytest.mark.real_models, requires_real_models]


def test_gold_long_meeting_real_production_models():
    gold = build_gold_transcript()
    transcript = "\n".join(f"[{seq}] {text}" for seq, text in gold["lines"].items() if str(text).strip())
    result = asyncio.run(
        run_event_pipeline(
            gold["chunks"],
            "gold-long-meeting-real",
            "user_1",
            "space_1",
            router=get_llm_router(),
            embedder=default_embedder(prefer_provider=True, lexical_fallback=False),
            polish_with_llm=True,
        )
    )
    report = pipeline_benchmark(
        result,
        gold["goldTasks"],
        gold["goldNotes"],
        case_id="gold-long-meeting-real",
        transcript=transcript,
        valid_additional_notes=gold.get("validAdditionalNotes"),
        valid_additional_tasks=gold.get("validAdditionalTasks"),
        gold_events=gold["events"],
        gold_threads=gold.get("goldThreads"),
        gold_complete=True,
        original_actionable_ids=gold.get("originalActionableEventIds"),
        reviewed_actionable_ids=gold.get("reviewedActionableEventIds"),
    )
    cost = cost_report(result.observability)
    print("GOLD_LONG_MEETING_BENCHMARK_REAL", json.dumps(report, default=str, ensure_ascii=True))
    print("REAL_MODEL_COST", json.dumps(cost, default=str, ensure_ascii=True))
    print("REAL_MODEL_ROUTES", json.dumps([item.model_dump() for item in result.observability.modelRoutes], default=str, ensure_ascii=True))
    print("STAGE_COUNTS", json.dumps(_stage_counts(result, report), default=str, ensure_ascii=True))
    print("QUALITY_METRICS", json.dumps(_quality_metrics(report), default=str, ensure_ascii=True))
    print("ARTIFACT_LABELS", json.dumps(_artifact_labels(report), default=str, ensure_ascii=True))
    print("SEMANTIC_CASE_HITS", json.dumps(_semantic_case_hits(result), default=str, ensure_ascii=True))
    print("NEGATIVE_CHECKS", json.dumps(_negative_checks(result, gold), default=str, ensure_ascii=True))
    print("TOPIC_INSPECT", json.dumps(_topic_inspect(result), default=str, ensure_ascii=True))
    print(
        "TASK_INSPECT",
        json.dumps(
            [{"title": task.title, "body": task.body, "sequences": [span.sequenceStart for span in task.evidence]} for task in result.tasks],
            default=str,
            ensure_ascii=True,
        ),
    )
    print("ACTION_INSPECT", json.dumps(_action_inspect(result), default=str, ensure_ascii=True))
    print("NOTE_INSPECT", json.dumps([{"title": note.title, "body": note.body[:160]} for note in result.notes], default=str, ensure_ascii=True))
    print("ACTION_OBJECT_FAILURES", json.dumps(report.get("actionObjectFailures") or [], default=str, ensure_ascii=True))
    print(
        "TASK_CLASSIFICATIONS",
        json.dumps(report.get("taskClassifications") or [], default=str, ensure_ascii=True),
    )
    print("ACTION_CLASSIFICATIONS", json.dumps(report.get("actionClassifications") or [], default=str, ensure_ascii=True))
    print("NOTE_QUALITY", json.dumps(report.get("noteQuality") or [], default=str, ensure_ascii=True))
    print("REQUIRED_RECALL", json.dumps({
        "requiredTaskRecall": report.get("requiredTaskRecall"),
        "requiredNoteRecall": report.get("requiredNoteRecall"),
        "optionalValidFound": report.get("optionalValidFound"),
        "lowValueSuppressed": report.get("lowValueSuppressed"),
        "invalidGoldCount": report.get("invalidGoldCount"),
    }, default=str))
    print("MONDAY_NOTE_TRACE", json.dumps(next((row for row in report.get("goldTraces") or [] if row.get("goldId") == "n-monday"), None), default=str, ensure_ascii=True))

    assert result.observability.embeddingItems > 0
    assert result.observability.gemmaCalls >= 1
    routes = {item.stage: item for item in result.observability.modelRoutes}
    if "atomic_event_extraction" in routes:
        assert routes["atomic_event_extraction"].capability == "SEMANTIC_EXTRACTION"
    if "task_synthesis" in routes:
        assert routes["task_synthesis"].capability == "HIGH_ACCURACY_REASONING"
    if "evidence_validation" in routes:
        assert routes["evidence_validation"].capability == "VALIDATION"
        assert "gpt-oss-20b" in (routes["evidence_validation"].model or "").casefold() or routes["evidence_validation"].fallback
    assert report["extractorMode"] != "scripted"
    assert report["unaccountedBlocks"] == 0
    assert report["genericTaskRate"] < 0.15
    assert result.notes, "memory events must not collapse to notes=0"
    assert "gptOss120bCalls" in report["observability"]
    event_count = max(len(result.events), 1)
    assert result.observability.gptOss120bCalls < event_count * (event_count - 1) / 2
    assert report["groundedPrecisionNotes"] != NOT_MEASURED


@pytest.mark.parametrize("size", [50, 150])
def test_shorter_real_meetings_report_honest_metrics(size: int):
    meeting = build_scale_meeting(size)
    result = asyncio.run(
        run_event_pipeline(
            meeting["chunks"][:size],
            meeting["id"],
            "u",
            "s",
            router=get_llm_router(),
            embedder=default_embedder(prefer_provider=True, lexical_fallback=False),
            polish_with_llm=True,
        )
    )
    from services.conversation.event_pipeline.gold_scoring import e2e_scale_report

    report = e2e_scale_report(result, gold={"goldComplete": False}, case_id=meeting["id"])
    print(f"REAL_SCALE_{size}", report)
    assert report["taskPrecision"] == NOT_MEASURED
    assert result.coverage is None or result.coverage.unaccounted_blocks == 0 or result.events


def test_reviewed_meetings_real_production_models():
    from tests.fixtures.reviewed_meetings import all_reviewed_meetings

    summaries = []
    for meeting in all_reviewed_meetings():
        transcript = "\n".join(f"[{seq}] {text}" for seq, text in meeting["lines"].items() if str(text).strip())
        result = asyncio.run(
            run_event_pipeline(
                meeting["chunks"],
                f"{meeting['id']}-real",
                "user_1",
                "space_1",
                router=get_llm_router(),
                embedder=default_embedder(prefer_provider=True, lexical_fallback=False),
                polish_with_llm=True,
            )
        )
        report = pipeline_benchmark(
            result,
            meeting["goldTasks"],
            meeting["goldNotes"],
            case_id=f"{meeting['id']}-real",
            transcript=transcript,
            valid_additional_notes=meeting.get("validAdditionalNotes"),
            valid_additional_tasks=meeting.get("validAdditionalTasks"),
            gold_events=meeting["events"],
            gold_threads=meeting.get("goldThreads"),
            gold_complete=True,
            original_actionable_ids=meeting.get("originalActionableEventIds"),
            reviewed_actionable_ids=meeting.get("reviewedActionableEventIds"),
        )
        row = {
            "id": meeting["id"],
            "size": meeting["size"],
            "groundedPrecisionTasks": report["groundedPrecisionTasks"],
            "taskRecall": report["taskRecall"],
            "requiredTaskRecall": report.get("requiredTaskRecall"),
            "noteFactualPrecision": report["noteFactualPrecision"],
            "noteUsefulnessPrecision": report["noteUsefulnessPrecision"],
            "noteRecall": report["noteRecall"],
            "requiredNoteRecall": report.get("requiredNoteRecall"),
            "optionalValidFound": report.get("optionalValidFound"),
            "lowValueSuppressed": report.get("lowValueSuppressed"),
            "invalidGoldCount": report.get("invalidGoldCount"),
            "duplicateTaskRate": report.get("duplicateTaskRate"),
            "noteDuplicateRate": report["noteDuplicateRate"],
            "duplicateArtifactRate": report.get("duplicateArtifactRate"),
            "mixedThreadRate": report["mixedThreadRate"],
            "genericTaskRate": report["genericTaskRate"],
            "unaccountedBlocks": report["unaccountedBlocks"],
            "memoryCoverageFailure": report.get("memoryCoverageFailure"),
            "memoryUnaccounted": report.get("memoryUnaccounted"),
        }
        summaries.append(row)
        print(f"REVIEWED_MEETING_REAL_{meeting['id']}", json.dumps(row, default=str, ensure_ascii=True))
        if meeting["id"] == "meeting-b":
            print("MEETING_B_GOLD_TRACES", json.dumps(report.get("goldTraces") or [], default=str, ensure_ascii=True))
            print("MEETING_B_GOLD_FAILURES", json.dumps(report.get("goldFailures") or [], default=str, ensure_ascii=True))
            print("MEETING_B_TASK_INSPECT", json.dumps(
                [{"title": task.title, "body": task.body, "sequences": [span.sequenceStart for span in task.evidence]} for task in result.tasks],
                default=str,
                ensure_ascii=True,
            ))
        if meeting["id"] == "meeting-c":
            print("MEETING_C_MEMORY_TRACES", json.dumps(result.diagnostics.get("memoryTraces") or [], default=str, ensure_ascii=True))
            print("MEETING_C_NOTE_INSPECT", json.dumps([{"title": note.title, "body": note.body[:200]} for note in result.notes], default=str, ensure_ascii=True))
            print("MEETING_C_NOTE_LABELS", json.dumps(report.get("noteClassifications") or [], default=str, ensure_ascii=True))
        assert report["unaccountedBlocks"] == 0
        assert report["genericTaskRate"] == 0
        assert report.get("memoryUnaccounted", 0) == 0
    macro_keys = [
        "groundedPrecisionTasks",
        "taskRecall",
        "noteUsefulnessPrecision",
        "noteRecall",
        "noteDuplicateRate",
        "mixedThreadRate",
        "genericTaskRate",
        "unaccountedBlocks",
    ]
    macro = {key: sum(item[key] for item in summaries) / len(summaries) for key in macro_keys}
    print("REVIEWED_MEETINGS_MACRO", json.dumps(macro, default=str, ensure_ascii=True))
    assert summaries


def test_anomaly_fixture_stability_real_models():
    """Repeat Meeting B and the 221-sequence gold on production models.

    Generative variance is the measurement. Do not treat a single run as readiness.
    """
    from tests.fixtures.reviewed_meetings import build_meeting_b

    runs = max(3, int(os.environ.get("EVENT_PIPELINE_STABILITY_RUNS", "3")))
    fixtures = [
        ("meeting-b", build_meeting_b()),
        ("gold-221", build_gold_transcript()),
    ]
    summaries: dict[str, list[dict]] = {}
    appearance: dict[str, dict[str, int]] = {}
    for label, meeting in fixtures:
        gold_tasks = meeting.get("goldTasks") or []
        gold_notes = meeting.get("goldNotes") or []
        appearance[label] = {item["id"]: 0 for item in [*gold_tasks, *gold_notes]}
        rows = []
        for run_index in range(runs):
            report, result = _run_real_benchmark(meeting, f"{meeting.get('id', label)}-stability-{run_index}")
            matched = {
                row.get("matchedGoldId")
                for row in [*(report.get("taskClassifications") or []), *(report.get("noteClassifications") or [])]
                if row.get("label") == "MATCHED_GOLD" and row.get("matchedGoldId")
            }
            for gold_id in appearance[label]:
                if gold_id in matched:
                    appearance[label][gold_id] += 1
            row = {
                "run": run_index + 1,
                "requiredTaskRecall": report.get("requiredTaskRecall"),
                "taskRecall": report.get("taskRecall"),
                "groundedPrecisionTasks": report.get("groundedPrecisionTasks"),
                "requiredNoteRecall": report.get("requiredNoteRecall"),
                "noteRecall": report.get("noteRecall"),
                "noteUsefulnessPrecision": report.get("noteUsefulnessPrecision"),
                "optionalValidFound": report.get("optionalValidFound"),
                "lowValueSuppressed": report.get("lowValueSuppressed"),
                "invalidGoldCount": report.get("invalidGoldCount"),
                "duplicateRate": report.get("duplicateArtifactRate", 0.0),
                "duplicateTaskRate": report.get("duplicateTaskRate", 0.0),
                "noteDuplicateRate": report.get("noteDuplicateRate", 0.0),
                "scoreCaseDuplicateRate": report.get("duplicateRate", 0.0),
                "mixedThreadRate": report.get("mixedThreadRate"),
                "genericTaskRate": report.get("genericTaskRate"),
                "unaccountedBlocks": report.get("unaccountedBlocks"),
                "goldFailures": report.get("goldFailures") or [],
                "tasks": [{"title": task.title, "body": task.body[:120]} for task in result.tasks],
                "notes": [{"title": note.title, "body": note.body[:120]} for note in result.notes],
            }
            rows.append(row)
            print(f"STABILITY_{label}_RUN_{run_index + 1}", json.dumps(row, default=str, ensure_ascii=True))
            assert report["unaccountedBlocks"] == 0
            assert report["genericTaskRate"] == 0
        summaries[label] = rows
        task_recalls = [item["requiredTaskRecall"] for item in rows]
        note_recalls = [item["requiredNoteRecall"] for item in rows]
        print(
            f"STABILITY_{label}_DISTRIBUTION",
            json.dumps(
                {
                    "runs": runs,
                    "requiredTaskRecall": {"mean": sum(task_recalls) / runs, "min": min(task_recalls), "max": max(task_recalls), "values": task_recalls},
                    "requiredNoteRecall": {"mean": sum(note_recalls) / runs, "min": min(note_recalls), "max": max(note_recalls), "values": note_recalls},
                    "appearanceRate": {gold_id: f"{count}/{runs}" for gold_id, count in appearance[label].items()},
                    "unstable": [gold_id for gold_id, count in appearance[label].items() if 0 < count < runs],
                },
                default=str,
                ensure_ascii=True,
            ),
        )
    print("STABILITY_APPEARANCE", json.dumps(appearance, default=str, ensure_ascii=True))
    assert summaries["meeting-b"]
    assert summaries["gold-221"]


def _run_real_benchmark(meeting: dict, conversation_id: str):
    transcript = "\n".join(f"[{seq}] {text}" for seq, text in meeting["lines"].items() if str(text).strip())
    result = asyncio.run(
        run_event_pipeline(
            meeting["chunks"],
            conversation_id,
            "user_1",
            "space_1",
            router=get_llm_router(),
            embedder=default_embedder(prefer_provider=True, lexical_fallback=False),
            polish_with_llm=True,
        )
    )
    report = pipeline_benchmark(
        result,
        meeting.get("goldTasks") or [],
        meeting.get("goldNotes") or [],
        case_id=conversation_id,
        transcript=transcript,
        valid_additional_notes=meeting.get("validAdditionalNotes"),
        valid_additional_tasks=meeting.get("validAdditionalTasks"),
        gold_events=meeting.get("events"),
        gold_threads=meeting.get("goldThreads"),
        gold_complete=True,
        original_actionable_ids=meeting.get("originalActionableEventIds"),
        reviewed_actionable_ids=meeting.get("reviewedActionableEventIds"),
    )
    return report, result


def _stage_counts(result, report: dict) -> dict:
    coverage = result.coverage
    return {
        "RAW": report["counts"]["rawChunks"],
        "USEFUL": report["counts"]["usefulChunks"],
        "MICRO_BLOCKS": report["counts"]["microBlocks"],
        "TOPICS": report["counts"]["topics"],
        "ATOMIC_EVENTS": report["counts"]["events"],
        "GLOBAL_THREADS": report["counts"]["threads"],
        "ACTION_EVENTS": report["counts"].get("actionEvents"),
        "MEMORY_EVENTS": report["counts"].get("memoryEvents"),
        "TASKS": report["counts"]["tasks"],
        "NOTES": report["counts"]["notes"],
        "REJECTED": coverage.rejected_events if coverage is not None else report["counts"].get("rejected"),
        "UNACCOUNTED": report["counts"].get("unaccounted"),
    }


def _quality_metrics(report: dict) -> dict:
    return {
        "atomicEventPrecision": report.get("eventPrecision"),
        "atomicEventRecall": report.get("eventRecall"),
        "eventTypeAccuracy": report.get("eventTypeAccuracy"),
        "actionabilityPrecision": report.get("actionabilityPrecision"),
        "actionabilityRecall": report.get("actionabilityRecall"),
        "actionVerbPrecision": report.get("actionVerbPrecision"),
        "actionVerbRecall": report.get("actionVerbRecall"),
        "actionObjectPrecision": report.get("actionObjectPrecision"),
        "actionObjectRecall": report.get("actionObjectRecall"),
        "explicitObjectAccuracy": report.get("explicitObjectAccuracy"),
        "coreferenceObjectAccuracy": report.get("coreferenceObjectAccuracy"),
        "inferredObjectRejectionRate": report.get("inferredObjectRejectionRate"),
        "actionObjectAccuracy": report.get("actionObjectAccuracy"),
        "actionObjectGroundingPrecision": report.get("actionObjectGroundingPrecision"),
        "actionObjectGroundingRecall": report.get("actionObjectGroundingRecall"),
        "strictGoldPrecisionTasks": report.get("strictGoldPrecisionTasks"),
        "groundedPrecisionTasks": report.get("groundedPrecisionTasks"),
        "taskPrecision": report.get("taskPrecision"),
        "taskRecall": report.get("taskRecall"),
        "requiredTaskRecall": report.get("requiredTaskRecall"),
        "noteGroundedPrecision": report.get("groundedPrecisionNotes"),
        "noteFactualPrecision": report.get("noteFactualPrecision"),
        "noteUsefulnessPrecision": report.get("noteUsefulnessPrecision"),
        "noteRecall": report.get("noteRecall"),
        "requiredNoteRecall": report.get("requiredNoteRecall"),
        "optionalValidFound": report.get("optionalValidFound"),
        "lowValueSuppressed": report.get("lowValueSuppressed"),
        "invalidGoldCount": report.get("invalidGoldCount"),
        "evidencePrecision": report.get("evidencePrecision"),
        "threadPrecision": report.get("threadPrecision"),
        "threadRecall": report.get("threadRecall"),
        "genericTaskRate": report.get("genericTaskRate"),
        "mixedThreadRate": report.get("mixedThreadRate"),
        "duplicateTaskRate": report.get("duplicateTaskRate"),
        "duplicateArtifactRate": report.get("duplicateArtifactRate"),
        "unaccountedBlockCount": report.get("unaccountedBlocks"),
        "memoryCoverageFailure": report.get("memoryCoverageFailure"),
        "memoryUnaccounted": report.get("memoryUnaccounted"),
        "memoryPublished": report.get("memoryPublished"),
        "memoryDuplicates": report.get("memoryDuplicates"),
        "memoryUpdates": report.get("memoryUpdates"),
        "strictOriginalGoldPrecision": report.get("strictOriginalGoldPrecision"),
        "reviewedActionPrecision": report.get("reviewedActionPrecision"),
        "reviewedActionRecall": report.get("reviewedActionRecall"),
        "objectSemanticAccuracy": report.get("objectSemanticAccuracy"),
        "objectSurfaceNormalizationAccuracy": report.get("objectSurfaceNormalizationAccuracy"),
        "objectGroundingAccuracy": report.get("objectGroundingAccuracy"),
        "topics": report.get("counts", {}).get("topics"),
        "actionEvents": report.get("counts", {}).get("actionEvents"),
        "groundedActionObjects": report.get("counts", {}).get("groundedActionObjects"),
        "tasks": report.get("counts", {}).get("tasks"),
        "notes": report.get("counts", {}).get("notes"),
    }


def _artifact_labels(report: dict) -> dict:
    return {
        "MATCHED_GOLD": {
            "tasks": report.get("matchedTasks"),
            "notes": report.get("matchedNotes"),
        },
        "VALID_ADDITIONAL": {
            "tasks": report.get("validAdditionalTasks"),
            "notes": report.get("validAdditionalNotes"),
        },
        "FALSE_POSITIVE": {
            "tasks": report.get("falsePositiveTasks"),
            "notes": report.get("falsePositiveNotes"),
        },
        "DUPLICATE": {
            "tasks": report.get("duplicateTasks"),
            "notes": report.get("duplicateNotes"),
        },
        "TOO_VAGUE": {
            "tasks": report.get("tooVagueTasks"),
            "notes": report.get("tooVagueNotes"),
        },
        "UNSUPPORTED": {
            "tasks": report.get("unsupportedTasks"),
            "notes": report.get("unsupportedNotes"),
        },
        "MISSING": {
            "tasks": report.get("missingTasks"),
            "notes": report.get("missingNotes"),
        },
    }


def _blob(result) -> str:
    parts = [f"{item.title} {item.body}" for item in [*result.tasks, *result.notes]]
    parts.extend(event.meaning for event in result.events)
    return " ".join(parts).casefold()


def _semantic_case_hits(result) -> dict:
    blob = _blob(result)
    probes = {
        "S3/frontend issue": ("s3", "frontend"),
        "old keys currently in use": ("old key", "purani key", "legacy key"),
        "create meeting page": ("meeting page",),
        "create server ID": ("server id", "server identifier"),
        "microphone-access requirement/problem": ("microphone", "mic "),
        "connection reported insecure": ("insecure",),
        "missing network parameter information": ("network param", "network parameter"),
        "Port ID follow-up": ("port id", "port tracking"),
        "Play Store problem": ("play store",),
        "master-prompt / GPT/OpenCV requirement": ("opencv", "master-prompt", "master prompt"),
    }
    return {label: any(needle in blob for needle in needles) for label, needles in probes.items()}


def _negative_checks(result, gold) -> dict:
    generic = [
        f"{task.title} {task.body}"
        for task in result.tasks
        if "complete pending" in f"{task.title} {task.body}".casefold()
    ]
    server = next(
        (task for task in result.tasks if "server" in task.title.casefold() and "id" in task.title.casefold()),
        None,
    )
    forbidden = []
    if server is not None:
        evidence_sequences = {span.sequenceStart for span in server.evidence} | {span.sequenceEnd for span in server.evidence}
        forbidden = sorted(evidence_sequences & set(gold.get("serverIdForbiddenSequences") or []))
    return {
        "completePendingTaskSurvived": bool(generic),
        "completePendingExamples": generic,
        "serverIdForbiddenEvidenceSequences": forbidden,
    }


def _topic_inspect(result) -> list[dict]:
    rows = []
    for topic in result.topics:
        rows.append(
            {
                "topicId": topic.topicId,
                "label": topic.label,
                "microBlocks": len(topic.microBlockIds),
                "sequences": list(topic.sequenceIds),
                "coherence": getattr(topic, "coherence", None),
                "boundaryReason": getattr(topic, "boundaryReason", None),
                "entities": list(topic.entities or []),
                "textPreview": (topic.text or "")[:160],
            }
        )
    return rows


def test_production_hinglish_commitments_real_models():
    """Same production path: STOP finalization → Gemma extraction → synthesis → validation."""
    from services.conversation.models import STTStatus, TranscriptChunkDocument

    lines = {
        0: "HRMS तो हमें बनाना ही है.",
        1: "candidate को onboard करेंगे share candidate detail link generate होगा",
        2: "AI hiring थोड़ा बना लेंगे इसको.",
        3: "payroll उसको भी बनाएंगे.",
        4: "इसको पूरा देखना है कि direct integration क्या use करते हैं.",
    }
    chunks = [
        TranscriptChunkDocument(
            conversationId="prod-hinglish-real",
            userId="u",
            spaceId="s",
            chunkId=f"chunk_{seq}",
            sequenceNumber=seq,
            rawText=text,
            sttStatus=STTStatus.COMPLETED,
        )
        for seq, text in lines.items()
    ]
    result = asyncio.run(
        run_event_pipeline(
            chunks,
            "prod-hinglish-real",
            "u",
            "s",
            router=get_llm_router(),
            embedder=default_embedder(prefer_provider=True, lexical_fallback=False),
            polish_with_llm=True,
        )
    )
    obs = result.observability
    print(
        "PROD_HINGLISH_STAGE_COUNTS",
        json.dumps(
            {
                "atomicEvents": obs.atomicEvents,
                "actionableEvents": obs.actionableEvents,
                "groundedActionObjects": obs.groundedActionObjects,
                "taskSynthesisInput": obs.taskSynthesisInputEvents,
                "taskCandidates": obs.taskCandidates,
                "accepted": obs.taskValidationAccepted,
                "persisted": obs.tasksPersisted,
                "returned": obs.tasksReturnedByApi,
                "tasks": [{"title": task.title, "body": task.body} for task in result.tasks],
                "notes": [{"title": note.title, "body": note.body[:200]} for note in result.notes],
            },
            default=str,
            ensure_ascii=True,
        ),
    )
    assert "[TASK_PIPELINE_TRACE]" in " ".join(obs.logs)
    if obs.groundedActionObjects > 0:
        assert result.tasks, "grounded explicit actions must not silently yield zero tasks"
    note_blob = " ".join(f"{note.title} {note.body}" for note in result.notes).casefold()
    assert "is directly integrated" not in note_blob or "whether" in note_blob or "determine" in note_blob


def _action_inspect(result) -> list[dict]:
    rows = []
    for event in result.events:
        signal = event.actionSignal
        if not (signal and signal.isActionable) and event.channel != "action":
            continue
        rows.append(
            {
                "eventId": event.eventId,
                "kind": event.kind.value if hasattr(event.kind, "value") else event.kind,
                "meaning": event.meaning,
                "verb": getattr(signal, "verb", None) if signal else None,
                "object": (signal.object if signal else None) or event.object,
                "rawActionObject": getattr(signal, "rawActionObject", None) if signal else None,
                "canonicalActionObject": getattr(signal, "canonicalActionObject", None) if signal else None,
                "actionStrength": getattr(signal, "actionStrength", None) if signal else None,
                "objectGroundingType": getattr(signal, "objectGroundingType", None) if signal else None,
                "artifactStatus": getattr(signal, "artifactStatus", None) if signal else None,
                "disposition": event.disposition.value if event.disposition else None,
                "grounded": bool(event.object),
            }
        )
    return rows
