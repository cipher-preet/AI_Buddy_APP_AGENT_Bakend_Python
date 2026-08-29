"""Scripted multi-meeting and note-quality checks. Does not require live models."""

from __future__ import annotations

import asyncio

from services.conversation.event_pipeline.embeddings import CachedEmbedder, LexicalEmbedder
from services.conversation.event_pipeline.events import ScriptedEventExtractor
from services.conversation.event_pipeline.gold_scoring import pipeline_benchmark
from services.conversation.event_pipeline.pipeline import run_event_pipeline
from tests.fixtures.reviewed_meetings import all_reviewed_meetings


def test_reviewed_meetings_scripted_quality():
    embedder = CachedEmbedder(LexicalEmbedder())
    summaries = []
    for meeting in all_reviewed_meetings():
        transcript = "\n".join(f"[{seq}] {text}" for seq, text in meeting["lines"].items() if str(text).strip())
        result = asyncio.run(
            run_event_pipeline(
                meeting["chunks"],
                meeting["id"],
                "user_1",
                "space_1",
                event_extractor=ScriptedEventExtractor(events=meeting["events"]),
                embedder=embedder,
            )
        )
        report = pipeline_benchmark(
            result,
            meeting["goldTasks"],
            meeting["goldNotes"],
            case_id=meeting["id"],
            transcript=transcript,
            valid_additional_notes=meeting.get("validAdditionalNotes"),
            valid_additional_tasks=meeting.get("validAdditionalTasks"),
            gold_events=meeting["events"],
            gold_threads=meeting.get("goldThreads"),
            gold_complete=True,
            original_actionable_ids=meeting.get("originalActionableEventIds"),
            reviewed_actionable_ids=meeting.get("reviewedActionableEventIds"),
        )
        assert report["unaccountedBlocks"] == 0
        assert report["genericTaskRate"] == 0
        assert report["taskRecall"] >= 0.85
        assert report["groundedPrecisionTasks"] >= 0.9
        assert report["noteRecall"] >= 0.85
        assert report.get("memoryUnaccounted", 0) == 0
        assert report.get("memoryCoverageFailure") is False
        summaries.append(
            {
                "id": meeting["id"],
                "size": meeting["size"],
                "taskRecall": report["taskRecall"],
                "requiredTaskRecall": report["requiredTaskRecall"],
                "groundedPrecisionTasks": report["groundedPrecisionTasks"],
                "noteUsefulnessPrecision": report["noteUsefulnessPrecision"],
                "noteRecall": report["noteRecall"],
                "requiredNoteRecall": report["requiredNoteRecall"],
                "optionalValidFound": report["optionalValidFound"],
                "lowValueSuppressed": report["lowValueSuppressed"],
                "invalidGoldCount": report["invalidGoldCount"],
                "duplicateRate": report["noteDuplicateRate"],
                "mixedThreadRate": report["mixedThreadRate"],
                "genericTaskRate": report["genericTaskRate"],
                "unaccountedBlocks": report["unaccountedBlocks"],
                "goldFailures": report.get("goldFailures") or [],
            }
        )
        if meeting["id"] == "meeting-b":
            print("MEETING_B_GOLD_TRACES", report.get("goldTraces"))
            assert report["requiredTaskRecall"] == 1.0
            assert not report.get("goldFailures")
    assert len(summaries) == 4
    print("REVIEWED_MEETINGS_SCRIPTED", summaries)
