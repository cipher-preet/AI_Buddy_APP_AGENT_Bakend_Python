import asyncio

from services.conversation.event_pipeline.embeddings import CachedEmbedder, LexicalEmbedder
from services.conversation.event_pipeline.events import ScriptedEventExtractor
from services.conversation.event_pipeline.gold_scoring import NOT_MEASURED, pipeline_benchmark
from services.conversation.event_pipeline.pipeline import run_event_pipeline
from tests.fixtures.long_meeting_gold import build_gold_transcript


def test_gold_long_meeting_regression():
    gold = build_gold_transcript()
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
        gold_threads=gold.get("goldThreads"),
        gold_complete=True,
        original_actionable_ids=gold.get("originalActionableEventIds"),
        reviewed_actionable_ids=gold.get("reviewedActionableEventIds"),
    )

    assert result.coverage is not None
    assert result.coverage.unaccounted_blocks == 0
    assert report["genericTaskRate"] == 0
    for title in gold["forbiddenTaskTitles"]:
        assert not any(title.casefold() in f"{task.title} {task.body}".casefold() for task in result.tasks)

    server = next((task for task in result.tasks if "server" in task.title.casefold() and "id" in task.title.casefold()), None)
    assert server is not None
    evidence_sequences = {span.sequenceStart for span in server.evidence} | {span.sequenceEnd for span in server.evidence}
    assert evidence_sequences.isdisjoint(gold["serverIdForbiddenSequences"])
    metadata = server.changes or {}
    assert "threadContextEvents" in metadata
    assert "artifactEvidence" in metadata

    blob = " ".join(f"{item.title} {item.body}" for item in [*result.tasks, *result.notes]).casefold()
    for concept in (
        "meeting page",
        "server id",
        "opencv",
        "microphone",
        "old keys",
        "play store",
        "insecure",
        "s3",
    ):
        assert concept in blob, concept

    s3_events = [event for event in result.events if "S3" in (event.entities or []) or "s3" in event.meaning.casefold()]
    s3_threads = {event.threadId for event in s3_events if event.threadId}
    assert len(s3_threads) == 1

    assert result.notes, "memory events must not collapse to notes=0"
    print("GOLD_LONG_MEETING_BENCHMARK_SCRIPTED", report)

    assert report["extractorMode"] == "scripted"
    assert report["matchedNotes"] == report["expectedNotes"] or report["noteRecall"] >= 0.85
    assert report["validAdditionalNotes"] >= 0
    # Valid extras must not be billed as false positives.
    classified_fp = [row for row in report["noteClassifications"] if row["label"] == "FALSE_POSITIVE"]
    classified_extra = [row for row in report["noteClassifications"] if row["label"] == "VALID_ADDITIONAL"]
    assert report["generatedNotes"] == report["matchedNotes"] + report["validAdditionalNotes"] + report["falsePositiveNotes"] + report["duplicateNotes"] + report["tooVagueNotes"] + report["unsupportedNotes"]
    assert report["generatedTasks"] == report["matchedTasks"] + report["validAdditionalTasks"] + report["falsePositiveTasks"] + report["duplicateTasks"] + report["tooVagueTasks"] + report["unsupportedTasks"]
    if report["generatedNotes"] > report["expectedNotes"]:
        assert classified_extra or report["validAdditionalNotes"] > 0
        assert len(classified_fp) < report["generatedNotes"] - report["expectedNotes"] or report["groundedPrecisionNotes"] >= 0.85
    assert report["groundedPrecisionNotes"] >= 0.85
    assert report["taskRecall"] >= 0.9 if report["taskRecall"] != NOT_MEASURED else True
    if report["evidencePrecision"] not in {None, NOT_MEASURED}:
        assert report["evidencePrecision"] >= 0.95
    assert report["mixedThreadRate"] < 0.05
    assert report["unaccountedBlocks"] == 0
    assert report.get("unaccountedSemanticUnits", 0) == 0
    assert report["genericTaskRate"] == 0
