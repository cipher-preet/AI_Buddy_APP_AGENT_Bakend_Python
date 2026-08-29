import asyncio

from services.conversation.event_pipeline.embeddings import CachedEmbedder, LexicalEmbedder
from services.conversation.event_pipeline.events import ScriptedEventExtractor
from services.conversation.event_pipeline.gold_scoring import NOT_MEASURED, e2e_scale_report
from services.conversation.event_pipeline.pipeline import run_event_pipeline
from services.conversation.event_pipeline.schemas import AtomicEvent, EventKind
from services.conversation.models import EvidenceSpan
from tests.fixtures.long_meeting_gold import build_gold_transcript
from tests.fixtures.scale_meetings import build_scale_meeting


def _extractor_for(meeting: dict) -> ScriptedEventExtractor:
    if meeting.get("events"):
        return ScriptedEventExtractor(events=meeting["events"])
    events = []
    for chunk in meeting["chunks"]:
        text = (chunk.rawText or "").strip()
        if not text or text.startswith("haan") or text.startswith("ok") or "filler" in text:
            continue
        if any(token in text for token in ("S3", "server ID", "Pricing", "Play Store", "microphone", "OpenCV", "meeting page")):
            kind = EventKind.REQUEST if "Please" in text or "create" in text.casefold() else EventKind.ISSUE
            events.append(
                AtomicEvent(
                    eventId=f"e-{chunk.sequenceNumber}",
                    topicId="T",
                    kind=kind,
                    meaning=text,
                    object=text[:40],
                    evidence=[EvidenceSpan(sequenceStart=chunk.sequenceNumber, sequenceEnd=chunk.sequenceNumber, text=text)],
                    sequenceIds=[chunk.sequenceNumber],
                    conversationId=meeting["id"],
                    userId="u",
                    spaceId="s",
                )
            )
    return ScriptedEventExtractor(events=events)


def _run(count: int) -> dict:
    meeting = build_scale_meeting(count)
    result = asyncio.run(
        run_event_pipeline(
            meeting["chunks"],
            meeting["id"],
            "u",
            "s",
            event_extractor=_extractor_for(meeting),
            embedder=CachedEmbedder(LexicalEmbedder()),
        )
    )
    transcript = ""
    if meeting.get("goldComplete"):
        gold = build_gold_transcript()
        transcript = "\n".join(f"[{seq}] {text}" for seq, text in gold["lines"].items() if str(text).strip())
        meeting["transcript"] = transcript
    report = e2e_scale_report(result, gold=meeting, case_id=meeting["id"])
    report["size"] = count
    return report


def test_e2e_scale_reports_do_not_invent_metrics():
    reports = {
        50: _run(50),
        150: _run(150),
        300: _run(300),
        500: _run(500),
    }
    print("E2E_SCALE_BENCHMARK_SCRIPTED", reports)
    for count, report in reports.items():
        assert report["counts"]["rawChunks"] >= count * 0.8 or report["counts"]["rawChunks"] == count
        assert "microBlocks" in report["counts"]
        assert "topics" in report["counts"]
        assert "events" in report["counts"]
        assert "threads" in report["counts"]
        assert "tasks" in report["counts"]
        assert "notes" in report["counts"]
        if not report.get("goldComplete"):
            assert report["taskPrecision"] == NOT_MEASURED
            assert report["notePrecision"] == NOT_MEASURED
            assert report["eventRecall"] == NOT_MEASURED
            assert report["evidencePrecision"] == NOT_MEASURED
            assert report["threadPrecision"] == NOT_MEASURED
        assert report["unaccountedBlocks"] == 0 or report["unaccountedBlocks"] == NOT_MEASURED
        assert report["genericTaskRate"] == 0 or isinstance(report["genericTaskRate"], float)
    assert reports[50]["counts"]["rawChunks"] < reports[500]["counts"]["rawChunks"]
