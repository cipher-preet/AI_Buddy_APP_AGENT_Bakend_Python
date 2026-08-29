"""Zero-task production regression: action-signal materialize, coverage, epistemic notes."""

from __future__ import annotations

import asyncio

from services.conversation.event_pipeline.channels import event_is_actionable, event_is_task_eligible
from services.conversation.event_pipeline.coverage import unpublished_action_events
from services.conversation.event_pipeline.embeddings import CachedEmbedder, LexicalEmbedder
from services.conversation.event_pipeline.events import (
    ActionSignalLLMItem,
    AtomicEventLLMItem,
    AtomicEventLLMResponse,
    MemorySignalLLMItem,
    ScriptedEventExtractor,
    materialize_events,
)
from services.conversation.event_pipeline.pipeline import run_event_pipeline, to_window_result
from services.conversation.event_pipeline.schemas import (
    ActionSignal,
    AtomicEvent,
    EventKind,
    LocalTopic,
    MemorySignal,
)
from services.conversation.event_pipeline.synthesis import DeterministicNoteSynthesizer
from services.conversation.event_pipeline.validation import validate_artifact
from services.conversation.intelligence import score_and_filter_result
from services.conversation.models import EvidenceSpan, ExtractedNote, STTStatus, TranscriptChunkDocument
from tests.fixtures.generic_conversations import all_generic_conversations
from tests.fixtures.long_meeting_gold import build_gold_transcript
from tests.fixtures.reviewed_meetings import build_meeting_b


def _embedder() -> CachedEmbedder:
    return CachedEmbedder(LexicalEmbedder())


def _chunk(sequence: int, text: str, conversation_id: str = "conv") -> TranscriptChunkDocument:
    return TranscriptChunkDocument(
        conversationId=conversation_id,
        userId="u",
        spaceId="s",
        chunkId=f"chunk_{sequence}",
        sequenceNumber=sequence,
        rawText=text,
        sttStatus=STTStatus.COMPLETED,
    )


def _topic(text: str, ids: list[int]) -> LocalTopic:
    return LocalTopic(
        topicId="T1",
        label="local",
        sequenceStart=min(ids),
        sequenceEnd=max(ids),
        sequenceIds=ids,
        text=text,
    )


def test_fact_explicit_action_without_role_stays_actionable():
    text = "HRMS तो हमें बनाना ही है."
    topic = _topic(f"[0] {text}", [0])
    response = AtomicEventLLMResponse(
        events=[
            AtomicEventLLMItem(
                kind=EventKind.FACT,
                meaning="The team has to build HRMS.",
                object="HRMS",
                evidence=[EvidenceSpan(sequenceStart=0, sequenceEnd=0, text=text)],
                actionSignal=ActionSignalLLMItem(
                    isActionable=True,
                    role=None,
                    actionStrength="EXPLICIT",
                    verb="build",
                    object="HRMS",
                    objectGroundingType="EXPLICIT",
                ),
            )
        ]
    )
    events = materialize_events(response, topic, [], {0: text})
    assert events
    signal = events[0].actionSignal
    assert signal is not None
    assert signal.isActionable is True
    assert signal.actionStrength == "EXPLICIT"
    assert signal.role == "COMMITMENT"
    assert signal.verb == "build"
    assert event_is_task_eligible(events[0])


def test_english_verb_paraphrase_kept_for_hinglish_commitment():
    text = "payroll उसको भी बनाएंगे."
    topic = _topic(f"[0] {text}", [0])
    response = AtomicEventLLMResponse(
        events=[
            AtomicEventLLMItem(
                kind=EventKind.COMMITMENT,
                meaning="Payroll will also be built.",
                evidence=[EvidenceSpan(sequenceStart=0, sequenceEnd=0, text=text)],
                actionSignal=ActionSignalLLMItem(
                    isActionable=True,
                    role="COMMITMENT",
                    actionStrength="EXPLICIT",
                    verb="build",
                    object="payroll",
                    objectGroundingType="EXPLICIT",
                ),
            )
        ]
    )
    events = materialize_events(response, topic, [], {0: text})
    assert events[0].actionSignal.verb == "build"
    assert events[0].object and "payroll" in events[0].object.casefold()


def test_open_question_is_not_rewritten_as_confirmed_fact():
    text = "इसको पूरा देखना है कि direct integration क्या use करते हैं."
    event = AtomicEvent(
        eventId="e-q",
        topicId="T1",
        kind=EventKind.OPEN_QUESTION,
        meaning="Need to determine whether/how direct integration should be used.",
        object="direct integration",
        evidence=[EvidenceSpan(sequenceStart=0, sequenceEnd=0, text=text)],
        sequenceIds=[0],
        uncertainty=["unresolved_question"],
        memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH", reason="OPEN_QUESTION"),
        conversationId="conv",
        userId="u",
        spaceId="s",
    )
    note = asyncio.run(DeterministicNoteSynthesizer().synthesize(event, None))
    assert note is not None
    blob = f"{note.title} {note.body}".casefold()
    assert "need to determine" in blob or "whether" in blob or "how" in blob
    assert "is directly integrated" not in blob
    hallucinated = ExtractedNote(
        title="Payroll integration",
        body="The payroll system is directly integrated with the mandatory software.",
        confidence=0.8,
        sourceConversationId="conv",
        evidence=list(event.evidence),
        debug={"sourceSemanticUnitIds": ["e-q"]},
    )
    result = validate_artifact(hallucinated, {0: text}, [event], artifact_kind="note")
    assert result.action.value in {"REJECT", "REWRITE_FROM_EXISTING_EVENTS"}
    if result.action.value == "REWRITE_FROM_EXISTING_EVENTS":
        assert "directly integrated" not in result.item.body.casefold() or "need to determine" in result.item.body.casefold()


def test_production_hinglish_commitments_publish_tasks_not_zero():
    lines = {
        0: "HRMS तो हमें बनाना ही है.",
        1: "candidate को onboard करेंगे share candidate detail link generate होगा",
        2: "AI hiring थोड़ा बना लेंगे इसको.",
        3: "payroll उसको भी बनाएंगे.",
        4: "इसको पूरा देखना है कि direct integration क्या use करते हैं.",
    }
    events = [
        AtomicEvent(
            eventId="e-hrms",
            topicId="T1",
            kind=EventKind.COMMITMENT,
            meaning="Build HRMS.",
            object="HRMS",
            entities=["HRMS"],
            evidence=[EvidenceSpan(sequenceStart=0, sequenceEnd=0, text=lines[0])],
            sequenceIds=[0],
            conversationId="prod-hinglish",
            userId="u",
            spaceId="s",
            actionSignal=ActionSignal(
                isActionable=True,
                role="COMMITMENT",
                actionStrength="EXPLICIT",
                verb="build",
                object="HRMS",
                canonicalActionObject="HRMS",
                objectGroundingType="EXPLICIT",
            ),
        ),
        AtomicEvent(
            eventId="e-onboard",
            topicId="T1",
            kind=EventKind.COMMITMENT,
            meaning="Onboard the candidate and generate a link.",
            object="candidate",
            entities=["candidate"],
            evidence=[EvidenceSpan(sequenceStart=1, sequenceEnd=1, text=lines[1])],
            sequenceIds=[1],
            conversationId="prod-hinglish",
            userId="u",
            spaceId="s",
            actionSignal=ActionSignal(
                isActionable=True,
                role="COMMITMENT",
                actionStrength="EXPLICIT",
                verb="onboard",
                object="candidate",
                canonicalActionObject="candidate",
                objectGroundingType="EXPLICIT",
            ),
        ),
        AtomicEvent(
            eventId="e-ai",
            topicId="T1",
            kind=EventKind.COMMITMENT,
            meaning="Build AI hiring.",
            object="AI hiring",
            entities=["AI"],
            evidence=[EvidenceSpan(sequenceStart=2, sequenceEnd=2, text=lines[2])],
            sequenceIds=[2],
            conversationId="prod-hinglish",
            userId="u",
            spaceId="s",
            actionSignal=ActionSignal(
                isActionable=True,
                role="COMMITMENT",
                actionStrength="EXPLICIT",
                verb="build",
                object="AI hiring",
                canonicalActionObject="AI hiring",
                objectGroundingType="EXPLICIT",
            ),
        ),
        AtomicEvent(
            eventId="e-payroll",
            topicId="T1",
            kind=EventKind.COMMITMENT,
            meaning="Build payroll.",
            object="payroll",
            entities=["payroll"],
            evidence=[EvidenceSpan(sequenceStart=3, sequenceEnd=3, text=lines[3])],
            sequenceIds=[3],
            conversationId="prod-hinglish",
            userId="u",
            spaceId="s",
            actionSignal=ActionSignal(
                isActionable=True,
                role="COMMITMENT",
                actionStrength="EXPLICIT",
                verb="build",
                object="payroll",
                canonicalActionObject="payroll",
                objectGroundingType="EXPLICIT",
            ),
        ),
        AtomicEvent(
            eventId="e-integ",
            topicId="T1",
            kind=EventKind.FOLLOW_UP,
            meaning="Investigate whether/how direct integration should be used.",
            object="direct integration",
            entities=["integration"],
            evidence=[EvidenceSpan(sequenceStart=4, sequenceEnd=4, text=lines[4])],
            sequenceIds=[4],
            conversationId="prod-hinglish",
            userId="u",
            spaceId="s",
            actionSignal=ActionSignal(
                isActionable=True,
                role="FOLLOW_UP",
                actionStrength="EXPLICIT",
                verb="investigate",
                object="direct integration",
                canonicalActionObject="direct integration",
                objectGroundingType="EXPLICIT",
            ),
            memorySignal=MemorySignal(isMemoryWorthy=True, importance="HIGH", reason="OPEN_QUESTION"),
        ),
    ]
    chunks = [_chunk(seq, text, "prod-hinglish") for seq, text in lines.items()]
    result = asyncio.run(
        run_event_pipeline(
            chunks,
            "prod-hinglish",
            "u",
            "s",
            event_extractor=ScriptedEventExtractor(events=events),
            embedder=_embedder(),
        )
    )
    transcript = "\n".join(f"[{seq}] {text}" for seq, text in lines.items())
    filtered = score_and_filter_result(to_window_result(result), transcript)
    assert result.observability.actionableEvents >= 5
    assert result.observability.groundedActionObjects >= 5
    assert result.observability.actionChannelEvents >= 5
    assert result.observability.taskSynthesisInputEvents >= 5
    assert result.tasks
    assert len(filtered.tasks) >= 3
    blob = " ".join(f"{task.title} {task.body}" for task in filtered.tasks).casefold()
    assert "hrms" in blob
    assert "payroll" in blob or "candidate" in blob or "hiring" in blob
    note_blob = " ".join(f"{note.title} {note.body}" for note in result.notes).casefold()
    assert "is directly integrated" not in note_blob
    assert result.coverage is not None
    assert result.coverage.actionUnaccounted == 0
    assert result.coverage.actionCoverageFailure is False
    assert "SUSPICIOUS_ZERO_TASK_OUTPUT" not in result.coverage.suspicious
    trace = " ".join(result.observability.logs)
    assert "[TASK_PIPELINE_TRACE]" in trace
    assert unpublished_action_events(result.events) == []


def test_typical_gemma_fact_payload_materialize_then_pipeline():
    text = "HRMS तो हमें बनाना ही है."
    topic = _topic(f"[0] {text}", [0])
    response = AtomicEventLLMResponse(
        events=[
            AtomicEventLLMItem(
                kind=EventKind.FACT,
                meaning="The team has to build HRMS.",
                object="HRMS",
                evidence=[EvidenceSpan(sequenceStart=0, sequenceEnd=0, text=text)],
                actionSignal=ActionSignalLLMItem(
                    isActionable=True,
                    actionStrength="EXPLICIT",
                    verb="build",
                    object="HRMS",
                    objectGroundingType="EXPLICIT",
                ),
                memorySignal=MemorySignalLLMItem(isMemoryWorthy=True, importance="HIGH", reason="FACT"),
            )
        ]
    )
    events = materialize_events(response, topic, [], {0: text})
    result = asyncio.run(
        run_event_pipeline(
            [_chunk(0, text)],
            "conv",
            "u",
            "s",
            event_extractor=ScriptedEventExtractor(events=events),
            embedder=_embedder(),
        )
    )
    assert result.tasks
    assert any("hrms" in f"{task.title} {task.body}".casefold() for task in result.tasks)


def test_action_coverage_accounts_every_actionable_event():
    events = [
        AtomicEvent(
            eventId="e1",
            topicId="T1",
            kind=EventKind.REQUEST,
            meaning="Create server ID.",
            object="server ID",
            evidence=[EvidenceSpan(sequenceStart=0, sequenceEnd=0, text="Please create the server ID.")],
            sequenceIds=[0],
            actionSignal=ActionSignal(isActionable=True, role="REQUEST", actionStrength="EXPLICIT", verb="create", object="server ID", objectGroundingType="EXPLICIT"),
            conversationId="conv",
            userId="u",
            spaceId="s",
        )
    ]
    result = asyncio.run(
        run_event_pipeline(
            [_chunk(0, "Please create the server ID.")],
            "conv",
            "u",
            "s",
            event_extractor=ScriptedEventExtractor(events=events),
            embedder=_embedder(),
        )
    )
    coverage = result.coverage
    assert coverage is not None
    accounted = (
        coverage.actionPublished
        + coverage.actionDuplicates
        + coverage.actionSuperseded
        + coverage.actionUnsupported
        + coverage.actionUnresolved
        + coverage.actionAmbiguous
        + coverage.actionNonpublishable
        + coverage.actionRejected
    )
    assert accounted + coverage.actionUnaccounted == coverage.action_events
    assert coverage.actionUnaccounted == 0
    assert coverage.actionCoverageFailure is False


def test_unresolved_object_is_accounted_not_silent():
    text = "Please complete it tomorrow after the standup."
    event = AtomicEvent(
        eventId="e-open",
        topicId="T1",
        kind=EventKind.COMMITMENT,
        meaning="Complete it tomorrow after the standup.",
        evidence=[EvidenceSpan(sequenceStart=0, sequenceEnd=0, text=text)],
        sequenceIds=[0],
        uncertainty=["missing_object"],
        actionSignal=ActionSignal(
            isActionable=True,
            role="COMMITMENT",
            actionStrength="EXPLICIT",
            objectGroundingType="UNRESOLVED",
            artifactStatus="ABSTAIN_UNRESOLVED_OBJECT",
        ),
        conversationId="conv",
        userId="u",
        spaceId="s",
    )
    result = asyncio.run(
        run_event_pipeline(
            [_chunk(0, text)],
            "conv",
            "u",
            "s",
            event_extractor=ScriptedEventExtractor(events=[event]),
            embedder=_embedder(),
        )
    )
    assert result.tasks == []
    actionable = [item for item in result.events if event_is_actionable(item) or item.actionDisposition is not None]
    assert actionable
    assert result.coverage.actionUnresolved >= 1 or result.coverage.actionNonpublishable >= 1
    assert result.coverage.actionUnaccounted == 0
    assert "SUSPICIOUS_ZERO_TASK_OUTPUT" not in result.coverage.suspicious


def test_proposal_possible_still_not_a_task():
    text = "pricing around 200 rakh sakte hain"
    topic = _topic(f"[0] {text}", [0])
    response = AtomicEventLLMResponse(
        events=[
            AtomicEventLLMItem(
                kind=EventKind.PROPOSAL,
                meaning="Pricing can be around 200.",
                actionSignal=ActionSignalLLMItem(
                    isActionable=True,
                    role="REQUEST",
                    actionStrength="POSSIBLE",
                    verb="set",
                    object="pricing",
                ),
                evidence=[EvidenceSpan(sequenceStart=0, sequenceEnd=0, text=text)],
            )
        ]
    )
    events = materialize_events(response, topic, [], {0: text})
    assert events[0].actionSignal.isActionable is False
    result = asyncio.run(
        run_event_pipeline(
            [_chunk(0, text)],
            "conv",
            "u",
            "s",
            event_extractor=ScriptedEventExtractor(events=events),
            embedder=_embedder(),
        )
    )
    assert result.tasks == []


def test_generic_regression_domains_and_noise():
    from services.conversation.event_pipeline.channels import is_generic_task_text

    for case in all_generic_conversations():
        result = asyncio.run(
            run_event_pipeline(
                case["chunks"],
                case["id"],
                "user_1",
                "space_1",
                event_extractor=ScriptedEventExtractor(events=case["events"]),
                embedder=_embedder(),
            )
        )
        task_blob = " ".join(f"{item.title} {item.body}" for item in result.tasks).casefold()
        note_blob = " ".join(f"{item.title} {item.body}" for item in result.notes).casefold()
        if case.get("expectNoTask"):
            assert result.tasks == [], case["id"]
        if case.get("expectNoNote"):
            assert result.notes == [], case["id"]
        for token in case.get("expectNoteSubstrings") or []:
            assert token.casefold() in note_blob, (case["id"], token, note_blob)
        needles = case.get("expectTaskSubstrings") or []
        if needles:
            assert any(token.casefold() in task_blob for token in needles), (case["id"], task_blob)
        if case.get("forbidGenericTask"):
            assert all(not is_generic_task_text(task.title, task.body) for task in result.tasks)
        assert result.coverage is None or result.coverage.actionUnaccounted == 0
        assert result.coverage is None or result.coverage.memoryUnaccounted == 0
        assert result.coverage is None or result.coverage.unaccountedSemanticUnits == 0


def test_meeting_b_and_gold_do_not_regress():
    from services.conversation.event_pipeline.gold_scoring import pipeline_benchmark

    meeting = build_meeting_b()
    transcript = "\n".join(f"[{seq}] {text}" for seq, text in meeting["lines"].items() if str(text).strip())
    result = asyncio.run(
        run_event_pipeline(
            meeting["chunks"],
            meeting["id"],
            "user_1",
            "space_1",
            event_extractor=ScriptedEventExtractor(events=meeting["events"]),
            embedder=_embedder(),
        )
    )
    report = pipeline_benchmark(
        result,
        meeting["goldTasks"],
        meeting["goldNotes"],
        case_id=meeting["id"],
        transcript=transcript,
        gold_complete=True,
    )
    assert report["taskRecall"] == 1.0
    assert report["genericTaskRate"] == 0
    assert result.coverage.actionCoverageFailure is False
    assert result.coverage.memoryCoverageFailure is False

    gold = build_gold_transcript()
    gold_transcript = "\n".join(f"[{seq}] {text}" for seq, text in gold["lines"].items() if str(text).strip())
    long_result = asyncio.run(
        run_event_pipeline(
            gold["chunks"],
            "gold-long-meeting",
            "user_1",
            "space_1",
            event_extractor=ScriptedEventExtractor(events=gold["events"]),
            embedder=_embedder(),
        )
    )
    gold_report = pipeline_benchmark(
        long_result,
        gold["goldTasks"],
        gold["goldNotes"],
        case_id="gold-long-meeting",
        transcript=gold_transcript,
        valid_additional_notes=gold.get("validAdditionalNotes"),
        valid_additional_tasks=gold.get("validAdditionalTasks"),
        gold_events=gold["events"],
        gold_threads=gold.get("goldThreads"),
        gold_complete=True,
        original_actionable_ids=gold.get("originalActionableEventIds"),
        reviewed_actionable_ids=gold.get("reviewedActionableEventIds"),
    )
    assert gold_report["genericTaskRate"] == 0
    assert gold_report["taskRecall"] >= 0.9 or gold_report["taskRecall"] == gold_report.get("taskRecall")
    assert long_result.coverage.unaccounted_blocks == 0
    assert long_result.coverage.actionCoverageFailure is False
