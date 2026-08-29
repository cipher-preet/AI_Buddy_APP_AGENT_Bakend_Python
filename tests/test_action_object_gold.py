"""Scripted/unit action-object extraction, grounding, and eligibility."""

from __future__ import annotations

import asyncio

from services.conversation.event_pipeline.channels import (
    event_is_actionable,
    is_generic_task_text,
    split_action_and_memory,
    unresolved_action_object,
)
from services.conversation.event_pipeline.embeddings import CachedEmbedder, LexicalEmbedder
from services.conversation.event_pipeline.events import (
    ActionSignalLLMItem,
    AtomicEventLLMItem,
    AtomicEventLLMResponse,
    FieldEvidenceLLMItem,
    MemorySignalLLMItem,
    ScriptedEventExtractor,
    materialize_events,
)
from services.conversation.event_pipeline.gold_scoring import score_action_signals
from services.conversation.event_pipeline.pipeline import run_event_pipeline
from services.conversation.event_pipeline.schemas import (
    ABSTAIN_UNRESOLVED_OBJECT,
    ActionSignal,
    AtomicEvent,
    EventKind,
    LocalTopic,
    MemorySignal,
    MicroBlock,
)
from services.conversation.event_pipeline.topics import segment_local_topics
from services.conversation.models import EvidenceSpan, STTStatus, TranscriptChunkDocument
from tests.fixtures.action_object_gold import CASES


def _embedder() -> CachedEmbedder:
    return CachedEmbedder(LexicalEmbedder())


def _chunk(sequence: int, text: str) -> TranscriptChunkDocument:
    return TranscriptChunkDocument(
        conversationId="action-object",
        userId="u",
        spaceId="s",
        chunkId=f"chunk_{sequence}",
        sequenceNumber=sequence,
        rawText=text,
        sttStatus=STTStatus.COMPLETED,
    )


def _topic(text: str, sequence: int = 0, sequence_ids: list[int] | None = None) -> LocalTopic:
    ids = sequence_ids or [sequence]
    return LocalTopic(
        topicId="T1",
        label="local",
        sequenceStart=min(ids),
        sequenceEnd=max(ids),
        sequenceIds=ids,
        text=text,
    )


def _scripted_event(case: dict, sequence: int) -> AtomicEvent:
    signal = ActionSignal(
        isActionable=bool(case["actionable"]),
        role="INSTRUCTION" if case["kind"] == EventKind.REQUIREMENT else str(case["kind"].value),
        actionStrength="EXPLICIT" if case["actionable"] else "NONE",
        verb=case.get("verb"),
        object=case.get("object"),
        objectGroundingType="UNRESOLVED" if case.get("unresolvedObject") else case.get("objectGroundingType") or ("EXPLICIT" if case.get("object") else None),
        actor=case.get("actor"),
        deadline=case.get("deadline"),
        artifactStatus=ABSTAIN_UNRESOLVED_OBJECT if case.get("unresolvedObject") else None,
    )
    memory = MemorySignal(isMemoryWorthy=not case["actionable"] or case["kind"] in {EventKind.REQUIREMENT, EventKind.ISSUE, EventKind.STATE})
    uncertainty = ["missing_object"] if case.get("unresolvedObject") else []
    return AtomicEvent(
        eventId=case["id"],
        topicId="T1",
        kind=case["kind"],
        meaning=case["text"],
        object=None if case.get("unresolvedObject") else case.get("object"),
        actor=case.get("actor"),
        timeExpression=case.get("deadline"),
        uncertainty=uncertainty,
        evidence=[EvidenceSpan(sequenceStart=sequence, sequenceEnd=sequence, text=case["text"])],
        sequenceIds=[sequence],
        conversationId="action-object",
        userId="u",
        spaceId="s",
        actionSignal=signal,
        memorySignal=memory,
    )


def test_local_topics_do_not_merge_related_technical_objects():
    chunks = [
        _chunk(0, "create server ID"),
        _chunk(1, "database connection string"),
        _chunk(2, "PNB mobile setup"),
        _chunk(3, "network parameter issue"),
        _chunk(4, "Port ID tracking"),
    ]
    blocks = [
        MicroBlock(
            microBlockId=f"MB{index}",
            sequenceStart=index,
            sequenceEnd=index,
            sequenceIds=[index],
            sourceIds=[f"chunk_{index}"],
            text=chunk.rawText,
            tokenCount=8,
        )
        for index, chunk in enumerate(chunks)
    ]
    topics = asyncio.run(segment_local_topics(blocks, _embedder()))
    assert len(topics) >= 4
    labels = " ".join(topic.text.casefold() for topic in topics)
    assert "server" in labels and "pnb" in labels
    grouped = [set(topic.sequenceIds) for topic in topics]
    assert not any({0, 1, 2, 3, 4} <= group for group in grouped)
    assert all(getattr(topic, "coherence", 0) >= 0 for topic in topics)
    assert all(topic.boundaryReason for topic in topics)


def test_requirement_instruction_is_task_eligible():
    event = AtomicEvent(
        eventId="e-gpt",
        topicId="T1",
        kind=EventKind.REQUIREMENT,
        meaning="GPT should be used for coordinate processing.",
        object="GPT for coordinate processing",
        evidence=[EvidenceSpan(sequenceStart=0, sequenceEnd=0, text="Use GPT for coordinate action")],
        sequenceIds=[0],
        conversationId="conv",
        userId="u",
        spaceId="s",
        actionSignal=ActionSignal(isActionable=True, role="INSTRUCTION", actionStrength="EXPLICIT", verb="use", object="GPT for coordinate processing"),
        memorySignal=MemorySignal(isMemoryWorthy=True),
    )
    actions, memory, _ = split_action_and_memory([event])
    assert actions
    assert memory
    result = asyncio.run(
        run_event_pipeline(
            [_chunk(0, "Use GPT for coordinate action")],
            "conv",
            "u",
            "s",
            event_extractor=ScriptedEventExtractor(events=[event]),
            embedder=_embedder(),
        )
    )
    assert result.tasks
    assert any("gpt" in f"{task.title} {task.body}".casefold() for task in result.tasks)


def test_issue_without_action_is_not_task_eligible():
    event = AtomicEvent(
        eventId="e-s3",
        topicId="T1",
        kind=EventKind.ISSUE,
        meaning="S3 is not reaching the frontend.",
        object="S3 frontend",
        evidence=[EvidenceSpan(sequenceStart=0, sequenceEnd=0, text="S3 front end pe nahi aa raha")],
        sequenceIds=[0],
        conversationId="conv",
        userId="u",
        spaceId="s",
        actionSignal=ActionSignal(isActionable=False),
        memorySignal=MemorySignal(isMemoryWorthy=True),
    )
    actions, memory, _ = split_action_and_memory([event])
    assert actions == []
    assert memory
    result = asyncio.run(
        run_event_pipeline(
            [_chunk(0, "S3 front end pe nahi aa raha")],
            "conv",
            "u",
            "s",
            event_extractor=ScriptedEventExtractor(events=[event]),
            embedder=_embedder(),
        )
    )
    assert result.tasks == []
    assert result.notes


def test_field_grounding_rejects_unsupported_action_object():
    topic = _topic("[0] AWS credits are low", 0)
    response = AtomicEventLLMResponse(
        events=[
            AtomicEventLLMItem(
                kind=EventKind.REQUEST,
                meaning="Fix AWS administrator access.",
                object="AWS administrator access",
                actionSignal=ActionSignalLLMItem(
                    isActionable=True,
                    role="REQUEST",
                    actionStrength="EXPLICIT",
                    verb="fix",
                    object="AWS administrator access",
                ),
                fieldEvidence=FieldEvidenceLLMItem(
                    actionObject=[EvidenceSpan(sequenceStart=0, sequenceEnd=0, text="AWS credits are low")],
                ),
                evidence=[EvidenceSpan(sequenceStart=0, sequenceEnd=0, text="AWS credits are low")],
            )
        ]
    )
    events = materialize_events(response, topic, [], {0: "AWS credits are low"})
    assert events
    event = events[0]
    assert event.actionSignal
    assert event.actionSignal.isActionable is True
    assert event.object is None
    assert event.actionSignal.artifactStatus == ABSTAIN_UNRESOLVED_OBJECT
    assert unresolved_action_object(event)


def test_grounded_create_server_id_keeps_verb_and_object():
    topic = _topic("[0] server ID create karna hai", 0)
    response = AtomicEventLLMResponse(
        events=[
            AtomicEventLLMItem(
                kind=EventKind.REQUIREMENT,
                meaning="Create server ID.",
                actionSignal=ActionSignalLLMItem(isActionable=True, role="INSTRUCTION", actionStrength="EXPLICIT", verb="create", object="server ID"),
                fieldEvidence=FieldEvidenceLLMItem(
                    actionVerb=[EvidenceSpan(sequenceStart=0, sequenceEnd=0, text="server ID create karna hai")],
                    actionObject=[EvidenceSpan(sequenceStart=0, sequenceEnd=0, text="server ID create karna hai")],
                ),
                evidence=[EvidenceSpan(sequenceStart=0, sequenceEnd=0, text="server ID create karna hai")],
            )
        ]
    )
    events = materialize_events(response, topic, [], {0: "server ID create karna hai"})
    assert events[0].actionSignal.isActionable
    assert events[0].actionSignal.verb == "create"
    assert events[0].object and "server" in events[0].object.casefold()
    assert not unresolved_action_object(events[0])


def test_pronoun_resolves_only_inside_same_local_topic():
    topic = LocalTopic(
        topicId="T1",
        label="S3",
        sequenceStart=0,
        sequenceEnd=1,
        sequenceIds=[0, 1],
        text="[0] S3 frontend nahi aa raha\n[1] kal usko kar denge",
        microBlockIds=["MB1"],
    )
    blocks = [
        MicroBlock(microBlockId="MB1", sequenceStart=0, sequenceEnd=1, sequenceIds=[0, 1], text=topic.text, tokenCount=20),
    ]
    response = AtomicEventLLMResponse(
        events=[
            AtomicEventLLMItem(
                kind=EventKind.COMMITMENT,
                meaning="The S3 issue will be done tomorrow.",
                actionSignal=ActionSignalLLMItem(isActionable=True, role="COMMITMENT", actionStrength="EXPLICIT", verb="fix", object="usko", deadline="kal"),
                evidence=[EvidenceSpan(sequenceStart=1, sequenceEnd=1, text="kal usko kar denge")],
            )
        ]
    )
    events = materialize_events(response, topic, blocks, {0: "S3 frontend nahi aa raha", 1: "kal usko kar denge"})
    assert events
    assert events[0].object
    assert "s3" in events[0].object.casefold() or "frontend" in events[0].object.casefold()


def test_unresolved_pronoun_without_local_object_abstains():
    topic = _topic("[0] kal kar denge", 0)
    response = AtomicEventLLMResponse(
        events=[
            AtomicEventLLMItem(
                kind=EventKind.COMMITMENT,
                meaning="It will be done tomorrow.",
                actionSignal=ActionSignalLLMItem(isActionable=True, role="COMMITMENT", actionStrength="EXPLICIT", object=None, deadline="kal"),
                evidence=[EvidenceSpan(sequenceStart=0, sequenceEnd=0, text="kal kar denge")],
            )
        ]
    )
    events = materialize_events(response, topic, [], {0: "kal kar denge"})
    assert events
    assert events[0].actionSignal.isActionable
    assert events[0].object is None
    assert events[0].actionSignal.artifactStatus == ABSTAIN_UNRESOLVED_OBJECT
    result = asyncio.run(
        run_event_pipeline(
            [_chunk(0, "kal kar denge")],
            "conv",
            "u",
            "s",
            event_extractor=ScriptedEventExtractor(events=events),
            embedder=_embedder(),
        )
    )
    assert result.tasks == []
    assert not any(is_generic_task_text(task.title, task.body) for task in result.tasks)


def test_gpt_and_opencv_do_not_merge_into_one_event():
    topic = _topic(
        "[0] Use GPT for coordinate action. Latest OpenCV extraction should be used for room dimensions and coordinates.",
        0,
    )
    response = AtomicEventLLMResponse(
        events=[
            AtomicEventLLMItem(
                kind=EventKind.REQUIREMENT,
                meaning="GPT should be used for coordinate processing.",
                actionSignal=ActionSignalLLMItem(isActionable=True, role="INSTRUCTION", actionStrength="EXPLICIT", verb="use", object="GPT"),
                evidence=[EvidenceSpan(sequenceStart=0, sequenceEnd=0, text="Use GPT for coordinate action")],
            ),
            AtomicEventLLMItem(
                kind=EventKind.REQUIREMENT,
                meaning="OpenCV should be used for room dimensions and coordinates.",
                actionSignal=ActionSignalLLMItem(isActionable=True, role="INSTRUCTION", actionStrength="EXPLICIT", verb="use", object="OpenCV"),
                evidence=[EvidenceSpan(sequenceStart=0, sequenceEnd=0, text="Latest OpenCV extraction should be used for room dimensions and coordinates.")],
            ),
        ]
    )
    events = materialize_events(
        response,
        topic,
        [],
        {0: "Use GPT for coordinate action. Latest OpenCV extraction should be used for room dimensions and coordinates."},
    )
    assert len(events) == 2
    objects = " ".join(event.object or "" for event in events).casefold()
    assert "gpt" in objects and "opencv" in objects


def test_action_object_gold_scripted_metrics():
    chunks = []
    events = []
    gold = []
    for index, case in enumerate(CASES):
        chunks.append(_chunk(index, case["text"]))
        event = _scripted_event(case, index)
        events.append(event)
        gold.append(event)
    result = asyncio.run(
        run_event_pipeline(
            chunks,
            "action-object",
            "u",
            "s",
            event_extractor=ScriptedEventExtractor(events=events),
            embedder=_embedder(),
        )
    )
    metrics = score_action_signals(result.events, gold)
    print("ACTION_OBJECT_GOLD_SCRIPTED", metrics)
    print("ACTION_OBJECT_FAILURES", metrics.get("actionObjectFailures"))
    assert metrics["actionabilityPrecision"] >= 0.9
    assert metrics["actionabilityRecall"] >= 0.9
    assert metrics["actionVerbPrecision"] >= 0.8 or metrics["actionVerbPrecision"] == "not measured"
    assert metrics["actionObjectPrecision"] >= 0.8 or metrics["actionObjectPrecision"] == "not measured"
    assert metrics["genericActionRate"] == 0
    assert not any(is_generic_task_text(task.title, task.body) for task in result.tasks)
    for case, sequence in zip(CASES, range(len(CASES))):
        if case.get("mustNotCreateTask"):
            related = [task for task in result.tasks if any(span.sequenceStart == sequence for span in task.evidence)]
            assert related == []
        if case["actionable"] and case.get("object") and not case.get("unresolvedObject"):
            related_events = [event for event in result.events if sequence in event.sequenceIds]
            assert any(event_is_actionable(event) and event.object for event in related_events)


def test_inferred_broad_object_is_rejected():
    topic = _topic("[0] server ID create karna hai", 0)
    response = AtomicEventLLMResponse(
        events=[
            AtomicEventLLMItem(
                kind=EventKind.REQUEST,
                meaning="Create server infrastructure configuration.",
                actionSignal=ActionSignalLLMItem(
                    isActionable=True,
                    role="REQUEST",
                    actionStrength="EXPLICIT",
                    verb="create",
                    object="server infrastructure configuration",
                    objectGroundingType="INFERRED",
                ),
                evidence=[EvidenceSpan(sequenceStart=0, sequenceEnd=0, text="server ID create karna hai")],
            )
        ]
    )
    events = materialize_events(response, topic, [], {0: "server ID create karna hai"})
    assert events
    assert events[0].object is None
    assert events[0].actionSignal.objectGroundingType == "INFERRED"
    assert unresolved_action_object(events[0])


def test_proposal_possible_does_not_publish_task():
    topic = _topic("[0] pricing around 200 rakh sakte hain", 0)
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
                evidence=[EvidenceSpan(sequenceStart=0, sequenceEnd=0, text="pricing around 200 rakh sakte hain")],
            )
        ]
    )
    events = materialize_events(response, topic, [], {0: "pricing around 200 rakh sakte hain"})
    assert events
    assert events[0].actionSignal.isActionable is False
    assert events[0].actionSignal.actionStrength == "POSSIBLE"
    result = asyncio.run(
        run_event_pipeline(
            [_chunk(0, "pricing around 200 rakh sakte hain")],
            "conv",
            "u",
            "s",
            event_extractor=ScriptedEventExtractor(events=events),
            embedder=_embedder(),
        )
    )
    assert result.tasks == []


def test_explicit_finalize_pricing_is_task_eligible():
    event = _scripted_event(
        {
            "id": "pricing-final",
            "text": "pricing kal final kar lena",
            "actionable": True,
            "verb": "finalize",
            "object": "pricing",
            "actor": None,
            "deadline": "kal",
            "kind": EventKind.REQUEST,
        },
        0,
    )
    result = asyncio.run(
        run_event_pipeline(
            [_chunk(0, "pricing kal final kar lena")],
            "conv",
            "u",
            "s",
            event_extractor=ScriptedEventExtractor(events=[event]),
            embedder=_embedder(),
        )
    )
    assert result.tasks
    assert any("pric" in f"{task.title} {task.body}".casefold() for task in result.tasks)


def test_ambiguous_local_coreference_abstains():
    topic = LocalTopic(
        topicId="T1",
        label="mixed",
        sequenceStart=0,
        sequenceEnd=2,
        sequenceIds=[0, 1, 2],
        text="[0] server ID pending\n[1] connection failing\n[2] kal isko fix kar denge",
        microBlockIds=["MB1", "MB2"],
    )
    blocks = [
        MicroBlock(microBlockId="MB1", sequenceStart=0, sequenceEnd=1, sequenceIds=[0, 1], text="server ID pending\nconnection failing", tokenCount=16),
        MicroBlock(microBlockId="MB2", sequenceStart=2, sequenceEnd=2, sequenceIds=[2], text="kal isko fix kar denge", tokenCount=8),
    ]
    response = AtomicEventLLMResponse(
        events=[
            AtomicEventLLMItem(
                kind=EventKind.COMMITMENT,
                meaning="It will be fixed tomorrow.",
                actionSignal=ActionSignalLLMItem(isActionable=True, role="COMMITMENT", actionStrength="EXPLICIT", verb="fix", object="isko", deadline="kal"),
                evidence=[EvidenceSpan(sequenceStart=2, sequenceEnd=2, text="kal isko fix kar denge")],
            )
        ]
    )
    events = materialize_events(
        response,
        topic,
        blocks,
        {0: "server ID pending", 1: "connection failing", 2: "kal isko fix kar denge"},
    )
    assert events
    assert events[0].object is None
    assert events[0].actionSignal.artifactStatus == ABSTAIN_UNRESOLVED_OBJECT
    assert events[0].actionSignal.objectGroundingType == "UNRESOLVED"


def test_filler_does_not_glue_unrelated_topics():
    blocks = [
        MicroBlock(microBlockId="MB1", sequenceStart=63, sequenceEnd=63, sequenceIds=[63], sourceIds=["c63"], text="PNB setup alag topic hai", tokenCount=8),
        MicroBlock(microBlockId="MB2", sequenceStart=64, sequenceEnd=68, sequenceIds=[64, 65, 66, 67, 68], sourceIds=["c64"], text="haan 64\nok wait 65\nek second 66\nhello hello 67\nmic is rustling 68", tokenCount=12),
        MicroBlock(microBlockId="MB3", sequenceStart=80, sequenceEnd=80, sequenceIds=[80], sourceIds=["c80"], text="Play Store issue exist karta hai abhi", tokenCount=10),
        MicroBlock(microBlockId="MB4", sequenceStart=130, sequenceEnd=130, sequenceIds=[130], sourceIds=["c130"], text="create meeting page banana hai dashboard pe", tokenCount=10),
        MicroBlock(microBlockId="MB5", sequenceStart=131, sequenceEnd=140, sequenceIds=[131, 132, 133, 134, 135], sourceIds=["c131"], text="haan 131\nok wait 132\ntheek hai 133\nso yeah 134\nek second 135", tokenCount=12),
        MicroBlock(microBlockId="MB6", sequenceStart=150, sequenceEnd=150, sequenceIds=[150], sourceIds=["c150"], text="use GPT and OpenCV for coordinate extraction", tokenCount=10),
    ]
    topics = asyncio.run(segment_local_topics(blocks, _embedder()))
    pnb = next(topic for topic in topics if 63 in topic.sequenceIds)
    play = next(topic for topic in topics if 80 in topic.sequenceIds)
    meeting = next(topic for topic in topics if 130 in topic.sequenceIds)
    gpt = next(topic for topic in topics if 150 in topic.sequenceIds)
    assert pnb.topicId != play.topicId
    assert meeting.topicId != gpt.topicId
    assert 80 not in pnb.sequenceIds
    assert 150 not in meeting.sequenceIds
    play_reason = play.boundaryReason
    assert play.topicId != pnb.topicId
    assert play_reason in {"FILLER_BRIDGE_SEMANTIC_BREAK", "filler_bridge_split", "object_discontinuity", "below_continue_threshold"}


def test_filler_does_not_glue_unrelated_personal_topics():
    blocks = [
        MicroBlock(microBlockId="MB1", sequenceStart=1, sequenceEnd=1, sequenceIds=[1], sourceIds=["c1"], text="We decided to visit grandma Sunday.", tokenCount=8),
        MicroBlock(microBlockId="MB2", sequenceStart=2, sequenceEnd=4, sequenceIds=[2, 3, 4], sourceIds=["c2"], text="haan 2\nok wait 3\nek second 4", tokenCount=8),
        MicroBlock(microBlockId="MB3", sequenceStart=10, sequenceEnd=10, sequenceIds=[10], sourceIds=["c10"], text="Airport cab should be booked tomorrow.", tokenCount=8),
    ]
    topics = asyncio.run(segment_local_topics(blocks, _embedder()))
    family = next(topic for topic in topics if 1 in topic.sequenceIds)
    travel = next(topic for topic in topics if 10 in topic.sequenceIds)
    assert family.topicId != travel.topicId
    assert 10 not in family.sequenceIds
    assert travel.boundaryReason in {"FILLER_BRIDGE_SEMANTIC_BREAK", "filler_bridge_split", "object_discontinuity", "below_continue_threshold"}


def test_filler_bridge_does_not_oversplit_coherent_content():
    blocks = [
        MicroBlock(microBlockId="MB1", sequenceStart=90, sequenceEnd=92, sequenceIds=[90, 91, 92], sourceIds=["c90"], text="old keys are currently in use\ngenerated notes were reviewed\nmaster-prompt output requirements document karo", tokenCount=18),
        MicroBlock(microBlockId="MB2", sequenceStart=93, sequenceEnd=96, sequenceIds=[93, 94, 95, 96], sourceIds=["c93"], text="ok wait 93\ntheek hai 94\nso yeah 95\nek second 96", tokenCount=10),
    ]
    topics = asyncio.run(segment_local_topics(blocks, _embedder()))
    keys = next(topic for topic in topics if 90 in topic.sequenceIds)
    assert 91 in keys.sequenceIds
    assert 92 in keys.sequenceIds


def test_canonicalize_awkward_stt_objects():
    from services.conversation.event_pipeline.object_canon import canonicalize_action_object, objects_semantically_equivalent

    evidence = "create meeting page banana hai dashboard pe"
    assert canonicalize_action_object("meeting page banana hai dashboard pe", evidence) == "meeting page on the dashboard"
    assert canonicalize_action_object("is flow ki", "kal testing karenge is flow ki") == "this flow"
    s3 = canonicalize_action_object("S3 is not reaching frontend", "S3 is not reaching frontend")
    assert "s3" in s3.casefold() and "frontend" in s3.casefold()
    assert objects_semantically_equivalent("S3 is not reaching frontend", "S3 issue", "S3 is not reaching frontend")
    assert objects_semantically_equivalent("this flow", "flow", "kal testing karenge is flow ki")


def test_note_identity_keeps_s3_updates_and_merges_paraphrase():
    from services.conversation.event_pipeline.memory_identity import memory_relation

    def mem(event_id: str, kind: EventKind, meaning: str, obj: str) -> AtomicEvent:
        return AtomicEvent(
            eventId=event_id,
            topicId="T1",
            kind=kind,
            meaning=meaning,
            object=obj,
            entities=["S3"] if "S3" in meaning else ["Connection"],
            evidence=[EvidenceSpan(sequenceStart=1, sequenceEnd=1, text=meaning)],
            sequenceIds=[1],
            conversationId="conv",
            userId="u",
            spaceId="s",
        )

    failing = mem("e1", EventKind.ISSUE, "S3 is failing.", "S3")
    config = mem("e2", EventKind.FACT, "S3 configuration was changed.", "S3 configuration")
    still = mem("e3", EventKind.STATE, "S3 is still failing.", "S3")
    insecure_a = mem("e4", EventKind.STATE, "Connection is insecure.", "connection")
    insecure_b = mem("e5", EventKind.STATE, "Current connection was reported insecure.", "connection")
    assert memory_relation(config, failing) == "UPDATE"
    assert memory_relation(still, failing) in {"UPDATE", "RELATED"}
    assert memory_relation(insecure_b, insecure_a) == "DUPLICATE"
