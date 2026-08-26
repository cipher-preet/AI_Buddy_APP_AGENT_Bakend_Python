import asyncio
from types import SimpleNamespace

from services.conversation import agents
from services.conversation.extraction_contract import (
    CORE_EVIDENCE_INVALID,
    EVIDENCE_VALID,
    OPTIONAL_METADATA_INVALID,
    hydrate_and_validate_unit_evidence,
)
from services.conversation.intelligence import validation_decision_for_task
from services.conversation.models import EvidenceSpan, ExtractedTask, SemanticUnit, WindowExtractionResult
from services.conversation.semantic_reconstruction import SemanticThread, SemanticTurn
from services.conversation.task_coverage import (
    CREATED_TASK,
    MERGED_INTO_TASK,
    TASK_COVERAGE_CONFLICT,
    annotate_semantic_units,
    evaluate_task_coverage,
    is_actionable_semantic_unit,
)
from services.llm.router import LLMCapability
from tests.fixtures.conversation_meetings import COMPLEX_MEETING_TRANSCRIPT
from tests.eval.conversations import BENCHMARK_CASES


def _span(sequence: int, text: str) -> EvidenceSpan:
    return EvidenceSpan(sequenceStart=sequence, sequenceEnd=sequence, text=text)


def _unit(key: str, kind: str, meaning: str, evidence: list[EvidenceSpan], **kwargs) -> SemanticUnit:
    return SemanticUnit(
        semanticKey=key,
        kind=kind,
        meaning=meaning,
        state=kwargs.get("state", "unresolved"),
        ownerText=kwargs.get("ownerText"),
        dueDateText=kwargs.get("dueDateText"),
        evidence=evidence,
        evidenceIds=kwargs.get("evidenceIds", [span.sequenceStart for span in evidence]),
        quality=kwargs.get("quality", {"grounded": True, "independentlyUseful": True}),
    )


def _task(title: str, body: str, evidence: list[EvidenceSpan], **kwargs) -> ExtractedTask:
    return ExtractedTask(
        title=title,
        body=body,
        operation=kwargs.get("operation", "CREATE"),
        ownerText=kwargs.get("ownerText"),
        dueDateText=kwargs.get("dueDateText"),
        confidence=0.8,
        sourceConversationId="conv",
        evidence=evidence,
        origin="explicit",
        changes={
            "synthesisSource": "llm",
            "semanticArtifactKey": kwargs.get("semanticArtifactKey", ""),
            "sourceSemanticUnitIds": kwargs.get("sourceSemanticUnitIds", []),
            "quality": {"grounded": True, "independentlyUseful": True},
        },
    )


def _thread(sequences: list[int], texts: dict[int, str], roles: list[str], key: str) -> SemanticThread:
    turns = [
        SemanticTurn(
            sequence=sequence,
            text=texts[sequence],
            normalized=texts[sequence],
            roles=set(roles),
            concepts=set(),
            transcript_quality=1.0,
            topic=key,
            meaning=texts[sequence],
            semantic_confidence=0.9,
            thread_key=key,
        )
        for sequence in sequences
    ]
    thread = SemanticThread(turns=turns, roles=set(roles), topic=key, thread_key=key)
    return thread


def test_paraphrased_cross_chunk_unit_survives_on_evidence_id_union():
    transcript = "\n".join(
        [
            "[0] Open a ticket.",
            "[1] Wait for every expected sequence ID.",
            "[2] Thursday evening.",
        ]
    )
    unit = _unit(
        "drain-ticket",
        "request",
        "Create a ticket so STOP waits for every expected sequence ID by Thursday evening.",
        [
            EvidenceSpan(
                sequenceStart=0,
                sequenceEnd=20,
                text="Create a ticket so STOP waits for every expected sequence before Thursday evening.",
            )
        ],
        evidenceIds=[0, 1, 2],
        ownerText="Priya",
        dueDateText="Thursday evening",
    )
    kept, rejected = hydrate_and_validate_unit_evidence([unit], transcript)
    assert rejected == 0
    assert len(kept) == 1
    assert kept[0].quality["evidenceOutcome"] == OPTIONAL_METADATA_INVALID
    assert kept[0].ownerText is None
    assert kept[0].dueDateText == "Thursday evening"
    assert kept[0].evidenceIds == [0, 1, 2]
    assert [span.sequenceStart for span in kept[0].evidence] == [0, 1, 2]


def test_unknown_owner_does_not_destroy_grounded_action():
    transcript = "[4] Please open the sequence-wait ticket."
    unit = _unit(
        "ticket",
        "request",
        "Open the sequence-wait ticket.",
        [_span(4, "Please open the sequence-wait ticket.")],
        ownerText="Rahul",
    )
    kept, rejected = hydrate_and_validate_unit_evidence([unit], transcript)
    assert rejected == 0
    assert kept[0].ownerText is None
    assert kept[0].quality["evidenceOutcome"] == OPTIONAL_METADATA_INVALID
    task = _task(
        "Open sequence-wait ticket",
        "Open the sequence-wait ticket using the cited sequence evidence.",
        [_span(4, "Please open the sequence-wait ticket.")],
        ownerText="Rahul",
    )
    keep, reason = validation_decision_for_task(task, transcript)
    assert keep is True and reason == "accepted"
    assert task.ownerText is None


def test_unknown_deadline_does_not_destroy_grounded_action():
    transcript = "[4] Please open the sequence-wait ticket."
    unit = _unit(
        "ticket",
        "request",
        "Open the sequence-wait ticket.",
        [_span(4, "Please open the sequence-wait ticket.")],
        dueDateText="next Friday",
    )
    kept, _ = hydrate_and_validate_unit_evidence([unit], transcript)
    assert kept[0].dueDateText is None
    task = _task(
        "Open sequence-wait ticket",
        "Open the sequence-wait ticket using the cited sequence evidence.",
        [_span(4, "Please open the sequence-wait ticket.")],
        dueDateText="next Friday",
    )
    keep, reason = validation_decision_for_task(task, transcript)
    assert keep is True and reason == "accepted"
    assert task.dueDateText is None


def test_invented_sequence_is_core_evidence_invalid():
    transcript = "[1] A harmless observation was mentioned."
    unit = _unit(
        "invented",
        "fact",
        "Invented meaning",
        [_span(99, "this line is not in the transcript")],
        evidenceIds=[99],
    )
    kept, rejected = hydrate_and_validate_unit_evidence([unit], transcript)
    assert not kept and rejected == 1
    assert unit.quality["evidenceOutcome"] == CORE_EVIDENCE_INVALID


def test_hindi_hinglish_action_survives_without_english_imperative():
    transcript = "[0] Mira ko sequence wait ticket kholna hai."
    unit = _unit(
        "ticket",
        "request",
        "Mira ko sequence wait ticket kholna hai.",
        [_span(0, "Mira ko sequence wait ticket kholna hai.")],
    )
    kept, rejected = hydrate_and_validate_unit_evidence([unit], transcript)
    assert rejected == 0
    assert is_actionable_semantic_unit(kept[0])
    assert kept[0].quality["evidenceOutcome"] == EVIDENCE_VALID


def test_noisy_stt_does_not_drop_grounded_action_when_ids_exist():
    transcript = "[0] Please open the sequence-wait ticket."
    unit = _unit(
        "ticket",
        "request",
        "Open the sequence-wait ticket.",
        [EvidenceSpan(sequenceStart=0, sequenceEnd=0, text="pls opn the sequense wait tkt")],
        evidenceIds=[0],
    )
    kept, rejected = hydrate_and_validate_unit_evidence([unit], transcript)
    assert rejected == 0
    assert kept[0].evidence[0].text == "Please open the sequence-wait ticket."


def test_completed_action_is_not_task_coverage():
    transcript = "[8] Rahul already sent the drain-safe finalizer notes to the channel."
    unit = _unit(
        "notes-sent",
        "commitment",
        "Rahul already sent the drain-safe finalizer notes.",
        [_span(8, "Rahul already sent the drain-safe finalizer notes to the channel.")],
        state="completed",
    )
    annotate_semantic_units([unit])
    assert is_actionable_semantic_unit(unit) is False
    coverage = evaluate_task_coverage([unit], WindowExtractionResult())
    assert coverage["validatedActionableUnitCount"] == 0
    assert coverage["taskCoverageConflict"] is False


def test_speculation_is_not_task_coverage():
    transcript = "[1] Maybe we should open a ticket if this happens again."
    unit = _unit(
        "maybe-ticket",
        "request",
        "Maybe open a ticket if this happens again.",
        [_span(1, "Maybe we should open a ticket if this happens again.")],
        quality={"grounded": True, "independentlyUseful": True, "semanticSpeculation": True},
    )
    annotate_semantic_units([unit])
    assert is_actionable_semantic_unit(unit) is False


def test_mixed_thread_can_yield_note_and_task_coverage():
    texts = {
        7: "We decided to keep raw transcript as the source of truth.",
        4: "Please also open a ticket to wait for every expected sequence ID before READY_FOR_PROCESSING.",
    }
    fact = _unit("truth", "decision", texts[7], [_span(7, texts[7])])
    action = _unit("ticket", "request", texts[4], [_span(4, texts[4])])
    threads = [
        _thread([7], texts, ["decision"], "truth"),
        _thread([4], texts, ["request", "assignment"], "ticket"),
    ]
    annotate_semantic_units([fact, action], threads)
    assert fact.quality["actionable"] is False
    assert action.quality["actionable"] is True
    task = _task(
        "Open sequence-wait ticket",
        "Open a ticket so STOP waits for every expected sequence ID.",
        [_span(4, texts[4])],
        sourceSemanticUnitIds=["ticket"],
        semanticArtifactKey="ticket",
    )
    coverage = evaluate_task_coverage(
        [fact, action],
        WindowExtractionResult(tasks=[task], notes=[]),
        threads,
    )
    assert coverage["taskCoverageConflict"] is False
    dispositions = {item["semanticKey"]: item["disposition"] for item in coverage["unitDispositions"]}
    assert dispositions["ticket"] == CREATED_TASK


def test_duplicate_units_merge_into_one_task_disposition():
    texts = {
        4: "Please also open a ticket to wait for every expected sequence ID before READY_FOR_PROCESSING.",
        5: "Mira should own the ticket instead of Rahul.",
    }
    open_ticket = _unit("ticket-open", "request", texts[4], [_span(4, texts[4])])
    assign = _unit("ticket-owner", "assignment", texts[5], [_span(5, texts[5])])
    annotate_semantic_units([open_ticket, assign])
    task = _task(
        "Track the STOP-drain ticket",
        "Open the Mira ticket so finalization waits for every expected sequence ID.",
        [_span(4, texts[4]), _span(5, texts[5])],
        sourceSemanticUnitIds=["ticket-open", "ticket-owner"],
        semanticArtifactKey="ticket",
    )
    coverage = evaluate_task_coverage(
        [open_ticket, assign],
        WindowExtractionResult(tasks=[task]),
    )
    dispositions = {item["semanticKey"]: item["disposition"] for item in coverage["unitDispositions"]}
    assert dispositions["ticket-open"] == MERGED_INTO_TASK
    assert dispositions["ticket-owner"] == MERGED_INTO_TASK
    assert coverage["taskCoverageConflict"] is False


def test_uncovered_request_thread_is_merged_into_validated_units():
    texts = {
        4: "Please also open a ticket to wait for every expected sequence ID before READY_FOR_PROCESSING.",
        5: "Mira should own the ticket instead of Rahul.",
        6: "Make the deadline Thursday evening.",
        7: "We decided to keep raw transcript as the source of truth.",
    }
    existing = [_unit("truth", "decision", texts[7], [_span(7, texts[7])])]
    thread = _thread([4, 5, 6], texts, ["request", "deadline", "decision"], "seq_id_ticket")
    from services.conversation.task_coverage import merge_uncovered_action_units

    transcript = "\n".join(f"[{seq}] {text}" for seq, text in texts.items())
    merged = merge_uncovered_action_units(existing, [thread], transcript)
    assert len(merged) == 2
    ticket = next(unit for unit in merged if unit.semanticKey == "seq_id_ticket")
    assert ticket.kind == "request"
    assert 4 in ticket.evidenceIds and 6 in ticket.evidenceIds


def test_uncertain_deadline_turn_does_not_drop_grounded_action_thread():
    texts = {
        4: "Please also open a ticket to wait for every expected sequence ID before READY_FOR_PROCESSING.",
        5: "Mira should own the ticket instead of Rahul.",
        6: "Make the deadline Thursday evening.",
    }
    thread = _thread([4, 5, 6], texts, ["request", "deadline", "assignment"], "seq_id_ticket")
    from dataclasses import replace

    thread.turns[2] = replace(thread.turns[2], uncertain=True)
    from services.conversation.task_coverage import merge_uncovered_action_units

    transcript = "\n".join(f"[{seq}] {text}" for seq, text in texts.items())
    merged = merge_uncovered_action_units([], [thread], transcript)
    assert any(unit.semanticKey == "seq_id_ticket" for unit in merged)
    assert is_actionable_semantic_unit(merged[0])


def test_actionable_units_with_zero_tasks_are_coverage_conflict():
    transcript = COMPLEX_MEETING_TRANSCRIPT
    unit = _unit(
        "mira-ticket",
        "request",
        "Open a Mira ticket so STOP waits for every expected sequence ID.",
        [
            _span(4, "Please also open a ticket to wait for every expected sequence ID before READY_FOR_PROCESSING."),
            _span(5, "Mira should own the ticket instead of Rahul."),
        ],
        evidenceIds=[4, 5, 6],
    )
    kept, rejected = hydrate_and_validate_unit_evidence([unit], transcript)
    assert rejected == 0
    annotate_semantic_units(kept)
    coverage = evaluate_task_coverage(kept, WindowExtractionResult(notes=[]))
    assert coverage["validatedActionableUnitCount"] == 1
    assert coverage["finalTaskCount"] == 0
    assert coverage["taskCoverageConflict"] is True


def test_long_meeting_fragmented_action_is_not_evidence_rejected():
    case = next(item for item in BENCHMARK_CASES if item["id"] == "long-meeting-two-hour")
    unit = _unit(
        "drain-gate",
        "assignment",
        "Mira will implement the drain gate so pending STT blocks READY_FOR_PROCESSING.",
        [
            EvidenceSpan(
                sequenceStart=0,
                sequenceEnd=12,
                text="Hour one reviewed the drain race and Mira will implement the drain gate.",
            )
        ],
        evidenceIds=[0, 1, 2],
        ownerText="Mira",
    )
    kept, rejected = hydrate_and_validate_unit_evidence([unit], case["transcript"])
    assert rejected == 0
    assert kept[0].ownerText == "Mira"
    assert 2 in kept[0].evidenceIds
    annotate_semantic_units(kept)
    assert kept[0].quality["actionable"] is True


def test_character_offset_span_still_grounds_core_action():
    case = next(item for item in BENCHMARK_CASES if item["id"] == "long-meeting-two-hour")
    unit = _unit(
        "drain-gate",
        "assignment",
        "Mira will implement the drain gate so pending STT blocks READY_FOR_PROCESSING.",
        [
            EvidenceSpan(
                sequenceStart=48,
                sequenceEnd=210,
                text="Mira will implement the drain gate after pending STT blocks READY_FOR_PROCESSING.",
            )
        ],
        evidenceIds=[],
        ownerText="Mira",
    )
    kept, rejected = hydrate_and_validate_unit_evidence([unit], case["transcript"])
    assert rejected == 0
    assert kept
    assert 2 in kept[0].evidenceIds
    annotate_semantic_units(kept)
    assert kept[0].quality["actionable"] is True


def test_synthesis_alias_defaults_missing_operation_and_confidence():
    from services.conversation.extraction_contract import alias_synthesis_payload

    payload = alias_synthesis_payload(
        {
            "tasks": [
                {
                    "title": "Create the STOP-drain ticket",
                    "body": "Track the Mira ticket so finalization waits for every expected sequence ID.",
                    "evidence": [{"sequenceStart": 4, "sequenceEnd": 4, "text": "Please also open a ticket."}],
                }
            ],
            "notes": [],
        }
    )
    task = payload["tasks"][0]
    assert task["operation"] == "CREATE"
    assert task["confidence"] == 0.5
    evidence_id_payload = alias_synthesis_payload(
        {
            "tasks": [
                {
                    "title": "Create the STOP-drain ticket",
                    "body": "Track the Mira ticket.",
                    "operation": "CREATE",
                    "confidence": 0.8,
                    "evidence": [{"evidenceId": 4, "text": "Please also open a ticket."}],
                }
            ],
            "notes": [],
        }
    )
    span = evidence_id_payload["tasks"][0]["evidence"][0]
    assert span["sequenceStart"] == 4
    assert span["sequenceEnd"] == 4
    content_payload = alias_synthesis_payload(
        {
            "tasks": [
                {
                    "content": "Fix the duplicate task issue today and then run long-meeting tests.",
                    "quality": {"independentlyUseful": True},
                }
            ],
            "notes": [
                {
                    "content": "The team plans internal testing with 1-hour and 2-4 hour recordings before release.",
                    "quality": {"independentlyUseful": True},
                }
            ],
        }
    )
    assert content_payload["tasks"][0]["title"]
    assert "duplicate task" in content_payload["tasks"][0]["body"]
    assert content_payload["notes"][0]["title"]
    assert "internal testing" in content_payload["notes"][0]["body"]


def test_notes_only_synthesis_triggers_targeted_coverage_repair():
    transcript = COMPLEX_MEETING_TRANSCRIPT
    mira = "Please also open a ticket to wait for every expected sequence ID before READY_FOR_PROCESSING."
    units = [
        _unit("mira-ticket", "request", mira, [_span(4, mira)], evidenceIds=[4, 5, 6]),
        _unit(
            "truth",
            "decision",
            "We decided to keep raw transcript as the source of truth.",
            [_span(7, "We decided to keep raw transcript as the source of truth.")],
        ),
    ]
    kept, _ = hydrate_and_validate_unit_evidence(units, transcript)
    annotate_semantic_units(kept)

    class _Router:
        def __init__(self):
            self.repair_calls = 0

        def route(self, capability: LLMCapability):
            return _Provider(self), "scripted-review-model"

    class _Provider:
        def __init__(self, router):
            self.router = router
            self.name = "scripted-review-provider"

        async def generate_structured(self, request, schema):
            self.router.repair_calls += 1
            assert schema.__name__ == "MissingItemRepairLLMResponse"
            return schema(
                tasks=[
                    {
                        "title": "Create Mira STOP-drain ticket",
                        "body": "Create and track the Mira ticket so finalization waits for every expected sequence ID before READY_FOR_PROCESSING.",
                        "operation": "CREATE",
                        "confidence": 0.9,
                        "ownerText": "Mira",
                        "dueDateText": "Thursday evening",
                        "semanticArtifactKey": "mira-ticket",
                        "sourceSemanticUnitIds": ["mira-ticket"],
                        "quality": {"grounded": True, "independentlyUseful": True},
                        "evidence": [
                            {"sequenceStart": 4, "sequenceEnd": 4, "text": mira},
                            {"sequenceStart": 5, "sequenceEnd": 5, "text": "Mira should own the ticket instead of Rahul."},
                            {"sequenceStart": 6, "sequenceEnd": 6, "text": "Make the deadline Thursday evening."},
                        ],
                    }
                ]
            )

    notes_only = WindowExtractionResult(
        notes=[],
        tasks=[],
        semanticUnits=kept,
        extractionDiagnostics={"finalSynthesisVerdict": "NO_PUBLISHABLE_ARTIFACTS", "finalSynthesisParsedNoteCount": 4},
    )
    diagnostics = {
        "finalSynthesisVerdict": "NO_PUBLISHABLE_ARTIFACTS",
        "finalSynthesisParsedTaskCount": 0,
        "finalSynthesisParsedNoteCount": 4,
        "qualityRepairAttempted": False,
    }
    router = _Router()
    result = asyncio.run(
        agents.apply_final_artifact_quality_gate(
            router, notes_only, kept, transcript, {}, "conv", "space", diagnostics
        )
    )
    assert router.repair_calls == 1
    assert diagnostics["finalSynthesisVerdict"] != "NO_PUBLISHABLE_ARTIFACTS"
    assert result.tasks
    assert "sequence" in result.tasks[0].body.casefold() or "ticket" in result.tasks[0].title.casefold()
    assert diagnostics["taskCoverageConflict"] is False or result.tasks
