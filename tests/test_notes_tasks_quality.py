from types import SimpleNamespace

from services.conversation.artifacts import artifacts_from_window
from services.conversation.intelligence import score_and_filter_result, validation_decision_for_note, validation_decision_for_task
from services.conversation.models import EvidenceSpan, ExtractedNote, ExtractedTask


def _span(text: str, sequence: int) -> EvidenceSpan:
    return EvidenceSpan(sequenceStart=sequence, sequenceEnd=sequence, text=text)


def _window(text: str):
    return SimpleNamespace(text=text, conversationId="conversation", userId="user", spaceId="space", id="window", sequenceStart=1, sequenceEnd=10, windowIndex=0)


def test_below_threshold_candidates_never_cross_the_persistence_gate():
    transcript = "[1] One short observation."
    candidate = ExtractedNote(title="Short observation", body="One short observation.", confidence=1, sourceConversationId="conversation", evidence=[_span("One short observation.", 1)])
    keep, reason = validation_decision_for_note(candidate, transcript)
    assert keep is False
    assert reason in {"generic_or_template_note", "low_value_or_low_confidence"}


def test_rich_multi_evidence_note_persists_with_validated_confidence():
    transcript = "\n".join([
        "[1] The ulari bridge expands when the tide rises.",
        "[2] Expansion keeps the central span above the waterline.",
        "[3] Inspectors need the tide record beside each crossing report.",
    ])
    note = ExtractedNote(
        title="Ulari bridge tide behavior",
        body="The ulari bridge expands as the tide rises, keeping its central span above the waterline. Crossing reports need the corresponding tide record for inspection.",
        confidence=0.01, sourceConversationId="conversation",
        evidence=[_span("The ulari bridge expands when the tide rises.", 1), _span("Expansion keeps the central span above the waterline.", 2), _span("Inspectors need the tide record beside each crossing report.", 3)],
    )
    artifacts = artifacts_from_window(_window(transcript), SimpleNamespace(tasks=[], notes=[note], decisions=[], issues=[], topics=[]))
    assert len(artifacts) == 1 and artifacts[0].evidence


def test_synthesized_task_without_origin_or_quality_metadata_is_accepted_when_evidence_matches():
    transcript = "[4] Rahul will verify the release OAuth credentials tomorrow before Play Store submission."
    task = ExtractedTask(
        title="Verify release OAuth credentials",
        body="Rahul will verify the release OAuth credentials tomorrow before Play Store submission using the cited sequence.",
        operation="CREATE",
        confidence=0.01,
        sourceConversationId="conv_1",
        evidence=[_span("Rahul will verify the release OAuth credentials tomorrow before Play Store submission.", 4)],
        origin="unknown",
        changes={"synthesisSource": "llm"},
    )
    keep, reason = validation_decision_for_task(task, transcript)
    assert keep and reason == "accepted"


def test_llm_quality_verdict_false_still_rejects_even_with_evidence():
    transcript = "[1] The ulari bridge expands when the tide rises."
    note = ExtractedNote(
        title="Bridge behavior",
        body="The bridge expands when the tide rises and must be recorded beside each crossing report.",
        confidence=1,
        sourceConversationId="conversation",
        evidence=[_span("The ulari bridge expands when the tide rises.", 1)],
        debug={"synthesisSource": "llm", "quality": {"grounded": False, "independentlyUseful": False}},
    )
    keep, reason = validation_decision_for_note(note, transcript)
    assert keep is False
    assert reason == "generic_or_template_note"


def test_semantic_quality_verdict_can_reject_vague_or_unsupported_synthesis():
    transcript = "[1] The ulari bridge expands when the tide rises."
    note = ExtractedNote(
        title="Bridge behavior", body="The bridge expands when the tide rises.", confidence=1, sourceConversationId="conversation",
        evidence=[_span("The ulari bridge expands when the tide rises.", 1)],
        debug={"quality": {"grounded": False, "independentlyUseful": False}},
    )
    assert validation_decision_for_note(note, transcript) == (False, "generic_or_template_note")


def test_distinct_actions_sharing_words_and_evidence_are_preserved_without_semantic_key():
    transcript = "[1] Before sunset, prepare the atlas and notify the keeper."
    evidence = [_span("Before sunset, prepare the atlas and notify the keeper.", 1)]
    tasks = [
        ExtractedTask(title="Atlas handoff", body="Prepare the atlas before sunset.", operation="CREATE", confidence=1, sourceConversationId="conversation", evidence=evidence, origin="explicit"),
        ExtractedTask(title="Keeper handoff", body="Notify the keeper before sunset.", operation="CREATE", confidence=1, sourceConversationId="conversation", evidence=evidence, origin="explicit"),
    ]
    result = SimpleNamespace(tasks=tasks, notes=[], decisions=[], issues=[])
    assert len(score_and_filter_result(result, transcript).tasks) == 2


def test_matching_model_semantic_artifact_keys_merge_true_duplicates():
    transcript = "[1] Before sunset, prepare the atlas."
    evidence = [_span("Before sunset, prepare the atlas.", 1)]
    tasks = [
        ExtractedTask(title="Atlas preparation", body="Prepare the atlas before sunset.", operation="CREATE", confidence=1, sourceConversationId="conversation", evidence=evidence, origin="explicit", changes={"semanticArtifactKey": "opaque-atlas-1"}),
        ExtractedTask(title="Prepare atlas", body="Prepare the atlas before sunset with the current entries.", operation="CREATE", confidence=1, sourceConversationId="conversation", evidence=evidence, origin="explicit", changes={"semanticArtifactKey": "opaque-atlas-1"}),
    ]
    result = SimpleNamespace(tasks=tasks, notes=[], decisions=[], issues=[])
    assert len(score_and_filter_result(result, transcript).tasks) == 1


def test_task_metadata_must_be_supported_even_when_the_action_is_valid():
    transcript = "[1] Prepare the atlas before sunset."
    task = ExtractedTask(title="Atlas preparation", body="Prepare the atlas before sunset.", operation="CREATE", ownerText="Mira", confidence=1, sourceConversationId="conversation", evidence=[_span("Prepare the atlas before sunset.", 1)], origin="explicit")
    assert validation_decision_for_task(task, transcript) == (False, "invented_owner")


def test_model_marked_speculation_is_not_persisted_as_a_task():
    transcript = "[1] The atlas might be prepared before sunset."
    task = ExtractedTask(
        title="Atlas preparation", body="Prepare the atlas before sunset.", operation="CREATE", confidence=1,
        sourceConversationId="conversation", evidence=[_span("The atlas might be prepared before sunset.", 1)],
        origin="strongly_inferred", changes={"semanticSpeculation": True},
    )
    assert validation_decision_for_task(task, transcript) == (False, "speculative_inference")
