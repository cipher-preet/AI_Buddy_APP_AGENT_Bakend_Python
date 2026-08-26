import asyncio
from types import SimpleNamespace

from services.conversation import agents
from services.conversation.intelligence import score_note, score_task, validation_decision_for_note, validation_decision_for_task
from services.conversation.models import EvidenceSpan, ExtractedNote, ExtractedTask, ExtractionOutcome
from services.conversation.semantic_reconstruction import reconstruct_window_intelligence
from services.llm.router import LLMCapability


def _span(text: str, sequence: int) -> EvidenceSpan:
    return EvidenceSpan(sequenceStart=sequence, sequenceEnd=sequence, text=text)


def _unit(ids, meaning, topic, roles, key, confidence=0.9):
    return {"evidenceIds": ids, "normalizedMeaning": meaning, "topic": topic, "roles": roles, "threadKey": key, "confidence": confidence}


class _FailingProvider:
    name = "failing-test-provider"

    async def generate_structured(self, request, schema):
        raise TimeoutError("simulated malformed JSON or timeout")


class _FailingRouter:
    def route(self, capability: LLMCapability):
        return _FailingProvider(), "failing-model"


def test_explicit_task_with_supported_metadata_scores_publishable():
    transcript = "[4] Rahul will verify the release OAuth credentials tomorrow before Play Store submission."
    task = ExtractedTask(
        title="Verify release OAuth credentials",
        body="Rahul will verify the release OAuth credentials tomorrow before Play Store submission.",
        operation="CREATE", ownerText="Rahul", dueDateText="tomorrow", dueDateStatus="ambiguous",
        confidence=0.01, sourceConversationId="conv_1",
        evidence=[_span("Rahul will verify the release OAuth credentials tomorrow before Play Store submission.", 4)], origin="explicit",
    )
    keep, reason = validation_decision_for_task(task, transcript)
    score, trace = score_task(task, transcript)
    assert keep and reason == "accepted" and score >= 0.55
    assert trace["llmConfidence"] == 0.01


def test_unsupported_evidence_or_metadata_never_persists():
    transcript = "[4] Please check the backend deployment before release."
    unsupported = ExtractedTask(
        title="Check deployment", body="Check the backend deployment before release.", operation="CREATE", ownerText="Rahul",
        confidence=1, sourceConversationId="conv_1", evidence=[_span("Please check the backend deployment before release.", 4)], origin="explicit",
    )
    wrong_sequence = unsupported.model_copy(update={"ownerText": None, "evidence": [_span("Please check the backend deployment before release.", 9)]})
    keep, reason = validation_decision_for_task(unsupported, transcript)
    assert keep is True and reason == "accepted"
    assert unsupported.ownerText is None
    assert validation_decision_for_task(wrong_sequence, transcript) == (False, "evidence_sequence_mismatch")


def test_confidence_is_evidence_based_not_a_model_confidence_floor():
    transcript = "[1] A single isolated detail was mentioned."
    note = ExtractedNote(title="Isolated detail", body="Isolated detail", confidence=0.99, sourceConversationId="conv_1", evidence=[_span("A single isolated detail was mentioned.", 1)])
    keep, reason = validation_decision_for_note(note, transcript)
    score, trace = score_note(note, transcript)
    assert not keep and reason == "generic_or_template_note"
    assert score < 0.70 and trace["llmConfidence"] == 0.99


def test_semantic_units_aggregate_related_evidence_before_fallback_note_synthesis():
    transcript = "\n".join([
        "[10] The soral membrane softens shock at its edge.",
        "[11] Its centre therefore stays stable during a drop.",
    ])
    reconstruction = reconstruct_window_intelligence(transcript, "conv", "space", [
        _unit([10], "The soral membrane softens shock at its edge.", "soral membrane behavior", ["fact"], "soral-thread"),
        _unit([11], "The softened edge keeps the centre stable during a drop.", "soral membrane behavior", ["explanation"], "soral-thread"),
    ])
    assert len(reconstruction.threads) == 1
    assert [span.sequenceStart for span in reconstruction.result.notes[0].evidence] == [10, 11]
    assert "established" not in reconstruction.result.notes[0].body.casefold()


def test_semantic_disagreement_reduces_confidence_without_english_conflict_markers():
    transcript = "\n".join(["[30] Ralum should be stored cold.", "[31] Ralum should be stored warm."])
    reconstruction = reconstruct_window_intelligence(transcript, "conv", "space", [
        _unit([30], "Ralum should be stored cold.", "ralum storage", ["requirement"], "ralum", 0.9),
        _unit([31], "Ralum should be stored warm.", "ralum storage", ["disagreement", "requirement"], "ralum", 0.9),
    ])
    assert reconstruction.facts
    assert max(item.confidence for item in reconstruction.facts) < 0.95


def test_no_classification_means_no_python_language_inference():
    transcript = "[1] Please arrange the impossible thing tomorrow."
    reconstruction = reconstruct_window_intelligence(transcript, "conv", "space")
    assert not reconstruction.result.notes and not reconstruction.result.tasks


def test_window_extraction_failure_abstains_instead_of_publishing_placeholder():
    window = SimpleNamespace(
        conversationId="conv", userId="user", spaceId="space", id="window", windowIndex=0,
        sequenceStart=8, sequenceEnd=10, text="[8] Some otherwise meaningful statement.",
    )
    result, provider, model = asyncio.run(agents.extract_window(_FailingRouter(), window, context={}, meeting_context={}))
    assert (provider, model) == ("failing-test-provider", "failing-model")
    assert result.extractionOutcome == ExtractionOutcome.EXTRACTION_FAILED
    assert not result.notes and not result.tasks
