from types import SimpleNamespace

from services.conversation.artifacts import artifacts_from_window
from services.conversation.semantic_reconstruction import reconstruct_window_intelligence


def _window(text: str):
    return SimpleNamespace(text=text, conversationId="conversation", userId="user", spaceId="space", id="window", sequenceStart=1, sequenceEnd=30, windowIndex=0)


def _unit(ids, meaning, topic, roles, key, confidence=0.88):
    return {
        "evidenceIds": ids,
        "normalizedMeaning": meaning,
        "topic": topic,
        "roles": roles,
        "threadKey": key,
        "confidence": confidence,
    }


def _reconstruct(text: str, units: list[dict]):
    result = reconstruct_window_intelligence(text, "conversation", "space", units).result
    return result, artifacts_from_window(_window(text), result)


def test_unseen_vocabulary_is_grouped_from_model_thread_identity_not_code_words():
    text = "\n".join([
        "[1] Vireli pollen changes the color of the dusk glass.",
        "[2] The effect fades after the glass is placed in shade.",
    ])
    result, artifacts = _reconstruct(text, [
        _unit([1], "Vireli pollen changes dusk glass color.", "Vireli pollen behavior", ["fact"], "vireli-1"),
        _unit([2], "The color effect fades in shade.", "Vireli pollen behavior", ["explanation"], "vireli-1"),
    ])
    assert len(result.notes) == 1
    assert len(result.notes[0].evidence) == 2
    assert artifacts and artifacts[0].evidence


def test_code_switched_turns_use_the_same_semantic_thread_without_language_rules():
    text = "\n".join([
        "[1] Kal zafran-lens ko andhere mein rakhna hai.",
        "[2] Keep the zafran-lens away from bright lamps overnight.",
    ])
    result, artifacts = _reconstruct(text, [
        _unit([1], "Keep the zafran lens in darkness overnight.", "zafran lens handling", ["requirement"], "zafran-2"),
        _unit([2], "Bright lamps must be avoided overnight.", "zafran lens handling", ["requirement"], "zafran-2"),
    ])
    assert len(result.notes) == 1
    assert len(result.notes[0].evidence) == 2
    assert artifacts


def test_descriptive_information_does_not_become_a_task():
    text = "\n".join([
        "[1] A nural weave absorbs vibration across its outer rings.",
        "[2] That is why the centre remains still during the demonstration.",
    ])
    result, artifacts = _reconstruct(text, [
        _unit([1], "A nural weave absorbs vibration across outer rings.", "nural weave behavior", ["fact"], "nural-3"),
        _unit([2], "The absorbed vibration keeps the centre still.", "nural weave behavior", ["explanation"], "nural-3"),
    ])
    assert result.notes and not result.tasks
    assert all(item.artifactType.value != "task" for item in artifacts)


def test_action_without_familiar_action_marker_is_created_only_from_semantic_intent():
    text = "[1] Mira’s turn: the aurora ledger before the next bell."
    result, artifacts = _reconstruct(text, [
        _unit([1], "Mira is assigned to prepare the aurora ledger before the next bell.", "aurora ledger", ["assignment", "action"], "aurora-4"),
    ])
    assert len(result.tasks) == 1
    assert result.tasks[0].ownerText is None  # not inferred from a name alone
    assert any(item.artifactType.value == "task" for item in artifacts)


def test_no_semantic_understanding_means_no_lexical_fallback_artifact():
    text = "[1] Blenko zuri quanta thel."
    result, artifacts = _reconstruct(text, [])
    assert not result.notes and not result.tasks and not artifacts


def test_distinct_actions_with_shared_evidence_are_not_merged_by_token_overlap():
    text = "[1] Before sunset, prepare the atlas and notify the keeper."
    units = [
        _unit([1], "Prepare the atlas before sunset.", "observatory handoff", ["action", "assignment"], "atlas-6"),
        _unit([1], "Notify the keeper before sunset.", "observatory handoff", ["action", "assignment"], "keeper-7"),
    ]
    result, _ = _reconstruct(text, units)
    assert len(result.tasks) == 2
