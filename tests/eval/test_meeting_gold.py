from tests.eval.meeting_gold import gold_cases


def test_gold_set_is_production_sized_and_labeled():
    cases = gold_cases()
    assert len(cases) >= 30
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids))
    assert "lenskart-hrms-meeting" in ids
    assert "atomic-dense-onboarding" in ids
    assert "cross-window-onboarding" in ids
    for case in cases:
        assert "expectedTasks" in case
        assert "expectedNotes" in case
        assert "expectedEvidence" in case
        assert "forbiddenArtifacts" in case
        assert case.get("transcript")
