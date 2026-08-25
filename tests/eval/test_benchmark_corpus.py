import re

from tests.eval.conversations import BENCHMARK_CASES, REQUIRED_CATEGORIES


def test_every_case_has_gold_evidence_inside_its_transcript():
    for case in BENCHMARK_CASES:
        sequences = {int(value) for value in re.findall(r"\[(\d+)\]", case["transcript"])}
        assert sequences, case["id"]
        for gold in [*case.get("goldTasks", []), *case.get("goldNotes", [])]:
            assert gold["kind"] in {"task", "note"}, case["id"]
            assert gold["evidenceSequences"], case["id"]
            assert set(gold["evidenceSequences"]) <= sequences, case["id"]
        assert set(case.get("nonTaskSequences", [])) <= sequences, case["id"]


def test_windowed_cases_keep_sequence_ids_in_the_full_transcript():
    windowed = [case for case in BENCHMARK_CASES if case.get("windows")]
    assert windowed
    for case in windowed:
        full = {int(value) for value in re.findall(r"\[(\d+)\]", case["transcript"])}
        for window in case["windows"]:
            window_ids = {int(value) for value in re.findall(r"\[(\d+)\]", window["transcript"])}
            assert window_ids <= full, f"{case['id']} {window['id']}"


def test_required_categories_each_have_at_least_one_case():
    by_category = {}
    for case in BENCHMARK_CASES:
        by_category.setdefault(case["category"], []).append(case["id"])
    for category in REQUIRED_CATEGORIES:
        assert by_category.get(category), category
    assert len(BENCHMARK_CASES) >= 20
