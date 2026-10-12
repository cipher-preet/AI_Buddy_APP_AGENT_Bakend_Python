from scripts.eval.goal_consultant_eval import load_scenarios, score_scenario


def test_eval_has_at_least_100_scenarios():
    rows = load_scenarios()
    assert len(rows) >= 100


def test_eval_subset_career_and_habit_pass():
    rows = {row["id"]: row for row in load_scenarios()}
    for key in ("career-zsm-student", "habit-reading", "skill-python-zero"):
        scores = score_scenario(rows[key])
        assert scores["domain"] == 1.0
        assert scores["relevance"] == 1.0
        assert scores["actionability"] == 1.0
