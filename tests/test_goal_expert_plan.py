from services.goals.plan import (
    _classic_fallback_plan,
    _is_generic_plan,
    _is_thin_expert,
    _launch_fallback_plan,
    assemble_expert_plan,
)
from services.goals.agents.pipeline import is_launch_style_goal, wants_expert_pipeline
from services.goals.schemas import GoalExecutionResult, GoalMilestone


def test_all_goals_use_expert_pipeline():
    assert wants_expert_pipeline({"title": "Run daily", "statement": "build a morning habit", "goalType": "habit"})
    assert wants_expert_pipeline({"title": "KukuNotes", "statement": "launch my app", "goalType": "app_build"})
    assert wants_expert_pipeline({"title": "Sales", "statement": "close 10 deals", "goalType": "sales"})


def test_launch_style_detection():
    assert is_launch_style_goal({"statement": "launch my AI meeting notes app", "goalType": "product_launch"})
    assert not is_launch_style_goal({"statement": "gain 5kg muscle", "goalType": "physique"})
    assert not is_launch_style_goal(
        {
            "statement": "I want to become a Zonal Sales Manager in a bank.",
            "understanding": {"domain": "CAREER"},
        }
    )
    assert not is_launch_style_goal(
        {"statement": "launch my career in banking", "understanding": {"domain": "CAREER"}}
    )


def test_launch_fallback_is_dense_not_three_phase():
    goal = {
        "title": "KukuNotes",
        "statement": "launch my AI meeting notes app",
        "goalType": "product_launch",
        "timeframe": "14 days",
        "successCriteria": "10 real testers complete a meeting",
    }
    plan = _launch_fallback_plan(goal)
    titles = [item["title"].lower() for item in plan["milestones"]]
    assert "specify the first shippable version" not in titles
    assert not _is_thin_expert(plan)
    assert len(plan["days"]) >= 8
    assert len(plan["strategies"]) >= 3
    assert plan["checklists"]


def test_classic_fallback_has_days_and_strategies():
    goal = {
        "title": "Stronger physique",
        "statement": "build muscle and get lean",
        "goalType": "physique",
        "timeframe": "12 months",
        "successCriteria": "Visible progress photos and stronger lifts",
    }
    plan = _classic_fallback_plan(goal)
    assert plan["days"]
    assert plan["strategies"]
    assert plan["milestones"]


def test_generic_three_phase_rejected():
    milestones = [
        {"title": "Specify the first shippable version"},
        {"title": "Build the vertical slice"},
        {"title": "Harden and polish"},
    ]
    assert _is_generic_plan(milestones)


def test_assemble_rejects_thin_execution():
    goal = {
        "title": "KukuNotes",
        "statement": "launch my app publicly",
        "goalType": "product_launch",
        "timeframe": "2 weeks",
        "successCriteria": "Public beta live",
    }
    thin = GoalExecutionResult(
        summary="Ship it",
        milestones=[
            GoalMilestone(title="Specify the first shippable version", successCriteria=["listed"], dueLabel="Week 1"),
            GoalMilestone(title="Build the vertical slice", successCriteria=["works"], dueLabel="Week 2"),
            GoalMilestone(title="Harden and polish", successCriteria=["done"], dueLabel="Later"),
        ],
    )
    plan = assemble_expert_plan(goal, strategy=None, execution=thin, models=["test"])
    assert plan["model"].startswith("fallback")
    assert not _is_thin_expert(plan)
