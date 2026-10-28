"""Steps have to say what to do, not repeat the outcome."""

import pytest

from services.goals.agents.llm import goal_step_pairs
from services.goals.agents.steps import evaluate_and_write_steps
from services.goals.schemas import DayTask, GoalStepWriterResult, PlanDay
from services.goals.step_actions import clarify_plan_steps, is_clear_action, steps_need_model


def _project_plan() -> dict:
    rows = [
        ("Scope Sign-off", "Scope Statement signed by client and internal sponsor"),
        (
            "Planning Completion",
            "WBS, effort estimates, resource capacity plan, risk register and 20-day schedule approved",
        ),
        ("Project Launch", "Kickoff meeting held, communication plan distributed"),
        (
            "Mid-project Review",
            "Progress vs. schedule reviewed, schedule re-baselined if needed, risk register updated",
        ),
        ("Final Delivery Sign-off", "Client acceptance sign-off obtained, hand-over package delivered"),
    ]
    days = []
    for index, (title, outcome) in enumerate(rows, start=1):
        days.append(
            {
                "id": f"day-{index}",
                "dayIndex": index,
                "title": title,
                "focus": f"Do the work for “{outcome}” and save something you can open again.",
                "definitionOfDone": outcome,
                "scheduledDate": f"2026-10-{26 + index - 1:02d}",
                "tasks": [
                    {
                        "id": f"day-{index}-t1",
                        "text": f"Do the work for “{outcome}” and save something you can open again.",
                        "done": False,
                    }
                ],
            }
        )
    return {"days": days}


def _blob(plan: dict) -> str:
    parts = []
    for day in plan["days"]:
        parts.append(day["title"])
        parts.append(day.get("focus") or "")
        for task in day["tasks"]:
            parts.append(task["text"])
            parts.append(task.get("why") or "")
    return " ".join(parts).lower()


def test_local_code_does_not_invent_actions():
    original = _project_plan()
    plan = clarify_plan_steps(original)
    assert plan["days"][0]["tasks"][0]["text"] == original["days"][0]["tasks"][0]["text"]
    assert steps_need_model(plan) is True


def test_a_real_action_is_not_rewritten():
    plan = clarify_plan_steps(
        {
            "days": [
                {
                    "id": "day-1",
                    "title": "Audit + scope freeze",
                    "tasks": [
                        {"id": "d1-t1", "text": "Smoke-test the core journey for the meeting notes app", "why": "Find the breaks before anyone else does."},
                        {"id": "d1-t2", "text": "Write a P0/P1/P2 backlog of launch blockers", "why": "The list is what you fix next."},
                    ],
                }
            ]
        }
    )
    texts = [task["text"] for task in plan["days"][0]["tasks"]]
    assert texts == [
        "Smoke-test the core journey for the meeting notes app",
        "Write a P0/P1/P2 backlog of launch blockers",
    ]
    assert plan["days"][0]["tasks"][0]["id"] == "d1-t1"


def test_step_writer_prefers_krutrim_gpt_oss_120b():
    assert goal_step_pairs()[0] == ("krutrim", "gpt-oss-120b")


def test_outcome_labels_need_the_model():
    assert steps_need_model(_project_plan()) is True


@pytest.mark.asyncio
async def test_model_rewrite_replaces_weak_steps_and_keeps_dates(monkeypatch):
    seen = {}

    async def fake_call(**kwargs):
        seen["pairs"] = kwargs["pairs"]
        seen["stage"] = kwargs["stage"]
        return (
            GoalStepWriterResult(
                accepted=False,
                issues=["The only checklist line repeats the outcome."],
                days=[
                    PlanDay(
                        title="Agree the scope",
                        focus="Write the scope and get both signatures before any delivery work.",
                        definitionOfDone="Both signatures are stored.",
                        tasks=[
                            DayTask(
                                text="Write the scope statement with inclusions, exclusions, and deliverables.",
                                why="The signature has to cover a real document.",
                                definitionOfDone="The draft is saved.",
                            ),
                            DayTask(
                                text="Send the scope statement to the client and the internal sponsor for signature.",
                                why="Delivery should wait for both of them.",
                                definitionOfDone="Both replies are saved.",
                            ),
                        ],
                    ),
                    PlanDay(
                        title="Approve the plan",
                        tasks=[
                            DayTask(text="Write the schedule, the risk register, and the effort estimates."),
                            DayTask(text="Ask the sponsor to approve the schedule before kickoff."),
                        ],
                    ),
                ],
            ),
            "krutrim:gpt-oss-120b",
        )

    monkeypatch.setattr("services.goals.agents.steps.call_goal_agent", fake_call)
    original = _project_plan()
    original["days"] = original["days"][:2]
    plan = await evaluate_and_write_steps(
        {"statement": "Deliver a client project as a project manager", "title": "Deliver a client project"},
        original,
    )
    assert seen["pairs"][0] == ("krutrim", "gpt-oss-120b")
    assert seen["stage"] == "goal_steps"
    assert plan["stepEvaluation"]["model"] == "krutrim:gpt-oss-120b"
    assert "do the work for" not in _blob(plan)
    assert plan["days"][0]["scheduledDate"] == "2026-10-26"
    assert "scope statement" in plan["days"][0]["tasks"][0]["text"].lower()
    assert len(plan["days"][0]["tasks"]) >= 2


@pytest.mark.asyncio
async def test_off_topic_model_steps_are_rejected(monkeypatch):
    async def fake_call(**kwargs):
        return (
            GoalStepWriterResult(
                accepted=True,
                days=[
                    PlanDay(
                        title="Morning workout",
                        tasks=[
                            DayTask(text="Run for twenty minutes and write the distance."),
                            DayTask(text="Stretch for ten minutes and note any pain."),
                        ],
                    ),
                    PlanDay(
                        title="Meal plan",
                        tasks=[
                            DayTask(text="Write tomorrow's meals and the portions."),
                            DayTask(text="Buy the food on that list."),
                        ],
                    ),
                ],
            ),
            "krutrim:gpt-oss-120b",
        )

    monkeypatch.setattr("services.goals.agents.steps.call_goal_agent", fake_call)
    plan = await evaluate_and_write_steps(
        {"statement": "Deliver a client project as a project manager"},
        _project_plan(),
    )
    assert plan["stepEvaluation"]["model"] == "krutrim:gpt-oss-120b"
    blob = _blob(plan)
    assert "workout" in blob or "run for twenty" in blob
    assert "language fundamentals" not in blob


_PHASE_SENTENCE = "This phase’s outcome is completed and can be shown"


def _skill_goal(*, level: str, hours: float = 5) -> dict:
    return {
        "title": "Learn a skill",
        "statement": "I want to learn a skill",
        "description": "watercolor painting",
        "goalType": "skill",
        "timeframe": "2 months",
        "availableHoursPerWeek": hours,
        "successCriteria": "I can paint a small scene on my own",
        "contextProfile": {"startingLevel": level, "description": "watercolor painting", "timeframe": "2 months"},
        "understanding": {"domain": "EDUCATION", "goalType": "skill", "startingLevel": "BEGINNER"},
    }


def _one_phase_plan() -> dict:
    return {
        "days": [
            {
                "id": "day-1",
                "dayIndex": 1,
                "title": "Week 1-2: Language Fundamentals",
                "focus": "Answer the Platform Preference Survey to determine the learning path.",
                "coachNote": "Answer the Platform Preference Survey to determine the learning path.",
                "tasks": [{"id": "day-1-t1", "text": _PHASE_SENTENCE, "done": False}],
            }
        ],
        "coachNote": "Answer the Platform Preference Survey to determine the learning path.",
        "immediateNextStep": "Answer the Platform Preference Survey to determine the learning path.",
        "milestones": [
            {
                "title": "Week 1-2: Language Fundamentals",
                "successCriteria": [_PHASE_SENTENCE],
            }
        ],
    }


def test_phase_outcome_sentence_is_left_for_the_model():
    plan = clarify_plan_steps(
        {
            "days": [
                {
                    "id": "day-1",
                    "title": "Practice the skill",
                    "tasks": [{"id": "day-1-t1", "text": _PHASE_SENTENCE, "done": False}],
                }
            ]
        }
    )
    assert plan["days"][0]["tasks"][0]["text"] == _PHASE_SENTENCE
    assert steps_need_model(plan) is True


def test_empty_milestone_criteria_are_not_the_filler_sentence():
    from services.goals.plan import _normalize_milestones

    rows = _normalize_milestones(
        [{"title": "Week 1-2: Language Fundamentals", "successCriteria": []}],
        "2 months",
    )
    assert "can be shown" not in str(rows).lower()


def _assert_two_month_shape(plan: dict, *, started: bool) -> None:
    days = plan["days"]
    assert 6 <= len(days) <= 10
    assert plan["horizonDays"] == 60
    assert days[-1]["scheduleDay"] == 60
    assert days[0]["scheduleDay"] == 1
    assert days[-1]["scheduleDay"] > days[0]["scheduleDay"]
    slots = [day["scheduleDay"] for day in days]
    assert slots == sorted(slots)
    assert len(set(slots)) == len(slots)
    blob = _blob(plan)
    assert "watercolor" in blob
    assert "language fundamentals" not in blob
    assert "platform preference" not in blob
    assert "can be shown" not in blob
    assert "do the work for" not in blob
    assert "save something you can open" not in blob
    if started:
        assert "fundamentals" not in blob
        assert "cannot do yet" not in blob
    else:
        assert "cannot do yet" in blob or "where you are" in blob
        assert "fundamentals" in blob
    for day in days:
        assert 2 <= len(day["tasks"]) <= 4
        for task in day["tasks"]:
            assert is_clear_action(task["text"], day["title"])
            assert task["text"].lower() != task["why"].lower()


@pytest.mark.asyncio
async def test_two_month_skill_goal_is_repaired_when_the_model_returns_one_phase(monkeypatch):
    seen = {}

    async def fake_call(**kwargs):
        seen["pairs"] = kwargs["pairs"]
        seen["payload"] = kwargs["payload"]
        return (
            GoalStepWriterResult(
                accepted=True,
                days=[
                    PlanDay(
                        title="Week 1-2: Language Fundamentals",
                        tasks=[DayTask(text=_PHASE_SENTENCE)],
                    )
                ],
            ),
            "krutrim:gpt-oss-120b",
        )

    monkeypatch.setattr("services.goals.agents.steps.call_goal_agent", fake_call)
    plan = await evaluate_and_write_steps(_skill_goal(level="From zero"), _one_phase_plan())
    assert seen["pairs"][0] == ("krutrim", "gpt-oss-120b")
    assert seen["payload"]["deadlineDays"] == 60
    assert seen["payload"]["description"] == "watercolor painting"
    assert "namedSkill" not in seen["payload"]
    assert "learnerStage" not in seen["payload"]
    assert plan["stepEvaluation"]["model"] == "kept"
    assert plan["days"][0]["title"] == "Week 1-2: Language Fundamentals"


@pytest.mark.asyncio
async def test_two_month_skill_goal_is_repaired_when_the_model_is_down(monkeypatch):
    async def fake_call(**kwargs):
        return None

    monkeypatch.setattr("services.goals.agents.steps.call_goal_agent", fake_call)
    plan = await evaluate_and_write_steps(_skill_goal(level="I already use it", hours=4), _one_phase_plan())
    assert plan["stepEvaluation"]["model"] == "kept"
    assert len(plan["days"]) == 1
    assert "beginner" not in _blob(plan)


@pytest.mark.asyncio
async def test_specific_multistep_plan_is_not_replaced(monkeypatch):
    called = False

    async def fake_call(**kwargs):
        nonlocal called
        called = True
        return None

    monkeypatch.setattr("services.goals.agents.steps.call_goal_agent", fake_call)
    original = {
        "days": [
            {
                "id": "day-1",
                "title": "Paint a small wash",
                "tasks": [
                    {"id": "a", "text": "Practice a flat watercolor painting wash and write what went wrong.", "why": "The wash is the piece you already need."},
                    {"id": "b", "text": "Compare the wash with a watercolor painting you made last month.", "why": "The comparison shows the gap."},
                ],
            },
            {
                "id": "day-2",
                "title": "Paint a small scene",
                "tasks": [
                    {"id": "c", "text": "Build a small watercolor painting of one object.", "why": "One object is enough for this proof."},
                    {"id": "d", "text": "Ask a friend what is unclear in the watercolor painting.", "why": "Their note is the next fix."},
                ],
            },
            {
                "id": "day-3",
                "title": "Revise the scene",
                "tasks": [
                    {"id": "e", "text": "Fix the one issue they named in the watercolor painting.", "why": "One fix finishes this round."},
                    {"id": "f", "text": "Write what the watercolor painting now shows.", "why": "The note is the result."},
                ],
            },
        ]
    }
    plan = await evaluate_and_write_steps(_skill_goal(level="Some practice"), original)
    assert called is False
    assert plan["stepEvaluation"]["model"] == "local-check"
    assert [day["title"] for day in plan["days"]] == [day["title"] for day in original["days"]]
    assert [task["text"] for day in plan["days"] for task in day["tasks"]] == [
        task["text"] for day in original["days"] for task in day["tasks"]
    ]
    assert [task["id"] for day in plan["days"] for task in day["tasks"]] == ["a", "b", "c", "d", "e", "f"]


@pytest.mark.asyncio
async def test_model_skill_plan_that_covers_the_deadline_is_kept(monkeypatch):
    async def fake_call(**kwargs):
        days = []
        for index in range(1, 9):
            days.append(
                PlanDay(
                    title=f"Watercolor painting block {index}",
                    tasks=[
                        DayTask(
                            text=f"Practice watercolor painting on exercise {index} and write what changed.",
                            why="This is the next piece of the painting.",
                        ),
                        DayTask(
                            text=f"Review the watercolor painting from exercise {index} and note the next change.",
                            why="The note decides the following block.",
                        ),
                    ],
                )
            )
        return GoalStepWriterResult(accepted=False, issues=["One phase was not enough."], days=days), "krutrim:gpt-oss-120b"

    monkeypatch.setattr("services.goals.agents.steps.call_goal_agent", fake_call)
    plan = await evaluate_and_write_steps(_skill_goal(level="From zero"), _one_phase_plan())
    assert plan["stepEvaluation"]["model"] == "krutrim:gpt-oss-120b"
    assert len(plan["days"]) == 8
    assert plan["days"][-1]["scheduleDay"] == 60
    assert "language fundamentals" not in _blob(plan)
    assert "exercise 8" in _blob(plan)
