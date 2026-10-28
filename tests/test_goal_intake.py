"""Question paths and plan briefs for different goals.

The intake must not ask one fixed script, and the brief must not paste a
user-growth plan onto a career, habit, or health goal.
"""

from services.goals.brief import apply_plan_brief, compose_plan_brief
from services.goals.plan import _fallback_plan
from services.goals.understanding import empty_understanding, next_interview_question, ready_for_spaces

BANNED = (
    "icp",
    "ideal customer profile",
    "pricing assumptions",
    "lead sourcing",
    "lead generation",
    "sales funnel",
    "landing page",
    "founder-led acquisition",
)


def _understood(statement: str):
    return empty_understanding(statement)


def _ask_all(statement: str, profile: dict | None = None, goal_extra: dict | None = None):
    goal = {"statement": statement, "title": statement[:80], **(goal_extra or {})}
    understanding = _understood(statement)
    profile = dict(profile or {})
    messages: list[dict] = []
    questions = []
    for _ in range(8):
        question = next_interview_question(goal, understanding, profile, messages)
        if question is None:
            break
        questions.append(question)
        messages.append({"role": "assistant", "stepId": question.stepId, "text": question.prompt})
    return questions


def _ids(questions) -> list[str]:
    return [question.stepId for question in questions]


def test_user_growth_asks_only_what_the_plan_needs():
    questions = _ask_all("Get 1,000 active users")
    assert _ids(questions) == ["subject", "audience", "stage", "timeframe", "approach"]
    assert "product-type" not in _ids(questions)
    assert "budget" not in _ids(questions)
    assert questions[0].control == "text"
    assert "1,000 active users" in questions[0].prompt.lower() or "active users" in questions[0].prompt.lower()
    assert "plan" in questions[0].note.lower()
    audience = next(question for question in questions if question.stepId == "audience")
    assert audience.control == "multi_select"
    labels = [option.label for option in audience.options]
    assert "Students" in labels
    assert "Businesses and startups" in labels
    assert "Not sure" in labels
    stage = next(question for question in questions if question.stepId == "stage")
    assert [option.label for option in stage.options][:3] == [
        "Not launched yet",
        "Launched, but very few users",
        "Already have some active users",
    ]
    deadline = next(question for question in questions if question.stepId == "timeframe")
    assert "3 months" in [option.label for option in deadline.options]
    assert questions[-1].control == "brief"
    assert "space" not in questions[-1].note.lower()
    assert "achieve" in " ".join(questions[-1].brief).lower()


def test_second_question_does_not_repeat_the_opening():
    questions = _ask_all("Get 1,000 active users")
    assert questions[0].note
    assert questions[1].note == ""


def test_india_budget_uses_rupees_and_other_places_do_not():
    india = _ask_all("Plan a trip to Japan", {"geography": "India"})
    budget = next(question for question in india if question.stepId == "budget")
    assert any("₹" in option.label for option in budget.options)
    assert "reaching people" not in budget.prompt.lower()

    plain = _ask_all("Plan a trip to Japan")
    budget = next(question for question in plain if question.stepId == "budget")
    assert all("₹" not in option.label for option in budget.options)


def test_named_product_and_audience_are_not_asked_again():
    questions = _ask_all("Launch a SaaS invoicing app for dentists")
    assert "subject" not in _ids(questions)
    assert "audience" not in _ids(questions)
    assert _ids(questions)[0] == "stage"


def test_bakery_audience_is_local_not_a_software_list():
    questions = _ask_all("Launch a local bakery")
    audience = next(question for question in questions if question.stepId == "audience")
    labels = [option.label for option in audience.options]
    assert "People who live nearby" in labels
    assert "Developers and tech professionals" not in labels


def test_career_path_never_asks_for_a_marketing_budget():
    questions = _ask_all("I want to become a Zonal Sales Manager in a bank.")
    assert _ids(questions)[0] == "current-level"
    assert "budget" not in _ids(questions)
    assert "audience" not in _ids(questions)
    assert "approach" == _ids(questions)[-1]
    assert "achieve" in " ".join(questions[-1].brief).lower()
    assert "sign-up" not in " ".join(questions[-1].brief).lower()


def test_python_from_zero_skips_the_level_question():
    questions = _ask_all("I want to learn Python from zero")
    assert _ids(questions)[0] == "timeframe"
    assert "current-level" not in _ids(questions)
    assert "stage" not in _ids(questions)
    assert "budget" not in _ids(questions)


def test_reading_habit_does_not_ask_audience_budget_or_hours():
    questions = _ask_all("Read 20 pages every day")
    assert _ids(questions) == ["stage", "timeframe", "approach"]
    labels = [option.label for option in questions[0].options]
    assert "Already doing it most days" in labels
    assert "60 days" in [option.label for option in questions[1].options]


def test_marathon_stage_differs_from_a_reading_habit():
    reading = _ask_all("Read 20 pages every day")[0]
    marathon = _ask_all("Train for a marathon")[0]
    assert marathon.stepId == "stage"
    assert [option.label for option in reading.options] != [option.label for option in marathon.options]
    assert any("train" in option.label.lower() for option in marathon.options)


def test_relationship_goal_is_not_a_user_growth_interview():
    questions = _ask_all("I want to be a better partner")
    assert questions[0].stepId == "subject"
    assert "relationship" in questions[0].prompt.lower()
    assert "budget" not in _ids(questions)
    assert "audience" not in _ids(questions)


def test_trip_budget_is_not_a_marketing_question():
    questions = _ask_all("Plan a two-week trip to Japan")
    budget = next(question for question in questions if question.stepId == "budget")
    assert "reaching people" not in budget.prompt.lower()


def test_signups_for_a_named_site_still_follow_the_people_path():
    questions = _ask_all("Get 1,000 signups for kukunotes.com")
    assert _ids(questions)[:4] == ["subject", "audience", "stage", "timeframe"]
    assert "product-type" not in _ids(questions)
    assert "plan" in questions[0].note.lower()


def test_sales_goal_asks_what_you_sell_and_keeps_a_stated_deadline():
    questions = _ask_all("Close 10 deals this quarter")
    assert questions[0].prompt == "What are you selling?"
    assert "timeframe" not in _ids(questions)
    labels = [option.label for option in next(question.options for question in questions if question.stepId == "audience")]
    assert "People who already buy from you" in labels
    assert "Developers and tech professionals" not in labels


def test_savings_and_a_novel_do_not_ask_what_the_statement_already_says():
    savings = _ask_all("Save 50000 for a house")
    assert savings[0].stepId != "subject"
    novel = _ask_all("Write a novel")
    assert novel[0].stepId != "subject"
    assert "making" not in novel[0].prompt.lower()


def test_different_goals_do_not_share_a_first_question():
    users = _ask_all("Get 1,000 active users")[0].prompt
    career = _ask_all("I want to become a Zonal Sales Manager in a bank.")[0].prompt
    habit = _ask_all("Read 20 pages every day")[0].prompt
    assert len({users, career, habit}) == 3


def test_approach_step_ends_intake():
    goal = {"statement": "Get 1,000 active users", "title": "1,000 active users"}
    understanding = _understood(goal["statement"])
    messages = [
        {"role": "assistant", "stepId": step_id}
        for step_id in ("subject", "audience", "stage", "budget", "timeframe", "approach")
    ]
    assert next_interview_question(goal, understanding, {}, messages) is None
    assert ready_for_spaces(goal, understanding, {}, messages)


def test_one_answer_does_not_finish_or_repeat():
    goal = {"statement": "Get 1,000 active users", "title": "1,000 users"}
    understanding = _understood(goal["statement"])
    messages = [
        {"role": "assistant", "stepId": "subject", "text": "What are you building?"},
        {"role": "user", "stepId": "subject", "text": "Not sure"},
    ]
    question = next_interview_question(goal, understanding, {}, messages)
    assert question is not None
    assert question.stepId == "audience"
    assert ready_for_spaces(goal, understanding, {}, messages) is False


def test_briefs_change_with_the_goal_and_do_not_promise_the_number():
    users = compose_plan_brief(
        {
            "statement": "Get 1,000 active users",
            "title": "1,000 active users",
            "description": "Records meetings into notes, mind maps, and documents",
            "audience": "Students, Developers and tech professionals",
            "timeframe": "30 days",
            "contextProfile": {"budget": "Under ₹5,000", "startingLevel": "Launched, but very few users"},
            "understanding": _understood("Get 1,000 active users"),
        }
    )
    career = compose_plan_brief(
        {
            "statement": "I want to become a Zonal Sales Manager in a bank.",
            "title": "Zonal Sales Manager",
            "timeframe": "Within a year",
            "contextProfile": {"startingLevel": "Student", "currentRole": "Student"},
            "understanding": _understood("I want to become a Zonal Sales Manager in a bank."),
        }
    )
    habit = compose_plan_brief(
        {
            "statement": "Read 20 pages every day",
            "title": "Daily reading",
            "understanding": _understood("Read 20 pages every day"),
        }
    )
    health = compose_plan_brief(
        {
            "statement": "Train for a marathon",
            "title": "Marathon",
            "contextProfile": {"startingLevel": "I train sometimes"},
            "understanding": _understood("Train for a marathon"),
        }
    )

    assert users["family"] == "people"
    assert career["family"] == "career"
    assert habit["family"] == "habit"
    assert health["family"] == "health"
    assert {item["title"] for item in users["methods"]} != {item["title"] for item in career["methods"]}
    assert {item["title"] for item in career["methods"]} != {item["title"] for item in habit["methods"]}
    assert {item["title"] for item in habit["methods"]} != {item["title"] for item in health["methods"]}

    assert "aggressive" in users["realityCheck"].lower()
    assert "not by a promised total" in users["realityCheck"].lower()
    assert "Students" in users["primaryFocus"]
    assert users["resourceSplit"]
    assert any("₹" in row["amount"] for row in users["resourceSplit"])
    assert "forecast" in " ".join(users["assumptions"]).lower()
    assert "registered" in users["whatCounts"].lower() or "core action" in users["whatCounts"].lower()
    assert "Records meetings" in users["starter"]

    assert career["resourceSplit"] == []
    assert "sign-up" not in career["realityCheck"].lower()
    assert habit["resourceSplit"] == []
    assert "customer" not in habit["primaryFocus"].lower()
    for brief in (career, habit, health):
        blob = str(brief).lower()
        for phrase in BANNED:
            assert phrase not in blob


def test_fallback_plan_keeps_the_brief_without_growth_language_on_a_career_goal():
    goal = {
        "statement": "I want to become a Zonal Sales Manager in a bank.",
        "title": "Zonal Sales Manager",
        "understanding": _understood("I want to become a Zonal Sales Manager in a bank."),
        "contextProfile": {"startingLevel": "student", "currentRole": "student"},
        "successCriteria": "Receive an offer",
    }
    plan = _fallback_plan(goal)
    blob = str(plan).lower()
    assert plan["realityCheck"]
    assert len(plan["strategies"]) >= 3
    assert plan["openQuestions"][0]["prompt"]
    for phrase in BANNED:
        assert phrase not in blob
    assert "saas" not in blob


def test_user_growth_fallback_defines_what_counts_and_splits_a_named_budget():
    goal = {
        "statement": "Get 1,000 active users",
        "title": "1,000 active users",
        "description": "Records meetings into notes",
        "audience": "Students, Businesses and startups",
        "timeframe": "30 days",
        "contextProfile": {"budget": "Under ₹5,000"},
        "understanding": _understood("Get 1,000 active users"),
    }
    plan = apply_plan_brief(_fallback_plan(goal), goal)
    titles = [item["title"] for item in plan["strategies"]]
    assert "Talk to the first likely users" in titles
    targets = " ".join(item.get("target", "") for item in plan["successDefinition"]["metrics"]).lower()
    assert "registered" in targets or "core action" in targets
    assert any(row["amount"].startswith("₹") for row in plan["resourceSplit"])
    assert "forecast" in " ".join(plan["assumptions"]).lower()
    assert plan["horizonDays"] == 30
    assert plan["days"][-1]["scheduleDay"] == 30
    assert plan["days"][-1]["dayIndex"] == len(plan["days"])
    assert "30 days" in plan["successDefinition"]["deadlineLabel"]
    for strategy in plan["strategies"]:
        assert strategy["title"].lower() in strategy["why"].lower()
        assert "this is the next path" not in strategy["why"].lower()
    for index, day in enumerate(plan["days"], start=1):
        assert day["dayIndex"] == index
        assert not day["title"].lower().startswith("day ")
        assert len(day["focus"]) > len(day["title"])
        assert "do this by day" not in day["focus"].lower()
        for task in day["tasks"]:
            assert task["why"].lower() != task["text"].lower()
            assert not task["why"].lower().startswith(f"{day['title'].lower()}:")
            assert "this is the work inside" not in task["why"].lower()
            assert "this is the next honest step" not in task["why"].lower()


def test_a_vague_platform_asks_what_it_is_then_who_it_is_for():
    questions = _ask_all("hey i want to make 1000 active user for my platform")
    assert _ids(questions)[:2] == ["subject", "audience"]
    assert "product-type" not in _ids(questions)
    assert "use or get" in questions[0].prompt.lower()


def test_three_months_spreads_the_timeline():
    goal = {
        "statement": "I want 1000 active users and I will achieve the goal in 3 months",
        "title": "1000 active users",
        "timeframe": "3 months",
        "understanding": _understood("I want 1000 active users and I will achieve the goal in 3 months"),
    }
    plan = _fallback_plan(goal)
    assert plan["horizonDays"] == 90
    assert plan["days"][-1]["scheduleDay"] == 90
    assert [day["dayIndex"] for day in plan["days"]] == list(range(1, len(plan["days"]) + 1))
    assert plan["days"][1]["scheduleDay"] > plan["days"][0]["scheduleDay"] + 1
    assert "3 months" in plan["successDefinition"]["deadlineLabel"]
    first = plan["days"][0]
    assert not first["title"].lower().startswith("day ")
    assert len(first["focus"]) > 40


def test_a_year_deadline_spans_the_year():
    goal = {
        "statement": "I want to become a Zonal Sales Manager in a bank.",
        "title": "Zonal Sales Manager",
        "timeframe": "Within a year",
        "understanding": _understood("I want to become a Zonal Sales Manager in a bank."),
    }
    plan = _fallback_plan(goal)
    assert plan["horizonDays"] == 365
    assert plan["days"][-1]["scheduleDay"] == 365
    assert plan["days"][-1]["dayIndex"] == len(plan["days"])
    assert "1 year" in plan["successDefinition"]["deadlineLabel"]
    assert "the deadline is 1 year" in plan["realityCheck"].lower()


def test_step_numbers_match_the_work_and_the_explanation_is_specific():
    from services.goals.brief import fit_plan_to_deadline

    goal = {
        "statement": "Deliver a client project as a project manager",
        "title": "Deliver a client project",
        "timeframe": "8 weeks",
        "understanding": _understood("Deliver a client project as a project manager"),
    }
    plan = fit_plan_to_deadline(
        {
            "days": [
                {
                    "dayIndex": 1,
                    "title": "Day 1 – Draft SOW",
                    "tasks": [{"text": "Draft and Send SOW"}],
                    "focus": "Draft and Send SOW",
                },
                {
                    "dayIndex": 4,
                    "title": "Day 2 – Finalize & Send SOW",
                    "tasks": [{"text": "Finalize"}],
                    "focus": "Finalize",
                },
            ],
            "successDefinition": {},
            "realityCheck": "The client has to accept the work.",
        },
        goal,
    )
    assert [day["dayIndex"] for day in plan["days"]] == [1, 2]
    assert plan["days"][-1]["scheduleDay"] == 56
    assert plan["days"][0]["title"] == "Draft SOW"
    assert "sow" in plan["days"][0]["focus"].lower()
    assert "price" not in plan["days"][0]["focus"].lower()
