from services.goals.clarify import detect_goal_type, fallback_clarify
from services.goals.domains import is_skip_answer
from services.goals.understanding import (
    apply_message_answers,
    empty_understanding,
    next_interview_question,
    ready_for_spaces,
)


def test_career_manager_is_not_sales_or_launch():
    statement = "I want to become a Zonal Sales Manager in a bank."
    assert detect_goal_type(statement) == "career"
    understood = empty_understanding(statement)
    assert understood["domain"] == "CAREER"
    assert understood["goalType"] == "career"


def test_become_a_does_not_classify_as_skill_for_jobs():
    assert detect_goal_type("become a full-stack engineer") == "career"


def test_saas_launch_still_product():
    assert detect_goal_type("launch a SaaS product for dentists") == "product_launch"


def test_habit_not_forced_into_career():
    assert detect_goal_type("read 20 pages every day") == "habit"


def test_skip_answers():
    assert is_skip_answer("Not sure")
    assert is_skip_answer("skip")
    assert is_skip_answer("suggest for me")
    assert not is_skip_answer("I am an area sales manager")


def test_interview_asks_current_level_for_career():
    goal = {"statement": "Become a zonal sales manager in a bank", "title": "ZSM"}
    understanding = empty_understanding(goal["statement"])
    question = next_interview_question(goal, understanding, {}, [])
    assert question is not None
    assert question.stepId == "current-level"


def test_messages_update_context_instead_of_reset():
    goal = {"statement": "Become a manager", "contextProfile": {"startingLevel": "student"}}
    messages = [
        {"role": "assistant", "stepId": "current-level", "text": "What is your current role?"},
        {"role": "user", "stepId": "current-level", "text": "Area Sales Manager"},
    ]
    profile = apply_message_answers(goal, messages)
    assert profile["startingLevel"] == "Area Sales Manager"


def test_ready_for_spaces_after_budget():
    goal = {
        "statement": "Become a zonal sales manager in a bank",
        "description": "Banking career",
        "successCriteria": "Receive a ZSM offer",
    }
    understanding = empty_understanding(goal["statement"])
    messages = [
        {"role": "assistant", "stepId": "current-level"},
        {"role": "assistant", "stepId": "timeframe"},
        {"role": "assistant", "stepId": "hours"},
    ]
    assert ready_for_spaces(goal, understanding, {}, messages)


def test_fallback_career_asks_current_level():
    goal = {"statement": "I want to become a Zonal Sales Manager in a bank.", "title": "ZSM"}
    built = fallback_clarify(goal, 0)
    assert built["step"]["stepId"] == "current-level"
